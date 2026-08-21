"""Phase 1b: encode offer shards -> packed SPLADE vector shards at the GPU ceiling.

The bottleneck is CPU (Jinja render + tokenize), not the GPU — a single-threaded
loop pegs one core at ~2.1k/s while the H100 idles, and naive multi-process (N
separate CUDA contexts) THRASHES the GPU (~0.9k/s). Correct pattern: ONE GPU
process fed by a fork-pool of CPU workers that render+tokenize in parallel
(imap prefetch), so the GPU stays saturated.

Per article: render each offer (description-free), max-merge the offers' SPLADE
reps in dense space on the GPU, top-k prune, pack. Output shard (out/<name>.bin):
  repeated [id_len u16][id bytes][u32 nnz][nnz*(u16 tok, f16 wt)]  records.
Resumable: shards with a .bin.done marker are skipped; --delete-input frees disk.

  PYTORCH_ALLOC_CONF=expandable_segments:True LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    uv run --extra train python scripts/encode_shards.py \
      --splade-ckpt checkpoints/capable-auk-759/best-*.ckpt \
      --in-dir data/offers --out-dir data/vectors --workers 24
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import struct
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.tokenization import load_fast_tokenizer

DEV = "cuda"
# fork-inherited worker globals (set in main before the pool forks)
_REN = _TOK = None
_PAD = 128


def _init(model_name, data_cfg_container, pad):
    global _REN, _TOK, _PAD
    from omegaconf import OmegaConf
    _REN = RowTextRenderer(OmegaConf.create(data_cfg_container))
    _TOK = load_fast_tokenizer(model_name)
    _PAD = pad


def worker_render(lines):
    """[raw json line] -> (ids, counts, input_ids[T,PAD] i32, attn[T,PAD] u8).
    Parses + renders + tokenizes in the WORKER (json.loads is heavy; keep it off
    the single main thread), dedups each article's offers, pads to fixed PAD."""
    ids, counts, tok_rows = [], [], []
    for line in lines:
        if not line.strip():
            continue
        a = json.loads(line)
        seen, texts = set(), []
        for off in a["offers"]:
            t = _REN.render_offer_text({**off, "description": ""})
            if t and t not in seen:
                seen.add(t); texts.append(t)
        if not texts:
            continue
        enc = _TOK(texts, truncation=True, max_length=_PAD)["input_ids"]
        ids.append(a["id"]); counts.append(len(enc)); tok_rows.extend(enc)
    if not tok_rows:
        return None
    T = len(tok_rows)
    input_ids = np.zeros((T, _PAD), dtype=np.int32)
    attn = np.zeros((T, _PAD), dtype=np.uint8)
    for i, row in enumerate(tok_rows):
        L = min(len(row), _PAD)
        input_ids[i, :L] = row[:L]; attn[i, :L] = 1
    return ids, np.array(counts, dtype=np.int64), input_ids, attn


def chunks(path, size):
    """Yield raw line batches — parsing happens in the workers, not here."""
    with open(path) as f:
        while True:
            batch = list(islice(f, size))
            if not batch:
                return
            yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splade-ckpt", required=True)
    ap.add_argument("--in-dir", default="data/offers")
    ap.add_argument("--out-dir", default="data/vectors")
    ap.add_argument("--pad", type=int, default=128)
    ap.add_argument("--gpu-batch", type=int, default=1024, help="offer-texts per GPU fwd")
    ap.add_argument("--art-chunk", type=int, default=3000, help="articles per CPU task")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--delete-input", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from omegaconf import OmegaConf
    model, cfg = load_embedding_module_from_checkpoint(args.splade_ckpt)  # CPU
    data_container = OmegaConf.to_container(cfg.data, resolve=True)

    # fork the CPU render/tokenize pool BEFORE any CUDA init (fork after .to(cuda)
    # inherits a broken CUDA context). Workers build their own renderer/tokenizer.
    ctx = mp.get_context("fork")
    pool = ctx.Pool(args.workers, initializer=_init,
                    initargs=(cfg.model.model_name, data_container, args.pad))

    model = model.to(DEV).eval()  # CUDA init only in the main process, after fork

    @torch.inference_mode()
    def gpu_encode(input_ids, attn):
        reps = []
        for i in range(0, input_ids.shape[0], args.gpu_batch):
            ii = torch.from_numpy(input_ids[i:i + args.gpu_batch]).to(DEV).long()
            aa = torch.from_numpy(attn[i:i + args.gpu_batch]).to(DEV).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r = model._encode_representations(ii, aa)
            reps.append(r.float())
        return torch.cat(reps, 0)

    def pack_batch(ids, counts, reps, fout):
        # Vectorized: one GPU max-merge (index_reduce amax over offers->articles),
        # one batched topk, ONE host transfer for the whole chunk — no per-article
        # GPU sync (that was the bottleneck).
        A = len(ids)
        art_idx = torch.repeat_interleave(
            torch.arange(A, device=DEV), torch.from_numpy(counts).to(DEV))
        merged = torch.zeros(A, reps.shape[1], device=DEV)
        merged.index_reduce_(0, art_idx, reps, "amax", include_self=False)
        vals, idx = torch.topk(merged, min(args.topk, reps.shape[1]), dim=1)
        vals_h = vals.to(torch.float16).cpu().numpy()
        idx_h = idx.to(torch.int32).cpu().numpy().astype(np.uint16)
        pos = vals_h > 0
        n = 0
        for j, aid in enumerate(ids):
            m = pos[j]
            if not m.any():
                continue
            tid = idx_h[j][m]; wt = vals_h[j][m]
            aidb = aid.encode()
            fout.write(struct.pack("<H", len(aidb)) + aidb
                       + struct.pack("<I", len(tid)) + tid.tobytes() + wt.tobytes())
            n += 1
        return n

    shards = sorted(glob.glob(f"{args.in_dir}/*.jsonl"))
    print(f"{len(shards)} shards, {args.workers} CPU workers", flush=True)
    t0, total = time.time(), 0
    for sp in shards:
        name = os.path.basename(sp).rsplit(".", 1)[0]
        outp = f"{args.out_dir}/{name}.bin"
        if os.path.exists(outp + ".done"):
            continue
        ts, n = time.time(), 0
        with open(outp, "wb") as fout:
            for res in pool.imap(worker_render, chunks(sp, args.art_chunk), chunksize=1):
                if res is None:
                    continue
                ids, counts, input_ids, attn = res
                reps = gpu_encode(input_ids, attn)
                n += pack_batch(ids, counts, reps, fout)
                total += 0  # counted below
        open(outp + ".done", "w").close()
        if args.delete_input:
            os.remove(sp)
        total += n
        el = time.time() - t0
        print(f"{name}: {n:,} ({n/max(time.time()-ts,1e-3):,.0f}/s) | "
              f"total {total:,} ({total/max(el,1e-3):,.0f}/s avg)", flush=True)
    pool.close()
    print(f"DONE {total:,} in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
