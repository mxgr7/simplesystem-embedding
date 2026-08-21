"""Pure SPLADE doc-inference benchmark on the GPU box — find the throughput ceiling.

No network / ES / tunnel. Loads capable-auk-759, renders real offer texts
(description-free, the deployed config), and sweeps the levers that matter:
precision (fp32 / bf16 autocast), batch size, padding strategy, torch.compile,
and the sparsification step (dense [B,vocab] -> {token_id: weight}).

Reports docs/s for forward-only and forward+sparsify so we can see where the
ceiling is and what the practical (usable output) rate is.

  PYTORCH_ALLOC_CONF=expandable_segments:True LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    uv run --extra train python scripts/splade_bench.py \
      --splade-ckpt checkpoints/capable-auk-759/best-*.ckpt
"""
import argparse
import json
import time

import numpy as np
import torch

from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer
from embedding_train.tokenization import load_fast_tokenizer

DEV = "cuda"


def load_texts(path, n, renderer):
    texts = []
    for line in open(path):
        if len(texts) >= n:
            break
        r = json.loads(line)
        r["description"] = ""
        t = renderer.render_offer_text(r)
        if t:
            texts.append(t)
    return texts


def sync():
    torch.cuda.synchronize()


@torch.inference_mode()
def encode_forward(model, input_ids, attn, amp):
    """Return dense reps [B, vocab]. amp: None|torch.bfloat16|torch.float16."""
    if amp is None:
        return model.encode({"input_ids": input_ids, "attention_mask": attn})
    with torch.autocast(device_type="cuda", dtype=amp):
        return model.encode({"input_ids": input_ids, "attention_mask": attn})


def sparsify_cpu_dict(reps):
    """Current path: to CPU numpy, per-row nonzero -> python dict."""
    r = reps.float().cpu().numpy()
    out = []
    for vec in r:
        idx = np.nonzero(vec > 0)[0]
        out.append({int(i): float(vec[i]) for i in idx})
    return out


def sparsify_gpu_topk(reps, k=256):
    """GPU top-k prune then move only k values/idx to CPU. Returns list of dicts."""
    vals, idx = torch.topk(reps, k=min(k, reps.shape[1]), dim=1)
    mask = vals > 0
    vals = vals.cpu().numpy(); idx = idx.cpu().numpy(); mask = mask.cpu().numpy()
    out = []
    for i in range(reps.shape[0]):
        m = mask[i]
        out.append(dict(zip(idx[i][m].tolist(), vals[i][m].tolist())))
    return out


def bench(model, tok, texts, bs, max_len, amp, pad, compile_fwd, sparsify, n_batches=20):
    # pre-tokenize batches (tokenization cost measured separately)
    batches = []
    t_tok = time.time()
    for i in range(0, min(len(texts), bs * n_batches), bs):
        chunk = texts[i:i + bs]
        enc = tok(chunk, padding=pad, truncation=True, max_length=max_len,
                  return_tensors="pt")
        batches.append((enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)))
    tok_s = (time.time() - t_tok)
    n_docs = sum(b[0].shape[0] for b in batches)

    # warmup
    for b in batches[:3]:
        encode_forward(model, b[0], b[1], amp)
    sync()

    # forward-only timing
    t0 = time.time()
    reps_cache = []
    for b in batches:
        reps = encode_forward(model, b[0], b[1], amp)
        reps_cache.append(reps)
    sync()
    fwd_s = time.time() - t0

    # sparsify timing (on the cached reps)
    t0 = time.time()
    total = 0
    for reps in reps_cache:
        out = sparsify(reps)
        total += len(out)
    sp_s = time.time() - t0

    avg_nnz = np.mean([len(o) for o in out]) if out else 0
    seqlen = int(np.mean([b[0].shape[1] for b in batches]))
    return {
        "docs": n_docs, "seqlen": seqlen,
        "fwd_docs_s": n_docs / fwd_s,
        "sparsify_docs_s": n_docs / sp_s,
        "combined_docs_s": n_docs / (fwd_s + sp_s),
        "tok_docs_s": n_docs / tok_s,
        "nnz": avg_nnz,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splade-ckpt", required=True)
    ap.add_argument("--data", default="data/desc_distractors.jsonl")
    ap.add_argument("--n-texts", type=int, default=40000)
    args = ap.parse_args()

    model, cfg = load_embedding_module_from_checkpoint(args.splade_ckpt)
    model = model.to(DEV).eval()
    tok = load_fast_tokenizer(cfg.model.model_name)
    ren = RowTextRenderer(cfg.data)
    max_cfg = int(cfg.data.max_offer_length)

    texts = load_texts(args.data, args.n_texts, ren)
    lens = [len(tok(t, truncation=True, max_length=512)["input_ids"]) for t in texts[:5000]]
    print(f"loaded {len(texts):,} texts | token len: mean {np.mean(lens):.0f} "
          f"p50 {np.percentile(lens,50):.0f} p90 {np.percentile(lens,90):.0f} "
          f"p99 {np.percentile(lens,99):.0f} max {max(lens)} | cfg max_offer_length={max_cfg}",
          flush=True)

    print(f"\n{'config':<52} {'seq':>4} {'fwd/s':>9} {'sparsify/s':>11} "
          f"{'comb/s':>9} {'nnz':>5}", flush=True)

    def row(label, **kw):
        r = bench(model, tok, texts, **kw)
        print(f"{label:<52} {r['seqlen']:>4} {r['fwd_docs_s']:>9,.0f} "
              f"{r['sparsify_docs_s']:>11,.0f} {r['combined_docs_s']:>9,.0f} "
              f"{r['nnz']:>5.0f}", flush=True)
        return r

    # baseline (current path): fp32, bs128, max_len=256 fixed pad, cpu-dict sparsify
    row("baseline fp32 bs128 pad=max_length(256) cpu-dict",
        bs=128, max_len=256, amp=None, pad="max_length", compile_fwd=False,
        sparsify=sparsify_cpu_dict)
    # dynamic padding (pad to batch max)
    row("fp32 bs128 pad=longest cpu-dict",
        bs=128, max_len=256, amp=None, pad="longest", compile_fwd=False,
        sparsify=sparsify_cpu_dict)
    # bf16 autocast, batch sweep, dynamic pad
    for bs in (128, 256, 512, 1024):
        row(f"bf16 bs{bs} pad=longest cpu-dict",
            bs=bs, max_len=256, amp=torch.bfloat16, pad="longest",
            compile_fwd=False, sparsify=sparsify_cpu_dict)
    # bf16 + gpu-topk sparsify (isolate sparsify cost vs cpu-dict)
    for bs in (512, 1024):
        row(f"bf16 bs{bs} pad=longest gpu-topk256",
            bs=bs, max_len=256, amp=torch.bfloat16, pad="longest",
            compile_fwd=False, sparsify=lambda r: sparsify_gpu_topk(r, 256))


if __name__ == "__main__":
    main()
