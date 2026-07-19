"""Prewarm splade:v1:<hash> Redis entries for every unique article hash.

Mirror of prewarm_v2_missing.py, but SPLADE has no TEI server — the forward is
run in-process on the GPU. Reuses the SAME render_inputs.parquet (one row per
unique article_hash, 8 canonical columns) the dense prewarm built, so the two
caches are keyed identically (article_hash is a hash of the raw fields, template-
agnostic). Renders with the checkpoint's own cfg.data (SPLADE template); encodes;
stores each vector packed via indexer.sparse_codec.

Resumable: pipelined Redis EXISTS skips hashes already present (re-run = no-op for
done hashes). Deployment default is description-free (--drop-description; the
description A/B showed descriptions don't help recall) — pass --keep-description
only if deploying the description-trained model.

Run on the H100 box (ask first — shared; hours for the full catalog):
  PYTORCH_ALLOC_CONF=expandable_segments:True LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    uv run --extra train python scripts/prewarm_splade.py \
      --splade-ckpt checkpoints/capable-auk-759/best-*.ckpt --drop-description
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset as pds
import redis
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from embedding_train.model import load_embedding_module_from_checkpoint  # noqa: E402
from embedding_train.rendering import RowTextRenderer  # noqa: E402
from embedding_train.tokenization import load_fast_tokenizer  # noqa: E402
from indexer.sparse_codec import pack_sparse  # noqa: E402

HASH_VERSION = "v1"  # splade cache generation
EXPORT_DIR = Path("/data/datasets/mongo_offers_export_20260512")
DEFAULT_RENDER_INPUTS = EXPORT_DIR / "render_inputs.parquet"
COLUMNS = ["article_hash", "name", "manufacturer_name", "description",
           "category_paths", "ean", "article_number",
           "manufacturer_article_number", "manufacturer_article_type"]
log = logging.getLogger("prewarm_splade")


@torch.inference_mode()
def encode_batch(model, tok, texts, max_len, device):
    inp = tok(texts, padding=True, truncation=True, max_length=max_len,
              return_tensors="pt")
    rep = model.encode({k: v.to(device) for k, v in inp.items()})
    return rep.float().cpu().numpy()


def to_dict(vec):
    idx = np.nonzero(vec > 0)[0]
    return {int(i): float(vec[i]) for i in idx}


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splade-ckpt", required=True)
    ap.add_argument("--render-inputs-parquet", default=str(DEFAULT_RENDER_INPUTS))
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--exists-batch", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--drop-description", action="store_true",
                    help="blank description before render (description-free deploy)")
    ap.add_argument("--keep-description", dest="drop_description",
                    action="store_false")
    ap.set_defaults(drop_description=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after N hashes (probe)")
    ap.add_argument("--shard", default=None, help="K/N — only buckets where bucket%%N==K")
    args = ap.parse_args()

    model, cfg = load_embedding_module_from_checkpoint(args.splade_ckpt)
    model = model.to(args.device).eval()
    tok = load_fast_tokenizer(cfg.model.model_name)
    ren = RowTextRenderer(cfg.data)
    max_len = int(cfg.data.max_offer_length)
    log.info("loaded %s | max_offer_length=%d | drop_description=%s",
             args.splade_ckpt, max_len, args.drop_description)

    rc = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    rc.ping()

    path = Path(args.render_inputs_parquet)
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        buckets = sorted(path.glob("bucket=*"), key=lambda p: int(p.name.split("=")[1]))
        mine = [str(f) for p in buckets if int(p.name.split("=")[1]) % n == k
                for f in p.glob("*.parquet")]
        dataset = pds.dataset(mine, format="parquet")
    else:
        dataset = pds.dataset(str(path), format="parquet", partitioning="hive")
    total = dataset.count_rows()
    log.info("render_inputs: %s unique hashes", f"{total:,}")

    t0, scanned, embedded, cached, nnz_sum = time.time(), 0, 0, 0, 0
    last = t0
    for chunk in dataset.to_batches(batch_size=10000, columns=COLUMNS):
        rows = chunk.to_pylist()
        if not rows:
            continue
        # pipelined EXISTS -> only encode missing hashes
        miss = []
        for i in range(0, len(rows), args.exists_batch):
            sub = rows[i:i + args.exists_batch]
            pipe = rc.pipeline(transaction=False)
            for r in sub:
                pipe.exists(f"splade:{HASH_VERSION}:{r['article_hash']}")
            for r, ex in zip(sub, pipe.execute()):
                if ex == 0:
                    miss.append(r)
        cached += len(rows) - len(miss)
        scanned += len(rows)

        for i in range(0, len(miss), args.batch_size):
            batch = miss[i:i + args.batch_size]
            texts, hashes = [], []
            for r in batch:
                if args.drop_description:
                    r = {**r, "description": ""}
                texts.append(ren.render_offer_text(r))
                hashes.append(r["article_hash"])
            reps = encode_batch(model, tok, texts, max_len, args.device)
            pipe = rc.pipeline(transaction=False)
            for h, vec in zip(hashes, reps):
                d = to_dict(vec)
                nnz_sum += len(d)
                pipe.set(f"splade:{HASH_VERSION}:{h}", pack_sparse(d))
            pipe.execute()
            embedded += len(batch)

        now = time.time()
        if now - last >= 15:
            el = now - t0
            rate = embedded / el if el else 0
            remain = (total - scanned) * (embedded / max(scanned, 1))
            eta = remain / rate / 60 if rate else -1
            log.info("scanned=%s/%s (%.1f%%) embedded=%s cached=%s "
                     "nnz~%d rate=%.0f/s eta=%.1fmin",
                     f"{scanned:,}", f"{total:,}", 100 * scanned / total,
                     f"{embedded:,}", f"{cached:,}",
                     nnz_sum / max(embedded, 1), rate, eta)
            last = now
        if args.limit and embedded >= args.limit:
            log.info("hit --limit %d, stopping", args.limit)
            break

    el = time.time() - t0
    log.info("DONE embedded=%s cached=%s mean_nnz=%.0f elapsed=%.0fs (%.1fmin) %.0f/s",
             f"{embedded:,}", f"{cached:,}", nnz_sum / max(embedded, 1),
             el, el / 60, embedded / max(el, 1e-3))


if __name__ == "__main__":
    main()
