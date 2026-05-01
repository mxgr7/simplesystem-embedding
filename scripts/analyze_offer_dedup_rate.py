"""Compute the duplication rate of offer-template inputs *after* the
embedding model's 256-token truncation, over the offers dataset.

We render each offer with the `offer_template` from
`configs/data/default.yaml`, tokenize with the production tokenizer
(`intfloat/multilingual-e5-base`), truncate to 256 tokens, and count
how many rows map to identical truncated token sequences.

Each parquet bucket is processed by a worker process. The worker writes
its per-bucket hashes to a `.npz` file; the driver concatenates and
counts at the end. Set `--workers N` to control parallelism (default: 8).

Skips buckets whose `.npz` already exists, so re-running picks up where
a previous invocation left off.

Run:
    uv run python scripts/analyze_offer_dedup_rate.py
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DATA_CFG_PATH = REPO_ROOT / "configs/data/default.yaml"
OFFERS_GLOB = "/data/datasets/offers_flat.parquet/bucket=*.parquet"
MODEL_NAME = "intfloat/multilingual-e5-base"
MAX_LEN = 256
DEFAULT_OUT_DIR = Path("/data/datasets/_offer_dedup_hashes")

# columns the offer_template actually consumes
PROJECT_COLS = [
    "name",
    "manufacturerName",
    "description",
    "categoryPaths",
    "ean",
    "article_number",
    "manufacturerArticleNumber",
    "manufacturerArticleType",
]

# offers_flat.parquet stores camelCase; default.yaml column_mapping uses
# snake_case. Rename source -> canonical so the renderer feeds the
# template correctly.
COLUMN_RENAME = {
    "manufacturerName": "manufacturer_name",
    "categoryPaths": "category_paths",
    "manufacturerArticleNumber": "manufacturer_article_number",
    "manufacturerArticleType": "manufacturer_article_type",
}


def _hash64(payload: bytes) -> int:
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "big", signed=False
    )


def process_bucket(args: tuple[str, str, int, int]) -> tuple[str, int, float]:
    bucket_path, out_path, batch_size, parquet_batch = args

    # silence the HF parallelism warning and let the rust tokenizer
    # use its own threadpool inside the worker
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    from embedding_train.rendering import RowTextRenderer
    from transformers import AutoTokenizer

    cfg = OmegaConf.load(DATA_CFG_PATH)
    cfg.column_rename = COLUMN_RENAME
    renderer = RowTextRenderer(cfg)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    pf = pq.ParquetFile(bucket_path)
    n_rows = pf.metadata.num_rows

    # preallocate; rendered_empty rows would shrink the array — we trim
    # at the end. uint64 is enough for 159M items (expected
    # false-collision count ≈ 7e-4).
    full_hashes = np.empty(n_rows, dtype=np.uint64)
    trunc_hashes = np.empty(n_rows, dtype=np.uint64)
    lengths = np.empty(n_rows, dtype=np.uint16)

    idx = 0
    pending_texts: list[str] = []
    pending_full: list[int] = []

    def flush() -> None:
        nonlocal idx
        if not pending_texts:
            return
        enc = tok(
            pending_texts,
            truncation=True,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        for ids, fh in zip(enc["input_ids"], pending_full):
            arr = np.asarray(ids, dtype=np.int32)
            th = _hash64(arr.tobytes())
            full_hashes[idx] = fh
            trunc_hashes[idx] = th
            lengths[idx] = len(ids) if len(ids) < 65535 else 65535
            idx += 1
        pending_texts.clear()
        pending_full.clear()

    t0 = time.time()
    for batch in pf.iter_batches(batch_size=parquet_batch, columns=PROJECT_COLS):
        for row in batch.to_pylist():
            txt = renderer.render_offer_text(row)
            if not txt:
                continue
            fh = _hash64(txt.encode("utf-8"))
            pending_texts.append(txt)
            pending_full.append(fh)
            if len(pending_texts) >= batch_size:
                flush()
    flush()

    full_hashes = full_hashes[:idx]
    trunc_hashes = trunc_hashes[:idx]
    lengths = lengths[:idx]
    np.savez(
        out_path, full=full_hashes, trunc=trunc_hashes, lens=lengths,
        n_input_rows=np.asarray([n_rows], dtype=np.int64),
    )

    return Path(bucket_path).name, idx, time.time() - t0


def collect_and_report(out_dir: Path) -> None:
    files = sorted(out_dir.glob("bucket=*.npz"))
    print(f"\n=== combining {len(files)} bucket result files ===")

    full_arrs, trunc_arrs, len_arrs = [], [], []
    n_input_total = 0
    for f in files:
        d = np.load(f)
        full_arrs.append(d["full"])
        trunc_arrs.append(d["trunc"])
        len_arrs.append(d["lens"])
        n_input_total += int(d["n_input_rows"][0])

    full = np.concatenate(full_arrs)
    trunc = np.concatenate(trunc_arrs)
    lens = np.concatenate(len_arrs)
    n = len(full)
    print(f"input rows total:    {n_input_total:,}")
    print(f"rendered (non-empty): {n:,}")
    print(f"rendered_empty:      {n_input_total - n:,}")

    print("\ncomputing unique counts...")
    sf = np.sort(full)
    n_full_unique = int(1 + np.count_nonzero(np.diff(sf)))
    del sf
    st = np.sort(trunc)
    n_trunc_unique = int(1 + np.count_nonzero(np.diff(st)))

    print(f"\n=== summary (n = {n:,}) ===")
    print(
        f"unique full-text inputs:     {n_full_unique:>14,}  "
        f"({n_full_unique/n*100:7.4f}% of rendered rows)"
    )
    print(
        f"unique truncated (256 tok):  {n_trunc_unique:>14,}  "
        f"({n_trunc_unique/n*100:7.4f}% of rendered rows)"
    )
    print()
    print(
        f"full-text dup rate:          {(1 - n_full_unique/n)*100:7.4f}%   "
        f"({n - n_full_unique:,} rows redundant under full-text dedup)"
    )
    print(
        f"truncated dup rate:          {(1 - n_trunc_unique/n)*100:7.4f}%   "
        f"({n - n_trunc_unique:,} rows redundant under 256-tok dedup)"
    )
    print(
        f"extra dups exposed by trunc: "
        f"{(n_full_unique - n_trunc_unique)/n*100:7.4f}% of rows "
        f"({n_full_unique - n_trunc_unique:,} rows unique pre-trunc but "
        "collide post-trunc)"
    )

    over_limit = int((lens >= MAX_LEN).sum())
    print()
    print(
        f"rows hitting truncation (>= {MAX_LEN} tok): "
        f"{over_limit:,} / {n:,} = {over_limit/n*100:.2f}%"
    )
    for pct in (50, 75, 90, 95, 99):
        v = int(np.percentile(lens, pct))
        print(f"  p{pct:>2} token length: {v}")

    # cluster-size distribution for truncated dups
    print("\ncluster-size histogram (truncated input groups):")
    _, counts = np.unique(trunc, return_counts=True)
    bins = Counter(counts.tolist())
    print(f"  total clusters: {len(counts):,}")
    for size in sorted(bins.keys()):
        if size <= 1 and len(bins) > 1:
            continue
        # only print up to size 30 explicitly, then aggregate
        if size > 30:
            break
        print(f"  size {size:>3}: {bins[size]:>10,} clusters  "
              f"({bins[size]*size/n*100:6.4f}% of rows)")
    big = sum(c for s, c in bins.items() if s > 30)
    big_rows = sum(s * c for s, c in bins.items() if s > 30)
    if big:
        print(f"  size >30: {big:,} clusters covering {big_rows:,} rows "
              f"({big_rows/n*100:6.4f}% of rows)")

    # top-10 largest dup clusters (size only — we don't keep raw text)
    top = sorted(bins.items(), reverse=True)[:10]
    print("\ntop-10 cluster sizes (truncated):")
    for size, count in top:
        print(f"  size {size:>6}: {count:,} clusters")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-per-bucket", type=int, default=None,
                    help="cap rows per bucket for testing (default: full)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--parquet-batch", type=int, default=8192)
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--report-only", action="store_true",
                    help="skip processing, just collect existing .npz files")
    args = ap.parse_args()

    if args.rows_per_bucket is not None:
        sys.exit("--rows-per-bucket is no longer supported; use the prior "
                 "version of this script for sampling runs")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        collect_and_report(out_dir)
        return

    files = sorted(glob.glob(OFFERS_GLOB))
    if not files:
        sys.exit(f"no parquet files matched {OFFERS_GLOB}")

    tasks: list[tuple[str, str, int, int]] = []
    for f in files:
        out_path = out_dir / f"{Path(f).stem}.npz"
        if out_path.exists():
            print(f"skip (already done): {out_path.name}")
            continue
        tasks.append((f, str(out_path), args.batch_size, args.parquet_batch))

    if not tasks:
        print("all buckets already processed")
        collect_and_report(out_dir)
        return

    print(f"buckets to process: {len(tasks)} / {len(files)}")
    print(f"workers: {args.workers}  batch_size: {args.batch_size}")

    t0 = time.time()
    ctx = get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for name, n, dt in pool.imap_unordered(process_bucket, tasks):
            print(
                f"[done] {name}: rendered={n:,} dt={dt:.1f}s "
                f"({n/dt:.0f} rows/s)  elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    print(f"\nall buckets done in {time.time()-t0:.1f}s")
    collect_and_report(out_dir)


if __name__ == "__main__":
    main()
