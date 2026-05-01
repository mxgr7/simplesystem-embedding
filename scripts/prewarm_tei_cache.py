"""Pre-populate the TEI Redis cache from `offers_embedded_full.parquet`.

The bulk indexer (`indexer/bulk.py`) consults `indexer/tei_cache.py`
before calling TEI; on a hit it skips the GPU. The parquet bundles
already-computed fp16 embeddings for every offer-grouped article
(produced by the same `RowTextRenderer` the indexer now reuses for
fallback misses), so populating the cache up front lets the bulk run
skip TEI for every article whose identity matches a parquet row.

Idempotent: each row maps to a deterministic key (`tei:{HASH_VERSION}:
{compute_article_hash(row)}`) and a fixed-bytes value; re-running
overwrites with identical content. Resumable per-bucket via
`--progress-dir/bucket=NN.done` markers — a bucket completes
atomically before its marker is written, so a crashed run resumes at
the bucket boundary, not mid-bucket.

Run example:
    uv run python scripts/prewarm_tei_cache.py \
        --parquet-dir /data/datasets/offers_embedded_full.parquet \
        --redis-url redis://localhost:6379/0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import redis

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.projection import HASH_VERSION, compute_article_hash  # noqa: E402

log = logging.getLogger("prewarm")

# Columns required to recompute the article hash. Matches the field set
# in `compute_article_hash`; if that grows, add here.
HASH_COLUMNS: list[str] = [
    "name",
    "manufacturerName",
    "description",
    "categoryPaths",
    "ean",
    "article_number",
    "manufacturerArticleNumber",
    "manufacturerArticleType",
]
EMBED_COLUMN = "offer_embedding"

# Per `indexer/tei_cache.py` — must match or `_redis_mget` rejects the
# value as the wrong byte length.
VECTOR_DIM = 128
VECTOR_BYTES = VECTOR_DIM * 2  # fp16


def _cache_key(article_hash: str) -> str:
    return f"tei:{HASH_VERSION}:{article_hash}"


def _embedding_array(table) -> np.ndarray:
    """Pull the embedding column out as a (n_rows, VECTOR_DIM) fp16
    array. Avoids the ~100× slowdown of materialising halffloat lists
    via `to_pylist`."""
    col = table.column(EMBED_COLUMN).combine_chunks()
    flat = col.values.to_numpy(zero_copy_only=False)
    if flat.dtype != np.float16:
        # PyArrow surfaces halffloat as np.float16 — guard against a
        # pyarrow upgrade silently changing this and corrupting Redis.
        raise RuntimeError(
            f"expected fp16 embedding values, got dtype={flat.dtype}"
        )
    n = table.num_rows
    if flat.size != n * VECTOR_DIM:
        raise RuntimeError(
            f"embedding flat size {flat.size} != n*dim ({n}*{VECTOR_DIM})"
        )
    return flat.reshape(n, VECTOR_DIM)


def prewarm_bucket(
    bucket_path: Path,
    r: redis.Redis,
    *,
    chunk_rows: int,
    pipeline_ops: int,
) -> tuple[int, int]:
    """Stream one parquet bucket into Redis. Returns (rows_written,
    bytes_written)."""
    t0 = time.time()
    log.info("[%s] reading parquet", bucket_path.name)
    table = pq.read_table(
        bucket_path, columns=HASH_COLUMNS + [EMBED_COLUMN]
    )
    n = table.num_rows
    log.info("[%s] %d rows", bucket_path.name, n)

    emb = _embedding_array(table)
    hash_table = table.select(HASH_COLUMNS)

    rows_written = 0
    bytes_written = 0
    last_log = time.time()

    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        chunk = hash_table.slice(start, end - start).to_pylist()

        pipe = r.pipeline(transaction=False)
        n_in_pipe = 0
        for j, row in enumerate(chunk):
            h = compute_article_hash(row)
            key = _cache_key(h)
            val = emb[start + j].tobytes()
            assert len(val) == VECTOR_BYTES
            pipe.set(key, val)
            n_in_pipe += 1
            if n_in_pipe >= pipeline_ops:
                pipe.execute()
                pipe = r.pipeline(transaction=False)
                n_in_pipe = 0
        if n_in_pipe:
            pipe.execute()

        rows_written += len(chunk)
        bytes_written += len(chunk) * VECTOR_BYTES

        now = time.time()
        if now - last_log >= 5.0:
            elapsed = now - t0
            rate = rows_written / max(elapsed, 1e-3)
            log.info(
                "[%s] %d/%d rows  elapsed=%.1fs  rate=%.0f/s  written=%.2f GB",
                bucket_path.name, rows_written, n, elapsed, rate,
                bytes_written / 1e9,
            )
            last_log = now

    elapsed = time.time() - t0
    log.info(
        "[%s] DONE  rows=%d  elapsed=%.1fs  rate=%.0f/s  written=%.2f GB",
        bucket_path.name, rows_written, elapsed,
        rows_written / max(elapsed, 1e-3), bytes_written / 1e9,
    )
    return rows_written, bytes_written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet-dir", required=True, type=Path)
    p.add_argument("--redis-url", required=True)
    p.add_argument(
        "--progress-dir", type=Path,
        default=Path("/tmp/prewarm_tei_cache.progress"),
        help="Per-bucket .done markers for resumability",
    )
    p.add_argument(
        "--restart", action="store_true",
        help="Wipe progress markers and re-process every bucket",
    )
    p.add_argument(
        "--bucket", type=int,
        help="Process only this bucket index (smoke testing)",
    )
    p.add_argument(
        "--chunk-rows", type=int, default=50_000,
        help="Rows per to_pylist materialisation slice",
    )
    p.add_argument(
        "--pipeline-ops", type=int, default=5_000,
        help="Redis pipeline flush threshold",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    bucket_files = sorted(args.parquet_dir.glob("bucket=*.parquet"))
    if not bucket_files:
        raise SystemExit(f"no bucket files in {args.parquet_dir}")
    if args.bucket is not None:
        bucket_files = [
            f for f in bucket_files
            if f.name == f"bucket={args.bucket:02d}.parquet"
        ]
        if not bucket_files:
            raise SystemExit(f"bucket {args.bucket} not found")

    args.progress_dir.mkdir(parents=True, exist_ok=True)
    if args.restart:
        for m in args.progress_dir.glob("bucket=*.done"):
            m.unlink()
        log.info("cleared progress markers under %s", args.progress_dir)

    log.info("connecting to %s", args.redis_url)
    r = redis.Redis.from_url(args.redis_url)
    r.ping()

    grand_t0 = time.time()
    total_rows = 0
    total_bytes = 0
    for bf in bucket_files:
        marker = args.progress_dir / f"{bf.stem}.done"
        if marker.exists():
            log.info("[%s] already done, skipping", bf.name)
            continue
        rows, b = prewarm_bucket(
            bf, r,
            chunk_rows=args.chunk_rows,
            pipeline_ops=args.pipeline_ops,
        )
        marker.write_text(f"{rows} {b}\n")
        total_rows += rows
        total_bytes += b

    log.info(
        "ALL DONE  buckets=%d  rows=%d  bytes=%.2f GB  elapsed=%.1fs  dbsize=%d",
        len(bucket_files), total_rows, total_bytes / 1e9,
        time.time() - grand_t0, r.dbsize(),
    )


if __name__ == "__main__":
    main()
