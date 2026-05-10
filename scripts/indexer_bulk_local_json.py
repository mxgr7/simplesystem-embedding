"""F9 bulk indexer CLI for the local MongoDB export JSON shards.

This is a simplified wrapper around `indexer.bulk.run_bulk_indexer` for
our current operating assumption:

  - source data always lives on the local filesystem
  - source format is always gzipped JSONL (`atlas-*.json.gz`)
  - collection layout matches `/data/mongodb-export-2026-03-04/`

It removes the generic source-selection surface from `scripts/indexer_bulk.py`
(S3, parquet source mode, source-format switching) while preserving the
Milvus/TEI/Redis/tuning knobs that are still operationally useful.

Typical full run:

    uv run python scripts/indexer_bulk_local_json.py \
        --articles-collection articles_v6 \
        --offers-collection offers_v6 \
        --milvus-uri http://localhost:19530 \
        --tei-url http://localhost:8080 \
        --redis-url redis://localhost:6379/0 \
        --sink-mode bulk_insert \
        --duckdb-temp-dir /data/duckdb_tmp \
        --duckdb-temp-dir-limit-gb 800 \
        --n-chunks 16 \
        --index-cycle

Smoke run against a subset of offer shards:

    uv run python scripts/indexer_bulk_local_json.py \
        --offers-glob 'atlas-fkxrb3-shard-0.*.json.gz' \
        --articles-collection articles_v6_smoke \
        --offers-collection offers_v6_smoke \
        --milvus-uri http://localhost:19530 \
        --tei-url http://localhost:8080 \
        --redis-url redis://localhost:6379/0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.bulk import run_bulk_indexer  # noqa: E402
from indexer.bulk_insert import BulkInsertConfig  # noqa: E402


COLLECTION_DIRS = {
    "offers": "offers",
    "pricings": "pricings",
    "markers": "coreArticleMarkers",
    "cans": "customerArticleNumbers",
}


def _build_globs(args):
    base = Path(args.source_root).expanduser()
    offers_dir = base / COLLECTION_DIRS["offers"]
    pricings_dir = base / COLLECTION_DIRS["pricings"]
    markers_dir = base / COLLECTION_DIRS["markers"]
    cans_dir = base / COLLECTION_DIRS["cans"]

    required_dirs = [offers_dir, pricings_dir, markers_dir, cans_dir]
    missing = [str(p) for p in required_dirs if not p.is_dir()]
    if missing:
        sys.exit(
            "source root is missing expected collection dirs: "
            + ", ".join(missing)
        )

    offers_pattern = args.offers_glob or "atlas-*.json.gz"
    return {
        "offers_glob": str(offers_dir / offers_pattern),
        "pricings_glob": str(pricings_dir / "atlas-*.json.gz"),
        "markers_glob": str(markers_dir / "atlas-*.json.gz"),
        "cans_glob": str(cans_dir / "atlas-*.json.gz"),
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_argument_group("Source")
    src.add_argument(
        "--source-root",
        default="/data/mongodb-export-2026-03-04",
        help="Local root containing offers/, pricings/, coreArticleMarkers/, "
             "customerArticleNumbers/ (default: /data/mongodb-export-2026-03-04).",
    )
    src.add_argument(
        "--offers-glob",
        default="",
        help="Filename pattern for the offers shard subset under offers/ "
             "(default 'atlas-*.json.gz'). pricings/markers/cans always read "
             "every shard for join completeness.",
    )

    sink = p.add_argument_group("Milvus sinks")
    sink.add_argument("--milvus-uri", default="http://localhost:19530")
    sink.add_argument("--articles-collection", required=True,
                      help="Versioned target name (e.g. articles_v6). Must exist.")
    sink.add_argument("--offers-collection", required=True,
                      help="Versioned target name (e.g. offers_v6). Must exist.")

    embed = p.add_argument_group("Embedding")
    embed.add_argument("--tei-url", required=True,
                       help="Base URL of TEI service (POST /embed contract).")
    embed.add_argument("--redis-url", required=True,
                       help="Redis URL for the hash-keyed embedding cache.")
    embed.add_argument("--tei-batch-size", type=int, default=4096,
                       help="Texts per TEI HTTP call (default 4096).")
    embed.add_argument("--tei-concurrency", type=int, default=8,
                       help="Parallel TEI HTTP calls per embed phase (default 8).")

    tune = p.add_argument_group("Tunables")
    tune.add_argument("--article-batch-size", type=int, default=1000,
                      help="Article rows per Milvus upsert + TEI cache lookup.")
    tune.add_argument("--offer-batch-size", type=int, default=5000,
                      help="Offer rows per Milvus upsert.")
    tune.add_argument("--duckdb-temp-dir", default=None,
                      help="DuckDB disk-spill directory (default per-connection temp).")
    tune.add_argument("--duckdb-temp-dir-limit-gb", type=int, default=500,
                      help="Hard cap on DuckDB disk-spill (default 500 GB).")
    tune.add_argument("--duckdb-memory-limit-gb", type=int, default=80,
                      help="DuckDB memory_limit (default 80 GB).")
    tune.add_argument("--duckdb-threads", type=int, default=0,
                      help="DuckDB worker threads (default 0 = system default).")
    tune.add_argument("--n-chunks", type=int, default=1,
                      help="Process the data in N hash-partitioned chunks.")
    tune.add_argument("--start-chunk", type=int, default=0,
                      help="Start chunk index when resuming a chunked run.")

    sink_mode = p.add_argument_group(
        "Sink mode",
        "Default 'upsert' is per-row idempotent + queryable immediately. "
        "'bulk_insert' stages parquet to MinIO/S3 and submits do_bulk_insert.",
    )
    sink_mode.add_argument("--sink-mode", choices=["upsert", "bulk_insert"], default="upsert",
                           help="How to write to Milvus (default 'upsert').")
    sink_mode.add_argument("--bulk-insert-s3-endpoint", default="http://localhost:9000",
                           help="MinIO/S3 endpoint URL for parquet staging.")
    sink_mode.add_argument("--bulk-insert-s3-bucket", default="a-bucket",
                           help="S3 bucket for staged parquets.")
    sink_mode.add_argument("--bulk-insert-s3-prefix", default="f9_indexer",
                           help="Key prefix under the bucket.")
    sink_mode.add_argument("--bulk-insert-s3-access-key", default="minioadmin",
                           help="S3 access key.")
    sink_mode.add_argument("--bulk-insert-s3-secret-key", default="minioadmin",
                           help="S3 secret key.")
    sink_mode.add_argument("--bulk-insert-stage-dir", default="/tmp/f9_indexer_stage",
                           help="Local directory for parquet staging before upload.")
    sink_mode.add_argument("--bulk-insert-chunk-rows", type=int, default=1_000_000,
                           help="Rows per parquet chunk (default 1M).")
    sink_mode.add_argument("--bulk-insert-upload-workers", type=int, default=4,
                           help="Parallel upload + submit workers (default 4).")
    sink_mode.add_argument("--bulk-insert-checkpoint", default="",
                           help="Optional JSON checkpoint path for resume.")
    sink_mode.add_argument("--bulk-insert-retry-attempts", type=int, default=5,
                           help="Total attempts for upload + do_bulk_insert.")
    sink_mode.add_argument("--bulk-insert-poll-interval-s", type=float, default=1.0,
                           help="Polling cadence for do_bulk_insert state.")
    sink_mode.add_argument("--index-cycle", action="store_true",
                           help="Drop indexes before bulk insert, rebuild after flush.")
    sink_mode.add_argument("--vector-index", default="HNSW",
                           choices=["HNSW", "IVF_FLAT", "FLAT"],
                           help="Vector index type for article index rebuild.")

    p.add_argument("--log-level", default="INFO",
                   help="Python logging level (default INFO).")

    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    globs = _build_globs(args)

    bulk_cfg = None
    if args.sink_mode == "bulk_insert":
        bulk_cfg = BulkInsertConfig(
            s3_endpoint=args.bulk_insert_s3_endpoint,
            s3_bucket=args.bulk_insert_s3_bucket,
            s3_prefix=args.bulk_insert_s3_prefix,
            s3_access_key=args.bulk_insert_s3_access_key,
            s3_secret_key=args.bulk_insert_s3_secret_key,
            stage_dir=Path(args.bulk_insert_stage_dir),
            chunk_rows=args.bulk_insert_chunk_rows,
            upload_workers=args.bulk_insert_upload_workers,
            checkpoint_path=Path(args.bulk_insert_checkpoint) if args.bulk_insert_checkpoint else None,
            retry_attempts=args.bulk_insert_retry_attempts,
            poll_interval_s=args.bulk_insert_poll_interval_s,
        )

    stats = run_bulk_indexer(
        offers_glob=globs["offers_glob"],
        pricings_glob=globs["pricings_glob"],
        markers_glob=globs["markers_glob"],
        cans_glob=globs["cans_glob"],
        milvus_uri=args.milvus_uri,
        articles_collection=args.articles_collection,
        offers_collection=args.offers_collection,
        tei_url=args.tei_url,
        redis_url=args.redis_url,
        article_batch_size=args.article_batch_size,
        offer_batch_size=args.offer_batch_size,
        tei_batch_size=args.tei_batch_size,
        tei_concurrency=args.tei_concurrency,
        duckdb_temp_dir=args.duckdb_temp_dir,
        duckdb_temp_dir_limit_gb=args.duckdb_temp_dir_limit_gb,
        duckdb_memory_limit_gb=args.duckdb_memory_limit_gb,
        duckdb_threads=args.duckdb_threads,
        n_chunks=args.n_chunks,
        start_chunk=args.start_chunk,
        sink_mode=args.sink_mode,
        bulk_insert=bulk_cfg,
        index_cycle=args.index_cycle,
        vector_index=args.vector_index,
    )

    print("DONE")
    print(f"  duckdb stage:  {stats.duckdb_seconds:>12.1f}s")
    print(f"  article sink:  {stats.article_upsert_seconds:>12.1f}s")
    print(f"  offer sink:    {stats.offer_upsert_seconds:>12.1f}s")
    print(f"  total:         {stats.total_seconds:>12.1f}s")
    print(f"  raw offers:    {stats.raw_offer_count:>12,d}")
    print(f"  raw pricings:  {stats.raw_pricing_count:>12,d}")
    print(f"  raw markers:   {stats.raw_marker_count:>12,d}")
    print(f"  raw cans:      {stats.raw_can_count:>12,d}")
    print(f"  articles:      {stats.article_count:>12,d}")
    print(f"  offer rows:    {stats.offer_row_count:>12,d}")
    print(f"  TEI cache hit: {stats.tei.hits:>12,d}")
    print(f"  TEI cache miss:{stats.tei.misses:>12,d}")


if __name__ == "__main__":
    main()
