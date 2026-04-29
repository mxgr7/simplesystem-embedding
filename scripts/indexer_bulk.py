"""F9 bulk indexer CLI — read raw Mongo Atlas snapshot from S3 (or
local cache), JOIN + project + aggregate in DuckDB, embed via TEI
(Redis-cached by article hash), upsert into the paired
`articles_v{N}` + `offers_v{N+1}` Milvus collections.

Typical production invocation (S3 source, dedicated TEI GPU box, Redis
sidecar on Milvus host):

    AWS_PROFILE=simplesystem uv run python scripts/indexer_bulk.py \\
        --s3-base s3://mongo-atlas-snapshot-for-lab/exported_snapshots/.../prod \\
        --milvus-uri http://milvus.internal:19530 \\
        --articles-collection articles_v4 \\
        --offers-collection offers_v5 \\
        --tei-url http://tei.internal:8080 \\
        --redis-url redis://redis.internal:6379/0 \\
        --duckdb-temp-dir /mnt/scratch/duckdb \\
        --duckdb-temp-dir-limit-gb 800

Local-cache invocation (smoke test against pulled shards under ~/s3-cache):

    uv run python scripts/indexer_bulk.py \\
        --local-cache ~/s3-cache \\
        --offers-glob 'atlas-fkxrb3-shard-0.*.json.gz' \\
        --milvus-uri http://localhost:19530 \\
        --articles-collection articles_v4_smoke \\
        --offers-collection offers_v5_smoke \\
        --tei-url http://localhost:8080 \\
        --redis-url redis://localhost:6379/0

The collections must already exist (run `create_articles_collection.py`
+ `create_offers_collection.py` first per the F9 alias workflow). This
script writes to the versioned names directly; the alias swing is a
separate operator step (see `scripts/MILVUS_ALIAS_WORKFLOW.md`).
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

# Mongo Atlas snapshot collection names — these become the per-collection
# subdirectory under both the S3 base and any local cache.
COLLECTIONS = {
    "offers":   "offers",
    "pricings": "pricings",
    "markers":  "coreArticleMarkers",
    "cans":     "customerArticleNumbers",
}


def _build_globs(args: argparse.Namespace) -> dict[str, str]:
    """Resolve the 4 per-collection globs from `--s3-base` or
    `--local-cache`. `--offers-glob` overrides the offers shard
    pattern (everything else stays at `atlas-*.json.gz`) — useful for
    targeted smoke runs against a single source shard while still
    pulling all pricings/markers/cans for full join coverage."""
    if args.s3_base and args.local_cache:
        sys.exit("--s3-base and --local-cache are mutually exclusive")
    if not args.s3_base and not args.local_cache:
        sys.exit("provide --s3-base or --local-cache")

    base = args.s3_base.rstrip("/") if args.s3_base else str(Path(args.local_cache).expanduser())
    sep = "/"

    offers_pattern = args.offers_glob or "atlas-*.json.gz"
    return {
        "offers_glob":   f"{base}{sep}{COLLECTIONS['offers']}{sep}{offers_pattern}",
        "pricings_glob": f"{base}{sep}{COLLECTIONS['pricings']}{sep}atlas-*.json.gz",
        "markers_glob":  f"{base}{sep}{COLLECTIONS['markers']}{sep}atlas-*.json.gz",
        "cans_glob":     f"{base}{sep}{COLLECTIONS['cans']}{sep}atlas-*.json.gz",
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_argument_group("Source")
    src.add_argument(
        "--s3-base",
        help="S3 prefix containing per-collection subdirs (offers/, pricings/, …). "
             "Example: s3://mongo-atlas-snapshot-for-lab/.../prod",
    )
    src.add_argument(
        "--local-cache",
        help="Local directory containing per-collection subdirs. Mutually "
             "exclusive with --s3-base. Use for smoke tests with pre-pulled shards.",
    )
    src.add_argument(
        "--offers-glob", default="",
        help="Filename pattern for the offers shard (default 'atlas-*.json.gz'). "
             "pricings/markers/cans always read every shard — joins need the full set.",
    )

    sink = p.add_argument_group("Milvus sinks")
    sink.add_argument("--milvus-uri", default="http://localhost:19530")
    sink.add_argument("--articles-collection", required=True,
                      help="Versioned target name (e.g. articles_v4). Must exist.")
    sink.add_argument("--offers-collection", required=True,
                      help="Versioned target name (e.g. offers_v5). Must exist.")

    embed = p.add_argument_group("Embedding")
    embed.add_argument("--tei-url", required=True,
                       help="Base URL of TEI service (POST /embed contract).")
    embed.add_argument("--redis-url", required=True,
                       help="Redis URL for the hash-keyed embedding cache. "
                            "Example: redis://localhost:6379/0")
    embed.add_argument("--tei-batch-size", type=int, default=64,
                       help="Texts per TEI HTTP call (default 64).")

    tune = p.add_argument_group("Tunables")
    tune.add_argument("--article-batch-size", type=int, default=1000,
                      help="Article rows per Milvus upsert + TEI cache lookup.")
    tune.add_argument("--offer-batch-size", type=int, default=5000,
                      help="Offer rows per Milvus upsert (no embedding here, can go bigger).")
    tune.add_argument("--duckdb-temp-dir", default=None,
                      help="DuckDB disk-spill directory (default: per-connection temp). "
                           "Point at a large NVMe for production runs.")
    tune.add_argument("--duckdb-temp-dir-limit-gb", type=int, default=500,
                      help="Hard cap on DuckDB disk-spill (default 500 GB). "
                           "Bump for full-catalog runs (158 GB offers + 28 GB pricings, "
                           "uncompressed factor ~3-4×).")
    tune.add_argument("--s3-region", default="eu-central-1",
                      help="S3 region for the credential_chain secret on the SOURCE side "
                           "(reading raw shards). Ignored on --local-cache runs. The MinIO "
                           "sink uses --bulk-insert-s3-* flags below instead.")

    sink = p.add_argument_group(
        "Sink mode",
        "Default 'upsert' is per-row idempotent + queryable immediately (right for smoke "
        "runs and small collections). 'bulk_insert' stages a parquet to MinIO and submits "
        "do_bulk_insert — ~50–100K rows/sec vs ~800/sec for upsert (right for a full reindex).",
    )
    sink.add_argument("--sink-mode", choices=["upsert", "bulk_insert"], default="upsert",
                      help="How to write to Milvus (default 'upsert').")
    sink.add_argument("--bulk-insert-s3-endpoint", default="http://localhost:9000",
                      help="MinIO/S3 endpoint URL for the parquet staging bucket "
                           "(default localhost:9000 = local docker-compose MinIO).")
    sink.add_argument("--bulk-insert-s3-bucket", default="a-bucket",
                      help="S3 bucket for staged parquets (default 'a-bucket' — Milvus's "
                           "default MinIO bucket).")
    sink.add_argument("--bulk-insert-s3-prefix", default="f9_indexer",
                      help="Key prefix under the bucket. Default 'f9_indexer'.")
    sink.add_argument("--bulk-insert-s3-access-key", default="minioadmin",
                      help="S3 access key (default 'minioadmin').")
    sink.add_argument("--bulk-insert-s3-secret-key", default="minioadmin",
                      help="S3 secret key (default 'minioadmin').")
    sink.add_argument("--bulk-insert-stage-dir", default="/tmp/f9_indexer_stage",
                      help="Local directory for parquet staging before upload "
                           "(default /tmp/f9_indexer_stage). Sized to hold the full "
                           "articles + offers parquets — at production scale ~60 GB.")

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
        duckdb_temp_dir=args.duckdb_temp_dir,
        duckdb_temp_dir_limit_gb=args.duckdb_temp_dir_limit_gb,
        s3_region=args.s3_region,
        sink_mode=args.sink_mode,
        bulk_insert=bulk_cfg,
    )

    print()
    print("=== Bulk indexer run summary ===")
    print(f"  raw_offers:    {stats.raw_offer_count:>12,}")
    print(f"  raw_pricings:  {stats.raw_pricing_count:>12,}")
    print(f"  raw_markers:   {stats.raw_marker_count:>12,}")
    print(f"  raw_cans:      {stats.raw_can_count:>12,}")
    print(f"  articles:      {stats.article_count:>12,}  upserted in {stats.article_upsert_seconds:.1f}s")
    print(f"  offers:        {stats.offer_row_count:>12,}  upserted in {stats.offer_upsert_seconds:.1f}s")
    print(f"  duckdb stage:  {stats.duckdb_seconds:>12.1f}s")
    print(f"  total wall:    {stats.total_seconds:>12.1f}s ({stats.total_seconds / 60:.1f} min)")
    print()
    print("  TEI cache:")
    print(f"    hits:        {stats.tei.hits:>12,}")
    print(f"    misses:      {stats.tei.misses:>12,}")
    print(f"    tei calls:   {stats.tei.tei_calls:>12,}")
    print(f"    bytes wrote: {stats.tei.bytes_written / 1e6:>11.1f} MB")
    if stats.article_count:
        hit_rate = stats.tei.hits / stats.article_count
        print(f"    hit rate:    {hit_rate:>12.1%}")

    if args.sink_mode == "bulk_insert":
        for label, b in (("articles", stats.articles_bulk_insert),
                         ("offers",   stats.offers_bulk_insert)):
            print()
            print(f"  bulk_insert ({label}):")
            print(f"    rows:           {b.rows_written:>12,}")
            print(f"    parquet size:   {b.parquet_bytes / 1e9:>10.2f} GB")
            print(f"    parquet write:  {b.write_seconds:>10.1f}s")
            print(f"    upload:         {b.upload_seconds:>10.1f}s")
            print(f"    bulk_insert:    {b.bulk_insert_seconds:>10.1f}s")


if __name__ == "__main__":
    main()
