"""One-time conversion: gzipped JSONL source → parquet.

Each source collection (offers, pricings, markers, cans) is loaded with
the same column schema the bulk indexer uses, then written as parquet.
Output preserves the schema so `load_raw_collections(...)` can swap in
the parquet path with no changes.

After conversion, point the bulk indexer at `--local-cache` containing
`*.parquet` files instead of `*.json.gz` — `load_raw_collections`
auto-dispatches on file extension.

Why convert: gzipped JSON is per-file single-threaded for parsing.
With ~4116 pricings shards on a shared NVMe, raw-load takes ~25 min
(disk-I/O + JSON parse bound). Parquet is column-parallel + zero-copy
+ predicate-pushdown; the same load takes ~3 min.

Usage:
    uv run python scripts/convert_source_to_parquet.py \\
        --source-dir /data/datasets/f9_indexer/s3-cache \\
        --target-dir /data/datasets/f9_indexer/s3-cache-parquet \\
        --duckdb-memory-limit-gb 100 \\
        --duckdb-threads 32
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import duckdb
from indexer.duckdb_projection import (
    RAW_OFFER_COLUMNS, RAW_PRICING_COLUMNS,
    RAW_MARKER_COLUMNS, RAW_CAN_COLUMNS,
)

log = logging.getLogger("convert")


def convert_one(
    con: duckdb.DuckDBPyConnection,
    *,
    src_glob: str,
    target_path: Path,
    columns: dict,
    name: str,
    row_group_size: int = 1_000_000,
    compression: str = "zstd",
) -> tuple[int, int]:
    """Read JSONL from glob → write parquet with the explicit column
    schema. Returns (rows, output_bytes)."""
    log.info("[%s] reading %s", name, src_glob)
    t0 = time.time()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream-style: read_json directly into COPY ... TO parquet without
    # materialising the full table in memory. DuckDB writes parquet
    # rowgroups as it parses, so peak memory ≈ rowgroup size, not table size.
    con.execute(
        "COPY (SELECT * FROM read_json(?, format='newline_delimited', "
        "maximum_object_size=?, columns=?)) "
        f"TO '{target_path}' (FORMAT PARQUET, COMPRESSION {compression}, "
        f"ROW_GROUP_SIZE {row_group_size})",
        [src_glob, 256 * 1024 * 1024, columns],
    )

    elapsed = time.time() - t0
    n = con.execute(f"SELECT count(*) FROM read_parquet('{target_path}')").fetchone()[0]
    out_bytes = target_path.stat().st_size
    log.info(
        "[%s] %s rows → %s (%.1f GB) in %.1fs",
        name, f"{n:,}", target_path.name,
        out_bytes / 1e9, elapsed,
    )
    return n, out_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", required=True, type=Path,
                   help="Root of source data (offers/, pricings/, etc.)")
    p.add_argument("--target-dir", required=True, type=Path,
                   help="Root for parquet output (mirrors source layout)")
    p.add_argument("--duckdb-memory-limit-gb", type=int, default=80,
                   help="DuckDB memory cap (default 80 GB)")
    p.add_argument("--duckdb-threads", type=int, default=0,
                   help="DuckDB threads (0 = system default)")
    p.add_argument("--duckdb-temp-dir", type=str, default="",
                   help="DuckDB spill dir (default per-conn temp)")
    p.add_argument("--row-group-size", type=int, default=1_000_000)
    p.add_argument("--compression", default="zstd",
                   choices=["snappy", "zstd", "gzip", "uncompressed"])
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{args.duckdb_memory_limit_gb}GB'")
    if args.duckdb_threads > 0:
        con.execute(f"SET threads = {args.duckdb_threads}")
    if args.duckdb_temp_dir:
        Path(args.duckdb_temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{args.duckdb_temp_dir}'")
    con.execute("SET preserve_insertion_order = false")

    src = args.source_dir
    tgt = args.target_dir

    jobs = [
        ("offers",   "offers/atlas-*.json.gz",                "offers.parquet",   RAW_OFFER_COLUMNS),
        ("pricings", "pricings/atlas-*.json.gz",              "pricings.parquet", RAW_PRICING_COLUMNS),
        ("markers",  "coreArticleMarkers/atlas-*.json.gz",    "markers.parquet",  RAW_MARKER_COLUMNS),
        ("cans",     "customerArticleNumbers/atlas-*.json.gz","cans.parquet",     RAW_CAN_COLUMNS),
    ]

    grand_t0 = time.time()
    for name, src_glob_rel, target_rel, columns in jobs:
        convert_one(
            con,
            src_glob=str(src / src_glob_rel),
            target_path=tgt / target_rel,
            columns=columns,
            name=name,
            row_group_size=args.row_group_size,
            compression=args.compression,
        )
    log.info("ALL DONE in %.1f min", (time.time() - grand_t0) / 60)


if __name__ == "__main__":
    main()
