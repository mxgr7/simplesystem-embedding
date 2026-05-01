"""Build offer_projected.parquet/chunk_KKKK.parquet from offers.parquet/.

For each hash bucket K (0..N_CHUNKS-1):
  Read offers.parquet/chunk_KKKK.parquet (already partitioned by hash)
  Run the offer-derived SQL (no JOINs to pricings/markers/cans)
  Write offer_projected.parquet/chunk_KKKK.parquet

Per-chunk indexer can then load this directly and skip the heavy
`projected` CTE work — saving ~40-60% of per-chunk materialise time.

Output schema mirrors the offer-derived columns of `_PROJECTION_CTE_SQL`,
plus carry-through fields for chunk-time JOIN-dependent reconstruction:
  vk, ak (join keys)
  vendor_id, article_number, id
  name, manufacturerName, ean, catalog_version_ids, delivery_time_days_max
  eclass5_code, eclass7_code, s2class_code (INTEGER[])
  relationship_accessory_for, relationship_spare_part_for, relationship_similar_to (VARCHAR[])
  category_l1..l5 (VARCHAR[])
  features (VARCHAR[])
  inline_pricings_open, inline_pricings_closed (raw STRUCTs from offer.pricings)
  inline_can_pair (STRUCT(value, version_id) | NULL)
  article_hash (32-char VARCHAR)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.duckdb_projection import init_macros, offer_projected_build_sql


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path,
                   default=Path("/data/datasets/f9_indexer/s3-cache-parquet"))
    p.add_argument("--n-chunks", type=int, default=16)
    p.add_argument("--duckdb-memory-limit-gb", type=int, default=80)
    p.add_argument("--duckdb-threads", type=int, default=32)
    p.add_argument("--duckdb-temp-dir", type=str,
                   default="/data/datasets/f9_indexer/duckdb_tmp")
    p.add_argument("--row-group-size", type=int, default=500_000)
    p.add_argument("--compression", default="zstd")
    args = p.parse_args()

    src_dir = args.source_dir / "offers.parquet"
    out_dir = args.source_dir / "offer_projected.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.is_dir():
        raise SystemExit(f"expected partitioned offers parquet dir at {src_dir}")

    Path(args.duckdb_temp_dir).mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.duckdb_memory_limit_gb}GB'")
    con.execute(f"SET threads={args.duckdb_threads}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{args.duckdb_temp_dir}'")
    init_macros(con)

    grand_t0 = time.time()
    total_in, total_out, total_bytes = 0, 0, 0

    for k in range(args.n_chunks):
        src = src_dir / f"chunk_{k:04d}.parquet"
        out = out_dir / f"chunk_{k:04d}.parquet"
        if not src.exists():
            print(f"  chunk_{k:04d}: SOURCE MISSING ({src})", flush=True)
            continue

        select_sql = offer_projected_build_sql(source_table_or_glob=str(src))
        t0 = time.time()
        con.execute(
            f"COPY ({select_sql}) TO '{out}' "
            f"(FORMAT PARQUET, COMPRESSION {args.compression}, "
            f"ROW_GROUP_SIZE {args.row_group_size})"
        )
        n_in = con.execute(f"SELECT count(*) FROM read_parquet('{src}')").fetchone()[0]
        n_out = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
        out_bytes = out.stat().st_size
        elapsed = time.time() - t0
        print(
            f"  chunk_{k:04d}: in={n_in:>11,}  out={n_out:>11,}  "
            f"{out_bytes/1e6:>5.0f} MB  {elapsed:>5.1f}s",
            flush=True,
        )
        total_in += n_in
        total_out += n_out
        total_bytes += out_bytes

    print()
    print(f"ALL DONE in {(time.time() - grand_t0)/60:.1f} min")
    print(f"  total input rows:    {total_in:,}")
    print(f"  total output rows:   {total_out:,}")
    print(f"  total output bytes:  {total_bytes/1e9:.2f} GB")
    print(f"  output dir:          {out_dir}")


if __name__ == "__main__":
    main()
