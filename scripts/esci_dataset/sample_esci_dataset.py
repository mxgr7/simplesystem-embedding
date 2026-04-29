#!/usr/bin/env python3
"""Build a stratified 100-row sample of the merged ESCI dataset.

20 rows from each of the 5 hit_band values {zero, 1-9, 10-99, 100-999, 1000+}.
Output is a separate parquet directory so the annotator can glob it
without pulling the full 644K-row file.
"""
from pathlib import Path

import duckdb

ROOT = Path("/data/datasets/queries_offers_esci")
SRC = ROOT / "queries_offers_merged.parquet" / "part-0.parquet"
OUT_DIR = ROOT / "queries_offers_merged_sample100.parquet"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "part-0.parquet"

SQL = f"""
COPY (
    WITH ranked AS (
        SELECT *,
               row_number() OVER (
                   PARTITION BY hit_band
                   ORDER BY hash(example_id || '-seed42')
               ) AS bucket_rank
        FROM read_parquet('{SRC}')
        WHERE name IS NOT NULL
    )
    SELECT * EXCLUDE (bucket_rank)
    FROM ranked
    WHERE bucket_rank <= 20
      AND hit_band IN ('zero', '1-9', '10-99', '100-999', '1000+')
    ORDER BY hit_band, bucket_rank
)
TO '{OUT_PATH}'
(FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 9);
"""


def main():
    con = duckdb.connect()
    con.execute(SQL)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_PATH}')").fetchone()[0]
    by_hb = con.execute(
        f"SELECT hit_band, COUNT(*) FROM read_parquet('{OUT_PATH}') "
        "GROUP BY 1 ORDER BY 1"
    ).fetchall()
    by_legs = con.execute(
        f"SELECT array_to_string(source_legs, ','), COUNT(*) "
        f"FROM read_parquet('{OUT_PATH}') GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    print(f"sample rows: {n}")
    print("by hit_band:")
    for hb, c in by_hb:
        print(f"  {hb:8} {c}")
    print("by source_legs:")
    for legs, c in by_legs:
        print(f"  {legs!r:30} {c}")
    print(f"output: {OUT_PATH} ({OUT_PATH.stat().st_size/1e3:.1f} KB)")


if __name__ == "__main__":
    main()
