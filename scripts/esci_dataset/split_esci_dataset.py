#!/usr/bin/env python3
"""Add a stratified 80/10/10 train/val/test `split` column to the labeled
ESCI dataset.

Stratification: by (frequency_band, hit_band), so each split sees the same
mix of head/torso/tail × zero/1-9/10-99/100-999/1000+. Splits are at the
query_id level (no query appears in more than one split — required for valid
ranking evaluation).

Reproducible: hash(query_id || seed) used inside NTILE(100). Seed = 42.

In-place: overwrites
    /data/datasets/queries_offers_esci/queries_offers_merged_labeled.parquet/part-0.parquet
"""
from pathlib import Path

import duckdb

ROOT = Path("/data/datasets/queries_offers_esci")
LABELED_DIR = ROOT / "queries_offers_merged_labeled.parquet"
LABELED = LABELED_DIR / "part-0.parquet"
TMP = LABELED_DIR / "part-0.parquet.tmp"
SEED = "esci-split-seed-42"

con = duckdb.connect()
con.execute("PRAGMA threads=8")

# Build per-query split assignment in a temp view, then re-write the
# labeled parquet with the new column appended.
con.execute(f"""
    CREATE OR REPLACE TEMP VIEW q_split AS
    WITH q AS (
        SELECT DISTINCT query_id, frequency_band, hit_band
        FROM read_parquet('{LABELED}')
    ),
    q_bucket AS (
        SELECT query_id,
               NTILE(100) OVER (
                   PARTITION BY frequency_band, hit_band
                   ORDER BY hash(CAST(query_id AS VARCHAR) || '{SEED}')
               ) AS bucket
        FROM q
    )
    SELECT query_id,
           CASE
               WHEN bucket <= 80 THEN 'train'
               WHEN bucket <= 90 THEN 'val'
               ELSE 'test'
           END AS split
    FROM q_bucket;
""")

print("[split] writing labeled parquet with new `split` column...")
con.execute(f"""
    COPY (
        SELECT lab.*, qs.split
        FROM read_parquet('{LABELED}') lab
        LEFT JOIN q_split qs ON lab.query_id = qs.query_id
        ORDER BY lab.example_id
    )
    TO '{TMP}'
    (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 9);
""")
TMP.replace(LABELED)

print("[verify] split distribution:")
print()

# Per-split row + query counts
print("  rows per split:")
for split, n_rows, n_queries in con.execute(f"""
    SELECT split, COUNT(*), COUNT(DISTINCT query_id)
    FROM read_parquet('{LABELED}')
    GROUP BY 1 ORDER BY 1
""").fetchall():
    print(f"    {split:6} rows={n_rows:>7,}  queries={n_queries:>6,}")

# Cell counts per (frequency_band, hit_band, split)
print()
print("  query stratification (queries per cell, train/val/test):")
print(f"    {'freq':6} {'hit':10} {'train':>7} {'val':>5} {'test':>5}")
for fb, hb, tr, va, te in con.execute(f"""
    WITH per_q AS (
        SELECT query_id, ANY_VALUE(frequency_band) AS frequency_band,
                          ANY_VALUE(hit_band)        AS hit_band,
                          ANY_VALUE(split)           AS split
        FROM read_parquet('{LABELED}')
        GROUP BY query_id
    )
    SELECT frequency_band, hit_band,
           SUM(CASE WHEN split='train' THEN 1 ELSE 0 END),
           SUM(CASE WHEN split='val'   THEN 1 ELSE 0 END),
           SUM(CASE WHEN split='test'  THEN 1 ELSE 0 END)
    FROM per_q
    GROUP BY 1, 2
    ORDER BY 1, 2
""").fetchall():
    print(f"    {fb:6} {hb:10} {tr:>7,} {va:>5,} {te:>5,}")

# Label distribution per split (sanity)
print()
print("  label × split:")
print(f"    {'label':12} {'train':>9} {'val':>7} {'test':>7}")
for label, tr, va, te in con.execute(f"""
    SELECT COALESCE(label, '∅NULL'),
           SUM(CASE WHEN split='train' THEN 1 ELSE 0 END),
           SUM(CASE WHEN split='val'   THEN 1 ELSE 0 END),
           SUM(CASE WHEN split='test'  THEN 1 ELSE 0 END)
    FROM read_parquet('{LABELED}')
    GROUP BY 1
    ORDER BY 1
""").fetchall():
    print(f"    {label:12} {tr:>9,} {va:>7,} {te:>7,}")

# Sanity: any query_id in more than one split?
leak = con.execute(f"""
    SELECT COUNT(*)
    FROM (
        SELECT query_id
        FROM read_parquet('{LABELED}')
        GROUP BY 1
        HAVING COUNT(DISTINCT split) > 1
    )
""").fetchone()[0]
print(f"\n  query_id leaks across splits: {leak}  (must be 0)")

# Final size
sz = LABELED.stat().st_size
print(f"\n  output: {LABELED} ({sz/1e6:.1f} MB)")
