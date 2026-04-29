#!/usr/bin/env python3
"""Materialize the mono-label-negatives candidates into a flat parquet ready
for the annotator.

Mirrors materialize_esci_dataset.py for the additional rows produced by
retrieve_mono_label_negatives.py. example_id continues from
MAX(example_id) + 1 in the existing labeled parquet so IDs stay globally
unique across the original dataset and this addendum.

Inputs:
  /data/datasets/queries_offers_esci/queries.parquet
  /data/datasets/queries_offers_esci/candidates_mono_label_negatives.parquet
  /data/datasets/queries_offers_esci/queries_offers_merged_labeled.parquet/part-0.parquet
  /data/datasets/offers_embedded_full.parquet/  (16 buckets)

Output:
  /data/datasets/queries_offers_esci/queries_offers_merged_mono_label_negatives.parquet/part-0.parquet
"""
import time
from pathlib import Path

import duckdb

ROOT = Path("/data/datasets/queries_offers_esci")
LABELED = ROOT / "queries_offers_merged_labeled.parquet" / "part-0.parquet"
CANDIDATES = ROOT / "candidates_mono_label_negatives.parquet"
QUERIES = ROOT / "queries.parquet"
OFFERS_GLOB = "/data/datasets/offers_embedded_full.parquet/*.parquet"
OUT_DIR = ROOT / "queries_offers_merged_mono_label_negatives.parquet"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "part-0.parquet"


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='32GB'")

    max_existing = con.execute(
        f"SELECT MAX(example_id) FROM read_parquet('{LABELED}')"
    ).fetchone()[0]
    print(f"[duckdb] continuing example_id from {max_existing + 1:,}")

    sql = f"""
    COPY (
        WITH q AS (
            SELECT
                query_id,
                qt_raw            AS query_term,
                normalized_qt,
                frequency_band,
                hit_band,
                mpn_shape,
                hit_count_at_search_time,
                platform_language
            FROM read_parquet('{QUERIES}')
        ),
        c AS (
            SELECT * FROM read_parquet('{CANDIDATES}')
        ),
        o AS (
            SELECT
                id,
                name,
                manufacturerName            AS manufacturer_name,
                description,
                categoryPaths               AS category_paths,
                ean,
                article_number,
                manufacturerArticleNumber   AS manufacturer_article_number,
                manufacturerArticleType     AS manufacturer_article_type
            FROM read_parquet('{OFFERS_GLOB}')
        )
        SELECT
            ({max_existing} + row_number() OVER (
                ORDER BY c.query_id, c.rank_hybrid_classified
            )) AS example_id,
            c.query_id,
            c.candidate_id,
            q.query_term,
            q.normalized_qt,
            q.frequency_band,
            q.hit_band,
            q.mpn_shape,
            q.hit_count_at_search_time,
            q.platform_language,
            c.rank_hybrid_classified,
            c.rank_vector,
            c.rank_bm25,
            c.score_hybrid_classified,
            c.score_vector,
            c.score_bm25,
            c.source_legs,
            o.name,
            o.manufacturer_name,
            o.description,
            o.category_paths,
            o.ean,
            o.article_number,
            o.manufacturer_article_number,
            o.manufacturer_article_type,
            c.retrieved_at
        FROM c
        INNER JOIN q ON c.query_id = q.query_id
        LEFT JOIN o  ON c.candidate_id = o.id
        ORDER BY example_id
    )
    TO '{OUT_PATH}'
    (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 9);
    """

    print(f"[duckdb] running join → {OUT_PATH}")
    t0 = time.time()
    con.execute(sql)
    elapsed = time.time() - t0
    print(f"[duckdb] join completed in {elapsed:.1f}s")

    n, n_null_name, n_distinct_q, min_ex, max_ex = con.execute(f"""
        SELECT COUNT(*),
               SUM(CASE WHEN name IS NULL THEN 1 ELSE 0 END),
               COUNT(DISTINCT query_id),
               MIN(example_id), MAX(example_id)
        FROM read_parquet('{OUT_PATH}')
    """).fetchone()
    sz = OUT_PATH.stat().st_size
    print(f"[verify] rows: {n:,}")
    print(f"[verify] distinct query_id: {n_distinct_q:,}")
    print(f"[verify] example_id range: {min_ex:,}..{max_ex:,}")
    print(f"[verify] rows with NULL name (unmatched candidate): {n_null_name:,}")
    print(f"[verify] file size: {sz/1e6:.1f} MB")


if __name__ == "__main__":
    main()
