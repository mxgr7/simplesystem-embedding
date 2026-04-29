#!/usr/bin/env python3
"""Materialize the joined ESCI flat dataset for the annotator.

Joins (DuckDB):
    candidates  ⋈  queries        on query_id
                ⋈  offers_embedded_full  on candidate_id = id

`offer_embedding` is intentionally excluded — the annotator never needs it.

Inputs:
  /data/datasets/queries_offers_esci/queries.parquet
  /data/datasets/queries_offers_esci/candidates.parquet
  /data/datasets/offers_embedded_full.parquet/  (16 buckets)

Output:
  /data/datasets/queries_offers_esci/queries_offers_merged.parquet/part-0.parquet
"""
from pathlib import Path
import time

import duckdb

ROOT = Path("/data/datasets/queries_offers_esci")
OFFERS_GLOB = "/data/datasets/offers_embedded_full.parquet/*.parquet"
OUT_DIR = ROOT / "queries_offers_merged.parquet"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "part-0.parquet"

SQL = f"""
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
        FROM read_parquet('{ROOT}/queries.parquet')
    ),
    c AS (
        SELECT *
        FROM read_parquet('{ROOT}/candidates.parquet')
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
        row_number() OVER (
            ORDER BY c.query_id, c.rank_hybrid_classified NULLS LAST,
                     c.rank_vector NULLS LAST, c.rank_bm25 NULLS LAST
        ) AS example_id,
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


def main():
    print(f"[duckdb] running join → {OUT_PATH}")
    con = duckdb.connect()
    # Use enough memory + threads to comfortably scan the 16 offer buckets
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='32GB'")

    t0 = time.time()
    con.execute(SQL)
    elapsed = time.time() - t0
    print(f"[duckdb] join completed in {elapsed:.1f}s")

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_PATH}')").fetchone()[0]
    n_null_name = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{OUT_PATH}') WHERE name IS NULL"
    ).fetchone()[0]
    n_distinct_q = con.execute(
        f"SELECT COUNT(DISTINCT query_id) FROM read_parquet('{OUT_PATH}')"
    ).fetchone()[0]
    sz = OUT_PATH.stat().st_size
    print(f"[verify] rows: {n:,}")
    print(f"[verify] distinct query_id: {n_distinct_q:,}")
    print(f"[verify] rows with NULL name (unmatched candidate): {n_null_name:,}")
    print(f"[verify] file size: {sz/1e6:.1f} MB")


if __name__ == "__main__":
    main()
