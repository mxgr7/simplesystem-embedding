"""Build the (vendor_id, article_number) -> [article_hash] lookup parquet
that the v5 importer expects at <export-dir>/article_hashes_v2/bucket=NN/*.parquet.

Source: gzipped Mongo offer exports from scripts/dump_mongo_offers.py.
Hashing: identical to scripts/prewarm_v2_missing.py's `compute_article_hash`
macro — SHA-256 over 8 NUL-joined canonical fields, first 32 hex chars.
Output: partitioned by `bucket` (16 buckets by default) so the v5 importer's
default glob `article_hashes_v2/**/*.parquet` picks it up.

Run:
    uv run python scripts/build_article_hashes_lookup.py \\
        --input-glob /data/datasets/mongo_offers_export_<YYYYMMDD>/vendor_*.json.gz \\
        --out-dir   /data/datasets/mongo_offers_export_<YYYYMMDD>/article_hashes_v2
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import duckdb


_DUCKDB_MACROS = r"""
CREATE OR REPLACE MACRO _v2_canon_paths(category_paths) AS
    encode(array_to_string(
        array_sort(
            list_transform(
                list_filter(
                    COALESCE(category_paths, []::STRUCT(elements VARCHAR[])[]),
                    cp -> cp.elements IS NOT NULL AND len(cp.elements) > 0
                ),
                cp -> array_to_string(cp.elements, '¦')
            )
        ),
        chr(30)
    ));

CREATE OR REPLACE MACRO compute_article_hash(
    a_name, a_mfg, a_desc, category_paths,
    a_ean, a_article_number, a_mfg_article_number, a_mfg_article_type
) AS
    substr(
        sha256(
            encode(COALESCE(a_name, '')) || '\x00'::BLOB ||
            encode(COALESCE(a_mfg, '')) || '\x00'::BLOB ||
            encode(COALESCE(a_desc, '')) || '\x00'::BLOB ||
            _v2_canon_paths(category_paths) || '\x00'::BLOB ||
            encode(COALESCE(a_ean, '')) || '\x00'::BLOB ||
            encode(COALESCE(a_article_number, '')) || '\x00'::BLOB ||
            encode(COALESCE(a_mfg_article_number, '')) || '\x00'::BLOB ||
            encode(COALESCE(a_mfg_article_type, ''))
        ),
        1, 32
    );
"""

# Same field schema as prewarm_v2_missing.py:_PINNED_COLUMNS plus vendorId
# at the top level.
_PINNED_COLUMNS = (
    "{"
    "'vendorId': 'VARCHAR', "
    "'articleNumber': 'VARCHAR', "
    "'offer': 'STRUCT(offerParams STRUCT("
    "\"name\" VARCHAR, "
    "manufacturerName VARCHAR, "
    "\"description\" VARCHAR, "
    "categoryPaths STRUCT(elements VARCHAR[])[], "
    "ean VARCHAR, "
    "manufacturerArticleNumber VARCHAR, "
    "manufacturerArticleType VARCHAR"
    "))'"
    "}"
)


def build(input_glob, out_dir, buckets):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads = {os.cpu_count() or 8}")
    con.execute("SET enable_progress_bar = false")
    con.execute("SET preserve_insertion_order = false")
    # The GROUP BY hash table for 455M rows -> ~150-250M (vendor,artno) groups
    # would not fit in RAM on this 7GB box; cap memory and spill to disk.
    con.execute("SET memory_limit = '4GB'")
    tmp_dir = out_dir.parent / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{tmp_dir}'")
    con.execute(_DUCKDB_MACROS)

    sql = f"""
    COPY (
      WITH src AS (
        SELECT
          vendorId      AS vendor_id,
          articleNumber AS article_number,
          offer.offerParams.name                       AS name,
          offer.offerParams.manufacturerName           AS manufacturer_name,
          offer.offerParams.description                AS description,
          offer.offerParams.categoryPaths              AS category_paths,
          offer.offerParams.ean                        AS ean,
          offer.offerParams.manufacturerArticleNumber  AS manufacturer_article_number,
          offer.offerParams.manufacturerArticleType    AS manufacturer_article_type
        FROM read_json(
          '{input_glob}',
          format='newline_delimited',
          compression='gzip',
          maximum_object_size=67108864,
          columns={_PINNED_COLUMNS}
        )
        WHERE vendorId IS NOT NULL AND articleNumber IS NOT NULL
      ),
      hashed AS (
        SELECT
          vendor_id,
          article_number,
          compute_article_hash(name, manufacturer_name, description, category_paths,
                               ean, article_number, manufacturer_article_number,
                               manufacturer_article_type) AS article_hash
        FROM src
      )
      SELECT
        vendor_id,
        article_number,
        list_distinct(list(article_hash)) AS article_hashes,
        CAST(abs(hash(vendor_id || '|' || article_number)) % {buckets} AS INT) AS bucket
      FROM hashed
      GROUP BY vendor_id, article_number
    ) TO '{out_dir}'
    (FORMAT PARQUET, COMPRESSION 'zstd', PARTITION_BY (bucket), OVERWRITE_OR_IGNORE);
    """
    t0 = time.time()
    print(f"building article_hashes_v2 lookup from {input_glob} -> {out_dir} "
          f"({buckets} buckets) ...", flush=True)
    con.execute(sql)
    n = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_dir}/**/*.parquet')"
    ).fetchone()[0]
    n_hashes = con.execute(
        f"SELECT SUM(len(article_hashes)) "
        f"FROM read_parquet('{out_dir}/**/*.parquet')"
    ).fetchone()[0]
    print(f"DONE: {n:,} (vendor, articleNumber) pairs, "
          f"{n_hashes:,} total hash references in {time.time()-t0:.1f}s",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-glob", required=True,
                    help="Glob for the gzipped Mongo offer exports.")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory; will hold bucket=NN/*.parquet.")
    ap.add_argument("--buckets", type=int, default=16)
    args = ap.parse_args()
    build(args.input_glob, args.out_dir, args.buckets)


if __name__ == "__main__":
    main()
