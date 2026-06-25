"""Catch up the article_hashes_v2 lookup with a Kafka-derived offer delta.

The original lookup at <export-dir>/article_hashes_v2/bucket=NN/*.parquet was
built by scripts/build_article_hashes_lookup.py from the 2026-05-27 Mongo
export. Since then offers changed; scripts/catchup_kafka_offers.py collected
every changed (vendorId, articleNumber) and re-fetched the current offers.

This script produces a NEW lookup that reflects those changes, via a keyed
merge on (vendor_id, article_number):

    new = (original ANTI JOIN all-changed-keys)   -- drop updated AND deleted
          UNION ALL
          delta                                    -- re-add the still-existing,
                                                   --   freshly hashed
  * updated key  -> old row dropped, fresh row from delta
  * deleted key  -> dropped, not re-added (in keys.tsv.gz, absent from delta)
  * new key      -> added by delta
  * untouched    -> survives the anti-join

The hash + bucket formulas are imported from build_article_hashes_lookup so
they stay bit-identical. Output goes to a fresh --out-dir; the original is left
untouched for rollback. Swap is a manual `mv` once validated.

Heavy job — run on the milvus box (32 cores / 247G). Example:
    uv run python scripts/update_article_hashes_lookup.py \\
        --original-map  /data/datasets/mongo_offers_export_20260527/article_hashes_v2 \\
        --catchup-dir   /data/datasets/catchup_20260529 \\
        --out-dir       /data/datasets/mongo_offers_export_20260527/article_hashes_v2_20260529 \\
        --threads 32 --memory-limit 200GB
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_article_hashes_lookup import _DUCKDB_MACROS, _PINNED_COLUMNS  # noqa: E402


def update(original_map, catchup_dir, out_dir, buckets, threads,
           memory_limit, apply_deletes):
    original_map = Path(original_map)
    catchup_dir = Path(catchup_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys_glob = str(catchup_dir / "keys.tsv.gz")
    offers_glob = str(catchup_dir / "vendor_*.json.gz")
    orig_glob = str(original_map / "**" / "*.parquet")

    con = duckdb.connect()
    con.execute(f"SET threads = {threads}")
    con.execute("SET enable_progress_bar = false")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    tmp_dir = out_dir.parent / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{tmp_dir}'")
    con.execute(_DUCKDB_MACROS)

    # The set of keys to drop from the original map. When apply_deletes is on,
    # this is EVERY changed key (so deletes disappear). When off, we only drop
    # keys that still exist (deletes keep their stale original row).
    if apply_deletes:
        drop_keys_cte = f"""
        drop_keys AS (
          SELECT DISTINCT vendor_id, article_number
          FROM read_csv('{keys_glob}', delim='\\t', header=false, quote='',
                        escape='', nullstr='',
                        columns={{'vendor_id': 'VARCHAR', 'article_number': 'VARCHAR'}})
        )"""
    else:
        drop_keys_cte = """
        drop_keys AS (SELECT vendor_id, article_number FROM delta)"""

    sql = f"""
    COPY (
      WITH delta_src AS (
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
          '{offers_glob}',
          format='newline_delimited', compression='gzip',
          maximum_object_size=67108864, columns={_PINNED_COLUMNS}
        )
        WHERE vendorId IS NOT NULL AND articleNumber IS NOT NULL
      ),
      delta_hashed AS (
        SELECT vendor_id, article_number,
               compute_article_hash(name, manufacturer_name, description, category_paths,
                                    ean, article_number, manufacturer_article_number,
                                    manufacturer_article_type) AS article_hash
        FROM delta_src
      ),
      delta AS (
        SELECT
          vendor_id,
          article_number,
          list_distinct(list(article_hash)) AS article_hashes,
          CAST(abs(hash(vendor_id || '|' || article_number)) % {buckets} AS INT) AS bucket
        FROM delta_hashed
        GROUP BY vendor_id, article_number
      ),
      {drop_keys_cte},
      original AS (
        SELECT vendor_id, article_number, article_hashes,
               CAST(bucket AS INT) AS bucket
        FROM read_parquet('{orig_glob}', hive_partitioning=true)
      ),
      kept AS (
        SELECT o.* FROM original o ANTI JOIN drop_keys d USING (vendor_id, article_number)
      )
      SELECT * FROM kept
      UNION ALL
      SELECT * FROM delta
    ) TO '{out_dir}'
    (FORMAT PARQUET, COMPRESSION 'zstd', PARTITION_BY (bucket), OVERWRITE_OR_IGNORE);
    """

    t0 = time.time()
    print(f"merging delta from {catchup_dir} into copy of {original_map} "
          f"-> {out_dir}  (apply_deletes={apply_deletes}, threads={threads})",
          flush=True)
    con.execute(sql)
    elapsed = time.time() - t0

    # Validation counts (cheap — parquet metadata / small reads).
    orig_n = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{orig_glob}', hive_partitioning=true)"
    ).fetchone()[0]
    new_glob = str(out_dir / "**" / "*.parquet")
    new_n = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{new_glob}')"
    ).fetchone()[0]
    delta_n = con.execute(f"""
        SELECT COUNT(*) FROM (
          SELECT DISTINCT vendorId, articleNumber
          FROM read_json('{offers_glob}', format='newline_delimited',
                         compression='gzip', maximum_object_size=67108864,
                         columns={_PINNED_COLUMNS})
          WHERE vendorId IS NOT NULL AND articleNumber IS NOT NULL
        )""").fetchone()[0]

    print("\n=== merge summary ===", flush=True)
    print(f"  original rows : {orig_n:,}")
    print(f"  delta rows    : {delta_n:,}  (still-existing changed keys)")
    print(f"  new map rows  : {new_n:,}")
    print(f"  net change    : {new_n - orig_n:+,}")
    print(f"  elapsed       : {elapsed:.1f}s -> {out_dir}", flush=True)
    print("Validate, then swap:", flush=True)
    print(f"  mv {original_map} {original_map}.pre_20260529 && "
          f"mv {out_dir} {original_map}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original-map", required=True,
                    help="Existing article_hashes_v2 dir (holds bucket=NN/*.parquet).")
    ap.add_argument("--catchup-dir", required=True,
                    help="catchup_kafka_offers.py output dir (keys.tsv.gz + vendor_*.json.gz).")
    ap.add_argument("--out-dir", required=True,
                    help="New lookup dir; original is left untouched.")
    ap.add_argument("--buckets", type=int, default=16)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--memory-limit", default="200GB")
    ap.add_argument("--no-apply-deletes", dest="apply_deletes",
                    action="store_false",
                    help="Keep stale rows for keys whose offer was deleted "
                         "(default: drop them so the map reflects deletes).")
    args = ap.parse_args()
    update(args.original_map, args.catchup_dir, args.out_dir, args.buckets,
           args.threads, args.memory_limit, args.apply_deletes)


if __name__ == "__main__":
    main()
