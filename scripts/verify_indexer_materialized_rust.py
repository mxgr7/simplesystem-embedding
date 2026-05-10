#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer import duckdb_projection as dp  # noqa: E402
from indexer.duckdb_projection import (  # noqa: E402
    aggregate_articles_from_collections,
    load_raw_collections,
    project_offer_rows_from_collections,
)
from indexer.s2class_mapper import S2CLASS_SOURCE_KEYS_DESC, mapping_for_version_key  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Verify Rust materialized parquet against DuckDB reference")
    p.add_argument("--offers-glob", required=True)
    p.add_argument("--pricings-glob", required=True)
    p.add_argument("--markers-glob", required=True)
    p.add_argument("--cans-glob", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--sample-mismatches", type=int, default=10)
    return p.parse_args()


def canon_price_list(values):
    values = values or []
    out = []
    for v in values:
        out.append({
            "price": round(float(v["price"]), 6),
            "currency": v.get("currency") or "",
            "priority": int(v.get("priority") or 0),
            "sourcePriceListId": v.get("sourcePriceListId") or "",
        })
    out.sort(key=lambda x: (x["currency"], x["priority"], x["sourcePriceListId"], x["price"]))
    return out


def canon_customer_numbers(values):
    values = values or []
    out = []
    for v in values:
        out.append({
            "value": v.get("value") or "",
            "version_ids": sorted(list(v.get("version_ids") or [])),
        })
    out.sort(key=lambda x: x["value"])
    return out


def canon_scalar_list(values, sort_values=False):
    values = list(values or [])
    if sort_values:
        values.sort()
    return values


def canon_offer(row):
    return {
        "id": row["id"],
        "article_hash": row["article_hash"],
        "_placeholder_vector": [round(float(x), 6) for x in (row["_placeholder_vector"] or [])],
        "ean": row["ean"] or "",
        "article_number": row["article_number"] or "",
        "vendor_id": row["vendor_id"] or "",
        "catalog_version_id": row["catalog_version_id"] or "",
        "prices": canon_price_list(row["prices"]),
        "delivery_time_days_max": int(row["delivery_time_days_max"] or 0),
        "core_marker_enabled_sources": canon_scalar_list(row["core_marker_enabled_sources"], sort_values=True),
        "core_marker_disabled_sources": canon_scalar_list(row["core_marker_disabled_sources"], sort_values=True),
        "features": canon_scalar_list(row["features"]),
        "relationship_accessory_for": canon_scalar_list(row["relationship_accessory_for"]),
        "relationship_spare_part_for": canon_scalar_list(row["relationship_spare_part_for"]),
        "relationship_similar_to": canon_scalar_list(row["relationship_similar_to"]),
        "price_list_ids": canon_scalar_list(row["price_list_ids"]),
        "currencies": canon_scalar_list(row["currencies"]),
        "eur_price_min": round(float(row["eur_price_min"]), 6),
        "eur_price_max": round(float(row["eur_price_max"]), 6),
        "chf_price_min": round(float(row["chf_price_min"]), 6),
        "chf_price_max": round(float(row["chf_price_max"]), 6),
        "huf_price_min": round(float(row["huf_price_min"]), 6),
        "huf_price_max": round(float(row["huf_price_max"]), 6),
        "pln_price_min": round(float(row["pln_price_min"]), 6),
        "pln_price_max": round(float(row["pln_price_max"]), 6),
        "gbp_price_min": round(float(row["gbp_price_min"]), 6),
        "gbp_price_max": round(float(row["gbp_price_max"]), 6),
        "czk_price_min": round(float(row["czk_price_min"]), 6),
        "czk_price_max": round(float(row["czk_price_max"]), 6),
        "cny_price_min": round(float(row["cny_price_min"]), 6),
        "cny_price_max": round(float(row["cny_price_max"]), 6),
    }


def canon_article(row):
    return {
        "article_hash": row["article_hash"],
        "name": row["name"] or "",
        "manufacturerName": row["manufacturerName"] or "",
        "category_l1": canon_scalar_list(row["category_l1"]),
        "category_l2": canon_scalar_list(row["category_l2"]),
        "category_l3": canon_scalar_list(row["category_l3"]),
        "category_l4": canon_scalar_list(row["category_l4"]),
        "category_l5": canon_scalar_list(row["category_l5"]),
        "eclass5_code": list(row["eclass5_code"] or []),
        "eclass7_code": list(row["eclass7_code"] or []),
        "s2class_code": list(row["s2class_code"] or []),
        "text_codes": row["text_codes"] or "",
        "customer_article_numbers": canon_customer_numbers(row["customer_article_numbers"]),
        "eur_price_min": round(float(row["eur_price_min"]), 6),
        "eur_price_max": round(float(row["eur_price_max"]), 6),
        "chf_price_min": round(float(row["chf_price_min"]), 6),
        "chf_price_max": round(float(row["chf_price_max"]), 6),
        "huf_price_min": round(float(row["huf_price_min"]), 6),
        "huf_price_max": round(float(row["huf_price_max"]), 6),
        "pln_price_min": round(float(row["pln_price_min"]), 6),
        "pln_price_max": round(float(row["pln_price_max"]), 6),
        "gbp_price_min": round(float(row["gbp_price_min"]), 6),
        "gbp_price_max": round(float(row["gbp_price_max"]), 6),
        "czk_price_min": round(float(row["czk_price_min"]), 6),
        "czk_price_max": round(float(row["czk_price_max"]), 6),
        "cny_price_min": round(float(row["cny_price_min"]), 6),
        "cny_price_max": round(float(row["cny_price_max"]), 6),
    }


def fast_init_macros(con):
    con.execute(dp._MACROS_SQL)
    rows = []
    for version_key in S2CLASS_SOURCE_KEYS_DESC:
        mapping = mapping_for_version_key(version_key)
        for from_code, to_code in mapping.items():
            rows.append((version_key, from_code, to_code))
    tmp = Path('/tmp/verify_indexer_materialized_rust_s2map.csv')
    with open(tmp, 'w', encoding='utf-8') as handle:
        handle.write('version_key,from_code,to_code\n')
        for version_key, from_code, to_code in rows:
            handle.write(f'{version_key},{from_code},{to_code}\n')
    con.execute(
        """
        CREATE OR REPLACE TABLE s2map AS
        SELECT version_key, from_code::INTEGER AS from_code, to_code::INTEGER AS to_code,
               expand_eclass(to_code::INTEGER) AS s2_groups
        FROM read_csv(?, header=true, columns={
            'version_key': 'VARCHAR',
            'from_code': 'INTEGER',
            'to_code': 'INTEGER'
        })
        """,
        [str(tmp)],
    )


def fetch_dicts(con, sql):
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    return [dict(zip(cols, row)) for row in rows]


def compare_maps(name, key_field, expected, actual, canon, sample_mismatches):
    exp = {row[key_field]: canon(row) for row in expected}
    act = {row[key_field]: canon(row) for row in actual}
    summary = {
        "expected_rows": len(exp),
        "actual_rows": len(act),
        "missing_in_actual": 0,
        "missing_in_expected": 0,
        "mismatched": 0,
    }
    missing_actual = sorted(set(exp) - set(act))
    missing_expected = sorted(set(act) - set(exp))
    mismatched = []
    for key in sorted(set(exp) & set(act)):
        if exp[key] != act[key]:
            mismatched.append(key)
    summary["missing_in_actual"] = len(missing_actual)
    summary["missing_in_expected"] = len(missing_expected)
    summary["mismatched"] = len(mismatched)
    print(name, json.dumps(summary, indent=2))
    for key in missing_actual[:sample_mismatches]:
        print(name, "missing_in_actual", key)
    for key in missing_expected[:sample_mismatches]:
        print(name, "missing_in_expected", key)
    for key in mismatched[:sample_mismatches]:
        print(name, "mismatch", key)
        print("expected", json.dumps(exp[key], ensure_ascii=False))
        print("actual", json.dumps(act[key], ensure_ascii=False))
    failed = summary["missing_in_actual"] or summary["missing_in_expected"] or summary["mismatched"]
    return summary, bool(failed)


def main():
    args = parse_args()
    started = time.time()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    fast_init_macros(con)
    load_raw_collections(
        con,
        offers_glob=args.offers_glob,
        pricings_glob=args.pricings_glob,
        markers_glob=args.markers_glob,
        cans_glob=args.cans_glob,
    )
    con.execute("create table expected_offers as " + project_offer_rows_from_collections(con).sql_query())
    con.execute("create table expected_articles as " + aggregate_articles_from_collections(con).sql_query())
    con.execute("create table actual_offers as select * from read_parquet(?)", [str(Path(args.output_root) / "offer_rows" / "*.parquet")])
    con.execute("create table actual_articles as select * from read_parquet(?)", [str(Path(args.output_root) / "articles" / "*.parquet")])

    expected_offers = fetch_dicts(con, "select * from expected_offers")
    actual_offers = fetch_dicts(con, "select * from actual_offers")
    expected_articles = fetch_dicts(con, "select * from expected_articles")
    actual_articles = fetch_dicts(con, "select * from actual_articles")

    offer_summary, offer_failed = compare_maps("offers", "id", expected_offers, actual_offers, canon_offer, args.sample_mismatches)
    article_summary, article_failed = compare_maps("articles", "article_hash", expected_articles, actual_articles, canon_article, args.sample_mismatches)

    result = {
        "offers": offer_summary,
        "articles": article_summary,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    print(json.dumps(result, indent=2))
    if offer_failed or article_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
