#!/usr/bin/env python3
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.s2class_mapper import (
    DEFAULT_S2CLASS_CODE,
    S2CLASS_SOURCE_KEYS_DESC,
    mapping_for_version_key,
)

RAW_OFFER_COLUMNS = {
    "_id": 'STRUCT("$oid" VARCHAR)',
    "offer": 'STRUCT(offerParams STRUCT(eclassGroups JSON))',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify offer_projected s2class_code parity between source Mongo exports and parquet output."
    )
    parser.add_argument(
        "--manifest-tsv",
        required=True,
        help="manifest.tsv written by the Rust converter",
    )
    parser.add_argument(
        "--parquet-glob",
        required=True,
        help="Parquet glob for converter output, e.g. /path/*.parquet",
    )
    parser.add_argument(
        "--db-path",
        default="/tmp/verify_offer_projected_s2class.duckdb",
        help="DuckDB database path for the verification run",
    )
    parser.add_argument(
        "--temp-dir",
        default="/tmp/verify_offer_projected_s2class.tmp",
        help="DuckDB temp spill directory",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="DuckDB thread count",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional manifest row limit for smaller trial runs",
    )
    parser.add_argument(
        "--sample-mismatches",
        type=int,
        default=20,
        help="How many mismatches to print when verification fails",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write a JSON summary",
    )
    return parser.parse_args()


def load_manifest_paths(manifest_tsv, limit=None):
    paths = []
    with open(manifest_tsv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            paths.append(row["path"])
            if limit is not None and len(paths) >= limit:
                break
    if not paths:
        raise SystemExit(f"no paths found in manifest: {manifest_tsv}")
    return paths


def load_s2map_rows():
    rows = []
    for version_key in S2CLASS_SOURCE_KEYS_DESC:
        mapping = mapping_for_version_key(version_key)
        for from_code, to_code in mapping.items():
            rows.append((version_key, from_code, to_code))
    return rows


def init_duckdb(con, temp_dir, threads):
    con.execute(f"PRAGMA threads={int(threads)}")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    con.execute("PRAGMA enable_progress_bar")
    con.execute("SET preserve_insertion_order=false")
    con.execute(
        """
        CREATE OR REPLACE MACRO unwrap_int(v) AS COALESCE(
            TRY_CAST(json_extract_string(v::JSON, '$."$numberInt"') AS INTEGER),
            TRY_CAST(json_extract_string(v::JSON, '$."$numberLong"') AS INTEGER),
            TRY_CAST(v::JSON AS INTEGER)
        );
        """
    )
    con.execute(
        """
        CREATE OR REPLACE MACRO expand_eclass(code) AS
            list_filter(
                [(code // 1000000) * 1000000, (code // 10000) * 10000, (code // 100) * 100, code],
                x -> x > 0
            );
        """
    )


def build_reference_sql():
    extracts = []
    for key in S2CLASS_SOURCE_KEYS_DESC:
        col = "ec_" + key.lower()
        extracts.append(
            "COALESCE((SELECT list(unwrap_int(value)) "
            f"FROM json_each(json_extract(eclass_groups, '$.{key}'))), []::INTEGER[]) AS {col}"
        )

    cases = []
    for key in S2CLASS_SOURCE_KEYS_DESC:
        col = "ec_" + key.lower()
        cases.append(
            f"WHEN len({col}) > 0 THEN COALESCE("
            f"(SELECT list_sort(list_distinct(flatten(list(expand_eclass(s.to_code))))) "
            f"FROM unnest({col}) AS src(leaf) "
            f"JOIN s2map s ON s.version_key = '{key}' AND s.from_code = src.leaf), "
            f"expand_eclass({DEFAULT_S2CLASS_CODE})"
            ")"
        )

    case_sql = (
        "CASE\n                "
        + "\n                ".join(cases)
        + f"\n                ELSE expand_eclass({DEFAULT_S2CLASS_CODE})\n            END AS s2class_code"
    )

    return f"""
        WITH src AS (
            SELECT
                _id."$oid" AS offer_id,
                offer.offerParams.eclassGroups AS eclass_groups
            FROM raw_offers
        ),
        extracted AS (
            SELECT
                offer_id,
                {',\n                '.join(extracts)}
            FROM src
        )
        SELECT
            offer_id,
            {case_sql}
        FROM extracted
    """


def create_s2map_table(con, temp_dir):
    rows = load_s2map_rows()
    csv_path = Path(temp_dir) / "s2map.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["version_key", "from_code", "to_code"])
        writer.writerows(rows)
    con.execute("DROP TABLE IF EXISTS s2map")
    con.execute(
        """
        CREATE TABLE s2map AS
        SELECT version_key, from_code::INTEGER AS from_code, to_code::INTEGER AS to_code
        FROM read_csv(?, header=true, columns={
            'version_key': 'VARCHAR',
            'from_code': 'INTEGER',
            'to_code': 'INTEGER'
        })
        """,
        [str(csv_path)],
    )


def build_summary(con):
    summary = {}
    summary["source_rows"] = con.execute("SELECT count(*) FROM source_expected").fetchone()[0]
    summary["parquet_rows"] = con.execute("SELECT count(*) FROM parquet_actual").fetchone()[0]
    summary["source_duplicate_offer_ids"] = con.execute(
        "SELECT count(*) FROM (SELECT offer_id FROM source_expected GROUP BY 1 HAVING count(*) > 1)"
    ).fetchone()[0]
    summary["parquet_duplicate_offer_ids"] = con.execute(
        "SELECT count(*) FROM (SELECT offer_id FROM parquet_actual GROUP BY 1 HAVING count(*) > 1)"
    ).fetchone()[0]
    summary["missing_in_parquet"] = con.execute(
        """
        SELECT count(*)
        FROM source_expected s
        LEFT JOIN parquet_actual p USING (offer_id)
        WHERE p.offer_id IS NULL
        """
    ).fetchone()[0]
    summary["missing_in_source"] = con.execute(
        """
        SELECT count(*)
        FROM parquet_actual p
        LEFT JOIN source_expected s USING (offer_id)
        WHERE s.offer_id IS NULL
        """
    ).fetchone()[0]
    summary["mismatched_s2class_code"] = con.execute(
        """
        SELECT count(*)
        FROM source_expected s
        JOIN parquet_actual p USING (offer_id)
        WHERE s.s2class_code IS DISTINCT FROM p.s2class_code
        """
    ).fetchone()[0]
    return summary


def fetch_samples(con, limit):
    return con.execute(
        f"""
        SELECT
            s.offer_id,
            s.s2class_code AS expected_s2class_code,
            p.s2class_code AS actual_s2class_code
        FROM source_expected s
        JOIN parquet_actual p USING (offer_id)
        WHERE s.s2class_code IS DISTINCT FROM p.s2class_code
        ORDER BY s.offer_id
        LIMIT {int(limit)}
        """
    ).fetchall()


def main():
    args = parse_args()
    start = time.perf_counter()

    manifest_path = Path(args.manifest_tsv)
    parquet_glob = args.parquet_glob
    db_path = Path(args.db_path)
    temp_dir = Path(args.temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    paths = load_manifest_paths(manifest_path, args.limit)
    print(json.dumps({
        "manifest_tsv": str(manifest_path),
        "parquet_glob": parquet_glob,
        "manifest_rows": len(paths),
        "db_path": str(db_path),
        "temp_dir": str(temp_dir),
        "threads": args.threads,
    }, indent=2))

    con = duckdb.connect(str(db_path))
    init_duckdb(con, str(temp_dir), args.threads)

    stage_start = time.perf_counter()
    create_s2map_table(con, temp_dir)
    print(f"loaded s2 mappings in {time.perf_counter() - stage_start:.2f}s")

    stage_start = time.perf_counter()
    con.execute(
        "CREATE TABLE raw_offers AS SELECT * FROM read_json(?, format='newline_delimited', columns=?)",
        [paths, RAW_OFFER_COLUMNS],
    )
    print(f"loaded raw_offers in {time.perf_counter() - stage_start:.2f}s")

    stage_start = time.perf_counter()
    con.execute("CREATE TABLE source_expected AS " + build_reference_sql())
    con.execute("DROP TABLE raw_offers")
    print(f"computed source_expected in {time.perf_counter() - stage_start:.2f}s")

    stage_start = time.perf_counter()
    if args.limit is None:
        con.execute(
            "CREATE TABLE parquet_actual AS SELECT offer_id, s2class_code FROM read_parquet(?)",
            [parquet_glob],
        )
    else:
        con.execute(
            """
            CREATE TABLE parquet_actual AS
            SELECT offer_id, s2class_code
            FROM read_parquet(?)
            WHERE offer_id IN (SELECT offer_id FROM source_expected)
            """,
            [parquet_glob],
        )
    print(f"loaded parquet_actual in {time.perf_counter() - stage_start:.2f}s")

    stage_start = time.perf_counter()
    summary = build_summary(con)
    print(f"compared source vs parquet in {time.perf_counter() - stage_start:.2f}s")

    summary["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    summary["status"] = "ok"
    print(json.dumps(summary, indent=2))

    failed = any(
        summary[key]
        for key in [
            "source_duplicate_offer_ids",
            "parquet_duplicate_offer_ids",
            "missing_in_parquet",
            "missing_in_source",
            "mismatched_s2class_code",
        ]
    )

    if failed:
        summary["status"] = "failed"
        samples = fetch_samples(con, args.sample_mismatches)
        if samples:
            print("sample mismatches:")
            for offer_id, expected, actual in samples:
                print(json.dumps({
                    "offer_id": offer_id,
                    "expected_s2class_code": expected,
                    "actual_s2class_code": actual,
                }))
        if args.report_json:
            Path(args.report_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(1)

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
