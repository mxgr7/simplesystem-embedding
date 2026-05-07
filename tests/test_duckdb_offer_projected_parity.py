"""Parity check: `offer_projected_build_sql()` vs Python projection.

`offer_projected` is the join-independent precompute used by the fast
bulk-indexer path. It must preserve every field it owns exactly, or the
`op_grouped` projection path diverges from the canonical wrapper/raw SQL
projection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import indexer.duckdb_projection as dp  # noqa: E402
from indexer.projection import compute_article_hash, project  # noqa: E402

FIXTURE_PATH = REPO_ROOT / "tests/fixtures/mongo_sample/sample_200.json"
COMPARABLE_FIELDS = [
    "vendor_id",
    "article_number",
    "id",
    "name",
    "manufacturerName",
    "ean",
    "catalog_version_id",
    "delivery_time_days_max",
    "eclass5_code",
    "eclass7_code",
    "s2class_code",
    "relationship_accessory_for",
    "relationship_spare_part_for",
    "relationship_similar_to",
    "category_l1",
    "category_l2",
    "category_l3",
    "category_l4",
    "category_l5",
    "features",
    "article_hash",
]


def _canon(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return {k: _canon(v[k]) for k in sorted(v.keys())}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    return str(v)


def _expected_py_rows() -> list[dict]:
    records = json.loads(FIXTURE_PATH.read_text())["records"]
    out = []
    for rec in records:
        row = project(rec).row
        expected = {k: row[k] for k in COMPARABLE_FIELDS if k != "article_hash"}
        expected["article_hash"] = compute_article_hash(row)
        out.append(expected)
    return out


def _actual_db_rows() -> list[dict]:
    con = duckdb.connect()
    dp.init_macros(con)
    dp._load_wrapper_fixture(con, FIXTURE_PATH)
    con.execute(
        """
        CREATE OR REPLACE TABLE raw_offers AS
        WITH unnested AS (
            SELECT unnest(records) AS rec FROM raw
        )
        SELECT
            rec.offer.articleNumber AS articleNumber,
            rec.offer.vendorId AS vendorId,
            rec.offer.catalogVersionId AS catalogVersionId,
            rec.offer.offer AS offer
        FROM unnested
        """
    )
    sql = dp.offer_projected_build_sql(source_table_or_glob="raw_offers")
    rows = dp._fetchall_dicts(con, sql)
    return [{k: row[k] for k in COMPARABLE_FIELDS} for row in rows]


def test_offer_projected_matches_python_projection_on_owned_fields() -> None:
    py_by_id = {row["id"]: row for row in _expected_py_rows()}
    db_by_id = {row["id"]: row for row in _actual_db_rows()}
    assert set(py_by_id) == set(db_by_id)
    for pk, py_row in py_by_id.items():
        db_row = db_by_id[pk]
        assert _canon(py_row) == _canon(db_row), pk
