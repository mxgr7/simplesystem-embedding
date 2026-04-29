"""Parity check: DuckDB SQL aggregate + offer-row emission vs Python.

The flat-row projection parity check (`test_duckdb_projection_parity.py`)
proves SQL == Python at the per-offer projection layer. This file extends
that proof to the F9 two-stream layer:

    `aggregate_articles(json_path)` (DuckDB) ==
        group flat rows by `compute_article_hash` →
        `aggregate_article(group)` per group (Python)

    `project_offer_rows(json_path)` (DuckDB) ==
        `to_offer_row(row, article_hash=…)` per row (Python)

Together with the projection parity these establish that every read
ftsearch makes against the F9 collections lands a SQL-emitted row that
the legacy Python code path would have produced byte-for-byte (modulo
canonicalisation — float precision, dict order, sentinel rounding).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.duckdb_projection import aggregate_articles, project_offer_rows  # noqa: E402
from indexer.projection import (  # noqa: E402
    aggregate_article,
    compute_article_hash,
    project,
    to_offer_row,
)

FIXTURES_DIR = REPO_ROOT / "tests/fixtures/mongo_sample"
DIFF_OUT = REPO_ROOT / "tests/.aggregate_parity_diff.json"

PARITY_FIXTURES = [
    ("sample_200",  FIXTURES_DIR / "sample_200.json",  True),
    ("sample_10k",  FIXTURES_DIR / "sample_10k.json",  False),
]


# ---------- canonicalisation ---------------------------------------------
# Identical to test_duckdb_projection_parity._canon — float rounding to 6
# decimals (DECIMAL(18,6) on the SQL side), sorted-by-key dict
# normalisation, list/tuple unification.
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


def _row_diff(py_row: dict, db_row: dict) -> dict[str, dict]:
    diff: dict[str, dict] = {}
    keys = set(py_row.keys()) | set(db_row.keys())
    for k in sorted(keys):
        p = _canon(py_row.get(k))
        d = _canon(db_row.get(k))
        if p != d:
            diff[k] = {"py": p, "db": d}
    return diff


def _resolve(label: str, path: Path, required: bool) -> Path:
    if path.exists():
        return path
    if required:
        pytest.fail(f"required fixture {label!r} missing at {path}")
    pytest.skip(f"optional fixture {label!r} missing at {path}")


@pytest.fixture(scope="module", params=PARITY_FIXTURES, ids=[f[0] for f in PARITY_FIXTURES])
def fixture_path(request) -> Path:
    label, path, required = request.param
    return _resolve(label, path, required)


@pytest.fixture(scope="module")
def py_flat_rows(fixture_path: Path) -> list[dict]:
    """Python-projected flat rows. The article + offer streams derive
    from these via `compute_article_hash` + `aggregate_article` /
    `to_offer_row`."""
    records = json.loads(fixture_path.read_text())["records"]
    return [project(r).row for r in records]


@pytest.fixture(scope="module")
def py_articles(py_flat_rows: list[dict]) -> dict[str, dict]:
    by_hash: dict[str, list[dict]] = {}
    for r in py_flat_rows:
        by_hash.setdefault(compute_article_hash(r), []).append(r)
    return {h: aggregate_article(group) for h, group in by_hash.items()}


@pytest.fixture(scope="module")
def py_offers(py_flat_rows: list[dict]) -> list[dict]:
    """Multi-set: a fixture with multiple records sharing
    `(vendorId, articleNumber)` produces multiple offer rows with the same
    `id`. Both implementations emit one offer row per input record;
    duplicates are real and must round-trip equally on both sides."""
    return [
        to_offer_row(r, article_hash=compute_article_hash(r))
        for r in py_flat_rows
    ]


@pytest.fixture(scope="module")
def db_articles(fixture_path: Path) -> dict[str, dict]:
    return {r["article_hash"]: r for r in aggregate_articles(fixture_path)}


@pytest.fixture(scope="module")
def db_offers(fixture_path: Path) -> list[dict]:
    return project_offer_rows(fixture_path)


# ---------- top-level shape ----------------------------------------------

def test_article_count_matches(py_articles: dict, db_articles: dict) -> None:
    assert len(py_articles) == len(db_articles), (
        f"article count mismatch: py={len(py_articles)} db={len(db_articles)}"
    )


def test_article_hash_set_matches(py_articles: dict, db_articles: dict) -> None:
    only_py = sorted(set(py_articles) - set(db_articles))
    only_db = sorted(set(db_articles) - set(py_articles))
    assert not only_py and not only_db, (
        f"article_hash set mismatch: only-in-python={only_py[:5]!r}... "
        f"only-in-duckdb={only_db[:5]!r}..."
    )


def test_offer_count_matches(py_offers: list, db_offers: list) -> None:
    assert len(py_offers) == len(db_offers), (
        f"offer count mismatch: py={len(py_offers)} db={len(db_offers)}"
    )


def test_offer_multiset_matches(py_offers: list, db_offers: list) -> None:
    """Compare canonicalised offer rows as multi-sets — fixtures may
    contain multiple input records sharing the same `(vendorId,
    articleNumber)` (and thus the same `id`). Both implementations emit
    one offer row per input record; the multi-set of rows must agree."""
    py_canon = sorted(json.dumps(_canon(r), sort_keys=True) for r in py_offers)
    db_canon = sorted(json.dumps(_canon(r), sort_keys=True) for r in db_offers)
    if py_canon != db_canon:
        py_set, db_set = set(py_canon), set(db_canon)
        only_py = sorted(py_set - db_set)
        only_db = sorted(db_set - py_set)
        diff_path = DIFF_OUT.with_suffix(".offer_multiset.json")
        diff_path.write_text(json.dumps({
            "only_in_python": [json.loads(s) for s in only_py[:5]],
            "only_in_duckdb": [json.loads(s) for s in only_db[:5]],
            "py_unique_count": len(py_canon) - sum(1 for x in py_canon if x in db_set),
            "db_unique_count": len(db_canon) - sum(1 for x in db_canon if x in py_set),
        }, indent=2, default=str))
        pytest.fail(
            f"offer multiset mismatch ({len(only_py)} only in python, "
            f"{len(only_db)} only in duckdb). Sample at {diff_path}."
        )


# ---------- per-row field-level parity -----------------------------------

def test_per_article_field_parity(
    fixture_path: Path, py_articles: dict, db_articles: dict
) -> None:
    diffs: dict[str, dict] = {}
    for h in py_articles:
        if h not in db_articles:
            continue
        d = _row_diff(py_articles[h], db_articles[h])
        if d:
            diffs[h] = d
    if diffs:
        diff_path = DIFF_OUT.with_suffix(f".articles.{fixture_path.stem}.json")
        diff_path.write_text(json.dumps(diffs, indent=2, default=str))
        sample = dict(list(diffs.items())[:3])
        pytest.fail(
            f"{fixture_path.name}: {len(diffs)}/{len(py_articles)} article rows have "
            f"field diffs. Full report at {diff_path}. First 3:\n"
            f"{json.dumps(sample, indent=2, default=str)[:4000]}"
        )


# Note: per-offer-row field-level parity is covered by
# `test_offer_multiset_matches` above — the multi-set canonicalisation
# diff identifies exactly which rows differ. A by-PK comparison breaks
# down on fixtures with duplicate `(vendorId, articleNumber)` records,
# which prod data in fact has (~30% of sample_10k).
