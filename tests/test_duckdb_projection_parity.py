"""Parity check: `indexer.duckdb_projection` vs `indexer.projection.project()`.

Validates the F9 design question "can the bulk indexer be DuckDB-native?"
by running both projection paths over the same joined-records fixture and
asserting field-level equality on every row.

Phased: starts at 200 rows (the existing fixture) for fast iteration. Once
clean at 200, the same test runs at 10K (after pulling a larger sample
from S3 — see `scripts/dump_s3_sample.py`).

Diffs report per-(record_index, field_name) so SQL bugs land on the
specific transform that's wrong. Failures are written to
`tests/.parity_diff.json` for inspection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.duckdb_projection import project_records  # noqa: E402
from indexer.projection import project  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "tests/fixtures/mongo_sample"
DIFF_OUT = REPO_ROOT / "tests/.parity_diff.json"

# Each entry: (label, path, required). `required=True` ones fail if missing
# (committed to the repo). Larger samples are opt-in — `dump_s3_sample.py`
# builds them locally; CI runs the 200 path only.
PARITY_FIXTURES = [
    ("sample_200",  FIXTURES_DIR / "sample_200.json",  True),
    ("sample_10k",  FIXTURES_DIR / "sample_10k.json",  False),
]


# ---------- canonicalisation ---------------------------------------------
# Both implementations should produce semantically identical rows, but the
# *representation* of a few fields differs (DuckDB lists vs Python lists;
# numeric-string vs Decimal vs float; struct dict vs plain dict). We
# canonicalise both sides into a JSON-comparable form before diffing.

def _canon(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        # Round to 6 decimals — matches DECIMAL(18,6) on the DuckDB side and
        # accommodates the legacy Decimal arithmetic. Sentinels (±MAX_PRICE_SENTINEL)
        # are preserved exactly.
        return round(v, 6)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return {k: _canon(v[k]) for k in sorted(v.keys())}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    # Fallback: stringify (covers UUID, Decimal, etc.)
    return str(v)


def _row_diff(py_row: dict, db_row: dict) -> dict[str, dict]:
    """Per-field diff. Returns {field: {py: ..., db: ...}} for mismatches."""
    diff: dict[str, dict] = {}
    keys = set(py_row.keys()) | set(db_row.keys())
    for k in sorted(keys):
        p = _canon(py_row.get(k))
        d = _canon(db_row.get(k))
        if p != d:
            diff[k] = {"py": p, "db": d}
    return diff


# ---------- fixtures ------------------------------------------------------

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
def py_rows(fixture_path: Path) -> list[dict]:
    records = json.loads(fixture_path.read_text())["records"]
    return [project(r).row for r in records]


@pytest.fixture(scope="module")
def db_rows(fixture_path: Path) -> list[dict]:
    return project_records(fixture_path)


# ---------- top-level shape ----------------------------------------------

def test_row_counts_match(fixture_path: Path, py_rows: list[dict], db_rows: list[dict]) -> None:
    assert len(py_rows) == len(db_rows), (
        f"{fixture_path.name}: row count mismatch python={len(py_rows)} duckdb={len(db_rows)}"
    )


def test_id_set_matches(fixture_path: Path, py_rows: list[dict], db_rows: list[dict]) -> None:
    """Both implementations must emit the same set of PKs. Catches missing
    rows or duplicates without depending on row order."""
    py_ids = sorted(r["id"] for r in py_rows)
    db_ids = sorted(r["id"] for r in db_rows)
    only_py = sorted(set(py_ids) - set(db_ids))
    only_db = sorted(set(db_ids) - set(py_ids))
    assert not only_py and not only_db, (
        f"{fixture_path.name}: PK set mismatch: only-in-python={only_py[:5]!r}... "
        f"only-in-duckdb={only_db[:5]!r}..."
    )


# ---------- per-row field-level parity ----------------------------------

def test_per_row_field_parity(fixture_path: Path, py_rows: list[dict], db_rows: list[dict]) -> None:
    """Every field on every row must match after canonicalisation."""
    py_by_id = {r["id"]: r for r in py_rows}
    db_by_id = {r["id"]: r for r in db_rows}
    diffs: dict[str, dict] = {}
    for pk in py_by_id:
        if pk not in db_by_id:
            continue  # caught by test_id_set_matches
        d = _row_diff(py_by_id[pk], db_by_id[pk])
        if d:
            diffs[pk] = d
    if diffs:
        diff_path = DIFF_OUT.with_suffix(f".{fixture_path.stem}.json")
        diff_path.write_text(json.dumps(diffs, indent=2, default=str))
        sample = dict(list(diffs.items())[:3])
        pytest.fail(
            f"{fixture_path.name}: {len(diffs)}/{len(py_by_id)} rows have field diffs. "
            f"Full report at {diff_path}. First 3:\n"
            f"{json.dumps(sample, indent=2, default=str)[:4000]}"
        )
