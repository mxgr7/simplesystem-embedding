"""Smoke: DuckDB can read S3 Atlas-snapshot shards with credential_chain.

Validates the F9 PR2b architectural assumption that the production bulk
indexer can pull JSONL.gz directly from
`s3://mongo-atlas-snapshot-for-lab/.../prod/{collection}/atlas-*.json.gz`
via DuckDB's `httpfs` extension — no S3 download step, no boto3 plumbing.

Skipped if `AWS_PROFILE` is unset or the credential chain can't reach S3.
This is a "does the wire work" test; per-row parity is in
`test_duckdb_projection_parity.py`.
"""

from __future__ import annotations

import os

import duckdb
import pytest

S3_BASE = (
    "s3://mongo-atlas-snapshot-for-lab/exported_snapshots/"
    "62fba96d8e949e4b845c9867/6308d553c10c0b62832b4a6e/"
    "s2-prod/2026-03-04T1103/1772628355/prod"
)
PROBE_SHARD = f"{S3_BASE}/offers/atlas-fkxrb3-shard-0.0.json.gz"


@pytest.fixture(scope="module")
def s3_con() -> duckdb.DuckDBPyConnection:
    if not os.environ.get("AWS_PROFILE") and not os.environ.get("AWS_ACCESS_KEY_ID"):
        pytest.skip("no AWS credentials in env (set AWS_PROFILE=simplesystem)")
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    # `credential_chain` follows the standard SDK lookup order (env → shared
    # config → instance metadata) — same as boto3 / aws-cli. Region pinned
    # to eu-central-1 to match the bucket.
    con.execute(
        "CREATE OR REPLACE SECRET s3_secret "
        "(TYPE S3, PROVIDER credential_chain, REGION 'eu-central-1')"
    )
    return con


def test_duckdb_can_count_s3_offers_shard(s3_con: duckdb.DuckDBPyConnection) -> None:
    """One shard of `offers` reads cleanly and the row count is in the
    expected ballpark (~30K records per shard — see the snapshot sizing
    table in scripts/dump_s3_sample.py)."""
    res = s3_con.execute(
        f"SELECT count(*) FROM read_json('{PROBE_SHARD}', format='newline_delimited')"
    ).fetchone()
    assert res is not None and res[0] > 1000, (
        f"unexpected row count from {PROBE_SHARD}: {res}"
    )
