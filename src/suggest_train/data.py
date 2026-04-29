"""Dataset paths and loaders for the suggest training pipeline.

The raw corpus is the day-partitioned parquet dump produced by
`scripts/fetch_posthog_suggest_data.py`. We expose the directory as both a
pyarrow dataset (zero-copy filtering / streaming) and as a DuckDB view (for
fast SQL aggregations during EDA).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow.dataset as ds

RAW_DIR = Path("/data/datasets/suggest/raw_search_events.parquet")


def open_dataset(path: Path | str = RAW_DIR) -> ds.Dataset:
    """Open the day-partitioned parquet dataset as a pyarrow Dataset."""
    return ds.dataset(str(path), partitioning="hive", format="parquet")


def duckdb_connect(path: Path | str = RAW_DIR,
                   view: str = "events") -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection with the raw events registered as
    a view called ``view`` (default ``events``).

    We keep the connection in memory, but the view is a lazy reference into
    the parquet files so DuckDB only reads the columns/partitions a query
    actually touches.
    """
    con = duckdb.connect(":memory:")
    con.execute(
        f"CREATE OR REPLACE VIEW {view} AS "
        f"SELECT * FROM read_parquet('{path}/**/*.parquet', "
        f"hive_partitioning = TRUE)"
    )
    return con
