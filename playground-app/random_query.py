"""In-memory sample of distinct queries for the "random query" button.

Loaded once at server start so clicks respond without touching disk. The
source parquet dataset has ~2M rows but only ~57K distinct ``query_term``
values, so deduplicating keeps the working set to a few MB of strings.
"""

from __future__ import annotations

import random
from pathlib import Path

import duckdb


class RandomQueryPicker:
    def __init__(self, data_dir: str | Path) -> None:
        glob = str(Path(data_dir) / "*.parquet")
        if "'" in glob:
            raise ValueError(f"invalid data_dir (contains quote): {data_dir!r}")
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(
                f"SELECT DISTINCT query_term FROM read_parquet('{glob}') "
                "WHERE query_term IS NOT NULL AND query_term <> ''"
            ).fetchall()
        finally:
            con.close()
        self._queries: list[str] = [r[0] for r in rows]
        if not self._queries:
            raise RuntimeError(f"No queries loaded from {data_dir}")

    def pick(self) -> str:
        return random.choice(self._queries)

    def __len__(self) -> int:
        return len(self._queries)
