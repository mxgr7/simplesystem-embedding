"""DuckDB-backed id -> full offer record lookup.

The source is the pre-flattened parquet dataset at
``offers_embedded_full.parquet/bucket=NN.parquet`` (16 buckets, ~160M rows).
Id row-groups have no min/max stats, so DuckDB scans all files on each
lookup — predicate pushdown still keeps it to ~200–300ms once the OS page
cache is warm, which is fine for a click-triggered modal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

_OUTPUT_COLUMNS = [
    "id",
    "name",
    "manufacturerName",
    "description",
    "ean",
    "article_number",
    "manufacturerArticleNumber",
    "manufacturerArticleType",
    "categoryPaths",
    "vendor_listings",
    "n",
]


@dataclass(slots=True)
class OfferRecord:
    id: str
    fields: dict[str, Any]


class OfferLookup:
    def __init__(self, data_dir: str | Path) -> None:
        glob = str(Path(data_dir) / "bucket=*.parquet")
        if "'" in glob:
            raise ValueError(f"invalid data_dir (contains quote): {data_dir!r}")
        self._con = duckdb.connect(database=":memory:")
        cols = ", ".join(_OUTPUT_COLUMNS)
        self._con.execute(
            f"CREATE VIEW offers AS SELECT {cols} FROM read_parquet('{glob}')"
        )

    def get(self, offer_id: str) -> OfferRecord | None:
        row = self._con.execute(
            "SELECT * FROM offers WHERE id = ?", [offer_id]
        ).fetchone()
        if row is None:
            return None
        description = self._con.description
        fields = {description[i][0]: row[i] for i in range(len(row))}
        return OfferRecord(id=fields["id"], fields=fields)

    def close(self) -> None:
        self._con.close()
