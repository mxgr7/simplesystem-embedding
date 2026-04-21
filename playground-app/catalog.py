"""DuckDB-backed lookup of offer records from the grouped parquet file."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb


@dataclass(slots=True)
class Offer:
    id: str
    name: str
    manufacturer: str
    ean: str
    article_number: str


class Catalog:
    def __init__(self, parquet_glob: str) -> None:
        self._parquet_glob = parquet_glob
        self._con = duckdb.connect()

    def lookup(self, ids: list[str]) -> dict[str, Offer]:
        if not ids:
            return {}
        df = self._con.execute(
            f"""
            SELECT id, name, manufacturerName, ean, article_number
            FROM read_parquet('{self._parquet_glob}')
            WHERE id = ANY($1)
            """,
            [ids],
        ).fetchdf()
        out: dict[str, Offer] = {}
        for _, row in df.iterrows():
            out[row["id"]] = Offer(
                id=row["id"],
                name=_str(row.get("name")),
                manufacturer=_str(row.get("manufacturerName")),
                ean=_str(row.get("ean")),
                article_number=_str(row.get("article_number")),
            )
        return out


def _str(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s
