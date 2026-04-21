"""Thin wrapper around the pymilvus client for the offers collection.

Card metadata is read from the Milvus collection directly via
``output_fields`` — the collection already stores the display fields, so
there is no separate catalog lookup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

log = logging.getLogger(__name__)

_OUTPUT_FIELDS = ["id", "name", "manufacturerName", "ean", "article_number"]


@dataclass(slots=True)
class Hit:
    id: str
    score: float
    name: str
    manufacturer: str
    ean: str
    article_number: str


class MilvusSearch:
    def __init__(self, uri: str, collection: str) -> None:
        self._client = MilvusClient(uri)
        self._collection = collection
        if not self._client.has_collection(collection):
            raise RuntimeError(f"Milvus collection {collection!r} missing at {uri}")

    def search(self, embedding: list[float], limit: int) -> list[Hit]:
        try:
            results = self._client.search(
                collection_name=self._collection,
                data=[embedding],
                limit=limit,
                search_params={"metric_type": "COSINE", "params": {}},
                output_fields=_OUTPUT_FIELDS,
            )
        except MilvusException as e:
            log.warning("Milvus search failed: %s", e)
            return []

        hits = results[0] if results else []
        out: list[Hit] = []
        for h in hits:
            ent = h["entity"]
            out.append(Hit(
                id=ent.get("id", ""),
                score=float(h["distance"]),
                name=_s(ent.get("name")),
                manufacturer=_s(ent.get("manufacturerName")),
                ean=_s(ent.get("ean")),
                article_number=_s(ent.get("article_number")),
            ))
        return out


def _s(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
