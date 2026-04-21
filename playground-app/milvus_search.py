"""Thin wrapper around the pymilvus client for the offers collection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Hit:
    id: str
    score: float


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
                output_fields=["id"],
            )
        except MilvusException as e:
            # Empty/unloaded collection: degrade to no-hits instead of 500.
            log.warning("Milvus search failed: %s", e)
            return []
        hits = results[0] if results else []
        return [Hit(id=h["entity"]["id"], score=float(h["distance"])) for h in hits]
