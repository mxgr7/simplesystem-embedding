"""Thin wrapper around the pymilvus client for the offers collection.

Card metadata is read from the Milvus collection directly via
``output_fields`` — the collection already stores the display fields, so
there is no separate catalog lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymilvus import MilvusClient

OUTPUT_FIELDS = ["id", "name", "manufacturerName", "ean", "article_number"]
_VECTOR_FIELD = "offer_embedding"


@dataclass(slots=True)
class Hit:
    id: str
    score: float
    name: str
    manufacturer: str
    ean: str
    article_number: str


@dataclass(slots=True)
class CollectionInfo:
    """Static info about the target collection, fetched once at startup."""

    uri: str
    collection: str
    row_count: int
    load_state: str
    num_partitions: int
    num_shards: int
    vector_field: str
    vector_dim: int
    vector_dtype: str
    index_type: str
    metric_type: str
    index_params: dict
    indexed_rows: int
    index_state: str
    consistency_level: int | None
    scalar_indexes: list[str]


class MilvusSearch:
    def __init__(self, uri: str, collection: str) -> None:
        self._client = MilvusClient(uri)
        self._uri = uri
        self.collection = collection
        if not self._client.has_collection(collection):
            raise RuntimeError(f"Milvus collection {collection!r} missing at {uri}")

    def describe(self) -> CollectionInfo:
        desc = self._client.describe_collection(self.collection)
        stats = self._client.get_collection_stats(collection_name=self.collection)
        load = self._client.get_load_state(collection_name=self.collection)

        vec_field = next(
            (f for f in desc.get("fields", []) if f.get("name") == _VECTOR_FIELD), {}
        )
        vec_dim = int(vec_field.get("params", {}).get("dim", 0))
        vec_dtype = getattr(vec_field.get("type"), "name", str(vec_field.get("type", "")))

        indexes = list(self._client.list_indexes(collection_name=self.collection))
        vec_idx = {}
        scalar_idx: list[str] = []
        for name in indexes:
            info = self._client.describe_index(
                collection_name=self.collection, index_name=name
            )
            if info.get("field_name") == _VECTOR_FIELD:
                vec_idx = info
            else:
                scalar_idx.append(
                    f"{info.get('field_name')} ({info.get('index_type')})"
                )

        return CollectionInfo(
            uri=self._uri,
            collection=self.collection,
            row_count=int(stats.get("row_count", 0)),
            load_state=getattr(load.get("state"), "name", str(load.get("state", ""))),
            num_partitions=int(desc.get("num_partitions", 0)),
            num_shards=int(desc.get("num_shards", 0)),
            vector_field=_VECTOR_FIELD,
            vector_dim=vec_dim,
            vector_dtype=vec_dtype,
            index_type=str(vec_idx.get("index_type", "")),
            metric_type=str(vec_idx.get("metric_type", "")),
            index_params=dict(vec_idx.get("params", {})),
            indexed_rows=int(vec_idx.get("indexed_rows", 0)),
            index_state=str(vec_idx.get("state", "")),
            consistency_level=desc.get("consistency_level"),
            scalar_indexes=scalar_idx,
        )

    def search(self, embedding: list[float], limit: int) -> list[Hit]:
        # Collection stores fp16 vectors; matching the query precision flushes
        # subnormals to 0 instead of tripping Milvus's underflow validator.
        query = np.asarray(embedding, dtype=np.float16)
        results = self._client.search(
            collection_name=self.collection,
            data=[query],
            limit=limit,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=OUTPUT_FIELDS,
        )

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
