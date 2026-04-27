"""Milvus access for the playground app.

The playground delegates the actual search call to the sibling search-api
service over HTTP (see ``main.py``); here we cover the two operations the
search-api does not return:

  * ``MilvusInfo.describe(...)`` — collection + index metadata for the
    debug panel, fetched once at startup.
  * ``OffersLookup.fetch(ids)`` — display-field lookup keyed on the offer
    ``id``s returned by the search response. The dense ``offers`` collection
    already carries name/manufacturer/category etc., so we issue a single
    ``client.query(filter='id in [...]', output_fields=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pymilvus import MilvusClient

OUTPUT_FIELDS = [
    "id", "name", "manufacturerName", "ean", "article_number",
    "catalog_version_ids",
    "category_l1", "category_l2", "category_l3", "category_l4", "category_l5",
]
_VECTOR_FIELD = "offer_embedding"
_CATEGORY_SEP = "¦"


@dataclass(slots=True)
class Display:
    id: str
    name: str
    manufacturer: str
    ean: str
    article_number: str
    catalog_version_ids: list[str]
    category_paths: list[list[str]]


@dataclass(slots=True)
class CollectionInfo:
    """Static info about a Milvus collection, fetched once at startup."""

    uri: str
    collection: str
    row_count: int
    load_state: str
    num_partitions: int
    num_shards: int
    vector_field: str            # may be "" for a code-only collection
    vector_dim: int
    vector_dtype: str
    index_type: str
    metric_type: str
    index_params: dict
    indexed_rows: int
    index_state: str
    consistency_level: int | None
    scalar_indexes: list[str]


def describe_collection(client: MilvusClient, uri: str, collection: str) -> CollectionInfo:
    desc = client.describe_collection(collection)
    stats = client.get_collection_stats(collection_name=collection)
    load = client.get_load_state(collection_name=collection)

    fields = desc.get("fields", [])
    # Pick the first field whose Milvus type name suggests a vector. Works
    # for FLOAT_VECTOR / FLOAT16_VECTOR / SPARSE_FLOAT_VECTOR alike.
    vec_field: dict = {}
    for f in fields:
        type_name = getattr(f.get("type"), "name", str(f.get("type", "")))
        if "VECTOR" in type_name:
            vec_field = f
            break
    vec_name = vec_field.get("name", "")
    vec_dim = int(vec_field.get("params", {}).get("dim", 0))
    vec_dtype = getattr(vec_field.get("type"), "name", str(vec_field.get("type", "")))

    indexes = list(client.list_indexes(collection_name=collection))
    vec_idx: dict = {}
    scalar_idx: list[str] = []
    for name in indexes:
        info = client.describe_index(collection_name=collection, index_name=name)
        if info.get("field_name") == vec_name:
            vec_idx = info
        else:
            scalar_idx.append(f"{info.get('field_name')} ({info.get('index_type')})")

    return CollectionInfo(
        uri=uri,
        collection=collection,
        row_count=int(stats.get("row_count", 0)),
        load_state=getattr(load.get("state"), "name", str(load.get("state", ""))),
        num_partitions=int(desc.get("num_partitions", 0)),
        num_shards=int(desc.get("num_shards", 0)),
        vector_field=vec_name,
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


class OffersLookup:
    """Resolve offer ``id``s to display fields by issuing a single
    ``client.query`` against the dense collection. The dense collection
    already stores everything the card needs."""

    def __init__(self, client: MilvusClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    def fetch(self, ids: Sequence[str]) -> dict[str, Display]:
        if not ids:
            return {}
        # Quote each id; Milvus query expr requires a list literal of strings.
        # IDs are 32-char hex hashes per the source data, so escaping is a
        # belt-and-braces measure rather than a real attack vector.
        quoted = ",".join(f'"{i.replace(chr(34), "")}"' for i in ids)
        rows = self._client.query(
            collection_name=self._collection,
            filter=f"id in [{quoted}]",
            output_fields=OUTPUT_FIELDS,
            limit=len(ids),
        )
        return {
            str(r.get("id", "")): _to_display(r)
            for r in rows
        }


def _to_display(entity: dict) -> Display:
    return Display(
        id=str(entity.get("id", "")),
        name=_s(entity.get("name")),
        manufacturer=_s(entity.get("manufacturerName")),
        ean=_s(entity.get("ean")),
        article_number=_s(entity.get("article_number")),
        catalog_version_ids=[_s(v) for v in (entity.get("catalog_version_ids") or [])],
        category_paths=_category_paths(entity),
    )


def _s(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _category_paths(entity: dict) -> list[list[str]]:
    """Pick the deepest non-empty level and split each entry into breadcrumbs.

    Categories are stored per level as arrays of full paths joined by ``¦``;
    every shallower level is redundant with the deepest, so we only keep the
    deepest. The returned list has one breadcrumb per taxonomy match.
    """
    for level in range(5, 0, -1):
        raw = entity.get(f"category_l{level}") or []
        if raw:
            return [
                [part for part in str(p).split(_CATEGORY_SEP) if part]
                for p in raw
            ]
    return []
