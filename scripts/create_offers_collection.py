"""Create a versioned `offers_v{N}` Milvus collection per spec §7 and
register the public alias (`offers` by default) to point at it.

Usage:
    python create_offers_collection.py --version 2
    python create_offers_collection.py --version 3 --alias offers
    python create_offers_collection.py --version 3 --no-alias
    python create_offers_collection.py --version 3 --dry-run

Naming is deliberately operator-driven: pick `--version N` higher than
the current version. ftsearch never embeds the versioned name; it talks
to the alias (see search-api/main.py — `MilvusClient.search` accepts
an alias for `collection_name`).

Schema mirrors `issues/article-search-replacement-spec.md` §7. Scalar
indexes cover every field that F3..F5 will filter, group, or aggregate
on. The vector index defaults to HNSW (matches current production
config).
"""

from __future__ import annotations

import argparse
import sys

from pymilvus import DataType, MilvusClient

DIM = 128

VECTOR_INDEX_DEFAULTS = {
    "HNSW": {"params": {"M": 16, "efConstruction": 200}},
    "IVF_FLAT": {"params": {"nlist": 4096}},
    "FLAT": {"params": {}},
}

# Each field listed here gets an INVERTED scalar index. Picked to cover
# every filter / group_by / aggregation path called out in spec §4.3-§4.4
# and in F3..F5. INVERTED handles equality, IN, range, and ARRAY
# membership uniformly on Milvus 2.6.15.
SCALAR_INDEX_FIELDS = (
    # Vendor / catalog
    "vendor_id",
    "catalog_version_ids",
    # eClass / S2Class hierarchy (§4.3, §4.4)
    "eclass5_code",
    "eclass7_code",
    "s2class_code",
    # Category hierarchy (§4.3 prefix `like` + §4.4 group_by)
    "category_l1",
    "category_l2",
    "category_l3",
    "category_l4",
    "category_l5",
    # Numeric range (§4.3 `maxDeliveryTime`)
    "delivery_time_days_max",
    # Features (§4.3 requiredFeatures, §4.4 FEATURES summary)
    "features",
    # Core sortiment (§4.3 `coreSortimentOnly`)
    "core_marker_enabled_sources",
    "core_marker_disabled_sources",
    # Relationships (§4.3)
    "relationship_accessory_for",
    "relationship_spare_part_for",
    "relationship_similar_to",
    # Identifier-level filters used by routed strict path
    "ean",
    "article_number",
)


def build_schema(client: MilvusClient):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

    # PK: legacy composite `{friendlyId}:{base64Url(articleNumber)}`. 256
    # leaves ample headroom (observed up to ~85 chars in fixtures).
    schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field("offer_embedding", DataType.FLOAT16_VECTOR, dim=DIM)

    # Straight projections.
    schema.add_field("name", DataType.VARCHAR, max_length=1024)
    schema.add_field("manufacturerName", DataType.VARCHAR, max_length=256)
    schema.add_field("ean", DataType.VARCHAR, max_length=64)
    schema.add_field("article_number", DataType.VARCHAR, max_length=256)

    # Single vendor per row (§7 / §9 #1).
    schema.add_field("vendor_id", DataType.VARCHAR, max_length=64)

    # Multi: catalog versions an offer participates in.
    schema.add_field(
        "catalog_version_ids", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=2048, max_length=64,
    )

    # Category prefix-paths joined with `¦` (`|` escape per CategoryPath.java).
    schema.add_field("category_l1", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=256)
    schema.add_field("category_l2", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=640)
    schema.add_field("category_l3", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=768)
    schema.add_field("category_l4", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=1024)
    schema.add_field("category_l5", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=1280)

    # NEW per §7.
    schema.add_field("prices", DataType.JSON)
    schema.add_field("delivery_time_days_max", DataType.INT32)
    schema.add_field(
        "core_marker_enabled_sources", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=64, max_length=64,
    )
    schema.add_field(
        "core_marker_disabled_sources", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=64, max_length=64,
    )
    # eClass / S2Class hierarchies — ARRAY<INT32> carrying every level of
    # the legacy hierarchy (root → leaf), mirroring ES `offers.eclass51Groups`
    # / `eclass71Groups` / `s2classGroups` keyword arrays. A `terms` query
    # at any level matches via `array_contains[_any]`. Single-INT collapsed
    # the hierarchy to one undefined-ordering scalar — silent recall bug.
    schema.add_field(
        "eclass5_code", DataType.ARRAY,
        element_type=DataType.INT32, max_capacity=16,
    )
    schema.add_field(
        "eclass7_code", DataType.ARRAY,
        element_type=DataType.INT32, max_capacity=16,
    )
    schema.add_field(
        "s2class_code", DataType.ARRAY,
        element_type=DataType.INT32, max_capacity=16,
    )
    schema.add_field(
        "features", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=512, max_length=512,
    )
    schema.add_field(
        "relationship_accessory_for", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=128, max_length=256,
    )
    schema.add_field(
        "relationship_spare_part_for", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=128, max_length=256,
    )
    schema.add_field(
        "relationship_similar_to", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=128, max_length=256,
    )
    return schema


def build_index_params(client: MilvusClient, vector_index: str):
    cfg = VECTOR_INDEX_DEFAULTS[vector_index]
    params = client.prepare_index_params()
    params.add_index(
        field_name="offer_embedding",
        index_type=vector_index,
        metric_type="COSINE",
        **cfg,
    )
    for field in SCALAR_INDEX_FIELDS:
        params.add_index(field_name=field, index_type="INVERTED", index_name=field)
    return params


def swing_alias(client: MilvusClient, alias: str, target: str) -> None:
    try:
        info = client.describe_alias(alias=alias)
        current = info.get("collection_name") if isinstance(info, dict) else None
    except Exception:
        current = None

    if current is None:
        print(f"Alias {alias!r} does not exist — creating → {target!r}.")
        client.create_alias(collection_name=target, alias=alias)
    elif current == target:
        print(f"Alias {alias!r} already points at {target!r} — nothing to do.")
    else:
        print(f"Alias {alias!r}: {current!r} → {target!r} (atomic swing).")
        client.alter_alias(collection_name=target, alias=alias)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", type=int, required=True, help="Integer N — produces collection name offers_v{N}.")
    p.add_argument("--alias", default="offers", help="Public alias (default 'offers').")
    p.add_argument("--no-alias", action="store_true", help="Skip alias create/swing.")
    p.add_argument("--uri", default="http://localhost:19530", help="Milvus URI.")
    p.add_argument("--vector-index", default="HNSW", choices=list(VECTOR_INDEX_DEFAULTS))
    p.add_argument("--dry-run", action="store_true", help="Print plan and exit.")
    args = p.parse_args()

    name = f"offers_v{args.version}"
    print(f"Target collection: {name}")
    print(f"Alias:             {'(skipped)' if args.no_alias else args.alias}")
    print(f"Milvus URI:        {args.uri}")
    print(f"Vector index:      {args.vector_index}")
    print(f"Scalar indexes:    {len(SCALAR_INDEX_FIELDS)}")
    if args.dry_run:
        print("(dry-run — no Milvus calls made)")
        return

    client = MilvusClient(uri=args.uri)
    if client.has_collection(name):
        sys.exit(f"Refusing to recreate {name!r} (already exists). Pick a higher --version.")

    schema = build_schema(client)
    index_params = build_index_params(client, args.vector_index)
    client.create_collection(collection_name=name, schema=schema, index_params=index_params)
    print(f"Created collection {name!r}.")

    client.load_collection(name)
    print(f"Loaded {name!r}.")

    if not args.no_alias:
        swing_alias(client, args.alias, name)


if __name__ == "__main__":
    main()
