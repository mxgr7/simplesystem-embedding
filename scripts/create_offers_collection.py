"""Create a versioned `offers_v{N}` Milvus collection per F9 topology
and register the public alias (`offers` by default) to point at it.

Usage:
    python create_offers_collection.py --version 3
    python create_offers_collection.py --version 3 --alias offers
    python create_offers_collection.py --version 3 --no-alias
    python create_offers_collection.py --version 3 --dry-run

Naming is operator-driven: pick `--version N` higher than the current
version. ftsearch never embeds the versioned name; it talks to the alias
(see search-api/main.py — `MilvusClient.search` accepts an alias for
`collection_name`).

F9 split this collection's vector + BM25 + article-level scalars out
into the new `articles_v{N}` collection (see
`scripts/create_articles_collection.py`). What remains here is the
per-offer scope:

  - `id` PK is `{vendor_uuid_dashed}:{base64Url(articleNumber)}` (post-F9
    PR2b — F9 dropped the legacy friendly_id base62 encoding so the
    DuckDB-native projection has no UDF requirement).
  - `article_hash` VARCHAR(32), INVERTED — join key into `articles_v{N}`.
  - Per-offer scalars (vendor, catalog, prices JSON, delivery, core
    markers, relationships, ean, article_number, features).
  - F8 per-offer price-scope envelope (`price_list_ids`, `currencies`,
    `{ccy}_price_min/max`) — Path B's bounded probe pushes price-list /
    currency / range pre-filters down via these columns instead of a
    full-collection scan + JSON post-pass. Recall is preserved exactly
    (the envelope is a superset of the precise filter).
  - NO `offer_embedding`, NO sparse codes — the dense vector and BM25
    index live on `articles_v{N}` keyed by hash.

See `issues/article-search-replacement-ftsearch-09-article-dedup.md` for
the topology rationale. Steady-state cutovers for `articles_v{N}` and
`offers_v{N}` are paired — see "Paired alias swing" in
`scripts/MILVUS_ALIAS_WORKFLOW.md`.
"""

from __future__ import annotations

import argparse
import sys

from pymilvus import DataType, MilvusClient

# Catalogue currencies that get per-offer envelope columns (F8). Mirror
# of the same constant in `scripts/create_articles_collection.py` and
# `indexer/projection.py`; `test_catalog_currencies_match_script` keeps
# all three in sync. Add a new currency: append here, in the other two
# spots, and bump the collection version.
CATALOG_CURRENCIES = ("eur", "chf", "huf", "pln", "gbp", "czk", "cny")

# Each field listed here gets an INVERTED scalar index. Picked to cover
# every filter / group_by / aggregation path called out in spec §4.3-§4.4
# and in F3..F5. INVERTED handles equality, IN, range, and ARRAY
# membership uniformly on Milvus 2.6.15.
#
# F9: `article_hash` is the join key into `articles_v{N}`. Path B's
# bounded-probe `query()` filters on the per-offer scope here, returns
# the matching distinct hashes, and feeds them into the article-collection
# ANN as `article_hash IN [...]`. INVERTED on a 32-char VARCHAR is the
# right shape for the IN-clause workload (validated at 25k hashes ≈ 430ms
# p95 on the hardware ceiling benchmark — see F9 PATH_B_HASH_LIMIT).
SCALAR_INDEX_FIELDS = (
    # F9 join key
    "article_hash",
    # Vendor / catalog
    "vendor_id",
    "catalog_version_ids",
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
    # F8 price-scope pre-filter — INVERTED handles array_contains[_any]
    # on both columns at array scope. Per-currency price ranges are
    # STL_SORT (added separately in build_index_params for range queries).
    "price_list_ids",
    "currencies",
)


def build_schema(client: MilvusClient):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

    # PK: `{vendor_uuid_dashed}:{base64Url(articleNumber)}`. 256 leaves
    # ample headroom — UUID head is 36 chars, observed b64 tail tops out
    # ~65 chars in fixtures (long INDUSTRIAL-PART article numbers).
    schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)

    # Milvus 2.6 requires every collection to declare at least one vector
    # field with an index. Path B never searches this collection — only
    # `query()` on filter expressions — so we declare a 2-dim FLOAT
    # placeholder + FLAT index. Storage cost: ~1.3 GB at 159M rows
    # (negligible). The dense vector for retrieval lives on `articles_v{N}`.
    schema.add_field("_placeholder_vector", DataType.FLOAT_VECTOR, dim=2)

    # F9 join key into `articles_v{N}`. sha256 truncated to 16 bytes,
    # hex-encoded. Path B's bounded probe queries this collection on the
    # per-offer scope, collects distinct hashes, then constrains the
    # article-collection ANN with `article_hash IN [...]`.
    schema.add_field("article_hash", DataType.VARCHAR, max_length=32)

    # Per-offer identifier scalars.
    schema.add_field("ean", DataType.VARCHAR, max_length=64)
    schema.add_field("article_number", DataType.VARCHAR, max_length=256)

    # Single vendor per row (§7 / §9 #1).
    schema.add_field("vendor_id", DataType.VARCHAR, max_length=64)

    # Multi: catalog versions an offer participates in.
    schema.add_field(
        "catalog_version_ids", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=2048, max_length=64,
    )

    # Per-offer scalars.
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

    # F8 price-scope pre-filter columns. The envelope is per-offer (the
    # F9 article-side envelope on `articles_v{N}` powers the sort-by-price
    # browse path; this one powers Path B's bounded probe).
    #
    # `price_list_ids`: union of every `prices[].sourcePriceListId` on
    # this offer. Median 4 entries, max ~470 in sampled prod data.
    # `currencies`: union of every `prices[].currency` on this offer.
    # Catalogue carries 7 currencies today.
    schema.add_field(
        "price_list_ids", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=512, max_length=64,
    )
    schema.add_field(
        "currencies", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=8, max_length=8,
    )
    # Per-currency (min, max) FLOAT envelope. Sentinel for "no price in
    # this currency on this offer": +MAX_PRICE_SENTINEL on _min,
    # -MAX_PRICE_SENTINEL on _max — Milvus 2.6 rejects NaN/±Inf so a
    # large finite value is the working substitute (see
    # `indexer/projection.py:MAX_PRICE_SENTINEL`). Range predicates
    # (`{ccy}_price_min <= X AND {ccy}_price_max >= Y`) naturally
    # exclude sentinel rows.
    for ccy in CATALOG_CURRENCIES:
        schema.add_field(f"{ccy}_price_min", DataType.FLOAT)
        schema.add_field(f"{ccy}_price_max", DataType.FLOAT)
    return schema


def build_index_params(client: MilvusClient):
    params = client.prepare_index_params()
    # Required by Milvus — see `_placeholder_vector` note in build_schema.
    params.add_index(
        field_name="_placeholder_vector",
        index_type="FLAT",
        metric_type="L2",
    )
    for field in SCALAR_INDEX_FIELDS:
        params.add_index(field_name=field, index_type="INVERTED", index_name=field)

    # F8: STL_SORT on every per-currency envelope column. Path B's probe
    # composes `{ccy}_price_min <= decoded_max AND {ccy}_price_max >=
    # decoded_min` against these — STL_SORT is the right index for range
    # queries on FLOAT scalars (Milvus 2.6 §"Scalar indexes" capability
    # matrix).
    for ccy in CATALOG_CURRENCIES:
        for suffix in ("min", "max"):
            field = f"{ccy}_price_{suffix}"
            params.add_index(field_name=field, index_type="STL_SORT", index_name=field)
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
    p.add_argument("--dry-run", action="store_true", help="Print plan and exit.")
    args = p.parse_args()

    name = f"offers_v{args.version}"
    print(f"Target collection: {name}")
    print(f"Alias:             {'(skipped)' if args.no_alias else args.alias}")
    print(f"Milvus URI:        {args.uri}")
    print(f"Currencies:        {len(CATALOG_CURRENCIES)} ({', '.join(CATALOG_CURRENCIES)})")
    print(f"Scalar indexes:    {len(SCALAR_INDEX_FIELDS)} + {len(CATALOG_CURRENCIES) * 2} STL_SORT")
    if args.dry_run:
        print("(dry-run — no Milvus calls made)")
        return

    client = MilvusClient(uri=args.uri)
    if client.has_collection(name):
        sys.exit(f"Refusing to recreate {name!r} (already exists). Pick a higher --version.")

    schema = build_schema(client)
    index_params = build_index_params(client)
    client.create_collection(collection_name=name, schema=schema, index_params=index_params)
    print(f"Created collection {name!r}.")

    client.load_collection(name)
    print(f"Loaded {name!r}.")

    if not args.no_alias:
        swing_alias(client, args.alias, name)


if __name__ == "__main__":
    main()
