"""Schema + index spec for the F9 paired collections (`articles_v{N}` +
`offers_v{N}`).

Single source of truth for what fields exist, how they're typed, and
which indexes get built. Used by:

  * `scripts/create_*_collection.py` — initial collection bring-up.
  * `indexer/bulk.py` — drops indexes before bulk_insert and re-applies
    them after the import + flush completes (the per-segment inline
    IndexBuilding stage is the dominant cost on small chunks; deferring
    indexes to a single post-import pass collapses the per-chunk floor).
  * tests — assert schema parity for the indexer's projection output.

Why one module instead of duplicated logic in the two create scripts:
keeping the index recipe in one place means the bulk indexer's
re-apply step can never drift out of sync with the collection's
original index layout. A drift would silently disable retrieval (e.g.
a missing INVERTED on `manufacturerName` falls back to a full scan).
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

from pymilvus import DataType, Function, FunctionType, MilvusClient

from indexer.projection import CATALOG_CURRENCIES

log = logging.getLogger(__name__)

DIM = 128

VECTOR_INDEX_DEFAULTS = {
    "HNSW": {"params": {"M": 16, "efConstruction": 200}},
    "IVF_FLAT": {"params": {"nlist": 4096}},
    "FLAT": {"params": {}},
}

# BM25 analyzer for `text_codes`. Conservative starting point matching
# today's `offers_codes` (whitespace + lowercase + length cap). F6's German
# pattern-replace + n-gram tokenization is absorbed by F9 PR3 (when
# `text_codes` content folding actually lands); revisit then.
BM25_ANALYZER_PARAMS = {
    "tokenizer": "whitespace",
    "filter": [
        "lowercase",
        {"type": "length", "min": 4, "max": 40},
    ],
}

# Article-level scalars that get an INVERTED index. Picked from the F9
# topology block — only fields that filter at article scope (categories,
# eclass hierarchies, manufacturer). Everything per-offer (vendor,
# catalog, price scope, core marker, relationships, ean, article_number,
# features, delivery) lives on `offers_v{N}` and is indexed there.
# `name` is stored but not indexed (retrieval/display field, BM25
# corpus is in `text_codes`).
#
# `manufacturerName` is INVERTED so the F9 dedup path's article-side
# `manufacturers_filter` can push down without a full collection scan.
ARTICLE_SCALAR_INDEX_FIELDS = (
    "manufacturerName",
    "category_l1",
    "category_l2",
    "category_l3",
    "category_l4",
    "category_l5",
    "eclass5_code",
    "eclass7_code",
    "s2class_code",
)

# Each field listed here gets an INVERTED scalar index on `offers_v{N}`.
# Picked to cover every filter / group_by / aggregation path called out
# in spec §4.3-§4.4 and in F3..F5. INVERTED handles equality, IN, range,
# and ARRAY membership uniformly on Milvus 2.6.15.
#
# F9: `article_hash` is the join key into `articles_v{N}`. Path B's
# bounded-probe `query()` filters on the per-offer scope here, returns
# the matching distinct hashes, and feeds them into the article-collection
# ANN as `article_hash IN [...]`. INVERTED on a 32-char VARCHAR is the
# right shape for the IN-clause workload (validated at 25k hashes ≈ 430ms
# p95 on the hardware ceiling benchmark — see F9 PATH_B_HASH_LIMIT).
OFFER_SCALAR_INDEX_FIELDS = (
    "article_hash",
    "vendor_id",
    "catalog_version_ids",
    "delivery_time_days_max",
    "features",
    "core_marker_enabled_sources",
    "core_marker_disabled_sources",
    "relationship_accessory_for",
    "relationship_spare_part_for",
    "relationship_similar_to",
    "ean",
    "article_number",
    "price_list_ids",
    "currencies",
)


# ---------- schemas -------------------------------------------------------


def build_articles_schema(client: MilvusClient):
    """Build the `articles_v{N}` schema. See
    `issues/article-search-replacement-ftsearch-09-article-dedup.md` for
    the F9 topology and field-by-field rationale."""
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

    # PK: sha256(name + manufacturerName + categories + eclass codes)
    # truncated to 16 bytes, hex-encoded → 32 chars.
    schema.add_field("article_hash", DataType.VARCHAR, max_length=32, is_primary=True)
    schema.add_field("offer_embedding", DataType.FLOAT16_VECTOR, dim=DIM)

    # BM25 input + output. `text_codes` is the union of identifier strings
    # across the article's offers (built by I1 / F9 PR2):
    #   name + " " + manufacturerName +
    #   " " + " ".join(distinct EANs across offers) +
    #   " " + " ".join(distinct article_numbers across offers)
    schema.add_field(
        "text_codes", DataType.VARCHAR, max_length=8192,
        enable_analyzer=True, analyzer_params=BM25_ANALYZER_PARAMS,
    )
    schema.add_field("sparse_codes", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="bm25_codes",
        function_type=FunctionType.BM25,
        input_field_names=["text_codes"],
        output_field_names=["sparse_codes"],
    ))

    schema.add_field("name", DataType.VARCHAR, max_length=1024)
    schema.add_field("manufacturerName", DataType.VARCHAR, max_length=256)

    # Category prefix-paths joined with `¦` (`|` escape per CategoryPath.java).
    schema.add_field("category_l1", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=256)
    schema.add_field("category_l2", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=640)
    schema.add_field("category_l3", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=768)
    schema.add_field("category_l4", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=1024)
    schema.add_field("category_l5", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=1280)

    # eClass / S2Class hierarchies — full root → leaf array.
    schema.add_field("eclass5_code", DataType.ARRAY, element_type=DataType.INT32, max_capacity=16)
    schema.add_field("eclass7_code", DataType.ARRAY, element_type=DataType.INT32, max_capacity=16)
    schema.add_field("s2class_code", DataType.ARRAY, element_type=DataType.INT32, max_capacity=16)

    # Customer-supplied SKU aliases. Inverted-by-value shape mirroring the
    # legacy ES Nested:
    #   [{"value": "BOLT-001", "version_ids": ["uuid-A", "uuid-C"]}, ...]
    # JSON is the only Milvus 2.6 shape that preserves the per-value→
    # version_ids relation without losing entitlement scoping. Filtering
    # uses the JSON predicate family (PR3); no scalar index here.
    schema.add_field("customer_article_numbers", DataType.JSON)

    # Per-currency envelope across all the article's offers. Sentinel for
    # "no price in this currency on this article": +MAX_PRICE_SENTINEL on
    # _min, -MAX_PRICE_SENTINEL on _max (Milvus 2.6 rejects NaN/±Inf;
    # see indexer/projection.py). Powers F4 sort-by-price browse via
    # STL_SORT ordered scan.
    for ccy in CATALOG_CURRENCIES:
        schema.add_field(f"{ccy}_price_min", DataType.FLOAT)
        schema.add_field(f"{ccy}_price_max", DataType.FLOAT)

    return schema


def build_offers_schema(client: MilvusClient):
    """Build the `offers_v{N}` schema. See
    `issues/article-search-replacement-ftsearch-09-article-dedup.md` for
    the F9 topology rationale (per-offer scope; vector + BM25 live on the
    paired articles collection)."""
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

    # PK: `{vendor_uuid_dashed}:{base64Url(articleNumber)}`. 256 leaves
    # ample headroom — UUID head is 36 chars, observed b64 tail tops out
    # ~65 chars in fixtures.
    schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)

    # Milvus 2.6 requires every collection to declare at least one vector
    # field with an index. Path B never searches this collection — only
    # `query()` on filter expressions — so we declare a 2-dim FLOAT
    # placeholder + FLAT index. Storage cost: ~1.3 GB at 159M rows
    # (negligible). The dense vector for retrieval lives on `articles_v{N}`.
    schema.add_field("_placeholder_vector", DataType.FLOAT_VECTOR, dim=2)

    schema.add_field("article_hash", DataType.VARCHAR, max_length=32)
    schema.add_field("ean", DataType.VARCHAR, max_length=64)
    schema.add_field("article_number", DataType.VARCHAR, max_length=256)
    schema.add_field("vendor_id", DataType.VARCHAR, max_length=64)

    schema.add_field(
        "catalog_version_ids", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=2048, max_length=64,
    )

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
    # Relationship arrays: per-offer references to other articles.
    # Empirically (10-shard prod sample) `relationship_accessory_for`
    # hit 3175 entries on a single offer; `relationship_similar_to`
    # 797. Capped at Milvus 2.6's hard ceiling (4096); the projection
    # truncates to this on emit. Some rows in this distribution look
    # pathological (a single offer claiming thousands of accessory
    # relationships) — flagged for separate data-quality review; the
    # cap + truncation here are sized to accept whatever the snapshot
    # ships rather than to validate it.
    schema.add_field(
        "relationship_accessory_for", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=4096, max_length=256,
    )
    schema.add_field(
        "relationship_spare_part_for", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=4096, max_length=256,
    )
    schema.add_field(
        "relationship_similar_to", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=4096, max_length=256,
    )

    # F8 price-scope pre-filter columns.
    schema.add_field(
        "price_list_ids", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=512, max_length=64,
    )
    schema.add_field(
        "currencies", DataType.ARRAY,
        element_type=DataType.VARCHAR, max_capacity=8, max_length=8,
    )
    for ccy in CATALOG_CURRENCIES:
        schema.add_field(f"{ccy}_price_min", DataType.FLOAT)
        schema.add_field(f"{ccy}_price_max", DataType.FLOAT)
    return schema


# ---------- index params -------------------------------------------------


def build_articles_index_params(client: MilvusClient, vector_index: str = "HNSW"):
    cfg = VECTOR_INDEX_DEFAULTS[vector_index]
    params = client.prepare_index_params()

    params.add_index(
        field_name="offer_embedding",
        index_type=vector_index,
        metric_type="COSINE",
        **cfg,
    )

    # BM25 sparse index. Mmap matches the existing offers_codes pattern —
    # only the inverted-index structures live in RAM, posting lists mmap
    # off disk.
    params.add_index(
        field_name="sparse_codes",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"mmap.enabled": True},
        index_name="sparse_codes",
    )

    # Mmap on all INVERTED scalar indexes. At 30% catalog scale the
    # combined heap of resident inverted dicts pushed Milvus standalone
    # to 165 GiB and triggered etcd-timeout-as-OOM-symptom restart loops;
    # mmap drops it to ~20 GiB by paging posting lists from disk.
    for field in ARTICLE_SCALAR_INDEX_FIELDS:
        params.add_index(
            field_name=field,
            index_type="INVERTED",
            index_name=field,
            params={"mmap.enabled": True},
        )

    # STL_SORT on every envelope column. Ordered scan for sort-by-price
    # browse (F4) is the hot path; range filters use the same index.
    # STL_SORT does not support mmap (Milvus rejects mmap.enabled on it).
    for ccy in CATALOG_CURRENCIES:
        for suffix in ("min", "max"):
            field = f"{ccy}_price_{suffix}"
            params.add_index(field_name=field, index_type="STL_SORT", index_name=field)

    return params


def build_offers_index_params(client: MilvusClient):
    params = client.prepare_index_params()
    # Required by Milvus — placeholder field so the collection has a
    # vector. Path B only `query()`s, never `search()`es this collection.
    params.add_index(
        field_name="_placeholder_vector",
        index_type="FLAT",
        metric_type="L2",
    )
    for field in OFFER_SCALAR_INDEX_FIELDS:
        params.add_index(
            field_name=field,
            index_type="INVERTED",
            index_name=field,
            params={"mmap.enabled": True},
        )

    # F8: STL_SORT on every per-currency envelope column. Path B's probe
    # composes `{ccy}_price_min <= decoded_max AND {ccy}_price_max >=
    # decoded_min` against these — STL_SORT is the right index for range
    # queries on FLOAT scalars. STL_SORT does not support mmap.
    for ccy in CATALOG_CURRENCIES:
        for suffix in ("min", "max"):
            field = f"{ccy}_price_{suffix}"
            params.add_index(field_name=field, index_type="STL_SORT", index_name=field)
    return params


# Vector-field DataType numeric IDs (FloatVector, Float16Vector,
# BFloat16Vector, SparseFloatVector, Int8Vector). Mmap doesn't apply to
# these via the field-level setter; vector mmap is configured on the
# index params (already done for sparse_codes).
_VECTOR_FIELD_TYPE_IDS = {101, 102, 103, 104, 105}


def enable_mmap_for_collection(client: MilvusClient, collection: str) -> None:
    """Enable field-level mmap on every non-vector field of `collection`.

    Field-level mmap pages raw scalar/array/JSON column storage from disk
    instead of resident heap. The collection should be unloaded before
    calling this; alter calls are safe afterwards but the change only
    takes effect on the next load.

    Index-level mmap is already wired into `build_*_index_params`; this
    handles the orthogonal field-level setting (which `drop_index` does
    not touch, so it persists across the bulk indexer's drop+rebuild)."""
    if not client.has_collection(collection):
        return
    desc = client.describe_collection(collection)
    n = 0
    for field in desc["fields"]:
        if field["type"] in _VECTOR_FIELD_TYPE_IDS:
            continue
        client.alter_collection_field(
            collection_name=collection,
            field_name=field["name"],
            field_params={"mmap.enabled": "true"},
        )
        n += 1
    log.info("  enabled field-level mmap on %d non-vector fields of %s", n, collection)


# ---------- index lifecycle (drop / re-apply) ----------------------------
#
# Used by the bulk indexer to avoid the per-segment inline IndexBuilding
# stage that dominates small-chunk bulk_insert wall time. Flow:
#
#   1. release_and_drop_indexes(client, collection)  ← before bulk_insert
#   2. ... do_bulk_insert into unindexed, unloaded collection ...
#   3. milvus.flush(collection)
#   4. apply_indexes_and_load(client, collection, params)  ← after flush
#
# Verified empirically (test on local Milvus 2.6.15): post-flush
# `create_index` on already-bulk-inserted segments DOES build the index
# to indexed_rows == total_rows. The earlier docstring claim of "silent
# no-op" referred to a different scenario (adding new index defs to a
# collection that already had inline-built indexes from bulk_insert).


def release_and_drop_indexes(
    client: MilvusClient,
    collection: str,
) -> list[str]:
    """Release `collection` from memory and drop every index on it.
    Returns the dropped index names (for logging). Idempotent — no-op
    if the collection has no indexes / isn't loaded.

    Releasing first is required: Milvus rejects `drop_index` on a
    loaded collection (the in-memory replicas reference the index).
    Sparse indexes in particular fail with a clear-but-late error if
    you skip the release."""
    if not client.has_collection(collection):
        return []
    # `release_collection` is idempotent — safe even if not loaded.
    client.release_collection(collection)
    names = list(client.list_indexes(collection))
    for n in names:
        client.drop_index(collection, index_name=n)
    return names


def apply_indexes_and_load(
    client: MilvusClient,
    collection: str,
    *,
    index_params,
    wait_for_index_seconds: float = 600.0,
    poll_interval_s: float = 2.0,
) -> None:
    """Create all indexes from `index_params`, wait for them to finish
    building (state=Finished, indexed_rows == total_rows), then load
    the collection.

    Loading before all indexes finish would block on the slowest builder
    anyway — we poll explicitly so we can surface progress and time out
    on a stuck IndexNode rather than hanging in `load_collection`."""
    # IndexParams iterates over IndexParam entries with `.field_name` and
    # `.index_name`. When `index_name` is unset (e.g. on the default-named
    # vector indexes) Milvus defaults the index name to the field name.
    index_names = [idx.index_name or idx.field_name for idx in index_params]
    log.info("  applying %d indexes to %s", len(index_names), collection)
    client.create_index(collection, index_params)

    deadline = time.time() + wait_for_index_seconds
    while True:
        states = [client.describe_index(collection, name) for name in index_names]
        unfinished = [
            (name, s.get("state"), s.get("indexed_rows"), s.get("total_rows"))
            for name, s in zip(index_names, states)
            if s.get("state") != "Finished"
        ]
        if not unfinished:
            break
        if time.time() > deadline:
            raise RuntimeError(
                f"apply_indexes_and_load: timed out waiting for indexes on "
                f"{collection}: still pending = {unfinished}"
            )
        log.info("  indexes pending on %s: %s", collection,
                 ", ".join(f"{n}={r}/{t}" for n, _, r, t in unfinished))
        time.sleep(poll_interval_s)

    client.load_collection(collection)
    log.info("  loaded %s after index rebuild", collection)


def list_collection_indexes(client: MilvusClient, collection: str) -> Iterable[str]:
    """Thin wrapper around `client.list_indexes`. Exists so callers can
    log/inspect without importing MilvusClient typing."""
    if not client.has_collection(collection):
        return []
    return client.list_indexes(collection)


__all__ = [
    "DIM",
    "CATALOG_CURRENCIES",
    "VECTOR_INDEX_DEFAULTS",
    "BM25_ANALYZER_PARAMS",
    "ARTICLE_SCALAR_INDEX_FIELDS",
    "OFFER_SCALAR_INDEX_FIELDS",
    "build_articles_schema",
    "build_articles_index_params",
    "build_offers_schema",
    "build_offers_index_params",
    "enable_mmap_for_collection",
    "release_and_drop_indexes",
    "apply_indexes_and_load",
    "list_collection_indexes",
]
