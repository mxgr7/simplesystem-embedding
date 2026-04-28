"""Create a versioned `articles_v{N}` Milvus collection per F9 topology
and register the public alias (`articles` by default) to point at it.

Usage:
    python create_articles_collection.py --version 1
    python create_articles_collection.py --version 2 --alias articles
    python create_articles_collection.py --version 2 --no-alias
    python create_articles_collection.py --version 2 --dry-run

F9 splits the legacy single-collection topology into two:

    `articles_v{N}` — vector + BM25 + article-level scalars + per-currency
                     envelope. PK is `article_hash` (sha256-truncated to
                     16 bytes, hex). One row per unique embedded-field
                     tuple.
    `offers_v{N}`   — per-offer scalars; `article_hash` join key; no
                     vectors. See `create_offers_collection.py`.

Together they support correlated per-offer filtering (catalog × price-list ×
price-range × core-marker on the *same* offer) at ~130M embeddings vs
~510M if we'd flattened to one row per offer. See
`issues/article-search-replacement-ftsearch-09-article-dedup.md` for the
design rationale and the Milvus 2.6 capability survey that pins the
two-collection split as the only viable shape.

Naming is operator-driven: pick `--version N` higher than the current
articles version. ftsearch never embeds the versioned name; it talks to
the alias.

Steady-state cutovers for `articles_v{N}` and `offers_v{N}` are paired —
see the "Paired alias swing" section in `scripts/MILVUS_ALIAS_WORKFLOW.md`.
"""

from __future__ import annotations

import argparse
import sys

from pymilvus import DataType, Function, FunctionType, MilvusClient

DIM = 128

# Currencies the catalogue carries today (per F8). Each gets a paired
# (min, max) FLOAT envelope column on the article row, refreshed at bulk
# reindex (streaming updates owned by I2). Used by the F4 sort-by-price
# browse path (no queryString) for ordered scan against
# `{ccy}_price_min ASC` — never used as a filter (the precise per-offer
# envelope lives on `offers_v{N}`, F8's territory).
#
# Add a new currency: append here and bump the collection version.
CATALOG_CURRENCIES = ("eur", "chf", "huf", "pln", "gbp", "czk", "cny")

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
# eclass hierarchies). Everything per-offer (vendor, catalog, price scope,
# core marker, relationships, ean, article_number, features, delivery)
# lives on `offers_v{N}` and is indexed there. `name` /
# `manufacturerName` are stored but not indexed — they're retrieval /
# response fields, not filters.
SCALAR_INDEX_FIELDS = (
    "category_l1",
    "category_l2",
    "category_l3",
    "category_l4",
    "category_l5",
    "eclass5_code",
    "eclass7_code",
    "s2class_code",
)


def build_schema(client: MilvusClient):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

    # PK: sha256(name + manufacturerName + categories + eclass codes)
    # truncated to 16 bytes, hex-encoded → 32 chars. See F9 "Hash function
    # and embedded-field set". Idempotent upsert by hash.
    schema.add_field("article_hash", DataType.VARCHAR, max_length=32, is_primary=True)
    schema.add_field("offer_embedding", DataType.FLOAT16_VECTOR, dim=DIM)

    # BM25 input + output. `text_codes` is the union of identifier strings
    # across the article's offers (built by I1 / F9 PR2):
    #   name + " " + manufacturerName +
    #   " " + " ".join(distinct EANs across offers) +
    #   " " + " ".join(distinct article_numbers across offers)
    # Sized to fit name(1024) + manufacturer(256) + many identifiers with
    # headroom.
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

    # Article-level retrieval / display fields.
    schema.add_field("name", DataType.VARCHAR, max_length=1024)
    schema.add_field("manufacturerName", DataType.VARCHAR, max_length=256)

    # Category prefix-paths joined with `¦` (`|` escape per CategoryPath.java).
    # Sized identically to offers_v{N} for parity.
    schema.add_field("category_l1", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=256)
    schema.add_field("category_l2", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=640)
    schema.add_field("category_l3", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=768)
    schema.add_field("category_l4", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=1024)
    schema.add_field("category_l5", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=1280)

    # eClass / S2Class hierarchies — full root → leaf array. See F1 / F3
    # for the recall-correctness rationale (single-INT collapses the
    # hierarchy).
    schema.add_field("eclass5_code", DataType.ARRAY, element_type=DataType.INT32, max_capacity=16)
    schema.add_field("eclass7_code", DataType.ARRAY, element_type=DataType.INT32, max_capacity=16)
    schema.add_field("s2class_code", DataType.ARRAY, element_type=DataType.INT32, max_capacity=16)

    # Per-currency envelope across all the article's offers. Two FLOAT
    # columns per currency; NaN sentinel for "no price in this currency
    # on this article" (Milvus comparison against NaN is false, so range
    # predicates naturally exclude these rows). Powers F4 sort-by-price
    # browse (ordered scan via STL_SORT). Never a filter input — F8
    # places the precise per-offer envelope on `offers_v{N}`.
    for ccy in CATALOG_CURRENCIES:
        schema.add_field(f"{ccy}_price_min", DataType.FLOAT)
        schema.add_field(f"{ccy}_price_max", DataType.FLOAT)

    return schema


def build_index_params(client: MilvusClient, vector_index: str):
    cfg = VECTOR_INDEX_DEFAULTS[vector_index]
    params = client.prepare_index_params()

    # Dense vector index.
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

    # Article-scope filter scalars.
    for field in SCALAR_INDEX_FIELDS:
        params.add_index(field_name=field, index_type="INVERTED", index_name=field)

    # STL_SORT on every envelope column. Ordered scan for sort-by-price
    # browse (F4) is the hot path; range filters use the same index.
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
    p.add_argument("--version", type=int, required=True, help="Integer N — produces collection name articles_v{N}.")
    p.add_argument("--alias", default="articles", help="Public alias (default 'articles').")
    p.add_argument("--no-alias", action="store_true", help="Skip alias create/swing.")
    p.add_argument("--uri", default="http://localhost:19530", help="Milvus URI.")
    p.add_argument("--vector-index", default="HNSW", choices=list(VECTOR_INDEX_DEFAULTS))
    p.add_argument("--dry-run", action="store_true", help="Print plan and exit.")
    args = p.parse_args()

    name = f"articles_v{args.version}"
    print(f"Target collection: {name}")
    print(f"Alias:             {'(skipped)' if args.no_alias else args.alias}")
    print(f"Milvus URI:        {args.uri}")
    print(f"Vector index:      {args.vector_index}")
    print(f"Currencies:        {len(CATALOG_CURRENCIES)} ({', '.join(CATALOG_CURRENCIES)})")
    print(f"Scalar indexes:    {len(SCALAR_INDEX_FIELDS)} + {len(CATALOG_CURRENCIES) * 2} STL_SORT")
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
