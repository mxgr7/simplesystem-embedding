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
    `{ccy}_price_min/max`).
  - NO `offer_embedding`, NO sparse codes — the dense vector and BM25
    index live on `articles_v{N}` keyed by hash.

Schema + index spec live in `indexer/collection_specs.py` so the bulk
indexer's drop-then-rebuild path can re-apply the exact same shape
post-bulk_insert. This script is a thin CLI shim over those builders.

See `issues/article-search-replacement-ftsearch-09-article-dedup.md` for
the topology rationale. Steady-state cutovers for `articles_v{N}` and
`offers_v{N}` are paired — see "Paired alias swing" in
`scripts/MILVUS_ALIAS_WORKFLOW.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymilvus import MilvusClient

from indexer.collection_specs import (  # noqa: E402  (sys.path hack above)
    CATALOG_CURRENCIES,
    OFFER_SCALAR_INDEX_FIELDS,
    build_offers_index_params,
    build_offers_schema,
)

# Re-export under the legacy name so existing imports
# (`from scripts.create_offers_collection import SCALAR_INDEX_FIELDS,
# build_schema, build_index_params`) keep working in tests + downstream.
SCALAR_INDEX_FIELDS = OFFER_SCALAR_INDEX_FIELDS


def build_schema(client: MilvusClient):
    return build_offers_schema(client)


def build_index_params(client: MilvusClient):
    return build_offers_index_params(client)


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
