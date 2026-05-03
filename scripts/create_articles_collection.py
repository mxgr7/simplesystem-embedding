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

Schema + index spec live in `indexer/collection_specs.py` so the bulk
indexer's drop-then-rebuild path can re-apply the exact same shape
post-bulk_insert. This script is a thin CLI shim over those builders.

Naming is operator-driven: pick `--version N` higher than the current
articles version. ftsearch never embeds the versioned name; it talks to
the alias.

Steady-state cutovers for `articles_v{N}` and `offers_v{N}` are paired —
see the "Paired alias swing" section in `scripts/MILVUS_ALIAS_WORKFLOW.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `indexer.*` importable when this script is invoked directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymilvus import MilvusClient

from indexer.collection_specs import (  # noqa: E402  (sys.path hack above)
    ARTICLE_SCALAR_INDEX_FIELDS,
    BM25_ANALYZER_PARAMS,
    CATALOG_CURRENCIES,
    DIM,
    VECTOR_INDEX_DEFAULTS,
    build_articles_index_params,
    build_articles_schema,
    enable_mmap_for_collection,
)

# Keep these names available at this module path so existing imports
# (`from scripts.create_articles_collection import SCALAR_INDEX_FIELDS,
# build_schema, build_index_params`) keep working in tests + downstream.
SCALAR_INDEX_FIELDS = ARTICLE_SCALAR_INDEX_FIELDS


def build_schema(client: MilvusClient):
    return build_articles_schema(client)


def build_index_params(client: MilvusClient, vector_index: str):
    return build_articles_index_params(client, vector_index)


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

    enable_mmap_for_collection(client, name)
    print(f"Enabled field-level mmap on {name!r}.")

    client.load_collection(name)
    print(f"Loaded {name!r}.")

    if not args.no_alias:
        swing_alias(client, args.alias, name)


if __name__ == "__main__":
    main()
