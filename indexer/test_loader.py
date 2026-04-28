"""Thin Milvus loader for tests + manual exploration.

Two entry points:

  - `load_rows()` — loads pre-projected flat rows into a single legacy
    `offers` collection. Pre-F9 single-collection topology.
  - `load_split()` — loads projected rows into the post-F9 pair
    (`articles_v{N}` + `offers_v{N}`). Groups by `compute_article_hash`,
    aggregates article-level rows (text_codes + per-currency envelope),
    seeds stub vectors keyed by hash so the deduped article gets a
    deterministic embedding.

Production bulk loading (real TEI cache, MongoDB scan, Redis sidecar,
resume-via-PK) lands in F9 PR2b (`indexer/bulk.py` + `scripts/indexer_bulk.py`).
"""

from __future__ import annotations

import hashlib
import time
from typing import Iterable

import numpy as np
from pymilvus import MilvusClient

from indexer.projection import (
    CATALOG_CURRENCIES,
    aggregate_article,
    compute_article_hash,
    to_offer_row,
)

DEFAULT_DIM = 128


def stub_vector(name: str, *, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Deterministic 128-d fp16 vector seeded by the SHA-256 of `name`.
    Two rows with identical names get identical vectors; otherwise the
    vectors are uncorrelated. Vector quality is irrelevant for filter
    compliance testing — F3..F5 assertions look at hit IDs, not order."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float16)


def load_rows(
    client: MilvusClient,
    collection: str,
    rows: Iterable[dict],
    *,
    dim: int = DEFAULT_DIM,
    embedding_field: str = "offer_embedding",
    name_field: str = "name",
    wait_timeout_s: float = 30.0,
) -> int:
    """Upsert `rows` into `collection`, filling in `offer_embedding` from
    a stub vector seeded by each row's `name`. Blocks until rows are
    queryable (Milvus growing-segment lag fix).

    Returns the row count actually visible after the wait."""
    materialised = []
    for r in rows:
        if embedding_field not in r:
            r = {**r, embedding_field: stub_vector(r.get(name_field, ""), dim=dim)}
        materialised.append(r)
    if not materialised:
        return 0
    client.upsert(collection_name=collection, data=materialised)

    return _wait_visible(client, collection, len(materialised), wait_timeout_s, pk_field="id")


def load_split(
    client: MilvusClient,
    *,
    articles_collection: str,
    offers_collection: str,
    rows: Iterable[dict],
    dim: int = DEFAULT_DIM,
    currencies: tuple[str, ...] = CATALOG_CURRENCIES,
    wait_timeout_s: float = 30.0,
) -> tuple[int, int]:
    """Load projected flat rows into the F9 two-collection pair.

    Steps:
      1. Compute `article_hash` per row.
      2. Group offers by hash; build one `articles_v{N}` row per group via
         `aggregate_article` (text_codes + per-currency envelope).
      3. Stub-embed each article keyed by its hash (so all offers in the
         dedup'd group share a vector — matches the production invariant
         where TEI is called once per unique embedded-field tuple).
      4. Upsert both collections; wait until both are queryable.

    Returns `(articles_visible, offers_visible)`."""
    materialised = list(rows)
    if not materialised:
        return (0, 0)

    by_hash: dict[str, list[dict]] = {}
    offer_rows: list[dict] = []
    for r in materialised:
        h = compute_article_hash(r)
        by_hash.setdefault(h, []).append(r)
        offer_rows.append(to_offer_row(r, article_hash=h))

    article_rows = []
    for group in by_hash.values():
        article = aggregate_article(group, currencies=currencies)
        article["offer_embedding"] = stub_vector(article["article_hash"], dim=dim)
        article_rows.append(article)

    client.upsert(collection_name=articles_collection, data=article_rows)
    client.upsert(collection_name=offers_collection, data=offer_rows)

    articles_visible = _wait_visible(
        client, articles_collection, len(article_rows), wait_timeout_s,
        pk_field="article_hash",
    )
    offers_visible = _wait_visible(
        client, offers_collection, len(offer_rows), wait_timeout_s,
        pk_field="id",
    )
    return (articles_visible, offers_visible)


def _wait_visible(
    client: MilvusClient,
    collection: str,
    expected: int,
    timeout_s: float,
    *,
    pk_field: str = "id",
) -> int:
    """Block until `expected` rows are queryable in `collection`. Works
    around Milvus's growing-segment lag — newly upserted rows aren't
    immediately visible to filter-based queries."""
    deadline = time.time() + timeout_s
    visible = 0
    while time.time() < deadline:
        rows_back = client.query(
            collection_name=collection,
            filter=f'{pk_field} != ""',
            output_fields=[pk_field],
            limit=expected + 1,
        )
        visible = len(rows_back)
        if visible >= expected:
            return visible
        time.sleep(0.5)
    return visible
