"""Thin Milvus loader for tests + manual exploration (Phase A).

Loads a list of pre-projected rows into a Milvus collection with
deterministic stub vectors derived from the row's `name`. Use this when
you want F3..F5 integration tests to run against the real 200-doc
sample without standing up TEI / a real embedder.

Production bulk loading (real TEI, MongoDB scan, resume) is Phase B
(`indexer/bulk.py`).
"""

from __future__ import annotations

import hashlib
import time
from typing import Iterable

import numpy as np
from pymilvus import MilvusClient

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

    expected = len(materialised)
    deadline = time.time() + wait_timeout_s
    visible = 0
    while time.time() < deadline:
        rows_back = client.query(
            collection_name=collection,
            filter='id != ""',
            output_fields=["id"],
            limit=expected + 1,
        )
        visible = len(rows_back)
        if visible >= expected:
            return visible
        time.sleep(0.5)
    return visible
