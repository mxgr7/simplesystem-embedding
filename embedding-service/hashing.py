"""Hashing + cache-key derivation for the embedding-service.

The hash is computed over the raw NUL-separated 8-field input string the
client sends — *not* over the rendered template. Two clients producing
the same NUL-joined input therefore share cache entries, regardless of
template changes that don't affect input semantics.

Wire format note: a HASH_VERSION bump must accompany any template/
preprocessing change that would shift the embedding for a fixed input,
so stale cache entries don't get served to the new code path. We share
the `tei:v2:` keyspace with `indexer/tei_cache.py` (which keeps its own
copy of HASH_VERSION in `indexer/projection.py` per the "copy, don't
share" policy).
"""

from __future__ import annotations

import hashlib

HASH_VERSION = "v2"


def article_hash(text: str) -> str:
    """SHA-256 truncated to 16 bytes → 32-char lowercase hex.

    Collision probability at 1.6e8 articles ~ 4e-23 — same posture as
    the indexer-side hash."""
    return hashlib.sha256(text.encode("utf-8")).digest()[:16].hex()


def cache_key(text_hash: str) -> str:
    return f"tei:{HASH_VERSION}:{text_hash}"
