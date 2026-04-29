"""Operational helpers around `pymilvus.MilvusClient`.

The F7 spec pins a few invariants that benefit from being applied at a
single layer rather than each call site:

  - **Bounded consistency** on every read. Milvus's default is
    `Bounded` for `search`/`query` *if no explicit value is provided
    AND the collection's per-collection default is bounded*, but the
    safer pattern (per F7 §"Milvus consistency level") is to set it
    explicitly so a future change to the cluster default doesn't
    silently shift latency-sensitive reads to `Strong`. We pin
    `Bounded` on every search/query/get unless the caller passes a
    different value explicitly.

  - Future hooks: tracing baggage propagation, retry, timeout, and
    metrics — the same wrapper is the natural place for them. F7
    follow-ups land here without a second ripple of call-site edits.

Use via:

    from milvus_helpers import BoundedMilvusClient
    raw = MilvusClient(uri=...)
    client = BoundedMilvusClient(raw)
    client.search(collection_name=..., data=..., ...)   # → consistency_level='Bounded'

Pass-through: any method not on this wrapper falls through to the
underlying client (e.g. `has_collection`, `describe_collection`,
`upsert`, `flush`) — those are write paths or schema reads where the
consistency level doesn't apply.
"""

from __future__ import annotations

from typing import Any

from pymilvus import MilvusClient

# Per F7 §"Milvus consistency level" — see the spec linked from the
# module docstring for the `Bounded` rationale (latency-bounded
# staleness instead of `Strong` synchronous coordination).
DEFAULT_CONSISTENCY_LEVEL = "Bounded"


class BoundedMilvusClient:
    """Thin wrapper that pins `consistency_level='Bounded'` on every
    read call (search/query/get). Pass-through for everything else."""

    __slots__ = ("_client",)

    def __init__(self, client: MilvusClient) -> None:
        # Bypass our own __setattr__ if we ever add one — for now
        # __slots__ keeps attribute creation explicit.
        object.__setattr__(self, "_client", client)

    # ---- consistency-pinned read methods ---------------------------

    def search(self, **kwargs: Any) -> Any:
        kwargs.setdefault("consistency_level", DEFAULT_CONSISTENCY_LEVEL)
        return self._client.search(**kwargs)

    def query(self, **kwargs: Any) -> Any:
        kwargs.setdefault("consistency_level", DEFAULT_CONSISTENCY_LEVEL)
        return self._client.query(**kwargs)

    def get(self, **kwargs: Any) -> Any:
        kwargs.setdefault("consistency_level", DEFAULT_CONSISTENCY_LEVEL)
        return self._client.get(**kwargs)

    # ---- pass-through for everything else --------------------------
    # `__getattr__` only fires when normal attribute lookup misses, so
    # `self._client` (set in __init__) doesn't recurse here.

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


__all__ = ["BoundedMilvusClient", "DEFAULT_CONSISTENCY_LEVEL"]
