"""HTTP client for the ftsearch service (`./search-api/`).

A2 builds the body via `mapping.request.map_request` and POSTs it
here. The response is returned raw (parsed JSON) — A3 wraps it in
the legacy envelope.

Connection pooling: a single `httpx.AsyncClient` per ACL process,
constructed in the FastAPI lifespan. Per-request timeout pinned at
4s by default — same shape as the F7 `BoundedMilvusClient` retry +
timeout pattern. A5 will layer on retries + tracing baggage
forwarding (mirror `search-api/embed_client.py`).
"""

from __future__ import annotations

from typing import Any

import httpx

# Match the ftsearch + indexer convention. A5 will turn the
# default into a config knob driven by the legacy SLO budget.
DEFAULT_TIMEOUT_S = 4.0


class FtsearchClient:
    """Async POST client for `{base_url}/{collection}/_search`.

    The legacy contract is endpoint-agnostic: `/article-features/search`
    doesn't carry a Milvus collection name. The ACL chooses one — by
    default the alias `articles` (post-F9 dedup topology) per the
    paired alias workflow in `scripts/MILVUS_ALIAS_WORKFLOW.md`."""

    def __init__(
        self,
        base_url: str,
        *,
        default_collection: str = "articles",
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_collection = default_collection
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def search(
        self,
        body: dict[str, Any],
        *,
        params: dict[str, Any] | None = None,
        collection: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        col = collection or self._default_collection
        resp = await self._client.post(
            f"{self._base_url}/{col}/_search",
            json=body,
            params=params,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["FtsearchClient", "DEFAULT_TIMEOUT_S"]
