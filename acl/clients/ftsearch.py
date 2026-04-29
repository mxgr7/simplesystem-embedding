"""HTTP client for the ftsearch service (`./search-api/`).

Wraps each call with the same operational glue F7 added on the
ftsearch side:

  - Per-call timeout (default 4.5s — fits inside the legacy p99 SLO
    of 5s, leaves ~500ms ACL budget for the request mapping +
    response shaping).
  - Retry on transient failures: 5 attempts, 500ms initial backoff,
    1.5× multiplier, capped at 5s per single backoff and 5s total
    backoff budget. Mirrors `search-api/embed_client.py:EmbedRetryPolicy`.
  - 4xx responses raise immediately (caller bug — bad input shouldn't
    spend retry attempts).
  - Tracing baggage propagation via the optional `headers=` kwarg
    populated by `acl/main.py` from the inbound trace context.

A4 owns the legacy error envelope; this client just raises typed
httpx exceptions and lets the FastAPI handler categorise.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from acl.metrics import (
    record_call,
    record_retry_exhausted,
    record_retry_fired,
)

log = logging.getLogger(__name__)

# Default 4.5s — fits inside the legacy p99 SLO of 5s with ~500ms
# headroom for the ACL's own request mapping + response shaping.
# Operators can override via FTSEARCH_TIMEOUT_MS.
DEFAULT_TIMEOUT_S = float(os.environ.get("FTSEARCH_TIMEOUT_MS", "4500")) / 1000.0


@dataclass(frozen=True)
class RetryPolicy:
    """Mirrors `search-api/milvus_helpers.py:RetryPolicy` and the
    legacy Java service (`management.tracing.retry.*`)."""
    max_attempts: int = 5
    initial_backoff_s: float = 0.5
    multiplier: float = 1.5
    max_single_backoff_s: float = 5.0
    total_budget_s: float = 5.0


# httpx exceptions worth retrying — 5xx, throttling, and network/timeouts.
# 4xx responses are caller bugs; not retryable.
_TRANSIENT_HTTP_STATUSES = frozenset({500, 502, 503, 504, 408, 429})


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_HTTP_STATUSES
    return isinstance(exc, (
        httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
        httpx.PoolTimeout, httpx.RemoteProtocolError,
    ))


def _outcome_for(exc: BaseException) -> str:
    """Map an exception type to one of `metrics.OUTCOME_LABELS`."""
    if isinstance(exc, httpx.HTTPStatusError):
        return "upstream_5xx" if exc.response.status_code >= 500 else "upstream_4xx"
    return "network_error"


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
        retry_policy: RetryPolicy | None = RetryPolicy(),
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_collection = default_collection
        self._client = httpx.AsyncClient(timeout=timeout_s)
        self._policy = retry_policy

    async def search(
        self,
        body: dict[str, Any],
        *,
        params: dict[str, Any] | None = None,
        collection: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        col = collection or self._default_collection
        if self._policy is None:
            return await self._search_once(col, body, params=params, headers=headers)

        loop = asyncio.get_event_loop()
        started = loop.time()
        last_exc: BaseException | None = None
        for attempt in range(self._policy.max_attempts):
            try:
                return await self._search_once(col, body, params=params, headers=headers)
            except BaseException as e:
                last_exc = e
                if not _is_transient(e):
                    log.warning("ftsearch: non-transient error (%s) — not retrying", e)
                    raise
                if attempt == self._policy.max_attempts - 1:
                    log.warning("ftsearch: exhausted %d attempts — %s",
                                self._policy.max_attempts, e)
                    record_retry_exhausted()
                    raise
                elapsed = loop.time() - started
                wait = min(
                    self._policy.initial_backoff_s * (self._policy.multiplier ** attempt),
                    self._policy.max_single_backoff_s,
                )
                if elapsed + wait > self._policy.total_budget_s:
                    log.warning(
                        "ftsearch: total budget %.1fs exhausted at attempt %d — %s",
                        self._policy.total_budget_s, attempt + 1, e,
                    )
                    record_retry_exhausted()
                    raise
                log.info("ftsearch attempt %d/%d failed (%s) — retrying in %.2fs",
                         attempt + 1, self._policy.max_attempts, e, wait)
                record_retry_fired()
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable") from last_exc

    async def _search_once(
        self,
        col: str,
        body: dict[str, Any],
        *,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        try:
            resp = await self._client.post(
                f"{self._base_url}/{col}/_search",
                json=body,
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            record_call("success", loop.time() - t0)
            return resp.json()
        except BaseException as e:
            record_call(_outcome_for(e), loop.time() - t0)
            raise

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["FtsearchClient", "RetryPolicy", "DEFAULT_TIMEOUT_S"]
