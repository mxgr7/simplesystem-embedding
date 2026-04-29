"""HTTP client for a TEI-compatible embedding service.

TEI (HuggingFace Text Embeddings Inference) exposes ``POST /embed`` with
payload ``{"inputs": [str, ...]}`` and returns ``[[float, ...], ...]``. Any
server that speaks this shape is a drop-in replacement.

Per F7 §"Retries"/"Timeouts":
  - Per-call HTTP timeout pinned via the httpx client (default 4s,
    matches `milvus_helpers.DEFAULT_PER_CALL_TIMEOUT_S`).
  - Transient failures (HTTP 5xx, network/timeout errors) retried
    with the same exponential-backoff policy as Milvus reads. 4xx
    responses raise immediately — those are caller bugs (bad input
    text, oversized batch, etc.) that retrying won't fix.
  - Server overload (503 + Retry-After) is treated as transient.
  - `truncate=True` is set per-request (was already set at the indexer
    layer; same flag here means a query string longer than the
    model's max_input_length doesn't 413 the request).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

DEFAULT_PER_CALL_TIMEOUT_S = 4.0


@dataclass(frozen=True)
class EmbedRetryPolicy:
    """Mirrors `milvus_helpers.RetryPolicy`. Defaults match the legacy
    Java service's retry policy (5/500ms/1.5×/5s)."""
    max_attempts: int = 5
    initial_backoff_s: float = 0.5
    multiplier: float = 1.5
    max_single_backoff_s: float = 5.0
    total_budget_s: float = 5.0


# httpx exceptions worth retrying — network-level failures, server-side
# 5xx, and timeouts. 4xx responses are caller errors; a retry won't fix
# a malformed payload or an oversized batch.
_TRANSIENT_HTTP_STATUSES = frozenset({500, 502, 503, 504, 408, 429})


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_HTTP_STATUSES
    return isinstance(exc, (
        httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
        httpx.PoolTimeout, httpx.RemoteProtocolError,
    ))


class EmbedClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_PER_CALL_TIMEOUT_S,
        retry_policy: EmbedRetryPolicy | None = EmbedRetryPolicy(),
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)
        self._policy = retry_policy

    async def embed(
        self,
        texts: list[str],
        *,
        headers: dict[str, str] | None = None,
    ) -> list[list[float]]:
        """`headers` propagates W3C trace context (traceparent /
        tracestate / baggage) per F7 §"Tracing baggage". Caller is
        `main.py:_search_dedup`'s `embed_fn` closure, which builds
        the headers dict from `request.state.trace_ctx`."""
        if self._policy is None:
            return await self._embed_once(texts, headers=headers)

        loop = asyncio.get_event_loop()
        started = loop.time()
        last_exc: BaseException | None = None
        for attempt in range(self._policy.max_attempts):
            try:
                return await self._embed_once(texts, headers=headers)
            except BaseException as e:
                last_exc = e
                if not _is_transient(e):
                    log.warning("embed: non-transient error (%s) — not retrying", e)
                    raise
                if attempt == self._policy.max_attempts - 1:
                    log.warning("embed: exhausted %d attempts — %s",
                                self._policy.max_attempts, e)
                    raise
                elapsed = loop.time() - started
                wait = min(
                    self._policy.initial_backoff_s * (self._policy.multiplier ** attempt),
                    self._policy.max_single_backoff_s,
                )
                if elapsed + wait > self._policy.total_budget_s:
                    log.warning("embed: total budget %.1fs exhausted at attempt %d — %s",
                                self._policy.total_budget_s, attempt + 1, e)
                    raise
                log.info("embed attempt %d/%d failed (%s) — retrying in %.2fs",
                         attempt + 1, self._policy.max_attempts, e, wait)
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable") from last_exc

    async def _embed_once(
        self,
        texts: list[str],
        *,
        headers: dict[str, str] | None = None,
    ) -> list[list[float]]:
        # `truncate=True` makes TEI right-truncate inputs longer than
        # the model's max_input_length instead of 413'ing — same
        # behaviour as the indexer-side `tei_cache.py:_tei_embed`.
        resp = await self._client.post(
            f"{self._base_url}/embed",
            json={"inputs": texts, "truncate": True},
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()
