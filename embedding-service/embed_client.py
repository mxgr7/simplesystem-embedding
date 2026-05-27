"""Async TEI HTTP client.

TEI's native contract: `POST /embed` with `{"inputs": [str, ...],
"truncate": bool}` returning `[[float, ...], ...]`. We split incoming
miss-text lists into chunks of `≤ max_client_batch` (matching TEI's
`--max-client-batch-size`) and dispatch up to `max_concurrency` in
parallel, gated by an `asyncio.Semaphore`. Same shape as the indexer
client in `indexer/tei_cache.py`, just async and with retries.

Retry policy mirrored from `search-api/embed_client.py:EmbedRetryPolicy`
— 5 attempts, 5s total budget, exp backoff, retry on 5xx/408/429/network.
Non-transient HTTP errors raise immediately (caller bug, no point
retrying).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    initial_backoff_s: float = 0.5
    multiplier: float = 1.5
    max_single_backoff_s: float = 5.0
    total_budget_s: float = 5.0


_TRANSIENT_HTTP_STATUSES = frozenset({500, 502, 503, 504, 408, 429})


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_HTTP_STATUSES
    return isinstance(exc, (
        httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
        httpx.PoolTimeout, httpx.RemoteProtocolError,
    ))


class TEIClient:
    """One client per process, lives on app.state. Thread-safe by virtue
    of httpx.AsyncClient + asyncio cooperative scheduling."""

    def __init__(
        self,
        base_url: str,
        *,
        max_client_batch: int,
        max_concurrency: int,
        per_call_timeout_s: float = 4.0,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_client_batch = max_client_batch
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            timeout=per_call_timeout_s,
            limits=httpx.Limits(
                max_connections=max_concurrency * 2,
                max_keepalive_connections=max_concurrency,
            ),
        )
        self._policy = retry_policy or RetryPolicy()
        # Histogram bucket for "time spent waiting for a TEI semaphore
        # slot". main.py reads this via the semaphore_wait_seconds metric.
        self._sem_wait_total_s = 0.0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, texts: list[str], *, truncate: bool = True) -> np.ndarray:
        """Embed `texts` via TEI; return shape (n, 128) fp16.

        `truncate` is forwarded to TEI per-request. Decoded directly into
        fp16 (no fp32 detour) — the model is fp16 and we never need the
        extra precision."""
        if not texts:
            return np.empty((0, 0), dtype=np.float16)

        chunks = [
            texts[i : i + self._max_client_batch]
            for i in range(0, len(texts), self._max_client_batch)
        ]
        # `gather` preserves input order in results.
        chunk_arrays = await asyncio.gather(
            *(self._embed_chunk(c, truncate=truncate) for c in chunks)
        )
        return np.concatenate(chunk_arrays, axis=0)

    async def _embed_chunk(self, chunk: list[str], *, truncate: bool) -> np.ndarray:
        """One TEI call, semaphore-gated, retried on transient failure.

        Direct fp32→fp16 decode in `np.asarray(..., dtype=np.float16)`."""
        wait_started = time.perf_counter()
        async with self._sem:
            self._sem_wait_total_s += time.perf_counter() - wait_started
            json_resp = await self._post_with_retry(chunk, truncate=truncate)
        arr = np.asarray(json_resp, dtype=np.float16)
        return arr

    async def _post_with_retry(
        self, chunk: list[str], *, truncate: bool
    ) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        started = loop.time()
        last_exc: BaseException | None = None
        for attempt in range(self._policy.max_attempts):
            try:
                return await self._post_once(chunk, truncate=truncate)
            except BaseException as e:
                last_exc = e
                if not _is_transient(e):
                    log.warning("TEI: non-transient error (%s) — not retrying", e)
                    raise
                if attempt == self._policy.max_attempts - 1:
                    log.warning("TEI: exhausted %d attempts — %s",
                                self._policy.max_attempts, e)
                    raise
                elapsed = loop.time() - started
                wait = min(
                    self._policy.initial_backoff_s * (self._policy.multiplier ** attempt),
                    self._policy.max_single_backoff_s,
                )
                if elapsed + wait > self._policy.total_budget_s:
                    log.warning("TEI: total budget %.1fs exhausted at attempt %d — %s",
                                self._policy.total_budget_s, attempt + 1, e)
                    raise
                log.info("TEI attempt %d/%d failed (%s) — retrying in %.2fs",
                         attempt + 1, self._policy.max_attempts, e, wait)
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable") from last_exc

    async def _post_once(
        self, chunk: list[str], *, truncate: bool
    ) -> list[list[float]]:
        resp = await self._client.post(
            f"{self._base_url}/embed",
            json={"inputs": chunk, "truncate": truncate},
        )
        resp.raise_for_status()
        return resp.json()

    @property
    def semaphore_wait_total_s(self) -> float:
        return self._sem_wait_total_s
