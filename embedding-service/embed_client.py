"""Async TEI HTTP client — a runtime-mutable pool of TEI backends.

TEI's native contract: `POST /embed` with `{"inputs": [str, ...],
"truncate": bool}` returning `[[float, ...], ...]`. We split incoming
miss-text lists into chunks of `≤ chunk_size` and dispatch them across a
*pool* of TEI backends, each gated by its own `asyncio.Semaphore`.

A `TEIBackend` is one TEI URL (its own httpx client, semaphore, health
state, weight). A `TEIPool` holds N of them and routes each chunk to the
healthiest least-loaded backend, with in-request failover to a different
backend on transient errors. Backends can be added / drained / removed at
runtime (see `main.py`'s `/admin/backends` routes) so a one-off indexing
run can fan out onto a bigger remote GPU instance without restarting the
service or any client noticing.

Retry policy mirrored from `search-api/embed_client.py:EmbedRetryPolicy`
— 5 attempts, 5s total budget, exp backoff, retry on 5xx/408/429/network.
Non-transient HTTP errors raise immediately (caller bug, no point
retrying). With a multi-backend pool a transient failure first fails over
to a *different* healthy backend before falling back to backoff-and-retry.
"""

from __future__ import annotations

import asyncio
import itertools
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

# Consecutive failures (real requests or health probes) before a backend is
# marked unhealthy and skipped in selection; one success restores it.
_UNHEALTHY_AFTER = 2


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_HTTP_STATUSES
    return isinstance(exc, (
        httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
        httpx.PoolTimeout, httpx.RemoteProtocolError, httpx.ConnectTimeout,
    ))


class TEIBackend:
    """One TEI endpoint. Owns its httpx client + concurrency semaphore and
    tracks live inflight count and health. Selection-relevant mutable state
    (`inflight`, `healthy`, `weight`, `draining`) is only ever touched from
    the event loop, so no locking is needed."""

    def __init__(
        self,
        backend_id: str,
        base_url: str,
        *,
        weight: float,
        max_client_batch: int,
        max_concurrency: int,
        timeout_s: float = 30.0,
    ) -> None:
        self.id = backend_id
        self.base_url = base_url.rstrip("/")
        self.weight = weight
        self.max_client_batch = max_client_batch
        self.max_concurrency = max_concurrency
        self.timeout_s = timeout_s

        self.inflight = 0            # assigned-or-running chunks (for load balancing)
        self.healthy = False         # unhealthy until the first probe succeeds
        self.draining = False        # set on remove → excluded from selection
        self.consecutive_failures = 0

        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_s,
            limits=httpx.Limits(
                max_connections=max_concurrency * 2,
                max_keepalive_connections=max_concurrency,
            ),
        )
        self._sem_wait_total_s = 0.0

    # --- selection helpers ---------------------------------------------

    @property
    def selectable(self) -> bool:
        return self.healthy and not self.draining

    @property
    def load(self) -> float:
        """Lower is better. inflight per unit weight, so a heavier (more
        capable) backend has to be proportionally busier before it's
        skipped in favour of a lighter one."""
        return self.inflight / self.weight if self.weight > 0 else float("inf")

    def mark_success(self) -> None:
        self.consecutive_failures = 0
        if not self.healthy:
            log.info("TEI backend %s (%s) healthy again", self.id, self.base_url)
        self.healthy = True

    def mark_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= _UNHEALTHY_AFTER and self.healthy:
            log.warning(
                "TEI backend %s (%s) marked unhealthy after %d failures",
                self.id, self.base_url, self.consecutive_failures,
            )
            self.healthy = False

    # --- IO ------------------------------------------------------------

    async def post(self, chunk: list[str], *, truncate: bool) -> list[list[float]]:
        """One TEI call, semaphore-gated. `inflight` is incremented by the
        pool's selector *before* this is called; we decrement it here."""
        wait_started = time.perf_counter()
        try:
            async with self._sem:
                self._sem_wait_total_s += time.perf_counter() - wait_started
                resp = await self._client.post(
                    "/embed", json={"inputs": chunk, "truncate": truncate},
                )
                resp.raise_for_status()
                return resp.json()
        finally:
            self.inflight -= 1

    async def probe(self) -> None:
        """Hit TEI's `/health`; feed the result into the same health
        counter real requests use. Short timeout — a slow health check
        shouldn't itself look like an outage."""
        try:
            resp = await self._client.get("/health", timeout=2.0)
            resp.raise_for_status()
            self.mark_success()
        except Exception as e:  # noqa: BLE001 — any failure = not healthy
            self.mark_failure()
            log.debug("TEI backend %s probe failed: %s", self.id, e)

    async def aclose(self) -> None:
        await self._client.aclose()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.base_url,
            "weight": self.weight,
            "healthy": self.healthy,
            "draining": self.draining,
            "inflight": self.inflight,
            "max_concurrency": self.max_concurrency,
            "max_client_batch": self.max_client_batch,
        }


class TEIPool:
    """Runtime-mutable set of `TEIBackend`s. Lives on `app.state.tei` in
    place of the old single client; `embed()` has the same signature so
    `main.py`'s handler is unchanged."""

    def __init__(
        self,
        *,
        retry_policy: RetryPolicy | None = None,
        probe_interval_s: float = 5.0,
        drain_timeout_s: float = 30.0,
        on_change=None,
    ) -> None:
        self._backends: dict[str, TEIBackend] = {}
        self._ids = itertools.count(1)
        self._policy = retry_policy or RetryPolicy()
        self._probe_interval_s = probe_interval_s
        self._drain_timeout_s = drain_timeout_s
        self._on_change = on_change  # callback(list[dict]) for metrics refresh
        self._mutate = asyncio.Lock()
        self._probe_task: asyncio.Task | None = None

    # --- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._probe_task is None:
            self._probe_task = asyncio.create_task(self._probe_loop())

    async def aclose(self) -> None:
        if self._probe_task is not None:
            self._probe_task.cancel()
            try:
                await self._probe_task
            except asyncio.CancelledError:
                pass
        await asyncio.gather(
            *(b.aclose() for b in self._backends.values()),
            return_exceptions=True,
        )

    # --- mutation ------------------------------------------------------

    async def add_backend(
        self,
        url: str,
        *,
        weight: float,
        max_concurrency: int,
        max_client_batch: int,
        timeout_s: float,
    ) -> dict:
        async with self._mutate:
            backend_id = f"b{next(self._ids)}"
            backend = TEIBackend(
                backend_id, url,
                weight=weight,
                max_client_batch=max_client_batch,
                max_concurrency=max_concurrency,
                timeout_s=timeout_s,
            )
            self._backends[backend_id] = backend
        log.info("added TEI backend %s (%s) weight=%s", backend_id, url, weight)
        asyncio.create_task(backend.probe())  # confirm health promptly
        self._notify()
        return backend.to_dict()

    async def set_weight(self, backend_id: str, weight: float) -> dict:
        backend = self._backends.get(backend_id)
        if backend is None:
            raise KeyError(backend_id)
        backend.weight = weight
        log.info("set TEI backend %s weight=%s", backend_id, weight)
        self._notify()
        return backend.to_dict()

    async def remove_backend(self, backend_id: str) -> dict:
        """Graceful: mark draining (excluded from new selections) and close
        once inflight reaches zero, in the background. Returns immediately
        with the draining snapshot."""
        backend = self._backends.get(backend_id)
        if backend is None:
            raise KeyError(backend_id)
        backend.draining = True
        log.info("draining TEI backend %s (%s)", backend_id, backend.base_url)
        asyncio.create_task(self._drain_and_close(backend))
        self._notify()
        return backend.to_dict()

    async def _drain_and_close(self, backend: TEIBackend) -> None:
        deadline = time.monotonic() + self._drain_timeout_s
        while backend.inflight > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        async with self._mutate:
            self._backends.pop(backend.id, None)
        await backend.aclose()
        log.info(
            "removed TEI backend %s (%s) — drained %d inflight",
            backend.id, backend.base_url, backend.inflight,
        )
        self._notify()

    def list_backends(self) -> list[dict]:
        return [b.to_dict() for b in self._backends.values()]

    def _notify(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change(self.list_backends())
            except Exception:  # noqa: BLE001 — metrics must never break routing
                log.exception("TEIPool on_change callback failed")

    # --- selection -----------------------------------------------------

    def _chunk_size(self) -> int:
        """A chunk can be routed to *any* backend (incl. on failover), so
        it must not exceed the smallest backend's max-client-batch or that
        backend would 413. Defaults to 8 when the pool is momentarily
        empty."""
        sizes = [b.max_client_batch for b in self._backends.values()]
        return min(sizes) if sizes else 8

    def _select(self, exclude: frozenset[str]) -> TEIBackend | None:
        """Pick the least-loaded selectable backend not in `exclude`,
        preferring weight>0 backends; weight==0 backends serve as
        fallback-only. Increments the winner's inflight (synchronously, no
        await) so concurrently-dispatched chunks spread instead of
        dogpiling the same backend."""
        candidates = [
            b for b in self._backends.values()
            if b.selectable and b.id not in exclude
        ]
        if not candidates:
            return None
        weighted = [b for b in candidates if b.weight > 0]
        pool = weighted or candidates  # fall back to weight-0 backends
        backend = min(pool, key=lambda b: b.load)
        backend.inflight += 1
        return backend

    def _has_selectable(self, exclude: frozenset[str]) -> bool:
        return any(
            b.selectable and b.id not in exclude
            for b in self._backends.values()
        )

    # --- embedding -----------------------------------------------------

    async def embed(self, texts: list[str], *, truncate: bool = True) -> np.ndarray:
        """Embed `texts` across the pool; return shape (n, 128) fp16.

        Chunks of `≤ _chunk_size()` are dispatched concurrently, each routed
        independently. `gather` preserves input order."""
        if not texts:
            return np.empty((0, 0), dtype=np.float16)

        size = self._chunk_size()
        chunks = [texts[i : i + size] for i in range(0, len(texts), size)]
        chunk_arrays = await asyncio.gather(
            *(self._embed_chunk(c, truncate=truncate) for c in chunks)
        )
        return np.concatenate(chunk_arrays, axis=0)

    async def _embed_chunk(self, chunk: list[str], *, truncate: bool) -> np.ndarray:
        """One chunk, with failover across backends and backoff retry.

        On a transient failure we first try a *different* healthy backend
        (immediate failover); only when none is left do we back off and
        retry, clearing the exclude set so a recovered backend is reusable.
        Direct fp32→fp16 decode."""
        loop = asyncio.get_event_loop()
        started = loop.time()
        exclude: frozenset[str] = frozenset()
        last_exc: BaseException | None = None

        for attempt in range(self._policy.max_attempts):
            backend = self._select(exclude)
            if backend is None:
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError("no healthy TEI backend available")

            try:
                json_resp = await backend.post(chunk, truncate=truncate)
                backend.mark_success()
                return np.asarray(json_resp, dtype=np.float16)
            except BaseException as e:  # noqa: BLE001 — classify then re-raise
                if not _is_transient(e):
                    log.warning(
                        "TEI backend %s: non-transient error (%s) — not retrying",
                        backend.id, e,
                    )
                    raise
                backend.mark_failure()
                last_exc = e
                if attempt == self._policy.max_attempts - 1:
                    log.warning(
                        "TEI: exhausted %d attempts — %s",
                        self._policy.max_attempts, e,
                    )
                    raise

                exclude = exclude | {backend.id}
                if self._has_selectable(exclude):
                    # Another healthy backend is available — fail over now,
                    # no backoff.
                    log.info(
                        "TEI backend %s failed (%s) — failing over", backend.id, e,
                    )
                    continue

                # No alternative left: back off, then allow any backend again.
                wait = min(
                    self._policy.initial_backoff_s * (self._policy.multiplier ** attempt),
                    self._policy.max_single_backoff_s,
                )
                elapsed = loop.time() - started
                if elapsed + wait > self._policy.total_budget_s:
                    log.warning(
                        "TEI: total budget %.1fs exhausted at attempt %d — %s",
                        self._policy.total_budget_s, attempt + 1, e,
                    )
                    raise
                log.info(
                    "TEI attempt %d/%d failed (%s) — retrying in %.2fs",
                    attempt + 1, self._policy.max_attempts, e, wait,
                )
                await asyncio.sleep(wait)
                exclude = frozenset()

        raise last_exc or RuntimeError("unreachable")

    # --- background ----------------------------------------------------

    async def _probe_loop(self) -> None:
        while True:
            await asyncio.sleep(self._probe_interval_s)
            backends = list(self._backends.values())
            if backends:
                await asyncio.gather(
                    *(b.probe() for b in backends), return_exceptions=True,
                )
                self._notify()

    @property
    def semaphore_wait_total_s(self) -> float:
        return sum(b._sem_wait_total_s for b in self._backends.values())
