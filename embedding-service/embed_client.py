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

Recovery (MXG-159). An unhealthy backend gets three independent ways back:

1. The probe loop, which runs on the backend's *serving* client so a
   success means "we can actually reach it", not just "it is up". Each
   probe round is bounded by `wait_for`, and the loop body catches
   everything, so a hung probe can no longer freeze recovery forever.
2. Client recycling: once a backend has been unhealthy for
   `client_recycle_after_s` the whole `httpx.AsyncClient` is replaced. A
   connection pool can be poisoned independently of the backend (a
   request cancelled by the caller's budget can leave httpcore's
   bookkeeping wedged), in which case the backend is fine and only a
   fresh client recovers it. This automates what an operator otherwise
   does by hand with `POST /admin/backends`.
3. Half-open admission: when no backend is selectable, one real chunk is
   let through to the stalest unhealthy backend, rate-limited to one
   trial in flight per `half_open_interval_s`. With a single-backend pool
   "no healthy backend" and "service down" are the same event, so failing
   fast forever is indistinguishable from being down.
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
_INPUT_REJECTION_HTTP_STATUSES = frozenset({400, 413, 422})

# Consecutive failures (real requests or health probes) before a backend is
# marked unhealthy and skipped in selection; one success restores it.
_UNHEALTHY_AFTER = 2


@dataclass(frozen=True)
class HealthPolicy:
    """Everything about how a backend leaves and re-enters the pool."""

    unhealthy_after: int = _UNHEALTHY_AFTER
    probe_interval_s: float = 5.0
    # Hard ceiling on one probe and on a whole probe round. httpx timeouts
    # do not cover every way a poisoned connection pool can block, so the
    # loop enforces its own.
    probe_timeout_s: float = 2.0
    probe_round_timeout_s: float = 10.0
    # Minimum spacing between half-open trial requests, per backend.
    half_open_interval_s: float = 5.0
    # How long a backend stays unhealthy before its client is replaced.
    client_recycle_after_s: float = 60.0
    # A pool timeout is not evidence about the backend, it is evidence about
    # us: every connection slot is held by a request that will never finish.
    # Waiting out the full recycle timer for that is waiting for information
    # we already have, so recycle after this many consecutive pool timeouts.
    pool_timeout_recycle_after: int = 3


class NoHealthyBackendError(RuntimeError):
    """No backend was selectable and no half-open trial was available.

    Subclasses RuntimeError so existing callers that catch RuntimeError
    (and the `/readyz` canary) behave exactly as before; `main.py` catches
    the specific type to answer 503 instead of a 500 with a traceback.
    """


class TEIInputError(RuntimeError):
    """TEI rejected request data rather than reporting a backend fault.

    The public exception deliberately carries no upstream response body.
    Article text can appear there, and callers need only the status to
    distinguish permanent input failures from retryable outages.
    """

    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"TEI rejected the embedding input (HTTP {status_code})")


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


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
        health: HealthPolicy | None = None,
    ) -> None:
        self.id = backend_id
        self.base_url = base_url.rstrip("/")
        self.weight = weight
        self.max_client_batch = max_client_batch
        self.max_concurrency = max_concurrency
        self.timeout_s = timeout_s
        self.health = health or HealthPolicy()

        self.inflight = 0            # assigned-or-running chunks (for load balancing)
        self.healthy = False         # unhealthy until the first probe succeeds
        self.draining = False        # set on remove → excluded from selection
        self.consecutive_failures = 0
        self.consecutive_pool_timeouts = 0
        self.unhealthy_since: float | None = None   # monotonic, None while healthy
        self.last_probe_error: str | None = None
        # Half-open bookkeeping: at most one trial request in flight per
        # backend, at most one per `half_open_interval_s`.
        self.trial_inflight = False
        self.last_trial_at = float("-inf")
        # Bumped on every client replacement, so operators and dashboards
        # can see that recycling happened at all.
        self.client_generation = 0
        self._recycled_at = float("-inf")

        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = self._new_client()
        self._sem_wait_total_s = 0.0
        self._closing: set[asyncio.Task] = set()

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_s,
            limits=httpx.Limits(
                max_connections=self.max_concurrency * 2,
                max_keepalive_connections=self.max_concurrency,
            ),
        )

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

    def mark_success(self, *, source: str = "request") -> None:
        self.consecutive_failures = 0
        self.consecutive_pool_timeouts = 0
        self.last_probe_error = None
        if not self.healthy:
            if self.unhealthy_since is None:
                # First confirmation after being added: not an incident.
                log.info(
                    "TEI backend %s (%s) healthy (via %s)",
                    self.id, self.base_url, source,
                )
            else:
                # WARNING, not INFO: nothing configures the root logger under
                # uvicorn, so INFO from this module never reaches the log. A
                # restore line nobody can see is how a four-hour outage gets
                # diagnosed from the wrong end.
                log.warning(
                    "TEI backend %s (%s) restored after %s unhealthy (via %s)",
                    self.id, self.base_url,
                    _fmt_duration(self.unhealthy_for()), source,
                )
        self.healthy = True
        self.unhealthy_since = None

    def mark_failure(self, exc: BaseException | None = None, *, source: str = "request") -> None:
        self.consecutive_failures += 1
        if isinstance(exc, httpx.PoolTimeout):
            self.consecutive_pool_timeouts += 1
        else:
            self.consecutive_pool_timeouts = 0
        if source == "probe":
            self.last_probe_error = (
                f"{type(exc).__name__}: {exc}" if exc is not None else "unknown"
            )
        if self.consecutive_failures >= self.health.unhealthy_after:
            if self.healthy:
                log.warning(
                    "TEI backend %s (%s) marked unhealthy after %d failures (%s: %s)",
                    self.id, self.base_url, self.consecutive_failures, source,
                    type(exc).__name__ if exc is not None else "unknown",
                )
            self.healthy = False
            # Also stamped for a backend that was never healthy in the first
            # place: "added and never answered" needs the same recovery as
            # "was fine and broke".
            if self.unhealthy_since is None:
                self.unhealthy_since = time.monotonic()

    def unhealthy_for(self) -> float:
        """Seconds since this backend went unhealthy, 0.0 while healthy."""
        if self.unhealthy_since is None:
            return 0.0
        return time.monotonic() - self.unhealthy_since

    # --- half-open -------------------------------------------------------

    def trial_due(self, now: float, interval_s: float) -> bool:
        return not self.trial_inflight and (now - self.last_trial_at) >= interval_s

    # --- client lifecycle ------------------------------------------------

    def recycle_due(self, now: float) -> bool:
        """A backend can be unreachable because *our* client is wedged, not
        because it is down. When probes have been failing for long enough,
        replace the client and let the next probe tell us which it was.

        Pool timeouts short-circuit that wait: they say the connection pool
        is full of requests that will never finish, which is a statement
        about this client and not about the backend. Measured on the box
        (MXG-159): with only the timer, a paused-then-resumed TEI took
        another 35s to come back; the pool-timeout path cuts that to the
        probe interval times the threshold."""
        if self.healthy or self.draining or self.unhealthy_since is None:
            return False
        since_recycle = now - self._recycled_at
        if (
            self.consecutive_pool_timeouts >= self.health.pool_timeout_recycle_after
            and since_recycle >= self.health.probe_interval_s * 3
        ):
            return True
        after = self.health.client_recycle_after_s
        return self.unhealthy_for() >= after and since_recycle >= after

    def recycle_client(self) -> None:
        """Swap in a fresh client; close the old one in the background.

        Closing is deliberately not awaited: if the pool is wedged, the
        close can block as thoroughly as the requests did. In-flight
        requests on the old client are already doomed, and the caller
        retries them."""
        old = self._client
        self._client = self._new_client()
        self.client_generation += 1
        self._recycled_at = time.monotonic()
        log.warning(
            "TEI backend %s (%s) client recycled (generation %d) after %s unhealthy",
            self.id, self.base_url, self.client_generation,
            _fmt_duration(self.unhealthy_for()),
        )
        task = asyncio.create_task(self._close_quietly(old))
        # Keep a strong ref, otherwise the task can be collected mid-flight.
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    @staticmethod
    async def _close_quietly(client: httpx.AsyncClient) -> None:
        try:
            await asyncio.wait_for(client.aclose(), timeout=5.0)
        except Exception as e:  # noqa: BLE001 — a stuck close must not propagate
            log.debug("closing recycled TEI client failed: %s", e)

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
        counter real requests use. Deliberately runs on the *serving*
        client: a probe on a private connection would answer "TEI is up"
        while the path we actually serve on stays wedged, and would flap
        the backend healthy once per interval. Short timeout — a slow
        health check shouldn't itself look like an outage."""
        try:
            resp = await self._client.get("/health", timeout=self.health.probe_timeout_s)
            resp.raise_for_status()
            self.mark_success(source="probe")
        except Exception as e:  # noqa: BLE001 — any failure = not healthy
            self.mark_failure(e, source="probe")
            log.debug("TEI backend %s probe failed: %s", self.id, e)

    async def aclose(self) -> None:
        for task in list(self._closing):
            task.cancel()
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
            # Why a backend is out, and whether recovery is being attempted.
            "consecutive_failures": self.consecutive_failures,
            "consecutive_pool_timeouts": self.consecutive_pool_timeouts,
            "unhealthy_for_s": round(self.unhealthy_for(), 1),
            "last_probe_error": self.last_probe_error,
            "client_generation": self.client_generation,
        }


class TEIPool:
    """Runtime-mutable set of `TEIBackend`s. Lives on `app.state.tei` in
    place of the old single client; `embed()` has the same signature so
    `main.py`'s handler is unchanged."""

    def __init__(
        self,
        *,
        retry_policy: RetryPolicy | None = None,
        health: HealthPolicy | None = None,
        drain_timeout_s: float = 30.0,
        on_change=None,
    ) -> None:
        self._backends: dict[str, TEIBackend] = {}
        self._ids = itertools.count(1)
        self._policy = retry_policy or RetryPolicy()
        self._health = health or HealthPolicy()
        self._drain_timeout_s = drain_timeout_s
        self._on_change = on_change  # callback(list[dict], dict) for metrics refresh
        self._mutate = asyncio.Lock()
        self._probe_task: asyncio.Task | None = None
        self._pending: set[asyncio.Task] = set()
        # Epoch seconds of the last completed probe round. Exported so a
        # frozen loop is visible as a stale timestamp: the health gauge is
        # push-updated, so a dead loop leaves it looking fine forever.
        self.probe_loop_last_iteration_at = 0.0
        self.probe_loop_errors = 0

    # --- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._probe_task is None:
            # Seed the liveness stamp so "loop never started" and "loop
            # started and is behind" are not the same reading.
            self.probe_loop_last_iteration_at = time.time()
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
                health=self._health,
            )
            self._backends[backend_id] = backend
        log.info("added TEI backend %s (%s) weight=%s", backend_id, url, weight)
        # Confirm health promptly. Keep a strong ref: a bare create_task can
        # be garbage-collected before it runs.
        task = asyncio.create_task(backend.probe())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
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

    def stats(self) -> dict:
        return {
            "probe_loop_last_iteration_at": self.probe_loop_last_iteration_at,
            "probe_loop_errors": self.probe_loop_errors,
        }

    def _notify(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change(self.list_backends(), self.stats())
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

    def _select_half_open(self, exclude: frozenset[str]) -> TEIBackend | None:
        """Pick an unhealthy backend to send one *real* chunk to.

        This is the only recovery path that proves the serving path works,
        so it is what gets the service back when the probe path and the
        request path disagree. Rate-limited to one trial in flight per
        backend and one per `half_open_interval_s`, so a hard-down backend
        costs one request per interval, not one per caller."""
        now = time.monotonic()
        candidates = [
            b for b in self._backends.values()
            if not b.draining
            and b.id not in exclude
            and b.trial_due(now, self._health.half_open_interval_s)
        ]
        if not candidates:
            return None
        backend = min(candidates, key=lambda b: b.last_trial_at)
        backend.last_trial_at = now
        backend.trial_inflight = True
        backend.inflight += 1
        log.warning(
            "TEI backend %s (%s) half-open trial after %s unhealthy",
            backend.id, backend.base_url, _fmt_duration(backend.unhealthy_for()),
        )
        return backend

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
            trial = False
            if backend is None:
                # Nothing selectable: try to claim a half-open trial before
                # giving up, so a recovered backend is found by traffic and
                # not only by the probe loop.
                backend = self._select_half_open(exclude)
                trial = backend is not None
            if backend is None:
                if last_exc is not None:
                    raise last_exc
                raise NoHealthyBackendError("no healthy TEI backend available")

            try:
                try:
                    json_resp = await backend.post(chunk, truncate=truncate)
                    backend.mark_success(source="trial" if trial else "request")
                    return np.asarray(json_resp, dtype=np.float16)
                except BaseException as e:  # noqa: BLE001 — classify then re-raise
                    if (
                        isinstance(e, httpx.HTTPStatusError)
                        and e.response.status_code in _INPUT_REJECTION_HTTP_STATUSES
                    ):
                        status = e.response.status_code
                        # Reaching TEI's input validator proves the serving
                        # connection works. Break any prior backend-failure
                        # streak, then report the caller's permanent error.
                        backend.mark_success(source="trial" if trial else "request")
                        log.warning(
                            "TEI backend %s rejected embedding input with HTTP %d",
                            backend.id, status,
                        )
                        raise TEIInputError(status) from e
                    if not _is_transient(e):
                        log.warning(
                            "TEI backend %s: non-transient error (%s) — not retrying",
                            backend.id, e,
                        )
                        raise
                    backend.mark_failure(e, source="trial" if trial else "request")
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
            finally:
                # Release the half-open slot on every exit path, including
                # `continue` and the re-raises above.
                if trial:
                    backend.trial_inflight = False

        raise last_exc or RuntimeError("unreachable")

    # --- background ----------------------------------------------------

    async def _probe_loop(self) -> None:
        """Probe every backend on an interval, forever.

        Every step is defensive on purpose (MXG-159): this loop is the only
        thing that brings a backend back when there is no traffic to run a
        half-open trial, and when it stopped, health froze at "unhealthy"
        for four hours across a TEI restart. It must not be stoppable by a
        hung probe (hence the round timeout) or by any exception (hence the
        catch-all), and its liveness must be observable (hence the
        timestamp)."""
        while True:
            try:
                await asyncio.sleep(self._health.probe_interval_s)
                backends = list(self._backends.values())
                if backends:
                    now = time.monotonic()
                    for b in backends:
                        if b.recycle_due(now):
                            b.recycle_client()
                    await self._probe_round(backends)
                self.probe_loop_last_iteration_at = time.time()
                if backends:
                    self._notify()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — the loop outlives every bug in it
                self.probe_loop_errors += 1
                self.probe_loop_last_iteration_at = time.time()
                log.exception("TEI probe round failed — continuing")

    async def _probe_round(self, backends: list[TEIBackend]) -> None:
        """One probe per backend, each bounded by the round budget.

        A probe that does not answer is a failure, not a reason to wait:
        httpx's own timeout did not save us in production (a wedged
        connection pool can block before any of it applies), so the round
        enforces the bound itself and charges the silence to the backend."""
        tasks = {b.id: asyncio.create_task(b.probe()) for b in backends}
        await asyncio.wait(
            tasks.values(), timeout=self._health.probe_round_timeout_s,
        )
        for backend in backends:
            task = tasks[backend.id]
            if task.done():
                continue
            task.cancel()
            self.probe_loop_errors += 1
            backend.mark_failure(
                asyncio.TimeoutError(
                    f"probe exceeded {self._health.probe_round_timeout_s:.1f}s"
                ),
                source="probe",
            )

    @property
    def semaphore_wait_total_s(self) -> float:
        return sum(b._sem_wait_total_s for b in self._backends.values())
