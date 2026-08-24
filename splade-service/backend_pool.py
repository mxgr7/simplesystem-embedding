import asyncio
import itertools
import logging
import time

import httpx

from constants import ENCODING_VERSION, model_metadata


log = logging.getLogger(__name__)
EXPECTED_METADATA = model_metadata()
TRANSIENT = {408, 429, 500, 502, 503, 504}

# `model_metadata()` pins WHICH CHECKPOINT a backend serves. It says nothing about
# HOW that checkpoint is executed, so two backends can pass it while producing
# different vectors -- an H100 on bf16 and a T4 on fp16 differ here and agree on
# every key above. The reindex client already pins these across backends
# (`validate_backends`); the serving pool did not, which is how a burst backend
# could land in the pool and write its vectors into the shared cache keyspace.
ENCODER_CONTRACT = ("document_compute_dtype", "document_encoding_version",
                    "fold_vocab_mask", "vocab_mask_sha256")


class HealthPolicy:
    """Everything about how a backend leaves and re-enters the pool.

    Names and defaults deliberately match `embedding-service/embed_client.py`'s
    `HealthPolicy`, so the dense and sparse stacks read and alert the same way.
    """

    def __init__(
        self,
        unhealthy_after=2,
        probe_interval_s=5.0,
        probe_timeout_s=2.0,
        # Hard ceiling on one probe. httpx timeouts do not cover every way a
        # poisoned connection pool can block, so the loop enforces its own.
        probe_round_timeout_s=10.0,
        # Run one real query encode every N probe rounds. Metadata proves
        # identity, while this canary proves the shared compute path still works.
        compute_probe_every=12,
        # Minimum spacing between half-open trial requests, per backend.
        half_open_interval_s=5.0,
        # How long a backend stays unhealthy before its client is replaced.
        client_recycle_after_s=60.0,
        # A pool timeout is not evidence about the backend, it is evidence about
        # us: every connection slot is held by a request that will never finish.
        # Waiting out the full recycle timer for that is waiting for information
        # we already have, so recycle after this many consecutive pool timeouts.
        pool_timeout_recycle_after=3,
    ):
        self.unhealthy_after = unhealthy_after
        self.probe_interval_s = probe_interval_s
        self.probe_timeout_s = probe_timeout_s
        self.probe_round_timeout_s = probe_round_timeout_s
        if compute_probe_every < 1:
            raise ValueError("compute_probe_every must be at least 1")
        self.compute_probe_every = compute_probe_every
        self.half_open_interval_s = half_open_interval_s
        self.client_recycle_after_s = client_recycle_after_s
        self.pool_timeout_recycle_after = pool_timeout_recycle_after


class RetryPolicy:
    """How one chunk retries across backends.

    Was `range(5)` and `min(0.25 * (2 ** attempt), 2)` inline. Named because the
    numbers matter -- five attempts against a single-backend pool is 5.75s of
    backoff before the caller sees anything -- and because a test that has to
    sleep through the real curve to assert on recovery is a test nobody runs.
    """

    def __init__(self, max_attempts=5, initial_backoff_s=0.25, max_backoff_s=2.0):
        self.max_attempts = max_attempts
        self.initial_backoff_s = initial_backoff_s
        self.max_backoff_s = max_backoff_s

    def backoff(self, attempt):
        return min(self.initial_backoff_s * (2 ** attempt), self.max_backoff_s)


class NoHealthyBackendError(RuntimeError):
    """No backend was selectable and no half-open trial was available.

    Subclasses RuntimeError so callers that already catch RuntimeError behave
    exactly as before; `main.py` catches the specific type to answer 503 instead
    of a 500 with a traceback per request.
    """


def _fmt_duration(seconds):
    if seconds < 60:
        return "%.1fs" % seconds
    if seconds < 3600:
        return "%.1fm" % (seconds / 60)
    return "%.1fh" % (seconds / 3600)


class Backend:
    def __init__(
        self,
        backend_id,
        url,
        weight,
        max_concurrency,
        max_client_batch,
        timeout_s,
        api_key,
        health=None,
    ):
        self.id = backend_id
        self.url = url.rstrip("/")
        self.weight = weight
        self.max_client_batch = max_client_batch
        self.max_concurrency = max_concurrency
        self.timeout_s = timeout_s
        self.api_key = api_key
        self.health = health or HealthPolicy()
        self.inflight = 0
        self.healthy = False
        self.draining = False
        self.consecutive_failures = 0
        self.consecutive_pool_timeouts = 0
        # Monotonic, None while healthy. Drives both the recycle timer and the
        # "restored after X" line.
        self.unhealthy_since = None
        self.last_probe_error = None
        # Monotonic count of real healthy/unhealthy transitions. The initial
        # registration confirmation is not an incident and does not count.
        self.health_transitions = 0
        # A failed compute canary stays latched. Metadata-only probes must not
        # promote a backend that can identify itself but cannot encode.
        self.compute_probe_failed = False
        # Half-open bookkeeping: at most one trial in flight per backend, at most
        # one per `half_open_interval_s`.
        self.trial_inflight = False
        self.last_trial_at = float("-inf")
        # Bumped on every client replacement, so operators and dashboards can see
        # that recycling happened at all.
        self.client_generation = 0
        self._recycled_at = float("-inf")
        self._closing = set()
        # Set by the pool on registration: `verify()` re-checks the encoder
        # contract on every promotion, not only at add() time.
        self.contract_check = None
        # Last payload seen by verify(). Kept rather than discarded so the frontend can answer
        # "which checkpoint is actually being served" without a second round trip: a client that
        # only ever calls /embed has no other way to tell, and the encoder-identity fields
        # (document_encoding_version, fold_vocab_mask, vocab_mask_sha256) exist only here.
        self.metadata = {}
        # Document calls can fill their configured connections without trapping
        # a live query behind them. One separate query connection reaches the
        # backend's priority queue; additional queries wait here.
        self.document_sem = asyncio.Semaphore(max_concurrency)
        self.query_sem = asyncio.Semaphore(1)
        self.client = self._new_client()

    def _new_client(self):
        """The single place a client is built, so recycling and tests both go
        through it."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return httpx.AsyncClient(
            base_url=self.url,
            headers=headers,
            timeout=self.timeout_s,
            limits=httpx.Limits(
                max_connections=self.max_concurrency * 2,
                max_keepalive_connections=self.max_concurrency,
            ),
        )

    # --- health state machine ---------------------------------------------

    def mark_success(self, source="request"):
        """Record a success. Only a source that ran `verify()` promotes.

        This is the one place the sparse pool deliberately differs from the dense
        one. Over there any successful request restores a backend; here the
        checkpoint contract is the whole point of `verify()`, so a backend that
        came back on a different dtype behind the same URL must not be readmitted
        by traffic alone. A plain request success still clears the counters --
        it is evidence the backend is answering, just not evidence about WHICH
        checkpoint is answering.
        """
        self.consecutive_failures = 0
        self.consecutive_pool_timeouts = 0
        if source == "request":
            return
        self.last_probe_error = None
        if not self.healthy:
            if self.unhealthy_since is None:
                # First confirmation after being added: not an incident.
                log.info("backend %s (%s) healthy (via %s)", self.id, self.url, source)
            else:
                self.health_transitions += 1
                # WARNING, not INFO: a restore line nobody reads is how a
                # four-hour outage gets diagnosed from the wrong end.
                log.warning(
                    "backend %s (%s) restored after %s unhealthy (via %s)",
                    self.id, self.url, _fmt_duration(self.unhealthy_for()), source,
                )
        self.healthy = True
        self.unhealthy_since = None

    def mark_failure(self, exc=None, source="request", immediate=False):
        self.consecutive_failures += 1
        if immediate:
            self.consecutive_failures = max(
                self.consecutive_failures, self.health.unhealthy_after
            )
        if isinstance(exc, httpx.PoolTimeout):
            self.consecutive_pool_timeouts += 1
        else:
            self.consecutive_pool_timeouts = 0
        if source != "request":
            self.last_probe_error = (
                f"{type(exc).__name__}: {exc}" if exc is not None else "unknown"
            )
        if self.consecutive_failures >= self.health.unhealthy_after:
            if self.healthy:
                self.health_transitions += 1
                log.warning(
                    "backend %s (%s) marked unhealthy after %d failures (%s: %s)",
                    self.id, self.url, self.consecutive_failures, source,
                    type(exc).__name__ if exc is not None else "unknown",
                )
            self.healthy = False
            # Also stamped for a backend that was never healthy in the first
            # place: "added and never answered" needs the same recovery as
            # "was fine and broke".
            if self.unhealthy_since is None:
                self.unhealthy_since = time.monotonic()

    def unhealthy_for(self):
        """Seconds since this backend went unhealthy, 0.0 while healthy."""
        if self.unhealthy_since is None:
            return 0.0
        return time.monotonic() - self.unhealthy_since

    # --- half-open --------------------------------------------------------

    def trial_due(self, now, interval_s):
        return not self.trial_inflight and (now - self.last_trial_at) >= interval_s

    # --- client lifecycle -------------------------------------------------

    def recycle_due(self, now):
        """A backend can be unreachable because *our* client is wedged, not
        because it is down. When probes have been failing for long enough,
        replace the client and let the next probe tell us which it was.

        Pool timeouts short-circuit that wait: they say the connection pool is
        full of requests that will never finish, which is a statement about this
        client and not about the backend. `REQUEST_BUDGET_S=120` here rather than
        the dense side's 5 makes the cancellations that poison a pool rarer, not
        impossible -- any sustained backend slowdown produces the same ones.
        """
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

    def recycle_client(self):
        """Swap in a fresh client; close the old one in the background.

        Closing is deliberately not awaited: if the pool is wedged, the close can
        block as thoroughly as the requests did. In-flight requests on the old
        client are already doomed, and the caller retries them.
        """
        old = self.client
        self.client = self._new_client()
        self.client_generation += 1
        self._recycled_at = time.monotonic()
        log.warning(
            "backend %s (%s) client recycled (generation %d) after %s unhealthy",
            self.id, self.url, self.client_generation,
            _fmt_duration(self.unhealthy_for()),
        )
        task = asyncio.create_task(self._close_quietly(old))
        # Keep a strong ref, otherwise the task can be collected mid-flight.
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    @staticmethod
    async def _close_quietly(client):
        try:
            await asyncio.wait_for(client.aclose(), timeout=5)
        except Exception as exc:
            log.debug("closing recycled client failed: %s", exc)

    # --- IO ---------------------------------------------------------------

    async def verify(self):
        """Confirm this backend serves the checkpoint we think it does.

        Runs on the *serving* client, not a private one: a probe on its own
        connection would answer "the backend is up" while the path we actually
        serve on stays wedged, and would flap the backend healthy once per
        interval. Does not set `healthy` -- `mark_success` owns that, so every
        promotion goes through one place.
        """
        response = await self.client.get(
            "/metadata", timeout=self.health.probe_timeout_s
        )
        response.raise_for_status()
        metadata = response.json()
        mismatches = {
            key: (EXPECTED_METADATA[key], metadata.get(key))
            for key in EXPECTED_METADATA
            if metadata.get(key) != EXPECTED_METADATA[key]
        }
        if mismatches:
            raise ValueError(f"backend model contract mismatch: {mismatches}")
        previous = self.metadata
        self.metadata = metadata
        if self.contract_check is not None:
            try:
                self.contract_check(self)
            except Exception:
                self.metadata = previous
                raise

    async def probe(self, check_compute=False):
        try:
            await self.verify()
        except Exception as exc:
            self.mark_failure(exc, source="probe")
            log.debug("backend %s metadata probe failed: %s", self.id, exc)
            return

        if check_compute or self.compute_probe_failed:
            try:
                vectors = await self.encode(["splade health probe"], document=False)
                if not vectors[0]:
                    raise ValueError("compute probe returned an empty vector")
            except Exception as exc:
                self.compute_probe_failed = True
                # A health canary is already the confirmation. Demote on its
                # first failure rather than waiting for unrelated traffic.
                self.mark_failure(exc, source="compute-probe", immediate=True)
                log.debug("backend %s compute probe failed: %s", self.id, exc)
                return
            self.compute_probe_failed = False

        self.mark_success(source="probe")

    async def encode(self, texts, document=True):
        admission = self.document_sem if document else self.query_sem
        async with admission:
            response = await self.client.post(
                "/encode", json={"inputs": texts, "document": document}
            )
            response.raise_for_status()
            vectors = response.json()
            if len(vectors) != len(texts) or not all(
                isinstance(vector, dict) for vector in vectors
            ):
                raise ValueError("backend returned an invalid vector batch")
            return vectors

    async def aclose(self):
        for task in list(self._closing):
            task.cancel()
        await self.client.aclose()

    def snapshot(self):
        return {
            "id": self.id,
            "url": self.url,
            "weight": self.weight,
            "healthy": self.healthy,
            "draining": self.draining,
            "inflight": self.inflight,
            "max_client_batch": self.max_client_batch,
            # Why a backend is out, and whether recovery is being attempted.
            "consecutive_failures": self.consecutive_failures,
            "consecutive_pool_timeouts": self.consecutive_pool_timeouts,
            "unhealthy_for_s": round(self.unhealthy_for(), 1),
            "last_probe_error": self.last_probe_error,
            "client_generation": self.client_generation,
            "health_transitions": self.health_transitions,
        }


class BackendPool:
    def __init__(self, health=None, retry=None):
        self.backends = {}
        self.ids = itertools.count(1)
        self.health = health or HealthPolicy()
        self.retry = retry or RetryPolicy()
        self.probe_task = None
        self.lock = asyncio.Lock()
        # Epoch seconds of the last completed probe round, and how many rounds
        # timed out or raised. Exported so a frozen loop is visible: `healthy` is
        # only ever written BY this loop, so a dead loop leaves it looking fine
        # forever and the outage reads as steady state.
        self.probe_loop_last_iteration_at = 0.0
        self.probe_loop_errors = 0
        self.probe_rounds = 0

    async def add(
        self,
        url,
        weight=1,
        max_concurrency=1,
        max_client_batch=8,
        timeout_s=120,
        api_key="",
        require_verify=True,
    ):
        """Register a backend.

        `require_verify=False` is for startup: a backend that is down or still
        loading its checkpoint should not take the frontend down with it, it
        should be registered unhealthy and left to the probe loop. The contract
        check then happens on the first successful verify instead of here.
        `POST /admin/backends` keeps `require_verify=True`, so a hand-added bad
        backend still fails loudly at the point of the request.
        """
        backend = Backend(
            f"b{next(self.ids)}",
            url,
            weight,
            max_concurrency,
            max_client_batch,
            timeout_s,
            api_key,
            self.health,
        )
        backend.contract_check = self._check_encoder_contract
        if require_verify:
            try:
                await backend.verify()
            except Exception:
                await backend.aclose()
                raise
            backend.mark_success(source="probe")
            async with self.lock:
                self.backends[backend.id] = backend
            return backend.snapshot()

        async with self.lock:
            self.backends[backend.id] = backend
        try:
            await backend.verify()
            backend.mark_success(source="probe")
        except Exception as exc:
            backend.mark_failure(exc, source="probe")
            log.warning(
                "backend %s (%s) registered unhealthy: %s: %s",
                backend.id, backend.url, type(exc).__name__, exc,
            )
        return backend.snapshot()

    def _check_encoder_contract(self, backend):
        """Reject a backend that executes the pinned checkpoint differently.

        Two places can define the expected contract, and both are checked because
        they fail in different ways. `SPLADE_ENCODING_VERSION` is the operator's
        declaration and also what the cache keyspace is namespaced by, so a
        mismatch there means cached vectors would be filed under a name that does
        not describe them. An already-registered backend is the empirical one: the
        first backend in defines the contract for the rest, exactly as
        `validate_backends` does on the reindex side.

        Called from `verify()`, so it runs on every promotion and not only at
        registration: a backend restarted onto a different dtype behind a stable
        URL used to be readmitted silently.
        """
        got = {key: backend.metadata.get(key) for key in ENCODER_CONTRACT}
        declared = ENCODING_VERSION
        if declared and got["document_encoding_version"] != declared:
            raise ValueError(
                f"backend {backend.url} encodes as "
                f"{got['document_encoding_version']!r} but SPLADE_ENCODING_VERSION "
                f"declares {declared!r}; the cache keyspace is namespaced by the "
                "declared value, so this would file its vectors under the wrong name"
            )
        for other in self.backends.values():
            # Skip itself (the backend is already registered when verify runs on
            # the startup path) and anything that has never verified, whose empty
            # metadata would read as a contract of all-None.
            if other is backend or not other.metadata:
                continue
            expected = {key: other.metadata.get(key) for key in ENCODER_CONTRACT}
            if got != expected:
                differing = {k: (expected[k], got[k]) for k in got if got[k] != expected[k]}
                raise ValueError(
                    f"backend {backend.url} encoder contract differs from "
                    f"{other.url}: {differing}"
                )
            break

    def start(self):
        if self.probe_task is None:
            # Seed the stamp so "loop never started" and "loop started and is
            # behind" are not the same reading.
            self.probe_loop_last_iteration_at = time.time()
            self.probe_task = asyncio.create_task(self._probe_loop())

    async def aclose(self):
        if self.probe_task:
            self.probe_task.cancel()
            try:
                await self.probe_task
            except asyncio.CancelledError:
                pass
        await asyncio.gather(
            *(backend.aclose() for backend in self.backends.values()),
            return_exceptions=True,
        )

    def snapshots(self):
        return [backend.snapshot() for backend in self.backends.values()]

    def stats(self):
        return {
            "probe_loop_last_iteration_at": self.probe_loop_last_iteration_at,
            "probe_loop_errors": self.probe_loop_errors,
        }

    def contracts(self):
        """Encoder identity per backend, for the frontend's GET /metadata.

        Deliberately narrow: these are the keys that decide whether a vector produced now is
        comparable with one already in the index. `model_id`/`model_sha256` name the checkpoint;
        `document_encoding_version` names the encoder build (dtype, codec, compiled head);
        `fold_vocab_mask` plus `vocab_mask_sha256` pin the exact kept-dimension set, which two
        booleans could not. Tuning knobs (batch size, overlap, device) are left out on purpose --
        a client asserting on those would break on a harmless redeploy.

        Backends that have never verified report an empty contract rather than being omitted, so a
        half-up pool is visible rather than silently looking like a healthy smaller one.
        """
        keys = (
            "model_id",
            "model_sha256",
            "document_encoding_version",
            "fold_vocab_mask",
            "vocab_mask_sha256",
        )
        return [
            {
                "id": backend.id,
                "healthy": backend.healthy,
                "draining": backend.draining,
                **{key: backend.metadata.get(key) for key in keys},
            }
            for backend in self.backends.values()
        ]

    def ready(self):
        return any(
            backend.healthy and not backend.draining
            for backend in self.backends.values()
        )

    async def set_weight(self, backend_id, weight):
        backend = self.backends.get(backend_id)
        if not backend:
            raise KeyError(backend_id)
        backend.weight = weight
        return backend.snapshot()

    async def remove(self, backend_id):
        backend = self.backends.get(backend_id)
        if not backend:
            raise KeyError(backend_id)
        backend.draining = True
        asyncio.create_task(self._drain(backend))
        return backend.snapshot()

    async def _drain(self, backend):
        for _ in range(600):
            if backend.inflight == 0:
                break
            await asyncio.sleep(0.1)
        async with self.lock:
            self.backends.pop(backend.id, None)
        await backend.aclose()

    def _select(self, excluded):
        candidates = [
            backend
            for backend in self.backends.values()
            if backend.healthy
            and not backend.draining
            and backend.id not in excluded
        ]
        if not candidates:
            return None
        weighted = [backend for backend in candidates if backend.weight > 0]
        candidates = weighted or candidates
        backend = min(
            candidates,
            key=lambda item: item.inflight / max(item.weight, 0.000001),
        )
        backend.inflight += 1
        return backend

    def _select_half_open(self, excluded):
        """Pick an unhealthy backend to run one trial against.

        With `BACKEND_URLS` holding a single entry, failing fast forever and being
        down are the same event: there is no second backend for the request to
        fall over to, so without this the only way back is the probe loop. The
        trial re-verifies the checkpoint contract before it serves anything --
        see `mark_success` for why that is not optional here. Rate-limited to one
        trial in flight per backend and one per `half_open_interval_s`, so a
        hard-down backend costs one round trip per interval, not one per caller.
        """
        now = time.monotonic()
        candidates = [
            backend
            for backend in self.backends.values()
            if not backend.draining
            and backend.id not in excluded
            and backend.trial_due(now, self.health.half_open_interval_s)
        ]
        if not candidates:
            return None
        backend = min(candidates, key=lambda item: item.last_trial_at)
        backend.last_trial_at = now
        backend.trial_inflight = True
        backend.inflight += 1
        log.warning(
            "backend %s (%s) half-open trial after %s unhealthy",
            backend.id, backend.url, _fmt_duration(backend.unhealthy_for()),
        )
        return backend

    async def _encode_chunk(self, chunk, document=True):
        excluded = set()
        last_error = None
        for attempt in range(self.retry.max_attempts):
            backend = self._select(excluded)
            trial = False
            if backend is None:
                # Nothing selectable: try to claim a half-open trial before giving
                # up, so a recovered backend is found by traffic and not only by
                # the probe loop.
                backend = self._select_half_open(excluded)
                trial = backend is not None
            if backend is None:
                raise self._no_backend_error(last_error)
            try:
                if trial:
                    # verify() first: a trial that promoted on a bare /encode
                    # success would readmit a backend serving a different
                    # checkpoint, which is the one thing this pool exists to stop.
                    # Promotion waits until the encode succeeds too, so a latched
                    # compute failure cannot flap healthy between the two calls.
                    await backend.verify()
                vectors = await backend.encode(chunk, document=document)
                if trial:
                    backend.compute_probe_failed = False
                backend.mark_success(source="trial" if trial else "request")
                return vectors
            except Exception as exc:
                last_error = exc
                transient = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code in TRANSIENT
                )
                if not transient:
                    raise
                backend.mark_failure(exc, source="trial" if trial else "request")
                excluded.add(backend.id)
                if not any(
                    candidate.healthy
                    and not candidate.draining
                    and candidate.id not in excluded
                    for candidate in self.backends.values()
                ):
                    await asyncio.sleep(self.retry.backoff(attempt))
                    excluded.clear()
            finally:
                # `_select`/`_select_half_open` claim the slot; releasing it here
                # covers every exit path, including the re-raise above.
                backend.inflight -= 1
                if trial:
                    backend.trial_inflight = False
        if not self.ready():
            raise self._no_backend_error(last_error)
        raise last_error or RuntimeError("SPLADE backend retries exhausted")

    @staticmethod
    def _no_backend_error(last_error):
        """Typed even when a transport error is what got us here.

        With one entry in BACKEND_URLS this is THE outage path, and the caller
        needs a 503 telling it to retry, not a 500 carrying an
        httpx.HTTPStatusError that nothing counts and no alert watches. The cause
        is chained, not discarded.
        """
        message = "no healthy SPLADE backend available"
        if last_error is not None:
            # httpx's timeout exceptions stringify to "", so name the type: the
            # drill log read "available: " and said nothing at all.
            detail = str(last_error) or type(last_error).__name__
            message = f"{message}: {detail}"
        error = NoHealthyBackendError(message)
        error.__cause__ = last_error
        return error

    async def encode(self, texts, document=True):
        if not texts:
            return []
        sizes = [
            backend.max_client_batch
            for backend in self.backends.values()
            if not backend.draining
        ]
        chunk_size = min(sizes) if sizes else 8
        chunks = [
            texts[index:index + chunk_size]
            for index in range(0, len(texts), chunk_size)
        ]
        batches = await asyncio.gather(
            *(
                self._encode_chunk(chunk, document=document)
                for chunk in chunks
            )
        )
        return [vector for batch in batches for vector in batch]

    async def _probe_loop(self):
        """Probe every backend on an interval, forever.

        Every step is defensive on purpose. This loop is the only thing that
        brings a backend back when there is no traffic to run a half-open trial,
        and on the dense side the same loop stopped on a single probe that never
        answered: health froze at "unhealthy" for four hours, across a restart of
        the backend itself. It must not be stoppable by a hung probe (hence the
        per-probe bound) or by any exception (hence the catch-all), and its
        liveness must be observable (hence the timestamp).
        """
        while True:
            try:
                await asyncio.sleep(self.health.probe_interval_s)
                backends = list(self.backends.values())
                if backends:
                    now = time.monotonic()
                    for backend in backends:
                        if backend.recycle_due(now):
                            backend.recycle_client()
                    self.probe_rounds += 1
                    check_compute = (
                        self.probe_rounds % self.health.compute_probe_every == 0
                    )
                    await self._probe_round(backends, check_compute)
                self.probe_loop_last_iteration_at = time.time()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Stamped on this path too: the metric measures whether the loop
                # is alive, not whether it is happy.
                self.probe_loop_errors += 1
                self.probe_loop_last_iteration_at = time.time()
                log.exception("probe round failed -- continuing")

    async def _probe_round(self, backends, check_compute=False):
        await asyncio.gather(
            *(
                self._bounded_probe(backend, check_compute)
                for backend in backends
            ),
            return_exceptions=True,
        )

    async def _bounded_probe(self, backend, check_compute=False):
        """One probe, bounded outside the client.

        Per backend rather than one budget shared across the round: a shared
        budget lets a single slow backend spend the allowance of every other one
        probed with it. The bound lives here rather than in httpx because httpx's
        own timeout provably did not bound anything during the dense outage -- a
        wedged connection pool blocks before any of it applies -- so silence is
        charged to the backend as a failure.
        """
        try:
            await asyncio.wait_for(
                backend.probe(check_compute), self.health.probe_round_timeout_s
            )
        except asyncio.TimeoutError:
            self.probe_loop_errors += 1
            backend.mark_failure(
                asyncio.TimeoutError(
                    "probe exceeded %.1fs" % self.health.probe_round_timeout_s
                ),
                source="probe",
            )
