"""BackendPool recovery: the probe loop, client recycling and half-open admission.

MXG-166. `splade-service/backend_pool.py` was the same pool design as the dense
wrapper's `embedding-service/embed_client.py` and carried the same defects: an
unguarded, unbounded probe loop that one silent probe could stop for good, probes
on a connection pool that only a process restart could clear, and no way for
traffic to find a backend that had come back. On the dense side that combination
cost four hours with the backend itself healthy the whole time.

`httpx.MockTransport` deliberately ignores httpx's own timeouts, which is exactly
the property the production failure had -- a wedged pool blocks before any client
timeout applies. That is what makes `mode="hang"` a real test of the bound rather
than a test of httpx.
"""

import asyncio
import contextlib
import json
import logging
from pathlib import Path

import httpx
import pytest


from conftest import load_flat_service

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "splade-service"
splade = load_flat_service(
    "splade_service", SERVICE, "backend_pool", "constants"
)
backend_pool = splade.backend_pool
constants = splade.constants


ENCODER_FIELDS = {
    "document_compute_dtype": "fp16",
    "document_encoding_version": "prod-soup-top256-fp16-fp16codec-wcast-v2",
    "fold_vocab_mask": True,
    "vocab_mask_sha256": "b" * 64,
}


class StubBackend:
    """One SPLADE backend behind a MockTransport, with a switchable mode.

    `mode` is flipped mid-test to simulate a backend going away and coming back:
    "ok" serves, "fail" 503s, "hang" never answers. `metadata_overrides` stands
    in for a backend that answers perfectly well while serving the wrong
    checkpoint -- the case a cheap /health probe cannot see and this service's
    verify() exists to catch.
    """

    def __init__(self, mode="ok", metadata_overrides=None):
        self.mode = mode
        self.metadata_overrides = metadata_overrides or {}
        self.metadata_calls = 0
        self.encode_calls = 0

    def payload(self):
        values = dict(constants.model_metadata())
        values.update(ENCODER_FIELDS)
        values.update(self.metadata_overrides)
        return values

    async def __call__(self, request):
        is_metadata = request.url.path == "/metadata"
        if is_metadata:
            self.metadata_calls += 1
        else:
            self.encode_calls += 1
        if self.mode == "hang":
            await asyncio.sleep(3600)
        if self.mode == "fail":
            return httpx.Response(503, text="backend down")
        if is_metadata:
            return httpx.Response(200, json=self.payload())
        inputs = json.loads(request.content)["inputs"]
        return httpx.Response(200, json=[{"1": 0.5} for _ in inputs])


@contextlib.contextmanager
def stubbed(stub):
    """Route every client the pool builds, including recycled ones, at `stub`.

    Patching the factory rather than the instance is the whole point: a test that
    swaps `backend.client` proves nothing about recycling, because the recycled
    client is built fresh from `_new_client`.
    """
    original = backend_pool.Backend._new_client

    def _new_client(self):
        handler = stub[self.url] if isinstance(stub, dict) else stub
        return httpx.AsyncClient(
            base_url=self.url, transport=httpx.MockTransport(handler)
        )

    backend_pool.Backend._new_client = _new_client
    try:
        yield
    finally:
        backend_pool.Backend._new_client = original


def _health(**overrides):
    values = {
        "probe_interval_s": 0.02,
        "probe_timeout_s": 0.05,
        "probe_round_timeout_s": 0.2,
        "half_open_interval_s": 5.0,
        "client_recycle_after_s": 60.0,
    }
    values.update(overrides)
    return backend_pool.HealthPolicy(**values)


def _no_sleep_retry(**overrides):
    values = {"max_attempts": 2, "initial_backoff_s": 0.0, "max_backoff_s": 0.0}
    values.update(overrides)
    return backend_pool.RetryPolicy(**values)


async def _make_pool(health=None, require_verify=True, url="http://splade.stub",
                     retry=None):
    pool = backend_pool.BackendPool(health or _health(), retry or _no_sleep_retry())
    await pool.add(
        url,
        max_concurrency=2,
        max_client_batch=8,
        timeout_s=1,
        require_verify=require_verify,
    )
    return pool


def _only(pool):
    return next(iter(pool.backends.values()))


async def _wait_until(predicate, timeout_s=3.0):
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


def test_two_failures_trip_the_backend_unhealthy():
    stub = StubBackend()

    async def body():
        with stubbed(stub):
            pool = await _make_pool()
            backend = _only(pool)
            assert backend.healthy

            stub.mode = "fail"
            await backend.probe()
            assert backend.healthy, "one failure must not trip the backend"
            await backend.probe()
            assert not backend.healthy
            assert backend.consecutive_failures == 2
            assert backend.unhealthy_for() >= 0.0
            await pool.aclose()

    asyncio.run(body())


def test_probe_loop_restores_a_backend_without_traffic():
    stub = StubBackend(mode="fail")

    async def body():
        with stubbed(stub):
            pool = await _make_pool(require_verify=False)
            backend = _only(pool)
            pool.start()
            assert await _wait_until(lambda: not backend.healthy)

            stub.mode = "ok"
            assert await _wait_until(lambda: backend.healthy)
            assert stub.encode_calls == 0, "recovery must not need traffic"
            await pool.aclose()

    asyncio.run(body())


def test_a_hung_probe_does_not_freeze_the_loop():
    """The regression test for the outage this ticket exists to prevent.

    Before the bound, one probe that never answered left `_probe_loop` awaiting
    it forever: health froze at its last value, the scrape-time collector kept
    re-exporting that value, and the outage read as steady state.
    """
    stub = StubBackend()

    async def body():
        with stubbed(stub):
            pool = await _make_pool()
            backend = _only(pool)
            pool.start()

            stub.mode = "hang"
            assert await _wait_until(lambda: not backend.healthy), (
                "a hung probe must count as a failure, not as silence"
            )
            assert pool.probe_loop_errors >= 1

            stamp = pool.probe_loop_last_iteration_at
            assert await _wait_until(
                lambda: pool.probe_loop_last_iteration_at > stamp
            ), "loop stopped ticking while a probe was hung"

            stub.mode = "ok"
            assert await _wait_until(lambda: backend.healthy)
            await pool.aclose()

    asyncio.run(body())


def test_client_is_recycled_after_prolonged_unhealth():
    stub = StubBackend()

    async def body():
        with stubbed(stub):
            pool = await _make_pool(_health(client_recycle_after_s=0.05))
            backend = _only(pool)
            pool.start()

            stub.mode = "fail"
            assert await _wait_until(lambda: backend.client_generation >= 1)

            stub.mode = "ok"
            assert await _wait_until(lambda: backend.healthy)
            await pool.aclose()

    asyncio.run(body())


def test_a_healthy_backend_client_is_not_recycled():
    stub = StubBackend()

    async def body():
        with stubbed(stub):
            pool = await _make_pool(_health(client_recycle_after_s=0.05))
            backend = _only(pool)
            pool.start()
            await asyncio.sleep(0.3)
            assert backend.healthy
            assert backend.client_generation == 0
            await pool.aclose()

    asyncio.run(body())


def test_pool_timeouts_recycle_the_client_early():
    """A PoolTimeout is evidence about us, not about the backend.

    `client_recycle_after_s=600` is load-bearing: it proves the fast path fired
    rather than the timer. So is the negative half -- a ConnectError says the
    backend is unreachable, which a new client cannot fix.
    """
    stub = StubBackend()

    async def body():
        with stubbed(stub):
            pool = await _make_pool(
                _health(
                    client_recycle_after_s=600.0,
                    pool_timeout_recycle_after=3,
                    probe_interval_s=0.001,
                )
            )
            backend = _only(pool)

            for _ in range(3):
                backend.mark_failure(httpx.PoolTimeout(""), source="probe")
            assert not backend.healthy
            assert backend.consecutive_pool_timeouts == 3
            await asyncio.sleep(0.01)
            assert backend.recycle_due(asyncio.get_event_loop().time())

            backend.mark_success(source="probe")
            for _ in range(3):
                backend.mark_failure(httpx.ConnectError(""), source="probe")
            assert backend.consecutive_pool_timeouts == 0
            assert not backend.recycle_due(asyncio.get_event_loop().time())
            await pool.aclose()

    asyncio.run(body())


def test_half_open_trial_recovers_from_traffic_alone():
    stub = StubBackend()

    async def body():
        with stubbed(stub):
            pool = await _make_pool(_health(half_open_interval_s=0.0))
            backend = _only(pool)

            stub.mode = "fail"
            with contextlib.suppress(Exception):
                await pool.encode(["a"], document=False)
            assert not backend.healthy

            stub.mode = "ok"
            vectors = await pool.encode(["a"], document=False)
            assert vectors == [{"1": 0.5}]
            assert backend.healthy
            assert not backend.trial_inflight
            assert backend.inflight == 0
            await pool.aclose()

    asyncio.run(body())


def test_half_open_is_rate_limited():
    stub = StubBackend()

    async def body():
        with stubbed(stub):
            pool = await _make_pool(_health(half_open_interval_s=30.0))
            backend = _only(pool)
            backend.mark_failure()
            backend.mark_failure()
            assert not backend.healthy

            stub.mode = "fail"
            before = stub.metadata_calls
            for _ in range(3):
                with contextlib.suppress(Exception):
                    await pool.encode(["a"], document=False)
            assert stub.metadata_calls - before == 1
            await pool.aclose()

    asyncio.run(body())


def test_a_trial_whose_verify_fails_does_not_promote():
    """The SPLADE-specific gate.

    A backend restarted onto a different checkpoint behind the same URL answers
    /encode perfectly well. Promoting on that alone is how its vectors get filed
    under another model's cache key, which is the one thing this pool exists to
    stop -- so the trial re-verifies before it serves.
    """
    stub = StubBackend(metadata_overrides={"model_id": "some-other-checkpoint"})

    async def body():
        with stubbed(stub):
            pool = await _make_pool(
                _health(half_open_interval_s=0.0), require_verify=False
            )
            backend = _only(pool)
            assert not backend.healthy, "a mismatched backend must not register healthy"

            before = stub.encode_calls
            with pytest.raises(Exception):
                await pool.encode(["a"], document=False)
            assert not backend.healthy
            assert stub.encode_calls == before, "verify must gate the trial's traffic"
            assert "contract mismatch" in (backend.last_probe_error or "")
            await pool.aclose()

    asyncio.run(body())


def test_no_backend_raises_a_typed_error():
    stub = StubBackend()

    async def body():
        with stubbed(stub):
            pool = await _make_pool()
            backend = _only(pool)
            # Draining is neither selectable nor trial-eligible.
            backend.draining = True
            with pytest.raises(backend_pool.NoHealthyBackendError) as excinfo:
                await pool.encode(["a"], document=False)
            assert isinstance(excinfo.value, RuntimeError)
            await pool.aclose()

    asyncio.run(body())


def test_a_down_backend_is_a_typed_error_not_a_transport_error():
    """With one entry in BACKEND_URLS this is THE outage path.

    Raising the raw httpx error here is how a total encoder outage became a 500
    per request: uncounted, unalerted, and a traceback in the log each time. The
    cause is still chained onto the typed error.
    """
    stub = StubBackend(mode="fail")

    async def body():
        with stubbed(stub):
            pool = await _make_pool(
                _health(half_open_interval_s=0.0), require_verify=False
            )
            with pytest.raises(backend_pool.NoHealthyBackendError) as excinfo:
                await pool.encode(["a"], document=False)
            assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)
            await pool.aclose()

    asyncio.run(body())


def test_startup_registers_an_unreachable_backend_unhealthy():
    """A backend that is down at boot must not restart-loop the frontend."""
    stub = StubBackend(mode="fail")

    async def body():
        with stubbed(stub):
            pool = await _make_pool(require_verify=False)
            backend = _only(pool)
            assert not backend.healthy
            assert not pool.ready()
            assert backend.consecutive_failures == 1

            stub.mode = "ok"
            pool.start()
            assert await _wait_until(lambda: pool.ready())
            await pool.aclose()

    asyncio.run(body())


def test_admin_add_still_rejects_a_backend_it_cannot_verify():
    stub = StubBackend(mode="fail")

    async def body():
        with stubbed(stub):
            pool = backend_pool.BackendPool(_health(), _no_sleep_retry())
            with pytest.raises(httpx.HTTPStatusError):
                await pool.add("http://splade.stub", require_verify=True)
            assert pool.backends == {}
            await pool.aclose()

    asyncio.run(body())


def test_a_second_backend_must_match_the_first_encoder_contract():
    """The contract check runs on every promotion, not only at registration."""
    good = StubBackend()
    bf16 = StubBackend(metadata_overrides={"document_compute_dtype": "bf16"})

    async def body():
        with stubbed({"http://splade.a": good, "http://splade.b": bf16}):
            pool = backend_pool.BackendPool(_health(), _no_sleep_retry())
            await pool.add("http://splade.a", require_verify=True)
            with pytest.raises(ValueError, match="encoder contract differs"):
                await pool.add("http://splade.b", require_verify=True)

            # And on the tolerant startup path the mismatch keeps it out too.
            await pool.add("http://splade.b", require_verify=False)
            second = pool.backends["b3"]
            assert not second.healthy
            assert "encoder contract differs" in (second.last_probe_error or "")
            await pool.aclose()

    asyncio.run(body())


def test_transitions_log_once_at_warning(caplog):
    """Six failures, one line.

    The dense outage produced 268k tracebacks and exactly one line that said what
    had happened. Per-event logging is how that line gets buried; per-state
    logging is how it gets read.
    """
    stub = StubBackend()

    async def body():
        with stubbed(stub):
            pool = await _make_pool()
            backend = _only(pool)
            with caplog.at_level(logging.WARNING, logger=backend_pool.log.name):
                stub.mode = "fail"
                for _ in range(6):
                    await backend.probe()
                stub.mode = "ok"
                await backend.probe()

            messages = [record.getMessage() for record in caplog.records]
            tripped = [m for m in messages if "marked unhealthy" in m]
            restored = [m for m in messages if "restored" in m]
            assert len(tripped) == 1, messages
            assert len(restored) == 1, messages
            await pool.aclose()

    asyncio.run(body())


def test_stats_expose_probe_loop_liveness():
    """The metric a frozen loop cannot fake.

    `healthy` is written only by the probe loop, so a dead loop leaves it frozen
    at its last value and a scrape-time collector re-exports that just as
    faithfully as a push gauge would.
    """
    stub = StubBackend()

    async def body():
        with stubbed(stub):
            pool = await _make_pool()
            assert pool.stats()["probe_loop_last_iteration_at"] == 0.0
            pool.start()
            seeded = pool.stats()["probe_loop_last_iteration_at"]
            assert seeded > 0.0, "start() must seed the stamp"
            assert await _wait_until(
                lambda: pool.stats()["probe_loop_last_iteration_at"] > seeded
            )
            assert pool.stats()["probe_loop_errors"] == 0
            await pool.aclose()

    asyncio.run(body())
