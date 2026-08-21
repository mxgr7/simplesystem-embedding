"""Recovery tests for `embedding-service/embed_client.py` (MXG-159).

The pool had no coverage at all, and the outage it caused was entirely in
the recovery path: a backend went unhealthy and stayed unhealthy for four
hours across a restart of the backend itself. These tests pin the three
independent ways back in:

  - the probe loop restores a backend with no traffic at all;
  - a probe that never answers cannot freeze that loop (the regression
    test for the actual production failure);
  - a half-open trial restores a backend from traffic alone, which is what
    recovers the service when the probe path and the serving path
    disagree, plus the client recycling that resolves that disagreement.

`httpx.MockTransport` deliberately ignores httpx's own timeouts, which is
exactly the property the production failure had: the timeout on the
request did not bound anything. So a "hang" stub here really does hang,
and only the pool's own bound can stop it.

Same flat-import convention as the other embedding-service tests, but the
module is loaded by path because `embed_client` is a colliding module
name (see conftest).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path

import httpx
import pytest

from conftest import load_flat_service

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVICE_DIR = REPO_ROOT / "embedding-service"
ec = load_flat_service(
    "embedding_service", SERVICE_DIR, "embed_client"
).embed_client


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class StubTEI:
    """One TEI behind a MockTransport, with a switchable failure mode.

    `mode` is flipped mid-test to simulate a backend going away and coming
    back: "ok" serves, "fail" 503s, "hang" never answers."""

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.health_calls = 0
        self.embed_calls = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        is_health = request.url.path == "/health"
        if is_health:
            self.health_calls += 1
        else:
            self.embed_calls += 1
        if self.mode == "hang":
            await asyncio.sleep(3600)
        if self.mode == "fail":
            return httpx.Response(503, text="backend down")
        if self.mode == "reject" and not is_health:
            return httpx.Response(422, text="input validation failed")
        if is_health:
            return httpx.Response(200, text="")
        n = len(json.loads(request.content)["inputs"])
        return httpx.Response(200, json=[[0.5, 0.25] for _ in range(n)])


@contextlib.contextmanager
def stubbed(stub):
    """Route every client the pool builds, including recycled ones, at
    `stub`, which is either one StubTEI or a {base_url: StubTEI} map."""
    original = ec.TEIBackend._new_client

    def _new_client(self):
        handler = stub[self.base_url] if isinstance(stub, dict) else stub
        return httpx.AsyncClient(
            base_url=self.base_url, transport=httpx.MockTransport(handler),
        )

    ec.TEIBackend._new_client = _new_client
    try:
        yield
    finally:
        ec.TEIBackend._new_client = original


def _health(**overrides) -> "ec.HealthPolicy":
    base = {
        "probe_interval_s": 0.02,
        "probe_timeout_s": 0.05,
        "probe_round_timeout_s": 0.2,
        "half_open_interval_s": 5.0,
        "client_recycle_after_s": 60.0,
    }
    base.update(overrides)
    return ec.HealthPolicy(**base)


def _no_sleep_policy(**overrides) -> "ec.RetryPolicy":
    base = {
        "max_attempts": 2,
        "initial_backoff_s": 0.0,
        "max_single_backoff_s": 0.0,
        "total_budget_s": 5.0,
    }
    base.update(overrides)
    return ec.RetryPolicy(**base)


async def _make_pool(*, health=None, policy=None) -> "ec.TEIPool":
    pool = ec.TEIPool(
        health=health or _health(),
        retry_policy=policy or _no_sleep_policy(),
    )
    await pool.add_backend(
        "http://tei.stub", weight=1.0, max_concurrency=2,
        max_client_batch=8, timeout_s=1.0,
    )
    # `add_backend` fires a one-shot confirmation probe as a task. Let it
    # land here, or it lands in the middle of a test and rewrites the health
    # state under it.
    await asyncio.sleep(0.01)
    return pool


def _only(pool: "ec.TEIPool") -> "ec.TEIBackend":
    return next(iter(pool._backends.values()))


async def _wait_until(predicate, timeout_s: float = 3.0) -> bool:
    """Poll `predicate` on the event loop until true or `timeout_s`."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


# ---------------------------------------------------------------------------
# Health transitions
# ---------------------------------------------------------------------------

def test_two_failures_trip_backend_unhealthy() -> None:
    stub = StubTEI("ok")

    async def body():
        with stubbed(stub):
            pool = await _make_pool()
            backend = _only(pool)
            await backend.probe()
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


def test_input_rejections_do_not_affect_backend_health() -> None:
    stub = StubTEI("ok")

    async def body():
        with stubbed(stub):
            pool = await _make_pool()
            backend = _only(pool)
            await backend.probe()
            assert backend.healthy
            backend.mark_failure(httpx.ConnectError("one transient failure"))
            assert backend.consecutive_failures == 1

            stub.mode = "reject"
            for _ in range(3):
                with pytest.raises(ec.TEIInputError) as raised:
                    await pool.embed(["bad input"])
                assert raised.value.status_code == 422
                assert "input validation failed" not in str(raised.value)

            assert backend.healthy
            assert backend.consecutive_failures == 0
            assert backend.client_generation == 0
            assert stub.embed_calls == 3
            await pool.aclose()

    asyncio.run(body())


def test_probe_loop_restores_backend_without_traffic() -> None:
    """The plain recovery the service was supposed to have all along: the
    backend comes back and nobody sends a request."""
    stub = StubTEI("fail")

    async def body():
        with stubbed(stub):
            pool = await _make_pool()
            backend = _only(pool)
            pool.start()
            assert await _wait_until(lambda: not backend.healthy)

            stub.mode = "ok"
            assert await _wait_until(lambda: backend.healthy), (
                "probe loop never restored the backend"
            )
            assert stub.embed_calls == 0, "recovery must not need traffic"
            await pool.aclose()

    asyncio.run(body())


def test_hung_probe_does_not_freeze_the_loop() -> None:
    """Regression test for the outage.

    A probe that never answers used to stop the loop for good: health froze
    at unhealthy, and the backend recovering changed nothing because
    nothing was left to notice."""
    stub = StubTEI("ok")

    async def body():
        with stubbed(stub):
            pool = await _make_pool()
            backend = _only(pool)
            pool.start()
            assert await _wait_until(lambda: backend.healthy)

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
            assert await _wait_until(lambda: backend.healthy), (
                "loop never recovered after the hang cleared"
            )
            await pool.aclose()

    asyncio.run(body())


def test_client_is_recycled_after_prolonged_unhealth() -> None:
    """A wedged connection pool looks exactly like a dead backend from the
    outside, and only a fresh client tells them apart."""
    stub = StubTEI("fail")

    async def body():
        with stubbed(stub):
            pool = await _make_pool(
                health=_health(client_recycle_after_s=0.05),
            )
            backend = _only(pool)
            pool.start()
            assert await _wait_until(lambda: backend.client_generation >= 1), (
                "client was never recycled"
            )

            stub.mode = "ok"
            assert await _wait_until(lambda: backend.healthy)
            await pool.aclose()

    asyncio.run(body())


def test_pool_timeouts_recycle_the_client_early() -> None:
    """A pool timeout says our own connection pool is full of requests that
    will never finish, so waiting out the recycle timer is waiting for
    information we already have. Measured on the box: it cost another 35s
    of downtime after TEI was already serving again."""
    stub = StubTEI("ok")

    async def body():
        with stubbed(stub):
            pool = await _make_pool(
                health=_health(
                    client_recycle_after_s=600.0,   # timer must not be what fires
                    pool_timeout_recycle_after=3,
                    probe_interval_s=0.001,         # spacing floor = 0.003s
                ),
            )
            backend = _only(pool)
            await backend.probe()

            for _ in range(3):
                backend.mark_failure(httpx.PoolTimeout(""), source="probe")
            assert not backend.healthy
            assert backend.consecutive_pool_timeouts == 3
            await asyncio.sleep(0.01)
            assert backend.recycle_due(asyncio.get_event_loop().time())

            # A non-pool failure must not take the fast path.
            backend.mark_success()
            backend.mark_failure(httpx.ConnectError(""), source="probe")
            backend.mark_failure(httpx.ConnectError(""), source="probe")
            backend.mark_failure(httpx.ConnectError(""), source="probe")
            assert backend.consecutive_pool_timeouts == 0
            assert not backend.recycle_due(asyncio.get_event_loop().time())
            await pool.aclose()

    asyncio.run(body())


def test_healthy_backend_client_is_not_recycled() -> None:
    stub = StubTEI("ok")

    async def body():
        with stubbed(stub):
            pool = await _make_pool(health=_health(client_recycle_after_s=0.05))
            backend = _only(pool)
            pool.start()
            assert await _wait_until(lambda: backend.healthy)
            await asyncio.sleep(0.3)
            assert backend.client_generation == 0
            await pool.aclose()

    asyncio.run(body())


# ---------------------------------------------------------------------------
# Half-open admission
# ---------------------------------------------------------------------------

def test_half_open_trial_recovers_from_traffic_alone() -> None:
    """No probe loop running: the request path itself has to find out that
    the backend is back, or a broken probe path means a permanent outage."""
    stub = StubTEI("ok")

    async def body():
        with stubbed(stub):
            pool = await _make_pool(health=_health(half_open_interval_s=0.0))
            backend = _only(pool)
            await backend.probe()

            stub.mode = "fail"
            with contextlib.suppress(Exception):
                await pool.embed(["a"])
            assert not backend.healthy

            stub.mode = "ok"
            out = await pool.embed(["a"])
            assert out.shape == (1, 2)
            assert backend.healthy
            assert not backend.trial_inflight
            await pool.aclose()

    asyncio.run(body())


def test_half_open_is_rate_limited() -> None:
    """A hard-down backend costs one trial per interval, not one per
    caller: the point of the circuit breaker is not to melt the backend
    while it is trying to come up."""
    stub = StubTEI("ok")

    async def body():
        with stubbed(stub):
            pool = await _make_pool(health=_health(half_open_interval_s=30.0))
            backend = _only(pool)
            await backend.probe()

            stub.mode = "fail"
            backend.mark_failure()
            backend.mark_failure()
            assert not backend.healthy

            before = stub.embed_calls
            for _ in range(3):
                with contextlib.suppress(Exception):
                    await pool.embed(["a"])
            assert stub.embed_calls - before == 1
            await pool.aclose()

    asyncio.run(body())


def test_no_backend_raises_typed_error() -> None:
    """`main.py` answers 503 off this type; a bare RuntimeError would be a
    500 with a traceback per request."""
    stub = StubTEI("ok")

    async def body():
        with stubbed(stub):
            pool = await _make_pool(health=_health(half_open_interval_s=30.0))
            backend = _only(pool)
            await backend.probe()
            backend.draining = True  # not selectable, not trial-eligible

            raised = None
            try:
                await pool.embed(["a"])
            except Exception as e:  # noqa: BLE001 — asserting the type below
                raised = e
            assert isinstance(raised, ec.NoHealthyBackendError)
            assert isinstance(raised, RuntimeError)
            await pool.aclose()

    asyncio.run(body())


# ---------------------------------------------------------------------------
# Failover (unchanged behaviour, pinned because the retry block was
# restructured to release the half-open slot on every exit path)
# ---------------------------------------------------------------------------

def test_transient_failure_fails_over_to_another_backend() -> None:
    bad, good = StubTEI("ok"), StubTEI("ok")

    async def body():
        with stubbed({"http://tei.bad": bad, "http://tei.good": good}):
            pool = ec.TEIPool(health=_health(), retry_policy=_no_sleep_policy())
            # weight 0 makes `good` fallback-only, so the failing backend is
            # picked first and the failover path is the one under test.
            await pool.add_backend(
                "http://tei.bad", weight=1.0, max_concurrency=2,
                max_client_batch=8, timeout_s=1.0,
            )
            await pool.add_backend(
                "http://tei.good", weight=0.0, max_concurrency=2,
                max_client_batch=8, timeout_s=1.0,
            )
            for b in pool._backends.values():
                await b.probe()

            bad.mode = "fail"
            out = await pool.embed(["a"])
            assert out.shape == (1, 2)
            assert bad.embed_calls == 1
            assert good.embed_calls == 1
            await pool.aclose()

    asyncio.run(body())


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def test_transitions_log_once_at_warning(caplog) -> None:
    """One line when it breaks, one when it comes back, nothing per
    rejected request. The outage produced 268k tracebacks and exactly one
    line that said what had happened."""
    stub = StubTEI("ok")

    async def body():
        with stubbed(stub):
            pool = await _make_pool()
            backend = _only(pool)
            await backend.probe()

            with caplog.at_level(logging.WARNING, logger=ec.log.name):
                stub.mode = "fail"
                for _ in range(6):
                    await backend.probe()
                stub.mode = "ok"
                await backend.probe()

            messages = [r.getMessage() for r in caplog.records]
            tripped = [m for m in messages if "marked unhealthy" in m]
            restored = [m for m in messages if "restored" in m]
            assert len(tripped) == 1, messages
            assert len(restored) == 1, messages
            await pool.aclose()

    asyncio.run(body())
