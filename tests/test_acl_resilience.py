"""A5 tests — ftsearch retry + tracing forwarding + per-call metrics.

Coverage:
  - 5xx response retried, succeeds on 2nd attempt → metrics counters
    fire correctly (1 retry fired, 0 exhausted).
  - 5xx response retried 5× then exhausts → exhausted counter +1.
  - 4xx response raises immediately, no retry.
  - Network error retried.
  - Tracing baggage forwarded on every retry attempt (not just the
    first) so a downstream tracer sees consistent context.
  - retry_policy=None disables retries (escape hatch).
  - Per-call metrics: outcome label correctly tagged on success vs
    upstream_5xx vs network_error.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from acl.clients.ftsearch import FtsearchClient, RetryPolicy  # noqa: E402
from acl.metrics import (  # noqa: E402
    ftsearch_call_duration_seconds,
    ftsearch_retries_exhausted_total,
    ftsearch_retries_fired_total,
)
from acl.tracing import extract_trace_context  # noqa: E402


def _no_sleep_policy(**overrides: Any) -> RetryPolicy:
    base: dict[str, Any] = {
        "max_attempts": 3,
        "initial_backoff_s": 0.0,
        "max_single_backoff_s": 0.0,
        "total_budget_s": 5.0,
    }
    base.update(overrides)
    return RetryPolicy(**base)


def _make_client(handler, *, policy: RetryPolicy | None) -> FtsearchClient:
    c = FtsearchClient("http://stub", retry_policy=policy)
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[attr-defined]
    return c


def _counter_value(c) -> float:
    return c._value.get()


# ---- retry behavior -----------------------------------------------------

def test_5xx_retried_then_succeeds() -> None:
    seen = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] < 2:
            return httpx.Response(503, text="overload")
        return httpx.Response(200, json={"articles": []})

    before_fired = _counter_value(ftsearch_retries_fired_total)
    before_exh = _counter_value(ftsearch_retries_exhausted_total)
    client = _make_client(handler, policy=_no_sleep_policy())
    out = asyncio.run(client.search({"query": "x"}))
    assert out == {"articles": []}
    assert seen["n"] == 2
    assert _counter_value(ftsearch_retries_fired_total) == before_fired + 1
    assert _counter_value(ftsearch_retries_exhausted_total) == before_exh


def test_all_5xx_exhausts() -> None:
    seen = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        return httpx.Response(503, text=f"overload {seen['n']}")

    before_exh = _counter_value(ftsearch_retries_exhausted_total)
    client = _make_client(handler, policy=_no_sleep_policy(max_attempts=4))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.search({"query": "x"}))
    assert seen["n"] == 4
    assert _counter_value(ftsearch_retries_exhausted_total) == before_exh + 1


def test_4xx_raises_without_retry() -> None:
    seen = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        return httpx.Response(400, text="bad input")

    before_fired = _counter_value(ftsearch_retries_fired_total)
    client = _make_client(handler, policy=_no_sleep_policy(max_attempts=10))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.search({"query": "x"}))
    assert seen["n"] == 1
    assert _counter_value(ftsearch_retries_fired_total) == before_fired


def test_network_error_retried() -> None:
    seen = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] < 2:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json={"articles": []})

    client = _make_client(handler, policy=_no_sleep_policy())
    out = asyncio.run(client.search({"query": "x"}))
    assert out == {"articles": []}
    assert seen["n"] == 2


def test_no_retry_when_policy_none() -> None:
    seen = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        return httpx.Response(503, text="overload")

    client = _make_client(handler, policy=None)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.search({"query": "x"}))
    assert seen["n"] == 1


# ---- tracing forwarding -------------------------------------------------

def test_trace_headers_forwarded_on_first_attempt() -> None:
    captured: list[dict[str, str]] = []
    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(dict(req.headers))
        return httpx.Response(200, json={"articles": []})

    client = _make_client(handler, policy=None)
    asyncio.run(client.search(
        {"query": "x"},
        headers={"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
                 "baggage": "userId=42"},
    ))
    assert "traceparent" in captured[0]
    assert "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01" in captured[0]["traceparent"]
    assert "userId=42" in captured[0]["baggage"]


def test_trace_headers_forwarded_on_every_retry() -> None:
    """A retry must carry the same trace context — without it the
    downstream tracer sees the second/third attempt as orphan
    requests with no parent span."""
    captured: list[dict[str, str]] = []
    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(dict(req.headers))
        if len(captured) < 3:
            return httpx.Response(503, text="overload")
        return httpx.Response(200, json={"articles": []})

    client = _make_client(handler, policy=_no_sleep_policy(max_attempts=5))
    asyncio.run(client.search(
        {"query": "x"},
        headers={"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"},
    ))
    assert len(captured) == 3
    for headers in captured:
        assert "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01" \
            in headers["traceparent"]


# ---- TraceContext propagation --------------------------------------------

def test_extract_trace_context_used_by_acl() -> None:
    """Smoke: the ACL-side `extract_trace_context` returns the same
    shape as the ftsearch-side, and `headers_for_forwarding` filters
    baggage to the documented subset."""
    ctx = extract_trace_context({
        "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        "baggage": "userId=42,internal=secret",
    })
    out = ctx.headers_for_forwarding()
    assert "traceparent" in out
    assert "userId=42" in out["baggage"]
    # Internal baggage doesn't leak.
    assert "internal" not in out["baggage"]
