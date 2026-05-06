"""Unit red-team tests for `acl/clients/ftsearch.py`.

Validates retry logic, timeout handling, error propagation, response
parsing edge cases, and concurrent-request isolation using
`httpx.MockTransport` — no network access required.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from acl.clients.ftsearch import FtsearchClient, RetryPolicy  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fast_policy(**overrides: Any) -> RetryPolicy:
    """Zero-sleep policy so tests run instantly."""
    base: dict[str, Any] = {
        "max_attempts": 5,
        "initial_backoff_s": 0.0,
        "multiplier": 1.0,
        "max_single_backoff_s": 0.0,
        "total_budget_s": 999.0,
    }
    base.update(overrides)
    return RetryPolicy(**base)


def _make_client(
    handler,
    *,
    policy: RetryPolicy | None = None,
    timeout_s: float = 30.0,
) -> FtsearchClient:
    """Build a FtsearchClient wired to a MockTransport handler."""
    if policy is None:
        policy = _fast_policy()
    client = FtsearchClient(
        "http://stub",
        default_collection="test-col",
        timeout_s=timeout_s,
        retry_policy=policy,
    )
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        timeout=timeout_s,
    )
    return client


# ===================================================================
# 1. Retry behavior — retryable status codes
# ===================================================================


class TestRetryOnTransientStatus:
    """503, 500, 502, 504, 408, 429 should all trigger retries."""

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 408, 429])
    def test_retries_on_transient_status_then_succeeds(self, status: int) -> None:
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            if counter["n"] == 1:
                return httpx.Response(status, text="transient")
            return httpx.Response(200, json={"hits": []})

        client = _make_client(handler)
        result = asyncio.run(client.search({"q": "test"}))
        assert result == {"hits": []}
        assert counter["n"] == 2, f"Expected 2 attempts for status {status}"

    def test_retries_exactly_max_attempts_on_503(self) -> None:
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            return httpx.Response(503, text="always overloaded")

        client = _make_client(handler, policy=_fast_policy(max_attempts=3))
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            asyncio.run(client.search({"q": "x"}))
        assert exc_info.value.response.status_code == 503
        assert counter["n"] == 3

    def test_retries_five_times_by_default(self) -> None:
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            return httpx.Response(500, text="fail")

        client = _make_client(handler, policy=_fast_policy(max_attempts=5))
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(client.search({"q": "x"}))
        assert counter["n"] == 5


# ===================================================================
# 2. Non-retryable status codes
# ===================================================================


class TestNonRetryable:
    """4xx statuses should raise immediately without retrying."""

    @pytest.mark.parametrize("status", [400, 404, 422, 401, 403, 405])
    def test_4xx_not_retried(self, status: int) -> None:
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            return httpx.Response(status, text="client error")

        client = _make_client(handler, policy=_fast_policy(max_attempts=10))
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            asyncio.run(client.search({"q": "x"}))
        assert exc_info.value.response.status_code == status
        assert counter["n"] == 1, f"Status {status} must not be retried"

    def test_no_retry_policy_means_single_attempt(self) -> None:
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            return httpx.Response(503, text="overload")

        client = _make_client(handler, policy=None)
        # With policy=None the constructor takes a different path; rebuild
        client = FtsearchClient(
            "http://stub",
            default_collection="test-col",
            retry_policy=None,
        )
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(client.search({"q": "x"}))
        assert counter["n"] == 1


# ===================================================================
# 3. Retry exhaustion
# ===================================================================


class TestRetryExhaustion:
    """When all retries fail the caller must get the last error."""

    def test_exhausted_raises_http_status_error(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(502, text="bad gateway")

        client = _make_client(handler, policy=_fast_policy(max_attempts=3))
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            asyncio.run(client.search({"q": "x"}))
        assert exc_info.value.response.status_code == 502

    def test_exhaustion_preserves_last_status_code(self) -> None:
        """If status codes vary across attempts, the LAST one is raised."""
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            code = [500, 502, 503][counter["n"] - 1]
            return httpx.Response(code, text="fail")

        client = _make_client(handler, policy=_fast_policy(max_attempts=3))
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            asyncio.run(client.search({"q": "x"}))
        # The last attempt's status is what the caller sees.
        assert exc_info.value.response.status_code == 503

    def test_total_budget_caps_retries(self) -> None:
        """Even with max_attempts=100, total_budget_s stops further tries."""
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            return httpx.Response(503, text="overload")

        # Budget of ~0s with non-zero backoff: should stop after first retry
        # because the next backoff would exceed the budget.
        policy = RetryPolicy(
            max_attempts=100,
            initial_backoff_s=0.01,
            multiplier=1.0,
            max_single_backoff_s=1.0,
            total_budget_s=0.0,
        )
        client = _make_client(handler, policy=policy)
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(client.search({"q": "x"}))
        # Budget is 0 so after the first failure the next backoff (0.01s)
        # exceeds remaining budget => stop at attempt 2 at most.
        assert counter["n"] <= 2


# ===================================================================
# 4. Timeout behavior
# ===================================================================


class TestTimeout:
    """Client must enforce request timeouts."""

    def test_read_timeout_raises(self) -> None:
        """ReadTimeout from the transport layer propagates to the caller."""

        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("upstream too slow")

        client = _make_client(handler, policy=_fast_policy(max_attempts=1))
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(client.search({"q": "x"}))

    def test_read_timeout_is_retried(self) -> None:
        """ReadTimeout is transient and should be retried."""
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            if counter["n"] == 1:
                raise httpx.ReadTimeout("slow first attempt")
            return httpx.Response(200, json={"ok": True})

        client = _make_client(handler, policy=_fast_policy(max_attempts=3))
        result = asyncio.run(client.search({"q": "x"}))
        assert result == {"ok": True}
        assert counter["n"] == 2

    def test_write_timeout_is_retried(self) -> None:
        """WriteTimeout is also in the transient set."""
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            if counter["n"] == 1:
                raise httpx.WriteTimeout("write stalled")
            return httpx.Response(200, json={"ok": True})

        client = _make_client(handler, policy=_fast_policy(max_attempts=3))
        result = asyncio.run(client.search({"q": "x"}))
        assert result == {"ok": True}
        assert counter["n"] == 2

    def test_pool_timeout_is_retried(self) -> None:
        """PoolTimeout (all connections busy) is transient."""
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            if counter["n"] == 1:
                raise httpx.PoolTimeout("pool exhausted")
            return httpx.Response(200, json={"ok": True})

        client = _make_client(handler, policy=_fast_policy(max_attempts=3))
        result = asyncio.run(client.search({"q": "x"}))
        assert result == {"ok": True}
        assert counter["n"] == 2

    def test_timeout_exhaustion(self) -> None:
        """All attempts timeout -- caller gets ReadTimeout."""

        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("always slow")

        client = _make_client(handler, policy=_fast_policy(max_attempts=3))
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(client.search({"q": "x"}))

    def test_client_has_configured_timeout(self) -> None:
        """The underlying httpx.AsyncClient must carry the configured
        timeout so real requests would be bounded."""
        client = FtsearchClient(
            "http://stub",
            timeout_s=2.5,
        )
        assert client._client.timeout.read == 2.5
        asyncio.run(client.aclose())


# ===================================================================
# 5. Response parsing edge cases
# ===================================================================


class TestResponseParsing:
    """What happens when the body is not well-formed JSON?"""

    def test_empty_body_200(self) -> None:
        """200 with empty body — json() will raise."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        client = _make_client(handler, policy=_fast_policy(max_attempts=1))
        with pytest.raises(json.JSONDecodeError):
            asyncio.run(client.search({"q": "x"}))

    def test_non_json_body_200(self) -> None:
        """200 with HTML body — json() must fail."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>Not Found</html>")

        client = _make_client(handler, policy=_fast_policy(max_attempts=1))
        with pytest.raises(json.JSONDecodeError):
            asyncio.run(client.search({"q": "x"}))

    def test_truncated_json_body_200(self) -> None:
        """Truncated JSON — should fail to parse."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b'{"hits": [{"id": "abc"')

        client = _make_client(handler, policy=_fast_policy(max_attempts=1))
        with pytest.raises(json.JSONDecodeError):
            asyncio.run(client.search({"q": "x"}))

    def test_valid_json_is_returned_as_dict(self) -> None:
        payload = {"hits": [{"id": "1", "score": 0.95}]}

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        client = _make_client(handler, policy=_fast_policy(max_attempts=1))
        result = asyncio.run(client.search({"q": "x"}))
        assert result == payload

    def test_5xx_with_non_json_body_is_retried(self) -> None:
        """503 with HTML body should still be retried, not crash on parse."""
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            if counter["n"] == 1:
                return httpx.Response(503, text="<h1>Service Unavailable</h1>")
            return httpx.Response(200, json={"ok": True})

        client = _make_client(handler, policy=_fast_policy(max_attempts=3))
        result = asyncio.run(client.search({"q": "x"}))
        assert result == {"ok": True}
        assert counter["n"] == 2


# ===================================================================
# 6. Connection errors
# ===================================================================


class TestConnectionErrors:
    """Simulate DNS, connect, and mid-stream failures."""

    def test_connect_error_is_retried(self) -> None:
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            if counter["n"] == 1:
                raise httpx.ConnectError("Connection refused")
            return httpx.Response(200, json={"ok": True})

        client = _make_client(handler, policy=_fast_policy(max_attempts=3))
        result = asyncio.run(client.search({"q": "x"}))
        assert result == {"ok": True}
        assert counter["n"] == 2

    def test_connect_error_exhaustion(self) -> None:
        """All attempts fail with ConnectError."""

        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("DNS resolution failed")

        client = _make_client(handler, policy=_fast_policy(max_attempts=3))
        with pytest.raises(httpx.ConnectError):
            asyncio.run(client.search({"q": "x"}))

    def test_remote_protocol_error_is_retried(self) -> None:
        """Connection reset mid-response (RemoteProtocolError) is retried."""
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            if counter["n"] == 1:
                raise httpx.RemoteProtocolError("Connection reset")
            return httpx.Response(200, json={"ok": True})

        client = _make_client(handler, policy=_fast_policy(max_attempts=3))
        result = asyncio.run(client.search({"q": "x"}))
        assert result == {"ok": True}
        assert counter["n"] == 2

    def test_non_transient_exception_not_retried(self) -> None:
        """Arbitrary exceptions outside the transient set should propagate
        immediately."""
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            raise httpx.DecodingError("bad encoding")

        client = _make_client(handler, policy=_fast_policy(max_attempts=5))
        with pytest.raises(httpx.DecodingError):
            asyncio.run(client.search({"q": "x"}))
        assert counter["n"] == 1


# ===================================================================
# 7. Concurrent requests through the same client
# ===================================================================


class TestConcurrentRequests:
    """Ensure concurrent calls through the same FtsearchClient
    don't share retry state or corrupt each other's results."""

    def test_concurrent_searches_independent_results(self) -> None:
        """Two concurrent searches must get their own results."""

        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            # Echo the query back so we can verify isolation.
            return httpx.Response(200, json={"echo": body.get("q")})

        client = _make_client(handler, policy=_fast_policy(max_attempts=1))

        async def run():
            a, b = await asyncio.gather(
                client.search({"q": "alpha"}),
                client.search({"q": "beta"}),
            )
            return a, b

        a, b = asyncio.run(run())
        assert a == {"echo": "alpha"}
        assert b == {"echo": "beta"}

    def test_concurrent_one_fails_other_succeeds(self) -> None:
        """One request gets a 503, the other 200 — they must be independent."""
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            if body.get("q") == "fail":
                return httpx.Response(503, text="fail")
            return httpx.Response(200, json={"ok": True})

        # Use max_attempts=1 so the failing one surfaces immediately.
        client = _make_client(handler, policy=_fast_policy(max_attempts=1))

        async def run():
            results = await asyncio.gather(
                client.search({"q": "fail"}),
                client.search({"q": "succeed"}),
                return_exceptions=True,
            )
            return results

        results = asyncio.run(run())
        # One should be a success dict, the other an exception.
        successes = [r for r in results if isinstance(r, dict)]
        errors = [r for r in results if isinstance(r, BaseException)]
        assert len(successes) == 1
        assert successes[0] == {"ok": True}
        assert len(errors) == 1
        assert isinstance(errors[0], httpx.HTTPStatusError)

    def test_concurrent_retries_dont_interfere(self) -> None:
        """Two requests both retrying should maintain independent counters."""
        counters: dict[str, int] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            key = body["q"]
            counters[key] = counters.get(key, 0) + 1
            if counters[key] < 3:
                return httpx.Response(503, text="retry me")
            return httpx.Response(200, json={"q": key, "attempt": counters[key]})

        client = _make_client(handler, policy=_fast_policy(max_attempts=5))

        async def run():
            a, b = await asyncio.gather(
                client.search({"q": "req-A"}),
                client.search({"q": "req-B"}),
            )
            return a, b

        a, b = asyncio.run(run())
        assert a["q"] == "req-A"
        assert b["q"] == "req-B"
        # Both should have needed 3 attempts.
        assert counters["req-A"] == 3
        assert counters["req-B"] == 3


# ===================================================================
# 8. URL construction and collection routing
# ===================================================================


class TestURLConstruction:
    """Verify the URL template `{base_url}/{collection}/_search`."""

    def test_default_collection_in_url(self) -> None:
        captured: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["url"] = str(req.url)
            return httpx.Response(200, json={})

        client = _make_client(handler, policy=_fast_policy(max_attempts=1))
        asyncio.run(client.search({"q": "x"}))
        assert captured["url"] == "http://stub/test-col/_search"

    def test_custom_collection_overrides_default(self) -> None:
        captured: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["url"] = str(req.url)
            return httpx.Response(200, json={})

        client = _make_client(handler, policy=_fast_policy(max_attempts=1))
        asyncio.run(client.search({"q": "x"}, collection="offers"))
        assert captured["url"] == "http://stub/offers/_search"

    def test_params_forwarded(self) -> None:
        captured: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["url"] = str(req.url)
            return httpx.Response(200, json={})

        client = _make_client(handler, policy=_fast_policy(max_attempts=1))
        asyncio.run(client.search({"q": "x"}, params={"limit": "10"}))
        assert "limit=10" in captured["url"]

    def test_headers_forwarded(self) -> None:
        captured: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["traceparent"] = req.headers.get("traceparent")
            return httpx.Response(200, json={})

        client = _make_client(handler, policy=_fast_policy(max_attempts=1))
        asyncio.run(
            client.search(
                {"q": "x"},
                headers={"traceparent": "00-abc-def-01"},
            )
        )
        assert captured["traceparent"] == "00-abc-def-01"


# ===================================================================
# 9. Backoff timing verification
# ===================================================================


class TestBackoffTiming:
    """Verify the exponential backoff and multiplier semantics."""

    def test_backoff_multiplier_applied(self) -> None:
        """With initial=0.05s and multiplier=2, delays should be
        0.05, 0.1, 0.2, ... (within tolerance)."""
        timestamps: list[float] = []

        def handler(req: httpx.Request) -> httpx.Response:
            timestamps.append(time.monotonic())
            if len(timestamps) < 4:
                return httpx.Response(503, text="retry")
            return httpx.Response(200, json={"ok": True})

        policy = RetryPolicy(
            max_attempts=5,
            initial_backoff_s=0.05,
            multiplier=2.0,
            max_single_backoff_s=10.0,
            total_budget_s=10.0,
        )
        client = _make_client(handler, policy=policy)
        result = asyncio.run(client.search({"q": "x"}))
        assert result == {"ok": True}

        # Verify increasing delays between attempts.
        delays = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        # Attempt 1->2 delay ~0.05s, 2->3 ~0.10s, 3->4 ~0.20s
        assert delays[0] >= 0.03, f"First delay too short: {delays[0]:.3f}s"
        assert delays[1] >= delays[0] * 1.5, (
            f"Multiplier not applied: {delays[1]:.3f}s <= {delays[0]*1.5:.3f}s"
        )
