"""Unit tests for `search-api/milvus_helpers.py`.

Coverage:
  - `BoundedMilvusClient` pins consistency_level + timeout on
    search/query/get; caller-supplied values win.
  - Pass-through forwards every other method to the underlying client.
  - `retry` exponential backoff: succeeds, retries on transient,
    raises on permanent error, exhausts attempts, exhausts total
    budget, ignores caller-passed timeout/consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "search-api"))

from milvus_helpers import (  # noqa: E402
    DEFAULT_CONSISTENCY_LEVEL,
    DEFAULT_PER_CALL_TIMEOUT_S,
    BoundedMilvusClient,
    RetryPolicy,
    retry,
)


class _RecordingClient:
    """Captures call kwargs for assertions. Stand-in for a real
    `pymilvus.MilvusClient`."""
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def search(self, **kwargs: Any) -> str:
        self.calls.append(("search", kwargs))
        return "search-result"

    def query(self, **kwargs: Any) -> str:
        self.calls.append(("query", kwargs))
        return "query-result"

    def get(self, **kwargs: Any) -> str:
        self.calls.append(("get", kwargs))
        return "get-result"

    def upsert(self, **kwargs: Any) -> str:
        self.calls.append(("upsert", kwargs))
        return "upsert-result"

    def has_collection(self, name: str) -> bool:
        self.calls.append(("has_collection", {"name": name}))
        return True


def test_search_pins_consistency_when_unspecified() -> None:
    raw = _RecordingClient()
    client = BoundedMilvusClient(raw)
    out = client.search(collection_name="c", data=[[1.0]])
    assert out == "search-result"
    assert raw.calls[0][1]["consistency_level"] == DEFAULT_CONSISTENCY_LEVEL


def test_query_pins_consistency_when_unspecified() -> None:
    raw = _RecordingClient()
    client = BoundedMilvusClient(raw)
    client.query(collection_name="c", filter='id == "x"')
    assert raw.calls[0][1]["consistency_level"] == DEFAULT_CONSISTENCY_LEVEL


def test_get_pins_consistency_when_unspecified() -> None:
    raw = _RecordingClient()
    client = BoundedMilvusClient(raw)
    client.get(collection_name="c", ids=["x"])
    assert raw.calls[0][1]["consistency_level"] == DEFAULT_CONSISTENCY_LEVEL


def test_caller_supplied_consistency_level_wins() -> None:
    """Explicit `consistency_level='Strong'` must propagate — required
    for the rare strongly-consistent read like a post-upsert verify
    after a swing test."""
    raw = _RecordingClient()
    client = BoundedMilvusClient(raw)
    client.search(collection_name="c", consistency_level="Strong", data=[[1.0]])
    assert raw.calls[0][1]["consistency_level"] == "Strong"


def test_passthrough_methods_unmodified() -> None:
    """Non-read methods should pass through with no kwarg additions —
    upserts, schema reads, alias methods, etc."""
    raw = _RecordingClient()
    client = BoundedMilvusClient(raw)
    client.upsert(collection_name="c", data=[{"x": 1}])
    assert raw.calls[0] == ("upsert", {"collection_name": "c", "data": [{"x": 1}]})
    assert "consistency_level" not in raw.calls[0][1]


def test_passthrough_handles_positional_args() -> None:
    """`has_collection` is positional in pymilvus — our wrapper must
    forward the call as-is via `__getattr__`."""
    raw = _RecordingClient()
    client = BoundedMilvusClient(raw)
    assert client.has_collection("foo") is True
    assert raw.calls[0] == ("has_collection", {"name": "foo"})


def test_underlying_client_attribute_access() -> None:
    """Direct attribute access (e.g. for a `_client.session_state`
    lookup) should reach the underlying object."""
    raw = _RecordingClient()
    raw.custom_attr = "hello"  # type: ignore[attr-defined]
    client = BoundedMilvusClient(raw)
    assert client.custom_attr == "hello"


# ---- timeout pinning ----------------------------------------------------

def test_search_pins_default_timeout_when_unspecified() -> None:
    raw = _RecordingClient()
    client = BoundedMilvusClient(raw, retry_policy=None)
    client.search(collection_name="c", data=[[1.0]])
    assert raw.calls[0][1]["timeout"] == DEFAULT_PER_CALL_TIMEOUT_S


def test_caller_supplied_timeout_wins() -> None:
    """A long-running operator query (e.g. a one-off audit) overrides
    the default by passing `timeout=N`."""
    raw = _RecordingClient()
    client = BoundedMilvusClient(raw, retry_policy=None)
    client.query(collection_name="c", filter='id == "x"', timeout=60.0)
    assert raw.calls[0][1]["timeout"] == 60.0


def test_timeout_disabled_when_constructor_passes_none() -> None:
    raw = _RecordingClient()
    client = BoundedMilvusClient(raw, timeout_s=None, retry_policy=None)
    client.query(collection_name="c", filter='id == "x"')
    assert "timeout" not in raw.calls[0][1]


# ---- retry semantics ----------------------------------------------------

class _TransientError(RuntimeError):
    """Stand-in for a transient gRPC failure from Milvus."""


def _no_sleep_policy(**overrides: Any) -> RetryPolicy:
    base = {"initial_backoff_s": 0.0, "max_single_backoff_s": 0.0,
            "total_budget_s": 1.0, "multiplier": 1.5, "max_attempts": 3}
    base.update(overrides)
    return RetryPolicy(**base)


def test_retry_succeeds_on_first_attempt() -> None:
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        return 42
    assert retry(fn, policy=_no_sleep_policy(), label="t") == 42
    assert calls["n"] == 1


def test_retry_recovers_after_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _TransientError("connection reset")
        return "ok"
    monkeypatch.setattr("milvus_helpers.time.sleep", lambda _: None)
    assert retry(fn, policy=_no_sleep_policy(max_attempts=4), label="t") == "ok"
    assert calls["n"] == 3


def test_retry_raises_immediately_on_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise RuntimeError("schema validation failed")
    monkeypatch.setattr("milvus_helpers.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="schema"):
        retry(fn, policy=_no_sleep_policy(max_attempts=10), label="t")
    assert calls["n"] == 1


def test_retry_exhausts_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise _TransientError(f"unavailable {calls['n']}")
    monkeypatch.setattr("milvus_helpers.time.sleep", lambda _: None)
    with pytest.raises(_TransientError):
        retry(fn, policy=_no_sleep_policy(max_attempts=4), label="t")
    assert calls["n"] == 4


def test_retry_total_budget_caps_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even with attempts=10, a tight `total_budget_s` should make us
    raise sooner — the sum of cumulative backoffs is checked. Use a
    fake monotonic clock that advances by the slept duration so the
    budget check sees realistic elapsed time without actually sleeping."""
    calls = {"n": 0}
    sleeps: list[float] = []
    clock = [0.0]

    def fn():
        calls["n"] += 1
        raise _TransientError("transient")

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        clock[0] += s

    monkeypatch.setattr("milvus_helpers.time.sleep", fake_sleep)
    monkeypatch.setattr("milvus_helpers.time.monotonic", lambda: clock[0])

    # Each backoff = 1.0s. With total_budget_s=2.5, after attempt 3
    # the cumulative would be ~3.0s which exceeds budget — raise.
    policy = RetryPolicy(
        max_attempts=10,
        initial_backoff_s=1.0,
        multiplier=1.0,  # constant 1.0 per attempt
        max_single_backoff_s=1.0,
        total_budget_s=2.5,
    )
    with pytest.raises(_TransientError):
        retry(fn, policy=policy, label="t")
    # 1st attempt: fail, sleep 1s (clock=1)
    # 2nd attempt: fail, sleep 1s (clock=2)
    # 3rd attempt: fail, would-sleep 1s but elapsed(2) + wait(1) > budget(2.5) → raise
    assert calls["n"] == 3
    assert sleeps == [1.0, 1.0]


def test_bounded_client_retries_on_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: BoundedMilvusClient.search retries when the
    underlying client raises transient, then succeeds."""
    calls = {"n": 0}

    class _RetryingClient:
        def search(self, **kwargs: Any) -> str:
            calls["n"] += 1
            if calls["n"] < 2:
                raise _TransientError("transient")
            return "ok"

    monkeypatch.setattr("milvus_helpers.time.sleep", lambda _: None)
    client = BoundedMilvusClient(
        _RetryingClient(),  # type: ignore[arg-type]
        retry_policy=_no_sleep_policy(),
    )
    assert client.search(collection_name="c", data=[[1.0]]) == "ok"
    assert calls["n"] == 2


def test_bounded_client_no_retry_when_policy_none() -> None:
    """`retry_policy=None` short-circuits — first failure raises
    immediately. Useful for tests + nested-retry callers."""
    class _FailingClient:
        def search(self, **kwargs: Any) -> str:
            raise _TransientError("transient")

    client = BoundedMilvusClient(_FailingClient(), retry_policy=None)  # type: ignore[arg-type]
    with pytest.raises(_TransientError):
        client.search(collection_name="c", data=[[1.0]])
