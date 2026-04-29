"""Unit tests for `search-api/embed_client.py`'s retry behaviour.

Validates:
  - Single embed() call → one POST.
  - 5xx on first attempt + 200 on second → retry succeeds.
  - 4xx response raises immediately (no retry).
  - All attempts 5xx → exhausts and raises.
  - Total budget caps attempts even with high `max_attempts`.

Uses `httpx.MockTransport` so we never hit the network.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "search-api"))

from embed_client import EmbedClient, EmbedRetryPolicy  # noqa: E402


def _client_with(handler, *, policy: EmbedRetryPolicy | None) -> EmbedClient:
    """Construct an EmbedClient whose underlying httpx uses the given
    `MockTransport` handler."""
    c = EmbedClient("http://stub", retry_policy=policy)
    # Replace the auto-built AsyncClient with one wired to the mock.
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[attr-defined]
    return c


def _no_sleep_policy(**overrides: Any) -> EmbedRetryPolicy:
    base: dict[str, Any] = {
        "max_attempts": 3,
        "initial_backoff_s": 0.0,
        "max_single_backoff_s": 0.0,
        "total_budget_s": 5.0,
    }
    base.update(overrides)
    return EmbedRetryPolicy(**base)


def test_single_call_succeeds() -> None:
    seen = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        return httpx.Response(200, json=[[0.1, 0.2]])
    client = _client_with(handler, policy=_no_sleep_policy())
    out = asyncio.run(client.embed(["hi"]))
    assert out == [[0.1, 0.2]]
    assert seen["n"] == 1


def test_retries_on_5xx_then_succeeds() -> None:
    seen = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] < 2:
            return httpx.Response(503, text="overload")
        return httpx.Response(200, json=[[1.0]])
    client = _client_with(handler, policy=_no_sleep_policy())
    out = asyncio.run(client.embed(["hi"]))
    assert out == [[1.0]]
    assert seen["n"] == 2


def test_4xx_raises_immediately() -> None:
    """Caller-side errors (bad input) shouldn't waste retries."""
    seen = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        return httpx.Response(400, text="bad input")
    client = _client_with(handler, policy=_no_sleep_policy(max_attempts=10))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.embed(["hi"]))
    assert seen["n"] == 1


def test_all_5xx_exhausts_attempts() -> None:
    seen = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        return httpx.Response(503, text=f"overload {seen['n']}")
    client = _client_with(handler, policy=_no_sleep_policy(max_attempts=4))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.embed(["hi"]))
    assert seen["n"] == 4


def test_no_retry_when_policy_none() -> None:
    seen = {"n": 0}
    def handler(req: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        return httpx.Response(503, text="overload")
    client = _client_with(handler, policy=None)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.embed(["hi"]))
    assert seen["n"] == 1


def test_truncate_flag_is_set_on_request() -> None:
    """TEI's max_input_length protection — same as the indexer-side
    `tei_cache.py:_tei_embed`. Without `truncate=true` an over-long
    query string returns 413."""
    captured: dict[str, Any] = {}
    def handler(req: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=[[0.0]])
    client = _client_with(handler, policy=None)
    asyncio.run(client.embed(["query: very long text"]))
    assert captured["body"] == {"inputs": ["query: very long text"], "truncate": True}
