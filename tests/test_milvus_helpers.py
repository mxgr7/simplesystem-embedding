"""Unit tests for `search-api/milvus_helpers.py`.

Validates the `BoundedMilvusClient` wrapper:
  - Pins consistency_level='Bounded' on search/query/get when caller
    didn't pass one.
  - Caller-supplied consistency_level wins (escape hatch for the rare
    `Strong`-required case).
  - Pass-through forwards every other method to the underlying client
    unmodified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "search-api"))

from milvus_helpers import DEFAULT_CONSISTENCY_LEVEL, BoundedMilvusClient  # noqa: E402


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
