"""Unit tests for `indexer.embedding_text` + `indexer.tei_cache`.

The cache uses a fake `redis.Redis`-compatible dict and stubs the TEI
HTTP call so the test runs offline. Validates:
  - Text rendering is stable, drops empty fields, sorts eclass codes.
  - First batch is all-miss → calls TEI once, populates cache.
  - Second batch with same hashes is all-hit → no TEI call.
  - Mixed batch (some new, some cached) → only the new hashes hit TEI,
    returned vectors stitch back in input order.
  - Duplicate hashes within a batch collapse to one TEI call.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.embedding_text import article_to_text  # noqa: E402
from indexer.tei_cache import TEICache  # noqa: E402


# ---------- text rendering --------------------------------------------------

def test_article_text_full() -> None:
    a = {
        "name": "Hammer X",
        "manufacturerName": "Acme",
        "category_l1": ["Tools"],
        "category_l2": ["Tools¦Hand"],
        "category_l3": [],
        "category_l4": [],
        "category_l5": [],
        "eclass5_code": [21050000, 27270911],
        "eclass7_code": [],
        "s2class_code": [99],
    }
    txt = article_to_text(a)
    assert txt.startswith("passage: Article Name: Hammer X")
    assert "Brand: Acme" in txt
    assert "Category: Tools Tools¦Hand" in txt
    assert "eClass5: 21050000 27270911" in txt
    assert "eClass7" not in txt  # empty omitted
    assert "S2Class: 99" in txt


def test_article_text_minimal() -> None:
    """Only `name` set — text should still be valid (no labels for
    empty fields)."""
    txt = article_to_text({"name": "Anonymous"})
    assert txt == "passage: Article Name: Anonymous"


def test_article_text_empty() -> None:
    """All fields empty — produce a stub passage so TEI never gets a
    blank string (model behaviour on empty input is undefined)."""
    txt = article_to_text({})
    assert txt == "passage:"


def test_article_text_eclass_sort_stable() -> None:
    """eClass codes are sorted — the upstream array order isn't
    guaranteed (DuckDB's `any_value` picks an arbitrary representative
    of the hash group), and the embedding must be invariant."""
    a1 = {"name": "X", "eclass5_code": [3, 1, 2]}
    a2 = {"name": "X", "eclass5_code": [2, 3, 1]}
    assert article_to_text(a1) == article_to_text(a2)


# ---------- cache fixtures --------------------------------------------------

class _FakeRedis:
    """Just enough of the redis.Redis API for TEICache: get / mget /
    set / pipeline (with MSET-style batch SETs)."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def mget(self, keys: list[str]) -> list[bytes | None]:
        return [self.store.get(k) for k in keys]

    def set(self, key: str, value: bytes) -> bool:
        self.store[key] = value
        return True

    def pipeline(self, transaction: bool = True) -> "_FakePipeline":
        return _FakePipeline(self)


class _FakePipeline:
    def __init__(self, parent: _FakeRedis) -> None:
        self.parent = parent
        self.queue: list[tuple[str, bytes]] = []

    def set(self, key: str, value: bytes) -> "_FakePipeline":
        self.queue.append((key, value))
        return self

    def execute(self) -> list[bool]:
        for k, v in self.queue:
            self.parent.set(k, v)
        out = [True] * len(self.queue)
        self.queue.clear()
        return out


class _CountingTEI:
    """Patches `TEICache._tei_embed` to return a deterministic vector
    per text and increment a call counter — no HTTP."""

    def __init__(self) -> None:
        self.call_count = 0
        self.texts_received: list[str] = []

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.call_count += 1
        self.texts_received.extend(texts)
        # Deterministic stub: hash the text bytes into 128 fp32 values.
        out = np.zeros((len(texts), 128), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(t.encode("utf-8")[:8].ljust(8, b"\x00"), "big")
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(128).astype(np.float32)
        return out


@pytest.fixture
def cache() -> TEICache:
    c = TEICache(
        tei_url="http://stub",
        redis_client=_FakeRedis(),  # type: ignore[arg-type]
        tei_batch_size=64,
    )
    # Patch out the HTTP call.
    c._tei_embed = _CountingTEI()  # type: ignore[method-assign]
    return c


def _articles(*hashes_and_names: tuple[str, str]) -> list[dict[str, Any]]:
    return [{"article_hash": h, "name": n} for h, n in hashes_and_names]


# ---------- cache behaviour -------------------------------------------------

def test_first_batch_all_miss(cache: TEICache) -> None:
    arts = _articles(("aaa", "Hammer"), ("bbb", "Wrench"))
    out = cache.embed_articles(arts)
    assert set(out) == {"aaa", "bbb"}
    assert all(v.dtype == np.float16 and v.shape == (128,) for v in out.values())
    assert cache._tei_embed.call_count == 1  # type: ignore[attr-defined]
    assert cache.stats.misses == 2 and cache.stats.hits == 0


def test_second_batch_all_hit(cache: TEICache) -> None:
    arts = _articles(("aaa", "Hammer"), ("bbb", "Wrench"))
    cache.embed_articles(arts)  # warm
    cache._tei_embed.call_count = 0  # type: ignore[attr-defined]
    cache.stats.hits = cache.stats.misses = 0

    out = cache.embed_articles(arts)
    assert set(out) == {"aaa", "bbb"}
    assert cache._tei_embed.call_count == 0  # type: ignore[attr-defined]
    assert cache.stats.hits == 2 and cache.stats.misses == 0


def test_mixed_batch_only_misses_hit_tei(cache: TEICache) -> None:
    """Pre-populate aaa; ask for [aaa, bbb, ccc]; only bbb + ccc go to TEI."""
    cache.embed_articles(_articles(("aaa", "Hammer")))
    cache._tei_embed.texts_received.clear()  # type: ignore[attr-defined]
    cache._tei_embed.call_count = 0  # type: ignore[attr-defined]

    out = cache.embed_articles(_articles(("aaa", "Hammer"), ("bbb", "Wrench"), ("ccc", "Drill")))
    assert set(out) == {"aaa", "bbb", "ccc"}
    assert cache._tei_embed.call_count == 1  # type: ignore[attr-defined]
    received = cache._tei_embed.texts_received  # type: ignore[attr-defined]
    assert len(received) == 2
    # Texts should be the wrench + drill renderings, not the hammer.
    assert any("Wrench" in t for t in received)
    assert any("Drill" in t for t in received)
    assert not any("Hammer" in t for t in received)


def test_duplicate_hashes_within_batch_collapse(cache: TEICache) -> None:
    """Two offers with the same article_hash → one TEI call, one
    Redis SET. The dict result has one entry per unique hash."""
    arts = _articles(("aaa", "Hammer"), ("aaa", "Hammer"), ("bbb", "Wrench"))
    out = cache.embed_articles(arts)
    assert set(out) == {"aaa", "bbb"}
    received = cache._tei_embed.texts_received  # type: ignore[attr-defined]
    assert len(received) == 2  # not 3 — duplicate collapsed


def test_corrupted_cache_entry_treated_as_miss(cache: TEICache) -> None:
    """A Redis value with the wrong byte length must trigger a fresh
    TEI call rather than serving garbage to Milvus."""
    # Manually plant a bad entry.
    cache._redis.store["tei:v1:aaa"] = b"\x00" * 128  # half the right size  # type: ignore[attr-defined]
    out = cache.embed_articles(_articles(("aaa", "Hammer")))
    assert "aaa" in out
    assert out["aaa"].shape == (128,)
    assert cache._tei_embed.call_count == 1  # type: ignore[attr-defined]
