"""Unit tests for the embedding-service.

Tests the pure-functional pieces (hashing, NUL-split validation, cache
key format, template rendering parity with the indexer) plus a couple
of HTTP-level paths via FastAPI's TestClient with stubbed cache + TEI.

Folder layout note: `embedding-service/` has a hyphen, so it's not a
Python package — same flat-import convention as `search-api/`. Tests
prepend it to `sys.path` to make `import cache`, `import main`, etc.
work.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVICE_DIR = REPO_ROOT / "embedding-service"
sys.path.insert(0, str(SERVICE_DIR))
sys.path.insert(0, str(REPO_ROOT))

import hashing  # noqa: E402
import rendering  # noqa: E402
from indexer.embedding_text import article_to_text as indexer_article_to_text  # noqa: E402


# ---------------------------------------------------------------------------
# Hashing + cache key
# ---------------------------------------------------------------------------

def test_hash_known_input() -> None:
    """Lock the hash to a known fixture so any drift in the algorithm
    (e.g. switching from sha256 to a different family) fails loudly."""
    h = hashing.article_hash("hello")
    assert len(h) == 32
    assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e"  # sha256("hello")[:16].hex()


def test_hash_handles_nul_bytes_in_input() -> None:
    """NUL bytes in the input are part of the hashed bytes (not split)."""
    h1 = hashing.article_hash("a\x00b")
    h2 = hashing.article_hash("ab")
    assert h1 != h2


def test_cache_key_format() -> None:
    assert hashing.cache_key("abcdef") == "tei:v2:abcdef"
    assert hashing.HASH_VERSION == "v2"


# ---------------------------------------------------------------------------
# NUL-split + 8-field validation
# ---------------------------------------------------------------------------

def _make_8field(
    name: str = "Hammer",
    mfr_name: str = "Acme",
    description: str = "A solid hammer.",
    category_paths: str = "Tools > Hand",
    ean: str = "1234567890123",
    article_number: str = "ART-001",
    mfr_article_number: str = "MFR-001",
    mfr_article_type: str = "standard",
) -> str:
    return "\x00".join([
        name, mfr_name, description, category_paths,
        ean, article_number, mfr_article_number, mfr_article_type,
    ])


def test_split_fields_exactly_8() -> None:
    fields = rendering.split_fields(_make_8field())
    assert len(fields) == 8
    assert fields[0] == "Hammer"
    assert fields[7] == "standard"


def test_split_fields_rejects_7() -> None:
    with pytest.raises(ValueError, match="exactly 8"):
        rendering.split_fields("a\x00b\x00c\x00d\x00e\x00f\x00g")


def test_split_fields_rejects_9() -> None:
    with pytest.raises(ValueError, match="exactly 8"):
        rendering.split_fields("a\x00b\x00c\x00d\x00e\x00f\x00g\x00h\x00i")


def test_fields_to_row_positional() -> None:
    fields = ["n", "m", "d", "c", "e", "a", "ma", "mat"]
    row = rendering.fields_to_row(fields)
    assert row["name"] == "n"
    assert row["manufacturer_name"] == "m"
    assert row["description"] == "d"
    assert row["category_paths"] == "c"
    assert row["ean"] == "e"
    assert row["article_number"] == "a"
    assert row["manufacturer_article_number"] == "ma"
    assert row["manufacturer_article_type"] == "mat"


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def test_render_from_nul_produces_template_output() -> None:
    """The rendered text must hit the expected passage shape from
    `configs/data/default.yaml` — `passage: Article Name: ...` with the
    8 conditional fields."""
    text = _make_8field()
    rendered = rendering.render_from_nul(text)
    assert rendered.startswith("passage: Article Name: Hammer")
    assert "EAN: 1234567890123" in rendered
    assert "Article Number: ART-001" in rendered
    assert "Article Number (Manufacturer): MFR-001" in rendered
    assert "Category: Tools > Hand" in rendered
    assert "Article Type: standard" in rendered
    assert "Brand: Acme" in rendered
    assert "Description: A solid hammer." in rendered


def test_render_drops_empty_fields() -> None:
    """A field with empty value should not render its label."""
    text = _make_8field(ean="", article_number="")
    rendered = rendering.render_from_nul(text)
    assert "EAN" not in rendered
    assert "Article Number:" not in rendered
    # But other fields still present
    assert "Brand: Acme" in rendered


def test_render_strips_html_in_description() -> None:
    """The template uses `clean_html: true` (see template.yaml), so HTML
    in the description must be stripped before rendering."""
    text = _make_8field(description="<p>solid <b>hammer</b></p>")
    rendered = rendering.render_from_nul(text)
    assert "<p>" not in rendered
    assert "<b>" not in rendered
    # The cleaned text should still be in the description
    assert "solid" in rendered and "hammer" in rendered


def test_render_parity_with_indexer() -> None:
    """The new service must produce byte-identical rendered text to the
    indexer-side `article_to_text` for the same 8-field row — otherwise
    cache entries written by the indexer and re-read here would mix two
    distributions, or vice versa."""
    text = _make_8field()
    new_render = rendering.render_from_nul(text)
    indexer_render = indexer_article_to_text(rendering.fields_to_row(
        rendering.split_fields(text),
    ))
    assert new_render == indexer_render


# ---------------------------------------------------------------------------
# HTTP handler — uses TestClient with stubbed cache + TEI
# ---------------------------------------------------------------------------

@pytest.fixture
def app_with_stubs(monkeypatch):
    """Build the FastAPI app with the cache + TEI client replaced by
    in-memory stubs so tests don't need a running KVRocks or TEI."""
    # Force imports of the service modules under SERVICE_DIR's sys.path.
    import main
    from cache import EmbeddingCache
    from embed_client import TEIClient

    # Stub cache: in-memory dict, no network.
    class StubCache:
        def __init__(self) -> None:
            self.store: dict[str, bytes] = {}
            from cache import CacheStats
            self.stats = CacheStats()

        async def mget(self, hashes):
            return [self.store.get(h) for h in hashes]

        async def mset(self, hash_to_bytes):
            self.store.update(hash_to_bytes)

        async def aclose(self):
            pass

    # Stub TEI: deterministic per-text fp16 vector.
    class StubTEI:
        def __init__(self) -> None:
            self.call_count = 0
            self.texts_received: list[str] = []
            self.semaphore_wait_total_s = 0.0

        async def embed(self, texts, *, truncate=True):
            import hashlib
            self.call_count += 1
            self.texts_received.extend(texts)
            out = np.zeros((len(texts), 128), dtype=np.float16)
            for i, t in enumerate(texts):
                # Seed from a hash of the whole text so distinct inputs
                # → distinct vectors (rendered texts share a common
                # prefix, so prefix-only seeding collides).
                seed = int.from_bytes(
                    hashlib.sha256(t.encode("utf-8")).digest()[:8], "big",
                )
                rng = np.random.default_rng(seed)
                out[i] = rng.standard_normal(128).astype(np.float16)
            return out

        async def aclose(self):
            pass

    monkeypatch.setattr(main, "EmbeddingCache",
                        lambda *a, **kw: StubCache())
    monkeypatch.setattr(main, "TEIClient",
                        lambda *a, **kw: StubTEI())

    with TestClient(main.app) as client:
        yield client, main.app


def test_embed_string_input_returns_vector(app_with_stubs):
    client, app = app_with_stubs
    text = _make_8field()
    r = client.post("/embed", json={"inputs": text})
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data, list) and len(data) == 1
    assert isinstance(data[0], list) and len(data[0]) == 128


def test_embed_list_input_returns_vectors_in_order(app_with_stubs):
    client, app = app_with_stubs
    t1 = _make_8field(name="A")
    t2 = _make_8field(name="B")
    r = client.post("/embed", json={"inputs": [t1, t2]})
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data) == 2
    # Different inputs → different vectors (sanity).
    assert data[0] != data[1]


def test_embed_second_call_hits_cache(app_with_stubs):
    client, app = app_with_stubs
    text = _make_8field()
    r1 = client.post("/embed", json={"inputs": text})
    # BackgroundTasks for cache write runs after the response under
    # TestClient — wait one event-loop turn for it to drain.
    asyncio.run(asyncio.sleep(0))
    r2 = client.post("/embed", json={"inputs": text})
    assert r1.status_code == 200 and r2.status_code == 200
    # Same hash → same vector (cache hit means we deserialised the stored
    # fp16 bytes back to a list — value must equal the original).
    assert r1.json()[0] == r2.json()[0]


def test_embed_rejects_wrong_field_count(app_with_stubs):
    client, _ = app_with_stubs
    r = client.post("/embed", json={"inputs": "only\x00seven\x00fields\x00here\x00is\x00not\x00enough"})
    assert r.status_code == 400
    assert "8 NUL-separated fields" in r.json()["detail"]


def test_embed_rejects_empty_list(app_with_stubs):
    client, _ = app_with_stubs
    r = client.post("/embed", json={"inputs": []})
    assert r.status_code == 400


def test_embed_rejects_oversize_request(app_with_stubs, monkeypatch):
    """Per-request input cap → 413."""
    client, app = app_with_stubs
    # Lower the cap so we don't have to build 257 fixtures.
    app.state.cfg = app.state.cfg.__class__(
        **{**app.state.cfg.__dict__, "max_inputs_per_request": 2}
    )
    r = client.post("/embed", json={
        "inputs": [_make_8field(), _make_8field(), _make_8field()],
    })
    assert r.status_code == 413
