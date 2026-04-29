"""End-to-end smoke for `indexer.bulk.run_bulk_indexer`.

The DuckDB / TEI cache / Milvus pieces have unit tests of their own
(`test_duckdb_*`, `test_tei_cache.py`); this test wires them together
through the production orchestrator with the externals mocked. Real
DuckDB processes the local-cached S3 shards; Milvus + TEI + Redis are
in-process fakes so the test runs offline in CI.

Validates:
  - Both Milvus collections receive upsert calls.
  - Each article row carries the fp16 `offer_embedding` (128-d) the
    cache returned.
  - Each offer row carries `article_hash` and `_placeholder_vector`.
  - Article row count <= offer row count (hash dedup).
  - All article hashes referenced from offer rows exist in the article
    upsert payload (no dangling `article_hash` join keys).
  - Cache hits + misses sum to the article count exactly once.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.bulk import run_bulk_indexer  # noqa: E402

CACHE_ROOT = Path.home() / "s3-cache"
SMOKE_OFFERS = CACHE_ROOT / "offers/atlas-fkxrb3-shard-0.0.json.gz"

pytestmark = pytest.mark.skipif(
    not SMOKE_OFFERS.exists(),
    reason="s3-cache shard 0.0 missing (run scripts/dump_s3_sample.py prerequisites)",
)


# ---------- fakes ----------------------------------------------------------

class _FakeRedis:
    """Minimal Redis API surface used by `TEICache`: mget, set, pipeline."""
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def mget(self, keys: list[str]) -> list[bytes | None]:
        return [self.store.get(k) for k in keys]

    def set(self, key: str, value: bytes) -> bool:
        self.store[key] = value
        return True

    def ping(self) -> bool:
        return True

    def pipeline(self, transaction: bool = True) -> "_FakePipeline":
        return _FakePipeline(self)


class _FakePipeline:
    def __init__(self, parent: _FakeRedis) -> None:
        self.parent = parent
        self.q: list[tuple[str, bytes]] = []

    def set(self, key: str, value: bytes) -> "_FakePipeline":
        self.q.append((key, value))
        return self

    def execute(self) -> list[bool]:
        for k, v in self.q:
            self.parent.set(k, v)
        out = [True] * len(self.q)
        self.q.clear()
        return out


class _RecordingMilvus:
    """Records every `upsert` call's data payload for later assertion.
    `has_collection` returns True for any name the test pre-registers
    so the orchestrator's existence check passes."""
    def __init__(self, *, known_collections: list[str]) -> None:
        self._collections = set(known_collections)
        self.upserts: dict[str, list[dict]] = {c: [] for c in known_collections}

    def has_collection(self, name: str) -> bool:
        return name in self._collections

    def upsert(self, *, collection_name: str, data: list[dict]) -> dict[str, Any]:
        self.upserts[collection_name].extend(data)
        return {"insert_count": len(data), "ids": [r.get("article_hash") or r.get("id") for r in data]}


class _DeterministicTEI:
    """Stand-in for `TEICache._tei_embed`. Deterministic 128-d fp32
    vector seeded by the SHA-256 of the input text — same shape as
    `indexer.test_loader.stub_vector`. Patched onto a real TEICache
    instance per test to avoid touching the network."""
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), 128), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(t.encode("utf-8")[:8].ljust(8, b"\x00"), "big")
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(128).astype(np.float32)
        return out


# ---------- e2e -----------------------------------------------------------

def test_run_bulk_indexer_end_to_end(tmp_path: Path) -> None:
    """Full path: read 1 offers shard + all pricings/markers/cans
    from local cache → DuckDB JOIN+aggregate → fake-TEI embed →
    fake-Milvus upsert. Assert payload shape on both collections."""

    fake_redis = _FakeRedis()
    fake_milvus = _RecordingMilvus(
        known_collections=["articles_smoke", "offers_smoke"],
    )

    with mock.patch("indexer.bulk.MilvusClient", return_value=fake_milvus), \
         mock.patch("indexer.bulk.redis.Redis.from_url", return_value=fake_redis), \
         mock.patch("indexer.tei_cache.TEICache._tei_embed", _DeterministicTEI()):
        stats = run_bulk_indexer(
            offers_glob=str(CACHE_ROOT / "offers/atlas-fkxrb3-shard-0.0.json.gz"),
            pricings_glob=str(CACHE_ROOT / "pricings/atlas-*.json.gz"),
            markers_glob=str(CACHE_ROOT / "coreArticleMarkers/atlas-*.json.gz"),
            cans_glob=str(CACHE_ROOT / "customerArticleNumbers/atlas-*.json.gz"),
            milvus_uri="memory://",
            articles_collection="articles_smoke",
            offers_collection="offers_smoke",
            tei_url="http://stub",
            redis_url="redis://stub",
            article_batch_size=500,
            offer_batch_size=2000,
            tei_batch_size=64,
        )

    # ---- aggregate counts
    assert stats.raw_offer_count > 1000
    assert stats.article_count > 0
    assert stats.offer_row_count > 0
    assert stats.article_count <= stats.offer_row_count, (
        f"more articles ({stats.article_count}) than offers ({stats.offer_row_count}) — dedup broken"
    )
    assert stats.tei.misses + stats.tei.hits == stats.article_count, (
        "every article must come from either the cache or a TEI miss exactly once"
    )

    article_payload = fake_milvus.upserts["articles_smoke"]
    offer_payload = fake_milvus.upserts["offers_smoke"]
    assert len(article_payload) == stats.article_count
    assert len(offer_payload) == stats.offer_row_count

    # ---- article-row shape
    art = article_payload[0]
    assert isinstance(art["article_hash"], str) and len(art["article_hash"]) == 32
    assert "name" in art and "manufacturerName" in art
    assert "text_codes" in art
    emb = art["offer_embedding"]
    assert isinstance(emb, np.ndarray) and emb.dtype == np.float16 and emb.shape == (128,)
    # Per-currency envelope columns present.
    assert "eur_price_min" in art and "eur_price_max" in art

    # ---- offer-row shape
    off = offer_payload[0]
    assert "id" in off and ":" in off["id"]  # `{vendor_uuid}:{b64url(article_number)}`
    assert isinstance(off["article_hash"], str) and len(off["article_hash"]) == 32
    assert off["_placeholder_vector"] == [0.0, 0.0]
    assert "prices" in off
    # F8 envelope columns present on the offer row.
    assert "price_list_ids" in off and "currencies" in off
    assert "eur_price_min" in off

    # ---- referential integrity: every offer's article_hash must exist
    # in the upserted articles. Catches a JOIN-key drift between the
    # two SQL streams.
    article_hashes = {a["article_hash"] for a in article_payload}
    offer_hashes = {o["article_hash"] for o in offer_payload}
    dangling = offer_hashes - article_hashes
    assert not dangling, (
        f"{len(dangling)} offer rows reference article_hashes "
        f"missing from articles_v* upsert (sample: {list(dangling)[:3]})"
    )


def test_rerun_uses_redis_cache(tmp_path: Path) -> None:
    """Two consecutive runs sharing one Redis instance: the second
    must hit the cache for every article and skip TEI entirely. This
    is the prod re-run safety net — if the indexer crashes mid-run,
    restarting must not re-embed everything."""
    fake_redis = _FakeRedis()
    deterministic_tei = _DeterministicTEI()

    fake_milvus_1 = _RecordingMilvus(known_collections=["articles_smoke", "offers_smoke"])
    with mock.patch("indexer.bulk.MilvusClient", return_value=fake_milvus_1), \
         mock.patch("indexer.bulk.redis.Redis.from_url", return_value=fake_redis), \
         mock.patch("indexer.tei_cache.TEICache._tei_embed", deterministic_tei):
        stats_1 = run_bulk_indexer(
            offers_glob=str(CACHE_ROOT / "offers/atlas-fkxrb3-shard-0.0.json.gz"),
            pricings_glob=str(CACHE_ROOT / "pricings/atlas-*.json.gz"),
            markers_glob=str(CACHE_ROOT / "coreArticleMarkers/atlas-*.json.gz"),
            cans_glob=str(CACHE_ROOT / "customerArticleNumbers/atlas-*.json.gz"),
            milvus_uri="memory://",
            articles_collection="articles_smoke",
            offers_collection="offers_smoke",
            tei_url="http://stub",
            redis_url="redis://stub",
            article_batch_size=500,
            offer_batch_size=2000,
        )

    assert stats_1.tei.hits == 0
    assert stats_1.tei.misses == stats_1.article_count

    tei_calls_after_first = deterministic_tei.calls

    fake_milvus_2 = _RecordingMilvus(known_collections=["articles_smoke", "offers_smoke"])
    with mock.patch("indexer.bulk.MilvusClient", return_value=fake_milvus_2), \
         mock.patch("indexer.bulk.redis.Redis.from_url", return_value=fake_redis), \
         mock.patch("indexer.tei_cache.TEICache._tei_embed", deterministic_tei):
        stats_2 = run_bulk_indexer(
            offers_glob=str(CACHE_ROOT / "offers/atlas-fkxrb3-shard-0.0.json.gz"),
            pricings_glob=str(CACHE_ROOT / "pricings/atlas-*.json.gz"),
            markers_glob=str(CACHE_ROOT / "coreArticleMarkers/atlas-*.json.gz"),
            cans_glob=str(CACHE_ROOT / "customerArticleNumbers/atlas-*.json.gz"),
            milvus_uri="memory://",
            articles_collection="articles_smoke",
            offers_collection="offers_smoke",
            tei_url="http://stub",
            redis_url="redis://stub",
            article_batch_size=500,
            offer_batch_size=2000,
        )

    assert stats_2.article_count == stats_1.article_count
    assert stats_2.tei.hits == stats_2.article_count, (
        "second run should be 100% cache hits"
    )
    assert stats_2.tei.misses == 0
    assert deterministic_tei.calls == tei_calls_after_first, (
        "TEI must not be called on rerun"
    )
