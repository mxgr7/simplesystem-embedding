import asyncio
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "splade-service"
sys.path.insert(0, str(SERVICE))

codec = importlib.import_module("codec")
constants = importlib.import_module("constants")
hashing = importlib.import_module("hashing")
rendering = importlib.import_module("rendering")


def make_input(**overrides):
    values = {
        "name": "Kühlschrank Pro",
        "manufacturer_name": "Müller",
        "description": "ignored description",
        "category_paths": "Alt > Pfad",
        "ean": "4000123456789",
        "article_number": "ART-1",
        "manufacturer_article_number": "MFR-1",
        "manufacturer_article_type": "Gerät",
        "customer_artnos_text": "KÜ-42",
        "vendor_text": "Händler GmbH",
        "category_leaf_text": "Küche > Kühlgeräte",
        "s2class_text": "Kühlgerät",
        "keywords_text": "kühlen groß",
        "features_text": "Größe: 60 cm.",
    }
    values.update(overrides)
    return "\x00".join(values[name] for name in constants.FIELD_ORDER)


def test_render_matches_training_fold_order_and_description_free_contract():
    rendered = rendering.render_from_nul(make_input())
    assert rendered.startswith("Article Name: kuehlschrank pro")
    assert "Brand: mueller" in rendered
    assert "Category: kueche > kuehlgeraete" in rendered
    assert "Features: groesse: 60 cm." in rendered
    assert "Description:" not in rendered
    # The training builder folds fields before rendering, not static labels.
    assert "Article Name:" in rendered


def test_description_is_canonicalized_out_of_cache_identity():
    left = rendering.canonical_input(make_input(description="one"))
    right = rendering.canonical_input(make_input(description="two"))
    assert left == right
    assert hashing.input_hash(left) == hashing.input_hash(right)


def test_field_count_is_strict():
    with pytest.raises(ValueError, match="14 NUL-separated"):
        rendering.render_from_nul("too\x00short")


def test_sparse_codec_prunes_and_round_trips_fp16():
    vector = {str(index): float(index + 1) / 100 for index in range(300)}
    packed = codec.pack_sparse(vector)
    restored = codec.unpack_sparse(packed)
    assert len(restored) == constants.TOP_K
    assert "299" in restored
    assert "0" not in restored
    assert restored["299"] == float(np.float16(3.0))


def test_sparse_codec_rejects_malformed_values():
    with pytest.raises(ValueError):
        codec.unpack_sparse(b"\x01")
    with pytest.raises(ValueError):
        codec.unpack_sparse(b"\xff\xff")


def test_sparse_batch_codec_round_trips_and_rejects_trailing_bytes():
    vectors = [{"2": 1.25, "1": 0.5}, {}, {"31000": 0.25}]
    packed = codec.pack_sparse_batch(vectors)
    assert codec.unpack_sparse_batch(packed) == [
        codec.unpack_sparse(codec.pack_sparse(vector)) for vector in vectors
    ]
    with pytest.raises(ValueError, match="trailing"):
        codec.unpack_sparse_batch(packed + b"x")


def test_sparse_array_codec_filters_fp16_underflow_and_sorts_ids():
    packed = codec.pack_sparse_arrays(
        np.array([20, 10, 30]), np.array([0.5, 1.0, 1e-12])
    )
    assert codec.unpack_sparse(packed) == {"10": 1.0, "20": 0.5}
    with pytest.raises(ValueError, match="invalid token"):
        codec.pack_sparse_arrays(np.array([constants.VOCAB_SIZE]), np.array([1.0]))


def test_backend_tokenizer_uses_wordpiece_fallback_on_missing_dependency(
    monkeypatch, tmp_path
):
    backend = importlib.import_module("backend")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ntest\n")
    config = tmp_path / "tokenizer_config.json"
    config.write_text(json.dumps({"do_lower_case": True}))

    def download(_model_name, filename, cache_dir):
        assert cache_dir == str(tmp_path)
        return str(vocab if filename == "vocab.txt" else config)

    def missing_dependency(*_args, **_kwargs):
        raise ImportError("protobuf is unavailable")

    monkeypatch.setattr(
        backend.AutoTokenizer, "from_pretrained", missing_dependency
    )
    monkeypatch.setattr(backend, "hf_hub_download", download)

    tokenizer = backend.load_tokenizer("example/model", str(tmp_path))

    assert tokenizer("TEST")["input_ids"] == [2, 5, 3]


class StubConfig:
    kvrocks_url = "redis://stub"
    backend_urls = ["http://backend"]
    backend_api_key = ""
    api_key = ""
    admin_api_key = ""
    max_inputs = 256
    max_inflight = 32
    request_budget_s = 5
    cache_read_timeout_s = 0.1
    cache_connections = 4
    probe_interval_s = 5


class StubCache:
    def __init__(self, *args):
        self.values = {}

    async def mget(self, hashes):
        return [self.values.get(value) for value in hashes]

    async def mset(self, values):
        self.values.update(values)

    async def ping(self):
        return True

    async def aclose(self):
        pass


class StubPool:
    def __init__(self, *args):
        self.calls = 0

    async def add(self, *args, **kwargs):
        return {"id": "b1"}

    def start(self):
        pass

    async def aclose(self):
        pass

    def ready(self):
        return True

    def snapshots(self):
        return [{"id": "b1", "healthy": True}]

    async def encode(self, texts, document=True):
        self.calls += 1
        return [{"10": 1.25, "20": 0.5} for _ in texts]


@pytest.fixture
def service_client(monkeypatch):
    main = importlib.import_module("main")
    cache = StubCache()
    pool = StubPool()
    monkeypatch.setattr(main, "Config", StubConfig)
    monkeypatch.setattr(main, "SparseCache", lambda *args: cache)
    monkeypatch.setattr(main, "BackendPool", lambda *args: pool)
    with TestClient(main.app) as client:
        yield client, cache, pool


def test_embed_miss_then_cache_hit(service_client):
    client, cache, pool = service_client
    first = client.post("/embed", json={"inputs": make_input()})
    second = client.post("/embed", json={"inputs": make_input()})
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert first.json() == [{"10": 1.25, "20": 0.5}]
    assert pool.calls == 1
    assert len(cache.values) == 1


def test_embed_deduplicates_same_miss_within_request(service_client):
    client, _, pool = service_client
    value = make_input()
    response = client.post("/embed", json={"inputs": [value, value]})
    assert response.status_code == 200
    assert response.json()[0] == response.json()[1]
    assert pool.calls == 1


def test_readyz_checks_cache_and_backend(service_client):
    client, _, _ = service_client
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_admin_lists_backends(service_client):
    client, _, _ = service_client
    response = client.get("/admin/backends")
    assert response.status_code == 200
    assert response.json() == [{"id": "b1", "healthy": True}]
