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
config_module = importlib.import_module("config")
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


class StubConfig(config_module.Config):
    """The real Config with the test's overrides on top.

    Subclassed rather than hand-listed: a flat stub goes stale silently every
    time a knob is added to Config, and this file spent a release erroring on a
    missing `backend_pool_concurrency` for exactly that reason.
    """

    def __init__(self):
        super().__init__()
        self.kvrocks_url = "redis://stub"
        self.backend_urls = ["http://backend"]
        self.backend_api_key = ""
        self.api_key = ""
        self.admin_api_key = ""
        self.max_inputs = 256
        self.max_inflight = 32
        self.request_budget_s = 5
        self.cache_read_timeout_s = 0.1
        self.cache_connections = 4



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
        self.healthy = True
        self.added = []

    async def add(self, *args, **kwargs):
        self.added.append((args, kwargs))
        return {"id": "b1"}

    def start(self):
        pass

    async def aclose(self):
        pass

    def ready(self):
        return True

    def stats(self):
        # The collector reads probe-loop liveness off this. A stub narrower than
        # the thing it stands in for is how a scrape-time AttributeError reaches
        # production green.
        return {"probe_loop_last_iteration_at": 1000.0, "probe_loop_errors": 0}

    def snapshots(self):
        # Mirrors BackendConnection.snapshot()'s real key set -- the metrics collector reads
        # url/draining off it, and a stub that is narrower than the thing it stands in for is
        # how a scrape-time KeyError reaches production green.
        return [
            {
                "id": "b1",
                "url": "http://backend-1:8138",
                "weight": 1.0,
                "healthy": self.healthy,
                "draining": False,
                "inflight": 0,
                "max_client_batch": 32,
            }
        ]

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


def test_startup_registers_backends_with_the_configured_batch_and_concurrency(
    service_client,
):
    """A restart must not quietly undo the pool's tuning.

    BackendPool.add's defaults (max_client_batch 8, max_concurrency 1) are a fallback,
    not an operating point: encode() chunks by the min max_client_batch across
    non-draining backends, so 8 splits a 128-input indexer batch into 16 chunks
    serialized behind one semaphore. On the T4 that measured ~13 s per request with
    inflight pinned at MAX_INFLIGHT and ~40% of requests shed as 429. Correcting it
    through POST /admin/backends works but dies with the process, so startup has to
    carry it.

    `require_verify=False` belongs to the same argument: a backend that is down
    or still loading its checkpoint at boot must be registered unhealthy and left
    to the probe loop, not allowed to abort lifespan and restart-loop the
    frontend. POST /admin/backends keeps the strict form.
    """
    _, _, pool = service_client
    assert [kwargs for _, kwargs in pool.added] == [
        {
            "max_concurrency": 3,
            "max_client_batch": 64,
            "api_key": "",
            "require_verify": False,
        }
    ]


def test_config_reads_pool_shape_from_the_environment(monkeypatch):
    config_module = importlib.import_module("config")
    monkeypatch.setenv("BACKEND_MAX_CLIENT_BATCH", "16")
    monkeypatch.setenv("BACKEND_POOL_CONCURRENCY", "2")
    tuned = config_module.Config()
    assert (tuned.backend_max_client_batch, tuned.backend_pool_concurrency) == (16, 2)

    monkeypatch.delenv("BACKEND_MAX_CLIENT_BATCH")
    monkeypatch.delenv("BACKEND_POOL_CONCURRENCY")
    default = config_module.Config()
    assert (default.backend_max_client_batch, default.backend_pool_concurrency) == (64, 3)


def test_admin_lists_backends(service_client):
    client, _, pool = service_client
    response = client.get("/admin/backends")
    assert response.status_code == 200
    assert response.json() == pool.snapshots()


def _metric_lines(client, name):
    body = client.get("/metrics").text
    return [line for line in body.splitlines() if line.startswith(name)]


def test_backend_healthy_metric_tracks_the_pool(service_client):
    """The gauge MXG-115 alerts on. It has to read the pool at scrape time, not at startup."""
    client, _, pool = service_client
    assert _metric_lines(client, "splade_service_backend_healthy{") == [
        'splade_service_backend_healthy{backend="b1",url="http://backend-1:8138"} 1.0'
    ]
    assert _metric_lines(client, "splade_service_backend_draining{") == [
        'splade_service_backend_draining{backend="b1",url="http://backend-1:8138"} 0.0'
    ]

    pool.healthy = False
    assert _metric_lines(client, "splade_service_backend_healthy{") == [
        'splade_service_backend_healthy{backend="b1",url="http://backend-1:8138"} 0.0'
    ]


def test_probe_loop_liveness_is_scraped_off_the_pool(service_client):
    """The one signal a frozen probe loop cannot fake.

    `splade_service_backend_healthy` above is written only BY the probe loop, so
    a loop that dies leaves it frozen at its last value and the outage reads as
    steady state -- a scrape-time collector re-exports the stale value just as
    faithfully as a push gauge would. `SpladeProbeLoopStalled` alerts on the age
    of this stamp instead.
    """
    client, _, _ = service_client
    assert _metric_lines(
        client, "splade_service_probe_loop_last_iteration_timestamp"
    ) == ["splade_service_probe_loop_last_iteration_timestamp 1000.0"]
    assert _metric_lines(client, "splade_service_probe_loop_errors_total") == [
        "splade_service_probe_loop_errors_total 0.0"
    ]
    assert _metric_lines(client, "splade_service_backend_client_generation{") == [
        'splade_service_backend_client_generation{backend="b1",url="http://backend-1:8138"} 0.0'
    ]


def test_backend_healthy_metric_survives_a_short_snapshot(service_client):
    """A collector that raises turns /metrics into a 500, i.e. the whole job reads as down."""
    client, _, pool = service_client
    pool.snapshots = lambda: [{"id": "b2"}]
    response = client.get("/metrics")
    assert response.status_code == 200
    assert 'splade_service_backend_healthy{backend="b2",url=""} 0.0' in response.text


def test_backend_health_collector_unregisters_on_shutdown(monkeypatch):
    """Two sequential app lifespans must not collide on the default registry."""
    from prometheus_client import REGISTRY

    main = importlib.import_module("main")
    monkeypatch.setattr(main, "Config", StubConfig)
    monkeypatch.setattr(main, "SparseCache", lambda *args: StubCache())
    monkeypatch.setattr(main, "BackendPool", lambda *args: StubPool())

    before = len(REGISTRY._collector_to_names)
    for _ in range(2):
        with TestClient(main.app):
            pass
    assert len(REGISTRY._collector_to_names) == before
