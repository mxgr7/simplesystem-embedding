import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient


from conftest import load_flat_service

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "splade-service"
splade = load_flat_service(
    "splade_service", SERVICE, "backend_pool", "config", "main"
)
backend_pool = splade.backend_pool
config_module = splade.config
main = splade.main


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
        self.api_key = "query-secret"
        self.admin_api_key = ""
        self.max_inputs = 2
        self.max_inflight = 32
        self.request_budget_s = 5
        self.cache_read_timeout_s = 0.1
        self.cache_connections = 4



class StubCache:
    def __init__(self, *args):
        self.calls = 0

    async def mget(self, hashes):
        self.calls += 1
        return [None for _ in hashes]

    async def mset(self, values):
        self.calls += 1

    async def ping(self):
        return True

    async def aclose(self):
        pass


class StubPool:
    def __init__(self, *args):
        self.calls = []

    async def add(self, *args, **kwargs):
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
        return [
            {
                "id": "b1",
                "url": "http://backend",
                "weight": 1,
                "healthy": True,
                "draining": False,
                "inflight": 0,
                "max_client_batch": 8,
                "client_generation": 0,
            }
        ]

    async def encode(self, texts, document=True):
        self.calls.append((texts, document))
        return [
            {"10": 1.23456789, "20": float(index + 1)}
            for index, _ in enumerate(texts)
        ]


@pytest.fixture
def query_client(monkeypatch):
    cache = StubCache()
    pool = StubPool()
    monkeypatch.setattr(main, "Config", StubConfig)
    monkeypatch.setattr(main, "SparseCache", lambda *args: cache)
    monkeypatch.setattr(main, "BackendPool", lambda *args: pool)
    with TestClient(main.app) as client:
        yield client, cache, pool


def auth_post(client, path, inputs):
    return client.post(
        path,
        json={"inputs": inputs},
        headers={"Authorization": "Bearer query-secret"},
    )


def test_embed_query_is_authenticated_and_folds_singleton(query_client):
    client, cache, pool = query_client

    unauthorized = client.post("/embed-query", json={"inputs": "query"})
    response = auth_post(client, "/embed-query", "  GRÖẞE\xa0für KÜHLUNG  ")

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.json() == [{"10": 1.23456789, "20": 1.0}]
    assert pool.calls == [(["groesse fuer kuehlung"], False)]
    assert cache.calls == 0


def test_embed_query_preserves_list_order_and_raw_weights(query_client):
    client, _, pool = query_client

    response = auth_post(client, "/embed-query", ["Zweite", "Erste"])

    assert response.status_code == 200
    assert response.json() == [
        {"10": 1.23456789, "20": 1.0},
        {"10": 1.23456789, "20": 2.0},
    ]
    assert pool.calls == [(["zweite", "erste"], False)]


@pytest.mark.parametrize(
    ("inputs", "status", "detail"),
    [
        ([], 400, "inputs must not be empty"),
        (["one", "two", "three"], 413, "too many inputs"),
        (["valid", " \x00\xa0 "], 400, "inputs must not contain empty queries"),
        (["x" * 4097], 413, "query input is too long"),
    ],
)
def test_embed_query_rejects_invalid_batches(
    query_client, inputs, status, detail
):
    client, cache, pool = query_client

    response = auth_post(client, "/embed-query", inputs)

    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert pool.calls == []
    assert cache.calls == 0


def test_embed_document_path_still_sets_document_true(query_client):
    client, cache, pool = query_client
    fields = [""] * 14
    fields[0] = "Kühlschrank"

    response = auth_post(client, "/embed", "\x00".join(fields))

    assert response.status_code == 200
    assert pool.calls[0][1] is True
    assert cache.calls > 0


def test_embed_query_bypasses_document_admission_limit(query_client):
    client, _, pool = query_client
    main.app.state.inflight = main.app.state.config.max_inflight
    response = auth_post(client, "/embed-query", "query")
    assert response.status_code == 200
    assert pool.calls == [(["query"], False)]


def test_embed_query_queue_is_not_cut_off_by_document_request_budget(query_client):
    client, _, pool = query_client
    main.app.state.config.request_budget_s = 0.001

    async def delayed(texts, document=True):
        await asyncio.sleep(0.02)
        return [{"10": 1.0} for _ in texts]

    pool.encode = delayed
    response = auth_post(client, "/embed-query", "query")

    assert response.status_code == 200


def test_embed_query_rejects_malformed_backend_vector(query_client):
    client, _, pool = query_client

    async def malformed(texts, document=True):
        return [{"bad": -1.0} for _ in texts]

    pool.encode = malformed
    with pytest.raises(ValueError, match="invalid query token"):
        auth_post(client, "/embed-query", "query")


def test_no_healthy_backend_is_a_503_not_a_500(query_client):
    """An unhandled NoHealthyBackendError is a 500 with a traceback per request.

    On the dense side that wrote 268k tracebacks and 2.3 GB of container log over
    one outage, and buried the single line that said what had happened. It is
    also invisible to every alert: `splade_service_requests_total` never counted
    it, `SpladeGatewayTimeouts` watches 504, and `SpladeNoBackendAtAll` only
    fires when the metric disappears entirely. 503 tells the indexer to retry
    rather than dead-letter.
    """
    client, _, pool = query_client

    async def no_backend(texts, document=True):
        raise backend_pool.NoHealthyBackendError("no healthy SPLADE backend available")

    pool.encode = no_backend
    response = auth_post(client, "/embed-query", "query")
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert response.json()["detail"] == "no healthy SPLADE backend available"


class FakeBackend:
    def __init__(self, backend_id, max_client_batch, fail=False):
        self.id = backend_id
        self.max_client_batch = max_client_batch
        self.fail = fail
        self.weight = 1
        self.inflight = 0
        self.healthy = True
        self.draining = False
        self.trial_inflight = False
        self.last_trial_at = float("-inf")
        self.consecutive_failures = 0
        self.calls = []
        self.marks = []

    def mark_success(self, source="request"):
        self.consecutive_failures = 0
        self.marks.append(("success", source))

    def mark_failure(self, exc=None, source="request"):
        self.consecutive_failures += 1
        self.marks.append(("failure", source))

    def trial_due(self, now, interval_s):
        return not self.trial_inflight and (now - self.last_trial_at) >= interval_s

    async def encode(self, texts, document=True):
        self.calls.append((texts, document))
        if self.fail:
            self.fail = False
            raise RuntimeError("temporary failure")
        await asyncio.sleep(0)
        return [{"text": text} for text in texts]


def test_backend_pool_propagates_document_through_chunking_and_retry():
    pool = backend_pool.BackendPool()
    first = FakeBackend("b1", 2, fail=True)
    second = FakeBackend("b2", 2)
    pool.backends = {first.id: first, second.id: second}

    vectors = asyncio.run(pool.encode(["a", "b", "c"], document=False))

    assert vectors == [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    assert first.calls == [(["a", "b"], False), (["c"], False)]
    assert second.calls == [(["a", "b"], False)]


def test_backend_posts_document_flag():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=[{"1": 0.25}])

    backend = backend_pool.Backend("b1", "http://backend", 1, 1, 8, 5, "")
    asyncio.run(backend.client.aclose())
    backend.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://backend",
    )
    try:
        result = asyncio.run(backend.encode(["query"], document=False))
    finally:
        asyncio.run(backend.aclose())

    assert result == [{"1": 0.25}]
    assert requests[0].read()
    assert requests[0].content == b'{"inputs":["query"],"document":false}'
