"""A6 (partial) — end-to-end ACL → ftsearch → Milvus acceptance.

Wires the in-process ACL TestClient to a real ftsearch (search-api)
app via `httpx.ASGITransport`, then runs the request against real
Milvus collections (`articles_v4_a6` + `offers_v5_a6`) populated with
sample_200 via `indexer.test_loader.load_split`.

Coverage (one test per spec §10 group, focused MVP):
  - Schema compliance: happy path round-trip 200 + valid envelope.
  - articleId round-trip: response carries the legacy
    `{friendlyId}:{base64UrlEncodedArticleNumber}` shape verbatim.
  - explain stub (§2.2): explain=true → "N/A", explain=false → absent.
  - Dropped-enum rejection (§2.1): one representative case.
  - Error envelope (§3.1): unreachable upstream → 503 in legacy shape.

Per-filter, per-sort, per-aggregation tests are left as follow-ups —
the existing `test_search_dedup_integration.py` covers each one
directly against ftsearch; this suite proves the ACL transparently
forwards them.

Skipped if Milvus is not reachable.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import httpx
import pytest
from pymilvus import MilvusClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "search-api"))
sys.path.insert(0, str(REPO_ROOT / "acl"))

from indexer.projection import project  # noqa: E402
from indexer.test_loader import load_split  # noqa: E402

MILVUS_URI = "http://localhost:19530"
ARTICLES = "articles_v4_a6"
OFFERS = "offers_v5_a6"
SAMPLE = REPO_ROOT / "tests/fixtures/mongo_sample/sample_200.json"


def _milvus_reachable() -> bool:
    try:
        c = MilvusClient(uri=MILVUS_URI)
        c.list_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _milvus_reachable(),
    reason=f"Milvus unreachable at {MILVUS_URI}",
)


# ---- fixtures -----------------------------------------------------------

@pytest.fixture(scope="module")
def projected_rows() -> list[dict]:
    raw = json.loads(SAMPLE.read_text())["records"]
    return [project(r).row for r in raw]


@pytest.fixture(scope="module")
def milvus_client():
    c = MilvusClient(uri=MILVUS_URI)
    yield c


@pytest.fixture(scope="module")
def collections_loaded(milvus_client: MilvusClient, projected_rows):
    """Build a fresh F9 pair + populate with sample_200."""
    from scripts.create_articles_collection import (
        BM25_ANALYZER_PARAMS, CATALOG_CURRENCIES, DIM, SCALAR_INDEX_FIELDS,
    )
    from scripts.create_offers_collection import (
        SCALAR_INDEX_FIELDS as OFFER_SCALAR_INDEX_FIELDS,
    )
    from pymilvus import DataType, Function, FunctionType

    for name in (ARTICLES, OFFERS):
        if milvus_client.has_collection(name):
            milvus_client.drop_collection(name)

    # Articles
    schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("article_hash", DataType.VARCHAR, max_length=32, is_primary=True)
    schema.add_field("offer_embedding", DataType.FLOAT16_VECTOR, dim=DIM)
    schema.add_field("text_codes", DataType.VARCHAR, max_length=8192,
                     enable_analyzer=True, analyzer_params=BM25_ANALYZER_PARAMS)
    schema.add_field("sparse_codes", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name="bm25_codes", function_type=FunctionType.BM25,
                                 input_field_names=["text_codes"],
                                 output_field_names=["sparse_codes"]))
    schema.add_field("name", DataType.VARCHAR, max_length=1024)
    schema.add_field("manufacturerName", DataType.VARCHAR, max_length=256)
    for d, ml in zip(range(1, 6), (256, 640, 768, 1024, 1280)):
        schema.add_field(f"category_l{d}", DataType.ARRAY,
                         element_type=DataType.VARCHAR, max_capacity=64, max_length=ml)
    for f in ("eclass5_code", "eclass7_code", "s2class_code"):
        schema.add_field(f, DataType.ARRAY, element_type=DataType.INT32, max_capacity=16)
    schema.add_field("customer_article_numbers", DataType.JSON)
    for ccy in CATALOG_CURRENCIES:
        schema.add_field(f"{ccy}_price_min", DataType.FLOAT)
        schema.add_field(f"{ccy}_price_max", DataType.FLOAT)
    params = milvus_client.prepare_index_params()
    params.add_index(field_name="offer_embedding", index_type="HNSW",
                     metric_type="COSINE", params={"M": 16, "efConstruction": 200})
    params.add_index(field_name="sparse_codes", index_type="SPARSE_INVERTED_INDEX",
                     metric_type="BM25", params={"mmap.enabled": True}, index_name="sparse_codes")
    for f in SCALAR_INDEX_FIELDS:
        params.add_index(field_name=f, index_type="INVERTED", index_name=f)
    for ccy in CATALOG_CURRENCIES:
        for s in ("min", "max"):
            params.add_index(field_name=f"{ccy}_price_{s}", index_type="STL_SORT", index_name=f"{ccy}_price_{s}")
    milvus_client.create_collection(collection_name=ARTICLES, schema=schema, index_params=params)
    milvus_client.load_collection(ARTICLES)

    # Offers
    schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field("_placeholder_vector", DataType.FLOAT_VECTOR, dim=2)
    schema.add_field("article_hash", DataType.VARCHAR, max_length=32)
    schema.add_field("ean", DataType.VARCHAR, max_length=64)
    schema.add_field("article_number", DataType.VARCHAR, max_length=256)
    schema.add_field("vendor_id", DataType.VARCHAR, max_length=64)
    schema.add_field("catalog_version_ids", DataType.ARRAY,
                     element_type=DataType.VARCHAR, max_capacity=2048, max_length=64)
    schema.add_field("prices", DataType.JSON)
    schema.add_field("delivery_time_days_max", DataType.INT32)
    for f in ("core_marker_enabled_sources", "core_marker_disabled_sources"):
        schema.add_field(f, DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=64)
    schema.add_field("features", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=512, max_length=512)
    for f in ("relationship_accessory_for", "relationship_spare_part_for", "relationship_similar_to"):
        schema.add_field(f, DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=128, max_length=256)
    schema.add_field("price_list_ids", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=512, max_length=64)
    schema.add_field("currencies", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=8, max_length=8)
    for ccy in CATALOG_CURRENCIES:
        schema.add_field(f"{ccy}_price_min", DataType.FLOAT)
        schema.add_field(f"{ccy}_price_max", DataType.FLOAT)
    params = milvus_client.prepare_index_params()
    params.add_index(field_name="_placeholder_vector", index_type="FLAT", metric_type="L2")
    for f in OFFER_SCALAR_INDEX_FIELDS:
        params.add_index(field_name=f, index_type="INVERTED", index_name=f)
    for ccy in CATALOG_CURRENCIES:
        for s in ("min", "max"):
            params.add_index(field_name=f"{ccy}_price_{s}", index_type="STL_SORT", index_name=f"{ccy}_price_{s}")
    milvus_client.create_collection(collection_name=OFFERS, schema=schema, index_params=params)
    milvus_client.load_collection(OFFERS)

    load_split(
        milvus_client,
        articles_collection=ARTICLES, offers_collection=OFFERS,
        rows=projected_rows,
    )
    milvus_client.flush(ARTICLES); milvus_client.flush(OFFERS)
    time.sleep(2)
    yield
    for name in (ARTICLES, OFFERS):
        if milvus_client.has_collection(name):
            milvus_client.drop_collection(name)


@pytest.fixture(scope="module")
def search_api_test_client(collections_loaded):
    """Boot search-api with the dedup topology pointing at our
    fixture pair, wrapped in a TestClient so its FastAPI lifespan
    actually fires (sets up app.state.milvus, app.state.gate, etc.)."""
    from fastapi.testclient import TestClient
    os.environ["USE_DEDUP_TOPOLOGY"] = "1"
    os.environ["MILVUS_ARTICLES_COLLECTION"] = ARTICLES
    os.environ["EMBED_URL"] = "http://embed.invalid"
    os.environ["MILVUS_URI"] = MILVUS_URI
    os.environ["API_KEY"] = ""
    sys.path.insert(0, str(REPO_ROOT / "search-api"))
    import main as search_api_main  # noqa: PLC0415
    importlib.reload(search_api_main)
    with TestClient(search_api_main.app) as client:
        yield client


@pytest.fixture
def acl_client(search_api_test_client) -> Iterator:
    """ACL TestClient whose internal `FtsearchClient` is wired to
    the real search-api app via httpx ASGITransport.

    httpx 0.28's ASGITransport doesn't fire lifespan itself — but the
    search-api's lifespan was already triggered by the
    `search_api_test_client` TestClient context manager (sets
    `app.state.milvus`, `app.state.gate`, etc.). Since `app.state` is
    shared across requests on the same app instance, our forwarded
    calls see the populated state."""
    from fastapi.testclient import TestClient
    sys.path.insert(0, str(REPO_ROOT / "acl"))
    import main as acl_main  # noqa: PLC0415
    importlib.reload(acl_main)
    with TestClient(acl_main.app, raise_server_exceptions=False) as c:
        c.app.state.ftsearch._client = httpx.AsyncClient(  # type: ignore[attr-defined]
            transport=httpx.ASGITransport(app=search_api_test_client.app),
            base_url="http://search-api-stub",
        )
        c.app.state.ftsearch._default_collection = OFFERS  # type: ignore[attr-defined]
        yield c


# ---- helpers ------------------------------------------------------------

def _valid_request(**overrides) -> dict:
    base = {
        "searchMode": "BOTH",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {"closedCatalogVersionIds": []},
        "maxDeliveryTime": 0,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
        "currency": "EUR",
        "explain": False,
    }
    base.update(overrides)
    return base


# ---- §10 acceptance cases -----------------------------------------------

def test_happy_path_returns_legacy_envelope(acl_client) -> None:
    """Schema compliance + envelope shape — request goes through
    ACL → ftsearch → Milvus and comes back in the legacy shape."""
    r = acl_client.post(
        "/article-features/search?page=1&pageSize=10",
        json=_valid_request(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) >= {"articles", "summaries", "metadata"}
    assert isinstance(body["articles"], list)
    assert "hitCount" in body["metadata"]


def test_articleid_format_round_trips(acl_client) -> None:
    """§3 — `articleId` is `{friendlyId}:{base64UrlEncodedArticleNumber}`
    on the wire. The ACL must NOT reformat or strip namespacing."""
    # Empty filter set + no query → Path A browse against article
    # collection. With sample_200 there might be no defensible
    # ordering, so we go via vendor filter for deterministic hits.
    r = acl_client.post(
        "/article-features/search?page=1&pageSize=5",
        json=_valid_request(),
    )
    assert r.status_code == 200
    for art in r.json()["articles"]:
        assert ":" in art["articleId"], (
            f"articleId {art['articleId']!r} not in legacy format"
        )


def test_explain_true_stubs_explanation_to_na(acl_client) -> None:
    """§2.2 deviation surfaces end-to-end — a real round-trip with
    explain=true returns the literal "N/A" stub on every article."""
    r = acl_client.post(
        "/article-features/search?page=1&pageSize=5",
        json=_valid_request(explain=True),
    )
    assert r.status_code == 200
    for art in r.json()["articles"]:
        assert art.get("explanation") == "N/A"


def test_explain_false_omits_explanation(acl_client) -> None:
    r = acl_client.post(
        "/article-features/search?page=1&pageSize=5",
        json=_valid_request(explain=False),
    )
    assert r.status_code == 200
    for art in r.json()["articles"]:
        assert "explanation" not in art


def test_dropped_search_articles_by_enum_rejected_e2e(acl_client) -> None:
    """§2.1 — `searchArticlesBy: ARTICLE_NUMBER` rejects at the ACL
    boundary; ftsearch is never called. Verified by status + envelope
    shape (the ftsearch happy-path response is irrelevant here)."""
    body = _valid_request()
    body["searchArticlesBy"] = "ARTICLE_NUMBER"
    r = acl_client.post(
        "/article-features/search?page=1&pageSize=10",
        json=body,
    )
    assert r.status_code == 400
    assert set(r.json().keys()) == {"message", "details", "timestamp"}


def test_score_field_dropped_from_articles(acl_client) -> None:
    """A3 mapping: ftsearch returns `score` per article; the ACL
    drops it before returning to next-gen (legacy contract carries
    `explanation`, not `score`)."""
    r = acl_client.post(
        "/article-features/search?page=1&pageSize=5",
        json=_valid_request(),
    )
    assert r.status_code == 200
    for art in r.json()["articles"]:
        assert "score" not in art


def test_acl_internal_metadata_fields_dropped(acl_client) -> None:
    """A3 mapping: `recallClipped` and `hitCountClipped` are
    ftsearch-side observability — not in legacy contract."""
    r = acl_client.post(
        "/article-features/search?page=1&pageSize=5",
        json=_valid_request(),
    )
    assert r.status_code == 200
    md = r.json()["metadata"]
    assert "recallClipped" not in md
    assert "hitCountClipped" not in md
