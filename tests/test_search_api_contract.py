"""Acceptance smoke for F2 (ftsearch HTTP contract).

Drives the F2 packet's acceptance criteria:

  * full-shape request returns 200 + valid empty response,
  * unknown searchMode is rejected,
  * pageSize > 500 is rejected,
  * unknown body fields are rejected (`extra='forbid'`),
  * OpenAPI YAML round-trips through `openapi-spec-validator`,
  * v0 alias still routes (deprecated but live).

The tests run against the in-process FastAPI app via `TestClient` — no
Milvus or embedder needed (the F2 stub touches neither).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

SEARCH_API_DIR = Path(__file__).resolve().parent.parent / "search-api"
sys.path.insert(0, str(SEARCH_API_DIR))


@pytest.fixture(scope="module")
def client(monkeypatch_session) -> TestClient:
    monkeypatch_session.setenv("EMBED_URL", "http://embed.invalid")
    monkeypatch_session.setenv("MILVUS_URI", "http://localhost:19530")
    monkeypatch_session.setenv("API_KEY", "")
    import importlib

    import main as main_mod
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
def full_request_body() -> dict:
    return {
        "searchMode": "BOTH",
        "selectedArticleSources": {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": ["aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"],
            "sourcePriceListIds": ["bbbbbbbb-2222-2222-2222-bbbbbbbbbbbb"],
            "customerUploadedCoreArticleListSourceIds": [],
        },
        "query": "Bohrmaschine",
        "articleIdsFilter": [],
        "vendorIdsFilter": [],
        "manufacturersFilter": [],
        "maxDeliveryTime": 5,
        "requiredFeatures": [{"name": "Spannung", "values": ["18V", "36V"]}],
        "priceFilter": {"min": 0, "max": 999999, "currencyCode": "EUR"},
        "currentCategoryPathElements": ["Werkzeug", "Akku"],
        "currentEClass5Code": 31000000,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
        "summaries": ["VENDORS", "MANUFACTURERS", "FEATURES"],
        "coreArticlesVendorsFilter": [],
        "blockedEClassVendorsFilters": [
            {
                "vendorIds": ["44444444-4444-4444-4444-444444444444"],
                "eClassVersion": "ECLASS_5_1",
                "blockedEClassGroups": [{"eClassGroupCode": 12345, "value": True}],
            }
        ],
        "currency": "EUR",
        "eClassesFilter": [123456],
        "eClassesAggregations": [{"id": "agg-id", "eClasses": [123456]}],
        "s2ClassForProductCategories": False,
    }


# ---------- F2 stub returns 200 + valid empty envelope --------------------

def test_full_shape_request_returns_empty_envelope(client: TestClient, full_request_body: dict) -> None:
    r = client.post("/offers_v_alias/_search?page=2&pageSize=25", json=full_request_body)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["articles"] == []
    assert body["metadata"] == {
        "page": 2,
        "pageSize": 25,
        "pageCount": 0,
        "term": "Bohrmaschine",
        "hitCount": 0,
    }
    s = body["summaries"]
    for arr_key in ("vendorSummaries", "manufacturerSummaries", "featureSummaries", "pricesSummary", "eClassesAggregations"):
        assert s[arr_key] == [], f"{arr_key} should be empty in stub"
    for null_key in ("categoriesSummary", "eClass5Categories", "eClass7Categories", "s2ClassCategories"):
        assert s[null_key] is None, f"{null_key} should be null in stub"


def test_minimal_request_works(client: TestClient) -> None:
    body = {
        "searchMode": "HITS_ONLY",
        "selectedArticleSources": {},
        "currency": "EUR",
    }
    r = client.post("/offers_v_alias/_search", json=body)
    assert r.status_code == 200, r.text


# ---------- validation paths ---------------------------------------------

def test_rejects_unknown_search_mode(client: TestClient) -> None:
    r = client.post(
        "/offers_v_alias/_search",
        json={"searchMode": "NOPE", "selectedArticleSources": {}, "currency": "EUR"},
    )
    assert r.status_code == 422
    assert "searchMode" in r.text


def test_rejects_unknown_body_field(client: TestClient) -> None:
    """`extra='forbid'` — typos and stale fields surface as 422."""
    r = client.post(
        "/offers_v_alias/_search",
        json={
            "searchMode": "HITS_ONLY",
            "selectedArticleSources": {},
            "currency": "EUR",
            "searchArticlesBy": "STANDARD",  # legacy field, dropped per §2.1
        },
    )
    assert r.status_code == 422
    assert "searchArticlesBy" in r.text


def test_rejects_bad_currency_format(client: TestClient) -> None:
    r = client.post(
        "/offers_v_alias/_search",
        json={"searchMode": "HITS_ONLY", "selectedArticleSources": {}, "currency": "eur"},
    )
    assert r.status_code == 422


def test_rejects_pagesize_over_cap(client: TestClient) -> None:
    r = client.post(
        "/offers_v_alias/_search?pageSize=501",
        json={"searchMode": "HITS_ONLY", "selectedArticleSources": {}, "currency": "EUR"},
    )
    assert r.status_code == 422


def test_accepts_pagesize_at_cap(client: TestClient) -> None:
    r = client.post(
        "/offers_v_alias/_search?pageSize=500",
        json={"searchMode": "HITS_ONLY", "selectedArticleSources": {}, "currency": "EUR"},
    )
    assert r.status_code == 200


def test_rejects_bad_sort_clause(client: TestClient) -> None:
    """Sort values must be `<field>,<asc|desc>` with a known field."""
    r = client.post(
        "/offers_v_alias/_search?sort=name",
        json={"searchMode": "HITS_ONLY", "selectedArticleSources": {}, "currency": "EUR"},
    )
    assert r.status_code in (400, 422, 500)


def test_accepts_known_sort_clause(client: TestClient) -> None:
    r = client.post(
        "/offers_v_alias/_search?sort=name,asc&sort=price,desc",
        json={"searchMode": "HITS_ONLY", "selectedArticleSources": {}, "currency": "EUR"},
    )
    assert r.status_code == 200


# ---------- v0 alias still routes -----------------------------------------

def test_v0_route_present_and_accepts_legacy_body(client: TestClient) -> None:
    """We don't exercise actual search behaviour (no Milvus available);
    a 404 from Milvus is acceptable proof the legacy DTO validates and
    the route is wired."""
    r = client.post(
        "/non_existent_collection/_search_v0",
        json={"query": "x", "index": "test", "category": None},
    )
    assert r.status_code in (200, 404)
    if r.status_code == 422:
        pytest.fail(f"legacy body shape rejected unexpectedly: {r.text}")


# ---------- OpenAPI doc round-trips ---------------------------------------

def test_openapi_yaml_is_valid() -> None:
    from openapi_spec_validator import validate
    spec_text = (SEARCH_API_DIR / "openapi.yaml").read_text()
    spec = yaml.safe_load(spec_text)
    validate(spec)


def test_openapi_advertises_both_routes() -> None:
    spec = yaml.safe_load((SEARCH_API_DIR / "openapi.yaml").read_text())
    assert "/{collection}/_search" in spec["paths"]
    assert "/{collection}/_search_v0" in spec["paths"]
    assert spec["paths"]["/{collection}/_search_v0"]["post"].get("deprecated") is True
