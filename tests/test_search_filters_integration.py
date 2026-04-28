"""F3.5 — fixture-driven integration tests for scalar filtering + price
post-pass.

Loads `tests/fixtures/offers_schema_smoke.json` into a fresh
`offers_f3_test` collection (created via the same script F1 uses), then
posts requests against the in-process FastAPI app and asserts each
filter from spec §4.3 demonstrably narrows the hit set, that AND
composition narrows further, and that the priceFilter currency-roles
split holds.

Skipped if Milvus is not reachable on localhost:19530.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pymilvus import MilvusClient

SEARCH_API_DIR = Path(__file__).resolve().parent.parent / "search-api"
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SEARCH_API_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

MILVUS_URI = "http://localhost:19530"
COLLECTION = "offers_f3_test"
DIM = 128
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "offers_schema_smoke.json"


def _vector(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float16)


def _to_milvus_row(row: dict) -> dict:
    out = {k: v for k, v in row.items() if not k.startswith("_") and k != "vector_seed"}
    out["offer_embedding"] = _vector(row["vector_seed"])
    return out


@pytest.fixture(scope="module")
def fixture_rows() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text())["rows"]


@pytest.fixture(scope="module")
def milvus_client() -> MilvusClient:
    try:
        c = MilvusClient(uri=MILVUS_URI)
        c.list_collections()
    except Exception as exc:
        pytest.skip(f"Milvus unreachable at {MILVUS_URI}: {exc}")
    return c


@pytest.fixture(scope="module", autouse=True)
def seed_collection(milvus_client: MilvusClient, fixture_rows: list[dict]):
    """Drop-recreate the test collection and load the smoke fixture."""
    from create_offers_collection import build_index_params, build_schema

    if milvus_client.has_collection(COLLECTION):
        milvus_client.drop_collection(COLLECTION)
    schema = build_schema(milvus_client)
    index_params = build_index_params(milvus_client, "HNSW")
    milvus_client.create_collection(
        collection_name=COLLECTION, schema=schema, index_params=index_params,
    )
    milvus_client.load_collection(COLLECTION)
    rows = [_to_milvus_row(r) for r in fixture_rows]
    milvus_client.upsert(collection_name=COLLECTION, data=rows)
    # Block until rows are queryable. Milvus' growing-segment search has
    # a brief lag after upsert; without this, the first scalar query
    # against a fresh collection can return zero.
    expected_count = len(rows)
    import time
    deadline = time.time() + 30
    while time.time() < deadline:
        got = milvus_client.query(
            collection_name=COLLECTION, filter='id != ""', output_fields=["id"], limit=expected_count + 1,
        )
        if len(got) >= expected_count:
            break
        time.sleep(0.5)
    else:
        pytest.fail(f"fixture rows not visible after 30s (got {len(got)}/{expected_count})")
    yield
    milvus_client.drop_collection(COLLECTION)


@pytest.fixture(scope="module")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def client(monkeypatch_session) -> TestClient:
    monkeypatch_session.setenv("EMBED_URL", "http://embed.invalid")
    monkeypatch_session.setenv("MILVUS_URI", MILVUS_URI)
    monkeypatch_session.setenv("API_KEY", "")
    import importlib

    import main as main_mod
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


# ---------- helpers ------------------------------------------------------

_BASE_BODY = {
    "searchMode": "HITS_ONLY",
    "selectedArticleSources": {},
    "currency": "EUR",
}


def _post(client: TestClient, **overrides) -> dict:
    """No query → handler runs filter-only browse, no embed needed."""
    body = {**_BASE_BODY, **overrides}
    r = client.post(f"/{COLLECTION}/_search?pageSize=100", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _ids(body: dict) -> set[str]:
    return {a["articleId"] for a in body["articles"]}


# ---------- baseline -----------------------------------------------------

def test_baseline_no_filters_returns_nothing_without_query(client: TestClient) -> None:
    """No query AND no filters → no defensible browse path → empty."""
    body = _post(client)
    assert body["articles"] == []


def test_baseline_with_one_filter_returns_some(client: TestClient, fixture_rows: list[dict]) -> None:
    body = _post(client, vendorIdsFilter=["11111111-1111-1111-1111-111111111111"])
    assert _ids(body) == {fixture_rows[0]["id"]}


# ---------- per-filter narrowing ----------------------------------------

def test_vendor_filter(client: TestClient, fixture_rows: list[dict]) -> None:
    body = _post(client, vendorIdsFilter=["44444444-4444-4444-4444-444444444444"])
    assert _ids(body) == {fixture_rows[3]["id"]}


def test_article_ids_filter(client: TestClient, fixture_rows: list[dict]) -> None:
    pks = [fixture_rows[1]["id"], fixture_rows[2]["id"]]
    body = _post(client, articleIdsFilter=pks)
    assert _ids(body) == set(pks)


def test_manufacturer_filter(client: TestClient, fixture_rows: list[dict]) -> None:
    body = _post(client, manufacturersFilter=["Bohrwerke"])
    assert _ids(body) == {fixture_rows[3]["id"]}


def test_max_delivery_time_filter(client: TestClient, fixture_rows: list[dict]) -> None:
    """delivery_time_days_max ≤ 3: rows with 3 (idx 1) and 2 (idx 7), plus
    the sparse row (delivery=0)."""
    body = _post(client, vendorIdsFilter=[r["vendor_id"] for r in fixture_rows], maxDeliveryTime=3)
    expected = {r["id"] for r in fixture_rows if r["delivery_time_days_max"] <= 3}
    assert _ids(body) == expected


def test_required_features_or_within(client: TestClient, fixture_rows: list[dict]) -> None:
    """Werkstoff=Stahl OR Werkstoff=Edelstahl matches rows 0 (Stahl) and 2 (Stahl+Edelstahl)."""
    body = _post(client, requiredFeatures=[
        {"name": "Werkstoff", "values": ["Stahl", "Edelstahl"]},
    ])
    assert _ids(body) == {fixture_rows[0]["id"], fixture_rows[2]["id"]}


def test_required_features_and_across(client: TestClient, fixture_rows: list[dict]) -> None:
    body = _post(client, requiredFeatures=[
        {"name": "Werkstoff", "values": ["Stahl"]},
        {"name": "Festigkeitsklasse", "values": ["10.9"]},
    ])
    assert _ids(body) == {fixture_rows[2]["id"]}


def test_category_prefix_at_depth(client: TestClient, fixture_rows: list[dict]) -> None:
    body = _post(client, currentCategoryPathElements=["Werkzeug", "Hand|Maschine"])
    # Row 1's category_l2 is 'Werkzeug¦Hand|Maschine' (the | already in source)
    assert _ids(body) == {fixture_rows[1]["id"]}


def test_eclass5_code_filter(client: TestClient, fixture_rows: list[dict]) -> None:
    body = _post(client, currentEClass5Code=23172001)
    assert _ids(body) == {fixture_rows[0]["id"]}


def test_eclasses_filter(client: TestClient, fixture_rows: list[dict]) -> None:
    body = _post(client, eClassesFilter=[23172001, 23110201])
    assert _ids(body) == {fixture_rows[0]["id"], fixture_rows[2]["id"]}


def test_closed_marketplace_only_without_cv_returns_nothing(client: TestClient) -> None:
    """Per legacy `OfferFilterBuilder`: an empty `closedCatalogVersionIds`
    list with `closedMarketplaceOnly=true` matches no offers."""
    body = _post(client, closedMarketplaceOnly=True)
    assert _ids(body) == set()


def test_closed_marketplace_only_intersects_closed_cv(
    client: TestClient, fixture_rows: list[dict],
) -> None:
    """`eeeeeeee-5555-...` is on rows 2 and 4 — both should pass."""
    body = _post(client, closedMarketplaceOnly=True, selectedArticleSources={
        "closedCatalogVersionIds": ["eeeeeeee-5555-5555-5555-eeeeeeeeeeee"],
    })
    assert _ids(body) == {fixture_rows[2]["id"], fixture_rows[4]["id"]}


def test_closed_catalog_versions_alone_is_noop(
    client: TestClient, fixture_rows: list[dict],
) -> None:
    """Without `closedMarketplaceOnly=true`, ftsearch treats the closed-CV
    list as request metadata only — no standalone CV intersection. (The
    ACL re-adds always-intersect for legacy parity.)"""
    body = _post(client,
                 vendorIdsFilter=[r["vendor_id"] for r in fixture_rows],
                 selectedArticleSources={
                     "closedCatalogVersionIds": ["aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"],
                 })
    # All fixture rows pass — closedCatalogVersionIds did not narrow.
    assert len(_ids(body)) == len(fixture_rows)


def test_relationship_accessory_for(client: TestClient, fixture_rows: list[dict]) -> None:
    body = _post(client, accessoriesForArticleNumber="ACC-001")
    assert _ids(body) == {fixture_rows[3]["id"]}


def test_relationship_spare_part_for(client: TestClient, fixture_rows: list[dict]) -> None:
    body = _post(client, sparePartsForArticleNumber="BASE-MODEL-A")
    assert _ids(body) == {fixture_rows[3]["id"]}


def test_relationship_similar_to(client: TestClient, fixture_rows: list[dict]) -> None:
    body = _post(client, similarToArticleNumber="SIMILAR-X")
    assert _ids(body) == {fixture_rows[3]["id"]}


def test_core_sortiment_with_customer_uploaded(client: TestClient, fixture_rows: list[dict]) -> None:
    body = _post(client, coreSortimentOnly=True, selectedArticleSources={
        "customerUploadedCoreArticleListSourceIds": ["11111111-aaaa-aaaa-aaaa-111111111111"],
    })
    # Row 4 has 11111111-... in enabled_sources (and not in disabled).
    assert _ids(body) == {fixture_rows[4]["id"]}


def test_blocked_eclass_vendors_excludes_listed_vendor_in_eclass(
    client: TestClient, fixture_rows: list[dict],
) -> None:
    """Block eclass5_code=23172001 for vendor 11111111-...; pre-filter to
    that vendor by id so the only remaining hit is whatever survives the
    block. Row 0 → blocked → expect zero results."""
    body = _post(
        client,
        vendorIdsFilter=["11111111-1111-1111-1111-111111111111"],
        blockedEClassVendorsFilters=[{
            "vendorIds": ["11111111-1111-1111-1111-111111111111"],
            "eClassVersion": "ECLASS_5_1",
            "blockedEClassGroups": [{"eClassGroupCode": 23172001, "value": True}],
        }],
    )
    assert _ids(body) == set()


def test_blocked_eclass_vendors_passes_other_vendors(
    client: TestClient, fixture_rows: list[dict],
) -> None:
    body = _post(
        client,
        vendorIdsFilter=[fixture_rows[0]["vendor_id"], fixture_rows[2]["vendor_id"]],
        blockedEClassVendorsFilters=[{
            "vendorIds": [fixture_rows[0]["vendor_id"]],  # only block vendor 0
            "eClassVersion": "ECLASS_5_1",
            "blockedEClassGroups": [{"eClassGroupCode": 23172001, "value": True}],
        }],
    )
    # Row 0 blocked, row 2 passes (vendor not in blocked list).
    assert _ids(body) == {fixture_rows[2]["id"]}


# ---------- AND composition ----------------------------------------------

def test_and_composition_narrows_further(client: TestClient, fixture_rows: list[dict]) -> None:
    a = _post(client, vendorIdsFilter=[r["vendor_id"] for r in fixture_rows])
    assert len(a["articles"]) >= 4

    b = _post(client,
              vendorIdsFilter=[r["vendor_id"] for r in fixture_rows],
              maxDeliveryTime=3)
    assert _ids(b) < _ids(a), "AND composition must produce a strict subset"


# ---------- price filter ------------------------------------------------

def test_price_filter_eur_within_range(client: TestClient, fixture_rows: list[dict]) -> None:
    """Row 1 priced 199.95 EUR; row 4 priced 299.00 EUR. Bounds 100.00–250.00 EUR keep
    only row 1."""
    body = _post(
        client,
        vendorIdsFilter=[r["vendor_id"] for r in fixture_rows],
        priceFilter={"min": 10000, "max": 25000, "currencyCode": "EUR"},
        selectedArticleSources={
            "sourcePriceListIds": [
                "dddddddd-4444-4444-4444-dddddddddddd",
                "ffffffff-6666-6666-6666-ffffffffffff",
            ],
        },
    )
    assert _ids(body) == {fixture_rows[1]["id"]}


def test_price_filter_currency_two_roles(client: TestClient, fixture_rows: list[dict]) -> None:
    """Top-level currency=EUR matches; bound currencyCode=JPY decodes 1500 → 1500 (no
    decimal scaling). Row 1's 199.95 EUR < 1500, so the filter rejects."""
    body = _post(
        client,
        vendorIdsFilter=[fixture_rows[1]["vendor_id"]],
        priceFilter={"min": 1000, "max": 2000, "currencyCode": "JPY"},
        selectedArticleSources={
            "sourcePriceListIds": ["dddddddd-4444-4444-4444-dddddddddddd"],
        },
    )
    assert _ids(body) == set()


def test_price_filter_resolves_highest_priority(client: TestClient, fixture_rows: list[dict]) -> None:
    """Row 0 has two EUR prices in scope: priority 2 (1499.99 EUR) wins over
    priority 1 (1234.56 EUR). Bounds 1400.00–1600.00 EUR pass it."""
    body = _post(
        client,
        vendorIdsFilter=[fixture_rows[0]["vendor_id"]],
        priceFilter={"min": 140000, "max": 160000, "currencyCode": "EUR"},
        selectedArticleSources={
            "sourcePriceListIds": [
                "aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa",
                "bbbbbbbb-2222-2222-2222-bbbbbbbbbbbb",
            ],
        },
    )
    assert _ids(body) == {fixture_rows[0]["id"]}


def test_price_filter_below_min_priority_one_only(client: TestClient, fixture_rows: list[dict]) -> None:
    """Same row 0 but only the priority-1 list is in scope → resolved price = 1234.56.
    Bounds 1400.00–1600.00 EUR reject it."""
    body = _post(
        client,
        vendorIdsFilter=[fixture_rows[0]["vendor_id"]],
        priceFilter={"min": 140000, "max": 160000, "currencyCode": "EUR"},
        selectedArticleSources={
            "sourcePriceListIds": ["aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"],
        },
    )
    assert _ids(body) == set()
