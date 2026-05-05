"""Comprehensive ACL OpenAPI contract tests against the real loaded catalog.

Tests every aspect of acl/openapi.yaml against articles_v6 + offers_v6
in Milvus. Requires:
  - search-api running on localhost:8001 (USE_DEDUP_TOPOLOGY=1, MILVUS_ARTICLES_COLLECTION=articles_v6)
  - ACL running on localhost:8081 (MILVUS_ARTICLES_COLLECTION=offers_v6)
  - Milvus on localhost:19530 with articles_v6 + offers_v6 loaded
"""

from __future__ import annotations

import base64
import re
from datetime import datetime, timezone

import httpx
import pytest
import yaml

ACL_BASE = "http://localhost:8081"

VENDOR_MAJOR = "01054f55-c50c-452b-8822-ee11be4788c9"
VENDOR_MINOR = "0106dc40-9e28-4ae5-8c84-cc5529e1aff8"
CV_EUR = "866b4863-8799-4046-9e84-0985a665c1c7"
CV_BIG = "403188af-9b0f-4019-b983-9b34bbe085c7"
CV_FEATURES = "0b85c4ab-b2e9-4d84-b728-b8be9dedf150"
CV_CHF = "b1470647-a6c7-4e60-8620-c09794a7df50"
CV_ACCESSORIES = "41550352-2c7f-403d-aecd-da9dbb2cf58c"
ARTICLE_NUM_ACCESSORIES = "5040662"
CORE_MARKER_CV = "43e21817-45c6-49a7-b21f-c34ca03546c3"
CORE_MARKER_SOURCE = "e4399ae6-f1ce-495b-8467-a418d54f0400"
PRICE_LIST_IDS_EUR = [
    "51a9dedc-efad-469b-8c81-33676f85630e",
    "fb6b5b83-3f9d-45d5-bab6-91dac4446183",
]

# ---- helpers ---------------------------------------------------------------

def _base_body(**overrides) -> dict:
    body = {
        "searchMode": "BOTH",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV_EUR],
        },
        "maxDeliveryTime": 0,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
        "currency": "EUR",
        "explain": False,
    }
    body.update(overrides)
    return body


def _post(body: dict, *, page: int = 1, page_size: int = 10,
          sort: list[str] | None = None) -> httpx.Response:
    params: dict = {"page": page, "pageSize": page_size}
    if sort:
        params["sort"] = sort
    return httpx.post(
        f"{ACL_BASE}/article-features/search",
        json=body,
        params=params,
        timeout=10,
    )


def _post_ok(body: dict, **kwargs) -> dict:
    r = _post(body, **kwargs)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:500]}"
    return r.json()


@pytest.fixture(scope="session", autouse=True)
def _check_services():
    try:
        r = httpx.get(f"{ACL_BASE}/healthz", timeout=3)
        assert r.status_code == 200
    except Exception:
        pytest.skip("ACL not running on localhost:8081")


# ===========================================================================
# 1. HEALTH & OPENAPI ENDPOINTS
# ===========================================================================

class TestHealthAndOpenAPI:
    def test_healthz_returns_ok(self):
        r = httpx.get(f"{ACL_BASE}/healthz", timeout=3)
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_openapi_yaml_served(self):
        r = httpx.get(f"{ACL_BASE}/openapi.yaml", timeout=3)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/yaml")
        spec = yaml.safe_load(r.text)
        assert spec["info"]["title"] == "article-search-acl"
        assert "/article-features/search" in spec["paths"]


# ===========================================================================
# 2. RESPONSE SCHEMA COMPLIANCE
# ===========================================================================

class TestResponseSchema:
    def test_top_level_keys(self):
        body = _post_ok(_base_body())
        assert set(body.keys()) == {"articles", "summaries", "metadata"}

    def test_articles_is_list(self):
        body = _post_ok(_base_body())
        assert isinstance(body["articles"], list)

    def test_article_shape(self):
        body = _post_ok(_base_body())
        assert len(body["articles"]) > 0, "Expected at least one article"
        art = body["articles"][0]
        assert "articleId" in art
        assert isinstance(art["articleId"], str)

    def test_no_extra_article_fields(self):
        body = _post_ok(_base_body())
        for art in body["articles"]:
            assert set(art.keys()) <= {"articleId", "explanation"}

    def test_metadata_required_fields(self):
        body = _post_ok(_base_body())
        md = body["metadata"]
        for field in ("page", "pageSize", "pageCount", "hitCount"):
            assert field in md, f"Missing required metadata field: {field}"
            assert isinstance(md[field], int)

    def test_metadata_no_internal_fields(self):
        body = _post_ok(_base_body())
        md = body["metadata"]
        assert "recallClipped" not in md
        assert "hitCountClipped" not in md

    def test_summaries_shape(self):
        body = _post_ok(_base_body(summaries=["VENDORS", "MANUFACTURERS", "PRICES"]))
        s = body["summaries"]
        assert isinstance(s.get("vendorSummaries"), list)
        assert isinstance(s.get("manufacturerSummaries"), list)
        assert isinstance(s.get("pricesSummary"), list)

    def test_score_field_never_in_articles(self):
        body = _post_ok(_base_body())
        for art in body["articles"]:
            assert "score" not in art


# ===========================================================================
# 3. ARTICLE ID FORMAT
# ===========================================================================

class TestArticleIdFormat:
    def test_articleid_colon_separated(self):
        body = _post_ok(_base_body())
        for art in body["articles"]:
            parts = art["articleId"].split(":")
            assert len(parts) == 3, (
                f"Expected 3 colon-separated parts, got {len(parts)}: {art['articleId']}"
            )

    def test_articleid_segments_valid(self):
        uuid_re = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        body = _post_ok(_base_body())
        for art in body["articles"]:
            parts = art["articleId"].split(":")
            assert uuid_re.match(parts[0]), f"Part 0 not UUID: {parts[0]}"
            decoded = base64.b64decode(parts[1]).decode("utf-8")
            assert len(decoded) > 0, "Empty article number"
            assert uuid_re.match(parts[2]), f"Part 2 not UUID: {parts[2]}"


# ===========================================================================
# 4. SEARCH MODES
# ===========================================================================

class TestSearchModes:
    def test_hits_only_returns_articles_empty_summaries(self):
        body = _post_ok(_base_body(searchMode="HITS_ONLY"))
        assert len(body["articles"]) > 0
        s = body["summaries"]
        assert s.get("vendorSummaries") == []
        assert s.get("manufacturerSummaries") == []

    def test_summaries_only_returns_no_articles(self):
        body = _post_ok(_base_body(
            searchMode="SUMMARIES_ONLY",
            summaries=["VENDORS"],
        ))
        assert body["articles"] == []
        assert body["metadata"]["hitCount"] > 0

    def test_both_returns_articles_and_summaries(self):
        body = _post_ok(_base_body(
            summaries=["VENDORS", "MANUFACTURERS"],
        ))
        assert len(body["articles"]) > 0
        assert len(body["summaries"]["vendorSummaries"]) > 0


# ===========================================================================
# 5. PAGINATION
# ===========================================================================

class TestPagination:
    def test_page1_default(self):
        body = _post_ok(_base_body())
        assert body["metadata"]["page"] == 1

    def test_pagesize_limits_results(self):
        body = _post_ok(_base_body(), page_size=3)
        assert len(body["articles"]) <= 3

    def test_page_beyond_results_empty(self):
        body = _post_ok(_base_body(), page=9999, page_size=10)
        assert body["articles"] == []
        assert body["metadata"]["page"] == 9999

    def test_pagesize_zero_returns_no_articles(self):
        body = _post_ok(_base_body(), page_size=0)
        assert body["articles"] == []
        assert body["metadata"]["hitCount"] > 0

    def test_pagesize_500_accepted(self):
        body = _post_ok(_base_body(), page_size=500)
        assert body["metadata"]["pageSize"] == 500

    def test_metadata_pagesize_echoes_request(self):
        body = _post_ok(_base_body(), page_size=3)
        assert body["metadata"]["pageSize"] == 3

    def test_pagecount_consistent_with_hitcount(self):
        body = _post_ok(_base_body(), page_size=5)
        md = body["metadata"]
        if md["hitCount"] > 0:
            expected_pages = (md["hitCount"] + 4) // 5
            assert md["pageCount"] == expected_pages

    def test_pagination_stable_across_pages(self):
        body1 = _post_ok(_base_body(), page=1, page_size=5, sort=["articleId,asc"])
        body2 = _post_ok(_base_body(), page=2, page_size=5, sort=["articleId,asc"])
        ids1 = {a["articleId"] for a in body1["articles"]}
        ids2 = {a["articleId"] for a in body2["articles"]}
        assert ids1.isdisjoint(ids2), "Page 1 and 2 overlap"


# ===========================================================================
# 6. SORTING
# ===========================================================================

class TestSorting:
    def test_sort_articleid_asc(self):
        body = _post_ok(_base_body(), sort=["articleId,asc"], page_size=20)
        ids = [a["articleId"] for a in body["articles"]]
        assert ids == sorted(ids), "Articles not sorted by articleId ascending"

    def test_sort_articleid_desc(self):
        body = _post_ok(_base_body(), sort=["articleId,desc"], page_size=20)
        ids = [a["articleId"] for a in body["articles"]]
        assert ids == sorted(ids, reverse=True), "Articles not sorted by articleId descending"

    def test_sort_name_asc_and_desc_differ(self):
        asc = _post_ok(_base_body(), sort=["name,asc"], page_size=10)
        desc = _post_ok(_base_body(), sort=["name,desc"], page_size=10)
        ids_asc = [a["articleId"] for a in asc["articles"]]
        ids_desc = [a["articleId"] for a in desc["articles"]]
        assert len(ids_asc) > 0
        assert len(ids_desc) > 0
        assert ids_asc != ids_desc, "asc and desc returned same order"

    def test_sort_price_asc_and_desc_differ(self):
        sas = {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV_EUR],
            "sourcePriceListIds": PRICE_LIST_IDS_EUR,
        }
        asc = _post_ok(_base_body(selectedArticleSources=sas), sort=["price,asc"], page_size=10)
        desc = _post_ok(_base_body(selectedArticleSources=sas), sort=["price,desc"], page_size=10)
        ids_asc = [a["articleId"] for a in asc["articles"]]
        ids_desc = [a["articleId"] for a in desc["articles"]]
        assert len(ids_asc) > 0
        assert len(ids_desc) > 0
        assert ids_asc != ids_desc, "price asc and desc returned same order"

    def test_multi_key_sort_accepted(self):
        body = _post_ok(
            _base_body(
                selectedArticleSources={
                    "closedCatalogVersionIds": [],
                    "catalogVersionIdsOrderedByPreference": [CV_EUR],
                    "sourcePriceListIds": PRICE_LIST_IDS_EUR,
                },
            ),
            sort=["price,asc", "articleId,asc"],
            page_size=5,
        )
        assert len(body["articles"]) > 0

    def test_sort_invalid_field_rejected(self):
        r = _post(_base_body(), sort=["invalid,asc"])
        assert r.status_code == 400

    def test_sort_invalid_direction_rejected(self):
        r = _post(_base_body(), sort=["price,up"])
        assert r.status_code == 400


# ===========================================================================
# 7. EXPLAIN (§2.2)
# ===========================================================================

class TestExplain:
    def test_explain_true_stubs_na(self):
        body = _post_ok(_base_body(explain=True))
        assert len(body["articles"]) > 0
        for art in body["articles"]:
            assert art.get("explanation") == "N/A"

    def test_explain_false_omits_explanation(self):
        body = _post_ok(_base_body(explain=False))
        for art in body["articles"]:
            assert "explanation" not in art


# ===========================================================================
# 8. FILTERS
# ===========================================================================

class TestFilters:
    def test_vendor_filter_narrows_results(self):
        unfiltered = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            summaries=["VENDORS"],
        ))
        filtered = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            vendorIdsFilter=[VENDOR_MAJOR],
            summaries=["VENDORS"],
        ))
        assert filtered["metadata"]["hitCount"] <= unfiltered["metadata"]["hitCount"]
        for vs in filtered["summaries"]["vendorSummaries"]:
            assert vs["vendorId"] == VENDOR_MAJOR

    def test_manufacturer_filter_narrows_results(self):
        sas = {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV_BIG],
        }
        unfiltered = _post_ok(_base_body(selectedArticleSources=sas, summaries=["MANUFACTURERS"]))
        filtered = _post_ok(_base_body(
            selectedArticleSources=sas,
            manufacturersFilter=["Würth"],
            summaries=["MANUFACTURERS"],
        ))
        assert filtered["metadata"]["hitCount"] <= unfiltered["metadata"]["hitCount"]

    def test_max_delivery_time_filter(self):
        all_results = _post_ok(_base_body())
        filtered = _post_ok(_base_body(maxDeliveryTime=1))
        assert filtered["metadata"]["hitCount"] <= all_results["metadata"]["hitCount"]

    def test_closed_marketplace_only(self):
        body = _post_ok(_base_body(
            closedMarketplaceOnly=True,
            selectedArticleSources={
                "closedCatalogVersionIds": [CV_EUR],
                "catalogVersionIdsOrderedByPreference": [],
            },
        ))
        assert body["metadata"]["hitCount"] >= 0

    def test_multiple_catalog_versions(self):
        single = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_EUR],
            },
        ))
        multi = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_EUR, CV_BIG],
            },
        ))
        assert multi["metadata"]["hitCount"] >= single["metadata"]["hitCount"]

    def test_empty_catalog_scope_returns_zero(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [],
            },
        ))
        assert body["metadata"]["hitCount"] == 0
        assert body["articles"] == []

    def test_nonexistent_catalog_version_returns_zero(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": ["00000000-0000-0000-0000-000000000000"],
            },
        ))
        assert body["metadata"]["hitCount"] == 0

    def test_article_ids_filter_returns_only_requested(self):
        browse = _post_ok(_base_body(), page_size=3, sort=["articleId,asc"])
        assert len(browse["articles"]) > 0
        target_id = browse["articles"][0]["articleId"]
        filtered = _post_ok(_base_body(articleIdsFilter=[target_id]))
        found_ids = {a["articleId"] for a in filtered["articles"]}
        assert target_id in found_ids
        assert found_ids == {target_id}

    def test_accessories_for_article_number(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_ACCESSORIES],
            },
            accessoriesForArticleNumber=ARTICLE_NUM_ACCESSORIES,
        ))
        assert body["metadata"]["hitCount"] >= 0

    def test_spare_parts_for_article_number(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_ACCESSORIES],
            },
            sparePartsForArticleNumber=ARTICLE_NUM_ACCESSORIES,
        ))
        assert body["metadata"]["hitCount"] >= 0

    def test_similar_to_article_number(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_ACCESSORIES],
            },
            similarToArticleNumber=ARTICLE_NUM_ACCESSORIES,
        ))
        assert body["metadata"]["hitCount"] >= 0

    def test_eclass_filter_narrows_results(self):
        sas = {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV_BIG],
        }
        unfiltered = _post_ok(_base_body(selectedArticleSources=sas))
        filtered = _post_ok(_base_body(selectedArticleSources=sas, eClassesFilter=[23110103]))
        assert filtered["metadata"]["hitCount"] <= unfiltered["metadata"]["hitCount"]

    def test_required_features_filter_narrows_results(self):
        sas = {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV_FEATURES],
        }
        unfiltered = _post_ok(_base_body(selectedArticleSources=sas))
        filtered = _post_ok(_base_body(
            selectedArticleSources=sas,
            requiredFeatures=[{"name": "für Schraube", "values": ["M 10"]}],
        ))
        assert filtered["metadata"]["hitCount"] <= unfiltered["metadata"]["hitCount"]

    def test_price_filter_min_max(self):
        body = _post_ok(_base_body(
            priceFilter={"min": 100, "max": 10000, "currencyCode": "EUR"},
        ))
        assert body["metadata"]["hitCount"] >= 0

    def test_price_filter_min_only(self):
        body = _post_ok(_base_body(
            priceFilter={"min": 500, "currencyCode": "EUR"},
        ))
        assert body["metadata"]["hitCount"] >= 0

    def test_price_filter_max_only(self):
        body = _post_ok(_base_body(
            priceFilter={"max": 100, "currencyCode": "EUR"},
        ))
        assert body["metadata"]["hitCount"] >= 0


# ===========================================================================
# 9. TEXT SEARCH (queryString)
# ===========================================================================

class TestTextSearch:
    def test_query_returns_results(self):
        body = _post_ok(_base_body(
            queryString="Schraube",
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
        ))
        assert body["metadata"]["hitCount"] > 0
        assert body["metadata"]["term"] == "Schraube"

    def test_query_null_is_browse(self):
        body = _post_ok(_base_body(queryString=None))
        assert body["metadata"]["term"] is None

    def test_nonsense_query_returns_valid_response(self):
        body = _post_ok(_base_body(queryString="xyznonexistent99999"))
        assert "articles" in body
        assert body["metadata"]["term"] == "xyznonexistent99999"

    def test_query_with_sort_returns_results(self):
        body = _post_ok(
            _base_body(
                queryString="Ventil",
                selectedArticleSources={
                    "closedCatalogVersionIds": [],
                    "catalogVersionIdsOrderedByPreference": [CV_BIG],
                },
            ),
            sort=["price,asc"],
        )
        assert body["metadata"]["hitCount"] >= 0

    def test_empty_query_string(self):
        body = _post_ok(_base_body(queryString=""))
        assert body["metadata"]["term"] is None or body["metadata"]["term"] == ""

    def test_metadata_term_echoes_query(self):
        body = _post_ok(_base_body(queryString="Bohrer"))
        assert body["metadata"]["term"] == "Bohrer"


# ===========================================================================
# 10. SUMMARIES
# ===========================================================================

class TestSummaries:
    def test_vendor_summaries(self):
        body = _post_ok(_base_body(summaries=["VENDORS"]))
        vs = body["summaries"]["vendorSummaries"]
        assert isinstance(vs, list)
        if vs:
            assert "vendorId" in vs[0]
            assert "count" in vs[0]
            assert isinstance(vs[0]["count"], int)
            assert vs[0]["count"] > 0

    def test_vendor_summary_shape(self):
        body = _post_ok(_base_body(summaries=["VENDORS"]))
        for vs in body["summaries"]["vendorSummaries"]:
            assert set(vs.keys()) == {"vendorId", "count"}

    def test_manufacturer_summaries(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            summaries=["MANUFACTURERS"],
        ))
        ms = body["summaries"]["manufacturerSummaries"]
        assert isinstance(ms, list)
        if ms:
            assert "name" in ms[0]
            assert "count" in ms[0]

    def test_feature_summaries(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_FEATURES],
            },
            summaries=["FEATURES"],
        ))
        fs = body["summaries"]["featureSummaries"]
        assert isinstance(fs, list)
        if fs:
            f = fs[0]
            assert "name" in f
            assert "count" in f
            assert "values" in f
            assert isinstance(f["values"], list)

    def test_feature_value_count_shape(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_FEATURES],
            },
            summaries=["FEATURES"],
        ))
        for fs in body["summaries"]["featureSummaries"]:
            for v in fs["values"]:
                assert set(v.keys()) == {"value", "count"}

    def test_prices_summary(self):
        body = _post_ok(_base_body(summaries=["PRICES"]))
        ps = body["summaries"]["pricesSummary"]
        assert isinstance(ps, list)
        if ps:
            p = ps[0]
            assert set(p.keys()) == {"min", "max", "currencyCode"}
            assert p["min"] <= p["max"]
            assert re.match(r"^[A-Z]{3}$", p["currencyCode"]), (
                f"Invalid currency code: {p['currencyCode']}"
            )

    def test_categories_summary(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            summaries=["CATEGORIES"],
        ))
        cs = body["summaries"].get("categoriesSummary")
        if cs is not None:
            assert isinstance(cs.get("sameLevel", []), list)
            assert isinstance(cs.get("children", []), list)

    def test_eclass5_summary(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            summaries=["ECLASS5"],
        ))
        ec = body["summaries"].get("eClass5Categories")
        if ec is not None:
            assert isinstance(ec.get("sameLevel", []), list)
            for bucket in ec.get("sameLevel", []):
                assert "group" in bucket
                assert "count" in bucket

    def test_eclass7_summary(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            summaries=["ECLASS7"],
        ))
        ec = body["summaries"].get("eClass7Categories")
        assert ec is None or isinstance(ec, dict)

    def test_s2class_summary(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            summaries=["S2CLASS"],
        ))
        sc = body["summaries"].get("s2ClassCategories")
        assert sc is None or isinstance(sc, dict)

    def test_eclass5set_summary(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            summaries=["ECLASS5SET"],
        ))
        assert "summaries" in body

    def test_platform_categories_summary(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            summaries=["PLATFORM_CATEGORIES"],
        ))
        assert "summaries" in body

    def test_summaries_no_extra_keys(self):
        body = _post_ok(_base_body(
            summaries=["VENDORS", "MANUFACTURERS", "FEATURES", "PRICES",
                        "CATEGORIES", "ECLASS5", "ECLASS7", "S2CLASS"],
        ))
        allowed = {
            "vendorSummaries", "manufacturerSummaries", "featureSummaries",
            "pricesSummary", "categoriesSummary", "eClass5Categories",
            "eClass7Categories", "s2ClassCategories", "eClassesAggregations",
        }
        assert set(body["summaries"].keys()) <= allowed

    def test_summaries_not_requested_are_empty(self):
        body = _post_ok(_base_body(summaries=[]))
        s = body["summaries"]
        assert s.get("vendorSummaries") == []
        assert s.get("manufacturerSummaries") == []

    def test_eclass_aggregations(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            eClassesAggregations=[
                {"id": "test-agg", "eClasses": [23110103]},
            ],
        ))
        aggs = body["summaries"].get("eClassesAggregations", [])
        assert isinstance(aggs, list)

    def test_summaries_only_with_all_kinds(self):
        body = _post_ok(_base_body(
            searchMode="SUMMARIES_ONLY",
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            summaries=["VENDORS", "MANUFACTURERS", "FEATURES", "PRICES",
                        "CATEGORIES", "ECLASS5"],
        ))
        assert body["articles"] == []
        assert body["metadata"]["hitCount"] >= 0


# ===========================================================================
# 11. CURRENCY
# ===========================================================================

class TestCurrency:
    def test_eur_currency(self):
        body = _post_ok(_base_body(currency="EUR"))
        assert body["metadata"]["hitCount"] >= 0

    def test_chf_currency_with_chf_catalog(self):
        body = _post_ok(_base_body(
            currency="CHF",
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_CHF],
            },
        ))
        assert body["metadata"]["hitCount"] >= 0


# ===========================================================================
# 12. VALIDATION / ERROR HANDLING
# ===========================================================================

class TestValidation:
    def test_missing_required_field_returns_400(self):
        body = {"searchMode": "BOTH"}
        r = _post(body)
        assert r.status_code == 400

    def test_error_envelope_shape(self):
        body = {"searchMode": "BOTH"}
        r = _post(body)
        assert r.status_code == 400
        err = r.json()
        assert set(err.keys()) == {"message", "details", "timestamp"}
        assert isinstance(err["details"], list)

    def test_error_timestamp_iso8601(self):
        body = {"searchMode": "BOTH"}
        r = _post(body)
        err = r.json()
        ts = err["timestamp"]
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None or "Z" in ts or "+" in ts

    def test_invalid_search_articles_by_rejected(self):
        body = _base_body()
        body["searchArticlesBy"] = "ARTICLE_NUMBER"
        r = _post(body)
        assert r.status_code == 400

    def test_all_legacy_search_articles_by_rejected(self):
        for val in ["ALL_ATTRIBUTES", "CUSTOMER_ARTICLE_NUMBER",
                     "VENDOR_ARTICLE_NUMBER", "EAN"]:
            body = _base_body()
            body["searchArticlesBy"] = val
            r = _post(body)
            assert r.status_code == 400, f"{val} should be rejected"

    def test_invalid_search_mode_rejected(self):
        body = _base_body()
        body["searchMode"] = "INVALID"
        r = _post(body)
        assert r.status_code == 400

    def test_invalid_currency_format_rejected(self):
        body = _base_body()
        body["currency"] = "eur"
        r = _post(body)
        assert r.status_code == 400

    def test_unknown_field_rejected(self):
        body = _base_body()
        body["unknownField"] = "value"
        r = _post(body)
        assert r.status_code == 400

    def test_price_filter_currency_required_when_min_set(self):
        body = _base_body(priceFilter={"min": 100})
        r = _post(body)
        assert r.status_code == 400

    def test_price_filter_currency_required_when_max_set(self):
        body = _base_body(priceFilter={"max": 500})
        r = _post(body)
        assert r.status_code == 400

    def test_unknown_field_in_selected_article_sources_rejected(self):
        body = _base_body()
        body["selectedArticleSources"]["unknownField"] = "value"
        r = _post(body)
        assert r.status_code == 400

    def test_unknown_field_in_price_filter_rejected(self):
        body = _base_body(priceFilter={"currencyCode": "EUR", "unknownField": 1})
        r = _post(body)
        assert r.status_code == 400

    def test_unknown_field_in_feature_filter_rejected(self):
        body = _base_body(
            requiredFeatures=[{"name": "test", "values": [], "unknownField": 1}],
        )
        r = _post(body)
        assert r.status_code == 400

    def test_negative_max_delivery_time_rejected(self):
        body = _base_body()
        body["maxDeliveryTime"] = -1
        r = _post(body)
        assert r.status_code == 400

    def test_page_zero_rejected(self):
        r = _post(_base_body(), page=0)
        assert r.status_code == 400 or r.status_code == 422

    def test_pagesize_over_500_rejected(self):
        r = _post(_base_body(), page_size=501)
        assert r.status_code == 400 or r.status_code == 422


# ===========================================================================
# 13. ACL CONTRACT SPECIFICS
# ===========================================================================

class TestACLContract:
    def test_customer_article_number_fields_accepted(self):
        body = _base_body()
        body["selectedArticleSources"]["customerArticleNumbersIndexingSourceIds"] = [
            "00000000-0000-0000-0000-000000000001"
        ]
        body["selectedArticleSources"]["customerManagedArticleNumberListId"] = (
            "00000000-0000-0000-0000-000000000002"
        )
        body["selectedArticleSources"]["uiCustomerArticleNumberSourceId"] = (
            "00000000-0000-0000-0000-000000000003"
        )
        r = _post(body)
        assert r.status_code == 200

    def test_source_price_list_ids_accepted(self):
        body = _base_body()
        body["selectedArticleSources"]["sourcePriceListIds"] = [
            "00000000-0000-0000-0000-000000000001"
        ]
        r = _post(body)
        assert r.status_code == 200

    def test_blocked_eclass_vendors_filters_accepted(self):
        body = _base_body(
            blockedEClassVendorsFilters=[{
                "vendorIds": [VENDOR_MAJOR],
                "eClassVersion": "ECLASS_5_1",
                "blockedEClassGroups": [
                    {"eClassGroupCode": 23110000, "value": True}
                ],
            }],
        )
        r = _post(body)
        assert r.status_code == 200

    def test_s2class_for_product_categories_accepted(self):
        body = _base_body(s2ClassForProductCategories=True)
        r = _post(body)
        assert r.status_code == 200

    def test_current_eclass5_code_accepted(self):
        body = _base_body(
            currentEClass5Code=23110103,
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            summaries=["ECLASS5"],
        )
        body_resp = _post_ok(body)
        ec = body_resp["summaries"].get("eClass5Categories")
        if ec is not None:
            assert "selectedEClassGroup" in ec or "sameLevel" in ec

    def test_current_eclass7_code_accepted(self):
        body = _base_body(
            currentEClass7Code=27260701,
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            summaries=["ECLASS7"],
        )
        r = _post(body)
        assert r.status_code == 200

    def test_current_s2class_code_accepted(self):
        body = _base_body(
            currentS2ClassCode=32039090,
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            summaries=["S2CLASS"],
        )
        r = _post(body)
        assert r.status_code == 200

    def test_current_category_path_elements_accepted(self):
        body = _base_body(
            currentCategoryPathElements=["Fachkreis Befestigungstechnik"],
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
            summaries=["CATEGORIES"],
        )
        r = _post(body)
        assert r.status_code == 200


# ===========================================================================
# 14. CORE SORTIMENT
# ===========================================================================

class TestCoreSortiment:
    def test_core_sortiment_only_accepted(self):
        body = _post_ok(_base_body(
            coreSortimentOnly=True,
            selectedArticleSources={
                "closedCatalogVersionIds": [CORE_MARKER_CV],
                "catalogVersionIdsOrderedByPreference": [CORE_MARKER_CV],
                "customerUploadedCoreArticleListSourceIds": [CORE_MARKER_SOURCE],
            },
        ))
        assert body["metadata"]["hitCount"] >= 0

    def test_core_articles_vendors_filter_accepted(self):
        body = _post_ok(_base_body(
            coreArticlesVendorsFilter=[VENDOR_MAJOR],
        ))
        assert body["metadata"]["hitCount"] >= 0


# ===========================================================================
# 15. IDEMPOTENCY & STABILITY
# ===========================================================================

# ===========================================================================
# 16. CROSS-CUTTING BEHAVIORAL CHECKS
# ===========================================================================

class TestBehavior:
    def test_vendor_summary_count_equals_hitcount(self):
        body = _post_ok(_base_body(summaries=["VENDORS"]))
        vsum = sum(v["count"] for v in body["summaries"]["vendorSummaries"])
        assert vsum == body["metadata"]["hitCount"]

    def test_price_filter_narrows_with_price_list(self):
        sas = {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV_EUR],
            "sourcePriceListIds": PRICE_LIST_IDS_EUR,
        }
        all_results = _post_ok(_base_body(selectedArticleSources=sas))
        filtered = _post_ok(_base_body(
            selectedArticleSources=sas,
            priceFilter={"min": 0, "max": 1000, "currencyCode": "EUR"},
        ))
        assert filtered["metadata"]["hitCount"] < all_results["metadata"]["hitCount"]

    def test_delivery_time_filter_narrows(self):
        all_results = _post_ok(_base_body())
        filtered = _post_ok(_base_body(maxDeliveryTime=1))
        assert filtered["metadata"]["hitCount"] < all_results["metadata"]["hitCount"]

    def test_combined_filters_are_and_composed(self):
        sas = {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV_EUR],
            "sourcePriceListIds": PRICE_LIST_IDS_EUR,
        }
        vendor_only = _post_ok(_base_body(
            selectedArticleSources=sas,
            vendorIdsFilter=[VENDOR_MAJOR],
        ))
        vendor_and_price = _post_ok(_base_body(
            selectedArticleSources=sas,
            vendorIdsFilter=[VENDOR_MAJOR],
            priceFilter={"min": 0, "max": 1000, "currencyCode": "EUR"},
        ))
        assert vendor_and_price["metadata"]["hitCount"] <= vendor_only["metadata"]["hitCount"]

    def test_query_with_price_sort_no_source_price_lists_returns_zero_articles(self):
        body = _post_ok(_base_body(), sort=["price,asc"], page_size=5)
        assert body["articles"] == []
        assert body["metadata"]["hitCount"] > 0

    def test_summaries_only_hitcount_matches_both_hitcount(self):
        sas = {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV_EUR],
        }
        both = _post_ok(_base_body(
            searchMode="BOTH", selectedArticleSources=sas, summaries=["VENDORS"],
        ))
        summaries = _post_ok(_base_body(
            searchMode="SUMMARIES_ONLY", selectedArticleSources=sas, summaries=["VENDORS"],
        ))
        assert both["metadata"]["hitCount"] == summaries["metadata"]["hitCount"]

    def test_page_beyond_range_has_correct_metadata(self):
        body = _post_ok(_base_body(), page=9999, page_size=10)
        assert body["articles"] == []
        assert body["metadata"]["hitCount"] > 0
        assert body["metadata"]["pageCount"] > 0


# ===========================================================================
# 17. IDEMPOTENCY & STABILITY
# ===========================================================================

# ===========================================================================
# 18. EDGE CASES
# ===========================================================================

class TestEdgeCases:
    def test_pagecount_single_result(self):
        browse = _post_ok(_base_body(), page_size=3, sort=["articleId,asc"])
        target_id = browse["articles"][0]["articleId"]
        body = _post_ok(_base_body(articleIdsFilter=[target_id]), page_size=10)
        assert body["metadata"]["hitCount"] == 1
        assert body["metadata"]["pageCount"] == 1

    def test_pagecount_exact_division(self):
        body = _post_ok(_base_body(), page_size=1)
        md = body["metadata"]
        assert md["pageCount"] == md["hitCount"]

    def test_sort_with_query_string_returns_results(self):
        sas = {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV_BIG],
        }
        body = _post_ok(
            _base_body(queryString="Stahl", selectedArticleSources=sas),
            sort=["name,asc"], page_size=10,
        )
        assert len(body["articles"]) > 0

    def test_unicode_query_string(self):
        body = _post_ok(_base_body(
            queryString="größe",
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_BIG],
            },
        ))
        assert body["metadata"]["term"] == "größe"

    def test_large_page_size_with_small_result_set(self):
        body = _post_ok(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_EUR],
            },
        ), page_size=500)
        assert len(body["articles"]) == body["metadata"]["hitCount"]

    def test_all_eclass_versions_in_blocked_filter_accepted(self):
        for version in ["ECLASS_5_1", "ECLASS_7_1", "S2CLASS"]:
            body = _base_body(
                blockedEClassVendorsFilters=[{
                    "vendorIds": [VENDOR_MAJOR],
                    "eClassVersion": version,
                    "blockedEClassGroups": [
                        {"eClassGroupCode": 23110000, "value": True}
                    ],
                }],
            )
            r = _post(body)
            assert r.status_code == 200, f"eClassVersion={version} rejected"


# ===========================================================================
# 19. IDEMPOTENCY & STABILITY
# ===========================================================================

class TestStability:
    def test_same_request_same_hitcount(self):
        body = _base_body()
        r1 = _post_ok(body)
        r2 = _post_ok(body)
        assert r1["metadata"]["hitCount"] == r2["metadata"]["hitCount"]

    def test_same_request_same_articles(self):
        body = _base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_EUR],
            },
        )
        r1 = _post_ok(body, sort=["articleId,asc"], page_size=10)
        r2 = _post_ok(body, sort=["articleId,asc"], page_size=10)
        ids1 = [a["articleId"] for a in r1["articles"]]
        ids2 = [a["articleId"] for a in r2["articles"]]
        assert ids1 == ids2
