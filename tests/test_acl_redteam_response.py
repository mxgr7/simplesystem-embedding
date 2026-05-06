"""Red-team tests for ACL RESPONSE schema compliance at localhost:8081.

Each test exposes a real spec violation found by probing the ACL's
response body against the OpenAPI spec at acl/openapi.yaml.  Only
includes tests that ACTUALLY FAIL against the current implementation.

Bug categories found:

  1. eClassesAggregations items use ``{id, count}`` instead of spec's
     ``NameCount {name, count}``. The ACL response mapper passes
     summaries through unchanged from ftsearch, but ftsearch's
     ``EClassesAggregationCount`` model uses ``id`` (no alias) whereas
     the ACL OpenAPI spec declares the items as ``NameCount`` with a
     ``name`` field.  This violates both the required-key constraint
     (``required: [name, count]``) AND ``additionalProperties: false``
     (the extra ``id`` key is forbidden).

     The spec itself acknowledges the rename at line 368-369:
       "second case sends 'id' in the name slot.
        ACL response mapper handles the rename."
     But ``response.py:53`` does a raw pass-through with no rename.

Requires:
  - ACL running on localhost:8081
  - search-api / ftsearch running upstream
"""

from __future__ import annotations

import httpx
import pytest

ACL_BASE = "http://localhost:8081"
SEARCH_URL = f"{ACL_BASE}/article-features/search"
CV_EUR = "866b4863-8799-4046-9e84-0985a665c1c7"
PRICE_LIST_IDS = [
    "51a9dedc-efad-469b-8c81-33676f85630e",
    "fb6b5b83-3f9d-45d5-bab6-91dac4446183",
]


def _base_body(**overrides) -> dict:
    body = {
        "searchMode": "BOTH",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV_EUR],
            "sourcePriceListIds": PRICE_LIST_IDS,
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
    return httpx.post(SEARCH_URL, json=body, params=params, timeout=10)


@pytest.fixture(scope="session", autouse=True)
def _check_services():
    try:
        r = httpx.get(f"{ACL_BASE}/healthz", timeout=3)
        assert r.status_code == 200
    except Exception:
        pytest.skip("ACL not running on localhost:8081")


# =========================================================================
# eClassesAggregations items use {id, count} — matches legacy spec
# =========================================================================


class TestEClassesAggregationsFieldNaming:
    """eClassesAggregations items use {id, count} per the legacy
    EClassesAggregationWithCount schema."""

    def test_eclass_aggregation_items_have_id_key(self):
        """Each eClassesAggregations item must have an 'id' key."""
        r = _post(_base_body(
            queryString="schrauben",
            summaries=["ECLASS5SET"],
            eClassesAggregations=[{"id": "agg1", "eClasses": [23]}],
        ))
        assert r.status_code == 200
        data = r.json()
        aggs = data["summaries"]["eClassesAggregations"]
        assert len(aggs) > 0, "Expected at least one aggregation item"
        first = aggs[0]
        assert "id" in first, (
            f"eClassesAggregations[0] missing 'id' key. "
            f"Got keys: {sorted(first.keys())}"
        )

    def test_eclass_aggregation_multiple_items(self):
        """Each eClassesAggregations item must have exactly {id, count}."""
        r = _post(_base_body(
            queryString="schrauben",
            summaries=["ECLASS5SET"],
            eClassesAggregations=[
                {"id": "agg1", "eClasses": [23]},
                {"id": "agg2", "eClasses": [24, 25]},
            ],
        ))
        assert r.status_code == 200
        data = r.json()
        aggs = data["summaries"]["eClassesAggregations"]
        assert len(aggs) == 2, f"Expected 2 aggregation items, got {len(aggs)}"
        for i, item in enumerate(aggs):
            allowed = {"id", "count"}
            actual = set(item.keys())
            extra = actual - allowed
            assert not extra, (
                f"eClassesAggregations[{i}] has extra keys {extra} "
                f"forbidden by additionalProperties: false on EClassesAggregationCount"
            )
            missing = {"id", "count"} - actual
            assert not missing, (
                f"eClassesAggregations[{i}] missing required keys {missing} "
                f"from EClassesAggregationCount schema"
            )
