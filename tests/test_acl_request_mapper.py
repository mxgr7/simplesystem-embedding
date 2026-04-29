"""Unit tests for `acl/mapping/request.py`.

Pure-function mapper — fast, no I/O. Covers the rename rules, the
deviation drops (§2.1, §2.2), the currency two-roles preservation,
and the page/sort move from body to query string.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from acl.mapping.request import map_request  # noqa: E402
from acl.models import LegacySearchRequest  # noqa: E402


def _minimal_request(**overrides) -> LegacySearchRequest:
    """Smallest valid `LegacySearchRequest` for tests — fill in
    only the required fields."""
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
    return LegacySearchRequest.model_validate(base)


def test_query_string_renamed_to_query() -> None:
    """ftsearch uses `query`; legacy uses `queryString`."""
    req = _minimal_request(queryString="schraube")
    out = map_request(req)
    assert out.body.get("query") == "schraube"
    assert "queryString" not in out.body


def test_query_absent_when_legacy_omits_it() -> None:
    """A None queryString shouldn't emit a `query: null` to ftsearch
    — ftsearch's stub already treats absence and None equivalently
    but we keep the wire compact."""
    req = _minimal_request()
    out = map_request(req)
    assert "query" not in out.body
    assert "queryString" not in out.body


def test_search_articles_by_dropped() -> None:
    """§2.1 — ftsearch never sees this field."""
    req = _minimal_request()
    out = map_request(req)
    assert "searchArticlesBy" not in out.body


def test_explain_dropped() -> None:
    """§2.2 — `explain` is response-only; ftsearch never sees it."""
    req = _minimal_request(explain=True)
    out = map_request(req)
    assert "explain" not in out.body


def test_currency_two_roles_preserved() -> None:
    """Top-level `currency` and `priceFilter.currencyCode` forward
    independently — no equality coercion."""
    req = _minimal_request(
        currency="EUR",
        priceFilter={"min": 100, "max": 1000, "currencyCode": "JPY"},
    )
    out = map_request(req)
    assert out.body["currency"] == "EUR"
    assert out.body["priceFilter"]["currencyCode"] == "JPY"
    assert out.body["priceFilter"]["min"] == 100
    assert out.body["priceFilter"]["max"] == 1000


def test_pagination_moves_from_body_to_query_params() -> None:
    out = map_request(_minimal_request(), page=3, page_size=42)
    assert out.params["page"] == 3
    assert out.params["pageSize"] == 42
    # Body must NOT carry these — ftsearch reads them from query string.
    assert "page" not in out.body
    assert "pageSize" not in out.body


def test_sort_forwarded_as_list_for_repeatable_query_params() -> None:
    """ftsearch's FastAPI Query(default_factory=list) accepts repeated
    `?sort=` params; httpx serialises a list value as repeated."""
    out = map_request(
        _minimal_request(), sort=["name,asc", "articleId,desc"],
    )
    assert out.params["sort"] == ["name,asc", "articleId,desc"]


def test_sort_omitted_when_empty() -> None:
    out = map_request(_minimal_request())
    assert "sort" not in out.params


def test_summaries_pass_through() -> None:
    req = _minimal_request(
        summaries=["VENDORS", "MANUFACTURERS", "CATEGORIES"],
    )
    out = map_request(req)
    assert out.body["summaries"] == ["VENDORS", "MANUFACTURERS", "CATEGORIES"]


def test_e_classes_aggregations_pass_through() -> None:
    req = _minimal_request(
        eClassesAggregations=[{"id": "agg-a", "eClasses": [123, 456]}],
    )
    out = map_request(req)
    assert out.body["eClassesAggregations"] == [
        {"id": "agg-a", "eClasses": [123, 456]},
    ]


def test_blocked_eclass_vendors_filters_pass_through() -> None:
    req = _minimal_request(
        blockedEClassVendorsFilters=[{
            "vendorIds": ["aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"],
            "eClassVersion": "ECLASS_5_1",
            "blockedEClassGroups": [{"eClassGroupCode": 12345, "value": True}],
        }],
    )
    out = map_request(req)
    assert len(out.body["blockedEClassVendorsFilters"]) == 1
    f = out.body["blockedEClassVendorsFilters"][0]
    assert f["eClassVersion"] == "ECLASS_5_1"
    assert f["blockedEClassGroups"][0]["eClassGroupCode"] == 12345


def test_selected_article_sources_pass_through() -> None:
    req = _minimal_request(
        selectedArticleSources={
            "closedCatalogVersionIds": ["aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"],
            "sourcePriceListIds": ["bbbbbbbb-2222-2222-2222-bbbbbbbbbbbb"],
        },
    )
    out = map_request(req)
    sas = out.body["selectedArticleSources"]
    assert sas["closedCatalogVersionIds"] == ["aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"]
    assert sas["sourcePriceListIds"] == ["bbbbbbbb-2222-2222-2222-bbbbbbbbbbbb"]


def test_mapper_is_pure() -> None:
    """Same input → same output. Run the mapper twice on the same
    request, assert both bodies + params equal."""
    req = _minimal_request(
        queryString="schraube",
        currency="CHF",
        priceFilter={"min": 1000, "max": 5000, "currencyCode": "CHF"},
    )
    out1 = map_request(req, page=2, page_size=25, sort=["name,asc"])
    out2 = map_request(req, page=2, page_size=25, sort=["name,asc"])
    assert out1.body == out2.body
    assert out1.params == out2.params
