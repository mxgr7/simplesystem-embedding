"""Spec-alignment regression tests for the ACL at localhost:8081.

Tests discrepancies found during a spec audit of the ACL OpenAPI
contract (acl/openapi.yaml) against the live service behaviour.

Requires:
  - ACL running on localhost:8081
  - search-api / ftsearch running upstream
"""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest

ACL_BASE = "http://localhost:8081"
SEARCH_URL = f"{ACL_BASE}/article-features/search"

BASE_BODY = {
    "searchMode": "BOTH",
    "searchArticlesBy": "STANDARD",
    "selectedArticleSources": {
        "closedCatalogVersionIds": [],
        "catalogVersionIdsOrderedByPreference": [
            "866b4863-8799-4046-9e84-0985a665c1c7",
        ],
        "sourcePriceListIds": [
            "51a9dedc-efad-469b-8c81-33676f85630e",
            "fb6b5b83-3f9d-45d5-bab6-91dac4446183",
        ],
    },
    "maxDeliveryTime": 0,
    "coreSortimentOnly": False,
    "closedMarketplaceOnly": False,
    "currency": "EUR",
    "explain": False,
}

# Allowed keys per schema (additionalProperties: false)
METADATA_KEYS = {"page", "pageSize", "pageCount", "hitCount", "term"}
ARTICLE_KEYS = {"articleId", "explanation"}
SUMMARIES_KEYS = {
    "vendorSummaries",
    "manufacturerSummaries",
    "featureSummaries",
    "pricesSummary",
    "categoriesSummary",
    "eClass5Categories",
    "eClass7Categories",
    "s2ClassCategories",
    "eClassesAggregations",
}


def _post(body: dict, *, page_size: int = 10,
          sort: list[str] | None = None) -> httpx.Response:
    params: dict = {"page": 1, "pageSize": page_size}
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
# pageSize validation (spec: minimum 1, maximum 500)
# =========================================================================


class TestPageSizeValidation:
    """Spec: pageSize has minimum=1, maximum=500."""

    def test_page_size_zero_rejected(self):
        """pageSize=0 is below minimum=1."""
        r = _post(BASE_BODY, page_size=0)
        assert r.status_code == 400

    def test_page_size_negative_rejected(self):
        """pageSize=-1 is below minimum=1."""
        r = _post(BASE_BODY, page_size=-1)
        assert r.status_code == 400

    def test_page_size_one_accepted(self):
        """pageSize=1 is the minimum valid value."""
        r = _post(BASE_BODY, page_size=1)
        assert r.status_code == 200

    def test_page_size_500_accepted(self):
        """pageSize=500 is the maximum valid value."""
        r = _post(BASE_BODY, page_size=500)
        assert r.status_code == 200

    def test_page_size_501_rejected(self):
        """pageSize=501 exceeds maximum=500."""
        r = _post(BASE_BODY, page_size=501)
        assert r.status_code == 400


# =========================================================================
# Error envelope format (spec: {message, details, timestamp}, no extras)
# =========================================================================


class TestErrorEnvelopeFormat:
    """Spec Error schema: {message: str, details: [str], timestamp: str(date-time)}, additionalProperties: false."""

    def test_missing_currency_returns_exact_error_keys(self):
        """Missing required field produces error with exactly the spec keys."""
        body = {k: v for k, v in BASE_BODY.items() if k != "currency"}
        r = _post(body)
        assert r.status_code == 400
        data = r.json()
        assert set(data.keys()) == {"message", "details", "timestamp"}

    def test_error_details_items_are_strings(self):
        """Spec: details items are type: string, not objects."""
        body = {k: v for k, v in BASE_BODY.items() if k != "currency"}
        r = _post(body)
        assert r.status_code == 400
        data = r.json()
        for item in data["details"]:
            assert isinstance(item, str), f"details item is {type(item).__name__}, expected str"

    def test_error_timestamp_is_iso8601(self):
        """Spec: timestamp is format: date-time (ISO 8601)."""
        body = {k: v for k, v in BASE_BODY.items() if k != "currency"}
        r = _post(body)
        assert r.status_code == 400
        ts = r.json()["timestamp"]
        # Must parse as ISO 8601
        datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def test_invalid_sort_returns_error_envelope(self):
        """Invalid sort param produces the spec error envelope shape."""
        r = _post(BASE_BODY, sort=["bogus,sideways"])
        assert r.status_code == 400
        data = r.json()
        assert set(data.keys()) == {"message", "details", "timestamp"}


# =========================================================================
# searchMode required (spec: required field, enum)
# =========================================================================


class TestSearchModeRequired:
    """Spec: searchMode is required with enum [HITS_ONLY, SUMMARIES_ONLY, BOTH]."""

    def test_omit_search_mode_rejected(self):
        """Omitting required searchMode produces 400."""
        body = {k: v for k, v in BASE_BODY.items() if k != "searchMode"}
        r = _post(body)
        assert r.status_code == 400

    def test_search_mode_both(self):
        """searchMode=BOTH is valid."""
        r = _post(BASE_BODY)
        assert r.status_code == 200

    def test_search_mode_hits_only(self):
        """searchMode=HITS_ONLY is valid."""
        body = {**BASE_BODY, "searchMode": "HITS_ONLY"}
        r = _post(body)
        assert r.status_code == 200

    def test_search_mode_summaries_only(self):
        """searchMode=SUMMARIES_ONLY is valid."""
        body = {**BASE_BODY, "searchMode": "SUMMARIES_ONLY"}
        r = _post(body)
        assert r.status_code == 200

    def test_invalid_search_mode_rejected(self):
        """Non-enum searchMode value produces 400."""
        body = {**BASE_BODY, "searchMode": "INVALID_MODE"}
        r = _post(body)
        assert r.status_code == 400


# =========================================================================
# Duplicate sort params
# =========================================================================


class TestDuplicateSortParams:
    """Repeatable sort query param edge cases."""

    def test_duplicate_sort_does_not_crash(self):
        """Duplicate sort=price,asc&sort=price,asc returns a valid status."""
        r = _post(BASE_BODY, sort=["price,asc", "price,asc"])
        assert r.status_code in (200, 400)

    def test_multi_key_sort_returns_valid_response(self):
        """Multi-key sort=price,asc&sort=name,desc returns a valid status."""
        r = _post(BASE_BODY, sort=["price,asc", "name,desc"])
        assert r.status_code in (200, 400)
        if r.status_code == 200:
            data = r.json()
            assert set(data.keys()) == {"articles", "summaries", "metadata"}


# =========================================================================
# Response structure (additionalProperties: false throughout)
# =========================================================================


class TestResponseStructure:
    """Verify response keys match the spec exactly (no extra keys)."""

    @pytest.fixture()
    def success_response(self) -> dict:
        """Fetch a 200 response for structural assertions."""
        r = _post(BASE_BODY)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        return r.json()

    def test_top_level_keys(self, success_response):
        """Response has exactly {articles, summaries, metadata}."""
        assert set(success_response.keys()) == {"articles", "summaries", "metadata"}

    def test_metadata_keys(self, success_response):
        """Metadata has only spec-allowed keys."""
        meta = success_response["metadata"]
        assert set(meta.keys()) <= METADATA_KEYS, (
            f"Extra metadata keys: {set(meta.keys()) - METADATA_KEYS}"
        )

    def test_article_item_keys(self, success_response):
        """Article items have only {articleId} or {articleId, explanation}."""
        for art in success_response["articles"]:
            assert set(art.keys()) <= ARTICLE_KEYS, (
                f"Extra article keys: {set(art.keys()) - ARTICLE_KEYS}"
            )

    def test_summaries_keys(self, success_response):
        """Summaries has only spec-allowed keys."""
        summ = success_response["summaries"]
        assert set(summ.keys()) <= SUMMARIES_KEYS, (
            f"Extra summaries keys: {set(summ.keys()) - SUMMARIES_KEYS}"
        )
