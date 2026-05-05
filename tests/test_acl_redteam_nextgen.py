"""Red-team tests derived from next-gen legacy codebase analysis.

Each test exposes a behavioral mismatch between the ACL and the legacy
article-search service that next-gen callers depend on. Findings from
reading the next-gen Java/Kotlin code at ../next-gen/article/search/.

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
    return httpx.post(SEARCH_URL, json=body, params=params, timeout=10)


@pytest.fixture(scope="session", autouse=True)
def _check_services():
    try:
        r = httpx.get(f"{ACL_BASE}/healthz", timeout=3)
        assert r.status_code == 200
    except Exception:
        pytest.skip("ACL not running on localhost:8081")


# =========================================================================
# Finding 12: pageCount must be >= 1 even for empty results
# =========================================================================
# Legacy: Math.max(1, articles.getTotalPages())
# Portal-bff uses pageCount for pagination UI.


class TestPageCountMinimum:
    """Legacy always returns pageCount >= 1. The portal-bff depends on this."""

    def test_empty_results_pagecount_at_least_1(self):
        """Empty catalog scope → zero hits, but pageCount must be >= 1."""
        body = _base_body()
        body["selectedArticleSources"]["catalogVersionIdsOrderedByPreference"] = []
        r = _post(body)
        assert r.status_code == 200
        md = r.json()["metadata"]
        assert md["hitCount"] == 0
        assert md["pageCount"] >= 1, (
            f"pageCount={md['pageCount']} for empty results; "
            f"legacy always returns >= 1"
        )


# =========================================================================
# Finding 3: Response page must be 1-based
# =========================================================================
# Legacy: .page(articles.getNumber() + 1)
# Portal-bff: PageRequest.of(metadata.getPage() - 1, ...)
# If page=0, portal-bff creates PageRequest.of(-1, ...) → crash.


class TestPageNumbering:
    """Response page must be 1-based to match legacy behavior."""

    def test_first_page_returns_page_1(self):
        r = _post(_base_body(), page=1)
        assert r.status_code == 200
        assert r.json()["metadata"]["page"] == 1

    def test_second_page_returns_page_2(self):
        r = _post(_base_body(), page=2)
        assert r.status_code == 200
        assert r.json()["metadata"]["page"] == 2

    def test_page_never_zero(self):
        """Even edge cases should never produce page=0."""
        body = _base_body()
        body["selectedArticleSources"]["catalogVersionIdsOrderedByPreference"] = []
        r = _post(body, page=1)
        assert r.status_code == 200
        assert r.json()["metadata"]["page"] >= 1


# =========================================================================
# Finding 7: articleId format — colon-split with exactly 2 parts
# =========================================================================
# Legacy ArticleId.fromString uses StringUtils.split(value, ':')
# expecting exactly 2 parts: {friendlyId}:{base64UrlEncodedArticleNumber}
# If the ACL returns 3+ colon-separated segments, parsing breaks.


class TestArticleIdFormat:
    """articleId must have exactly one colon: {part1}:{part2}."""

    def test_article_id_has_exactly_two_colon_parts(self):
        """Portal-bff's ArticleId.fromString splits on ':' and expects
        exactly 2 parts. More than one colon breaks the parser."""
        r = _post(_base_body(), page_size=20)
        assert r.status_code == 200
        for art in r.json()["articles"]:
            aid = art["articleId"]
            parts = aid.split(":")
            assert len(parts) == 2, (
                f"articleId {aid!r} has {len(parts)} colon-separated parts; "
                f"legacy expects exactly 2 (friendlyId:base64ArticleNumber)"
            )


# =========================================================================
# Finding 14: Empty catalogVersionIdsOrderedByPreference → empty results
# =========================================================================
# Legacy returns Page.empty() silently. ACL must not error.


class TestEmptyCatalogScope:
    """Empty catalog scope must return zero results gracefully, not error."""

    def test_empty_catalog_version_ids_returns_200(self):
        body = _base_body()
        body["selectedArticleSources"]["catalogVersionIdsOrderedByPreference"] = []
        r = _post(body)
        assert r.status_code == 200
        assert r.json()["metadata"]["hitCount"] == 0

    def test_empty_source_price_list_ids_returns_200(self):
        body = _base_body()
        body["selectedArticleSources"]["sourcePriceListIds"] = []
        r = _post(body)
        assert r.status_code == 200
