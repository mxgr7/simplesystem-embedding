"""Edge-case tests for the dedup-topology routing logic.

Red-team tests targeting sort interactions, browse mode, dedup behaviour,
score semantics, hitCount accuracy, SUMMARIES_ONLY mode, and large
catalog version lists. All tests run against the live search-api at
localhost:8001 with USE_DEDUP_TOPOLOGY=1.
"""

from __future__ import annotations

import httpx
import pytest

SEARCH_URL = "http://localhost:8001/offers_v6/_search"

# Catalog version with live data.
CV = "866b4863-8799-4046-9e84-0985a665c1c7"

# A real sourcePriceListId present in offers_v6.
REAL_PRICE_LIST_ID = "51a9dedc-efad-469b-8c81-33676f85630e"


def _base_body(**overrides):
    body = {
        "searchMode": "BOTH",
        "selectedArticleSources": {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV],
        },
        "maxDeliveryTime": 0,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
        "currency": "EUR",
    }
    body.update(overrides)
    return body


@pytest.fixture(scope="module")
def client():
    c = httpx.Client(base_url="http://localhost:8001", timeout=30)
    try:
        r = c.get("/openapi.yaml")
        if r.status_code != 200:
            pytest.skip("search-api not reachable at localhost:8001")
    except httpx.ConnectError:
        pytest.skip("search-api not reachable at localhost:8001")
    yield c
    c.close()


# ──────────────────────────────────────────────────────────────────────
# 1. Sort interactions: sort=name,asc stability with identical names
# ──────────────────────────────────────────────────────────────────────

class TestSortStability:
    """When articles share the same name, sort=name,asc must be stable
    (tiebreak on articleId,asc). Running the same request twice must
    produce the same order."""

    def test_name_sort_deterministic_across_runs(self, client):
        """Two identical sort=name,asc requests must return articles in
        the same order, proving a stable tiebreak exists."""
        body = _base_body(query="screw")
        params = {"sort": "name,asc", "pageSize": 50}
        r1 = client.post("/offers_v6/_search", params=params, json=body)
        r2 = client.post("/offers_v6/_search", params=params, json=body)
        assert r1.status_code == 200
        assert r2.status_code == 200
        ids1 = [a["articleId"] for a in r1.json()["articles"]]
        ids2 = [a["articleId"] for a in r2.json()["articles"]]
        assert ids1 == ids2, "sort=name,asc is not deterministic across repeated requests"

    def test_name_sort_asc_order_valid(self, client):
        """Articles returned by sort=name,asc should not have a name
        that sorts after a subsequent name (case-insensitive)."""
        # Use browse mode to get articles purely by name
        body = _base_body()
        params = {"sort": "name,asc", "pageSize": 50}
        r = client.post("/offers_v6/_search", params=params, json=body)
        assert r.status_code == 200
        articles = r.json()["articles"]
        # We cannot check the name from the response (articles only carry
        # articleId + score), but we verify ordering is consistent with
        # re-fetching: page 1 IDs should not appear in page 2 and vice versa.
        if len(articles) >= 2:
            ids_page1 = {a["articleId"] for a in articles}
            r2 = client.post(
                "/offers_v6/_search",
                params={"sort": "name,asc", "pageSize": 50, "page": 2},
                json=body,
            )
            ids_page2 = {a["articleId"] for a in r2.json()["articles"]}
            assert not ids_page1 & ids_page2, "page 1 and page 2 share articles under sort=name,asc"

    def test_name_sort_desc_reverses_asc(self, client):
        """sort=name,desc should produce the opposite ordering to asc."""
        body = _base_body(query="screw")
        r_asc = client.post(
            "/offers_v6/_search",
            params={"sort": "name,asc", "pageSize": 20},
            json=body,
        )
        r_desc = client.post(
            "/offers_v6/_search",
            params={"sort": "name,desc", "pageSize": 20},
            json=body,
        )
        assert r_asc.status_code == 200
        assert r_desc.status_code == 200
        ids_asc = [a["articleId"] for a in r_asc.json()["articles"]]
        ids_desc = [a["articleId"] for a in r_desc.json()["articles"]]
        # They should not be identical unless there is only 1 result.
        if len(ids_asc) > 1:
            assert ids_asc != ids_desc, "asc and desc sort=name produce identical order"


# ──────────────────────────────────────────────────────────────────────
# 2. Empty query with filters: browse mode + various filter combos
# ──────────────────────────────────────────────────────────────────────

class TestBrowseModeFilters:
    """Browse mode (no query string) should still return results when
    at least one filter narrows the set."""

    def test_browse_no_filters_returns_results(self, client):
        """No query, no filters except catalog version. Should return
        articles (browse the whole catalog)."""
        body = _base_body()
        r = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body)
        assert r.status_code == 200
        data = r.json()
        # With a valid catalog version the hitCount should be > 0
        assert data["metadata"]["hitCount"] > 0

    def test_browse_with_vendor_filter(self, client):
        """Browse + vendorIdsFilter should narrow results."""
        # First get some results to discover a vendor
        body = _base_body(query="screw")
        r = client.post("/offers_v6/_search", params={"pageSize": 5}, json=body)
        assert r.status_code == 200
        # Now browse with a made-up vendor — should return 0 results
        body = _base_body(vendorIdsFilter=["nonexistent-vendor-id-xyz"])
        r2 = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body)
        assert r2.status_code == 200
        assert r2.json()["metadata"]["hitCount"] == 0

    def test_browse_with_category_filter(self, client):
        """Browse + currentCategoryPathElements narrows results."""
        body = _base_body(
            currentCategoryPathElements=["nonexistent_cat_path_element"],
        )
        r = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body)
        assert r.status_code == 200
        assert r.json()["metadata"]["hitCount"] == 0

    def test_browse_empty_string_query_is_browse(self, client):
        """query="" should behave identically to query=null (browse mode)."""
        body_null = _base_body()
        body_empty = _base_body(query="")
        r_null = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body_null)
        r_empty = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body_empty)
        assert r_null.status_code == 200
        assert r_empty.status_code == 200
        assert r_null.json()["metadata"]["hitCount"] == r_empty.json()["metadata"]["hitCount"]

    def test_browse_whitespace_query_is_browse(self, client):
        """query="   " (whitespace only) should be treated as browse."""
        body = _base_body(query="   ")
        r = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body)
        assert r.status_code == 200
        # Should have results like browse mode (not query mode)
        data = r.json()
        # In browse mode, scores should be null
        for a in data["articles"]:
            assert a["score"] is None, f"article {a['articleId']} has non-null score in browse mode"


# ──────────────────────────────────────────────────────────────────────
# 3. Dedup behavior: same article from multiple catalog versions
# ──────────────────────────────────────────────────────────────────────

class TestDedupBehavior:
    """The dedup topology de-duplicates on article_hash. If the same
    article exists in multiple catalog versions within the preference
    list, it should appear only once in results."""

    def test_no_duplicate_article_ids_in_page(self, client):
        """A single page should never contain the same articleId twice."""
        body = _base_body(query="screw")
        r = client.post("/offers_v6/_search", params={"pageSize": 100}, json=body)
        assert r.status_code == 200
        ids = [a["articleId"] for a in r.json()["articles"]]
        assert len(ids) == len(set(ids)), "duplicate articleIds found in a single page"

    def test_dedup_across_pages(self, client):
        """Articles on page 1 should not re-appear on page 2."""
        body = _base_body(query="screw")
        r1 = client.post("/offers_v6/_search", params={"pageSize": 20, "page": 1}, json=body)
        r2 = client.post("/offers_v6/_search", params={"pageSize": 20, "page": 2}, json=body)
        assert r1.status_code == 200
        assert r2.status_code == 200
        ids1 = {a["articleId"] for a in r1.json()["articles"]}
        ids2 = {a["articleId"] for a in r2.json()["articles"]}
        overlap = ids1 & ids2
        assert not overlap, f"articles appear on both page 1 and 2: {overlap}"

    def test_multiple_catalog_versions_same_article_appears_once(self, client):
        """Sending the same CV duplicated in the preference list should
        not multiply results."""
        body_single = _base_body(query="screw")
        body_double = _base_body(
            query="screw",
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV, CV],
            },
        )
        r1 = client.post("/offers_v6/_search", params={"pageSize": 50}, json=body_single)
        r2 = client.post("/offers_v6/_search", params={"pageSize": 50}, json=body_double)
        assert r1.status_code == 200
        assert r2.status_code == 200
        # hitCount should be the same — dedup on article_hash
        assert r1.json()["metadata"]["hitCount"] == r2.json()["metadata"]["hitCount"]


# ──────────────────────────────────────────────────────────────────────
# 4. Score semantics: null when no query, ordered when query present
# ──────────────────────────────────────────────────────────────────────

class TestScoreSemantics:
    """Score is null in browse/non-relevance sort, and monotonically
    decreasing in relevance+query mode."""

    def test_browse_mode_scores_are_null(self, client):
        """Browse mode (no query) should emit score=null for all articles."""
        body = _base_body()
        r = client.post("/offers_v6/_search", params={"pageSize": 20}, json=body)
        assert r.status_code == 200
        for a in r.json()["articles"]:
            assert a["score"] is None, f"article {a['articleId']} score should be null in browse"

    def test_non_relevance_sort_scores_are_null(self, client):
        """sort=name,asc + query present should still emit score=null
        (non-relevance sort nulls scores per spec)."""
        body = _base_body(query="screw")
        r = client.post(
            "/offers_v6/_search",
            params={"sort": "name,asc", "pageSize": 20},
            json=body,
        )
        assert r.status_code == 200
        for a in r.json()["articles"]:
            assert a["score"] is None, f"non-relevance sort should emit null score, got {a['score']}"

    def test_relevance_sort_with_query_scores_ordered(self, client):
        """Default sort (relevance) + query should produce descending scores."""
        body = _base_body(query="screw")
        r = client.post("/offers_v6/_search", params={"pageSize": 50}, json=body)
        assert r.status_code == 200
        articles = r.json()["articles"]
        scores = [a["score"] for a in articles]
        assert all(s is not None for s in scores), "relevance sort with query should have non-null scores"
        # Scores should be non-increasing
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"scores not monotonically descending at position {i}: "
                f"{scores[i]} < {scores[i+1]}"
            )

    def test_relevance_scores_are_positive(self, client):
        """RRF scores should be > 0 for all returned articles."""
        body = _base_body(query="screw")
        r = client.post("/offers_v6/_search", params={"pageSize": 20}, json=body)
        assert r.status_code == 200
        for a in r.json()["articles"]:
            assert a["score"] is not None and a["score"] > 0, (
                f"article {a['articleId']} has non-positive score: {a['score']}"
            )


# ──────────────────────────────────────────────────────────────────────
# 5. hitCount accuracy: hitCount vs actual articles across pages
# ──────────────────────────────────────────────────────────────────────

class TestHitCountAccuracy:
    """hitCount should be >= the total number of articles retrievable
    across all pages (unless hitCountClipped). It should also equal
    the page iteration sum when not clipped."""

    def test_hitcount_ge_page_articles(self, client):
        """hitCount >= len(articles) on the first page."""
        body = _base_body(query="screw")
        r = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["metadata"]["hitCount"] >= len(data["articles"])

    def test_hitcount_consistent_with_pagecount(self, client):
        """pageCount should equal ceil(hitCount / pageSize)."""
        body = _base_body(query="screw")
        page_size = 10
        r = client.post("/offers_v6/_search", params={"pageSize": page_size}, json=body)
        assert r.status_code == 200
        meta = r.json()["metadata"]
        expected_page_count = (meta["hitCount"] + page_size - 1) // page_size if meta["hitCount"] > 0 else 0
        assert meta["pageCount"] == expected_page_count, (
            f"pageCount {meta['pageCount']} != ceil(hitCount={meta['hitCount']} / pageSize={page_size}) = {expected_page_count}"
        )

    def test_hitcount_matches_article_sum_across_pages(self, client):
        """Iterate pages until empty: total article count should match
        hitCount (when not clipped). Limited to first 5 pages for speed."""
        body = _base_body(query="screw")
        page_size = 50
        r = client.post("/offers_v6/_search", params={"pageSize": page_size, "page": 1}, json=body)
        assert r.status_code == 200
        data = r.json()
        hit_count = data["metadata"]["hitCount"]
        if data["metadata"].get("hitCountClipped"):
            pytest.skip("hitCount clipped, cannot verify exact count")
        total = len(data["articles"])
        page = 2
        max_pages = min(5, data["metadata"]["pageCount"])
        while page <= max_pages:
            r = client.post(
                "/offers_v6/_search",
                params={"pageSize": page_size, "page": page},
                json=body,
            )
            assert r.status_code == 200
            articles = r.json()["articles"]
            if not articles:
                break
            total += len(articles)
            page += 1
        # If we iterated all pages, total should equal hitCount
        if page > data["metadata"]["pageCount"]:
            assert total == hit_count, (
                f"sum of articles across pages ({total}) != hitCount ({hit_count})"
            )

    def test_last_page_not_overfilled(self, client):
        """The last page should have <= pageSize articles."""
        body = _base_body(query="screw")
        page_size = 10
        r = client.post("/offers_v6/_search", params={"pageSize": page_size, "page": 1}, json=body)
        assert r.status_code == 200
        meta = r.json()["metadata"]
        if meta["pageCount"] == 0:
            pytest.skip("no pages")
        r_last = client.post(
            "/offers_v6/_search",
            params={"pageSize": page_size, "page": meta["pageCount"]},
            json=body,
        )
        assert r_last.status_code == 200
        assert len(r_last.json()["articles"]) <= page_size

    def test_page_beyond_pagecount_is_empty(self, client):
        """Requesting a page beyond pageCount should return 0 articles."""
        body = _base_body(query="screw")
        r = client.post("/offers_v6/_search", params={"pageSize": 10, "page": 1}, json=body)
        assert r.status_code == 200
        page_count = r.json()["metadata"]["pageCount"]
        r_beyond = client.post(
            "/offers_v6/_search",
            params={"pageSize": 10, "page": page_count + 1},
            json=body,
        )
        assert r_beyond.status_code == 200
        assert len(r_beyond.json()["articles"]) == 0


# ──────────────────────────────────────────────────────────────────────
# 6. SUMMARIES_ONLY mode: skips rank/paginate, returns empty hits
# ──────────────────────────────────────────────────────────────────────

class TestSummariesOnlyMode:
    """SUMMARIES_ONLY should short-circuit the rank/sort/page path,
    return empty articles[], but still provide hitCount and summaries."""

    def test_summaries_only_returns_empty_articles(self, client):
        """SUMMARIES_ONLY returns zero articles regardless of data."""
        body = _base_body(
            query="screw",
            searchMode="SUMMARIES_ONLY",
            summaries=["VENDORS"],
        )
        r = client.post("/offers_v6/_search", params={"pageSize": 50}, json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["articles"] == [], "SUMMARIES_ONLY should return empty articles"

    def test_summaries_only_has_hitcount(self, client):
        """SUMMARIES_ONLY still reports a non-zero hitCount."""
        body = _base_body(
            query="screw",
            searchMode="SUMMARIES_ONLY",
            summaries=["VENDORS"],
        )
        r = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body)
        assert r.status_code == 200
        assert r.json()["metadata"]["hitCount"] > 0

    def test_summaries_only_has_vendor_summaries(self, client):
        """SUMMARIES_ONLY with summaries=[VENDORS] returns vendor data."""
        body = _base_body(
            query="screw",
            searchMode="SUMMARIES_ONLY",
            summaries=["VENDORS"],
        )
        r = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body)
        assert r.status_code == 200
        summaries = r.json()["summaries"]
        assert len(summaries.get("vendorSummaries", [])) > 0, (
            "SUMMARIES_ONLY with VENDORS requested should return vendorSummaries"
        )

    def test_summaries_only_browse_mode(self, client):
        """SUMMARIES_ONLY without a query (browse) should still work."""
        body = _base_body(
            searchMode="SUMMARIES_ONLY",
            summaries=["VENDORS"],
        )
        r = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["articles"] == []
        assert data["metadata"]["hitCount"] >= 0

    def test_summaries_only_does_not_change_with_sort(self, client):
        """SUMMARIES_ONLY should ignore sort param (no ranking happens)."""
        body = _base_body(
            query="screw",
            searchMode="SUMMARIES_ONLY",
            summaries=["VENDORS"],
        )
        r_default = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body)
        r_sorted = client.post(
            "/offers_v6/_search",
            params={"pageSize": 10, "sort": "name,asc"},
            json=body,
        )
        assert r_default.status_code == 200
        assert r_sorted.status_code == 200
        # hitCount should be the same regardless of sort
        assert (
            r_default.json()["metadata"]["hitCount"]
            == r_sorted.json()["metadata"]["hitCount"]
        )

    def test_summaries_only_with_filter(self, client):
        """SUMMARIES_ONLY + filter should respect the filter."""
        body = _base_body(
            searchMode="SUMMARIES_ONLY",
            summaries=["VENDORS"],
            vendorIdsFilter=["nonexistent-vendor-xyz"],
        )
        r = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body)
        assert r.status_code == 200
        assert r.json()["metadata"]["hitCount"] == 0


# ──────────────────────────────────────────────────────────────────────
# 7. Large catalog version list: 50+ IDs in the preference list
# ──────────────────────────────────────────────────────────────────────

class TestLargeCatalogVersionList:
    """The preference list can hold many catalog version IDs. The system
    should not crash or return errors with large lists."""

    def test_50_catalog_versions_does_not_crash(self, client):
        """50+ catalogVersionIds in preference list should still return
        200 (most IDs will be no-ops if not loaded, but shouldn't crash)."""
        # Generate 50 fake CVs + the real one
        fake_cvs = [f"00000000-0000-0000-0000-{i:012d}" for i in range(49)]
        all_cvs = [CV] + fake_cvs
        body = _base_body(
            query="screw",
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": all_cvs,
            },
        )
        r = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body)
        assert r.status_code == 200
        # Should still find results from the one real CV
        assert r.json()["metadata"]["hitCount"] > 0

    def test_100_catalog_versions_in_closed_list(self, client):
        """100 closedCatalogVersionIds should not crash."""
        fake_closed = [f"10000000-0000-0000-0000-{i:012d}" for i in range(100)]
        body = _base_body(
            query="screw",
            selectedArticleSources={
                "closedCatalogVersionIds": fake_closed,
                "catalogVersionIdsOrderedByPreference": [CV],
            },
        )
        r = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body)
        assert r.status_code == 200

    def test_large_preference_list_results_match_single(self, client):
        """Results with CV + 49 non-existent CVs should match results
        with just CV (the fakes don't have data)."""
        fake_cvs = [f"20000000-0000-0000-0000-{i:012d}" for i in range(49)]
        body_single = _base_body(query="screw")
        body_large = _base_body(
            query="screw",
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV] + fake_cvs,
            },
        )
        r1 = client.post("/offers_v6/_search", params={"pageSize": 20}, json=body_single)
        r2 = client.post("/offers_v6/_search", params={"pageSize": 20}, json=body_large)
        assert r1.status_code == 200
        assert r2.status_code == 200
        # hitCount should be the same since fakes have no data
        assert r1.json()["metadata"]["hitCount"] == r2.json()["metadata"]["hitCount"]

    def test_empty_preference_list_returns_nothing(self, client):
        """An empty catalogVersionIdsOrderedByPreference should match nothing
        (match_nothing sentinel fires)."""
        body = _base_body(
            query="screw",
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [],
            },
        )
        r = client.post("/offers_v6/_search", params={"pageSize": 10}, json=body)
        assert r.status_code == 200
        assert r.json()["metadata"]["hitCount"] == 0
        assert r.json()["articles"] == []
