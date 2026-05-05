"""Red-team tests for the search-api (ftsearch) service.

Each test targets a real bug confirmed by probing the running service at
http://localhost:8001.  All tests use the dedup topology
(USE_DEDUP_TOPOLOGY=1) against the live articles_v6 + offers_v6
collections.

Bugs found
----------

1.  **Page overflow → 500**
    ``page * pageSize > 16384`` with a query string causes
    ``_rank_limit`` to exceed Milvus's ``proxy.maxResultWindow``
    (default 16384) on the ``search()`` call, producing an unhandled
    500 instead of an empty page or a 400.

2.  **hitCount / pageCount ignore price-sort drops**
    ``sort=price`` drops articles whose representative offer has no
    resolved price (by design — matches legacy ES).  But ``hitCount``
    and ``pageCount`` are computed from the article-expr count *before*
    the price-sort materialisation drops, so the response advertises
    pages that are empty.

3.  **Whitespace-only query leaks raw whitespace into metadata.term**
    ``query="   "`` is treated as browse (correctly — empty after
    strip) but ``metadata.term`` carries the raw whitespace string
    instead of ``null``, which is inconsistent with ``query=null``
    browse behaviour.

4.  **sort=price + empty sourcePriceListIds → 0 articles but non-zero
    pageCount**
    When no ``sourcePriceListIds`` are provided, ``resolve_price``
    returns ``None`` for every offer, so ``pick_representative`` drops
    every article.  All pages are empty, yet ``hitCount`` and
    ``pageCount`` claim results exist.

5.  **Category path depth > 5 silently ignored**
    ``currentCategoryPathElements`` with 6+ elements is silently
    treated as no-filter (returns unfiltered results) instead of
    returning an error.  A caller sending a deep path would get
    unexpectedly broad results.
"""

from __future__ import annotations

import httpx
import pytest

SEARCH_URL = "http://localhost:8001/offers_v6/_search"

# Catalog-version present in the live data.
CV_EUR = "866b4863-8799-4046-9e84-0985a665c1c7"

# A real sourcePriceListId that exists in offers_v6 price entries.
REAL_PRICE_LIST_ID = "51a9dedc-efad-469b-8c81-33676f85630e"


def _base_body(**overrides):
    body = {
        "searchMode": "BOTH",
        "selectedArticleSources": {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV_EUR],
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
# 1. Page overflow → unhandled 500
# ──────────────────────────────────────────────────────────────────────

class TestPageOverflow:
    """page * pageSize > 16384 with a query string causes a 500 because
    _rank_limit feeds the result into Milvus search() which enforces
    proxy.maxResultWindow = 16384."""

    def test_page_times_pagesize_exceeds_milvus_window_crashes(self, client):
        """page=1639, pageSize=10 → rank_limit = max(16390, 200) = 16390
        which exceeds Milvus's 16384 limit."""
        r = client.post(
            "/offers_v6/_search",
            params={"page": 1639, "pageSize": 10},
            json=_base_body(query="screw"),
        )
        # BUG: returns 500 instead of a graceful empty page or 400.
        # A correct implementation would either clamp rank_limit to
        # 16384 and return an empty page, or return 400 with
        # "page out of range".
        assert r.status_code == 500, (
            f"Expected 500 from Milvus overflow, got {r.status_code}"
        )

    def test_max_pagesize_moderate_page_crashes(self, client):
        """pageSize=500 (the schema max), page=33 → rank_limit = 16500.
        This crashes so hard it resets the TCP connection (worse than 500)."""
        try:
            r = client.post(
                "/offers_v6/_search",
                params={"page": 33, "pageSize": 500},
                json=_base_body(query="screw"),
            )
            # If we get a response at all, it should be a 500.
            assert r.status_code == 500
        except (httpx.ReadError, httpx.RemoteProtocolError):
            # BUG: the server crashes so hard the TCP connection resets.
            # This is even worse than a 500 — the client gets no
            # response at all.
            pass

    def test_just_below_boundary_succeeds(self, client):
        """page=1638, pageSize=10 → rank_limit = max(16380, 200) = 16380
        which is within the 16384 window."""
        r = client.post(
            "/offers_v6/_search",
            params={"page": 1638, "pageSize": 10},
            json=_base_body(query="screw"),
        )
        assert r.status_code == 200

    def test_browse_mode_does_not_crash_on_high_page(self, client):
        """Browse (no query) uses rank_limit = hitcount_cap, so it
        doesn't hit the Milvus search() limit."""
        r = client.post(
            "/offers_v6/_search",
            params={"page": 1639, "pageSize": 10},
            json=_base_body(),
        )
        assert r.status_code == 200

    def test_non_relevance_sort_does_not_crash_on_high_page(self, client):
        """Non-relevance sort + query uses rank_limit =
        relevance_pool_max (200), so it doesn't overflow."""
        r = client.post(
            "/offers_v6/_search",
            params={"page": 99999, "pageSize": 10, "sort": "name,asc"},
            json=_base_body(query="screw"),
        )
        assert r.status_code == 200


# ──────────────────────────────────────────────────────────────────────
# 2. hitCount / pageCount ignore price-sort drops
# ──────────────────────────────────────────────────────────────────────

class TestPriceSortHitCountMismatch:
    """sort=price drops articles without a resolved price from the page,
    but hitCount and pageCount are computed pre-drop."""

    def test_price_sort_hitcount_exceeds_displayable_articles(self, client):
        """sort=price,asc with real sourcePriceListIds.  Some articles may
        not have prices under this price list, so they drop from the page.
        hitCount should reflect the displayable (price-having) count, but
        it reflects the article-expr count instead."""
        r = client.post(
            "/offers_v6/_search",
            params={"sort": "price,asc", "pageSize": 500},
            json=_base_body(
                query="screw",
                selectedArticleSources={
                    "closedCatalogVersionIds": [],
                    "catalogVersionIdsOrderedByPreference": [CV_EUR],
                    "sourcePriceListIds": [REAL_PRICE_LIST_ID],
                },
            ),
        )
        assert r.status_code == 200
        data = r.json()
        articles = data["articles"]
        hit_count = data["metadata"]["hitCount"]
        # BUG: if any articles were dropped by price-sort (no resolved
        # price), hitCount overstates the displayable count.
        # In the test data, all 113 articles happen to have prices under
        # the real price list, so this test documents the structural bug
        # rather than necessarily catching it on this dataset. The next
        # test (empty sourcePriceListIds) demonstrates the extreme case.
        if len(articles) < hit_count:
            pytest.fail(
                f"hitCount={hit_count} but only {len(articles)} articles "
                f"have resolved prices for sort=price — hitCount is wrong"
            )

    def test_price_sort_empty_pricelists_zero_articles_but_nonzero_hitcount(self, client):
        """sort=price,asc with NO sourcePriceListIds: resolve_price
        returns None for all offers, so every article is dropped by
        pick_representative. Yet hitCount and pageCount claim results."""
        r = client.post(
            "/offers_v6/_search",
            params={"sort": "price,asc", "pageSize": 5},
            json=_base_body(query="screw"),
        )
        assert r.status_code == 200
        data = r.json()
        articles = data["articles"]
        hit_count = data["metadata"]["hitCount"]
        page_count = data["metadata"]["pageCount"]

        # BUG: articles is empty, but hitCount > 0 and pageCount > 0.
        assert len(articles) == 0, "Expected 0 articles (no prices resolve)"
        assert hit_count > 0, f"Expected hitCount > 0, got {hit_count}"
        assert page_count > 0, f"Expected pageCount > 0, got {page_count}"
        # The combination articles=[] + pageCount>0 is the bug:
        # a UI paging through would see empty pages everywhere.

    def test_price_sort_browse_also_mismatches(self, client):
        """Browse (no query) + sort=price,asc has the same bug."""
        r = client.post(
            "/offers_v6/_search",
            params={"sort": "price,asc", "pageSize": 3},
            json=_base_body(
                selectedArticleSources={
                    "closedCatalogVersionIds": [],
                    "catalogVersionIdsOrderedByPreference": [CV_EUR],
                    "sourcePriceListIds": [REAL_PRICE_LIST_ID],
                },
            ),
        )
        assert r.status_code == 200
        data = r.json()
        articles = data["articles"]
        hit_count = data["metadata"]["hitCount"]
        page_count = data["metadata"]["pageCount"]

        # hitCount counts articles matching article_expr, not the subset
        # that survives price resolution. If any articles have no price,
        # hitCount overstates.
        # On this dataset every article has a price under the test price
        # list, so len(articles) == min(3, price_surviving). Document the
        # structural issue: pageCount is based on hitCount, not the price-
        # filtered count.
        if len(articles) == 0 and page_count > 0:
            pytest.fail(
                f"0 articles returned but pageCount={page_count} — "
                f"hitCount={hit_count} ignores price drops"
            )


# ──────────────────────────────────────────────────────────────────────
# 3. Whitespace-only query → metadata.term not null
# ──────────────────────────────────────────────────────────────────────

class TestWhitespaceQuery:
    """query="   " is stripped to "" and treated as browse. But
    metadata.term preserves the raw whitespace instead of null."""

    def test_whitespace_query_term_is_raw_whitespace(self, client):
        r = client.post(
            "/offers_v6/_search",
            json=_base_body(query="   "),
        )
        assert r.status_code == 200
        data = r.json()
        term = data["metadata"]["term"]
        # BUG: term should be null (matching query=null browse behaviour)
        # but it's "   " — the raw unstripped input.
        assert term == "   ", f"Expected raw whitespace, got {term!r}"
        # The correct behaviour would be term=null or term=""

    def test_null_query_term_is_null(self, client):
        """Baseline: query=null → term=null."""
        r = client.post(
            "/offers_v6/_search",
            json=_base_body(),  # no query key → null
        )
        assert r.status_code == 200
        assert r.json()["metadata"]["term"] is None

    def test_whitespace_query_returns_browse_results(self, client):
        """query="   " should behave identically to query=null."""
        r_ws = client.post(
            "/offers_v6/_search",
            json=_base_body(query="   "),
        )
        r_null = client.post(
            "/offers_v6/_search",
            json=_base_body(),
        )
        assert r_ws.status_code == 200
        assert r_null.status_code == 200
        ws_hit = r_ws.json()["metadata"]["hitCount"]
        null_hit = r_null.json()["metadata"]["hitCount"]
        assert ws_hit == null_hit, (
            f"Whitespace query hitCount={ws_hit} != null query hitCount={null_hit}"
        )
        # But their term fields differ — that's the bug.
        assert r_ws.json()["metadata"]["term"] != r_null.json()["metadata"]["term"], (
            "Expected term to differ between whitespace and null query "
            "(this IS the bug — they should be the same)"
        )


# ──────────────────────────────────────────────────────────────────────
# 4. Category path depth > 5 silently ignored
# ──────────────────────────────────────────────────────────────────────

class TestCategoryPathDepth:
    """_category_prefix returns None for depth > 5, silently making the
    filter a no-op. The caller gets unfiltered results."""

    def test_depth_6_returns_unfiltered_results(self, client):
        """6 path elements → filter is silently dropped → same hitCount
        as no-filter browse."""
        r_filtered = client.post(
            "/offers_v6/_search",
            json=_base_body(
                query="screw",
                currentCategoryPathElements=["a", "b", "c", "d", "e", "f"],
            ),
        )
        r_unfiltered = client.post(
            "/offers_v6/_search",
            json=_base_body(query="screw"),
        )
        assert r_filtered.status_code == 200
        assert r_unfiltered.status_code == 200
        filtered_hit = r_filtered.json()["metadata"]["hitCount"]
        unfiltered_hit = r_unfiltered.json()["metadata"]["hitCount"]
        # BUG: depth-6 filter is silently ignored, so hitCount matches
        # the unfiltered count. A real depth-6 category would not match
        # any articles, so the filter should either reject or return 0.
        assert filtered_hit == unfiltered_hit, (
            f"Expected depth-6 to be silently ignored (hitCount={filtered_hit} "
            f"== {unfiltered_hit}), but it wasn't"
        )

    def test_depth_5_is_applied(self, client):
        """5 elements is the max supported depth — filter IS applied."""
        r = client.post(
            "/offers_v6/_search",
            json=_base_body(
                query="screw",
                currentCategoryPathElements=["a", "b", "c", "d", "e"],
            ),
        )
        assert r.status_code == 200
        # Depth 5 with bogus path → 0 results (filter applies).
        assert r.json()["metadata"]["hitCount"] == 0


# ──────────────────────────────────────────────────────────────────────
# 5. Miscellaneous edge cases
# ──────────────────────────────────────────────────────────────────────

class TestMiscEdgeCases:

    def test_pagesize_zero_returns_no_articles_but_hitcount(self, client):
        """pageSize=0 is accepted (schema allows ge=0). Articles should
        be empty but hitCount should reflect the full count."""
        r = client.post(
            "/offers_v6/_search",
            params={"pageSize": 0},
            json=_base_body(query="screw"),
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["articles"]) == 0
        assert data["metadata"]["hitCount"] > 0
        assert data["metadata"]["pageCount"] == 0

    def test_page_beyond_results_returns_empty(self, client):
        """page=100 with only 113 hits at pageSize=10 → empty articles
        but hitCount still accurate."""
        r = client.post(
            "/offers_v6/_search",
            params={"page": 100, "pageSize": 10},
            json=_base_body(query="screw"),
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["articles"]) == 0
        assert data["metadata"]["hitCount"] > 0

    def test_hits_only_skips_summaries(self, client):
        """HITS_ONLY mode should return empty summaries even when
        summaries are requested."""
        r = client.post(
            "/offers_v6/_search",
            json=_base_body(
                query="screw",
                searchMode="HITS_ONLY",
                summaries=["VENDORS", "MANUFACTURERS"],
            ),
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["articles"]) > 0
        assert len(data["summaries"]["vendorSummaries"]) == 0
        assert len(data["summaries"]["manufacturerSummaries"]) == 0

    def test_unknown_field_rejected(self, client):
        """extra='forbid' on the Pydantic model should reject unknown
        top-level fields with a 422."""
        r = client.post(
            "/offers_v6/_search",
            json=_base_body(unknownField="test"),
        )
        assert r.status_code == 422

    def test_invalid_sort_field_returns_400(self, client):
        """Unrecognised sort field should return 400."""
        r = client.post(
            "/offers_v6/_search",
            params={"sort": "INVALID,asc"},
            json=_base_body(),
        )
        assert r.status_code == 400

    def test_sort_without_direction_returns_400(self, client):
        """sort=name (missing direction) should return 400."""
        r = client.post(
            "/offers_v6/_search",
            params={"sort": "name"},
            json=_base_body(),
        )
        assert r.status_code == 400

    def test_lowercase_currency_rejected(self, client):
        """currency must match ^[A-Z]{3}$."""
        r = client.post(
            "/offers_v6/_search",
            json=_base_body(currency="eur"),
        )
        assert r.status_code == 422

    def test_inverted_price_range_returns_zero(self, client):
        """priceFilter.min > priceFilter.max should return 0 results
        (no articles can satisfy the inverted range)."""
        r = client.post(
            "/offers_v6/_search",
            json=_base_body(
                query="screw",
                selectedArticleSources={
                    "closedCatalogVersionIds": [],
                    "catalogVersionIdsOrderedByPreference": [CV_EUR],
                    "sourcePriceListIds": [REAL_PRICE_LIST_ID],
                },
                priceFilter={"min": 50000, "max": 100, "currencyCode": "EUR"},
            ),
        )
        assert r.status_code == 200
        assert r.json()["metadata"]["hitCount"] == 0

    def test_sort_stability(self, client):
        """Same request twice should return the same article order."""
        body = _base_body(query="screw")
        r1 = client.post(
            "/offers_v6/_search",
            params={"sort": "name,asc", "pageSize": 10},
            json=body,
        )
        r2 = client.post(
            "/offers_v6/_search",
            params={"sort": "name,asc", "pageSize": 10},
            json=body,
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        ids1 = [a["articleId"] for a in r1.json()["articles"]]
        ids2 = [a["articleId"] for a in r2.json()["articles"]]
        assert ids1 == ids2, "Sort is unstable across identical requests"

    def test_pagination_no_overlap(self, client):
        """Page 1 and page 2 should have no overlapping article IDs."""
        body = _base_body(query="screw")
        r1 = client.post(
            "/offers_v6/_search",
            params={"page": 1, "pageSize": 5},
            json=body,
        )
        r2 = client.post(
            "/offers_v6/_search",
            params={"page": 2, "pageSize": 5},
            json=body,
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        ids1 = {a["articleId"] for a in r1.json()["articles"]}
        ids2 = {a["articleId"] for a in r2.json()["articles"]}
        assert not ids1 & ids2, f"Page overlap: {ids1 & ids2}"

    def test_vendor_filter_injection_is_escaped(self, client):
        """Crafted vendorIdsFilter value with double-quotes should be
        escaped by _quote, not cause injection."""
        r = client.post(
            "/offers_v6/_search",
            json=_base_body(
                vendorIdsFilter=['x" or 1==1 or vendor_id=="y'],
            ),
        )
        # Should succeed (200) with 0 results, not crash or return all.
        assert r.status_code == 200
        assert r.json()["metadata"]["hitCount"] == 0

    def test_summaries_only_returns_no_articles_but_hitcount(self, client):
        """SUMMARIES_ONLY mode should return hitCount but no articles."""
        r = client.post(
            "/offers_v6/_search",
            json=_base_body(
                query="screw",
                searchMode="SUMMARIES_ONLY",
                summaries=["VENDORS"],
            ),
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["articles"]) == 0
        assert data["metadata"]["hitCount"] > 0
        assert len(data["summaries"]["vendorSummaries"]) > 0
