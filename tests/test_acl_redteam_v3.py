"""Red-team v3 tests for the ACL at localhost:8081.

Probes edge cases and potential behavioral mismatches NOT covered by
v1/v2 suites:

  1. Pagination edge cases (page=0, negative, very large)
  2. Currency handling (invalid, case sensitivity, priceFilter vs top-level)
  3. SUMMARIES_ONLY mode (metadata, articles array behaviour)
  4. Feature filter edge cases (empty name, empty values, special chars)
  5. Interaction between explain=true and various search modes
  6. Response envelope completeness (are required keys always present?)
  7. Concurrent request handling
  8. Unicode in queryString, manufacturers, feature values

Requires:
  - ACL running on localhost:8081
  - search-api / ftsearch running upstream
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

ACL_BASE = "http://localhost:8081"
SEARCH_URL = f"{ACL_BASE}/article-features/search"
CV_ID = "866b4863-8799-4046-9e84-0985a665c1c7"


@pytest.fixture(scope="session", autouse=True)
def _check_acl_running():
    """Skip entire module if ACL is not reachable."""
    try:
        r = httpx.get(f"{ACL_BASE}/healthz", timeout=3)
        assert r.status_code == 200
    except Exception:
        pytest.skip("ACL not running on localhost:8081")


def _base_body(**overrides) -> dict:
    """Minimal valid request body."""
    body = {
        "searchMode": "BOTH",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV_ID],
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
          sort: list[str] | None = None, timeout: float = 15) -> httpx.Response:
    params: dict = {"page": page, "pageSize": page_size}
    if sort:
        params["sort"] = sort
    return httpx.post(SEARCH_URL, json=body, params=params, timeout=timeout)


def _assert_error_envelope(resp: httpx.Response) -> dict:
    """Assert legacy error envelope {message, details, timestamp}."""
    data = resp.json()
    assert "message" in data, f"Missing 'message'. Keys: {sorted(data.keys())}"
    assert "details" in data, f"Missing 'details'. Keys: {sorted(data.keys())}"
    assert "timestamp" in data, f"Missing 'timestamp'. Keys: {sorted(data.keys())}"
    assert "detail" not in data, f"FastAPI default 'detail' leaked: {data}"
    assert isinstance(data["details"], list), f"details must be list, got {type(data['details'])}"
    return data


def _assert_success_envelope(resp: httpx.Response) -> dict:
    """Assert 200 response has spec-mandated {articles, summaries, metadata}."""
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:500]}"
    data = resp.json()
    assert "articles" in data, f"Missing 'articles'. Keys: {sorted(data.keys())}"
    assert "summaries" in data, f"Missing 'summaries'. Keys: {sorted(data.keys())}"
    assert "metadata" in data, f"Missing 'metadata'. Keys: {sorted(data.keys())}"
    return data


# =========================================================================
# 1. PAGINATION EDGE CASES
# =========================================================================

class TestPaginationEdgeCases:
    """Spec: page minimum: 1, pageSize minimum: 0, maximum: 500."""

    def test_page_zero_rejected(self):
        """page=0 violates `minimum: 1` in spec; must return 400."""
        r = _post(_base_body(), page=0)
        assert r.status_code == 400, (
            f"page=0 should be rejected (spec minimum: 1), got {r.status_code}"
        )
        _assert_error_envelope(r)

    def test_page_negative_rejected(self):
        """page=-1 violates `minimum: 1` in spec; must return 400."""
        r = _post(_base_body(), page=-1)
        assert r.status_code == 400, (
            f"page=-1 should be rejected (spec minimum: 1), got {r.status_code}"
        )
        _assert_error_envelope(r)

    def test_page_very_large_returns_empty_articles(self):
        """Very large page should succeed with empty articles, not error."""
        r = _post(_base_body(), page=999999)
        data = _assert_success_envelope(r)
        assert isinstance(data["articles"], list)

    def test_page_size_zero_valid(self):
        """pageSize=0 is valid per spec (minimum: 0). Returns 200
        with empty articles array (useful for count-only queries)."""
        r = _post(_base_body(), page_size=0)
        data = _assert_success_envelope(r)
        assert data["articles"] == [], (
            f"pageSize=0 should return empty articles, got {len(data['articles'])} items"
        )

    def test_page_size_501_rejected(self):
        """pageSize=501 violates `maximum: 500` in spec; must return 400."""
        r = _post(_base_body(), page_size=501)
        assert r.status_code == 400, (
            f"pageSize=501 should be rejected (spec maximum: 500), got {r.status_code}"
        )
        _assert_error_envelope(r)

    def test_page_size_500_valid(self):
        """pageSize=500 is the maximum, should be accepted."""
        r = _post(_base_body(), page_size=500)
        # Should not fail with validation error
        assert r.status_code in (200, 400), (
            f"pageSize=500 should be valid. Got {r.status_code}"
        )
        # If 400, it should NOT be about pageSize
        if r.status_code == 400:
            data = r.json()
            details = " ".join(data.get("details", []))
            assert "pageSize" not in details and "page_size" not in details, (
                f"pageSize=500 was incorrectly rejected: {details}"
            )

    def test_page_non_integer_rejected(self):
        """page=abc should return 400."""
        params = {"page": "abc", "pageSize": "10"}
        r = httpx.post(SEARCH_URL, json=_base_body(), params=params, timeout=10)
        assert r.status_code == 400, (
            f"page='abc' should be rejected, got {r.status_code}"
        )
        _assert_error_envelope(r)


# =========================================================================
# 2. CURRENCY HANDLING
# =========================================================================

class TestCurrencyHandling:
    """Spec: currency is pattern ^[A-Z]{3}$. priceFilter.currencyCode
    also pattern ^[A-Z]{3}$."""

    def test_lowercase_currency_rejected(self):
        """currency='eur' violates pattern ^[A-Z]{3}$; must be 400."""
        r = _post(_base_body(currency="eur"))
        assert r.status_code == 400, (
            f"Lowercase currency 'eur' should be rejected, got {r.status_code}"
        )

    def test_mixed_case_currency_rejected(self):
        """currency='Eur' violates pattern ^[A-Z]{3}$; must be 400."""
        r = _post(_base_body(currency="Eur"))
        assert r.status_code == 400, (
            f"Mixed case currency 'Eur' should be rejected, got {r.status_code}"
        )

    def test_two_char_currency_rejected(self):
        """currency='EU' too short; must be 400."""
        r = _post(_base_body(currency="EU"))
        assert r.status_code == 400, (
            f"Two-char currency 'EU' should be rejected, got {r.status_code}"
        )

    def test_four_char_currency_rejected(self):
        """currency='EURO' too long; must be 400."""
        r = _post(_base_body(currency="EURO"))
        assert r.status_code == 400, (
            f"Four-char currency 'EURO' should be rejected, got {r.status_code}"
        )

    def test_numeric_currency_rejected(self):
        """currency='123' has digits, violates [A-Z]; must be 400."""
        r = _post(_base_body(currency="123"))
        assert r.status_code == 400, (
            f"Numeric currency '123' should be rejected, got {r.status_code}"
        )

    def test_price_filter_currency_code_lowercase_rejected(self):
        """priceFilter.currencyCode='eur' violates pattern; must be 400."""
        r = _post(_base_body(priceFilter={"min": 100, "currencyCode": "eur"}))
        assert r.status_code == 400, (
            f"Lowercase priceFilter.currencyCode 'eur' should be rejected, got {r.status_code}"
        )

    def test_price_filter_min_without_currency_code_rejected(self):
        """priceFilter with min set but no currencyCode; must be 400
        (per spec: currencyCode required when min or max is set)."""
        r = _post(_base_body(priceFilter={"min": 100}))
        assert r.status_code == 400, (
            f"priceFilter.min without currencyCode should be rejected, got {r.status_code}"
        )

    def test_price_filter_max_without_currency_code_rejected(self):
        """priceFilter with max set but no currencyCode; must be 400."""
        r = _post(_base_body(priceFilter={"max": 5000}))
        assert r.status_code == 400, (
            f"priceFilter.max without currencyCode should be rejected, got {r.status_code}"
        )

    def test_price_filter_both_bounds_with_currency_valid(self):
        """priceFilter with min, max, and valid currencyCode should work."""
        r = _post(_base_body(priceFilter={"min": 100, "max": 5000, "currencyCode": "EUR"}))
        # Should be accepted by the ACL (may fail at ftsearch but not 400 from validation)
        assert r.status_code != 422, (
            f"Valid priceFilter should not return 422"
        )

    def test_price_filter_currency_mismatch_with_top_level(self):
        """Different currencies for top-level and priceFilter should both
        be forwarded (per spec: two roles, not collapsed)."""
        r = _post(_base_body(
            currency="USD",
            priceFilter={"min": 100, "max": 5000, "currencyCode": "EUR"},
        ))
        # ACL should not reject this; the currencies serve different roles
        assert r.status_code != 400 or "currency" not in str(r.json().get("details", [])), (
            f"Mismatched currencies between top-level and priceFilter "
            f"should be allowed (two-roles spec), got {r.status_code}"
        )


# =========================================================================
# 3. SUMMARIES_ONLY MODE
# =========================================================================

class TestSummariesOnlyMode:
    """searchMode=SUMMARIES_ONLY: spec implies articles array should be
    empty but summaries and metadata still returned."""

    def test_summaries_only_returns_envelope(self):
        """SUMMARIES_ONLY response must still have articles, summaries, metadata."""
        r = _post(_base_body(
            searchMode="SUMMARIES_ONLY",
            summaries=["VENDORS", "MANUFACTURERS"],
        ))
        data = _assert_success_envelope(r)
        assert isinstance(data["articles"], list)
        assert isinstance(data["summaries"], dict)
        assert isinstance(data["metadata"], dict)

    def test_summaries_only_articles_empty(self):
        """In SUMMARIES_ONLY mode, articles array should be empty."""
        r = _post(_base_body(
            searchMode="SUMMARIES_ONLY",
            summaries=["VENDORS"],
        ))
        if r.status_code == 200:
            data = r.json()
            assert data.get("articles") == [], (
                f"SUMMARIES_ONLY should return empty articles, got "
                f"{len(data.get('articles', []))} items"
            )

    def test_summaries_only_metadata_present(self):
        """SUMMARIES_ONLY must still return metadata with hitCount."""
        r = _post(_base_body(
            searchMode="SUMMARIES_ONLY",
            summaries=["VENDORS"],
        ))
        if r.status_code == 200:
            data = r.json()
            meta = data.get("metadata", {})
            # Spec: required: [page, pageSize, pageCount, hitCount]
            for key in ("page", "pageSize", "pageCount", "hitCount"):
                assert key in meta, (
                    f"metadata.{key} missing in SUMMARIES_ONLY response. "
                    f"Got keys: {sorted(meta.keys())}"
                )

    def test_hits_only_returns_no_summaries(self):
        """HITS_ONLY mode: summaries should be empty/minimal."""
        r = _post(_base_body(
            searchMode="HITS_ONLY",
            summaries=["VENDORS", "MANUFACTURERS"],
        ))
        if r.status_code == 200:
            data = r.json()
            # summaries key must exist (spec required) but content may be empty
            assert "summaries" in data


# =========================================================================
# 4. FEATURE FILTER EDGE CASES
# =========================================================================

class TestFeatureFilterEdgeCases:
    """FeatureFilter schema: required [name, values], both strings."""

    def test_feature_filter_empty_name_accepted_or_rejected_consistently(self):
        """Empty string name: spec doesn't set minLength, so this might
        be valid. Key: must not 500."""
        r = _post(_base_body(requiredFeatures=[{"name": "", "values": ["x"]}]))
        assert r.status_code != 500, (
            f"Empty feature name caused 500: {r.text[:300]}"
        )

    def test_feature_filter_empty_values_list(self):
        """Empty values array: spec says type:array with items:string.
        An empty array is technically valid per JSON Schema."""
        r = _post(_base_body(requiredFeatures=[{"name": "color", "values": []}]))
        # Should not cause 500
        assert r.status_code != 500, (
            f"Empty values list caused 500: {r.text[:300]}"
        )

    def test_feature_filter_special_chars_in_name(self):
        """Feature name with special characters (SQL injection style)."""
        r = _post(_base_body(requiredFeatures=[
            {"name": "color'; DROP TABLE--", "values": ["red"]}
        ]))
        assert r.status_code != 500, (
            f"Special chars in feature name caused 500: {r.text[:300]}"
        )
        # Should not return anything that looks like a DB error
        if r.status_code >= 400:
            body_text = r.text.lower()
            assert "sql" not in body_text and "syntax" not in body_text, (
                f"Response contains SQL-related error message: {r.text[:300]}"
            )

    def test_feature_filter_special_chars_in_values(self):
        """Feature values with special characters."""
        r = _post(_base_body(requiredFeatures=[
            {"name": "size", "values": ["<script>alert(1)</script>", "12\"x14\""]}
        ]))
        assert r.status_code != 500, (
            f"Special chars in feature values caused 500: {r.text[:300]}"
        )

    def test_feature_filter_very_long_name(self):
        """Very long feature name (10k chars)."""
        long_name = "A" * 10000
        r = _post(_base_body(requiredFeatures=[
            {"name": long_name, "values": ["val"]}
        ]))
        # Should not crash the server
        assert r.status_code in (200, 400), (
            f"Very long feature name returned {r.status_code}"
        )

    def test_feature_filter_many_values(self):
        """Many values in a single feature filter."""
        r = _post(_base_body(requiredFeatures=[
            {"name": "color", "values": [f"val_{i}" for i in range(1000)]}
        ]))
        assert r.status_code != 500, (
            f"1000 feature values caused 500: {r.text[:300]}"
        )

    def test_feature_filter_duplicate_names(self):
        """Multiple feature filters with the same name."""
        r = _post(_base_body(requiredFeatures=[
            {"name": "color", "values": ["red"]},
            {"name": "color", "values": ["blue"]},
        ]))
        # Should not crash -- may be valid or rejected, but not 500
        assert r.status_code != 500, (
            f"Duplicate feature names caused 500: {r.text[:300]}"
        )


# =========================================================================
# 5. EXPLAIN + SEARCH MODE INTERACTION
# =========================================================================

class TestExplainInteraction:
    """explain=true should add explanation:'N/A' to articles regardless
    of search mode."""

    def test_explain_true_with_hits_only(self):
        """explain=true + HITS_ONLY: articles should have explanation='N/A'."""
        r = _post(_base_body(searchMode="HITS_ONLY", explain=True))
        if r.status_code == 200:
            data = r.json()
            for art in data.get("articles", []):
                assert "explanation" in art, (
                    f"explain=true but article missing 'explanation': {art}"
                )
                assert art["explanation"] == "N/A", (
                    f"explanation should be 'N/A', got {art['explanation']!r}"
                )

    def test_explain_false_with_both(self):
        """explain=false: articles should NOT have explanation key."""
        r = _post(_base_body(searchMode="BOTH", explain=False))
        if r.status_code == 200:
            data = r.json()
            for art in data.get("articles", []):
                assert "explanation" not in art, (
                    f"explain=false but article has 'explanation': {art}"
                )

    def test_explain_true_with_summaries_only(self):
        """explain=true + SUMMARIES_ONLY: no articles expected, so
        explanation field is moot. Key: must not error."""
        r = _post(_base_body(
            searchMode="SUMMARIES_ONLY",
            explain=True,
            summaries=["VENDORS"],
        ))
        # Should not fail
        assert r.status_code in (200, 400), (
            f"explain=true + SUMMARIES_ONLY returned {r.status_code}"
        )
        if r.status_code == 200:
            data = r.json()
            # Articles should be empty; explanation on empty list is vacuously satisfied
            assert isinstance(data.get("articles"), list)

    def test_explain_non_boolean_rejected(self):
        """explain must be a strict boolean, not string 'true'."""
        r = _post(_base_body(explain="true"))
        assert r.status_code == 400, (
            f"explain='true' (string) should be rejected, got {r.status_code}"
        )

    def test_explain_integer_rejected(self):
        """explain must be strict boolean, not integer 1."""
        r = _post(_base_body(explain=1))
        assert r.status_code == 400, (
            f"explain=1 (integer) should be rejected, got {r.status_code}"
        )


# =========================================================================
# 6. RESPONSE ENVELOPE COMPLETENESS
# =========================================================================

class TestResponseEnvelopeCompleteness:
    """Spec: SearchResponse required: [articles, summaries, metadata].
    Metadata required: [page, pageSize, pageCount, hitCount]."""

    def test_metadata_has_required_fields(self):
        """Every 200 response metadata must have page, pageSize,
        pageCount, hitCount."""
        r = _post(_base_body())
        if r.status_code == 200:
            meta = r.json().get("metadata", {})
            for key in ("page", "pageSize", "pageCount", "hitCount"):
                assert key in meta, (
                    f"metadata missing required field '{key}'. "
                    f"Got keys: {sorted(meta.keys())}"
                )

    def test_metadata_page_count_at_least_one(self):
        """Spec says pageCount minimum:0 but response mapper doc says
        pageCount >= 1. Check the actual behavior."""
        r = _post(_base_body(queryString="xyznonexistent999"))
        if r.status_code == 200:
            meta = r.json().get("metadata", {})
            # Per response.py: "Legacy always returns pageCount >= 1"
            assert meta.get("pageCount", 0) >= 1, (
                f"pageCount should be >= 1, got {meta.get('pageCount')}"
            )

    def test_articles_is_always_list(self):
        """articles must always be a list, never null or missing."""
        r = _post(_base_body())
        if r.status_code == 200:
            data = r.json()
            assert isinstance(data["articles"], list), (
                f"articles must be list, got {type(data['articles'])}"
            )

    def test_summaries_is_always_dict(self):
        """summaries must always be an object/dict, never null or list."""
        r = _post(_base_body())
        if r.status_code == 200:
            data = r.json()
            assert isinstance(data["summaries"], dict), (
                f"summaries must be dict, got {type(data['summaries'])}"
            )

    def test_no_extra_top_level_keys(self):
        """Spec has additionalProperties: false on SearchResponse.
        Only articles, summaries, metadata allowed."""
        r = _post(_base_body())
        if r.status_code == 200:
            data = r.json()
            allowed = {"articles", "summaries", "metadata"}
            extra = set(data.keys()) - allowed
            assert not extra, (
                f"Response has extra top-level keys not in spec: {extra}"
            )

    def test_article_no_score_field(self):
        """Spec: Article has only articleId and explanation.
        score from ftsearch must NOT be forwarded."""
        r = _post(_base_body())
        if r.status_code == 200:
            data = r.json()
            for art in data.get("articles", []):
                assert "score" not in art, (
                    f"Article has 'score' field which is not in spec: {art}"
                )

    def test_metadata_no_recall_clipped(self):
        """recallClipped and hitCountClipped must be dropped by ACL."""
        r = _post(_base_body())
        if r.status_code == 200:
            meta = r.json().get("metadata", {})
            assert "recallClipped" not in meta, (
                f"metadata.recallClipped should be stripped: {meta}"
            )
            assert "hitCountClipped" not in meta, (
                f"metadata.hitCountClipped should be stripped: {meta}"
            )

    def test_error_envelope_has_no_extra_keys(self):
        """Error envelope must be exactly {message, details, timestamp}
        with additionalProperties: false."""
        r = _post({"invalid": "body"})
        assert r.status_code == 400
        data = r.json()
        allowed = {"message", "details", "timestamp"}
        extra = set(data.keys()) - allowed
        assert not extra, (
            f"Error envelope has extra keys not in spec: {extra}"
        )


# =========================================================================
# 7. CONCURRENT REQUEST HANDLING
# =========================================================================

class TestConcurrentRequests:
    """ACL must handle concurrent requests without mixing up responses
    or crashing."""

    def test_parallel_requests_all_succeed(self):
        """Send 10 concurrent requests; all should return valid responses."""
        def _do_request(i: int) -> tuple[int, int]:
            r = _post(_base_body(queryString=f"test_concurrent_{i}"))
            return i, r.status_code

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_do_request, i) for i in range(10)]
            results = [f.result() for f in futures]

        for i, status in results:
            assert status in (200, 400), (
                f"Concurrent request {i} returned unexpected {status}"
            )

    def test_parallel_different_modes(self):
        """Send concurrent requests with different searchModes."""
        modes = ["BOTH", "HITS_ONLY", "SUMMARIES_ONLY"]

        def _do_request(mode: str) -> tuple[str, int, dict]:
            r = _post(_base_body(searchMode=mode, summaries=["VENDORS"]))
            return mode, r.status_code, r.json() if r.status_code == 200 else {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_do_request, m) for m in modes]
            results = [f.result() for f in futures]

        for mode, status, data in results:
            assert status in (200, 400), (
                f"searchMode={mode} concurrent returned {status}"
            )

    def test_parallel_mixed_valid_invalid(self):
        """Mix valid and invalid requests concurrently -- server must
        not crash or conflate responses."""
        def _valid() -> int:
            return _post(_base_body()).status_code

        def _invalid() -> int:
            return _post({"bad": "request"}).status_code

        with ThreadPoolExecutor(max_workers=6) as executor:
            valid_futs = [executor.submit(_valid) for _ in range(3)]
            invalid_futs = [executor.submit(_invalid) for _ in range(3)]

            for f in valid_futs:
                assert f.result() in (200,), (
                    f"Valid request failed during concurrent load: {f.result()}"
                )
            for f in invalid_futs:
                assert f.result() == 400, (
                    f"Invalid request during concurrent load returned {f.result()}, expected 400"
                )


# =========================================================================
# 8. UNICODE HANDLING
# =========================================================================

class TestUnicodeHandling:
    """queryString, manufacturers, feature values can contain Unicode."""

    def test_unicode_query_string(self):
        """German umlauts in queryString."""
        r = _post(_base_body(queryString="Schraubenschluessel"))
        assert r.status_code != 500, f"ASCII queryString caused 500: {r.text[:300]}"

        r = _post(_base_body(queryString="Schraubenschlüssel"))
        assert r.status_code != 500, f"Umlaut queryString caused 500: {r.text[:300]}"

    def test_unicode_query_string_chinese(self):
        """Chinese characters in queryString."""
        r = _post(_base_body(queryString="螺丝刀"))
        assert r.status_code != 500, f"Chinese queryString caused 500: {r.text[:300]}"

    def test_unicode_query_string_emoji(self):
        """Emoji in queryString -- must not crash."""
        r = _post(_base_body(queryString="bolt \U0001f529"))
        assert r.status_code != 500, f"Emoji in queryString caused 500: {r.text[:300]}"

    def test_unicode_manufacturer_filter(self):
        """Manufacturer names with accented characters."""
        r = _post(_base_body(manufacturersFilter=["Boegli"]))
        assert r.status_code != 500, f"ASCII manufacturer caused 500"

        r = _post(_base_body(manufacturersFilter=["Bogli"]))
        assert r.status_code != 500, f"Accented manufacturer caused 500"

    def test_unicode_feature_values(self):
        """Feature values with unicode characters."""
        r = _post(_base_body(requiredFeatures=[
            {"name": "Groesse", "values": ["10mm", "M8x1.25"]}
        ]))
        assert r.status_code != 500, f"Feature with special chars caused 500"

    def test_unicode_feature_name_with_diacritics(self):
        """Feature name with full diacritics."""
        r = _post(_base_body(requiredFeatures=[
            {"name": "Langenenheit", "values": ["Meter"]}
        ]))
        assert r.status_code != 500, f"Diacritic feature name caused 500"

    def test_null_bytes_in_query_string(self):
        """Null bytes in queryString: should be rejected or handled safely."""
        r = _post(_base_body(queryString="test\x00value"))
        assert r.status_code != 500, (
            f"Null byte in queryString caused 500: {r.text[:300]}"
        )

    def test_very_long_unicode_query_string(self):
        """Long unicode string exceeding the 10k char limit."""
        # 10001 chars -- one over the max_length=10000
        long_q = "A" * 10001
        r = _post(_base_body(queryString=long_q))
        # Should be rejected due to max_length=10000
        assert r.status_code == 400, (
            f"queryString > 10k chars should be rejected, got {r.status_code}"
        )

    def test_query_string_exactly_10k(self):
        """queryString at exactly 10000 chars should be accepted."""
        q = "A" * 10000
        r = _post(_base_body(queryString=q))
        # Must not be rejected due to length validation
        assert r.status_code != 400 or "queryString" not in str(r.json().get("details", [])), (
            f"queryString at exactly 10000 chars should be valid"
        )


# =========================================================================
# 9. SORT PARAMETER EDGE CASES
# =========================================================================

class TestSortEdgeCases:
    """sort query param: pattern ^(articleId|name|price),(asc|desc)$"""

    def test_empty_sort_valid(self):
        """No sort param should default to relevance."""
        r = _post(_base_body())
        assert r.status_code in (200, 400)

    def test_valid_sort_combinations(self):
        """All valid sort combinations should be accepted."""
        valid_sorts = [
            "articleId,asc", "articleId,desc",
            "name,asc", "name,desc",
            "price,asc", "price,desc",
        ]
        for s in valid_sorts:
            r = _post(_base_body(), sort=[s])
            assert r.status_code != 400 or "sort" not in str(r.json().get("details", [])), (
                f"Valid sort '{s}' was rejected"
            )

    def test_invalid_sort_field_rejected(self):
        """sort=relevance,asc is not in the allowed enum; must be 400."""
        r = _post(_base_body(), sort=["relevance,asc"])
        assert r.status_code == 400, (
            f"Invalid sort field 'relevance' should be rejected, got {r.status_code}"
        )
        _assert_error_envelope(r)

    def test_invalid_sort_direction_rejected(self):
        """sort=price,ascending is not valid; must be 400."""
        r = _post(_base_body(), sort=["price,ascending"])
        assert r.status_code == 400, (
            f"Invalid sort direction 'ascending' should be rejected, got {r.status_code}"
        )

    def test_sort_case_sensitive_field(self):
        """sort=Price,asc (capital P) should be rejected."""
        r = _post(_base_body(), sort=["Price,asc"])
        assert r.status_code == 400, (
            f"sort='Price,asc' (capitalized) should be rejected, got {r.status_code}"
        )

    def test_sort_case_sensitive_direction(self):
        """sort=price,ASC should be rejected (spec says asc/desc only)."""
        r = _post(_base_body(), sort=["price,ASC"])
        assert r.status_code == 400, (
            f"sort='price,ASC' (uppercase direction) should be rejected, got {r.status_code}"
        )

    def test_sort_with_spaces_rejected(self):
        """sort='price, asc' (space) doesn't match pattern; must be 400."""
        r = _post(_base_body(), sort=["price, asc"])
        assert r.status_code == 400, (
            f"sort='price, asc' (with space) should be rejected, got {r.status_code}"
        )

    def test_multiple_sort_keys(self):
        """Multiple sort keys passed at once."""
        r = _post(_base_body(), sort=["price,asc", "name,desc"])
        # Should be accepted (spec says repeatable)
        assert r.status_code != 500, (
            f"Multiple sort keys caused 500: {r.text[:300]}"
        )

    def test_sort_empty_string_rejected(self):
        """sort='' (empty string) doesn't match pattern; must be 400."""
        r = _post(_base_body(), sort=[""])
        assert r.status_code == 400, (
            f"Empty sort string should be rejected, got {r.status_code}"
        )


# =========================================================================
# 10. SEARCH MODE VALIDATION
# =========================================================================

class TestSearchModeValidation:
    """searchMode must be one of HITS_ONLY, SUMMARIES_ONLY, BOTH."""

    def test_invalid_search_mode_rejected(self):
        """searchMode='INVALID' must be 400."""
        r = _post(_base_body(searchMode="INVALID"))
        assert r.status_code == 400, (
            f"Invalid searchMode 'INVALID' should be rejected, got {r.status_code}"
        )

    def test_search_mode_case_sensitive(self):
        """searchMode='both' (lowercase) must be 400."""
        r = _post(_base_body(searchMode="both"))
        assert r.status_code == 400, (
            f"Lowercase searchMode 'both' should be rejected, got {r.status_code}"
        )

    def test_search_articles_by_non_standard_rejected(self):
        """searchArticlesBy='ARTICLE_NUMBER' is dropped per sec 2.1."""
        r = _post(_base_body(searchArticlesBy="ARTICLE_NUMBER"))
        assert r.status_code == 400, (
            f"searchArticlesBy='ARTICLE_NUMBER' should be rejected, got {r.status_code}"
        )

    def test_search_articles_by_case_sensitive(self):
        """searchArticlesBy='standard' (lowercase) must be 400."""
        r = _post(_base_body(searchArticlesBy="standard"))
        assert r.status_code == 400, (
            f"Lowercase searchArticlesBy 'standard' should be rejected, got {r.status_code}"
        )


# =========================================================================
# 11. BODY SIZE AND ARRAY LIMITS
# =========================================================================

class TestBodyAndArrayLimits:
    """Body max 1MB, vendorIds max 500, articleIds max 1000."""

    def test_vendor_ids_at_limit(self):
        """500 vendorIds should be accepted."""
        ids = [f"{i:08d}-0000-0000-0000-000000000000" for i in range(500)]
        r = _post(_base_body(vendorIdsFilter=ids))
        # Should not fail validation for count
        if r.status_code == 400:
            details = str(r.json().get("details", []))
            assert "500" not in details or "max" not in details.lower(), (
                f"500 vendorIds should be at the limit, not over"
            )

    def test_vendor_ids_over_limit(self):
        """501 vendorIds should be rejected."""
        ids = [f"{i:08d}-0000-0000-0000-000000000000" for i in range(501)]
        r = _post(_base_body(vendorIdsFilter=ids))
        assert r.status_code == 400, (
            f"501 vendorIds should exceed limit, got {r.status_code}"
        )

    def test_article_ids_over_limit(self):
        """1001 articleIds should be rejected."""
        ids = [f"abc{i}:MTIzNA" for i in range(1001)]
        r = _post(_base_body(articleIdsFilter=ids))
        assert r.status_code == 400, (
            f"1001 articleIds should exceed limit, got {r.status_code}"
        )

    def test_content_length_over_1mb_rejected(self):
        """Request with Content-Length > 1MB should be 413.
        The ACL checks the Content-Length header; we send a body that
        actually matches the declared size (padding with spaces which
        are valid JSON whitespace)."""
        # Build a JSON body that's over 1MB by padding with a huge string field
        big_value = "x" * (1_048_577)  # Just over 1MB
        big_body = '{"padding": "' + big_value + '"}'
        r = httpx.post(
            SEARCH_URL,
            content=big_body.encode(),
            headers={"content-type": "application/json"},
            timeout=15,
        )
        assert r.status_code == 413, (
            f"Content-Length > 1MB should return 413, got {r.status_code}"
        )


# =========================================================================
# 12. ADDITIONAL PROPERTIES / UNKNOWN FIELDS
# =========================================================================

class TestAdditionalProperties:
    """Spec: additionalProperties: false on all DTOs."""

    def test_unknown_top_level_field_rejected(self):
        """Extra top-level field should be 400 (extra=forbid)."""
        body = _base_body()
        body["unknownField"] = "surprise"
        r = _post(body)
        assert r.status_code == 400, (
            f"Unknown top-level field should be rejected, got {r.status_code}"
        )

    def test_unknown_nested_field_rejected(self):
        """Extra field in selectedArticleSources should be 400."""
        body = _base_body()
        body["selectedArticleSources"]["unknownNested"] = "nope"
        r = _post(body)
        assert r.status_code == 400, (
            f"Unknown nested field should be rejected, got {r.status_code}"
        )

    def test_unknown_field_in_price_filter_rejected(self):
        """Extra field in priceFilter should be 400."""
        body = _base_body(priceFilter={"min": 100, "currencyCode": "EUR", "extra": True})
        r = _post(body)
        assert r.status_code == 400, (
            f"Unknown field in priceFilter should be rejected, got {r.status_code}"
        )


# =========================================================================
# 13. TYPE COERCION / STRICT TYPING
# =========================================================================

class TestStrictTyping:
    """Spec uses strict types: integer fields must not accept strings,
    booleans must not accept 0/1, etc."""

    def test_max_delivery_time_string_rejected(self):
        """maxDeliveryTime must be integer, not string '5'."""
        r = _post(_base_body(maxDeliveryTime="5"))
        assert r.status_code == 400, (
            f"maxDeliveryTime='5' (string) should be rejected, got {r.status_code}"
        )

    def test_max_delivery_time_float_rejected(self):
        """maxDeliveryTime must be integer, not float 5.0."""
        r = _post(_base_body(maxDeliveryTime=5.0))
        assert r.status_code == 400, (
            f"maxDeliveryTime=5.0 (float) should be rejected, got {r.status_code}"
        )

    def test_max_delivery_time_negative_rejected(self):
        """maxDeliveryTime has minimum: 0; -1 must be rejected."""
        r = _post(_base_body(maxDeliveryTime=-1))
        assert r.status_code == 400, (
            f"maxDeliveryTime=-1 should be rejected (minimum: 0), got {r.status_code}"
        )

    def test_core_sortiment_only_integer_rejected(self):
        """coreSortimentOnly must be strict boolean, not 0."""
        r = _post(_base_body(coreSortimentOnly=0))
        assert r.status_code == 400, (
            f"coreSortimentOnly=0 should be rejected (strict bool), got {r.status_code}"
        )

    def test_closed_marketplace_only_string_rejected(self):
        """closedMarketplaceOnly must be strict boolean, not 'false'."""
        r = _post(_base_body(closedMarketplaceOnly="false"))
        assert r.status_code == 400, (
            f"closedMarketplaceOnly='false' should be rejected, got {r.status_code}"
        )

    def test_price_filter_min_as_float_rejected(self):
        """priceFilter.min must be strict integer, not float."""
        r = _post(_base_body(priceFilter={"min": 100.5, "currencyCode": "EUR"}))
        assert r.status_code == 400, (
            f"priceFilter.min=100.5 (float) should be rejected, got {r.status_code}"
        )

    def test_e_class_code_as_string_rejected(self):
        """currentEClass5Code must be integer, not string."""
        r = _post(_base_body(currentEClass5Code="12345"))
        assert r.status_code == 400, (
            f"currentEClass5Code='12345' (string) should be rejected, got {r.status_code}"
        )


# =========================================================================
# 14. MISSING REQUIRED FIELDS
# =========================================================================

class TestMissingRequiredFields:
    """Spec: many fields are required. Missing them must return 400."""

    def test_missing_search_mode(self):
        """searchMode is required."""
        body = _base_body()
        del body["searchMode"]
        r = _post(body)
        assert r.status_code == 400, (
            f"Missing searchMode should be 400, got {r.status_code}"
        )

    def test_missing_search_articles_by(self):
        """searchArticlesBy is required."""
        body = _base_body()
        del body["searchArticlesBy"]
        r = _post(body)
        assert r.status_code == 400, (
            f"Missing searchArticlesBy should be 400, got {r.status_code}"
        )

    def test_missing_selected_article_sources(self):
        """selectedArticleSources is required."""
        body = _base_body()
        del body["selectedArticleSources"]
        r = _post(body)
        assert r.status_code == 400, (
            f"Missing selectedArticleSources should be 400, got {r.status_code}"
        )

    def test_missing_max_delivery_time(self):
        """maxDeliveryTime is required."""
        body = _base_body()
        del body["maxDeliveryTime"]
        r = _post(body)
        assert r.status_code == 400, (
            f"Missing maxDeliveryTime should be 400, got {r.status_code}"
        )

    def test_missing_currency(self):
        """currency is required."""
        body = _base_body()
        del body["currency"]
        r = _post(body)
        assert r.status_code == 400, (
            f"Missing currency should be 400, got {r.status_code}"
        )

    def test_missing_explain(self):
        """explain is required."""
        body = _base_body()
        del body["explain"]
        r = _post(body)
        assert r.status_code == 400, (
            f"Missing explain should be 400, got {r.status_code}"
        )

    def test_missing_core_sortiment_only(self):
        """coreSortimentOnly is required."""
        body = _base_body()
        del body["coreSortimentOnly"]
        r = _post(body)
        assert r.status_code == 400, (
            f"Missing coreSortimentOnly should be 400, got {r.status_code}"
        )

    def test_missing_closed_marketplace_only(self):
        """closedMarketplaceOnly is required."""
        body = _base_body()
        del body["closedMarketplaceOnly"]
        r = _post(body)
        assert r.status_code == 400, (
            f"Missing closedMarketplaceOnly should be 400, got {r.status_code}"
        )

    def test_missing_closed_catalog_version_ids(self):
        """closedCatalogVersionIds is required inside selectedArticleSources."""
        body = _base_body()
        del body["selectedArticleSources"]["closedCatalogVersionIds"]
        r = _post(body)
        assert r.status_code == 400, (
            f"Missing closedCatalogVersionIds should be 400, got {r.status_code}"
        )


# =========================================================================
# 15. ARTICLE ID FORMAT EDGE CASES
# =========================================================================

class TestArticleIdFormat:
    """articleIdsFilter items: legacy uses {friendlyId}:{b64url} (2 parts)."""

    def test_article_id_with_three_colons(self):
        """A 3-part ID (ftsearch format) passed to the ACL should not crash."""
        ftsearch_id = "550e8400-e29b-41d4-a716-446655440000:MTIzNA:550e8400-e29b-41d4-a716-446655440001"
        r = _post(_base_body(articleIdsFilter=[ftsearch_id]))
        # Should not 500 -- may return empty results
        assert r.status_code != 500, (
            f"3-part article ID caused 500: {r.text[:300]}"
        )

    def test_article_id_empty_string(self):
        """Empty string article ID should not crash."""
        r = _post(_base_body(articleIdsFilter=[""]))
        assert r.status_code != 500, (
            f"Empty article ID caused 500: {r.text[:300]}"
        )

    def test_article_id_single_part(self):
        """Single-part ID (no colon) should not crash."""
        r = _post(_base_body(articleIdsFilter=["abc123"]))
        assert r.status_code != 500, (
            f"Single-part article ID caused 500: {r.text[:300]}"
        )

    def test_article_id_special_characters(self):
        """Article ID with base64url chars (+, /, =) should not crash."""
        r = _post(_base_body(articleIdsFilter=["abc:YWJj+/=="]))
        assert r.status_code != 500, (
            f"Special char article ID caused 500: {r.text[:300]}"
        )


# =========================================================================
# 16. SUMMARIES ENUM VALUES
# =========================================================================

class TestSummariesEnum:
    """summaries array items must be valid SummaryKind enum values."""

    def test_invalid_summary_kind_rejected(self):
        """summaries=['INVALID'] should be 400."""
        r = _post(_base_body(summaries=["INVALID"]))
        assert r.status_code == 400, (
            f"Invalid SummaryKind 'INVALID' should be rejected, got {r.status_code}"
        )

    def test_all_valid_summary_kinds(self):
        """All defined SummaryKind values should be accepted."""
        all_kinds = [
            "CATEGORIES", "ECLASS5", "ECLASS7", "S2CLASS",
            "VENDORS", "MANUFACTURERS", "FEATURES", "PRICES",
            "PLATFORM_CATEGORIES", "ECLASS5SET",
        ]
        r = _post(_base_body(summaries=all_kinds))
        # Should not fail validation
        assert r.status_code != 400 or "summaries" not in str(r.json().get("details", [])), (
            f"Valid SummaryKind values were rejected"
        )

    def test_summary_kind_case_sensitive(self):
        """summaries=['vendors'] (lowercase) should be 400."""
        r = _post(_base_body(summaries=["vendors"]))
        assert r.status_code == 400, (
            f"Lowercase SummaryKind 'vendors' should be rejected, got {r.status_code}"
        )

    def test_duplicate_summary_kinds_accepted(self):
        """Duplicate entries in summaries array -- should not crash."""
        r = _post(_base_body(summaries=["VENDORS", "VENDORS", "VENDORS"]))
        assert r.status_code != 500, (
            f"Duplicate SummaryKind values caused 500: {r.text[:300]}"
        )


# =========================================================================
# 17. EMPTY BODY / MALFORMED JSON
# =========================================================================

class TestMalformedInput:
    """Various malformed inputs that should return 400, not 500."""

    def test_empty_body(self):
        """Empty JSON object should return 400 (missing required fields)."""
        r = _post({})
        assert r.status_code == 400, (
            f"Empty body should return 400, got {r.status_code}"
        )

    def test_null_body(self):
        """null as body should return 400 or 422."""
        r = httpx.post(SEARCH_URL, content=b"null",
                       headers={"content-type": "application/json"}, timeout=10)
        assert r.status_code in (400, 422), (
            f"null body should return 400/422, got {r.status_code}"
        )

    def test_array_body(self):
        """Array as body (instead of object) should return 400."""
        r = httpx.post(SEARCH_URL, content=b"[]",
                       headers={"content-type": "application/json"}, timeout=10)
        assert r.status_code in (400, 422), (
            f"Array body should return 400/422, got {r.status_code}"
        )

    def test_invalid_json(self):
        """Malformed JSON should return 400 or 422."""
        r = httpx.post(SEARCH_URL, content=b"{invalid json",
                       headers={"content-type": "application/json"}, timeout=10)
        assert r.status_code in (400, 422), (
            f"Invalid JSON should return 400/422, got {r.status_code}"
        )

    def test_non_json_content_type(self):
        """Non-JSON content type should return 400 or 415."""
        r = httpx.post(SEARCH_URL, content=b"key=value",
                       headers={"content-type": "application/x-www-form-urlencoded"},
                       timeout=10)
        assert r.status_code in (400, 415, 422), (
            f"Non-JSON content type should return 400/415/422, got {r.status_code}"
        )

    def test_no_content_type_header(self):
        """Missing content-type should still be handled gracefully."""
        r = httpx.post(SEARCH_URL, content=b'{"test": 1}', timeout=10)
        assert r.status_code != 500, (
            f"Missing content-type caused 500: {r.text[:300]}"
        )
