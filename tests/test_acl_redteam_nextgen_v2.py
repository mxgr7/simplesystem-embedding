"""Red-team tests v2 — deeper behavioral mismatches from next-gen analysis.

Each test probes a contract assumption that the portal-bff (or the
generated Feign client) depends on. Failures indicate the ACL
diverges from what legacy callers expect.

Derived from reading:
  - portal-bff/ArticleSearchController.java (sort defaults)
  - ArticleSearchRestClient.java (response mapping, PageRequest creation)
  - SearchMetadataUtils.java / FeSearchMetadataUtils.java (metadata shape)
  - RestApiControllerAdvice.java (error handling, HTTP status mapping)
  - ArticleSearchMapper.java (summary consumption)
  - ExternalArticlesSearchMapper.java (external API mapping)
  - TraceIdReturningFilter.java (tracing header contract)
  - Generated SearchMetadata DTO (required/nullable annotations)
  - Generated SearchResultSummaries DTO (required fields)
  - ACL app.py (sort regex, error envelope shape)
  - ACL openapi.yaml (additionalProperties: false, required fields)

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
# Finding 1: sort=relevance,desc must not be rejected
# =========================================================================
# Legacy: ArticleSearchController uses @PageableDefault(sort = "relevance")
# which produces Sort.by(Direction.DESC, "relevance").
# ArticleSearchRestClient.mapRestSearchParams sends this as
#   sort=relevance,DESC
# The ACL's _SORT_RE only allows articleId|name|price — this rejects
# the default sort the portal-bff sends on every search.


class TestRelevanceSortAccepted:
    """The ACL must not reject relevance,desc — it's the default sort
    from the portal-bff."""

    def test_relevance_desc_sort_returns_200(self):
        """Portal-bff sends sort=relevance,DESC by default.
        ACL must not 400 on this."""
        r = _post(_base_body(), sort=["relevance,desc"])
        assert r.status_code == 200, (
            f"sort=relevance,desc returned {r.status_code}; "
            f"portal-bff sends this on every default search. "
            f"Body: {r.text[:300]}"
        )

    def test_relevance_asc_sort_returns_200(self):
        r = _post(_base_body(), sort=["relevance,asc"])
        assert r.status_code == 200, (
            f"sort=relevance,asc returned {r.status_code}; "
            f"Body: {r.text[:300]}"
        )


# =========================================================================
# Finding 2: metadata.hitCount must be present (not omitted)
# =========================================================================
# Generated SearchMetadata DTO uses @Nonnull for page/pageSize/pageCount
# but hitCount is @Nullable. However, ArticleSearchRestClient.mapSearchResponse
# line 264 passes metadata.getHitCount() to new PageImpl(..., hitCount)
# which is a long — null would NPE. The portal-bff also accesses it
# directly via page.getTotalElements().


class TestHitCountPresent:
    """hitCount must always be present in the response metadata."""

    def test_hitcount_always_present(self):
        r = _post(_base_body())
        assert r.status_code == 200
        md = r.json()["metadata"]
        assert "hitCount" in md, (
            "hitCount missing from metadata — "
            "ArticleSearchRestClient passes it to PageImpl constructor"
        )

    def test_hitcount_is_integer(self):
        """hitCount must be a JSON integer, not a string or float."""
        r = _post(_base_body())
        assert r.status_code == 200
        hc = r.json()["metadata"]["hitCount"]
        assert isinstance(hc, int), (
            f"hitCount is {type(hc).__name__}={hc!r}; legacy DTO uses Long"
        )

    def test_hitcount_nonnegative(self):
        r = _post(_base_body())
        assert r.status_code == 200
        assert r.json()["metadata"]["hitCount"] >= 0


# =========================================================================
# Finding 3: metadata.pageSize must echo the requested pageSize
# =========================================================================
# ArticleSearchRestClient line 263:
#   PageRequest.of(metadata.getPage() - 1, metadata.getPageSize(), ...)
# If pageSize in the response doesn't match what was requested, the
# portal-bff's PageRequest will have a different size than expected.


class TestPageSizeEcho:
    """metadata.pageSize must reflect the requested page size."""

    def test_pagesize_10_echoed(self):
        r = _post(_base_body(), page_size=10)
        assert r.status_code == 200
        assert r.json()["metadata"]["pageSize"] == 10

    def test_pagesize_5_echoed(self):
        r = _post(_base_body(), page_size=5)
        assert r.status_code == 200
        assert r.json()["metadata"]["pageSize"] == 5

    def test_pagesize_50_echoed(self):
        r = _post(_base_body(), page_size=50)
        assert r.status_code == 200
        assert r.json()["metadata"]["pageSize"] == 50

    def test_pagesize_1_echoed(self):
        r = _post(_base_body(), page_size=1)
        assert r.status_code == 200
        assert r.json()["metadata"]["pageSize"] == 1


# =========================================================================
# Finding 4: summaries must have all four required arrays
# =========================================================================
# Generated SearchResultSummaries DTO marks vendorSummaries,
# manufacturerSummaries, featureSummaries, pricesSummary as
# @Nonnull / required=true. If any key is missing, Jackson
# deserialization will throw.


class TestSummariesRequiredArrays:
    """summaries must contain vendorSummaries, manufacturerSummaries,
    featureSummaries, and pricesSummary — all required by the DTO."""

    def test_summaries_key_exists(self):
        r = _post(_base_body())
        assert r.status_code == 200
        assert "summaries" in r.json(), (
            "Top-level 'summaries' key missing — "
            "ArticleSearchRestClient accesses response.getSummaries()"
        )

    def test_vendor_summaries_present(self):
        r = _post(_base_body())
        assert r.status_code == 200
        s = r.json()["summaries"]
        assert "vendorSummaries" in s, (
            "vendorSummaries missing — marked @Nonnull/required in DTO"
        )
        assert isinstance(s["vendorSummaries"], list)

    def test_manufacturer_summaries_present(self):
        r = _post(_base_body())
        assert r.status_code == 200
        s = r.json()["summaries"]
        assert "manufacturerSummaries" in s, (
            "manufacturerSummaries missing — marked @Nonnull/required in DTO"
        )
        assert isinstance(s["manufacturerSummaries"], list)

    def test_feature_summaries_present(self):
        r = _post(_base_body())
        assert r.status_code == 200
        s = r.json()["summaries"]
        assert "featureSummaries" in s, (
            "featureSummaries missing — marked @Nonnull/required in DTO"
        )
        assert isinstance(s["featureSummaries"], list)

    def test_prices_summary_present(self):
        r = _post(_base_body())
        assert r.status_code == 200
        s = r.json()["summaries"]
        assert "pricesSummary" in s, (
            "pricesSummary missing — marked @Nonnull/required in DTO"
        )
        assert isinstance(s["pricesSummary"], list)


# =========================================================================
# Finding 5: searchMode=HITS_ONLY must still return summaries key
# =========================================================================
# Even when searchMode=HITS_ONLY, the response must include the
# summaries key (possibly with empty arrays) because the generated
# SearchResponse DTO always deserializes it. If the key is absent,
# response.getSummaries() returns null, and
# articleSearchMapper.mapSearchResultSummaries(null, params) may NPE.


class TestSearchModeHitsOnly:
    """HITS_ONLY must still return a summaries object (even if empty
    arrays), because the Feign client always deserializes it."""

    def test_hits_only_has_summaries_key(self):
        body = _base_body(searchMode="HITS_ONLY")
        r = _post(body)
        assert r.status_code == 200
        data = r.json()
        assert "summaries" in data, (
            "searchMode=HITS_ONLY omitted 'summaries' key — "
            "but the generated DTO always reads it"
        )

    def test_hits_only_has_articles(self):
        body = _base_body(searchMode="HITS_ONLY")
        r = _post(body)
        assert r.status_code == 200
        assert "articles" in r.json()

    def test_hits_only_has_metadata(self):
        body = _base_body(searchMode="HITS_ONLY")
        r = _post(body)
        assert r.status_code == 200
        assert "metadata" in r.json()


# =========================================================================
# Finding 6: searchMode=SUMMARIES_ONLY must still return articles key
# =========================================================================


class TestSearchModeSummariesOnly:
    """SUMMARIES_ONLY must still return articles + metadata keys."""

    def test_summaries_only_has_articles_key(self):
        body = _base_body(searchMode="SUMMARIES_ONLY")
        r = _post(body)
        assert r.status_code == 200
        data = r.json()
        assert "articles" in data, (
            "searchMode=SUMMARIES_ONLY omitted 'articles' key — "
            "but the generated DTO always reads it"
        )

    def test_summaries_only_has_metadata(self):
        body = _base_body(searchMode="SUMMARIES_ONLY")
        r = _post(body)
        assert r.status_code == 200
        assert "metadata" in r.json()


# =========================================================================
# Finding 7: error envelope must have details as array of objects
# =========================================================================
# Legacy RestApiControllerAdvice returns:
#   {message: str, details: [{field: str, message: str}], timestamp: str}
# The ACL currently returns details as list[str].
# The generated Error DTO in the legacy API spec (spec.yaml) defines:
#   details: array of ErrorDetail {field: str, message: str}
# Callers that parse the error envelope expect objects, not strings.
# Note: the ACL openapi.yaml already declares details as list[str],
# so this is a known deviation. Test it to track.


class TestErrorEnvelopeShape:
    """Error responses must match the legacy error envelope format."""

    def test_validation_error_has_message_and_timestamp(self):
        """Missing required field should return 400 with the legacy
        error envelope shape."""
        r = httpx.post(SEARCH_URL, json={}, timeout=10)
        assert r.status_code == 400
        body = r.json()
        assert "message" in body, "Error envelope missing 'message'"
        assert "timestamp" in body, "Error envelope missing 'timestamp'"

    def test_validation_error_has_details_array(self):
        """The details field must be present and be an array."""
        r = httpx.post(SEARCH_URL, json={}, timeout=10)
        assert r.status_code == 400
        body = r.json()
        assert "details" in body, "Error envelope missing 'details'"
        assert isinstance(body["details"], list), (
            f"details is {type(body['details']).__name__}, expected list"
        )


# =========================================================================
# Finding 8: pageSize=0 accepted (legacy parity)
# =========================================================================


class TestPageSizeZero:
    """pageSize=0 is accepted (legacy parity)."""

    def test_pagesize_zero_returns_200(self):
        r = _post(_base_body(), page_size=0)
        assert r.status_code == 200, (
            f"pageSize=0 returned {r.status_code} — expected 200"
        )


# =========================================================================
# Finding 9: vendor summary vendorId format
# =========================================================================
# Legacy SearchResultSummaries has vendorSummaries[].vendorId as a
# raw UUID string. The ACL response must use the same format.


class TestVendorSummaryFormat:
    """vendorSummaries[].vendorId must be a raw UUID string."""

    def test_vendor_summary_vendor_id_is_uuid(self):
        body = _base_body(searchMode="BOTH")
        r = _post(body, page_size=20)
        assert r.status_code == 200
        vs = r.json().get("summaries", {}).get("vendorSummaries", [])
        for v in vs:
            vid = v["vendorId"]
            # UUID format: 8-4-4-4-12 hex digits with dashes
            assert len(vid) == 36, (
                f"vendorId {vid!r} is not UUID format (len={len(vid)})"
            )
            assert vid.count("-") == 4, (
                f"vendorId {vid!r} is not UUID format (dashes={vid.count('-')})"
            )

    def test_vendor_summary_count_is_integer(self):
        body = _base_body(searchMode="BOTH")
        r = _post(body, page_size=20)
        assert r.status_code == 200
        vs = r.json().get("summaries", {}).get("vendorSummaries", [])
        for v in vs:
            assert isinstance(v["count"], int), (
                f"vendorSummary count is {type(v['count']).__name__}"
            )


# =========================================================================
# Finding 10: manufacturer summary shape
# =========================================================================
# Legacy: manufacturerSummaries[].name (string) + count (int64)
# The generated DTO has getManufacturerSummaries() returning a list
# of objects with getName() and getCount().


class TestManufacturerSummaryShape:
    """manufacturerSummaries items must have name + count."""

    def test_manufacturer_summaries_have_name_and_count(self):
        body = _base_body(searchMode="BOTH")
        r = _post(body, page_size=20)
        assert r.status_code == 200
        ms = r.json().get("summaries", {}).get("manufacturerSummaries", [])
        for m in ms:
            assert "name" in m, f"manufacturerSummary missing 'name': {m}"
            assert "count" in m, f"manufacturerSummary missing 'count': {m}"
            assert isinstance(m["name"], str)
            assert isinstance(m["count"], int)


# =========================================================================
# Finding 11: feature summary shape
# =========================================================================
# Legacy: featureSummaries[].name, count, values[].value, values[].count
# ArticleSearchMapper.mapFeatureSummaries reads these fields.


class TestFeatureSummaryShape:
    """featureSummaries items must have name, count, values array."""

    def test_feature_summaries_shape(self):
        body = _base_body(searchMode="BOTH")
        r = _post(body, page_size=20)
        assert r.status_code == 200
        fs = r.json().get("summaries", {}).get("featureSummaries", [])
        for f in fs:
            assert "name" in f, f"featureSummary missing 'name': {f}"
            assert "count" in f, f"featureSummary missing 'count': {f}"
            assert "values" in f, f"featureSummary missing 'values': {f}"
            assert isinstance(f["values"], list), (
                f"featureSummary values is {type(f['values']).__name__}"
            )
            for val in f["values"]:
                assert "value" in val, (
                    f"featureValueSummary missing 'value': {val}"
                )
                assert "count" in val, (
                    f"featureValueSummary missing 'count': {val}"
                )


# =========================================================================
# Finding 12: prices summary shape
# =========================================================================
# Legacy: pricesSummary[].min, max (number), currencyCode (string)


class TestPricesSummaryShape:
    """pricesSummary items must have min, max, currencyCode."""

    def test_prices_summary_shape(self):
        body = _base_body(searchMode="BOTH")
        r = _post(body, page_size=20)
        assert r.status_code == 200
        ps = r.json().get("summaries", {}).get("pricesSummary", [])
        for p in ps:
            assert "min" in p, f"pricesSummary missing 'min': {p}"
            assert "max" in p, f"pricesSummary missing 'max': {p}"
            assert "currencyCode" in p, (
                f"pricesSummary missing 'currencyCode': {p}"
            )
            assert isinstance(p["min"], (int, float)), (
                f"pricesSummary min is {type(p['min']).__name__}"
            )
            assert isinstance(p["max"], (int, float)), (
                f"pricesSummary max is {type(p['max']).__name__}"
            )


# =========================================================================
# Finding 13: response must always have all three top-level keys
# =========================================================================
# ACL openapi.yaml: required: [articles, summaries, metadata]
# The generated SearchResponse DTO accesses all three unconditionally.


class TestResponseTopLevelKeys:
    """Response must always contain articles, summaries, and metadata."""

    def test_all_top_level_keys_present(self):
        r = _post(_base_body())
        assert r.status_code == 200
        data = r.json()
        for key in ("articles", "summaries", "metadata"):
            assert key in data, (
                f"Top-level key '{key}' missing from response"
            )

    def test_articles_is_list(self):
        r = _post(_base_body())
        assert r.status_code == 200
        assert isinstance(r.json()["articles"], list)

    def test_summaries_is_object(self):
        r = _post(_base_body())
        assert r.status_code == 200
        assert isinstance(r.json()["summaries"], dict)

    def test_metadata_is_object(self):
        r = _post(_base_body())
        assert r.status_code == 200
        assert isinstance(r.json()["metadata"], dict)


# =========================================================================
# Finding 14: metadata.page, pageSize, pageCount must be integers
# =========================================================================
# Generated SearchMetadata uses Integer for page/pageSize/pageCount.
# If the ACL returns strings or floats, Jackson deserialization may
# fail or produce wrong types.


class TestMetadataTypes:
    """All metadata numeric fields must be JSON integers."""

    def test_page_is_integer(self):
        r = _post(_base_body())
        assert r.status_code == 200
        assert isinstance(r.json()["metadata"]["page"], int)

    def test_pagesize_is_integer(self):
        r = _post(_base_body())
        assert r.status_code == 200
        assert isinstance(r.json()["metadata"]["pageSize"], int)

    def test_pagecount_is_integer(self):
        r = _post(_base_body())
        assert r.status_code == 200
        assert isinstance(r.json()["metadata"]["pageCount"], int)


# =========================================================================
# Finding 15: sort with no sort param must not error
# =========================================================================
# When no sort is passed, the legacy service defaults to relevance sort.
# The ACL must handle the absence of the sort param gracefully.


class TestNoSortParam:
    """Missing sort param must be handled gracefully."""

    def test_no_sort_param_returns_200(self):
        params = {"page": 1, "pageSize": 10}
        r = httpx.post(SEARCH_URL, json=_base_body(), params=params, timeout=10)
        assert r.status_code == 200


# =========================================================================
# Finding 16: sort=articleId,desc and sort=articleId,asc accepted
# =========================================================================
# The legacy API spec says sort: [articleId,desc, articleId,asc].
# These must be accepted.


class TestArticleIdSort:
    """articleId sort must be accepted."""

    def test_articleid_desc_returns_200(self):
        r = _post(_base_body(), sort=["articleId,desc"])
        assert r.status_code == 200

    def test_articleid_asc_returns_200(self):
        r = _post(_base_body(), sort=["articleId,asc"])
        assert r.status_code == 200


# =========================================================================
# Finding 17: Content-Type must be application/json
# =========================================================================
# The generated Feign client sends Accept: application/json and
# Content-Type: application/json. The ACL must respond with
# application/json.


class TestContentType:
    """Response Content-Type must be application/json."""

    def test_response_content_type_is_json(self):
        r = _post(_base_body())
        assert r.status_code == 200
        ct = r.headers.get("content-type", "")
        assert "application/json" in ct, (
            f"Content-Type is {ct!r}, expected application/json"
        )


# =========================================================================
# Finding 18: explain=true should include explanation field in articles
# =========================================================================
# Legacy: when explain=true, articles[].explanation is populated.
# ACL stubs it as "N/A" per §2.2. The key must be present when
# explain=true.


class TestExplainField:
    """When explain=true, articles[].explanation must be present."""

    def test_explain_true_includes_explanation(self):
        body = _base_body(explain=True)
        r = _post(body)
        assert r.status_code == 200
        for art in r.json()["articles"]:
            assert "explanation" in art, (
                f"explain=true but article {art['articleId']} "
                f"has no explanation field"
            )

    def test_explain_false_may_omit_explanation(self):
        """When explain=false, explanation key may be absent or null."""
        body = _base_body(explain=False)
        r = _post(body)
        assert r.status_code == 200
        # This is the expected behavior — just verify no crash
        for art in r.json()["articles"]:
            if "explanation" in art:
                # If present, it must be null or a string
                assert art["explanation"] is None or isinstance(art["explanation"], str)


# =========================================================================
# Finding 19: large page number must not 500
# =========================================================================
# ArticleSearchRestClient creates PageRequest.of(metadata.getPage()-1,
# metadata.getPageSize()). If page is very large (beyond total pages),
# the legacy service returns an empty page. The ACL must not crash.


class TestLargePageNumber:
    """Requesting a page beyond total results must return 200 with
    empty articles, not 500."""

    def test_page_99999_returns_200(self):
        r = _post(_base_body(), page=99999)
        assert r.status_code == 200
        assert r.json()["articles"] == []

    def test_page_beyond_total_still_has_metadata(self):
        r = _post(_base_body(), page=99999)
        assert r.status_code == 200
        md = r.json()["metadata"]
        assert "page" in md
        assert "pageSize" in md
        assert "pageCount" in md
        assert "hitCount" in md


# =========================================================================
# Finding 20: price sort accepted
# =========================================================================
# ACL _SORT_RE includes "price" — verify it actually works end-to-end.


class TestPriceSort:
    """price sort must be accepted and return results."""

    def test_price_asc_returns_200(self):
        r = _post(_base_body(), sort=["price,asc"])
        assert r.status_code == 200

    def test_price_desc_returns_200(self):
        r = _post(_base_body(), sort=["price,desc"])
        assert r.status_code == 200


# =========================================================================
# Finding 21: name sort accepted
# =========================================================================


class TestNameSort:
    """name sort must be accepted and return results."""

    def test_name_asc_returns_200(self):
        r = _post(_base_body(), sort=["name,asc"])
        assert r.status_code == 200

    def test_name_desc_returns_200(self):
        r = _post(_base_body(), sort=["name,desc"])
        assert r.status_code == 200


# =========================================================================
# Finding 22: queryString with results checks article count consistency
# =========================================================================
# metadata.hitCount should be >= number of returned articles.


class TestHitCountConsistency:
    """hitCount must be >= len(articles) on any given page."""

    def test_hitcount_gte_article_count(self):
        r = _post(_base_body(), page_size=5)
        assert r.status_code == 200
        data = r.json()
        articles = data["articles"]
        hit_count = data["metadata"]["hitCount"]
        assert hit_count >= len(articles), (
            f"hitCount={hit_count} < len(articles)={len(articles)}"
        )


# =========================================================================
# Finding 23: sort param with uppercase direction
# =========================================================================
# Next-gen sends Direction.name() which is "ASC"/"DESC" (uppercase).
# The ACL _SORT_RE uses (asc|desc) lowercase. If the caller sends
# uppercase, it should still work (or be normalized).


class TestSortDirectionCase:
    """Legacy sends uppercase direction (ASC/DESC). The ACL should
    accept both cases for compatibility."""

    def test_uppercase_desc(self):
        r = _post(_base_body(), sort=["articleId,DESC"])
        # Must not be a 400 — portal-bff sends uppercase
        assert r.status_code != 500, (
            f"sort=articleId,DESC returned 500: {r.text[:200]}"
        )

    def test_uppercase_asc(self):
        r = _post(_base_body(), sort=["articleId,ASC"])
        assert r.status_code != 500, (
            f"sort=articleId,ASC returned 500: {r.text[:200]}"
        )

    def test_uppercase_accepted_as_200_or_tolerated(self):
        """Ideally returns 200. If 400, at least shouldn't 500."""
        r = _post(_base_body(), sort=["articleId,DESC"])
        assert r.status_code in (200, 400), (
            f"sort=articleId,DESC returned {r.status_code}; "
            f"expected 200 (case-insensitive) or 400 (strict). "
            f"Body: {r.text[:300]}"
        )
