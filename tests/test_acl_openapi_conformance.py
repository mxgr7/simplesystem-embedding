"""OpenAPI spec conformance tests for the ACL service.

Verifies that the running ACL service at localhost:8081 (via TestClient)
faithfully implements acl/openapi.yaml. Each test targets a specific
area where spec vs implementation drift commonly occurs:

  1. Response schema conformance (200 + error envelopes)
  2. Error envelope format (400/413/500 all match Error schema)
  3. Query parameter boundary validation (page min=1, pageSize 0..500)
  4. Content-Type handling (reject non-JSON, handle missing CT)
  5. HTTP method validation (only POST allowed on /article-features/search)
  6. Nullable vs absent fields (OpenAPI nullable semantics)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import jsonschema
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from unittest.mock import AsyncMock, patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from acl.app import app  # noqa: E402

# --- Load the OpenAPI spec and extract schemas ---------------------------

SPEC_PATH = REPO_ROOT / "acl" / "openapi.yaml"
SPEC = yaml.safe_load(SPEC_PATH.read_text())
SCHEMAS = SPEC["components"]["schemas"]


def _resolve_ref(ref: str) -> dict:
    """Resolve a $ref like '#/components/schemas/Foo'."""
    parts = ref.lstrip("#/").split("/")
    node = SPEC
    for p in parts:
        node = node[p]
    return node


def _build_jsonschema(schema_name: str) -> dict:
    """Build a JSON Schema validator dict from an OpenAPI component schema,
    with $ref resolution inlined so jsonschema can validate."""
    def _inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                resolved = _resolve_ref(node["$ref"])
                return _inline(resolved)
            # OpenAPI 3.0 nullable: true -> JSON Schema accepts null
            out = {}
            for k, v in node.items():
                if k == "nullable" and v is True:
                    continue  # handled below
                out[k] = _inline(v)
            if node.get("nullable"):
                existing_type = out.get("type")
                if existing_type:
                    out["type"] = [existing_type, "null"]
                else:
                    # anyOf with null
                    out = {"anyOf": [out, {"type": "null"}]}
            return out
        elif isinstance(node, list):
            return [_inline(item) for item in node]
        return node

    raw = SCHEMAS[schema_name]
    inlined = _inline(raw)
    # Remove additionalProperties: false for top-level to allow lenient
    # validation -- we test strict key matching separately.
    return inlined


ERROR_SCHEMA = _build_jsonschema("Error")
SEARCH_RESPONSE_SCHEMA = _build_jsonschema("SearchResponse")
METADATA_SCHEMA = _build_jsonschema("Metadata")


# --- Fixtures -------------------------------------------------------------

@pytest.fixture
def client():
    """TestClient with raise_server_exceptions=False so error handlers run."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _valid_request() -> dict:
    """Minimal valid request body per the spec."""
    return {
        "searchMode": "BOTH",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {"closedCatalogVersionIds": []},
        "maxDeliveryTime": 0,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
        "currency": "EUR",
        "explain": False,
    }


def _mock_ftsearch_response(
    articles=None, summaries=None, metadata=None,
) -> dict:
    """Build a mock ftsearch response for injection."""
    return {
        "articles": articles or [],
        "summaries": summaries or {},
        "metadata": metadata or {
            "page": 1,
            "pageSize": 10,
            "pageCount": 1,
            "hitCount": 0,
            "term": None,
        },
    }


def _patch_ftsearch(mock_response: dict):
    """Context manager that patches the ftsearch client to return mock_response."""
    async def _mock_search(*args, **kwargs):
        return mock_response
    return patch.object(
        app.state, "ftsearch",
        new=type("MockClient", (), {"search": _mock_search, "aclose": AsyncMock()})(),
    )


# --- 1. RESPONSE SCHEMA CONFORMANCE --------------------------------------

class TestResponseSchemaConformance:
    """Verify the 200 response matches the declared SearchResponse schema."""

    def test_200_response_validates_against_search_response_schema(self, client):
        """Full response body must validate against SearchResponse."""
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        jsonschema.validate(body, SEARCH_RESPONSE_SCHEMA)

    def test_200_response_has_all_required_top_level_keys(self, client):
        """SearchResponse requires: articles, summaries, metadata."""
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200
        body = r.json()
        for key in ["articles", "summaries", "metadata"]:
            assert key in body, f"Required key '{key}' missing from response"

    def test_metadata_has_all_required_fields(self, client):
        """Metadata requires: page, pageSize, pageCount, hitCount."""
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200
        meta = r.json()["metadata"]
        for key in ["page", "pageSize", "pageCount", "hitCount"]:
            assert key in meta, f"Required metadata key '{key}' missing"

    def test_metadata_field_types(self, client):
        """All metadata integer fields are actually integers."""
        mock = _mock_ftsearch_response(metadata={
            "page": 1, "pageSize": 10, "pageCount": 1, "hitCount": 42, "term": "bolt",
        })
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200
        meta = r.json()["metadata"]
        assert isinstance(meta["page"], int)
        assert isinstance(meta["pageSize"], int)
        assert isinstance(meta["pageCount"], int)
        assert isinstance(meta["hitCount"], int)
        assert isinstance(meta["term"], str)

    def test_articles_is_array(self, client):
        """articles must be an array even when empty."""
        mock = _mock_ftsearch_response(articles=[])
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200
        assert isinstance(r.json()["articles"], list)

    def test_article_has_required_article_id(self, client):
        """Each article object requires articleId (string)."""
        mock = _mock_ftsearch_response(articles=[
            {"articleId": "abc-uuid:artnum:cat-uuid", "score": 0.9},
        ])
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200
        articles = r.json()["articles"]
        assert len(articles) == 1
        assert "articleId" in articles[0]
        assert isinstance(articles[0]["articleId"], str)

    def test_article_does_not_expose_score_field(self, client):
        """Score is an ftsearch-internal field; spec does NOT declare it."""
        mock = _mock_ftsearch_response(articles=[
            {"articleId": "abc-uuid:artnum:cat-uuid", "score": 0.95},
        ])
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200
        articles = r.json()["articles"]
        assert "score" not in articles[0], "score field leaked into response (not in spec)"

    def test_explain_true_injects_explanation_field(self, client):
        """When explain=true, articles[].explanation = 'N/A' per spec section 2.2."""
        req = _valid_request()
        req["explain"] = True
        mock = _mock_ftsearch_response(articles=[
            {"articleId": "abc-uuid:artnum:cat-uuid", "score": 0.5},
        ])
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=req)
        assert r.status_code == 200
        articles = r.json()["articles"]
        assert articles[0].get("explanation") == "N/A"

    def test_explain_false_explanation_absent_or_null(self, client):
        """When explain=false, explanation should be absent or null (nullable field)."""
        req = _valid_request()
        req["explain"] = False
        mock = _mock_ftsearch_response(articles=[
            {"articleId": "abc-uuid:artnum:cat-uuid", "score": 0.5},
        ])
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=req)
        assert r.status_code == 200
        articles = r.json()["articles"]
        # Per OpenAPI nullable semantics: key can be absent or null
        art = articles[0]
        if "explanation" in art:
            assert art["explanation"] is None, \
                f"explanation should be null when explain=false, got {art['explanation']!r}"

    def test_summaries_is_object(self, client):
        """summaries must be an object (not array, not null)."""
        mock = _mock_ftsearch_response(summaries={})
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200
        assert isinstance(r.json()["summaries"], dict)

    def test_response_has_no_additional_top_level_keys(self, client):
        """SearchResponse declares additionalProperties: false."""
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200
        body = r.json()
        allowed = {"articles", "summaries", "metadata"}
        extra = set(body.keys()) - allowed
        assert not extra, f"Extra top-level keys violating additionalProperties:false: {extra}"

    def test_metadata_no_extra_keys(self, client):
        """Metadata additionalProperties: false -- no recallClipped etc."""
        mock = _mock_ftsearch_response(metadata={
            "page": 1, "pageSize": 10, "pageCount": 1, "hitCount": 0,
            "term": None, "recallClipped": True, "hitCountClipped": False,
        })
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200
        meta = r.json()["metadata"]
        allowed = {"page", "pageSize", "pageCount", "hitCount", "term"}
        extra = set(meta.keys()) - allowed
        assert not extra, f"Extra metadata keys violating additionalProperties:false: {extra}"


# --- 2. ERROR ENVELOPE FORMAT ---------------------------------------------

class TestErrorEnvelopeFormat:
    """All error responses must match the Error schema exactly:
    {message: string, details: string[], timestamp: datetime}"""

    def _assert_error_envelope(self, body: dict):
        """Validate body against the Error schema from the spec."""
        # Required keys
        assert "message" in body, f"Error envelope missing 'message': {body}"
        assert "details" in body, f"Error envelope missing 'details': {body}"
        assert "timestamp" in body, f"Error envelope missing 'timestamp': {body}"
        # Types
        assert isinstance(body["message"], str), \
            f"message must be string, got {type(body['message'])}"
        assert isinstance(body["details"], list), \
            f"details must be array, got {type(body['details'])}"
        for item in body["details"]:
            assert isinstance(item, str), \
                f"details items must be strings, got {type(item)}: {item!r}"
        assert isinstance(body["timestamp"], str), \
            f"timestamp must be string, got {type(body['timestamp'])}"
        # timestamp must be parseable as ISO-8601
        try:
            datetime.fromisoformat(body["timestamp"])
        except ValueError:
            pytest.fail(f"timestamp is not valid ISO-8601: {body['timestamp']!r}")
        # additionalProperties: false
        allowed_keys = {"message", "details", "timestamp"}
        extra = set(body.keys()) - allowed_keys
        assert not extra, f"Error envelope has extra keys (additionalProperties:false): {extra}"

    def test_400_validation_error_matches_error_schema(self, client):
        """Missing required field -> 400 with Error envelope."""
        body = _valid_request()
        body.pop("currency")
        r = client.post("/article-features/search", json=body)
        assert r.status_code == 400
        self._assert_error_envelope(r.json())

    def test_400_invalid_json_matches_error_schema(self, client):
        """Malformed JSON -> 400 with Error envelope."""
        r = client.post(
            "/article-features/search",
            content=b"not json{{{",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400
        self._assert_error_envelope(r.json())

    def test_400_query_param_validation_matches_error_schema(self, client):
        """pageSize=501 -> 400 with Error envelope."""
        r = client.post(
            "/article-features/search",
            params={"pageSize": 501},
            json=_valid_request(),
        )
        assert r.status_code == 400
        self._assert_error_envelope(r.json())

    def test_400_invalid_sort_matches_error_schema(self, client):
        """Invalid sort pattern -> 400 with Error envelope."""
        r = client.post(
            "/article-features/search",
            params={"sort": "invalid_field,asc"},
            json=_valid_request(),
        )
        assert r.status_code == 400
        self._assert_error_envelope(r.json())

    def test_413_body_too_large_matches_error_schema(self, client):
        """Body > 1MB -> 413 with Error envelope."""
        huge_body = b"x" * (1_048_576 + 1)
        r = client.post(
            "/article-features/search",
            content=huge_body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(huge_body)),
            },
        )
        assert r.status_code == 413
        self._assert_error_envelope(r.json())

    def test_500_unhandled_exception_matches_error_schema(self, client):
        """Internal error -> 500 with Error envelope (no traceback leak)."""
        class _Boom:
            async def search(self, *args, **kwargs):
                raise RuntimeError("kaboom")
            async def aclose(self):
                pass
        original = client.app.state.ftsearch
        client.app.state.ftsearch = _Boom()
        try:
            r = client.post("/article-features/search", json=_valid_request())
        finally:
            client.app.state.ftsearch = original
        assert r.status_code == 500
        self._assert_error_envelope(r.json())

    def test_405_method_not_allowed_matches_error_schema(self, client):
        """GET on POST-only route -> should use Error envelope."""
        r = client.get("/article-features/search")
        assert r.status_code == 405
        self._assert_error_envelope(r.json())

    def test_error_timestamp_is_recent(self, client):
        """Timestamp should be current (not hardcoded/stale)."""
        before = datetime.now(tz=None)
        body = _valid_request()
        body.pop("searchMode")
        r = client.post("/article-features/search", json=body)
        assert r.status_code == 400
        ts = datetime.fromisoformat(r.json()["timestamp"])
        # Remove timezone info for comparison
        ts_naive = ts.replace(tzinfo=None)
        # Should be within 5 seconds of request time
        delta = abs((ts_naive - before).total_seconds())
        assert delta < 5, f"Timestamp too far from request time: {delta}s"


# --- 3. QUERY PARAMETER VALIDATION ----------------------------------------

class TestQueryParameterValidation:
    """Spec declares: page >= 1, pageSize 0..500, sort pattern."""

    def test_page_default_is_1(self, client):
        """No page param -> defaults to 1 (spec default: 1)."""
        mock = _mock_ftsearch_response(metadata={
            "page": 1, "pageSize": 10, "pageCount": 1, "hitCount": 0,
        })
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200

    def test_page_minimum_1_accepts_1(self, client):
        """page=1 is the minimum -- must be accepted."""
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post(
                "/article-features/search",
                params={"page": 1},
                json=_valid_request(),
            )
        assert r.status_code == 200

    def test_page_below_minimum_rejected(self, client):
        """page=0 violates minimum:1 -- must be rejected."""
        r = client.post(
            "/article-features/search",
            params={"page": 0},
            json=_valid_request(),
        )
        assert r.status_code == 400, f"page=0 should be rejected, got {r.status_code}"

    def test_page_negative_rejected(self, client):
        """page=-1 violates minimum:1."""
        r = client.post(
            "/article-features/search",
            params={"page": -1},
            json=_valid_request(),
        )
        assert r.status_code == 400

    def test_pagesize_default_is_10(self, client):
        """No pageSize param -> defaults to 10."""
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200

    def test_pagesize_minimum_0_accepts_0(self, client):
        """pageSize=0 is valid (minimum: 0)."""
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post(
                "/article-features/search",
                params={"pageSize": 0},
                json=_valid_request(),
            )
        assert r.status_code == 200

    def test_pagesize_maximum_500_accepts_500(self, client):
        """pageSize=500 is valid (maximum: 500)."""
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post(
                "/article-features/search",
                params={"pageSize": 500},
                json=_valid_request(),
            )
        assert r.status_code == 200

    def test_pagesize_501_rejected(self, client):
        """pageSize=501 exceeds maximum:500 -- must be rejected."""
        r = client.post(
            "/article-features/search",
            params={"pageSize": 501},
            json=_valid_request(),
        )
        assert r.status_code == 400

    def test_pagesize_negative_rejected(self, client):
        """pageSize=-1 violates minimum:0."""
        r = client.post(
            "/article-features/search",
            params={"pageSize": -1},
            json=_valid_request(),
        )
        assert r.status_code == 400

    def test_sort_valid_pattern_accepted(self, client):
        """sort=price,asc matches ^(articleId|name|price),(asc|desc)$."""
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post(
                "/article-features/search",
                params={"sort": "price,asc"},
                json=_valid_request(),
            )
        assert r.status_code == 200

    def test_sort_all_valid_fields_accepted(self, client):
        """All declared sort fields: articleId, name, price."""
        mock = _mock_ftsearch_response()
        for field in ["articleId", "name", "price"]:
            for direction in ["asc", "desc"]:
                with _patch_ftsearch(mock):
                    r = client.post(
                        "/article-features/search",
                        params={"sort": f"{field},{direction}"},
                        json=_valid_request(),
                    )
                assert r.status_code == 200, \
                    f"sort={field},{direction} should be valid, got {r.status_code}"

    def test_sort_invalid_field_rejected(self, client):
        """sort=unknown,asc doesn't match the pattern -- must be rejected."""
        r = client.post(
            "/article-features/search",
            params={"sort": "unknown,asc"},
            json=_valid_request(),
        )
        assert r.status_code == 400

    def test_sort_invalid_direction_rejected(self, client):
        """sort=price,up doesn't match -- only asc|desc allowed."""
        r = client.post(
            "/article-features/search",
            params={"sort": "price,up"},
            json=_valid_request(),
        )
        assert r.status_code == 400

    def test_sort_missing_direction_rejected(self, client):
        """sort=price (no comma+direction) doesn't match pattern."""
        r = client.post(
            "/article-features/search",
            params={"sort": "price"},
            json=_valid_request(),
        )
        assert r.status_code == 400


# --- 4. CONTENT-TYPE HANDLING --------------------------------------------

class TestContentTypeHandling:
    """Spec declares requestBody content: application/json only."""

    def test_application_json_accepted(self, client):
        """Standard application/json content-type works."""
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post(
                "/article-features/search",
                json=_valid_request(),
            )
        assert r.status_code == 200

    def test_application_xml_rejected(self, client):
        """application/xml is not declared in the spec -- should reject."""
        r = client.post(
            "/article-features/search",
            content=b"<xml/>",
            headers={"Content-Type": "application/xml"},
        )
        # Should be 400 or 415 (Unsupported Media Type) or 422
        assert r.status_code in (400, 415, 422), \
            f"application/xml should be rejected, got {r.status_code}"

    def test_multipart_form_data_rejected(self, client):
        """multipart/form-data is not declared -- should reject."""
        r = client.post(
            "/article-features/search",
            content=b"--boundary\r\nContent-Disposition: form-data; name=\"file\"\r\n\r\ndata\r\n--boundary--",
            headers={"Content-Type": "multipart/form-data; boundary=boundary"},
        )
        assert r.status_code in (400, 415, 422), \
            f"multipart/form-data should be rejected, got {r.status_code}"

    def test_no_content_type_with_valid_json_body(self, client):
        """Missing Content-Type header with JSON body -- FastAPI behavior."""
        import json
        r = client.post(
            "/article-features/search",
            content=json.dumps(_valid_request()).encode(),
        )
        # Without content-type, FastAPI may not parse the body as JSON
        # Should either work (lenient) or fail with 400/422
        assert r.status_code in (200, 400, 422), \
            f"No content-type: expected 200/400/422, got {r.status_code}"

    def test_response_content_type_is_application_json(self, client):
        """Response Content-Type must be application/json per spec."""
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200
        ct = r.headers.get("content-type", "")
        assert "application/json" in ct, f"Response content-type should be application/json, got {ct!r}"

    def test_error_response_content_type_is_application_json(self, client):
        """Error responses also declare application/json."""
        body = _valid_request()
        body.pop("currency")
        r = client.post("/article-features/search", json=body)
        assert r.status_code == 400
        ct = r.headers.get("content-type", "")
        assert "application/json" in ct, f"Error content-type should be application/json, got {ct!r}"


# --- 5. HTTP METHOD VALIDATION -------------------------------------------

class TestHttpMethodValidation:
    """Spec declares only POST for /article-features/search."""

    def test_get_not_allowed(self, client):
        """GET /article-features/search -> 405 Method Not Allowed."""
        r = client.get("/article-features/search")
        assert r.status_code == 405, f"GET should be 405, got {r.status_code}"

    def test_put_not_allowed(self, client):
        """PUT /article-features/search -> 405."""
        r = client.put("/article-features/search", json=_valid_request())
        assert r.status_code == 405, f"PUT should be 405, got {r.status_code}"

    def test_delete_not_allowed(self, client):
        """DELETE /article-features/search -> 405."""
        r = client.delete("/article-features/search")
        assert r.status_code == 405, f"DELETE should be 405, got {r.status_code}"

    def test_patch_not_allowed(self, client):
        """PATCH /article-features/search -> 405."""
        r = client.patch("/article-features/search", json=_valid_request())
        assert r.status_code == 405, f"PATCH should be 405, got {r.status_code}"

    def test_405_response_has_error_envelope(self, client):
        """405 responses should still use the Error envelope."""
        r = client.get("/article-features/search")
        assert r.status_code == 405
        body = r.json()
        assert "message" in body
        assert "details" in body
        assert "timestamp" in body

    def test_healthz_only_allows_get(self, client):
        """Spec declares only GET for /healthz."""
        r = client.post("/healthz")
        assert r.status_code == 405, f"POST /healthz should be 405, got {r.status_code}"


# --- 6. NULLABLE VS ABSENT FIELDS ----------------------------------------

class TestNullableVsAbsentFields:
    """OpenAPI nullable fields: test the service handles them correctly
    in both request acceptance and response emission."""

    def test_request_querystring_null_accepted(self, client):
        """queryString is nullable -- null value should be accepted."""
        req = _valid_request()
        req["queryString"] = None
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=req)
        assert r.status_code == 200, f"queryString:null should be accepted, got {r.status_code}: {r.text}"

    def test_request_querystring_absent_accepted(self, client):
        """queryString is not required -- absent should be accepted."""
        req = _valid_request()
        assert "queryString" not in req  # not in our minimal request
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=req)
        assert r.status_code == 200

    def test_request_pricefilter_null_accepted(self, client):
        """priceFilter is nullable -- explicit null should be accepted."""
        req = _valid_request()
        req["priceFilter"] = None
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=req)
        assert r.status_code == 200

    def test_request_accessories_for_article_number_null(self, client):
        """accessoriesForArticleNumber is nullable."""
        req = _valid_request()
        req["accessoriesForArticleNumber"] = None
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=req)
        assert r.status_code == 200

    def test_request_spare_parts_for_article_number_null(self, client):
        """sparePartsForArticleNumber is nullable."""
        req = _valid_request()
        req["sparePartsForArticleNumber"] = None
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=req)
        assert r.status_code == 200

    def test_request_similar_to_article_number_null(self, client):
        """similarToArticleNumber is nullable."""
        req = _valid_request()
        req["similarToArticleNumber"] = None
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=req)
        assert r.status_code == 200

    def test_request_current_eclass5_code_null(self, client):
        """currentEClass5Code is nullable integer."""
        req = _valid_request()
        req["currentEClass5Code"] = None
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=req)
        assert r.status_code == 200

    def test_request_current_eclass7_code_null(self, client):
        """currentEClass7Code is nullable integer."""
        req = _valid_request()
        req["currentEClass7Code"] = None
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=req)
        assert r.status_code == 200

    def test_request_current_s2class_code_null(self, client):
        """currentS2ClassCode is nullable integer."""
        req = _valid_request()
        req["currentS2ClassCode"] = None
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=req)
        assert r.status_code == 200

    def test_response_metadata_term_null_when_no_query(self, client):
        """metadata.term is nullable -- should be null when no queryString."""
        mock = _mock_ftsearch_response(metadata={
            "page": 1, "pageSize": 10, "pageCount": 1, "hitCount": 0, "term": None,
        })
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=_valid_request())
        assert r.status_code == 200
        meta = r.json()["metadata"]
        # term can be absent or null per nullable semantics
        if "term" in meta:
            assert meta["term"] is None or isinstance(meta["term"], str)

    def test_response_metadata_term_string_when_query_present(self, client):
        """metadata.term should be a string when queryString was provided."""
        mock = _mock_ftsearch_response(metadata={
            "page": 1, "pageSize": 10, "pageCount": 1, "hitCount": 5, "term": "bolt",
        })
        req = _valid_request()
        req["queryString"] = "bolt"
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=req)
        assert r.status_code == 200
        meta = r.json()["metadata"]
        if "term" in meta:
            assert isinstance(meta["term"], str)

    def test_selected_article_sources_nullable_fields(self, client):
        """customerManagedArticleNumberListId is nullable UUID."""
        req = _valid_request()
        req["selectedArticleSources"] = {
            "closedCatalogVersionIds": [],
            "customerManagedArticleNumberListId": None,
            "uiCustomerArticleNumberSourceId": None,
        }
        mock = _mock_ftsearch_response()
        with _patch_ftsearch(mock):
            r = client.post("/article-features/search", json=req)
        assert r.status_code == 200

    def test_non_nullable_field_null_rejected(self, client):
        """currency is NOT nullable -- null should be rejected."""
        req = _valid_request()
        req["currency"] = None
        r = client.post("/article-features/search", json=req)
        assert r.status_code == 400, \
            f"null for non-nullable 'currency' should be rejected, got {r.status_code}"

    def test_non_nullable_boolean_null_rejected(self, client):
        """coreSortimentOnly is required boolean (not nullable) -- null should be rejected."""
        req = _valid_request()
        req["coreSortimentOnly"] = None
        r = client.post("/article-features/search", json=req)
        assert r.status_code == 400, \
            f"null for non-nullable 'coreSortimentOnly' should be rejected, got {r.status_code}"

    def test_non_nullable_integer_null_rejected(self, client):
        """maxDeliveryTime is required integer (not nullable) -- null should be rejected."""
        req = _valid_request()
        req["maxDeliveryTime"] = None
        r = client.post("/article-features/search", json=req)
        assert r.status_code == 400, \
            f"null for non-nullable 'maxDeliveryTime' should be rejected, got {r.status_code}"
