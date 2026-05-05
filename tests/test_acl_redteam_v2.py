"""Red-team v2 tests for the ACL at localhost:8081.

Each test exposes a real spec violation found by adversarial probing
against the running ACL service. Only includes tests that ACTUALLY FAIL
against the current implementation.

Bug categories found:

  1. Error-envelope violation on 404/405: FastAPI's default
     ``{"detail":"..."}`` leaks through instead of the spec-mandated
     ``{message, details, timestamp}`` Error schema.

  2. Required-field defaults: three nested schemas mark fields as
     ``required`` in the OpenAPI spec, but the Pydantic model gives
     them ``default_factory=list``, silently accepting requests that
     omit those fields.

  3. Upstream status-code leak: when ftsearch returns a non-2xx status
     (e.g. 422), the ACL forwards the status code verbatim. The spec
     only defines 200, 400, 500, 501 — a 422 response is a contract
     violation. The error details also contain raw ftsearch internals.

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


def _assert_error_envelope(resp: httpx.Response) -> dict:
    """Assert the response body matches the spec Error schema:
    ``{message: str, details: [str], timestamp: str(date-time)}``."""
    data = resp.json()
    assert "message" in data, (
        f"Error response missing 'message' key. Got keys: {sorted(data.keys())}"
    )
    assert "details" in data, (
        f"Error response missing 'details' key. Got keys: {sorted(data.keys())}"
    )
    assert "timestamp" in data, (
        f"Error response missing 'timestamp' key. Got keys: {sorted(data.keys())}"
    )
    # Must NOT contain FastAPI's default "detail" key
    assert "detail" not in data, (
        f"Error response contains FastAPI default 'detail' key instead of "
        f"spec error envelope. Body: {data}"
    )
    return data


@pytest.fixture(scope="session", autouse=True)
def _check_services():
    try:
        r = httpx.get(f"{ACL_BASE}/healthz", timeout=3)
        assert r.status_code == 200
    except Exception:
        pytest.skip("ACL not running on localhost:8081")


# =========================================================================
# BUG 1: 404 response uses FastAPI default envelope, not spec Error schema
# =========================================================================
# The spec defines an Error schema with {message, details, timestamp} and
# additionalProperties: false. ALL error responses should use this shape.
# FastAPI's default 404 returns {"detail": "Not Found"} which is a
# completely different shape.


class TestNotFoundErrorEnvelope:
    """Spec: all error responses must use {message, details, timestamp}.
    Bug: 404 for unknown paths returns FastAPI's {"detail":"Not Found"}."""

    def test_unknown_path_returns_spec_error_envelope(self):
        """GET /nonexistent should return the spec Error schema, not
        FastAPI's default {"detail":"Not Found"}."""
        r = httpx.get(f"{ACL_BASE}/nonexistent", timeout=10)
        assert r.status_code == 404
        _assert_error_envelope(r)

    def test_unknown_path_deep_returns_spec_error_envelope(self):
        """GET /foo/bar/baz should return the spec Error schema."""
        r = httpx.get(f"{ACL_BASE}/foo/bar/baz", timeout=10)
        assert r.status_code == 404
        _assert_error_envelope(r)


# =========================================================================
# BUG 2: 405 response uses FastAPI default envelope, not spec Error schema
# =========================================================================
# Wrong HTTP method on a known endpoint returns 405 with
# {"detail":"Method Not Allowed"} instead of the spec error envelope.


class TestMethodNotAllowedErrorEnvelope:
    """Spec: all error responses must use {message, details, timestamp}.
    Bug: 405 for wrong methods returns FastAPI's {"detail":"Method Not Allowed"}."""

    def test_get_search_returns_spec_error_envelope(self):
        """GET /article-features/search should use the spec error envelope."""
        r = httpx.get(SEARCH_URL, timeout=10)
        assert r.status_code == 405
        _assert_error_envelope(r)

    def test_delete_search_returns_spec_error_envelope(self):
        """DELETE /article-features/search should use the spec error envelope."""
        r = httpx.delete(SEARCH_URL, timeout=10)
        assert r.status_code == 405
        _assert_error_envelope(r)

    def test_put_search_returns_spec_error_envelope(self):
        """PUT /article-features/search should use the spec error envelope."""
        r = httpx.put(SEARCH_URL, json={}, timeout=10)
        assert r.status_code == 405
        _assert_error_envelope(r)

    def test_patch_search_returns_spec_error_envelope(self):
        """PATCH /article-features/search should use the spec error envelope."""
        r = httpx.patch(SEARCH_URL, timeout=10)
        assert r.status_code == 405
        _assert_error_envelope(r)

    def test_post_healthz_returns_spec_error_envelope(self):
        """POST /healthz should use the spec error envelope (healthz only
        allows GET)."""
        r = httpx.post(f"{ACL_BASE}/healthz", json={}, timeout=10)
        assert r.status_code == 405
        _assert_error_envelope(r)


# =========================================================================
# BUG 3: FeatureFilter.values not required (spec says required: [name, values])
# =========================================================================
# The OpenAPI spec declares FeatureFilter with required: [name, values].
# The Pydantic model gives `values` a default_factory=list, so omitting
# it is silently accepted with 200 instead of being rejected as 400.


class TestFeatureFilterValuesRequired:
    """Spec: FeatureFilter has required: [name, values].
    Bug: Pydantic model defaults values to [] so omitting it succeeds."""

    def test_feature_filter_without_values_rejected(self):
        """Omitting the required 'values' field from a FeatureFilter
        item should return 400, not 200."""
        r = _post(_base_body(
            requiredFeatures=[{"name": "color"}],
        ))
        assert r.status_code == 400, (
            "requiredFeatures[0] without 'values' accepted as 200; "
            "spec says required: [name, values]"
        )


# =========================================================================
# BUG 4: EClassesAggregation.eClasses not required
#         (spec says required: [id, eClasses])
# =========================================================================
# The OpenAPI spec declares EClassesAggregation with required: [id, eClasses].
# The Pydantic model gives `eClasses` a default_factory=list, so omitting
# it is silently accepted.


class TestEClassesAggregationEClassesRequired:
    """Spec: EClassesAggregation has required: [id, eClasses].
    Bug: Pydantic model defaults eClasses to [] so omitting it succeeds."""

    def test_eclasses_aggregation_without_eclasses_rejected(self):
        """Omitting the required 'eClasses' field from an
        EClassesAggregation item should return 400, not 200."""
        r = _post(_base_body(
            eClassesAggregations=[{"id": "agg1"}],
        ))
        assert r.status_code == 400, (
            "eClassesAggregations[0] without 'eClasses' accepted as 200; "
            "spec says required: [id, eClasses]"
        )


# =========================================================================
# BUG 5: BlockedEClassVendorsFilter.blockedEClassGroups not required
#         (spec says required: [vendorIds, eClassVersion, blockedEClassGroups])
# =========================================================================
# The OpenAPI spec declares BlockedEClassVendorsFilter with
# required: [vendorIds, eClassVersion, blockedEClassGroups].
# The Pydantic model gives `blockedEClassGroups` a default_factory=list.


class TestBlockedEClassVendorsFilterGroupsRequired:
    """Spec: BlockedEClassVendorsFilter has
    required: [vendorIds, eClassVersion, blockedEClassGroups].
    Bug: Pydantic model defaults blockedEClassGroups to []."""

    def test_blocked_eclass_vendors_filter_without_groups_rejected(self):
        """Omitting the required 'blockedEClassGroups' field should
        return 400, not 200."""
        r = _post(_base_body(
            blockedEClassVendorsFilters=[{
                "vendorIds": ["01054f55-c50c-452b-8822-ee11be4788c9"],
                "eClassVersion": "ECLASS_5_1",
            }],
        ))
        assert r.status_code == 400, (
            "blockedEClassVendorsFilters[0] without 'blockedEClassGroups' "
            "accepted as 200; spec says it is required"
        )


# =========================================================================
# BUG 6: Upstream 422 status code leaks through
# =========================================================================
# The spec defines only 200, 400, 500, and 501 as valid response status
# codes.  When ftsearch rejects a request with 422, the ACL forwards
# that status code verbatim instead of mapping it to a spec-defined code
# (400 for input errors, 500 for unexpected upstream failures).
# Additionally, the error details expose ftsearch's raw JSON error
# payload, leaking implementation details.


class TestUpstreamStatusCodeLeak:
    """Spec: only 200, 400, 500, 501 are valid response status codes.
    Bug: ftsearch's 422 is forwarded verbatim to the client."""

    def test_empty_price_filter_does_not_return_422(self):
        """An empty priceFilter {} is valid per the ACL spec (no required
        fields in PriceFilter). If ftsearch rejects it, the ACL must
        translate the error to a spec-defined status code, not forward
        422 verbatim.

        The spec only defines 200, 400, 500, 501 as valid response codes."""
        r = _post(_base_body(priceFilter={}))
        assert r.status_code != 422, (
            f"ACL returned 422, which is not in the spec. "
            f"ftsearch's 422 was forwarded verbatim."
        )
        # The status must be one of the spec-defined codes
        assert r.status_code in (200, 400, 500, 501), (
            f"ACL returned {r.status_code}, which is not a spec-defined "
            f"response code (valid: 200, 400, 500, 501)"
        )

    def test_empty_price_filter_error_does_not_mention_ftsearch(self):
        """Error messages must not leak internal service names."""
        r = _post(_base_body(priceFilter={}))
        if r.status_code >= 400:
            body = r.json()
            msg = body.get("message", "")
            assert "ftsearch" not in msg.lower(), (
                f"Error message leaks internal service name 'ftsearch': {msg}"
            )

    def test_empty_price_filter_error_does_not_leak_upstream_body(self):
        """Error details must not expose raw upstream error payloads."""
        r = _post(_base_body(priceFilter={}))
        if r.status_code >= 400:
            body = r.json()
            details = body.get("details", [])
            for d in details:
                assert '"type":"missing"' not in d, (
                    f"Error details leak raw ftsearch validation payload: {d}"
                )
