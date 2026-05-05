"""Red-team tests for the ACL at localhost:8081.

Each test exposes a real spec violation or behavioral bug found by
systematic adversarial probing against the running ACL service.
Only includes tests that ACTUALLY FAIL against the current implementation.

Bug categories found:
  1. Pydantic lax-mode type coercion: strings/bools/ints accepted where
     the OpenAPI spec requires strict JSON types (boolean, integer).
  2. Snake-case field names accepted alongside camelCase due to
     populate_by_name=True, violating additionalProperties:false.
  3. UUID format: uuid not validated on uuid-typed fields.
  4. Sort query-param regex not enforced by the ACL (delegated to ftsearch,
     leaking upstream internals in the error message).
  5. 404/405 responses use FastAPI's default {"detail":"..."} envelope
     instead of the spec's {message, details, timestamp} Error schema.

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
# BUG 1: String-to-boolean coercion (spec says type: boolean)
# =========================================================================
# The OpenAPI spec declares `explain`, `coreSortimentOnly`,
# `closedMarketplaceOnly`, `s2ClassForProductCategories` as
# `type: boolean`.  In JSON Schema, only literal `true`/`false` are
# valid booleans.  Pydantic's lax mode coerces strings like "yes",
# "on", "true", "1" into booleans.  The ACL should reject these with
# 400; instead it returns 200.


class TestStringToBoolCoercion:
    """Spec: boolean fields must only accept JSON true/false.
    Bug: Pydantic lax mode coerces strings to bool."""

    @pytest.mark.parametrize("value", ["yes", "no", "true", "false",
                                       "True", "False", "1", "0",
                                       "on", "off"])
    def test_explain_rejects_string(self, value: str):
        """explain is type:boolean. String values should be rejected."""
        r = _post(_base_body(explain=value))
        assert r.status_code == 400, (
            f"explain={value!r} (string) accepted as 200; "
            f"spec requires type:boolean (only true/false literals)"
        )

    @pytest.mark.parametrize("value", ["yes", "true", "1", "on"])
    def test_core_sortiment_only_rejects_string(self, value: str):
        r = _post(_base_body(coreSortimentOnly=value))
        assert r.status_code == 400, (
            f"coreSortimentOnly={value!r} (string) accepted as 200"
        )

    @pytest.mark.parametrize("value", ["yes", "true", "1", "on"])
    def test_closed_marketplace_only_rejects_string(self, value: str):
        r = _post(_base_body(closedMarketplaceOnly=value))
        assert r.status_code == 400, (
            f"closedMarketplaceOnly={value!r} (string) accepted as 200"
        )


# =========================================================================
# BUG 2: Int-to-boolean coercion (spec says type: boolean)
# =========================================================================
# JSON Schema boolean only allows true/false. Integers 0 and 1 are NOT
# valid booleans. Pydantic coerces them.


class TestIntToBoolCoercion:
    """Spec: boolean fields must only accept JSON true/false.
    Bug: Pydantic lax mode coerces ints 0/1 to bool."""

    @pytest.mark.parametrize("value", [0, 1])
    def test_explain_rejects_int(self, value: int):
        r = _post(_base_body(explain=value))
        assert r.status_code == 400, (
            f"explain={value} (int) accepted as 200; "
            f"spec requires type:boolean"
        )

    @pytest.mark.parametrize("value", [0, 1])
    def test_core_sortiment_only_rejects_int(self, value: int):
        r = _post(_base_body(coreSortimentOnly=value))
        assert r.status_code == 400, (
            f"coreSortimentOnly={value} (int) accepted as 200"
        )


# =========================================================================
# BUG 3: Bool-to-integer coercion (spec says type: integer)
# =========================================================================
# maxDeliveryTime is type:integer. JSON booleans are not integers per
# JSON Schema.  Pydantic coerces True->1, False->0.


class TestBoolToIntCoercion:
    """Spec: integer fields must only accept JSON numbers.
    Bug: Pydantic lax mode coerces booleans to int."""

    @pytest.mark.parametrize("value", [True, False])
    def test_max_delivery_time_rejects_bool(self, value: bool):
        r = _post(_base_body(maxDeliveryTime=value))
        assert r.status_code == 400, (
            f"maxDeliveryTime={value} (bool) accepted as 200; "
            f"spec requires type:integer"
        )


# =========================================================================
# BUG 4: String-to-integer coercion (spec says type: integer)
# =========================================================================
# Fields like maxDeliveryTime (type:integer) accept string "0".
# JSON Schema requires a JSON number, not a string.


class TestStringToIntCoercion:
    """Spec: integer fields must only accept JSON number literals.
    Bug: Pydantic lax mode coerces numeric strings to int."""

    def test_max_delivery_time_rejects_string(self):
        r = _post(_base_body(maxDeliveryTime="0"))
        assert r.status_code == 400, (
            'maxDeliveryTime="0" (string) accepted as 200'
        )

    def test_current_eclass5_code_rejects_string(self):
        r = _post(_base_body(currentEClass5Code="23110103"))
        assert r.status_code == 400, (
            'currentEClass5Code="23110103" (string) accepted as 200'
        )

    def test_price_filter_min_rejects_string(self):
        r = _post(_base_body(
            priceFilter={"min": "100", "currencyCode": "EUR"},
        ))
        assert r.status_code == 400, (
            'priceFilter.min="100" (string) accepted as 200'
        )

    def test_eclass_group_code_rejects_string(self):
        r = _post(_base_body(
            blockedEClassVendorsFilters=[{
                "vendorIds": ["01054f55-c50c-452b-8822-ee11be4788c9"],
                "eClassVersion": "ECLASS_5_1",
                "blockedEClassGroups": [
                    {"eClassGroupCode": "23110000", "value": True},
                ],
            }],
        ))
        assert r.status_code == 400, (
            'eClassGroupCode="23110000" (string) accepted as 200'
        )

    def test_eclass_filter_items_reject_strings(self):
        r = _post(_base_body(eClassesFilter=["23110103"]))
        assert r.status_code == 400, (
            'eClassesFilter=["23110103"] (string items) accepted as 200'
        )


# =========================================================================
# BUG 5: Bool-to-integer coercion in priceFilter.min
# =========================================================================


class TestBoolToIntInPriceFilter:
    """priceFilter.min is type:integer, nullable:true.
    Bug: True (a JSON boolean) is accepted and coerced to 1."""

    def test_price_filter_min_rejects_bool(self):
        r = _post(_base_body(
            priceFilter={"min": True, "currencyCode": "EUR"},
        ))
        assert r.status_code == 400, (
            "priceFilter.min=True (bool) accepted as 200"
        )


# =========================================================================
# BUG 6: Float-to-integer coercion for lossless floats
# =========================================================================
# currentEClass5Code=23110103.0 (a float that happens to be an exact
# integer) is accepted. The spec says type:integer.


class TestFloatToIntCoercion:
    """Spec: type:integer. A JSON float like 23110103.0 is not an integer.
    Bug: Pydantic coerces lossless floats to int."""

    def test_current_eclass5_code_rejects_float(self):
        r = _post(_base_body(currentEClass5Code=23110103.0))
        assert r.status_code == 400, (
            "currentEClass5Code=23110103.0 (float) accepted as 200"
        )


# =========================================================================
# BUG 7: snake_case field names accepted (additionalProperties: false)
# =========================================================================
# The OpenAPI spec defines all fields in camelCase and declares
# additionalProperties:false. The Pydantic model uses
# `populate_by_name=True`, which also accepts the Python snake_case
# attribute name. This means a request with `search_mode` instead of
# `searchMode` is accepted -- an undocumented, non-spec input surface.


class TestSnakeCaseFieldNames:
    """Spec: additionalProperties:false with camelCase field names.
    Bug: populate_by_name=True accepts snake_case aliases."""

    def test_full_snake_case_body_rejected(self):
        """All top-level fields sent as snake_case should be rejected."""
        body = {
            "search_mode": "BOTH",
            "search_articles_by": "STANDARD",
            "selected_article_sources": {
                "closed_catalog_version_ids": [],
                "catalog_version_ids_ordered_by_preference": [CV_EUR],
            },
            "max_delivery_time": 0,
            "core_sortiment_only": False,
            "closed_marketplace_only": False,
            "currency": "EUR",
            "explain": False,
        }
        r = httpx.post(SEARCH_URL, json=body, params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 400, (
            "Full snake_case body accepted as 200; spec uses camelCase "
            "with additionalProperties:false"
        )

    def test_mixed_camel_and_snake_rejected(self):
        """Mixing camelCase + snake_case should be rejected."""
        body = {
            "searchMode": "BOTH",
            "search_articles_by": "STANDARD",  # snake_case
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
        r = httpx.post(SEARCH_URL, json=body, params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 200 or r.status_code == 400, "unexpected status"
        # The real bug: search_articles_by is accepted as a valid field name
        # even though the spec only defines searchArticlesBy.
        assert r.status_code == 400, (
            "snake_case 'search_articles_by' accepted; spec only defines "
            "'searchArticlesBy'"
        )

    def test_snake_case_price_filter_rejected(self):
        """price_filter (snake_case) should be rejected per the spec."""
        body = _base_body()
        body["price_filter"] = {"min": 100, "currency_code": "EUR"}
        r = httpx.post(SEARCH_URL, json=body, params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 400, (
            "snake_case 'price_filter' accepted; spec only defines 'priceFilter'"
        )

    def test_snake_case_in_nested_object_rejected(self):
        """Snake_case in SelectedArticleSources should be rejected."""
        body = _base_body()
        body["selectedArticleSources"] = {
            "closed_catalog_version_ids": [],
            "catalog_version_ids_ordered_by_preference": [CV_EUR],
        }
        r = httpx.post(SEARCH_URL, json=body, params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 400, (
            "snake_case nested fields accepted; spec uses camelCase"
        )


# =========================================================================
# BUG 8: UUID format not validated
# =========================================================================
# The spec declares vendorIdsFilter items as `format: uuid`, but non-UUID
# strings are silently accepted.


class TestUUIDFormatValidation:
    """Spec: vendorIdsFilter items are format:uuid.
    Bug: arbitrary strings pass validation."""

    def test_vendor_ids_filter_rejects_non_uuid(self):
        r = _post(_base_body(vendorIdsFilter=["not-a-uuid"]))
        assert r.status_code == 400, (
            'vendorIdsFilter=["not-a-uuid"] accepted as 200; '
            "spec says format:uuid"
        )

    def test_closed_catalog_version_ids_rejects_non_uuid(self):
        r = _post(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": ["not-a-uuid"],
                "catalogVersionIdsOrderedByPreference": [],
            },
        ))
        assert r.status_code == 400, (
            'closedCatalogVersionIds=["not-a-uuid"] accepted as 200; '
            "spec says format:uuid"
        )

    def test_source_price_list_ids_rejects_non_uuid(self):
        r = _post(_base_body(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "catalogVersionIdsOrderedByPreference": [CV_EUR],
                "sourcePriceListIds": ["not-a-uuid"],
            },
        ))
        assert r.status_code == 400, (
            'sourcePriceListIds=["not-a-uuid"] accepted as 200; '
            "spec says format:uuid"
        )

    def test_core_articles_vendors_filter_rejects_non_uuid(self):
        r = _post(_base_body(coreArticlesVendorsFilter=["not-a-uuid"]))
        assert r.status_code == 400, (
            'coreArticlesVendorsFilter=["not-a-uuid"] accepted as 200; '
            "spec says format:uuid"
        )


# =========================================================================
# BUG 9: Sort validation delegated to ftsearch (error leaks internals)
# =========================================================================
# The ACL spec defines a regex pattern on sort items:
#   pattern: "^(articleId|name|price),(asc|desc)$"
# The ACL code does NOT validate this -- it passes raw strings to
# ftsearch, which validates them. The error message then says
# "ftsearch returned a non-2xx response" and includes ftsearch's
# internal error detail -- leaking an implementation detail.


class TestSortValidationLeak:
    """Spec: sort items must match the ACL's regex pattern.
    Bug: ACL delegates validation to ftsearch, leaking internal details."""

    def test_invalid_sort_error_does_not_mention_ftsearch(self):
        """Error for invalid sort should NOT mention ftsearch."""
        r = _post(_base_body(), sort=["score,asc"])
        assert r.status_code == 400
        err = r.json()
        assert "ftsearch" not in err["message"].lower(), (
            f"Error message leaks ftsearch internals: {err['message']}"
        )

    def test_empty_sort_string_error_does_not_mention_ftsearch(self):
        """Error for empty sort string should NOT mention ftsearch."""
        r = _post(_base_body(), sort=[""])
        assert r.status_code == 400
        err = r.json()
        assert "ftsearch" not in err["message"].lower(), (
            f"Error message leaks ftsearch internals: {err['message']}"
        )
