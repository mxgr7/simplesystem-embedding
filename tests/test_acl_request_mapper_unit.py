"""Red-team unit tests for `acl/mapping/request.py`.

Targets FriendlyId conversion edge cases, alias serialization, field
stripping, pagination params, selectedArticleSources dropped fields,
dual-currency forwarding, and boundary values.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from acl.mapping.request import (  # noqa: E402
    FtsearchRequest,
    _convert_legacy_article_id,
    _friendly_to_uuid,
    map_request,
)
from acl.models import LegacySearchRequest  # noqa: E402


def _minimal_request(**overrides) -> LegacySearchRequest:
    """Smallest valid LegacySearchRequest for isolated tests."""
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


# ============================================================================
# 1. FriendlyId conversion edge cases
# ============================================================================


class TestFriendlyIdConversion:
    """Edge cases in _friendly_to_uuid and _convert_legacy_article_id."""

    def test_very_short_friendly_id_one_char(self) -> None:
        """Single base62 character '1' = decimal 1 -> UUID int=1."""
        result = _friendly_to_uuid("1")
        assert result == uuid.UUID(int=1)

    def test_very_short_friendly_id_two_chars(self) -> None:
        """Two-char base62: 'A0' = 10*62 + 0 = 620 -> UUID int=620."""
        result = _friendly_to_uuid("A0")
        assert result == uuid.UUID(int=620)

    def test_single_zero_char(self) -> None:
        """'0' maps to value 0 -> UUID int=0 (all-zeros UUID)."""
        result = _friendly_to_uuid("0")
        assert result == uuid.UUID(int=0)

    def test_invalid_base62_character_exclamation(self) -> None:
        """Characters outside base62 alphabet raise KeyError."""
        with pytest.raises(KeyError):
            _friendly_to_uuid("abc!def")

    def test_invalid_base62_character_space(self) -> None:
        """Space is not in base62 alphabet."""
        with pytest.raises(KeyError):
            _friendly_to_uuid("abc def")

    def test_invalid_base62_character_plus(self) -> None:
        """'+' is not in base62 alphabet."""
        with pytest.raises(KeyError):
            _friendly_to_uuid("abc+def")

    def test_empty_string_friendly_id(self) -> None:
        """Empty string produces UUID(int=0) since no iterations occur."""
        # An empty string iterates zero times, leaving n=0.
        result = _friendly_to_uuid("")
        assert result == uuid.UUID(int=0)

    def test_maximum_128bit_uuid(self) -> None:
        """Maximum UUID value (2^128 - 1) round-trips through base62.

        The max UUID is ffffffff-ffff-ffff-ffff-ffffffffffff.
        We compute its base62 encoding and verify round-trip.
        """
        max_uuid = uuid.UUID(int=(2**128) - 1)
        # Encode the max value to base62 for the test
        n = max_uuid.int
        chars = []
        base62_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        while n > 0:
            chars.append(base62_alphabet[n % 62])
            n //= 62
        friendly = "".join(reversed(chars))
        # Now round-trip
        result = _friendly_to_uuid(friendly)
        assert result == max_uuid

    def test_overflow_beyond_128bit_raises_value_error(self) -> None:
        """A FriendlyId encoding a value > 2^128-1 should fail at UUID()."""
        # Construct a base62 string that decodes to 2^128 (one more than max)
        # 2^128 in base62 — we just need a very long string of max chars
        # The simplest approach: prepend a '1' to the max-value encoding
        max_val = (2**128) - 1
        n = max_val + 1  # 2^128
        chars = []
        base62_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        while n > 0:
            chars.append(base62_alphabet[n % 62])
            n //= 62
        friendly = "".join(reversed(chars))
        with pytest.raises(ValueError):
            _friendly_to_uuid(friendly)

    def test_convert_article_id_no_colon_passthrough(self) -> None:
        """An articleId with no colon is returned unchanged."""
        result = _convert_legacy_article_id("noColonHere")
        assert result == "noColonHere"

    def test_convert_article_id_three_colons_passthrough(self) -> None:
        """An articleId with 3+ colons (parts != 2) is returned unchanged."""
        input_id = "part1:part2:part3:part4"
        result = _convert_legacy_article_id(input_id)
        assert result == input_id

    def test_convert_article_id_empty_string(self) -> None:
        """Empty string has no colon -> returned unchanged."""
        result = _convert_legacy_article_id("")
        assert result == ""

    def test_convert_article_id_invalid_base62_in_first_part(self) -> None:
        """Invalid base62 chars in the FriendlyId part -> returned unchanged."""
        result = _convert_legacy_article_id("abc!xyz:MTIzNA")
        assert result == "abc!xyz:MTIzNA"

    def test_convert_article_id_valid_conversion(self) -> None:
        """A well-formed friendlyId:b64 is correctly converted."""
        # '1' in base62 = UUID(int=1) = 00000000-0000-0000-0000-000000000001
        result = _convert_legacy_article_id("1:MTIzNA")
        expected_uuid = uuid.UUID(int=1)
        assert result == f"{expected_uuid}:MTIzNA"

    def test_convert_article_id_empty_second_part(self) -> None:
        """FriendlyId with empty base64 part still converts."""
        result = _convert_legacy_article_id("1:")
        expected_uuid = uuid.UUID(int=1)
        assert result == f"{expected_uuid}:"


# ============================================================================
# 2. Body dump with alias serialization (camelCase output)
# ============================================================================


class TestCamelCaseAliasSerialization:
    """Verify model_dump(by_alias=True) produces camelCase keys."""

    def test_top_level_fields_are_camel_case(self) -> None:
        req = _minimal_request(queryString="bolt")
        out = map_request(req)
        body = out.body
        # After rename: query not queryString
        assert "query" in body
        # These should be camelCase, not snake_case
        assert "searchMode" in body
        assert "search_mode" not in body
        assert "maxDeliveryTime" in body
        assert "max_delivery_time" not in body
        assert "coreSortimentOnly" in body
        assert "core_sortiment_only" not in body
        assert "closedMarketplaceOnly" in body
        assert "closed_marketplace_only" not in body

    def test_nested_selected_article_sources_camel_case(self) -> None:
        req = _minimal_request(
            selectedArticleSources={
                "closedCatalogVersionIds": ["aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"],
                "catalogVersionIdsOrderedByPreference": [
                    "bbbbbbbb-2222-2222-2222-bbbbbbbbbbbb"
                ],
            },
        )
        out = map_request(req)
        sas = out.body["selectedArticleSources"]
        assert "closedCatalogVersionIds" in sas
        assert "closed_catalog_version_ids" not in sas
        assert "catalogVersionIdsOrderedByPreference" in sas
        assert "catalog_version_ids_ordered_by_preference" not in sas

    def test_price_filter_camel_case(self) -> None:
        req = _minimal_request(
            priceFilter={"min": 100, "max": 500, "currencyCode": "USD"},
        )
        out = map_request(req)
        pf = out.body["priceFilter"]
        assert "currencyCode" in pf
        assert "currency_code" not in pf

    def test_blocked_eclass_vendors_camel_case(self) -> None:
        req = _minimal_request(
            blockedEClassVendorsFilters=[{
                "vendorIds": ["aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"],
                "eClassVersion": "ECLASS_5_1",
                "blockedEClassGroups": [{"eClassGroupCode": 99999, "value": True}],
            }],
        )
        out = map_request(req)
        f = out.body["blockedEClassVendorsFilters"][0]
        assert "vendorIds" in f
        assert "vendor_ids" not in f
        assert "eClassVersion" in f
        assert "blockedEClassGroups" in f
        assert "eClassGroupCode" in f["blockedEClassGroups"][0]


# ============================================================================
# 3. Fields that should NOT appear in the output body
# ============================================================================


class TestDroppedFields:
    """Fields that the mapper must strip before sending to ftsearch."""

    def test_search_articles_by_never_in_output(self) -> None:
        req = _minimal_request()
        out = map_request(req)
        assert "searchArticlesBy" not in out.body
        assert "search_articles_by" not in out.body

    def test_explain_true_never_in_output(self) -> None:
        req = _minimal_request(explain=True)
        out = map_request(req)
        assert "explain" not in out.body

    def test_explain_false_never_in_output(self) -> None:
        req = _minimal_request(explain=False)
        out = map_request(req)
        assert "explain" not in out.body

    def test_page_page_size_sort_not_in_body(self) -> None:
        """Pagination/sort lives in params, never in body."""
        req = _minimal_request()
        out = map_request(req, page=5, page_size=50, sort=["name,asc"])
        assert "page" not in out.body
        assert "pageSize" not in out.body
        assert "sort" not in out.body

    def test_query_string_key_never_in_output(self) -> None:
        """Even with queryString set, the output key must be 'query'."""
        req = _minimal_request(queryString="test")
        out = map_request(req)
        assert "queryString" not in out.body


# ============================================================================
# 4. page/pageSize/sort params interaction
# ============================================================================


class TestPaginationParams:
    """Verify params dict construction for various page/pageSize/sort combos."""

    def test_defaults(self) -> None:
        out = map_request(_minimal_request())
        assert out.params["page"] == 1
        assert out.params["pageSize"] == 10
        assert "sort" not in out.params

    def test_custom_page_and_page_size(self) -> None:
        out = map_request(_minimal_request(), page=7, page_size=100)
        assert out.params["page"] == 7
        assert out.params["pageSize"] == 100

    def test_sort_single_value(self) -> None:
        out = map_request(_minimal_request(), sort=["price,desc"])
        assert out.params["sort"] == ["price,desc"]

    def test_sort_multiple_values(self) -> None:
        out = map_request(
            _minimal_request(), sort=["name,asc", "price,desc", "id,asc"]
        )
        assert out.params["sort"] == ["name,asc", "price,desc", "id,asc"]

    def test_sort_empty_list_omitted(self) -> None:
        """An empty sort list should not appear in params (same as None)."""
        out = map_request(_minimal_request(), sort=[])
        assert "sort" not in out.params

    def test_sort_none_omitted(self) -> None:
        out = map_request(_minimal_request(), sort=None)
        assert "sort" not in out.params

    def test_page_zero(self) -> None:
        """Page 0 is passed as-is — validation is the caller's job."""
        out = map_request(_minimal_request(), page=0)
        assert out.params["page"] == 0


# ============================================================================
# 5. selectedArticleSources with dropped fields present
# ============================================================================


class TestSelectedArticleSourcesDroppedFields:
    """Fields listed in _DROPPED_SELECTED_ARTICLE_SOURCES_FIELDS are stripped."""

    def test_customer_article_numbers_indexing_source_ids_stripped(self) -> None:
        req = _minimal_request(
            selectedArticleSources={
                "closedCatalogVersionIds": ["aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"],
                "customerArticleNumbersIndexingSourceIds": [
                    "cccccccc-3333-3333-3333-cccccccccccc"
                ],
            },
        )
        out = map_request(req)
        sas = out.body["selectedArticleSources"]
        assert "customerArticleNumbersIndexingSourceIds" not in sas

    def test_customer_managed_article_number_list_id_stripped(self) -> None:
        req = _minimal_request(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "customerManagedArticleNumberListId": "dddddddd-4444-4444-4444-dddddddddddd",
            },
        )
        out = map_request(req)
        sas = out.body["selectedArticleSources"]
        assert "customerManagedArticleNumberListId" not in sas

    def test_ui_customer_article_number_source_id_stripped(self) -> None:
        req = _minimal_request(
            selectedArticleSources={
                "closedCatalogVersionIds": [],
                "uiCustomerArticleNumberSourceId": "eeeeeeee-5555-5555-5555-eeeeeeeeeeee",
            },
        )
        out = map_request(req)
        sas = out.body["selectedArticleSources"]
        assert "uiCustomerArticleNumberSourceId" not in sas

    def test_all_dropped_fields_stripped_simultaneously(self) -> None:
        req = _minimal_request(
            selectedArticleSources={
                "closedCatalogVersionIds": ["aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"],
                "customerArticleNumbersIndexingSourceIds": [
                    "cccccccc-3333-3333-3333-cccccccccccc"
                ],
                "customerManagedArticleNumberListId": "dddddddd-4444-4444-4444-dddddddddddd",
                "uiCustomerArticleNumberSourceId": "eeeeeeee-5555-5555-5555-eeeeeeeeeeee",
                "sourcePriceListIds": ["ffffffff-6666-6666-6666-ffffffffffff"],
            },
        )
        out = map_request(req)
        sas = out.body["selectedArticleSources"]
        assert "customerArticleNumbersIndexingSourceIds" not in sas
        assert "customerManagedArticleNumberListId" not in sas
        assert "uiCustomerArticleNumberSourceId" not in sas
        # Non-dropped fields should still be there
        assert "closedCatalogVersionIds" in sas
        assert "sourcePriceListIds" in sas

    def test_non_dropped_fields_preserved(self) -> None:
        req = _minimal_request(
            selectedArticleSources={
                "closedCatalogVersionIds": ["aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"],
                "catalogVersionIdsOrderedByPreference": [
                    "bbbbbbbb-2222-2222-2222-bbbbbbbbbbbb"
                ],
                "sourcePriceListIds": ["cccccccc-3333-3333-3333-cccccccccccc"],
                "customerUploadedCoreArticleListSourceIds": [
                    "dddddddd-4444-4444-4444-dddddddddddd"
                ],
            },
        )
        out = map_request(req)
        sas = out.body["selectedArticleSources"]
        assert sas["closedCatalogVersionIds"] == [
            "aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"
        ]
        assert sas["catalogVersionIdsOrderedByPreference"] == [
            "bbbbbbbb-2222-2222-2222-bbbbbbbbbbbb"
        ]
        assert sas["sourcePriceListIds"] == ["cccccccc-3333-3333-3333-cccccccccccc"]
        assert sas["customerUploadedCoreArticleListSourceIds"] == [
            "dddddddd-4444-4444-4444-dddddddddddd"
        ]


# ============================================================================
# 6. priceFilter and currency both present (dual currency forwarding)
# ============================================================================


class TestDualCurrencyForwarding:
    """Top-level currency and priceFilter.currencyCode are independent."""

    def test_different_currencies_both_forwarded(self) -> None:
        """Typical case: display currency EUR, filter bounds in JPY."""
        req = _minimal_request(
            currency="EUR",
            priceFilter={"min": 10000, "max": 50000, "currencyCode": "JPY"},
        )
        out = map_request(req)
        assert out.body["currency"] == "EUR"
        assert out.body["priceFilter"]["currencyCode"] == "JPY"
        assert out.body["priceFilter"]["min"] == 10000
        assert out.body["priceFilter"]["max"] == 50000

    def test_same_currency_both_forwarded_without_collapse(self) -> None:
        """Even when both are EUR, mapper does NOT collapse them."""
        req = _minimal_request(
            currency="EUR",
            priceFilter={"min": 100, "max": 200, "currencyCode": "EUR"},
        )
        out = map_request(req)
        assert out.body["currency"] == "EUR"
        assert out.body["priceFilter"]["currencyCode"] == "EUR"

    def test_price_filter_without_bounds_only_currency_code(self) -> None:
        """priceFilter with only currencyCode (no min/max) is valid per model
        but unusual — verify it passes through."""
        req = _minimal_request(
            currency="USD",
            priceFilter={"currencyCode": "GBP"},
        )
        out = map_request(req)
        assert out.body["currency"] == "USD"
        pf = out.body["priceFilter"]
        assert pf["currencyCode"] == "GBP"
        # min/max excluded because they are None (exclude_none=True)
        assert "min" not in pf
        assert "max" not in pf

    def test_no_price_filter_only_top_level_currency(self) -> None:
        """priceFilter absent -> only top-level currency in output."""
        req = _minimal_request(currency="CHF")
        out = map_request(req)
        assert out.body["currency"] == "CHF"
        assert "priceFilter" not in out.body


# ============================================================================
# 7. Empty/null queryString handling
# ============================================================================


class TestQueryStringHandling:
    """Cover the queryString -> query rename with edge values."""

    def test_null_query_string_excluded_from_body(self) -> None:
        """None queryString means no `query` key in output (exclude_none)."""
        req = _minimal_request()  # queryString defaults to None
        out = map_request(req)
        assert "query" not in out.body
        assert "queryString" not in out.body

    def test_empty_string_query_forwarded(self) -> None:
        """An explicit empty string '' is a valid queryString — forwarded."""
        req = _minimal_request(queryString="")
        out = map_request(req)
        assert out.body["query"] == ""
        assert "queryString" not in out.body

    def test_whitespace_only_query_forwarded_as_is(self) -> None:
        """Mapper does NOT trim — that's ftsearch's job."""
        req = _minimal_request(queryString="   ")
        out = map_request(req)
        assert out.body["query"] == "   "

    def test_unicode_query_preserved(self) -> None:
        """Non-ASCII in queryString passes through unchanged."""
        req = _minimal_request(queryString="Schräubchen ☃")
        out = map_request(req)
        assert out.body["query"] == "Schräubchen ☃"


# ============================================================================
# 8. Maximum-size UUID FriendlyId (128-bit max value)
# ============================================================================


class TestMaxUuidFriendlyId:
    """Boundary: the largest possible UUID via FriendlyId encoding."""

    def test_max_uuid_in_article_ids_filter(self) -> None:
        """articleIdsFilter entry using the max UUID FriendlyId."""
        max_uuid = uuid.UUID(int=(2**128) - 1)
        # Encode to base62
        n = max_uuid.int
        chars = []
        base62_alphabet = (
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
        )
        while n > 0:
            chars.append(base62_alphabet[n % 62])
            n //= 62
        friendly = "".join(reversed(chars))

        req = _minimal_request(
            articleIdsFilter=[f"{friendly}:SU9NRQ"],
        )
        out = map_request(req)
        expected = f"{max_uuid}:SU9NRQ"
        assert out.body["articleIdsFilter"] == [expected]

    def test_zero_uuid_in_article_ids_filter(self) -> None:
        """articleIdsFilter with friendlyId '0' -> all-zeros UUID."""
        req = _minimal_request(articleIdsFilter=["0:YXJ0"])
        out = map_request(req)
        zero_uuid = uuid.UUID(int=0)
        assert out.body["articleIdsFilter"] == [f"{zero_uuid}:YXJ0"]

    def test_multiple_article_ids_mixed_valid_invalid(self) -> None:
        """Mix of valid conversions and passthrough (invalid) entries."""
        req = _minimal_request(
            articleIdsFilter=[
                "1:MTIz",           # valid: UUID(int=1):MTIz
                "no-colon-here",    # no colon -> passthrough
                "bad!char:b64",     # invalid base62 -> passthrough
                "a:b:c",            # 3 parts -> passthrough
            ],
        )
        out = map_request(req)
        expected = [
            f"{uuid.UUID(int=1)}:MTIz",
            "no-colon-here",
            "bad!char:b64",
            "a:b:c",
        ]
        assert out.body["articleIdsFilter"] == expected

    def test_empty_article_ids_filter_not_in_output(self) -> None:
        """Empty list -> excluded by exclude_none? No, empty list is not None.
        But the field has default_factory=list, so it may or may not appear
        depending on whether the model_dump includes defaults."""
        req = _minimal_request(articleIdsFilter=[])
        out = map_request(req)
        # Empty list is falsy, so body.get("articleIdsFilter") won't trigger
        # conversion, but the field may still be in the serialized output.
        # The key point: no crash, and if present it's still an empty list.
        if "articleIdsFilter" in out.body:
            assert out.body["articleIdsFilter"] == []


# ============================================================================
# Additional: FtsearchRequest dataclass output type
# ============================================================================


class TestFtsearchRequestOutput:
    """Verify the output dataclass structure."""

    def test_output_is_ftsearch_request(self) -> None:
        out = map_request(_minimal_request())
        assert isinstance(out, FtsearchRequest)

    def test_body_is_dict(self) -> None:
        out = map_request(_minimal_request())
        assert isinstance(out.body, dict)

    def test_params_is_dict(self) -> None:
        out = map_request(_minimal_request())
        assert isinstance(out.params, dict)

    def test_ftsearch_request_is_frozen(self) -> None:
        """FtsearchRequest is a frozen dataclass — assignment raises."""
        out = map_request(_minimal_request())
        with pytest.raises(Exception):  # FrozenInstanceError
            out.body = {}  # type: ignore[misc]
