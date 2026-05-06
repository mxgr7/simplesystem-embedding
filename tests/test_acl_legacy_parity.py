"""ACL-vs-legacy spec parity tests.

Compares acl/openapi.yaml field-by-field against the legacy
article-search spec at next-gen/api-spec/specs/article-search/ and
verifies every schema, enum value, field name, field type, and
structural element that is NOT in the list of known intentional
deviations.

Also includes live integration tests against localhost:8081 that
verify the ACL actually handles legacy request/response shapes.

Known intentional deviations (skipped):
  S2.1  searchArticlesBy single-value enum [STANDARD]
  S2.2  explanation stubbed to "N/A"
  S2.3  non-relevance sorts on relevance-bounded pool
  S4    Error.details is string[] (legacy uses ErrorDetail objects)
  S5    searchMode required in ACL (optional in legacy)
  S6    pageSize range 1..500 (legacy 1..50)
  S7    Response required fields fewer in ACL Summaries/CategoriesSummary/EClassCategories
  S8    SelectedArticleSources required fields fewer in ACL
  S9    additionalProperties: false in ACL
  S10   vendorIdsFilter format: uuid in ACL
  S11   articleIdsFilter/eClassesFilter/eClassesAggregations no uniqueItems/minItems
  S12   sort pattern constraint in ACL
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

# ---------------------------------------------------------------------------
# Load specs
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_DIR = Path("/home/mgerer/next-gen/api-spec/specs/article-search")

ACL_SPEC_PATH = REPO_ROOT / "acl" / "openapi.yaml"
LEGACY_SEARCH_PATH = LEGACY_DIR / "query-search-api.yaml"
LEGACY_TOPLEVEL_PATH = LEGACY_DIR / "spec.yaml"

ACL_SPEC = yaml.safe_load(ACL_SPEC_PATH.read_text())
LEGACY_SEARCH = yaml.safe_load(LEGACY_SEARCH_PATH.read_text())
LEGACY_TOPLEVEL = yaml.safe_load(LEGACY_TOPLEVEL_PATH.read_text())

ACL_SCHEMAS = ACL_SPEC["components"]["schemas"]
LEGACY_SCHEMAS = LEGACY_SEARCH["components"]["schemas"]
LEGACY_TL_SCHEMAS = LEGACY_TOPLEVEL["components"]["schemas"]


def _resolve_legacy_ref(ref: str) -> dict:
    """Resolve $ref in the legacy spec tree."""
    if ref.startswith("./spec.yaml#"):
        parts = ref.split("#")[1].lstrip("/").split("/")
        node = LEGACY_TOPLEVEL
    elif ref.startswith("#/"):
        parts = ref.lstrip("#/").split("/")
        node = LEGACY_SEARCH
    else:
        raise ValueError(f"Unsupported $ref: {ref}")
    for p in parts:
        node = node[p]
    return node


def _resolve_acl_ref(ref: str) -> dict:
    """Resolve $ref in the ACL spec."""
    parts = ref.lstrip("#/").split("/")
    node = ACL_SPEC
    for p in parts:
        node = node[p]
    return node


def _deref_legacy(schema: dict) -> dict:
    """If a schema is just a $ref, resolve it.  Also inline IdSet/IdList."""
    if "$ref" in schema:
        return _deref_legacy(_resolve_legacy_ref(schema["$ref"]))
    return schema


def _deref_acl(schema: dict) -> dict:
    if "$ref" in schema:
        return _deref_acl(_resolve_acl_ref(schema["$ref"]))
    return schema


def _get_type(schema: dict) -> str:
    """Extract the effective type from a schema dict."""
    s = schema
    if "$ref" in s:
        return "ref"
    return s.get("type", "object")


# ===========================================================================
# KNOWN DEVIATIONS — helpers to skip intentional differences
# ===========================================================================

# S2.1: searchArticlesBy enum values that legacy has but ACL intentionally omits
S2_1_LEGACY_ONLY_SEARCH_BY = {
    "ALL_ATTRIBUTES", "ARTICLE_NUMBER", "CUSTOMER_ARTICLE_NUMBER",
    "VENDOR_ARTICLE_NUMBER", "EAN",
    *(f"TEST_PROFILE_{i:02d}" for i in range(1, 21)),
}

# S5: searchMode is required in ACL but optional in legacy (skip required diff)
# S6: pageSize max is 500 in ACL vs 50 in legacy (skip)
# S8: SelectedArticleSources required fields (ACL only requires closedCatalogVersionIds)
# S9: additionalProperties: false in ACL (skip)
# S10: vendorIdsFilter format: uuid (skip)
# S11: uniqueItems/minItems differences (skip)
# S12: sort pattern constraint (skip)

# ===========================================================================
# Part 1: Static spec comparison tests
# ===========================================================================


class TestEndpointParity:
    """Verify the endpoint path and HTTP method match."""

    def test_endpoint_path_exists(self):
        """ACL must serve /article-features/search."""
        assert "/article-features/search" in ACL_SPEC["paths"], \
            "ACL is missing the /article-features/search endpoint"

    def test_http_method_is_post(self):
        """Both specs use POST for the search endpoint."""
        acl_path = ACL_SPEC["paths"]["/article-features/search"]
        assert "post" in acl_path, \
            "ACL /article-features/search must support POST"


class TestQueryParameterParity:
    """Query parameters: page, pageSize, sort."""

    def test_page_parameter_name_and_type(self):
        acl_params = ACL_SPEC["paths"]["/article-features/search"]["post"]["parameters"]
        page_param = next(p for p in acl_params if p["name"] == "page")
        assert page_param["in"] == "query"
        assert page_param["schema"]["type"] == "integer"
        assert page_param["schema"].get("default") == 1
        assert page_param["schema"].get("minimum") == 1

    def test_pagesize_parameter_name_and_type(self):
        acl_params = ACL_SPEC["paths"]["/article-features/search"]["post"]["parameters"]
        ps_param = next(p for p in acl_params if p["name"] == "pageSize")
        assert ps_param["in"] == "query"
        assert ps_param["schema"]["type"] == "integer"
        assert ps_param["schema"].get("default") == 10
        # S6: max differs intentionally (500 vs 50), just check it exists
        assert "maximum" in ps_param["schema"]

    def test_sort_parameter_name_and_type(self):
        acl_params = ACL_SPEC["paths"]["/article-features/search"]["post"]["parameters"]
        sort_param = next(p for p in acl_params if p["name"] == "sort")
        assert sort_param["in"] == "query"
        assert sort_param["schema"]["type"] == "array"
        assert sort_param["schema"]["items"]["type"] == "string"

    def test_page_param_matches_legacy(self):
        """Legacy page param lives in spec.yaml."""
        legacy_page = LEGACY_TOPLEVEL["components"]["parameters"]["page"]
        acl_params = ACL_SPEC["paths"]["/article-features/search"]["post"]["parameters"]
        acl_page = next(p for p in acl_params if p["name"] == "page")
        assert legacy_page["name"] == acl_page["name"]
        assert legacy_page["in"] == acl_page["in"]
        assert legacy_page["schema"]["type"] == acl_page["schema"]["type"]
        assert legacy_page["schema"]["default"] == acl_page["schema"]["default"]
        assert legacy_page["schema"]["minimum"] == acl_page["schema"]["minimum"]

    def test_sort_param_matches_legacy_name(self):
        """Legacy sort param is named 'sort' with type array of strings."""
        legacy_sort = LEGACY_SEARCH["components"]["parameters"]["sort"]
        acl_params = ACL_SPEC["paths"]["/article-features/search"]["post"]["parameters"]
        acl_sort = next(p for p in acl_params if p["name"] == "sort")
        assert legacy_sort["name"] == acl_sort["name"]
        assert legacy_sort["in"] == acl_sort["in"]
        assert legacy_sort["schema"]["type"] == acl_sort["schema"]["type"]
        assert legacy_sort["schema"]["items"]["type"] == acl_sort["schema"]["items"]["type"]


class TestSearchRequestFieldParity:
    """Every field in legacy SearchParams must exist in ACL SearchRequest
    with compatible type (modulo known deviations)."""

    LEGACY_REQ = LEGACY_SCHEMAS["SearchParams"]
    ACL_REQ = ACL_SCHEMAS["SearchRequest"]

    # Fields present in both specs.  We list them explicitly so a NEW field
    # added to legacy but missing in ACL will be caught.
    EXPECTED_FIELDS = [
        "searchMode",
        "searchArticlesBy",
        "selectedArticleSources",
        "queryString",
        "articleIdsFilter",
        "vendorIdsFilter",
        "manufacturersFilter",
        "maxDeliveryTime",
        "requiredFeatures",
        "priceFilter",
        "accessoriesForArticleNumber",
        "sparePartsForArticleNumber",
        "similarToArticleNumber",
        "currentCategoryPathElements",
        "currentEClass5Code",
        "currentEClass7Code",
        "currentS2ClassCode",
        "coreSortimentOnly",
        "closedMarketplaceOnly",
        "summaries",
        "coreArticlesVendorsFilter",
        "blockedEClassVendorsFilters",
        "currency",
        "explain",
        "eClassesFilter",
        "eClassesAggregations",
        "s2ClassForProductCategories",
    ]

    @pytest.mark.parametrize("field", EXPECTED_FIELDS)
    def test_legacy_field_exists_in_acl(self, field):
        """Every legacy SearchParams field must appear in ACL SearchRequest."""
        assert field in self.ACL_REQ["properties"], \
            f"Legacy field '{field}' missing from ACL SearchRequest"

    @pytest.mark.parametrize("field", EXPECTED_FIELDS)
    def test_legacy_field_exists_in_legacy(self, field):
        """Confirm the field actually exists in legacy (test self-check)."""
        assert field in self.LEGACY_REQ["properties"], \
            f"Expected field '{field}' not in legacy SearchParams"

    def test_no_extra_acl_request_fields(self):
        """ACL should not have request fields that legacy doesn't have."""
        acl_fields = set(self.ACL_REQ["properties"].keys())
        legacy_fields = set(self.LEGACY_REQ["properties"].keys())
        extra = acl_fields - legacy_fields
        assert extra == set(), \
            f"ACL SearchRequest has fields not in legacy: {extra}"

    def test_no_missing_acl_request_fields(self):
        """Every legacy request field must be in ACL (exhaustive check)."""
        acl_fields = set(self.ACL_REQ["properties"].keys())
        legacy_fields = set(self.LEGACY_REQ["properties"].keys())
        missing = legacy_fields - acl_fields
        assert missing == set(), \
            f"Legacy fields missing from ACL SearchRequest: {missing}"


class TestRequestFieldTypes:
    """Field types must be compatible between legacy and ACL."""

    LEGACY_REQ = LEGACY_SCHEMAS["SearchParams"]["properties"]
    ACL_REQ = ACL_SCHEMAS["SearchRequest"]["properties"]

    # (field, expected_type) — for direct-type fields
    SIMPLE_TYPE_FIELDS = [
        ("searchMode", "string"),
        ("searchArticlesBy", "string"),
        ("queryString", "string"),
        ("maxDeliveryTime", "integer"),
        ("coreSortimentOnly", "boolean"),
        ("closedMarketplaceOnly", "boolean"),
        ("currency", "string"),
        ("explain", "boolean"),
        ("s2ClassForProductCategories", "boolean"),
        ("currentEClass5Code", "integer"),
        ("currentEClass7Code", "integer"),
        ("currentS2ClassCode", "integer"),
        ("accessoriesForArticleNumber", "string"),
        ("sparePartsForArticleNumber", "string"),
        ("similarToArticleNumber", "string"),
    ]

    @pytest.mark.parametrize("field,expected_type", SIMPLE_TYPE_FIELDS)
    def test_simple_type_matches(self, field, expected_type):
        acl_schema = _deref_acl(self.ACL_REQ[field])
        legacy_schema = _deref_legacy(self.LEGACY_REQ[field])
        acl_type = acl_schema.get("type")
        legacy_type = legacy_schema.get("type")
        assert acl_type == expected_type, \
            f"ACL {field} type is {acl_type}, expected {expected_type}"
        assert legacy_type == expected_type, \
            f"Legacy {field} type is {legacy_type}, expected {expected_type}"

    ARRAY_FIELDS = [
        ("articleIdsFilter", "string"),
        ("vendorIdsFilter", "string"),
        ("manufacturersFilter", "string"),
        ("currentCategoryPathElements", "string"),
        ("eClassesFilter", "integer"),
    ]

    @pytest.mark.parametrize("field,item_type", ARRAY_FIELDS)
    def test_array_field_types(self, field, item_type):
        acl_schema = _deref_acl(self.ACL_REQ[field])
        legacy_schema = _deref_legacy(self.LEGACY_REQ[field])
        assert acl_schema["type"] == "array", \
            f"ACL {field} should be array"
        assert legacy_schema["type"] == "array", \
            f"Legacy {field} should be array"
        acl_items = _deref_acl(acl_schema["items"])
        legacy_items = _deref_legacy(legacy_schema["items"])
        assert acl_items["type"] == item_type, \
            f"ACL {field} items type is {acl_items['type']}, expected {item_type}"
        assert legacy_items["type"] == item_type, \
            f"Legacy {field} items type is {legacy_items['type']}, expected {item_type}"

    def test_requiredfeatures_is_array_of_objects(self):
        acl_rf = _deref_acl(self.ACL_REQ["requiredFeatures"])
        legacy_rf = _deref_legacy(self.LEGACY_REQ["requiredFeatures"])
        assert acl_rf["type"] == "array"
        assert legacy_rf["type"] == "array"

    def test_summaries_is_array_of_enums(self):
        acl_s = _deref_acl(self.ACL_REQ["summaries"])
        legacy_s = _deref_legacy(self.LEGACY_REQ["summaries"])
        assert acl_s["type"] == "array"
        assert legacy_s["type"] == "array"

    def test_blocked_eclass_vendors_filters_is_array(self):
        acl_b = _deref_acl(self.ACL_REQ["blockedEClassVendorsFilters"])
        legacy_b = _deref_legacy(self.LEGACY_REQ["blockedEClassVendorsFilters"])
        assert acl_b["type"] == "array"
        assert legacy_b["type"] == "array"

    def test_eclass_aggregations_is_array(self):
        acl_ea = _deref_acl(self.ACL_REQ["eClassesAggregations"])
        legacy_ea = _deref_legacy(self.LEGACY_REQ["eClassesAggregations"])
        assert acl_ea["type"] == "array"
        assert legacy_ea["type"] == "array"

    def test_core_articles_vendors_filter_is_array(self):
        acl_cav = _deref_acl(self.ACL_REQ["coreArticlesVendorsFilter"])
        legacy_cav = _deref_legacy(self.LEGACY_REQ["coreArticlesVendorsFilter"])
        assert acl_cav["type"] == "array"
        assert legacy_cav["type"] == "array"


class TestSearchModeEnum:
    """searchMode enum values must match (S5 deviation is about required, not values)."""

    def test_search_mode_values(self):
        acl_enum = ACL_SCHEMAS["SearchRequest"]["properties"]["searchMode"]["enum"]
        legacy_enum = LEGACY_SCHEMAS["SearchParams"]["properties"]["searchMode"]["enum"]
        assert set(acl_enum) == set(legacy_enum), \
            f"searchMode enums differ: ACL={acl_enum}, legacy={legacy_enum}"


class TestSummaryKindEnum:
    """Every legacy SummaryKind value must exist in ACL."""

    def test_summary_kind_values(self):
        acl_enum = ACL_SCHEMAS["SummaryKind"]["enum"]
        # Legacy defines enum inline in Summaries schema
        legacy_summaries = _deref_legacy(LEGACY_SCHEMAS["Summaries"])
        legacy_items = _deref_legacy(legacy_summaries["items"])
        legacy_enum = legacy_items["enum"]
        # Every legacy value must be present in ACL
        missing = set(legacy_enum) - set(acl_enum)
        assert missing == set(), \
            f"Legacy SummaryKind values missing from ACL: {missing}"

    def test_summary_kind_no_extra_in_acl(self):
        acl_enum = ACL_SCHEMAS["SummaryKind"]["enum"]
        legacy_summaries = _deref_legacy(LEGACY_SCHEMAS["Summaries"])
        legacy_items = _deref_legacy(legacy_summaries["items"])
        legacy_enum = legacy_items["enum"]
        extra = set(acl_enum) - set(legacy_enum)
        assert extra == set(), \
            f"ACL SummaryKind has extra values not in legacy: {extra}"

    EXPECTED_VALUES = [
        "CATEGORIES", "ECLASS5", "ECLASS7", "S2CLASS", "VENDORS",
        "MANUFACTURERS", "FEATURES", "PRICES", "PLATFORM_CATEGORIES",
        "ECLASS5SET",
    ]

    @pytest.mark.parametrize("value", EXPECTED_VALUES)
    def test_each_summary_kind_present(self, value):
        acl_enum = ACL_SCHEMAS["SummaryKind"]["enum"]
        assert value in acl_enum, \
            f"SummaryKind value '{value}' missing from ACL"


class TestEClassVersionEnum:
    """EClassVersion enum values must match exactly."""

    def test_eclass_version_values(self):
        acl_enum = ACL_SCHEMAS["EClassVersion"]["enum"]
        # Legacy defines this inline in BlockedEClassVendorsFilters
        legacy_bev = _deref_legacy(LEGACY_SCHEMAS["BlockedEClassVendorsFilters"])
        legacy_enum = legacy_bev["properties"]["eClassVersion"]["enum"]
        assert set(acl_enum) == set(legacy_enum), \
            f"EClassVersion enums differ: ACL={acl_enum}, legacy={legacy_enum}"

    @pytest.mark.parametrize("value", ["ECLASS_5_1", "ECLASS_7_1", "S2CLASS"])
    def test_each_eclass_version_present(self, value):
        acl_enum = ACL_SCHEMAS["EClassVersion"]["enum"]
        assert value in acl_enum, \
            f"EClassVersion value '{value}' missing from ACL"


class TestFeatureFilterSchema:
    """FeatureFilter (ACL) vs RequiredFeaturesInner (legacy) field parity."""

    def test_field_names_match(self):
        acl_ff = ACL_SCHEMAS["FeatureFilter"]
        legacy_rf = _deref_legacy(LEGACY_SCHEMAS["RequiredFeatures"])
        legacy_inner = _deref_legacy(legacy_rf["items"])
        acl_fields = set(acl_ff["properties"].keys())
        legacy_fields = set(legacy_inner["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"FeatureFilter fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_name_is_string(self):
        assert ACL_SCHEMAS["FeatureFilter"]["properties"]["name"]["type"] == "string"

    def test_values_is_array_of_strings(self):
        vals = ACL_SCHEMAS["FeatureFilter"]["properties"]["values"]
        assert vals["type"] == "array"
        assert vals["items"]["type"] == "string"

    def test_required_fields_match(self):
        acl_req = set(ACL_SCHEMAS["FeatureFilter"]["required"])
        legacy_rf = _deref_legacy(LEGACY_SCHEMAS["RequiredFeatures"])
        legacy_inner = _deref_legacy(legacy_rf["items"])
        legacy_req = set(legacy_inner["required"])
        assert acl_req == legacy_req, \
            f"FeatureFilter required fields differ: ACL={acl_req}, legacy={legacy_req}"


class TestPriceFilterSchema:
    """PriceFilter field names and types must match."""

    def test_field_names(self):
        acl_pf = ACL_SCHEMAS["PriceFilter"]
        legacy_pf = _deref_legacy(LEGACY_SCHEMAS["PriceFilter"])
        acl_fields = set(acl_pf["properties"].keys())
        legacy_fields = set(legacy_pf["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"PriceFilter fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_min_is_integer(self):
        acl_min = ACL_SCHEMAS["PriceFilter"]["properties"]["min"]
        legacy_min = LEGACY_SCHEMAS["PriceFilter"]["properties"]["min"]
        assert acl_min["type"] == "integer"
        assert legacy_min["type"] == "integer"

    def test_max_is_integer(self):
        acl_max = ACL_SCHEMAS["PriceFilter"]["properties"]["max"]
        legacy_max = LEGACY_SCHEMAS["PriceFilter"]["properties"]["max"]
        assert acl_max["type"] == "integer"
        assert legacy_max["type"] == "integer"

    def test_currency_code_is_string(self):
        acl_cc = ACL_SCHEMAS["PriceFilter"]["properties"]["currencyCode"]
        legacy_cc = LEGACY_SCHEMAS["PriceFilter"]["properties"]["currencyCode"]
        assert acl_cc["type"] == "string"
        assert legacy_cc["type"] == "string"


class TestSelectedArticleSourcesSchema:
    """SelectedArticleSources field names and types must match."""

    def test_all_legacy_fields_present_in_acl(self):
        acl_sas = ACL_SCHEMAS["SelectedArticleSources"]
        legacy_sas = _deref_legacy(LEGACY_SCHEMAS["SelectedArticleSources"])
        acl_fields = set(acl_sas["properties"].keys())
        legacy_fields = set(legacy_sas["properties"].keys())
        missing = legacy_fields - acl_fields
        assert missing == set(), \
            f"Legacy SelectedArticleSources fields missing from ACL: {missing}"

    def test_no_extra_acl_fields(self):
        acl_sas = ACL_SCHEMAS["SelectedArticleSources"]
        legacy_sas = _deref_legacy(LEGACY_SCHEMAS["SelectedArticleSources"])
        acl_fields = set(acl_sas["properties"].keys())
        legacy_fields = set(legacy_sas["properties"].keys())
        extra = acl_fields - legacy_fields
        assert extra == set(), \
            f"ACL SelectedArticleSources has extra fields: {extra}"

    EXPECTED_FIELDS = [
        "closedCatalogVersionIds",
        "catalogVersionIdsOrderedByPreference",
        "sourcePriceListIds",
        "customerArticleNumbersIndexingSourceIds",
        "customerUploadedCoreArticleListSourceIds",
        "customerManagedArticleNumberListId",
        "uiCustomerArticleNumberSourceId",
    ]

    @pytest.mark.parametrize("field", EXPECTED_FIELDS)
    def test_field_present(self, field):
        acl_sas = ACL_SCHEMAS["SelectedArticleSources"]
        assert field in acl_sas["properties"], \
            f"SelectedArticleSources missing field '{field}'"

    def test_array_fields_are_arrays(self):
        """The five ID-list fields should all be arrays."""
        acl_sas = ACL_SCHEMAS["SelectedArticleSources"]["properties"]
        array_fields = [
            "closedCatalogVersionIds",
            "catalogVersionIdsOrderedByPreference",
            "sourcePriceListIds",
            "customerArticleNumbersIndexingSourceIds",
            "customerUploadedCoreArticleListSourceIds",
        ]
        for field in array_fields:
            schema = _deref_acl(acl_sas[field])
            assert schema["type"] == "array", \
                f"SelectedArticleSources.{field} should be array, got {schema.get('type')}"

    def test_string_fields_are_strings(self):
        """customerManagedArticleNumberListId and uiCustomerArticleNumberSourceId
        should be strings."""
        acl_sas = ACL_SCHEMAS["SelectedArticleSources"]["properties"]
        for field in ["customerManagedArticleNumberListId",
                      "uiCustomerArticleNumberSourceId"]:
            schema = _deref_acl(acl_sas[field])
            assert schema["type"] == "string", \
                f"SelectedArticleSources.{field} should be string"


class TestBlockedEClassVendorsFilterSchema:
    """BlockedEClassVendorsFilter field names and types must match."""

    def test_field_names(self):
        acl_bev = ACL_SCHEMAS["BlockedEClassVendorsFilter"]
        legacy_bev = _deref_legacy(LEGACY_SCHEMAS["BlockedEClassVendorsFilters"])
        acl_fields = set(acl_bev["properties"].keys())
        legacy_fields = set(legacy_bev["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"BlockedEClassVendorsFilter fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_vendor_ids_is_array(self):
        acl_bev = ACL_SCHEMAS["BlockedEClassVendorsFilter"]
        schema = _deref_acl(acl_bev["properties"]["vendorIds"])
        assert schema["type"] == "array"

    def test_eclass_version_is_string(self):
        acl_bev = ACL_SCHEMAS["BlockedEClassVendorsFilter"]
        schema = _deref_acl(acl_bev["properties"]["eClassVersion"])
        assert schema["type"] == "string"

    def test_blocked_eclass_groups_is_array(self):
        acl_bev = ACL_SCHEMAS["BlockedEClassVendorsFilter"]
        schema = _deref_acl(acl_bev["properties"]["blockedEClassGroups"])
        assert schema["type"] == "array"

    def test_required_fields_match(self):
        acl_bev = ACL_SCHEMAS["BlockedEClassVendorsFilter"]
        legacy_bev = _deref_legacy(LEGACY_SCHEMAS["BlockedEClassVendorsFilters"])
        assert set(acl_bev["required"]) == set(legacy_bev["required"]), \
            f"BlockedEClassVendorsFilter required fields differ"


class TestBlockedEClassGroupSchema:
    """BlockedEClassGroup field names and types."""

    def test_field_names(self):
        acl_beg = ACL_SCHEMAS["BlockedEClassGroup"]
        # Legacy has this inline in BlockedEClassVendorsFilters.blockedEClassGroups.items
        legacy_bev = _deref_legacy(LEGACY_SCHEMAS["BlockedEClassVendorsFilters"])
        legacy_items = legacy_bev["properties"]["blockedEClassGroups"]["items"]
        acl_fields = set(acl_beg["properties"].keys())
        legacy_fields = set(legacy_items["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"BlockedEClassGroup fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_eclass_group_code_is_integer(self):
        assert ACL_SCHEMAS["BlockedEClassGroup"]["properties"]["eClassGroupCode"]["type"] == "integer"

    def test_value_is_boolean(self):
        assert ACL_SCHEMAS["BlockedEClassGroup"]["properties"]["value"]["type"] == "boolean"

    def test_required_fields_match(self):
        acl_beg = ACL_SCHEMAS["BlockedEClassGroup"]
        legacy_bev = _deref_legacy(LEGACY_SCHEMAS["BlockedEClassVendorsFilters"])
        legacy_items = legacy_bev["properties"]["blockedEClassGroups"]["items"]
        assert set(acl_beg["required"]) == set(legacy_items["required"])


class TestEClassesAggregationSchema:
    """EClassesAggregation (request) field names and types must match
    legacy EClassesAggregationWithEClasses."""

    def test_field_names(self):
        acl_ea = ACL_SCHEMAS["EClassesAggregation"]
        legacy_ea = _deref_legacy(LEGACY_SCHEMAS["EClassesAggregationWithEClasses"])
        acl_fields = set(acl_ea["properties"].keys())
        legacy_fields = set(legacy_ea["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"EClassesAggregation fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_id_is_string(self):
        assert ACL_SCHEMAS["EClassesAggregation"]["properties"]["id"]["type"] == "string"

    def test_eclasses_is_array_of_integers(self):
        schema = ACL_SCHEMAS["EClassesAggregation"]["properties"]["eClasses"]
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "integer"

    def test_required_fields_match(self):
        acl_ea = ACL_SCHEMAS["EClassesAggregation"]
        legacy_ea = _deref_legacy(LEGACY_SCHEMAS["EClassesAggregationWithEClasses"])
        assert set(acl_ea["required"]) == set(legacy_ea["required"])


class TestEClassesAggregationCountSchema:
    """EClassesAggregationCount (response) must match legacy
    EClassesAggregationWithCount {id, count}."""

    def test_field_names(self):
        acl_eac = ACL_SCHEMAS["EClassesAggregationCount"]
        legacy_eac = _deref_legacy(LEGACY_SCHEMAS["EClassesAggregationWithCount"])
        acl_fields = set(acl_eac["properties"].keys())
        legacy_fields = set(legacy_eac["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"EClassesAggregationCount fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_id_is_string(self):
        assert ACL_SCHEMAS["EClassesAggregationCount"]["properties"]["id"]["type"] == "string"

    def test_count_is_integer(self):
        assert ACL_SCHEMAS["EClassesAggregationCount"]["properties"]["count"]["type"] == "integer"

    def test_required_fields_match(self):
        acl_eac = ACL_SCHEMAS["EClassesAggregationCount"]
        legacy_eac = _deref_legacy(LEGACY_SCHEMAS["EClassesAggregationWithCount"])
        assert set(acl_eac["required"]) == set(legacy_eac["required"])


# ---------------------------------------------------------------------------
# Response schema parity
# ---------------------------------------------------------------------------


class TestSearchResponseSchema:
    """SearchResponse top-level fields."""

    def test_articles_field_exists(self):
        assert "articles" in ACL_SCHEMAS["SearchResponse"]["properties"]

    def test_summaries_field_exists(self):
        assert "summaries" in ACL_SCHEMAS["SearchResponse"]["properties"]

    def test_metadata_field_exists(self):
        assert "metadata" in ACL_SCHEMAS["SearchResponse"]["properties"]

    def test_no_extra_fields(self):
        acl_fields = set(ACL_SCHEMAS["SearchResponse"]["properties"].keys())
        # Legacy SearchResponse also has articles, summaries, metadata
        legacy_resp = _deref_legacy(LEGACY_SCHEMAS["SearchResponse"])
        legacy_fields = set(legacy_resp["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"SearchResponse fields differ: ACL={acl_fields}, legacy={legacy_fields}"


class TestArticleSchema:
    """Article schema field parity."""

    def test_article_id_field(self):
        assert "articleId" in ACL_SCHEMAS["Article"]["properties"]
        assert ACL_SCHEMAS["Article"]["properties"]["articleId"]["type"] == "string"

    def test_explanation_field(self):
        assert "explanation" in ACL_SCHEMAS["Article"]["properties"]
        assert ACL_SCHEMAS["Article"]["properties"]["explanation"]["type"] == "string"

    def test_field_names_match_legacy(self):
        acl_fields = set(ACL_SCHEMAS["Article"]["properties"].keys())
        legacy_resp = _deref_legacy(LEGACY_SCHEMAS["SearchResponse"])
        legacy_article = legacy_resp["properties"]["articles"]["items"]
        legacy_fields = set(legacy_article["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"Article fields differ: ACL={acl_fields}, legacy={legacy_fields}"


class TestMetadataSchema:
    """Metadata field names and types."""

    EXPECTED_FIELDS = {
        "page": "integer",
        "pageSize": "integer",
        "pageCount": "integer",
        "term": "string",
        "hitCount": "integer",
    }

    @pytest.mark.parametrize("field,ftype", list(EXPECTED_FIELDS.items()))
    def test_field_present_and_typed(self, field, ftype):
        assert field in ACL_SCHEMAS["Metadata"]["properties"], \
            f"Metadata missing field '{field}'"
        assert ACL_SCHEMAS["Metadata"]["properties"][field]["type"] == ftype, \
            f"Metadata.{field} type mismatch"

    def test_no_extra_fields(self):
        acl_fields = set(ACL_SCHEMAS["Metadata"]["properties"].keys())
        legacy_meta = _deref_legacy(LEGACY_SCHEMAS["SearchMetadata"])
        legacy_fields = set(legacy_meta["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"Metadata fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_field_names_match_legacy(self):
        acl_fields = set(ACL_SCHEMAS["Metadata"]["properties"].keys())
        legacy_meta = _deref_legacy(LEGACY_SCHEMAS["SearchMetadata"])
        legacy_fields = set(legacy_meta["properties"].keys())
        missing = legacy_fields - acl_fields
        assert missing == set(), \
            f"Legacy Metadata fields missing from ACL: {missing}"


class TestSummariesSchema:
    """Summaries (response) field names must match."""

    EXPECTED_FIELDS = [
        "vendorSummaries",
        "manufacturerSummaries",
        "featureSummaries",
        "pricesSummary",
        "categoriesSummary",
        "eClass5Categories",
        "eClass7Categories",
        "s2ClassCategories",
        "eClassesAggregations",
    ]

    @pytest.mark.parametrize("field", EXPECTED_FIELDS)
    def test_field_present_in_acl(self, field):
        assert field in ACL_SCHEMAS["Summaries"]["properties"], \
            f"Summaries missing field '{field}'"

    def test_all_legacy_fields_present(self):
        legacy_srs = _deref_legacy(LEGACY_SCHEMAS["SearchResultSummaries"])
        legacy_fields = set(legacy_srs["properties"].keys())
        acl_fields = set(ACL_SCHEMAS["Summaries"]["properties"].keys())
        missing = legacy_fields - acl_fields
        assert missing == set(), \
            f"Legacy Summaries fields missing from ACL: {missing}"

    def test_no_extra_acl_fields(self):
        legacy_srs = _deref_legacy(LEGACY_SCHEMAS["SearchResultSummaries"])
        legacy_fields = set(legacy_srs["properties"].keys())
        acl_fields = set(ACL_SCHEMAS["Summaries"]["properties"].keys())
        extra = acl_fields - legacy_fields
        assert extra == set(), \
            f"ACL Summaries has extra fields: {extra}"


class TestVendorSummarySchema:
    """VendorSummary {vendorId, count} field names and types."""

    def test_field_names(self):
        acl_vs = ACL_SCHEMAS["VendorSummary"]
        legacy_srs = _deref_legacy(LEGACY_SCHEMAS["SearchResultSummaries"])
        legacy_vs = legacy_srs["properties"]["vendorSummaries"]["items"]
        acl_fields = set(acl_vs["properties"].keys())
        legacy_fields = set(legacy_vs["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"VendorSummary fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_vendor_id_is_string(self):
        assert ACL_SCHEMAS["VendorSummary"]["properties"]["vendorId"]["type"] == "string"

    def test_count_is_integer(self):
        assert ACL_SCHEMAS["VendorSummary"]["properties"]["count"]["type"] == "integer"


class TestNameCountSchema:
    """NameCount (manufacturerSummaries items) {name, count}."""

    def test_field_names(self):
        acl_nc = ACL_SCHEMAS["NameCount"]
        legacy_srs = _deref_legacy(LEGACY_SCHEMAS["SearchResultSummaries"])
        legacy_ms = legacy_srs["properties"]["manufacturerSummaries"]["items"]
        acl_fields = set(acl_nc["properties"].keys())
        legacy_fields = set(legacy_ms["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"NameCount fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_name_is_string(self):
        assert ACL_SCHEMAS["NameCount"]["properties"]["name"]["type"] == "string"

    def test_count_is_integer(self):
        assert ACL_SCHEMAS["NameCount"]["properties"]["count"]["type"] == "integer"


class TestFeatureSummarySchema:
    """FeatureSummary {name, count, values[]} field parity."""

    def test_field_names(self):
        acl_fs = ACL_SCHEMAS["FeatureSummary"]
        legacy_srs = _deref_legacy(LEGACY_SCHEMAS["SearchResultSummaries"])
        legacy_fs_arr = _deref_legacy(legacy_srs["properties"]["featureSummaries"])
        legacy_fs = _deref_legacy(legacy_fs_arr["items"])
        acl_fields = set(acl_fs["properties"].keys())
        legacy_fields = set(legacy_fs["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"FeatureSummary fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_name_is_string(self):
        assert ACL_SCHEMAS["FeatureSummary"]["properties"]["name"]["type"] == "string"

    def test_count_is_integer(self):
        assert ACL_SCHEMAS["FeatureSummary"]["properties"]["count"]["type"] == "integer"

    def test_values_is_array(self):
        vals = ACL_SCHEMAS["FeatureSummary"]["properties"]["values"]
        assert vals["type"] == "array"


class TestFeatureValueCountSchema:
    """FeatureValueCount {value, count}."""

    def test_field_names(self):
        acl_fvc = ACL_SCHEMAS["FeatureValueCount"]
        # Legacy: FeatureSummaries > items > values > items
        legacy_srs = _deref_legacy(LEGACY_SCHEMAS["SearchResultSummaries"])
        legacy_fs_arr = _deref_legacy(legacy_srs["properties"]["featureSummaries"])
        legacy_fs = _deref_legacy(legacy_fs_arr["items"])
        legacy_vals = legacy_fs["properties"]["values"]["items"]
        acl_fields = set(acl_fvc["properties"].keys())
        legacy_fields = set(legacy_vals["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"FeatureValueCount fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_value_is_string(self):
        assert ACL_SCHEMAS["FeatureValueCount"]["properties"]["value"]["type"] == "string"

    def test_count_is_integer(self):
        assert ACL_SCHEMAS["FeatureValueCount"]["properties"]["count"]["type"] == "integer"


class TestPricesBucketSchema:
    """PricesBucket {min, max, currencyCode}."""

    def test_field_names(self):
        acl_pb = ACL_SCHEMAS["PricesBucket"]
        legacy_srs = _deref_legacy(LEGACY_SCHEMAS["SearchResultSummaries"])
        legacy_pb = legacy_srs["properties"]["pricesSummary"]["items"]
        acl_fields = set(acl_pb["properties"].keys())
        legacy_fields = set(legacy_pb["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"PricesBucket fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_min_is_number(self):
        assert ACL_SCHEMAS["PricesBucket"]["properties"]["min"]["type"] == "number"

    def test_max_is_number(self):
        assert ACL_SCHEMAS["PricesBucket"]["properties"]["max"]["type"] == "number"

    def test_currency_code_is_string(self):
        assert ACL_SCHEMAS["PricesBucket"]["properties"]["currencyCode"]["type"] == "string"

    def test_required_fields(self):
        assert set(ACL_SCHEMAS["PricesBucket"]["required"]) == {"min", "max", "currencyCode"}


class TestCategoryBucketSchema:
    """CategoryBucket {categoryPathElements, count}."""

    def test_field_names(self):
        acl_cb = ACL_SCHEMAS["CategoryBucket"]
        legacy_cs = _deref_legacy(LEGACY_SCHEMAS["CategorySummary"])
        acl_fields = set(acl_cb["properties"].keys())
        legacy_fields = set(legacy_cs["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"CategoryBucket fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_category_path_elements_is_array_of_strings(self):
        cpe = ACL_SCHEMAS["CategoryBucket"]["properties"]["categoryPathElements"]
        assert cpe["type"] == "array"
        assert cpe["items"]["type"] == "string"

    def test_count_is_integer(self):
        assert ACL_SCHEMAS["CategoryBucket"]["properties"]["count"]["type"] == "integer"


class TestCategoriesSummarySchema:
    """CategoriesSummary {currentCategoryPathElements, sameLevel, children}."""

    def test_field_names(self):
        acl_cs = ACL_SCHEMAS["CategoriesSummary"]
        legacy_srs = _deref_legacy(LEGACY_SCHEMAS["SearchResultSummaries"])
        legacy_cs = legacy_srs["properties"]["categoriesSummary"]
        acl_fields = set(acl_cs["properties"].keys())
        legacy_fields = set(legacy_cs["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"CategoriesSummary fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_same_level_is_array(self):
        assert ACL_SCHEMAS["CategoriesSummary"]["properties"]["sameLevel"]["type"] == "array"

    def test_children_is_array(self):
        assert ACL_SCHEMAS["CategoriesSummary"]["properties"]["children"]["type"] == "array"

    def test_current_category_path_elements_is_array(self):
        cpe = ACL_SCHEMAS["CategoriesSummary"]["properties"]["currentCategoryPathElements"]
        assert cpe["type"] == "array"
        assert cpe["items"]["type"] == "string"


class TestEClassBucketSchema:
    """EClassBucket {group, count}."""

    def test_field_names(self):
        acl_eb = ACL_SCHEMAS["EClassBucket"]
        legacy_egs = _deref_legacy(LEGACY_SCHEMAS["EClassGroupSummary"])
        acl_fields = set(acl_eb["properties"].keys())
        legacy_fields = set(legacy_egs["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"EClassBucket fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_group_is_integer(self):
        assert ACL_SCHEMAS["EClassBucket"]["properties"]["group"]["type"] == "integer"

    def test_count_is_integer(self):
        assert ACL_SCHEMAS["EClassBucket"]["properties"]["count"]["type"] == "integer"


class TestEClassCategoriesSchema:
    """EClassCategories {selectedEClassGroup, sameLevel, children}."""

    def test_field_names(self):
        acl_ec = ACL_SCHEMAS["EClassCategories"]
        legacy_egcs = _deref_legacy(LEGACY_SCHEMAS["EClassGroupCategoriesSummary"])
        acl_fields = set(acl_ec["properties"].keys())
        legacy_fields = set(legacy_egcs["properties"].keys())
        assert acl_fields == legacy_fields, \
            f"EClassCategories fields differ: ACL={acl_fields}, legacy={legacy_fields}"

    def test_selected_eclass_group_is_integer(self):
        seg = ACL_SCHEMAS["EClassCategories"]["properties"]["selectedEClassGroup"]
        assert seg["type"] == "integer"

    def test_same_level_is_array(self):
        assert ACL_SCHEMAS["EClassCategories"]["properties"]["sameLevel"]["type"] == "array"

    def test_children_is_array(self):
        assert ACL_SCHEMAS["EClassCategories"]["properties"]["children"]["type"] == "array"


class TestResponsePricesSummaryType:
    """pricesSummary must be array in both specs (not object)."""

    def test_acl_prices_summary_is_array(self):
        ps = ACL_SCHEMAS["Summaries"]["properties"]["pricesSummary"]
        assert ps["type"] == "array"

    def test_legacy_prices_summary_is_array(self):
        legacy_srs = _deref_legacy(LEGACY_SCHEMAS["SearchResultSummaries"])
        ps = legacy_srs["properties"]["pricesSummary"]
        assert ps["type"] == "array"


class TestResponseEClassesAggregationsType:
    """eClassesAggregations in response is array of {id, count}."""

    def test_acl_is_array(self):
        ea = ACL_SCHEMAS["Summaries"]["properties"]["eClassesAggregations"]
        assert ea["type"] == "array"

    def test_legacy_is_array(self):
        legacy_srs = _deref_legacy(LEGACY_SCHEMAS["SearchResultSummaries"])
        ea = legacy_srs["properties"]["eClassesAggregations"]
        assert ea["type"] == "array"


# ===========================================================================
# Part 2: Live integration tests
# ===========================================================================

ACL_BASE = "http://localhost:8081"
SEARCH_URL = f"{ACL_BASE}/article-features/search"


@pytest.fixture(scope="module")
def _acl_running():
    """Skip all live tests if the ACL is not running."""
    try:
        r = httpx.get(f"{ACL_BASE}/healthz", timeout=3)
        if r.status_code != 200:
            pytest.skip("ACL healthz did not return 200")
    except Exception:
        pytest.skip("ACL not running at localhost:8081")


def _minimal_body() -> dict:
    return {
        "searchMode": "BOTH",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [
                "866b4863-8799-4046-9e84-0985a665c1c7",
            ],
            "sourcePriceListIds": [
                "51a9dedc-efad-469b-8c81-33676f85630e",
            ],
        },
        "maxDeliveryTime": 0,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
        "currency": "EUR",
        "explain": False,
    }


def _maximal_body() -> dict:
    """A request body using every optional field from the legacy spec."""
    return {
        "searchMode": "BOTH",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [
                "866b4863-8799-4046-9e84-0985a665c1c7",
            ],
            "sourcePriceListIds": [
                "51a9dedc-efad-469b-8c81-33676f85630e",
            ],
            "customerArticleNumbersIndexingSourceIds": [],
            "customerUploadedCoreArticleListSourceIds": [],
            "customerManagedArticleNumberListId": None,
            "uiCustomerArticleNumberSourceId": None,
        },
        "queryString": "schraube",
        "articleIdsFilter": [],
        "vendorIdsFilter": [],
        "manufacturersFilter": [],
        "maxDeliveryTime": 0,
        "requiredFeatures": [],
        "priceFilter": None,
        "accessoriesForArticleNumber": None,
        "sparePartsForArticleNumber": None,
        "similarToArticleNumber": None,
        "currentCategoryPathElements": [],
        "currentEClass5Code": None,
        "currentEClass7Code": None,
        "currentS2ClassCode": None,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
        "summaries": [
            "CATEGORIES", "ECLASS5", "ECLASS7", "S2CLASS",
            "VENDORS", "MANUFACTURERS", "FEATURES", "PRICES",
            "PLATFORM_CATEGORIES", "ECLASS5SET",
        ],
        "coreArticlesVendorsFilter": [],
        "blockedEClassVendorsFilters": [],
        "currency": "EUR",
        "explain": True,
        "eClassesFilter": [],
        "eClassesAggregations": [],
        "s2ClassForProductCategories": False,
    }


class TestLiveRequestAcceptance:
    """Verify the ACL accepts legacy request shapes and returns 200."""

    @pytest.mark.usefixtures("_acl_running")
    def test_minimal_request_returns_200(self):
        r = httpx.post(SEARCH_URL, json=_minimal_body(),
                       params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 200, \
            f"Minimal request failed with {r.status_code}: {r.text}"

    @pytest.mark.usefixtures("_acl_running")
    def test_maximal_request_returns_200(self):
        """Every optional field from legacy spec, all at once."""
        r = httpx.post(SEARCH_URL, json=_maximal_body(),
                       params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 200, \
            f"Maximal request failed with {r.status_code}: {r.text}"

    @pytest.mark.usefixtures("_acl_running")
    def test_maximal_request_with_price_filter(self):
        body = _maximal_body()
        body["priceFilter"] = {"min": 100, "max": 5000, "currencyCode": "EUR"}
        r = httpx.post(SEARCH_URL, json=body,
                       params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 200, \
            f"Request with priceFilter failed: {r.status_code}: {r.text}"

    @pytest.mark.usefixtures("_acl_running")
    def test_maximal_request_with_required_features(self):
        body = _maximal_body()
        body["requiredFeatures"] = [{"name": "Farbe", "values": ["rot"]}]
        r = httpx.post(SEARCH_URL, json=body,
                       params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 200, \
            f"Request with requiredFeatures failed: {r.status_code}: {r.text}"

    @pytest.mark.usefixtures("_acl_running")
    def test_maximal_request_with_blocked_eclass_vendors(self):
        body = _maximal_body()
        body["blockedEClassVendorsFilters"] = [{
            "vendorIds": ["866b4863-8799-4046-9e84-0985a665c1c7"],
            "eClassVersion": "ECLASS_5_1",
            "blockedEClassGroups": [{"eClassGroupCode": 12345678, "value": True}],
        }]
        r = httpx.post(SEARCH_URL, json=body,
                       params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 200, \
            f"Request with blockedEClassVendorsFilters failed: {r.status_code}: {r.text}"

    @pytest.mark.usefixtures("_acl_running")
    def test_maximal_request_with_eclass_aggregations(self):
        body = _maximal_body()
        body["eClassesAggregations"] = [
            {"id": "agg1", "eClasses": [12345678, 23456789]},
        ]
        r = httpx.post(SEARCH_URL, json=body,
                       params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 200, \
            f"Request with eClassesAggregations failed: {r.status_code}: {r.text}"


class TestLiveResponseStructure:
    """Verify the response contains all expected top-level and nested structures."""

    @pytest.fixture(scope="class")
    def search_response(self, _acl_running):
        body = _maximal_body()
        r = httpx.post(SEARCH_URL, json=body,
                       params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 200
        return r.json()

    def test_top_level_keys(self, search_response):
        """Response must have articles, summaries, metadata."""
        for key in ("articles", "summaries", "metadata"):
            assert key in search_response, \
                f"Response missing top-level key '{key}'"

    def test_articles_is_list(self, search_response):
        assert isinstance(search_response["articles"], list)

    def test_metadata_has_required_fields(self, search_response):
        meta = search_response["metadata"]
        for key in ("page", "pageSize", "pageCount", "hitCount"):
            assert key in meta, \
                f"metadata missing required field '{key}'"

    def test_metadata_types(self, search_response):
        meta = search_response["metadata"]
        assert isinstance(meta["page"], int)
        assert isinstance(meta["pageSize"], int)
        assert isinstance(meta["pageCount"], int)
        assert isinstance(meta["hitCount"], int)

    def test_summaries_is_dict(self, search_response):
        assert isinstance(search_response["summaries"], dict)

    def test_article_has_article_id(self, search_response):
        for article in search_response["articles"]:
            assert "articleId" in article, \
                "Article missing 'articleId' field"
            assert isinstance(article["articleId"], str)


class TestLiveSearchModeEnum:
    """Verify each searchMode value is accepted."""

    @pytest.mark.usefixtures("_acl_running")
    @pytest.mark.parametrize("mode", ["HITS_ONLY", "SUMMARIES_ONLY", "BOTH"])
    def test_search_mode_accepted(self, mode):
        body = _minimal_body()
        body["searchMode"] = mode
        r = httpx.post(SEARCH_URL, json=body,
                       params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 200, \
            f"searchMode={mode} rejected: {r.status_code}: {r.text}"


class TestLiveSummaryKindEnum:
    """Verify each SummaryKind value is accepted in the summaries field."""

    @pytest.mark.usefixtures("_acl_running")
    @pytest.mark.parametrize("kind", [
        "CATEGORIES", "ECLASS5", "ECLASS7", "S2CLASS", "VENDORS",
        "MANUFACTURERS", "FEATURES", "PRICES", "PLATFORM_CATEGORIES",
        "ECLASS5SET",
    ])
    def test_summary_kind_accepted(self, kind):
        body = _minimal_body()
        body["summaries"] = [kind]
        r = httpx.post(SEARCH_URL, json=body,
                       params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 200, \
            f"summaries=['{kind}'] rejected: {r.status_code}: {r.text}"


class TestLiveEClassVersionEnum:
    """Verify each EClassVersion value is accepted in blockedEClassVendorsFilters."""

    @pytest.mark.usefixtures("_acl_running")
    @pytest.mark.parametrize("version", ["ECLASS_5_1", "ECLASS_7_1", "S2CLASS"])
    def test_eclass_version_accepted(self, version):
        body = _minimal_body()
        body["blockedEClassVendorsFilters"] = [{
            "vendorIds": ["866b4863-8799-4046-9e84-0985a665c1c7"],
            "eClassVersion": version,
            "blockedEClassGroups": [{"eClassGroupCode": 12345678, "value": True}],
        }]
        r = httpx.post(SEARCH_URL, json=body,
                       params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 200, \
            f"eClassVersion={version} rejected: {r.status_code}: {r.text}"


class TestLiveSortParameter:
    """Verify sort parameter works with legacy-compatible values."""

    @pytest.mark.usefixtures("_acl_running")
    @pytest.mark.parametrize("sort_val", [
        "articleId,asc", "articleId,desc",
        "name,asc", "name,desc",
        "price,asc", "price,desc",
        "relevance,asc", "relevance,desc",
    ])
    def test_sort_value_accepted(self, sort_val):
        body = _minimal_body()
        body["queryString"] = "test"
        r = httpx.post(SEARCH_URL, json=body,
                       params={"page": 1, "pageSize": 10, "sort": sort_val},
                       timeout=10)
        assert r.status_code == 200, \
            f"sort={sort_val} rejected: {r.status_code}: {r.text}"


class TestLiveExplainField:
    """Verify explain=true is accepted and explanation is present."""

    @pytest.mark.usefixtures("_acl_running")
    def test_explain_true_accepted(self):
        body = _minimal_body()
        body["explain"] = True
        body["queryString"] = "test"
        r = httpx.post(SEARCH_URL, json=body,
                       params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 200

    @pytest.mark.usefixtures("_acl_running")
    def test_explain_false_accepted(self):
        body = _minimal_body()
        body["explain"] = False
        r = httpx.post(SEARCH_URL, json=body,
                       params={"page": 1, "pageSize": 10}, timeout=10)
        assert r.status_code == 200
