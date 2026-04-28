"""Unit tests for `search-api/filters.py` (F3.1).

Each filter is exercised in isolation, then composition is validated. We
assert string equality on emitted exprs — the F3.5 integration tests
prove the exprs Milvus actually parses and narrows the hit set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SEARCH_API_DIR = Path(__file__).resolve().parent.parent / "search-api"
sys.path.insert(0, str(SEARCH_API_DIR))

from filters import build_milvus_expr, encode_category_path  # noqa: E402
from models import (  # noqa: E402
    BlockedEClassGroup,
    BlockedEClassVendorsFilter,
    EClassVersion,
    FeatureFilter,
    SearchMode,
    SearchRequest,
    SelectedArticleSources,
)


def _req(**overrides) -> SearchRequest:
    base = {
        "search_mode": SearchMode.HITS_ONLY,
        "selected_article_sources": SelectedArticleSources(),
        "currency": "EUR",
    }
    base.update(overrides)
    return SearchRequest(**base)


# ---------- atom-level shape ---------------------------------------------

def test_no_filters_emits_none() -> None:
    assert build_milvus_expr(_req()) is None


def test_vendor_ids() -> None:
    expr = build_milvus_expr(_req(vendor_ids_filter=["v1", "v2"]))
    assert expr == 'vendor_id in ["v1", "v2"]'


def test_article_ids() -> None:
    expr = build_milvus_expr(_req(article_ids_filter=["a:1", "b:2"]))
    assert expr == 'id in ["a:1", "b:2"]'


def test_manufacturers() -> None:
    expr = build_milvus_expr(_req(manufacturers_filter=["Bosch", "Makita"]))
    assert expr == 'manufacturerName in ["Bosch", "Makita"]'


def test_max_delivery_time_zero_is_noop() -> None:
    assert build_milvus_expr(_req(max_delivery_time=0)) is None


def test_max_delivery_time_positive() -> None:
    assert build_milvus_expr(_req(max_delivery_time=5)) == "delivery_time_days_max <= 5"


def test_required_features_or_within_and_across() -> None:
    expr = build_milvus_expr(_req(required_features=[
        FeatureFilter(name="Spannung", values=["18V", "36V"]),
        FeatureFilter(name="Akkutyp", values=["LiIon"]),
    ]))
    # AND across names, OR within values (encoded via array_contains_any).
    assert expr == (
        '(array_contains_any(features, ["Spannung=18V", "Spannung=36V"]))'
        ' and (array_contains_any(features, ["Akkutyp=LiIon"]))'
    )


def test_required_features_skips_empty_values() -> None:
    expr = build_milvus_expr(_req(required_features=[
        FeatureFilter(name="Empty", values=[]),
        FeatureFilter(name="Has", values=["x"]),
    ]))
    assert expr == 'array_contains_any(features, ["Has=x"])'


def test_category_prefix_at_depth_2() -> None:
    expr = build_milvus_expr(_req(current_category_path_elements=["Werkzeug", "Akku"]))
    assert expr == 'array_contains(category_l2, "Werkzeug¦Akku")'


def test_category_prefix_escapes_path_separator() -> None:
    # Element containing the separator (¦) gets it replaced with the
    # escape (|) per `CategoryPath.asStringPath`.
    expr = build_milvus_expr(_req(
        current_category_path_elements=["Werkzeug", "Hand¦Maschine"]
    ))
    assert expr == 'array_contains(category_l2, "Werkzeug¦Hand|Maschine")'


def test_category_prefix_depth_out_of_range_is_noop() -> None:
    too_deep = ["a", "b", "c", "d", "e", "f"]  # depth 6, schema only has l1..l5
    assert build_milvus_expr(_req(current_category_path_elements=too_deep)) is None


def test_eclass_codes_compose_with_and() -> None:
    expr = build_milvus_expr(_req(
        current_eclass5_code=23172001,
        current_eclass7_code=23172090,
        current_s2class_code=1001,
    ))
    # eclass{5,7}_code / s2class_code are ARRAY<INT32> hierarchies; a
    # parent or leaf code matches via array_contains.
    assert expr == (
        "(array_contains(eclass5_code, 23172001))"
        " and (array_contains(eclass7_code, 23172090))"
        " and (array_contains(s2class_code, 1001))"
    )


def test_eclasses_filter_default_eclass5() -> None:
    assert build_milvus_expr(_req(eclasses_filter=[1001, 1002])) == (
        "array_contains_any(eclass5_code, [1001, 1002])"
    )


def test_eclasses_filter_s2class_when_flag_set() -> None:
    expr = build_milvus_expr(_req(
        eclasses_filter=[5042, 5043],
        s2class_for_product_categories=True,
    ))
    assert expr == "array_contains_any(s2class_code, [5042, 5043])"


def test_closed_marketplace_only_without_cv_matches_nothing() -> None:
    """Per legacy `OfferFilterBuilder`, `closedMarketplaceOnly=true` emits a
    `terms` query against `closedCatalogVersionIds`. An empty list matches
    nothing in ES; we replicate via an always-false expr."""
    assert build_milvus_expr(_req(closed_marketplace_only=True)) == 'id == ""'


def test_closed_marketplace_only_intersects_closed_cv() -> None:
    expr = build_milvus_expr(_req(
        closed_marketplace_only=True,
        selected_article_sources=SelectedArticleSources(
            closedCatalogVersionIds=["c-1", "c-2"],
        ),
    ))
    assert expr == 'array_contains_any(catalog_version_ids, ["c-1", "c-2"])'


def test_closed_catalog_versions_alone_is_noop() -> None:
    """Without `closedMarketplaceOnly=True`, `closedCatalogVersionIds` is
    metadata for the core-sortiment logic only. Legacy never intersects on
    it standalone (`OfferFilterBuilder` switches lists by the flag)."""
    assert build_milvus_expr(_req(
        selected_article_sources=SelectedArticleSources(
            closedCatalogVersionIds=["c-1", "c-2"],
        ),
    )) is None


def test_relationships_compose_with_and() -> None:
    expr = build_milvus_expr(_req(
        accessories_for_article_number="ACC-001",
        spare_parts_for_article_number="BASE-A",
        similar_to_article_number="SIM-X",
    ))
    assert expr == (
        '(array_contains(relationship_accessory_for, "ACC-001"))'
        ' and (array_contains(relationship_spare_part_for, "BASE-A"))'
        ' and (array_contains(relationship_similar_to, "SIM-X"))'
    )


def test_core_sortiment_uses_closed_catalog_version_ids() -> None:
    # `closedCatalogVersionIds` drives the core-sortiment source set when
    # `coreSortimentOnly=true`; ftsearch does not impose a standalone
    # CV intersection on it (see _closed_marketplace docstring).
    expr = build_milvus_expr(_req(
        core_sortiment_only=True,
        selected_article_sources=SelectedArticleSources(
            closedCatalogVersionIds=["c-1", "c-2"],
        ),
    ))
    assert expr == 'array_contains_any(core_marker_enabled_sources, ["c-1", "c-2"])'


def test_core_sortiment_with_customer_uploaded() -> None:
    expr = build_milvus_expr(_req(
        core_sortiment_only=True,
        selected_article_sources=SelectedArticleSources(
            closedCatalogVersionIds=["c-1"],
            customerUploadedCoreArticleListSourceIds=["u-1"],
        ),
    ))
    assert expr == (
        '(array_contains_any(core_marker_enabled_sources, ["u-1"]))'
        ' or ((array_contains_any(core_marker_enabled_sources, ["c-1"]))'
        ' and (not array_contains_any(core_marker_disabled_sources, ["u-1"])))'
    )


def test_core_sortiment_only_uploaded() -> None:
    expr = build_milvus_expr(_req(
        core_sortiment_only=True,
        selected_article_sources=SelectedArticleSources(
            customerUploadedCoreArticleListSourceIds=["u-1"],
        ),
    ))
    assert expr == 'array_contains_any(core_marker_enabled_sources, ["u-1"])'


def test_core_sortiment_off_is_noop() -> None:
    assert build_milvus_expr(_req(core_sortiment_only=False)) is None


def test_core_sortiment_no_sources_is_noop() -> None:
    # An empty source context shouldn't mass-exclude rows; treat as no-op.
    assert build_milvus_expr(_req(core_sortiment_only=True)) is None


def test_core_articles_vendors_filter() -> None:
    expr = build_milvus_expr(_req(
        core_articles_vendors_filter=["v-strict"],
        selected_article_sources=SelectedArticleSources(
            closedCatalogVersionIds=["c-1"],
        ),
    ))
    assert expr == (
        '(vendor_id not in ["v-strict"])'
        ' or (array_contains_any(core_marker_enabled_sources, ["c-1"]))'
    )


def test_blocked_eclass_vendors_basic() -> None:
    expr = build_milvus_expr(_req(blocked_eclass_vendors_filters=[
        BlockedEClassVendorsFilter(
            vendorIds=["v-1"],
            eClassVersion=EClassVersion.ECLASS_5_1,
            blockedEClassGroups=[BlockedEClassGroup(eClassGroupCode=23172001, value=True)],
        ),
    ]))
    assert expr == (
        '(vendor_id not in ["v-1"]) or '
        '(not (array_contains_any(eclass5_code, [23172001])))'
    )


def test_blocked_eclass_vendors_with_exception() -> None:
    expr = build_milvus_expr(_req(blocked_eclass_vendors_filters=[
        BlockedEClassVendorsFilter(
            vendorIds=["v-1"],
            eClassVersion=EClassVersion.ECLASS_5_1,
            blockedEClassGroups=[
                BlockedEClassGroup(eClassGroupCode=23170000, value=True),
                BlockedEClassGroup(eClassGroupCode=23172001, value=False),
            ],
        ),
    ]))
    # block-true minus block-false: blocked iff hierarchy contains [23170000]
    # AND does not contain [23172001].
    assert expr == (
        '(vendor_id not in ["v-1"]) or '
        '(not ((array_contains_any(eclass5_code, [23170000]))'
        ' and (not array_contains_any(eclass5_code, [23172001]))))'
    )


def test_blocked_eclass_vendors_no_vendors_filter_applies_globally() -> None:
    expr = build_milvus_expr(_req(blocked_eclass_vendors_filters=[
        BlockedEClassVendorsFilter(
            eClassVersion=EClassVersion.S2CLASS,
            blockedEClassGroups=[BlockedEClassGroup(eClassGroupCode=5000, value=True)],
        ),
    ]))
    assert expr == "not (array_contains_any(s2class_code, [5000]))"


def test_blocked_eclass_vendors_skip_entries_with_no_block_true() -> None:
    expr = build_milvus_expr(_req(blocked_eclass_vendors_filters=[
        BlockedEClassVendorsFilter(
            vendorIds=["v-1"],
            eClassVersion=EClassVersion.ECLASS_5_1,
            blockedEClassGroups=[BlockedEClassGroup(eClassGroupCode=1, value=False)],
        ),
    ]))
    assert expr is None


# ---------- composition ---------------------------------------------------

def test_two_filters_and_composed() -> None:
    expr = build_milvus_expr(_req(
        vendor_ids_filter=["v-1"],
        max_delivery_time=3,
    ))
    assert expr == '(vendor_id in ["v-1"]) and (delivery_time_days_max <= 3)'


def test_quote_escaping_for_quotes_and_backslashes() -> None:
    expr = build_milvus_expr(_req(manufacturers_filter=['He said "hi"', "back\\slash"]))
    assert expr == 'manufacturerName in ["He said \\"hi\\"", "back\\\\slash"]'


# ---------- encoding helpers ---------------------------------------------

def test_encode_category_path_no_separator_in_elements() -> None:
    assert encode_category_path(["A", "B", "C"]) == "A¦B¦C"


def test_encode_category_path_escapes_separator_in_element() -> None:
    assert encode_category_path(["Hand¦Maschine", "Akku"]) == "Hand|Maschine¦Akku"
