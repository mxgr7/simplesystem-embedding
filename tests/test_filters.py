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

from filters import (  # noqa: E402
    build_article_expr,
    build_milvus_expr,
    build_offer_expr,
    encode_category_path,
    has_per_vendor_blocked_eclass,
)
from models import (  # noqa: E402
    BlockedEClassGroup,
    BlockedEClassVendorsFilter,
    EClassVersion,
    FeatureFilter,
    PriceFilter,
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


# ---------- F9 split: build_article_expr / build_offer_expr --------------
#
# Article-side fields live on `articles_v{N}`: name, manufacturerName,
# category_l1..l5, eclass{5,7}_code, s2class_code. Everything else is
# offer-side.

def test_split_no_filters_emits_none_on_both_sides() -> None:
    req = _req()
    assert build_article_expr(req) is None
    assert build_offer_expr(req) is None


def test_split_vendor_is_offer_side() -> None:
    req = _req(vendor_ids_filter=["v1"])
    assert build_article_expr(req) is None
    assert build_offer_expr(req) == 'vendor_id in ["v1"]'


def test_split_article_ids_is_offer_side() -> None:
    """`articleIdsFilter` is the legacy offer PK — stays on offers_v{N}.
    Spec semantic ("user asks for these specific offers") is preserved
    without a hash-resolution round-trip."""
    req = _req(article_ids_filter=["a:1", "b:2"])
    assert build_article_expr(req) is None
    assert build_offer_expr(req) == 'id in ["a:1", "b:2"]'


def test_split_manufacturer_is_article_side() -> None:
    req = _req(manufacturers_filter=["Bosch"])
    assert build_article_expr(req) == 'manufacturerName in ["Bosch"]'
    assert build_offer_expr(req) is None


def test_split_category_is_article_side() -> None:
    req = _req(current_category_path_elements=["Werkzeug", "Akku"])
    assert build_article_expr(req) == 'array_contains(category_l2, "Werkzeug¦Akku")'
    assert build_offer_expr(req) is None


def test_split_eclass_codes_are_article_side() -> None:
    req = _req(current_eclass5_code=23172001, current_eclass7_code=23172090)
    article = build_article_expr(req)
    assert article is not None
    assert "eclass5_code" in article
    assert "eclass7_code" in article
    assert build_offer_expr(req) is None


def test_split_eclasses_filter_is_article_side() -> None:
    req = _req(eclasses_filter=[1001, 1002])
    assert build_article_expr(req) == "array_contains_any(eclass5_code, [1001, 1002])"
    assert build_offer_expr(req) is None


def test_split_eclasses_filter_with_s2class_flag_is_article_side() -> None:
    req = _req(eclasses_filter=[5042], s2class_for_product_categories=True)
    assert build_article_expr(req) == "array_contains_any(s2class_code, [5042])"
    assert build_offer_expr(req) is None


def test_split_required_features_is_offer_side() -> None:
    req = _req(required_features=[FeatureFilter(name="Spannung", values=["18V"])])
    assert build_article_expr(req) is None
    assert build_offer_expr(req) == 'array_contains_any(features, ["Spannung=18V"])'


def test_split_delivery_time_is_offer_side() -> None:
    req = _req(max_delivery_time=5)
    assert build_article_expr(req) is None
    assert build_offer_expr(req) == "delivery_time_days_max <= 5"


def test_split_closed_marketplace_is_offer_side() -> None:
    req = _req(
        closed_marketplace_only=True,
        selected_article_sources=SelectedArticleSources(
            closedCatalogVersionIds=["c-1"],
        ),
    )
    assert build_article_expr(req) is None
    assert build_offer_expr(req) == 'array_contains_any(catalog_version_ids, ["c-1"])'


def test_split_relationships_are_offer_side() -> None:
    req = _req(
        accessories_for_article_number="ACC-001",
        spare_parts_for_article_number="BASE-A",
    )
    assert build_article_expr(req) is None
    offer = build_offer_expr(req)
    assert offer is not None
    assert "relationship_accessory_for" in offer
    assert "relationship_spare_part_for" in offer


def test_split_core_sortiment_is_offer_side() -> None:
    req = _req(
        core_sortiment_only=True,
        selected_article_sources=SelectedArticleSources(
            closedCatalogVersionIds=["c-1"],
        ),
    )
    assert build_article_expr(req) is None
    assert build_offer_expr(req) == 'array_contains_any(core_marker_enabled_sources, ["c-1"])'


def test_split_core_articles_vendors_is_offer_side() -> None:
    req = _req(
        core_articles_vendors_filter=["v-strict"],
        selected_article_sources=SelectedArticleSources(
            closedCatalogVersionIds=["c-1"],
        ),
    )
    assert build_article_expr(req) is None
    offer = build_offer_expr(req)
    assert offer is not None
    assert "vendor_id not in" in offer
    assert "core_marker_enabled_sources" in offer


def test_split_blocked_eclass_global_is_article_side() -> None:
    """No `vendorIds` → applies to every article, eclass-only. Lands on
    the article side as `not array_contains_any(...)` with no vendor
    correlation."""
    req = _req(blocked_eclass_vendors_filters=[
        BlockedEClassVendorsFilter(
            eClassVersion=EClassVersion.S2CLASS,
            blockedEClassGroups=[BlockedEClassGroup(eClassGroupCode=5000, value=True)],
        ),
    ])
    assert build_article_expr(req) == "not (array_contains_any(s2class_code, [5000]))"
    assert build_offer_expr(req) is None


def test_split_blocked_eclass_per_vendor_drops_from_expr() -> None:
    """Per-vendor entries correlate offer.vendor with article.eclass —
    not expressible as a single Milvus expr post-F9. Splitter omits;
    routing.py applies as a Python post-pass after the join."""
    req = _req(blocked_eclass_vendors_filters=[
        BlockedEClassVendorsFilter(
            vendorIds=["v-1"],
            eClassVersion=EClassVersion.ECLASS_5_1,
            blockedEClassGroups=[BlockedEClassGroup(eClassGroupCode=23172001, value=True)],
        ),
    ])
    assert build_article_expr(req) is None
    assert build_offer_expr(req) is None
    assert has_per_vendor_blocked_eclass(req) is True


def test_split_blocked_eclass_global_and_per_vendor_separate() -> None:
    """Global entry pushes down on article side; per-vendor entry stays
    in the SearchRequest for the routing post-pass."""
    req = _req(blocked_eclass_vendors_filters=[
        BlockedEClassVendorsFilter(
            eClassVersion=EClassVersion.ECLASS_5_1,
            blockedEClassGroups=[BlockedEClassGroup(eClassGroupCode=1000, value=True)],
        ),
        BlockedEClassVendorsFilter(
            vendorIds=["v-1"],
            eClassVersion=EClassVersion.ECLASS_5_1,
            blockedEClassGroups=[BlockedEClassGroup(eClassGroupCode=2000, value=True)],
        ),
    ])
    article = build_article_expr(req)
    assert article is not None
    assert "1000" in article
    assert "2000" not in article  # per-vendor entry omitted
    assert has_per_vendor_blocked_eclass(req) is True


def test_split_composes_atoms_with_and_per_side() -> None:
    """Cross-scope request: vendor (offer) + manufacturer (article) +
    eclass (article) → composed independently on each side."""
    req = _req(
        vendor_ids_filter=["v1"],
        manufacturers_filter=["Bosch"],
        current_eclass5_code=23172001,
    )
    article = build_article_expr(req)
    offer = build_offer_expr(req)
    assert article == (
        '(manufacturerName in ["Bosch"])'
        ' and (array_contains(eclass5_code, 23172001))'
    )
    assert offer == 'vendor_id in ["v1"]'


def test_split_legacy_build_milvus_expr_unchanged() -> None:
    """`build_milvus_expr` (legacy single-collection path) preserves the
    full AND-composition across both scopes — used when
    `USE_DEDUP_TOPOLOGY=false`."""
    req = _req(
        vendor_ids_filter=["v1"],
        manufacturers_filter=["Bosch"],
    )
    expr = build_milvus_expr(req)
    assert expr == '(vendor_id in ["v1"]) and (manufacturerName in ["Bosch"])'


def test_has_per_vendor_blocked_eclass_false_when_global() -> None:
    req = _req(blocked_eclass_vendors_filters=[
        BlockedEClassVendorsFilter(
            eClassVersion=EClassVersion.ECLASS_5_1,
            blockedEClassGroups=[BlockedEClassGroup(eClassGroupCode=1, value=True)],
        ),
    ])
    assert has_per_vendor_blocked_eclass(req) is False


def test_has_per_vendor_blocked_eclass_false_when_only_block_false() -> None:
    """Entry with vendorIds but no `value=true` codes is functionally a
    no-op — `_blocked_eclass_vendors` skips it. Detection mirrors that."""
    req = _req(blocked_eclass_vendors_filters=[
        BlockedEClassVendorsFilter(
            vendorIds=["v-1"],
            eClassVersion=EClassVersion.ECLASS_5_1,
            blockedEClassGroups=[BlockedEClassGroup(eClassGroupCode=1, value=False)],
        ),
    ])
    assert has_per_vendor_blocked_eclass(req) is False


# ---------- F8 price-scope pre-filter ------------------------------------
#
# Both atoms gate on `priceFilter` being set — preserves the F3
# post-pass-only path's behaviour when there's no priceFilter (offers
# without a resolvable price are kept with `resolved_price=None`,
# representative-pick still proceeds). When priceFilter IS set, the
# clauses are conservative supersets of the precise post-pass.

def test_price_band_skipped_without_price_filter() -> None:
    """No priceFilter → no band clause (preserves F3 parity)."""
    assert build_offer_expr(_req()) is None


def test_price_band_emits_both_bounds_when_present() -> None:
    """`{ccy}_price_min <= decoded_max AND {ccy}_price_max >= decoded_min`."""
    req = _req(price_filter=PriceFilter(min=1525, max=10000, currencyCode="EUR"))
    # 1525 minor → 15.25; 10000 minor → 100. Top-level currency=EUR → eur_*.
    assert build_offer_expr(req) == (
        "(eur_price_min <= 100) and (eur_price_max >= 15.25)"
    )


def test_price_band_min_only_emits_max_column_only() -> None:
    """Only `min` set → only `{ccy}_price_max >= decoded_min` (the `_max`
    column captures the upper end of the offer's range; if it's below
    the requested floor the offer is out)."""
    req = _req(price_filter=PriceFilter(min=2000, currencyCode="EUR"))
    assert build_offer_expr(req) == "eur_price_max >= 20"


def test_price_band_max_only_emits_min_column_only() -> None:
    """Only `max` set → only `{ccy}_price_min <= decoded_max`."""
    req = _req(price_filter=PriceFilter(max=5000, currencyCode="EUR"))
    assert build_offer_expr(req) == "eur_price_min <= 50"


def test_price_band_uses_top_level_currency_lowercased() -> None:
    """Spec §3 currency two-roles: top-level `currency` selects the
    column, NOT `priceFilter.currencyCode` (which only decodes bounds)."""
    req = _req(
        currency="CHF",
        price_filter=PriceFilter(min=1000, max=2000, currencyCode="CHF"),
    )
    assert build_offer_expr(req) == (
        "(chf_price_min <= 20) and (chf_price_max >= 10)"
    )


def test_price_band_decodes_via_priceFilter_currency_code() -> None:
    """Spec §3: `priceFilter.currencyCode` drives bound decoding via
    ISO-4217 fraction digits. JPY has 0 digits → minor units == decimal."""
    req = _req(
        currency="EUR",
        price_filter=PriceFilter(min=1500, max=10000, currencyCode="JPY"),
    )
    # Decoded via JPY (0 digits) → 1500/10000 verbatim.
    assert build_offer_expr(req) == (
        "(eur_price_min <= 10000) and (eur_price_max >= 1500)"
    )


def test_price_band_skipped_when_currency_not_in_catalog() -> None:
    """Out-of-catalogue top-level currency (e.g., USD) has no
    `usd_price_*` column on offers_v{N} — clause is omitted, post-pass
    handles the precise check alone."""
    req = _req(
        currency="USD",
        price_filter=PriceFilter(min=1500, max=10000, currencyCode="USD"),
    )
    assert build_offer_expr(req) is None


def test_price_list_scope_skipped_without_price_filter() -> None:
    """sourcePriceListIds alone (no priceFilter) → no clause. Routing.py's
    resolver only drops on price-list mismatch when price_filter_active —
    emitting this clause without priceFilter would break parity."""
    req = _req(selected_article_sources=SelectedArticleSources(
        sourcePriceListIds=["pl-1", "pl-2"],
    ))
    assert build_offer_expr(req) is None


def test_price_list_scope_skipped_when_source_ids_empty() -> None:
    """priceFilter alone, empty sourcePriceListIds → no list clause
    (band clause may still emit)."""
    req = _req(price_filter=PriceFilter(min=1000, currencyCode="EUR"))
    expr = build_offer_expr(req)
    assert expr is not None
    assert "price_list_ids" not in expr


def test_price_list_scope_emits_when_priceFilter_and_ids_set() -> None:
    """Both priceFilter and sourcePriceListIds set → list clause +
    band clause AND-composed."""
    req = _req(
        selected_article_sources=SelectedArticleSources(
            sourcePriceListIds=["pl-1", "pl-2"],
        ),
        price_filter=PriceFilter(min=1000, max=5000, currencyCode="EUR"),
    )
    expr = build_offer_expr(req)
    assert expr is not None
    assert 'array_contains_any(price_list_ids, ["pl-1", "pl-2"])' in expr
    assert "eur_price_min <= 50" in expr
    assert "eur_price_max >= 10" in expr


def test_price_clauses_compose_with_other_offer_atoms() -> None:
    """F8 clauses AND-compose with vendor / delivery / etc."""
    req = _req(
        vendor_ids_filter=["v1"],
        max_delivery_time=5,
        selected_article_sources=SelectedArticleSources(
            sourcePriceListIds=["pl-1"],
        ),
        price_filter=PriceFilter(max=10000, currencyCode="EUR"),
    )
    expr = build_offer_expr(req)
    assert expr is not None
    assert 'vendor_id in ["v1"]' in expr
    assert "delivery_time_days_max <= 5" in expr
    assert 'array_contains_any(price_list_ids, ["pl-1"])' in expr
    assert "eur_price_min <= 100" in expr


def test_price_clauses_are_offer_side_only() -> None:
    """F8 envelope columns live on offers_v{N} — never article side."""
    req = _req(
        selected_article_sources=SelectedArticleSources(
            sourcePriceListIds=["pl-1"],
        ),
        price_filter=PriceFilter(min=1000, max=5000, currencyCode="EUR"),
    )
    assert build_article_expr(req) is None


def test_price_band_skipped_when_no_bounds() -> None:
    """priceFilter present but min and max both None → no band clause
    (no actual bound to apply)."""
    req = _req(price_filter=PriceFilter(currencyCode="EUR"))
    expr = build_offer_expr(req)
    # No sourcePriceListIds either → both atoms skip → None.
    assert expr is None
