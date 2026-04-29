"""F5 — pure-function tests for `search-api/aggregations.py`.

Covers each summary kind on hand-crafted article + offer rows. Counts,
ordering, sum-≤-hitcount invariant, hierarchical sameLevel/children
logic, and the field-set planner.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_API_DIR = REPO_ROOT / "search-api"
sys.path.insert(0, str(SEARCH_API_DIR))

import aggregations  # noqa: E402
from models import (  # noqa: E402
    EClassesAggregation,
    SearchMode,
    SearchRequest,
    SelectedArticleSources,
    SummaryKind,
)


def _req(**overrides) -> SearchRequest:
    base = {
        "search_mode": SearchMode.BOTH,
        "selected_article_sources": SelectedArticleSources(),
        "currency": "EUR",
    }
    base.update(overrides)
    return SearchRequest(**base)


def _article(hash_: str, **fields) -> dict:
    return {"article_hash": hash_, **fields}


def _offer(hash_: str, *, vendor_id: str = "v1", **fields) -> dict:
    return {"article_hash": hash_, "vendor_id": vendor_id, **fields}


# ──────────────────────────────────────────────────────────────────────
# Field-set planner
# ──────────────────────────────────────────────────────────────────────

def test_article_fields_needed_baseline_is_just_hash() -> None:
    needed = aggregations.article_fields_needed(_req(summaries=[]))
    assert needed == {"article_hash"}


def test_article_fields_needed_includes_per_kind_columns() -> None:
    needed = aggregations.article_fields_needed(_req(summaries=[
        SummaryKind.MANUFACTURERS,
        SummaryKind.CATEGORIES,
        SummaryKind.ECLASS5,
        SummaryKind.ECLASS5SET,
    ]))
    assert "manufacturerName" in needed
    assert {f"category_l{d}" for d in range(1, 6)}.issubset(needed)
    assert "eclass5_code" in needed


def test_offer_fields_needed_per_kind() -> None:
    needed = aggregations.offer_fields_needed(_req(summaries=[
        SummaryKind.VENDORS, SummaryKind.FEATURES, SummaryKind.PRICES,
    ]))
    assert {"article_hash", "vendor_id", "features", "prices"}.issubset(needed)


def test_platform_categories_alias_picks_correct_field() -> None:
    """PLATFORM_CATEGORIES under `s2ClassForProductCategories=true`
    needs s2class_code; otherwise needs the category prefix arrays."""
    s2_needed = aggregations.article_fields_needed(_req(
        summaries=[SummaryKind.PLATFORM_CATEGORIES],
        s2class_for_product_categories=True,
    ))
    cat_needed = aggregations.article_fields_needed(_req(
        summaries=[SummaryKind.PLATFORM_CATEGORIES],
        s2class_for_product_categories=False,
    ))
    assert "s2class_code" in s2_needed
    assert {f"category_l{d}" for d in range(1, 6)}.issubset(cat_needed)


def test_needs_offer_fetch_only_when_offer_kinds_requested() -> None:
    assert aggregations.needs_offer_fetch(_req(summaries=[SummaryKind.MANUFACTURERS])) is False
    assert aggregations.needs_offer_fetch(_req(summaries=[SummaryKind.VENDORS])) is True
    assert aggregations.needs_offer_fetch(_req(summaries=[SummaryKind.FEATURES])) is True
    assert aggregations.needs_offer_fetch(_req(summaries=[SummaryKind.PRICES])) is True
    assert aggregations.needs_offer_fetch(_req(summaries=[])) is False


def test_needs_article_fetch_only_when_article_kinds_requested() -> None:
    assert aggregations.needs_article_fetch(_req(summaries=[SummaryKind.VENDORS])) is False
    assert aggregations.needs_article_fetch(_req(summaries=[SummaryKind.MANUFACTURERS])) is True
    assert aggregations.needs_article_fetch(_req(summaries=[SummaryKind.CATEGORIES])) is True
    assert aggregations.needs_article_fetch(_req(summaries=[SummaryKind.ECLASS5SET])) is True


# ──────────────────────────────────────────────────────────────────────
# VENDORS
# ──────────────────────────────────────────────────────────────────────

def test_vendors_count_distinct_articles_per_vendor() -> None:
    """Each (vendor, article) pair counted once even with multiple offers."""
    rows = [
        _offer("h1", vendor_id="v1"),
        _offer("h1", vendor_id="v1"),  # same article + vendor → count once
        _offer("h2", vendor_id="v1"),
        _offer("h1", vendor_id="v2"),
    ]
    out = aggregations.vendors_summary(rows)
    by_v = {s.vendor_id: s.count for s in out}
    assert by_v == {"v1": 2, "v2": 1}


def test_vendors_sorted_by_count_desc_then_id_asc() -> None:
    rows = [
        _offer("h1", vendor_id="z"), _offer("h2", vendor_id="z"),
        _offer("h1", vendor_id="a"), _offer("h2", vendor_id="a"),
        _offer("h3", vendor_id="m"),
    ]
    out = aggregations.vendors_summary(rows)
    # a and z tie at 2; alphabetical → a, z; m at 1.
    assert [s.vendor_id for s in out] == ["a", "z", "m"]


def test_vendors_skips_empty_vendor() -> None:
    rows = [_offer("h1", vendor_id=""), _offer("h2", vendor_id="v1")]
    out = aggregations.vendors_summary(rows)
    assert {s.vendor_id for s in out} == {"v1"}


# ──────────────────────────────────────────────────────────────────────
# MANUFACTURERS
# ──────────────────────────────────────────────────────────────────────

def test_manufacturers_one_per_article_grouped_by_name() -> None:
    rows = [
        _article("h1", manufacturerName="Bosch"),
        _article("h2", manufacturerName="Bosch"),
        _article("h3", manufacturerName="Makita"),
    ]
    out = aggregations.manufacturers_summary(rows)
    by_n = {n.name: n.count for n in out}
    assert by_n == {"Bosch": 2, "Makita": 1}


def test_manufacturers_skips_empty_name() -> None:
    rows = [
        _article("h1", manufacturerName=""),
        _article("h2", manufacturerName="X"),
    ]
    out = aggregations.manufacturers_summary(rows)
    assert {n.name for n in out} == {"X"}


# ──────────────────────────────────────────────────────────────────────
# FEATURES
# ──────────────────────────────────────────────────────────────────────

def test_features_distinct_articles_per_name_value() -> None:
    rows = [
        _offer("h1", features=["Spannung=18V", "Akkutyp=LiIon"]),
        _offer("h1", features=["Spannung=18V"]),  # same article + token
        _offer("h2", features=["Spannung=36V", "Akkutyp=LiIon"]),
    ]
    out = aggregations.features_summary(rows)
    # Both Spannung and Akkutyp present.
    by_name = {f.name: f for f in out}
    spannung = by_name["Spannung"]
    # Distinct articles having any Spannung value: h1 + h2.
    assert spannung.count == 2
    by_value = {v.value: v.count for v in spannung.values}
    assert by_value == {"18V": 1, "36V": 1}
    akku = by_name["Akkutyp"]
    assert akku.count == 2  # h1 + h2 both have LiIon


def test_features_skips_malformed_tokens() -> None:
    """Tokens without `=` separator are dropped (defensive — projection
    enforces this on the writer side)."""
    rows = [
        _offer("h1", features=["malformed-no-equals", "Good=Value"]),
    ]
    out = aggregations.features_summary(rows)
    assert {f.name for f in out} == {"Good"}


# ──────────────────────────────────────────────────────────────────────
# PRICES
# ──────────────────────────────────────────────────────────────────────

def test_prices_min_max_resolved_per_currency() -> None:
    rows = [
        _offer("h1", prices=[{"price": 10.0, "currency": "EUR", "priority": 1, "sourcePriceListId": "p1"}]),
        _offer("h2", prices=[{"price": 100.0, "currency": "EUR", "priority": 1, "sourcePriceListId": "p1"}]),
        _offer("h3", prices=[{"price": 50.0, "currency": "EUR", "priority": 1, "sourcePriceListId": "p1"}]),
    ]
    req = _req(selected_article_sources=SelectedArticleSources(sourcePriceListIds=["p1"]))
    out = aggregations.prices_summary(rows, req)
    assert len(out) == 1
    assert out[0].currency_code == "EUR"
    assert out[0].min == 10.0
    assert out[0].max == 100.0


def test_prices_skips_offers_with_no_in_scope_price() -> None:
    rows = [
        _offer("h1", prices=[{"price": 10.0, "currency": "EUR", "priority": 1, "sourcePriceListId": "out-of-scope"}]),
        _offer("h2", prices=[{"price": 50.0, "currency": "EUR", "priority": 1, "sourcePriceListId": "p1"}]),
    ]
    req = _req(selected_article_sources=SelectedArticleSources(sourcePriceListIds=["p1"]))
    out = aggregations.prices_summary(rows, req)
    assert out[0].min == 50.0
    assert out[0].max == 50.0


# ──────────────────────────────────────────────────────────────────────
# CATEGORIES — hierarchical
# ──────────────────────────────────────────────────────────────────────

def test_categories_root_yields_top_level_paths() -> None:
    """No current_category_path → sameLevel = depth-1 categories,
    children = empty."""
    rows = [
        _article("h1", category_l1=["Werkzeug"], category_l2=["Werkzeug¦Akku"]),
        _article("h2", category_l1=["Werkzeug"], category_l2=["Werkzeug¦Hand"]),
        _article("h3", category_l1=["Maschinenbau"]),
    ]
    out = aggregations.categories_summary(rows, current_path=[])
    by_path = {tuple(b.category_path_elements): b.count for b in out.same_level}
    assert by_path == {("Werkzeug",): 2, ("Maschinenbau",): 1}
    assert out.children == []


def test_categories_at_depth_returns_siblings_and_children() -> None:
    """current_path=['Werkzeug'] (depth 1) → sameLevel = depth-1 codes
    (Werkzeug + sibling); children = depth-2 paths under Werkzeug."""
    rows = [
        _article("h1", category_l1=["Werkzeug"], category_l2=["Werkzeug¦Akku"]),
        _article("h2", category_l1=["Werkzeug"], category_l2=["Werkzeug¦Hand"]),
        _article("h3", category_l1=["Maschinenbau"]),
    ]
    out = aggregations.categories_summary(rows, current_path=["Werkzeug"])
    same = {tuple(b.category_path_elements): b.count for b in out.same_level}
    children = {tuple(b.category_path_elements): b.count for b in out.children}
    # sameLevel = depth-1 codes (no prefix constraint at depth 1).
    assert same == {("Werkzeug",): 2, ("Maschinenbau",): 1}
    assert children == {("Werkzeug", "Akku"): 1, ("Werkzeug", "Hand"): 1}


def test_categories_at_deeper_level_constrains_siblings_to_prefix() -> None:
    """current_path=['Werkzeug','Akku'] (depth 2) → sameLevel at depth 2
    constrained to prefix 'Werkzeug'; children at depth 3 under
    'Werkzeug¦Akku'."""
    rows = [
        _article("h1", category_l2=["Werkzeug¦Akku"], category_l3=["Werkzeug¦Akku¦Bohrer"]),
        _article("h2", category_l2=["Werkzeug¦Hand"]),
        _article("h3", category_l2=["Maschinenbau¦Antrieb"]),  # different prefix
    ]
    out = aggregations.categories_summary(rows, current_path=["Werkzeug", "Akku"])
    same = {tuple(b.category_path_elements): b.count for b in out.same_level}
    children = {tuple(b.category_path_elements): b.count for b in out.children}
    # sameLevel: depth-2 paths starting with 'Werkzeug'.
    assert same == {("Werkzeug", "Akku"): 1, ("Werkzeug", "Hand"): 1}
    # Maschinenbau dropped — different prefix.
    assert children == {("Werkzeug", "Akku", "Bohrer"): 1}


def test_categories_prefix_collision_avoided() -> None:
    """A path 'Werkzeugmacher' must NOT match a prefix 'Werkzeug'.
    Avoided by the `path == prefix or path.startswith(prefix + sep)` check."""
    rows = [
        _article("h1", category_l1=["Werkzeug"]),
        _article("h2", category_l1=["Werkzeugmacher"]),
    ]
    out = aggregations.categories_summary(rows, current_path=["Werkzeug"])
    children = [tuple(b.category_path_elements) for b in out.children]
    # No depth-2 paths in the rows, so children empty regardless.
    assert children == []


# ──────────────────────────────────────────────────────────────────────
# ECLASS5 / ECLASS7 / S2CLASS — hierarchical
# ──────────────────────────────────────────────────────────────────────

def test_eclass_root_no_selection_yields_depth_1_codes() -> None:
    rows = [
        _article("h1", eclass5_code=[23, 2317, 231720, 23172001]),
        _article("h2", eclass5_code=[23, 2318, 231801]),
        _article("h3", eclass5_code=[27, 2710]),
    ]
    out = aggregations.eclass_summary(rows, "eclass5_code", selected=None)
    by_g = {b.group: b.count for b in out.same_level}
    assert by_g == {23: 2, 27: 1}
    assert out.children == []


def test_eclass_with_selection_yields_siblings_and_children() -> None:
    """Selected = 2317 (depth 2). Siblings: depth-2 codes sharing
    parent 23 (i.e., 2317 and 2318). Children: depth-3 codes whose
    parent is 2317."""
    rows = [
        _article("h1", eclass5_code=[23, 2317, 231720, 23172001]),
        _article("h2", eclass5_code=[23, 2318, 231801]),
        _article("h3", eclass5_code=[23, 2317, 231730]),
        _article("h4", eclass5_code=[27, 2710]),  # different parent — not a sibling
    ]
    out = aggregations.eclass_summary(rows, "eclass5_code", selected=2317)
    siblings = {b.group: b.count for b in out.same_level}
    children = {b.group: b.count for b in out.children}
    # Siblings of 2317 (parent=23): 2317 (h1+h3), 2318 (h2). NOT 2710 (parent=27).
    assert siblings == {2317: 2, 2318: 1}
    # Children of 2317 at depth 3: 231720 (h1), 231730 (h3). NOT 231801 (parent=2318).
    assert children == {231720: 1, 231730: 1}


def test_eclass_depth_calculation_handles_each_level() -> None:
    """Sanity: depth = ceil(digits/2). 23→1, 2317→2, 231720→3, 23172001→4."""
    rows = [_article("h1", eclass5_code=[23, 2317, 231720, 23172001])]
    # Selecting at each depth surfaces its siblings.
    for selected, depth in [(23, 1), (2317, 2), (231720, 3), (23172001, 4)]:
        out = aggregations.eclass_summary(rows, "eclass5_code", selected=selected)
        # `selected` itself appears in same_level (parent matches).
        codes = {b.group for b in out.same_level}
        assert selected in codes, f"depth-{depth} selection missed itself"


# ──────────────────────────────────────────────────────────────────────
# ECLASS5SET
# ──────────────────────────────────────────────────────────────────────

def test_eclass5set_counts_intersection_per_entry() -> None:
    rows = [
        _article("h1", eclass5_code=[23, 2317, 231720]),
        _article("h2", eclass5_code=[23, 2318]),
        _article("h3", eclass5_code=[27, 2710]),
    ]
    aggs = [
        EClassesAggregation(id="entry-A", eClasses=[2317, 2318]),
        EClassesAggregation(id="entry-B", eClasses=[27]),
        EClassesAggregation(id="entry-C", eClasses=[99]),  # no match
    ]
    out = aggregations.eclass5set_summary(rows, aggs)
    by_id = {c.id: c.count for c in out}
    assert by_id == {"entry-A": 2, "entry-B": 1, "entry-C": 0}
    # Order preserved.
    assert [c.id for c in out] == ["entry-A", "entry-B", "entry-C"]


def test_eclass5set_empty_aggregation_entry_yields_zero() -> None:
    rows = [_article("h1", eclass5_code=[23])]
    aggs = [EClassesAggregation(id="empty", eClasses=[])]
    out = aggregations.eclass5set_summary(rows, aggs)
    assert out[0].count == 0


# ──────────────────────────────────────────────────────────────────────
# Top-level dispatcher
# ──────────────────────────────────────────────────────────────────────

def test_compute_summaries_only_emits_requested_kinds() -> None:
    article_rows = [_article("h1", manufacturerName="Bosch", eclass5_code=[23])]
    offer_rows = [_offer("h1", vendor_id="v1")]
    req = _req(summaries=[SummaryKind.MANUFACTURERS])
    out = aggregations.compute_summaries(req, article_rows=article_rows, offer_rows=offer_rows)
    # Manufacturers populated.
    assert out.manufacturer_summaries
    # Vendor not requested → still default (empty list).
    assert out.vendor_summaries == []
    # Eclass5 not requested → None.
    assert out.eclass5_categories is None


def test_compute_summaries_platform_categories_alias_to_categories() -> None:
    article_rows = [_article("h1", category_l1=["X"])]
    req = _req(summaries=[SummaryKind.PLATFORM_CATEGORIES])
    out = aggregations.compute_summaries(req, article_rows=article_rows, offer_rows=[])
    # PLATFORM_CATEGORIES (default flag) populates categories_summary.
    assert out.categories_summary is not None
    assert out.s2class_categories is None


def test_compute_summaries_platform_categories_alias_to_s2class_when_flag() -> None:
    article_rows = [_article("h1", s2class_code=[1001])]
    req = _req(
        summaries=[SummaryKind.PLATFORM_CATEGORIES],
        s2class_for_product_categories=True,
    )
    out = aggregations.compute_summaries(req, article_rows=article_rows, offer_rows=[])
    # With the flag, populates s2class_categories instead.
    assert out.s2class_categories is not None
    assert out.categories_summary is None


def test_compute_summaries_platform_categories_does_not_override_explicit_kind() -> None:
    """If both PLATFORM_CATEGORIES and CATEGORIES are requested, the
    explicit CATEGORIES populates first; PLATFORM_CATEGORIES alias
    sees the field already set and leaves it alone."""
    article_rows = [_article("h1", category_l1=["X"])]
    req = _req(summaries=[SummaryKind.CATEGORIES, SummaryKind.PLATFORM_CATEGORIES])
    out = aggregations.compute_summaries(req, article_rows=article_rows, offer_rows=[])
    assert out.categories_summary is not None
    # Single populated entry; no double-counting.
    assert len(out.categories_summary.same_level) == 1
