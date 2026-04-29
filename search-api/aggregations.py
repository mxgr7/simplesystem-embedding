"""F5 — summary aggregations over the full filtered hit set.

Pure-function module: every aggregation takes pre-fetched article and/or
offer rows + the relevant request fields, and returns the matching slice
of `models.Summaries`. Routing is responsible for fetching the rows;
this module owns the counting logic.

Article-level data lives on `articles_v{N}` (one row per dedup'd article):
manufacturer, name, category_l1..l5, eclass5_code, eclass7_code,
s2class_code. Offer-level data lives on `offers_v{N}` (one row per offer):
vendor_id, features, prices, article_hash (the join key).

Counts are per *distinct article*, not per offer — even for offer-level
sources (vendors/features/prices) the unit is the article. That matches
the user-facing "filter by vendor V → see N products" interpretation
under the F9 dedup topology, which differs from the legacy single-doc-
per-offer schema where the count would be per-offer.

Hierarchy navigation is data-driven:

  - Categories: `category_l1..l5` ARRAY<VARCHAR> already carries each
    article's prefix paths at every depth. sameLevel/children at depth N
    derive from `category_l{N}` / `category_l{N+1}` directly.
  - eClass / S2Class: `eclassN_code` / `s2class_code` ARRAY<INT32>
    carries the full root→leaf hierarchy. The parent of a code is
    `code // 100` for eClass51/eClass71 (2-digit per level); the depth
    of a code is `len(str(code)) // 2`. We use these to compute
    siblings (same-parent codes at the same depth) and children
    (codes one level deeper under the selected one).

Aggregations are clipped at the hitcount cap upstream; counts here are
exact within the rows we received. If the upstream cap fired, every
summary count is a lower bound — surfaced via `metadata.hitCountClipped`
which is already wired in F4.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Iterable

from models import (
    CategoriesSummary,
    CategoryBucket,
    EClassBucket,
    EClassCategories,
    EClassesAggregation,
    EClassesAggregationCount,
    FeatureSummary,
    FeatureValueCount,
    NameCount,
    PricesSummary,
    SearchRequest,
    Summaries,
    SummaryKind,
    VendorSummary,
)
from prices import resolve_price


# ──────────────────────────────────────────────────────────────────────
# Field-set planner — only fetch what's needed for the requested kinds.
# ──────────────────────────────────────────────────────────────────────

def article_fields_needed(req: SearchRequest) -> set[str]:
    """Article-level fields required to compute the requested summaries.
    `article_hash` is always included so callers can group by it."""
    needed: set[str] = {"article_hash"}
    requested = set(req.summaries)

    if SummaryKind.MANUFACTURERS in requested:
        needed.add("manufacturerName")
    if SummaryKind.CATEGORIES in requested:
        needed.update({f"category_l{d}" for d in range(1, 6)})
    if SummaryKind.ECLASS5 in requested:
        needed.add("eclass5_code")
    if SummaryKind.ECLASS7 in requested:
        needed.add("eclass7_code")
    if SummaryKind.S2CLASS in requested:
        needed.add("s2class_code")
    if SummaryKind.ECLASS5SET in requested:
        needed.add("eclass5_code")
    if SummaryKind.PLATFORM_CATEGORIES in requested:
        if req.s2class_for_product_categories:
            needed.add("s2class_code")
        else:
            needed.update({f"category_l{d}" for d in range(1, 6)})
    return needed


def offer_fields_needed(req: SearchRequest) -> set[str]:
    """Offer-level fields required for the requested summaries.
    `article_hash` is always included so we can group by article."""
    needed: set[str] = {"article_hash"}
    requested = set(req.summaries)

    if SummaryKind.VENDORS in requested:
        needed.add("vendor_id")
    if SummaryKind.FEATURES in requested:
        needed.add("features")
    if SummaryKind.PRICES in requested:
        needed.add("prices")
    return needed


def needs_offer_fetch(req: SearchRequest) -> bool:
    """True iff at least one requested summary is offer-sourced."""
    requested = set(req.summaries)
    return bool(requested & {SummaryKind.VENDORS, SummaryKind.FEATURES, SummaryKind.PRICES})


def needs_article_fetch(req: SearchRequest) -> bool:
    """True iff at least one requested summary is article-sourced."""
    requested = set(req.summaries)
    return bool(requested & {
        SummaryKind.MANUFACTURERS, SummaryKind.CATEGORIES,
        SummaryKind.ECLASS5, SummaryKind.ECLASS7, SummaryKind.S2CLASS,
        SummaryKind.ECLASS5SET, SummaryKind.PLATFORM_CATEGORIES,
    })


# ──────────────────────────────────────────────────────────────────────
# Top-level dispatcher
# ──────────────────────────────────────────────────────────────────────

def compute_summaries(
    req: SearchRequest,
    *,
    article_rows: list[dict],
    offer_rows: list[dict],
) -> Summaries:
    """Compute the requested summary kinds. Kinds not in `req.summaries`
    are left at default (empty list / None)."""
    out = Summaries()
    requested = set(req.summaries)

    if SummaryKind.VENDORS in requested:
        out.vendor_summaries = vendors_summary(offer_rows)
    if SummaryKind.MANUFACTURERS in requested:
        out.manufacturer_summaries = manufacturers_summary(article_rows)
    if SummaryKind.FEATURES in requested:
        out.feature_summaries = features_summary(offer_rows)
    if SummaryKind.PRICES in requested:
        out.prices_summary = prices_summary(offer_rows, req)
    if SummaryKind.CATEGORIES in requested:
        out.categories_summary = categories_summary(
            article_rows, req.current_category_path_elements,
        )
    if SummaryKind.ECLASS5 in requested:
        out.eclass5_categories = eclass_summary(
            article_rows, "eclass5_code", req.current_eclass5_code,
        )
    if SummaryKind.ECLASS7 in requested:
        out.eclass7_categories = eclass_summary(
            article_rows, "eclass7_code", req.current_eclass7_code,
        )
    if SummaryKind.S2CLASS in requested:
        out.s2class_categories = eclass_summary(
            article_rows, "s2class_code", req.current_s2class_code,
        )
    if SummaryKind.ECLASS5SET in requested:
        out.eclasses_aggregations = eclass5set_summary(
            article_rows, req.eclasses_aggregations,
        )
    if SummaryKind.PLATFORM_CATEGORIES in requested:
        # Per spec: PLATFORM_CATEGORIES is an alias of CATEGORIES, OR
        # of S2CLASS when `s2ClassForProductCategories` is true. The
        # response shape doesn't have a separate field — the alias
        # populates the corresponding existing field if it isn't
        # already populated by an explicit kind.
        if req.s2class_for_product_categories:
            if out.s2class_categories is None:
                out.s2class_categories = eclass_summary(
                    article_rows, "s2class_code", req.current_s2class_code,
                )
        else:
            if out.categories_summary is None:
                out.categories_summary = categories_summary(
                    article_rows, req.current_category_path_elements,
                )

    return out


# ──────────────────────────────────────────────────────────────────────
# VENDORS
# ──────────────────────────────────────────────────────────────────────

def vendors_summary(offer_rows: Iterable[dict]) -> list[VendorSummary]:
    """Distinct (article_hash, vendor_id) pairs, grouped by vendor.
    Count is the number of distinct articles each vendor has offers for
    in the filtered hit set."""
    by_vendor: dict[str, set[str]] = defaultdict(set)
    for o in offer_rows:
        vendor = o.get("vendor_id")
        h = o.get("article_hash")
        if not vendor or not h:
            continue
        by_vendor[str(vendor)].add(str(h))
    # Sort by count desc, vendor_id asc for determinism.
    items = sorted(
        ((v, len(hs)) for v, hs in by_vendor.items()),
        key=lambda r: (-r[1], r[0]),
    )
    return [VendorSummary(vendorId=v, count=c) for v, c in items]


# ──────────────────────────────────────────────────────────────────────
# MANUFACTURERS
# ──────────────────────────────────────────────────────────────────────

def manufacturers_summary(article_rows: Iterable[dict]) -> list[NameCount]:
    """One article per row by construction; count distinct manufacturer
    names. Empty manufacturers are skipped (matches legacy ES `terms`
    behaviour on missing keyword field)."""
    counts: dict[str, int] = defaultdict(int)
    for a in article_rows:
        name = a.get("manufacturerName")
        if not name:
            continue
        counts[str(name)] += 1
    items = sorted(counts.items(), key=lambda r: (-r[1], r[0]))
    return [NameCount(name=n, count=c) for n, c in items]


# ──────────────────────────────────────────────────────────────────────
# FEATURES
# ──────────────────────────────────────────────────────────────────────

def features_summary(offer_rows: Iterable[dict]) -> list[FeatureSummary]:
    """Distinct articles per (feature name, value), aggregated to
    {name → {value → distinct articles}}. Top-level count per feature is
    "distinct articles having at least one offer with any value for that
    feature"."""
    # name → value → set(article_hash)
    by_name: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for o in offer_rows:
        h = o.get("article_hash")
        if not h:
            continue
        for token in o.get("features") or []:
            name, eq, value = str(token).partition("=")
            if not eq or not name:
                continue
            by_name[name][value].add(str(h))
    out: list[FeatureSummary] = []
    for name in sorted(by_name.keys()):
        values = by_name[name]
        # Distinct articles having ANY value for this feature.
        article_set: set[str] = set()
        for vs in values.values():
            article_set |= vs
        value_buckets = sorted(
            ((v, len(hs)) for v, hs in values.items()),
            key=lambda r: (-r[1], r[0]),
        )
        out.append(FeatureSummary(
            name=name,
            count=len(article_set),
            values=[FeatureValueCount(value=v, count=c) for v, c in value_buckets],
        ))
    # Sort feature names by count desc, name asc to surface most-populated first.
    out.sort(key=lambda f: (-f.count, f.name))
    return out


# ──────────────────────────────────────────────────────────────────────
# PRICES
# ──────────────────────────────────────────────────────────────────────

def prices_summary(
    offer_rows: Iterable[dict],
    req: SearchRequest,
) -> list[PricesSummary]:
    """For each offer, resolve its in-scope price under the request
    (currency × sourcePriceListIds × priority). Group by currency,
    return min/max. The legacy contract returns one entry per request
    currency in practice; the array shape is preserved."""
    sas = req.selected_article_sources
    by_ccy: dict[str, list[Decimal]] = defaultdict(list)
    for o in offer_rows:
        resolved = resolve_price(
            o.get("prices"),
            currency=req.currency,
            source_price_list_ids=sas.source_price_list_ids,
        )
        if resolved is None:
            continue
        by_ccy[req.currency.upper()].append(resolved)
    out: list[PricesSummary] = []
    for ccy in sorted(by_ccy.keys()):
        prices = by_ccy[ccy]
        out.append(PricesSummary(
            min=float(min(prices)),
            max=float(max(prices)),
            currencyCode=ccy,
        ))
    return out


# ──────────────────────────────────────────────────────────────────────
# CATEGORIES
# ──────────────────────────────────────────────────────────────────────

_CATEGORY_PATH_SEPARATOR = "¦"


def _path_to_elements(path: str) -> list[str]:
    """Inverse of `filters.encode_category_path` for a stored path."""
    return path.split(_CATEGORY_PATH_SEPARATOR) if path else []


def categories_summary(
    article_rows: Iterable[dict],
    current_path: list[str],
) -> CategoriesSummary:
    """Hierarchical: given the depth the user is browsing,
    `sameLevel` are paths at that depth (siblings of the current node,
    or top-level children when depth=0); `children` are paths one level
    deeper under the current path.

    Per F4/spec, depth 0 (no current path) → sameLevel surfaces the
    top-level categories (depth 1), children is empty (no deeper level
    is implied without a selection).
    """
    depth = len(current_path)
    same_depth = max(depth, 1)
    children_depth = depth + 1 if depth > 0 else 0

    if same_depth > 5:
        # Out of schema — return an empty hierarchy with the user's
        # path echoed back. Defensive; the F3 filter rejects these too.
        return CategoriesSummary(
            currentCategoryPathElements=list(current_path),
            sameLevel=[], children=[],
        )

    prefix = current_path[: same_depth - 1] if same_depth > 0 else []
    prefix_path = _CATEGORY_PATH_SEPARATOR.join(prefix) if prefix else ""

    # sameLevel: count distinct articles per path at same_depth that
    # share the prefix.
    same_level_counts: dict[str, set[str]] = defaultdict(set)
    children_counts: dict[str, set[str]] = defaultdict(set)
    selected_path = _CATEGORY_PATH_SEPARATOR.join(current_path) if current_path else None

    for a in article_rows:
        h = a.get("article_hash")
        if not h:
            continue
        h = str(h)
        for path in a.get(f"category_l{same_depth}") or []:
            if prefix and not _starts_with_prefix(path, prefix_path):
                continue
            same_level_counts[str(path)].add(h)
        if children_depth and 1 <= children_depth <= 5 and selected_path:
            for path in a.get(f"category_l{children_depth}") or []:
                if not _starts_with_prefix(path, selected_path):
                    continue
                children_counts[str(path)].add(h)

    same_level = _to_category_buckets(same_level_counts)
    children = _to_category_buckets(children_counts)
    return CategoriesSummary(
        currentCategoryPathElements=list(current_path),
        sameLevel=same_level, children=children,
    )


def _starts_with_prefix(path: str, prefix: str) -> bool:
    """A path starts with a prefix iff it equals the prefix or the prefix
    + a separator is a strict prefix. Avoids `Werkzeug` false-matching
    `Werkzeugmacher`."""
    if not prefix:
        return True
    return path == prefix or path.startswith(prefix + _CATEGORY_PATH_SEPARATOR)


def _to_category_buckets(counts: dict[str, set[str]]) -> list[CategoryBucket]:
    items = sorted(
        ((p, len(hs)) for p, hs in counts.items()),
        key=lambda r: (-r[1], r[0]),
    )
    return [
        CategoryBucket(categoryPathElements=_path_to_elements(p), count=c)
        for p, c in items
    ]


# ──────────────────────────────────────────────────────────────────────
# ECLASS5 / ECLASS7 / S2CLASS
# ──────────────────────────────────────────────────────────────────────

def eclass_summary(
    article_rows: Iterable[dict],
    field: str,
    selected: int | None,
) -> EClassCategories:
    """Hierarchical eClass / S2Class summary, derived from the per-article
    `ARRAY<INT32>` carrying the full root→leaf chain.

    Hierarchy convention: each level is two trailing digits (eClass51,
    eClass71). The parent of code C is `C // 100`; the depth is
    `len(str(C)) // 2`. S2Class follows the same pattern in this codebase.

    Without a selection, sameLevel surfaces top-level (depth 1) codes
    that appear in the filtered hit set; children is empty.

    With a selection at depth N: sameLevel surfaces sibling codes (same
    parent at depth N-1), children surfaces codes at depth N+1 whose
    parent is the selection.
    """
    if selected is None:
        return _eclass_root_summary(article_rows, field)

    sel_depth = _eclass_depth(selected)
    parent = selected // 100 if sel_depth > 1 else None

    sibling_counts: dict[int, set[str]] = defaultdict(set)
    children_counts: dict[int, set[str]] = defaultdict(set)
    for a in article_rows:
        h = a.get("article_hash")
        if not h:
            continue
        h = str(h)
        for code in a.get(field) or []:
            code = int(code)
            d = _eclass_depth(code)
            if d == sel_depth:
                if parent is None or code // 100 == parent:
                    sibling_counts[code].add(h)
            elif d == sel_depth + 1 and code // 100 == selected:
                children_counts[code].add(h)

    return EClassCategories(
        selectedEClassGroup=selected,
        sameLevel=_to_eclass_buckets(sibling_counts),
        children=_to_eclass_buckets(children_counts),
    )


def _eclass_root_summary(
    article_rows: Iterable[dict],
    field: str,
) -> EClassCategories:
    """No selection → sameLevel = depth-1 codes in the hit set, children empty."""
    root_counts: dict[int, set[str]] = defaultdict(set)
    for a in article_rows:
        h = a.get("article_hash")
        if not h:
            continue
        h = str(h)
        for code in a.get(field) or []:
            code = int(code)
            if _eclass_depth(code) == 1:
                root_counts[code].add(h)
    return EClassCategories(
        selectedEClassGroup=None,
        sameLevel=_to_eclass_buckets(root_counts),
        children=[],
    )


def _eclass_depth(code: int) -> int:
    """eClass / S2Class depth from code. Two digits per level.
    23 → 1, 2317 → 2, 231720 → 3, 23172001 → 4. Codes with an odd
    number of digits round up — defensive against rare malformed
    inputs."""
    if code <= 0:
        return 0
    return (len(str(code)) + 1) // 2


def _to_eclass_buckets(counts: dict[int, set[str]]) -> list[EClassBucket]:
    items = sorted(
        ((g, len(hs)) for g, hs in counts.items()),
        key=lambda r: (-r[1], r[0]),
    )
    return [EClassBucket(group=g, count=c) for g, c in items]


# ──────────────────────────────────────────────────────────────────────
# ECLASS5SET
# ──────────────────────────────────────────────────────────────────────

def eclass5set_summary(
    article_rows: Iterable[dict],
    aggregations: list[EClassesAggregation],
) -> list[EClassesAggregationCount]:
    """For each named aggregation entry, count distinct articles whose
    `eclass5_code` array intersects `entry.e_classes`. Order of the
    returned list mirrors the request order — callers identify entries
    by `id`."""
    out: list[EClassesAggregationCount] = []
    for entry in aggregations:
        target = {int(c) for c in entry.e_classes}
        if not target:
            out.append(EClassesAggregationCount(id=entry.id, count=0))
            continue
        matched: set[str] = set()
        for a in article_rows:
            h = a.get("article_hash")
            if not h:
                continue
            codes = {int(c) for c in (a.get("eclass5_code") or [])}
            if codes & target:
                matched.add(str(h))
        out.append(EClassesAggregationCount(id=entry.id, count=len(matched)))
    return out
