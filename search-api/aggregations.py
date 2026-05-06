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
        needed.add("s2class_code")
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
    return bool(requested & {
        SummaryKind.VENDORS, SummaryKind.FEATURES, SummaryKind.PRICES,
        SummaryKind.MANUFACTURERS,
    })


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
    category_article_rows: list[dict] | None = None,
    s2class_article_rows: list[dict] | None = None,
    vendor_offer_rows: list[dict] | None = None,
    mfg_article_rows: list[dict] | None = None,
    mfg_offer_rows: list[dict] | None = None,
) -> Summaries:
    """Compute the requested summary kinds. Kinds not in `req.summaries`
    are left at default (empty list / None).

    Disjunctive faceting: legacy ES computes each aggregation excluding
    its own post-filter. When provided, the `*_rows` overrides use
    unscoped rows instead of the default `article_rows`/`offer_rows`."""
    out = Summaries()
    requested = set(req.summaries)

    cat_rows = category_article_rows if category_article_rows is not None else article_rows
    s2c_rows = s2class_article_rows if s2class_article_rows is not None else article_rows
    vnd_rows = vendor_offer_rows if vendor_offer_rows is not None else offer_rows
    mfg_a = mfg_article_rows if mfg_article_rows is not None else article_rows
    mfg_o = mfg_offer_rows if mfg_offer_rows is not None else offer_rows

    if SummaryKind.VENDORS in requested:
        out.vendor_summaries = vendors_summary(vnd_rows)
    if SummaryKind.MANUFACTURERS in requested:
        out.manufacturer_summaries = manufacturers_summary(mfg_a, mfg_o)
    if SummaryKind.FEATURES in requested:
        out.feature_summaries = features_summary(offer_rows)
    if SummaryKind.PRICES in requested:
        out.prices_summary = prices_summary(offer_rows, req)
    if SummaryKind.CATEGORIES in requested:
        out.categories_summary = categories_summary(
            cat_rows, req.current_category_path_elements,
        )
    if SummaryKind.ECLASS5 in requested:
        out.eclass5_categories = eclass_summary(
            article_rows, "eclass5_code", req.current_eclass5_code, offer_rows,
        )
    if SummaryKind.ECLASS7 in requested:
        out.eclass7_categories = eclass_summary(
            article_rows, "eclass7_code", req.current_eclass7_code, offer_rows,
        )
    if SummaryKind.S2CLASS in requested:
        out.s2class_categories = eclass_summary(
            s2c_rows, "s2class_code", req.current_s2class_code, offer_rows,
        )
    if SummaryKind.ECLASS5SET in requested:
        out.eclasses_aggregations = eclass5set_summary(
            article_rows, req.eclasses_aggregations, offer_rows,
        )
    if SummaryKind.PLATFORM_CATEGORIES in requested:
        if req.s2class_for_product_categories:
            if out.s2class_categories is None:
                out.s2class_categories = eclass_summary(
                    s2c_rows, "s2class_code", req.current_s2class_code, offer_rows,
                )
        else:
            if out.categories_summary is None:
                out.categories_summary = categories_summary(
                    cat_rows, req.current_category_path_elements,
                )

    return out


# ──────────────────────────────────────────────────────────────────────
# VENDORS
# ──────────────────────────────────────────────────────────────────────

def vendors_summary(offer_rows: Iterable[dict]) -> list[VendorSummary]:
    """Per-offer count grouped by vendor. Legacy ES counts each nested
    offer document, so we count each offer row (not distinct articles)."""
    by_vendor: dict[str, int] = defaultdict(int)
    for o in offer_rows:
        vendor = o.get("vendor_id")
        if not vendor:
            continue
        by_vendor[str(vendor)] += 1
    items = sorted(by_vendor.items(), key=lambda r: (-r[1], r[0]))
    return [VendorSummary(vendorId=v, count=c) for v, c in items]


# ──────────────────────────────────────────────────────────────────────
# MANUFACTURERS
# ──────────────────────────────────────────────────────────────────────

def manufacturers_summary(
    article_rows: Iterable[dict],
    offer_rows: Iterable[dict],
) -> list[NameCount]:
    """Per-offer count grouped by manufacturer name. Legacy ES stores
    manufacturerName inside nested offer docs, so the count is per-offer.
    We join article → manufacturer via article_hash, then count each offer."""
    hash_to_mfg: dict[str, str] = {}
    for a in article_rows:
        h = a.get("article_hash")
        if h:
            name = a.get("manufacturerName")
            hash_to_mfg[str(h)] = str(name) if name else ""
    counts: dict[str, int] = defaultdict(int)
    for o in offer_rows:
        h = o.get("article_hash")
        if h:
            mfg = hash_to_mfg.get(str(h), "")
            counts[mfg] += 1
    items = sorted(counts.items(), key=lambda r: (-r[1], r[0]))
    return [NameCount(name=n, count=c) for n, c in items]


# ──────────────────────────────────────────────────────────────────────
# FEATURES
# ──────────────────────────────────────────────────────────────────────

def features_summary(offer_rows: Iterable[dict]) -> list[FeatureSummary]:
    """Per-offer count grouped by (feature name, value). Legacy ES stores
    features as nested offer-level fields, counting each offer doc."""
    # name → value → count (per-offer)
    by_name: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # name → total offers having any value
    name_totals: dict[str, int] = defaultdict(int)
    for o in offer_rows:
        seen_names: set[str] = set()
        for token in o.get("features") or []:
            name, eq, value = str(token).partition("=")
            if not eq or not name:
                continue
            by_name[name][value] += 1
            seen_names.add(name)
        for name in seen_names:
            name_totals[name] += 1
    out: list[FeatureSummary] = []
    for name in sorted(by_name.keys()):
        values = by_name[name]
        value_buckets = sorted(
            values.items(),
            key=lambda r: (-r[1], r[0]),
        )
        out.append(FeatureSummary(
            name=name,
            count=name_totals[name],
            values=[FeatureValueCount(value=v, count=c) for v, c in value_buckets],
        ))
    out.sort(key=lambda f: (-f.count, f.name))
    return out[:100]


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
) -> CategoriesSummary | None:
    """Hierarchical: given the depth the user is browsing,
    `sameLevel` are paths at that depth (siblings of the current node,
    or top-level children when depth=0); `children` are paths one level
    deeper under the current path.

    Legacy parity: returns None when `currentCategoryPathElements` is
    not set (Java `CategorySummaryExtractor` returns null when
    `currentCategoryPath == null`).
    """
    if not current_path:
        return None

    depth = len(current_path)
    children_depth = depth + 1

    if depth > 5:
        return CategoriesSummary(
            currentCategoryPathElements=list(current_path),
            sameLevel=[], children=[],
        )

    prefix = current_path[: depth - 1] if depth > 1 else []
    prefix_path = _CATEGORY_PATH_SEPARATOR.join(prefix) if prefix else ""

    same_level_counts: dict[str, set[str]] = defaultdict(set)
    children_counts: dict[str, set[str]] = defaultdict(set)
    selected_path = _CATEGORY_PATH_SEPARATOR.join(current_path)

    for a in article_rows:
        h = a.get("article_hash")
        if not h:
            continue
        h = str(h)
        for path in a.get(f"category_l{depth}") or []:
            if prefix and not _starts_with_prefix(path, prefix_path):
                continue
            same_level_counts[str(path)].add(h)
        if 1 <= children_depth <= 5:
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
    offer_rows: Iterable[dict] | None = None,
) -> EClassCategories | None:
    """Hierarchical eClass / S2Class summary, derived from the per-article
    `ARRAY<INT32>` carrying the full root→leaf chain.

    Legacy counts per-offer (ES documents). When `offer_rows` is provided,
    bucket counts reflect the number of offers per article hash.
    """
    if selected is None:
        return None

    hash_offer_count: dict[str, int] | None = None
    if offer_rows is not None:
        hash_offer_count = {}
        for o in offer_rows:
            h = str(o.get("article_hash", ""))
            if h:
                hash_offer_count[h] = hash_offer_count.get(h, 0) + 1

    sel_depth = _eclass_depth(selected)
    parent = _eclass_parent(selected)

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
                if parent is None or _eclass_parent(code) == parent:
                    sibling_counts[code].add(h)
            elif d == sel_depth + 1 and _eclass_parent(code) == selected:
                children_counts[code].add(h)

    same = _to_eclass_buckets(sibling_counts, hash_offer_count)
    kids = _to_eclass_buckets(children_counts, hash_offer_count)
    if not same and not kids:
        return None
    return EClassCategories(
        selectedEClassGroup=selected,
        sameLevel=same,
        children=kids,
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
    roots = _to_eclass_buckets(root_counts)
    if not roots:
        return None
    return EClassCategories(
        selectedEClassGroup=None,
        sameLevel=roots,
        children=[],
    )


def _eclass_depth(code: int) -> int:
    """eClass / S2Class depth from a zero-padded 8-digit code.

    Codes are stored as 8-digit ints with trailing zero pairs:
    21000000 → 1, 21040000 → 2, 21042100 → 3, 21042101 → 4.
    Compact codes (no zero padding) also work: 23 → 1, 2317 → 2."""
    if code <= 0:
        return 0
    s = f"{code:08d}"
    depth = 0
    for i in range(0, 8, 2):
        if s[i:i+2] != "00":
            depth = i // 2 + 1
    return depth


def _eclass_parent(code: int) -> int | None:
    """Parent code: zero out the deepest non-zero 2-digit group.
    21042101 → 21042100, 21040000 → 21000000, 21000000 → None."""
    d = _eclass_depth(code)
    if d <= 1:
        return None
    s = f"{code:08d}"
    return int(s[: (d - 1) * 2] + "0" * (8 - (d - 1) * 2))


def _to_eclass_buckets(
    counts: dict[int, set[str]],
    hash_offer_count: dict[str, int] | None = None,
) -> list[EClassBucket]:
    def _count(hs: set[str]) -> int:
        if hash_offer_count:
            return sum(hash_offer_count.get(h, 1) for h in hs)
        return len(hs)
    items = sorted(
        ((g, _count(hs)) for g, hs in counts.items()),
        key=lambda r: (-r[1], r[0]),
    )
    return [EClassBucket(group=g, count=c) for g, c in items]


# ──────────────────────────────────────────────────────────────────────
# ECLASS5SET
# ──────────────────────────────────────────────────────────────────────

def eclass5set_summary(
    article_rows: Iterable[dict],
    aggregations: list[EClassesAggregation],
    offer_rows: Iterable[dict] | None = None,
) -> list[EClassesAggregationCount]:
    """For each named aggregation entry, count offers whose article's
    `s2class_code` array intersects `entry.e_classes`. Legacy counts
    per-offer (ES documents), not per-article. When `offer_rows` is
    provided, the count reflects the number of offers; otherwise falls
    back to counting distinct article hashes."""
    hash_to_s2: dict[str, set[int]] = {}
    for a in article_rows:
        h = a.get("article_hash")
        if not h:
            continue
        codes = {int(c) for c in (a.get("s2class_code") or [])}
        hash_to_s2[str(h)] = codes

    hash_offer_count: dict[str, int] = {}
    if offer_rows is not None:
        for o in offer_rows:
            h = str(o.get("article_hash", ""))
            if h:
                hash_offer_count[h] = hash_offer_count.get(h, 0) + 1

    out: list[EClassesAggregationCount] = []
    for entry in aggregations:
        target = {int(c) for c in entry.e_classes}
        if not target:
            out.append(EClassesAggregationCount(id=entry.id, count=0))
            continue
        count = 0
        for h, codes in hash_to_s2.items():
            if codes & target:
                count += hash_offer_count.get(h, 1) if hash_offer_count else 1
        out.append(EClassesAggregationCount(id=entry.id, count=count))
    return out
