"""F3 — translate a `SearchRequest` into a Milvus scalar `expr` string.

Every scalar filter from spec §4.3 (legacy `*FilterProvider.java` parity)
is encoded here. `priceFilter` is intentionally NOT included — it cannot
be expressed as a Milvus expr (the per-row `prices` JSON array needs
currency × priceList × priority resolution in Python) and is handled by
`prices.passes_price_filter` in a post-Milvus pass.

The returned expression is AND-composed across filters at the top level.
Within multi-valued filters, semantics match legacy:

  * `requiredFeatures` — AND across feature names, OR within values
  * `closedMarketplaceOnly` — `array_contains_any(catalog_version_ids,
    [closedCatalogVersionIds])`. Per legacy `OfferFilterBuilder`: when
    the flag is set, intersect on the closed-CV list; on an empty list
    legacy emits a `terms` query against `[]` which matches nothing —
    we replicate via the always-false `id == ""` sentinel.
    `closedCatalogVersionIds` on its own (without the flag) is treated
    as request-side metadata for the core-sortiment logic; ftsearch
    does not impose a standalone CV intersection. The ACL is the layer
    that re-adds always-intersect semantics for legacy parity.
  * `coreSortimentOnly` — `enabled ∋ closedCatalogVersionIds` OR (with
    customer-uploaded sources) `enabled ∋ uploaded` OR
    (`enabled ∋ closedCatalogVersionIds` AND NOT `disabled ∋ uploaded`).
    Source list comes from `selectedArticleSources` per
    `CoreSortimentFilterProvider` + `ArticleSearchContext.getCoreArticleListSourceIds`.
  * `coreArticlesVendorsFilter` — for the listed vendors only the core
    sortiment counts; other vendors pass through. Modeled as
    `(vendor_id NOT IN [vendors]) OR <core sortiment expr>` (per
    `VendorSpecificPreFilterProvider` + `CatalogViewPreFilterFactory`).
  * `blockedEClassVendorsFilters` — for the listed vendors, articles in
    the blocked eClass codes are filtered out. `value=true` codes are
    blocked; `value=false` codes are kept (legacy `INVERSE` mode treats
    them as exceptions). Modeled per entry as
    `(vendor_id NOT IN [vendors]) OR NOT (eclassN_code IN [block-true]
     AND eclassN_code NOT IN [block-false])`.
"""

from __future__ import annotations

from models import EClassVersion, SearchRequest

# CategoryPath encoding mirrors `commons/CategoryPath.asStringPath`: per
# element, replace U+00A6 (¦) with U+007C (|), then join with U+00A6.
CATEGORY_PATH_SEPARATOR = "¦"
CATEGORY_PATH_ESCAPE = "|"


def encode_category_path(elements: list[str]) -> str:
    return CATEGORY_PATH_SEPARATOR.join(
        e.replace(CATEGORY_PATH_SEPARATOR, CATEGORY_PATH_ESCAPE) for e in elements
    )


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _str_array(values: list[str]) -> str:
    return "[" + ", ".join(_quote(v) for v in values) + "]"


def _int_array(values: list[int]) -> str:
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def _and(parts: list[str | None]) -> str | None:
    keep = [p for p in parts if p]
    if not keep:
        return None
    if len(keep) == 1:
        return keep[0]
    return " and ".join(f"({p})" for p in keep)


# ---------- per-filter atoms ---------------------------------------------

def _vendor_ids(req: SearchRequest) -> str | None:
    if not req.vendor_ids_filter:
        return None
    return f"vendor_id in {_str_array(req.vendor_ids_filter)}"


def _article_ids(req: SearchRequest) -> str | None:
    if not req.article_ids_filter:
        return None
    return f"id in {_str_array(req.article_ids_filter)}"


def _manufacturers(req: SearchRequest) -> str | None:
    if not req.manufacturers_filter:
        return None
    return f"manufacturerName in {_str_array(req.manufacturers_filter)}"


def _delivery_time(req: SearchRequest) -> str | None:
    # Legacy `DeliveryTimeFilterProvider` is a no-op when maxDeliveryTime <= 0.
    if req.max_delivery_time <= 0:
        return None
    return f"delivery_time_days_max <= {int(req.max_delivery_time)}"


def _required_features(req: SearchRequest) -> str | None:
    parts: list[str] = []
    for ff in req.required_features:
        if not ff.values:
            continue
        tokens = [f"{ff.name}={v}" for v in ff.values]
        parts.append(f"array_contains_any(features, {_str_array(tokens)})")
    return _and(parts) if parts else None


def _category_prefix(req: SearchRequest) -> str | None:
    elements = req.current_category_path_elements
    if not elements:
        return None
    depth = len(elements)
    if depth < 1 or depth > 5:
        # Schema only supports l1..l5. Out-of-range = no-op (mirrors legacy
        # `IllegalStateException` defensively rather than raising).
        return None
    return f"array_contains(category_l{depth}, {_quote(encode_category_path(elements))})"


def _eclass_codes(req: SearchRequest) -> str | None:
    parts: list[str] = []
    if req.current_eclass5_code is not None:
        parts.append(f"eclass5_code == {int(req.current_eclass5_code)}")
    if req.current_eclass7_code is not None:
        parts.append(f"eclass7_code == {int(req.current_eclass7_code)}")
    if req.current_s2class_code is not None:
        parts.append(f"s2class_code == {int(req.current_s2class_code)}")
    return _and(parts)


def _eclasses_filter(req: SearchRequest) -> str | None:
    if not req.eclasses_filter:
        return None
    field = "s2class_code" if req.s2class_for_product_categories else "eclass5_code"
    return f"{field} in {_int_array(req.eclasses_filter)}"


_MATCH_NOTHING_EXPR = 'id == ""'


def _closed_marketplace(req: SearchRequest) -> str | None:
    if not req.closed_marketplace_only:
        return None
    cv = req.selected_article_sources.closed_catalog_version_ids
    if not cv:
        return _MATCH_NOTHING_EXPR
    return f"array_contains_any(catalog_version_ids, {_str_array(cv)})"


def _relationships(req: SearchRequest) -> str | None:
    parts: list[str] = []
    if req.accessories_for_article_number:
        parts.append(
            f"array_contains(relationship_accessory_for, {_quote(req.accessories_for_article_number)})"
        )
    if req.spare_parts_for_article_number:
        parts.append(
            f"array_contains(relationship_spare_part_for, {_quote(req.spare_parts_for_article_number)})"
        )
    if req.similar_to_article_number:
        parts.append(
            f"array_contains(relationship_similar_to, {_quote(req.similar_to_article_number)})"
        )
    return _and(parts)


def _core_sortiment_inner(req: SearchRequest) -> str | None:
    """The boolean expression equivalent to legacy CoreSortimentFilterProvider.filteringQuery,
    independent of the `coreSortimentOnly` toggle. Used by `_core_sortiment` and reused
    inside `_core_articles_vendors`."""
    sas = req.selected_article_sources
    base = list(sas.closed_catalog_version_ids)
    uploaded = list(sas.customer_uploaded_core_article_list_source_ids)
    if not base and not uploaded:
        # Legacy builds a filter against an empty list which matches nothing.
        # We treat as no-op here to avoid surprising mass-exclusion when
        # the ACL forwards an under-populated context.
        return None
    if not uploaded:
        return f"array_contains_any(core_marker_enabled_sources, {_str_array(base)})"
    sub_a = f"array_contains_any(core_marker_enabled_sources, {_str_array(uploaded)})"
    if not base:
        return sub_a
    sub_b_left = f"array_contains_any(core_marker_enabled_sources, {_str_array(base)})"
    sub_b_right = f"not array_contains_any(core_marker_disabled_sources, {_str_array(uploaded)})"
    return f"({sub_a}) or (({sub_b_left}) and ({sub_b_right}))"


def _core_sortiment(req: SearchRequest) -> str | None:
    if not req.core_sortiment_only:
        return None
    return _core_sortiment_inner(req)


def _core_articles_vendors(req: SearchRequest) -> str | None:
    """For listed vendors → only their core sortiment counts; other vendors pass."""
    vendors = req.core_articles_vendors_filter
    if not vendors:
        return None
    inner = _core_sortiment_inner(req)
    if inner is None:
        return None
    return f"(vendor_id not in {_str_array(vendors)}) or ({inner})"


_ECLASS_FIELD = {
    EClassVersion.ECLASS_5_1: "eclass5_code",
    EClassVersion.ECLASS_7_1: "eclass7_code",
    EClassVersion.S2CLASS: "s2class_code",
}


def _blocked_eclass_vendors(req: SearchRequest) -> str | None:
    if not req.blocked_eclass_vendors_filters:
        return None
    parts: list[str] = []
    for entry in req.blocked_eclass_vendors_filters:
        field = _ECLASS_FIELD[entry.e_class_version]
        block_true = [g.e_class_group_code for g in entry.blocked_e_class_groups if g.value]
        block_false = [g.e_class_group_code for g in entry.blocked_e_class_groups if not g.value]
        if not block_true:
            continue
        block_expr = f"{field} in {_int_array(block_true)}"
        if block_false:
            block_expr = f"({block_expr}) and ({field} not in {_int_array(block_false)})"
        if entry.vendor_ids:
            parts.append(
                f"(vendor_id not in {_str_array(entry.vendor_ids)}) or (not ({block_expr}))"
            )
        else:
            parts.append(f"not ({block_expr})")
    return _and(parts)


# ---------- top-level entry point ----------------------------------------

def build_milvus_expr(req: SearchRequest) -> str | None:
    """Return the AND-composed Milvus expression, or None if no scalar
    filters apply. `priceFilter` is excluded by design (post-Milvus pass).
    """
    return _and([
        _vendor_ids(req),
        _article_ids(req),
        _manufacturers(req),
        _delivery_time(req),
        _required_features(req),
        _category_prefix(req),
        _eclass_codes(req),
        _eclasses_filter(req),
        _closed_marketplace(req),
        _relationships(req),
        _core_sortiment(req),
        _core_articles_vendors(req),
        _blocked_eclass_vendors(req),
    ])
