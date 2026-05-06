# F3 — Scalar filtering + price-resolution module

**Category**: ftsearch (`./search-api/`)
**Depends on**: F1 (schema), F2 (request contract)
**Unblocks**: F4, F5, A2, A6

References: spec §4.3, §3 (currency two-roles), §7.

**Legacy reference** (next-gen): per-filter providers in `article/search/query/src/main/java/com/simplesystem/nextgen/article/search/query/infrastructure/search/filter/` — read `VendorFilterProvider`, `CategoryFilterProvider`, `EClassFilterProvider`, `EClassesFilterProvider`, `CoreSortimentFilterProvider`, `ArticleIdFilterProvider`, `ManufacturerFilterProvider`, `DeliveryTimeFilterProvider`, `FeaturesFilterProvider` for the legacy semantics. Price resolution: `PriceFilterProvider.java:56-68` (priority by `PricingType.priority`, currency match, range filter; nested with `priceListFilter` from context).

## Status

✅ **Done** — commits `ad3a361` (initial) + `d0cb6f4` (correction).

  * `search-api/filters.py` — translator with 13 filter atoms; AND-composed top-level expression
  * `search-api/prices.py` — `decode_minor_units`, `resolve_price`, `passes_price_filter` (currency two-roles split + ISO-4217 fraction-digit decoding)
  * `search-api/main.py` + `search-api/hybrid.py` — F3 wired through hybrid search path with BM25-leg post-intersection + price-filter over-fetch (default `PRICE_FILTER_OVERFETCH_N=10`)
  * Correction (`d0cb6f4`): dropped phantom `closed_catalog == true` filter and `_closed_catalog_versions` standalone filter; rewrote `_closed_marketplace` to intersect on `closedCatalogVersionIds` with `id == ""` match-nothing sentinel for the empty-list case (matches legacy `OfferFilterBuilder`).
  * Always-on legacy CV parity (this iteration): `_closed_marketplace` now switches between `closedCatalogVersionIds` (flag on) and `catalogVersionIdsOrderedByPreference` (flag off) directly inside ftsearch. Replaces the earlier "ftsearch is permissive, ACL re-adds parity" placeholder — that ACL wrapper was never built and the natural reading of `OfferFilterBuilder` puts the switch in ftsearch. `MATCH_NOTHING_EXPR` short-circuits at `main.py` / `routing.py` so empty active CV lists skip Milvus entirely.
  * eClass-hierarchy correction: `_eclass_codes` / `_eclasses_filter` / `_blocked_eclass_vendors` now emit `array_contains[_any]` against the `ARRAY<INT>` hierarchy fields. Previous `==`/`in` on collapsed scalars only matched the (undefined-ordering) first element — a silent recall bug for any non-leaf or non-first-element filter.
  * Tests: 32 unit (`test_filters.py`) + 16 unit (`test_prices.py`) + 25 integration (`test_search_filters_integration.py`).
  * Two intentional simplifications vs. legacy: priceFilter picks the single highest-priority entry (no per-tier mustNot dance); `coreSortimentOnly` with no source context is a no-op rather than mass-excluding.

## Scope

Translate every filter on the new ftsearch request into Milvus expressions (or post-Milvus filtering, where Milvus expr cannot express the predicate) so that hits returned by the search reflect every parity-critical filter. Build the price-resolution module that F4 (sort-by-price) and F5 (PRICES aggregation) will reuse.

Filters are AND-composed at the top level. Within each multi-valued filter the existing legacy semantics apply (see §4.3 table).

## In scope

- Filter translator that walks the request and emits a Milvus `expr` string. Per filter:
  - `selectedArticleSources.closedCatalogVersionIds` → `catalog_version_id in [...]` (scalar per source mongo offer post-F9 correction)
  - `articleIdsFilter` → `id in [...]`
  - `vendorIdsFilter` → `vendor_id in [...]` (single field per F1).
  - `manufacturersFilter` → `manufacturerName in [...]`
  - `maxDeliveryTime` → `delivery_time_days_max <= ...`
  - `requiredFeatures` → per `{name, values:[…]}` group: `array_contains_any(features, ["name=v1", …])`; AND across groups
  - `currentCategoryPathElements` → prefix match on the appropriate `category_l*` field; encode with `¦` separator
  - `currentEClass5Code` / `currentEClass7Code` / `currentS2ClassCode` → `array_contains(eclassN_code, X)` against the matching `ARRAY<INT>` hierarchy field; matches whether `X` is the leaf or any parent level. Legacy treats `0` as absent (no filter); the portal sends `0` as the default when no eClass is selected.
  - `eClassesFilter` → `array_contains_any(eclass5_code, [...])` (legacy `EClassesFilterProvider` operates on the eClass5 code tree; the hierarchy array carries every level so a `terms` query at any depth matches).
  - `coreSortimentOnly` → `array_contains_any(core_marker_enabled_sources, [...])` AND NOT `array_contains_any(core_marker_disabled_sources, [...])`. The source IDs come from `selectedArticleSources` — see `CoreSortimentFilterProvider.java` for the exact field set the legacy uses.
  - `closedMarketplaceOnly` → always-on `catalog_version_id in [<active list>]` (legacy `OfferFilterBuilder`; scalar `IN` against the per-source-offer column post-F9 correction). The flag selects which list drives the intersection: `closedCatalogVersionIds` when true, `catalogVersionIdsOrderedByPreference` when false. Empty active list ⇒ `terms` against `[]` matches nothing — replicated via the `id == ""` sentinel (`MATCH_NOTHING_EXPR`), short-circuited at the dispatch layer to skip the Milvus round-trip. `closedCatalogVersionIds` continues to drive the marker-source set inside `_core_sortiment_inner` (separate concern, same field — legacy quirk). The article-side mirror lands as F10 (`articles_v{N}.catalog_version_ids` ARRAY envelope, filtered with `array_contains_any`).
  - `coreArticlesVendorsFilter` → vendor-id intersected with core marker; compose from the two
  - `blockedEClassVendorsFilters` → negative expr (NOT (vendor_id in [...] AND `array_contains_any(eclassN_code, [...])`)) per entry; the hierarchy array means a vendor's row is blocked when *any* level of its eClass tree falls into the block list.
  - `accessoriesForArticleNumber` / `sparePartsForArticleNumber` / `similarToArticleNumber` → `array_contains(relationship_*, "<articleNumber>")`
- **Price filter** is special: cannot be expressed as a Milvus expr because the `prices` JSON is a per-row array that needs filtering by `currency` × `sourcePriceListIds`, then min-by-priority, then range-check. Approach:
  1. Use the Milvus expr filter for everything except `priceFilter`.
  2. Over-fetch (configurable headroom), then resolve the per-row `prices` array in Python:
     - Filter to entries matching request top-level `currency` AND `sourcePriceListId ∈ selectedArticleSources.sourcePriceListIds`.
     - Pick the highest-priority entry (use `PricingType.priority`).
     - Decode `priceFilter.min/max` (int64) into decimals using `priceFilter.currencyCode`'s ISO-4217 fraction-digit count.
     - Range-check; drop rows that don't match.
  3. **Over-fetch factor: `N = 10`** (i.e. fetch `10 × pageSize` candidates pre-price-filter), tunable via `PRICE_FILTER_OVERFETCH_N` env. Cap absolute candidates at the same hit-count safety bound F4 uses.
- **Price-resolution module** (`search-api/prices.py` or similar) exposing one helper used by F3, F4, F5: given a row's `prices` array + request currency + request priceListIds, return the resolved scalar (or `None` if no entry qualifies). F3 uses it to range-check; F4 uses it as the sort key; F5 uses it to compute min/max.
- Wire filters into the existing hybrid search path. Filters apply equally to the dense leg, the BM25 leg, and the strict-identifier path. RRF fusion runs over the already-filtered candidate sets.
- `category_l1..5` translation: build the parent prefix from the request and translate to a Milvus expression that matches the leaf level via the `¦` separator.

## Out of scope

- Sorting by these fields — F4.
- Aggregating these fields — F5.
- Filter populating data — comes from I1.

## Deliverables

- Filter translator module with unit tests covering each filter's expr.
- Price-resolution module with unit tests (currency × priceList × priority).
- Integration-ish tests against a small fixture-loaded Milvus collection demonstrating each filter narrows the hit set.

## Acceptance

- Each filter in the §4.3 table demonstrably narrows the hit set on a fixture collection.
- AND composition of two filters narrows further.
- `priceFilter` correctly handles JPY (0 fraction digits) and EUR (2) bound decoding.
- The currency two-roles split holds: top-level `currency` drives matching, `priceFilter.currencyCode` only decodes bounds.

## Open questions for this packet

(none — over-fetch factor locked at N=10 (configurable); core-sortiment source-field mapping reads from `CoreSortimentFilterProvider.java` during implementation.)
