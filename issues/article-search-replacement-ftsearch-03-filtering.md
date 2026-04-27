# F3 — Scalar filtering + price-resolution module

**Category**: ftsearch (`./search-api/`)
**Depends on**: F1 (schema), F2 (request contract)
**Unblocks**: F4, F5, A2, A6

References: spec §4.3, §3 (currency two-roles), §7.

## Scope

Translate every filter on the new ftsearch request into Milvus expressions (or post-Milvus filtering, where Milvus expr cannot express the predicate) so that hits returned by the search reflect every parity-critical filter. Build the price-resolution module that F4 (sort-by-price) and F5 (PRICES aggregation) will reuse.

Filters are AND-composed at the top level. Within each multi-valued filter the existing legacy semantics apply (see §4.3 table).

## In scope

- Filter translator that walks the request and emits a Milvus `expr` string. Per filter:
  - `selectedArticleSources.closedCatalogVersionIds` → `array_contains_any(catalog_version_ids, [...])`
  - `articleIdsFilter` → `id in [...]`
  - `vendorIdsFilter` → `vendor_id in [...]` (single field) or `array_contains_any(vendor_ids, [...])` (array — confirm with F1)
  - `manufacturersFilter` → `manufacturerName in [...]`
  - `maxDeliveryTime` → `delivery_time_days_max <= ...`
  - `requiredFeatures` → per `{name, values:[…]}` group: `array_contains_any(features, ["name=v1", …])`; AND across groups
  - `currentCategoryPathElements` → prefix match on the appropriate `category_l*` field; encode with `¦` separator
  - `currentEClass5Code` / `currentEClass7Code` / `currentS2ClassCode` → equality on the matching int field
  - `eClassesFilter` → `eclass5_code in [...]` (or whichever eClass version is implied — confirm)
  - `coreSortimentOnly` → `array_contains_any(core_marker_enabled_sources, [...])` AND NOT `array_contains_any(core_marker_disabled_sources, [...])` (resolve via the request's source IDs — see open question)
  - `closedMarketplaceOnly` → `closed_catalog == true`
  - `coreArticlesVendorsFilter` → vendor-id intersected with core marker; compose from the two
  - `blockedEClassVendorsFilters` → negative expr (NOT (vendor_id in [...] AND eclassN_code in [...])) per entry
  - `accessoriesForArticleNumber` / `sparePartsForArticleNumber` / `similarToArticleNumber` → `array_contains(relationship_*, "<articleNumber>")`
- **Price filter** is special: cannot be expressed as a Milvus expr because the `prices` JSON is a per-row array that needs filtering by `currency` × `sourcePriceListIds`, then min-by-priority, then range-check. Approach:
  1. Use the Milvus expr filter for everything except `priceFilter`.
  2. Over-fetch (configurable headroom), then resolve the per-row `prices` array in Python:
     - Filter to entries matching request top-level `currency` AND `sourcePriceListId ∈ selectedArticleSources.sourcePriceListIds`.
     - Pick the highest-priority entry (use `PricingType.priority`).
     - Decode `priceFilter.min/max` (int64) into decimals using `priceFilter.currencyCode`'s ISO-4217 fraction-digit count.
     - Range-check; drop rows that don't match.
  3. Document the over-fetch headroom and the cap (so it's tunable via env).
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

- "Core sortiment only" semantics: the legacy expression is `enabled markers ∩ request sources, MINUS disabled markers ∩ request sources`. Confirm the exact request fields that supply the source IDs (likely `selectedArticleSources.customerUploadedCoreArticleListSourceIds` and friends — verify against legacy code before implementing).
- Over-fetch headroom for the priceFilter resolution: what cap is acceptable when many rows don't survive the currency × priceList filter? Pick a budget aligned with the §4.7 latency SLO.
