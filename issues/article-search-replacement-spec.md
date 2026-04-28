# Legacy Search → ftsearch Replacement: Complete Spec

**Scope:** `POST /article-features/search` only. The other endpoints under `/article-features/*` and `/article-sources/*` are out of scope for this iteration.

## Terminology

- **next-gen** — the customer-facing application (frontend + backend). Hosts the existing search controller and its callers.
- **legacy search** — the current `/article-features/search` endpoint inside next-gen, backed by Elasticsearch with German-tuned analyzers, deep aggregations, and multi-source filtering.
- **ftsearch** ("fine-tuned search") — the new search service. Status quo: FastAPI → Milvus (HNSW + BM25), ranking logic in the FastAPI layer. Exposes a single hybrid-search call today; will be extended with filtering, faceting, and non-relevance sorting.
- **ACL** — a thin anti-corruption layer service that exposes a narrowed derivative of legacy search's OpenAPI (see §2 Deviations) and translates each request into one or more ftsearch calls. Holds no business logic of its own.

## TL;DR

The replacement is implemented by extending ftsearch (FastAPI + Milvus) with the filtering, faceting, sorting, and metadata-join logic legacy search owns today, then placing a thin ACL service in front of ftsearch that speaks a *narrowed* derivative of legacy search's OpenAPI. Conscious deviations are encoded in the schema itself (see §2) rather than papered over.

Next-gen integration therefore requires two things:
1. Reroute the legacy-search base URL to the ACL.
2. Regenerate the next-gen search client from the updated OpenAPI and clean up any callers passing values that have been dropped from the contract.

This is no longer a literal drop-in replacement — but every deviation is documented, surfaces at compile time in next-gen, and falls into one of three categories: requirement no longer relevant, complexity not worth carrying, or Milvus / approach can't support the exact behavior cleanly.

---

## 1. Integration Architecture

```
┌─────────────────┐    unchanged HTTP     ┌──────────────┐   HTTP   ┌─────────────────────┐
│ next-gen client │ ────────────────────► │  ACL service │ ───────► │ ftsearch (FastAPI)  │
│ (Java/frontend) │  POST /article-       │  (thin DTO   │          │  ranking · filters  │
│                 │   features/search     │   mapping)   │          │  facets · sort      │
└─────────────────┘                       └──────────────┘          └──────────┬──────────┘
                                                                               │
                                                                               ▼
                                                                        ┌────────────┐
                                                                        │   Milvus   │
                                                                        │ HNSW+BM25  │
                                                                        └────────────┘
```

**ACL responsibilities (thin, by design):**
- Expose a narrowed derivative of `api-spec/specs/article-search/*.yaml` — same paths, same error envelope, DTOs trimmed where §2 records a deviation.
- Translate each legacy request into one or more ftsearch calls and assemble the response.
- Forward request-scoped baggage (tracing/observability only — `userId`, `companyId`, `customerOciSessionId`, W3C traceparent) onto ftsearch.
- Return shaped errors in the legacy `{message, details, timestamp}` envelope.

**ftsearch responsibilities (everything else):**
- Hybrid relevance ranking (already in place).
- Scalar filtering, hierarchical-category filtering, vendor/manufacturer filters, price/feature filters — implemented in the FastAPI layer over Milvus expr filters. No auxiliary stores at query time (§6).
- Faceting / `summaries.*` aggregation.
- Sorting beyond relevance (name, price, articleId).
- Hit-count semantics. (`explain` is honored as a stub returning `"N/A"` — §2.2.)
- Identifier-query tokenization (article number / EAN / vendor SKU).
- Catalog visibility scoping — request-scoped via `selectedArticleSources` fields, not tenant-authenticated (the service is internal; see §9 #7).

Next-gen integration: base-URL config flip + a one-time client regeneration against the updated OpenAPI (see §2 for the deviations that drive Java-side cleanup).

---

## 2. Deviations from Legacy Contract

This section is the canonical list of every place the new stack intentionally diverges from legacy search's behavior. **Any divergence not listed here is a bug.** Each deviation is encoded into the ACL's OpenAPI — the schema is the single source of truth for what's accepted; runtime behavior matches.

The reasons for not supporting certain behavior are:
- **Requirement no longer relevant** — the legacy behavior exists but isn't needed going forward.
- **Complexity reduction** — implementable, but cost is high relative to value.

For each entry: the category, what changes on the wire (OpenAPI delta), runtime behavior, and what next-gen callers need to do.

### 2.1 `searchArticlesBy` collapses to `STANDARD` only

**OpenAPI delta.** The `searchArticlesBy` enum shrinks from

```
ALL_ATTRIBUTES | ARTICLE_NUMBER | CUSTOMER_ARTICLE_NUMBER | VENDOR_ARTICLE_NUMBER | EAN | STANDARD | TEST_PROFILE_01..20
```

to

```
STANDARD
```

The field stays in the schema (still required, single-value enum) so the request shape is unchanged otherwise. A future deviation may drop the field entirely.

**Runtime behavior.** All requests are served by ftsearch's `classified hybrid` strategy. The classifier inside ftsearch decides per query whether to lean on the dense vector or the BM25 codes leg — there is no externally selectable mode. Requests carrying any other enum value (stale clients) are rejected with HTTP 400 in the legacy envelope:

```jsonc
{
  "message": "Validation failure",
  "details": [
    {
      "field": "searchArticlesBy",
      "message": "must be 'STANDARD'"
    }
  ],
  "timestamp": "2026-04-27T11:10:57.849+00:00"
}
```

The ACL validates against its OpenAPI; ftsearch validates again as defense-in-depth.

**Functional consequences.** Several legacy use cases lose their dedicated mode and fall through to classified hybrid:

- *EAN, article-number, vendor-article-number search* — handled by classified hybrid via the strict-identifier classifier; should remain serviceable but quality is now a property of the classifier rather than a guarantee.
- *Customer-article-number search* — depended on a tenant-scoped lookup keyed off `customerArticleNumbersIndexingSourceIds`. **Dropped.** Classified hybrid does not have this lookup and we are not building one; the feature loss is accepted as part of removing the `CUSTOMER_ARTICLE_NUMBER` mode.
- *ALL_ATTRIBUTES, TEST_PROFILE_01..20* — no remaining users expected; these go away cleanly.

**Next-gen migration.** Regenerating the search client from the new OpenAPI removes the dropped enum constants from the generated Java enum, turning every caller that still references them into a compile error. All such callers should switch to `STANDARD`. Callers that relied on customer-article-number search semantically lose that capability (see above) — there is no replacement.

### 2.2 `explain` returns `"N/A"` instead of a score breakdown — (M)

**OpenAPI delta.** None. Request `explain: boolean` and response `articles[].explanation: string?` stay in the schema verbatim.

**Runtime behavior.** When `explain: true`, every hit's `explanation` is set to the literal string `"N/A"`. When `explain: false` (or omitted), `explanation` is `null` / absent — same as legacy. ftsearch is not asked to synthesize anything ranking-meaningful; the placeholder lives in the ACL.

**Why a stub instead of a real string.** Producing a Lucene-equivalent score breakdown for a hybrid Milvus result has no clean translation (dense + BM25 + RRF doesn't decompose the way Lucene's tree does). Audit of the only consumer (`frontend/.../ExplanationButton.tsx`) shows the field is treated as opaque text — clicking the icon downloads it as `search_result_calculations_<timestamp>.txt`. Returning `"N/A"` keeps the consumer's "string is present" expectation intact and the download still works; the file just contains `N/A`. Revisit later if a useful explanation string is wanted.

**Functional consequences.** The diagnostic download exists but no longer contains useful information. No code path breaks.

### 2.3 Future deviations

Additional deviations will be added here as they are decided — particularly around fields that depend on data ftsearch doesn't carry today (price-sourced filters, eClass hierarchies). When in doubt during implementation: prefer recording a deviation here over silently degrading behavior.

---

## 3. The Contract the ACL MUST Implement (Verbatim)

`POST /article-features/search`, JSON in/out, no authentication on either hop (next-gen → ACL or ACL → ftsearch); both endpoints expose `security: []`. Default app port: **8081**, actuators: **9090**.

**Query params**: `page` (1-indexed, default 1), `pageSize` (default 10, **max 500**), `sort` (repeatable; values like `articleId,desc`, `name,asc`, `price,desc`).

**Request body — required fields**: `searchMode`, `searchArticlesBy`, `selectedArticleSources`, `maxDeliveryTime`, `coreSortimentOnly`, `closedMarketplaceOnly`, `currency` (`^[A-Z]{3}$`), `explain`.

**Currency fields — two roles, both consulted (legacy parity).** The top-level `currency` and `priceFilter.currencyCode` are *not* duplicates:
- **Top-level `currency`** pins which `prices.*` entries qualify for the price filter, sort-by-price, and the `PRICES` aggregation. It governs the match.
- **`priceFilter.currencyCode`** is consumed only to decode the integer `priceFilter.min`/`max` into a decimal amount via that currency's default fraction-digit count (EUR → 2 decimals: `1500` → `15.00`; JPY → 0 decimals: `1500` → `1500`). Required (non-null) whenever `min` or `max` is set; never used to match `prices.currency`.

The two are independent on the wire. In practice the frontend sends them matched, but legacy does *not* enforce equality and the ACL/ftsearch must not either — top-level currency drives matching, `priceFilter.currencyCode` drives bound-decoding, and that is what gets forwarded to ftsearch.

**Full request body** (all fields the ACL must accept and honor, even when ignored):

```jsonc
{
  "searchMode": "HITS_ONLY|SUMMARIES_ONLY|BOTH",
  "searchArticlesBy": "STANDARD",  // single-value enum per §2.1
  "selectedArticleSources": {
    "closedCatalogVersionIds": ["uuid", ...],            // required
    "catalogVersionIdsOrderedByPreference": ["uuid", ...],
    "sourcePriceListIds": ["uuid", ...],
    "customerArticleNumbersIndexingSourceIds": ["uuid", ...],
    "customerUploadedCoreArticleListSourceIds": ["uuid", ...],
    "customerManagedArticleNumberListId": "uuid?",
    "uiCustomerArticleNumberSourceId": "uuid?"
  },
  "queryString": "string?",
  "articleIdsFilter": ["articleId", ...],
  "vendorIdsFilter": ["uuid", ...],
  "manufacturersFilter": ["string", ...],
  "maxDeliveryTime": 0,
  "requiredFeatures": [{"name": "...", "values": ["..."]}],
  "priceFilter": {"min": 0, "max": 999999, "currencyCode": "EUR"},
  "accessoriesForArticleNumber": "string?",
  "sparePartsForArticleNumber": "string?",
  "similarToArticleNumber": "string?",
  "currentCategoryPathElements": ["...", "..."],
  "currentEClass5Code": 31000000,
  "currentEClass7Code": 31010001,
  "currentS2ClassCode": 31010001,
  "coreSortimentOnly": false,
  "closedMarketplaceOnly": false,
  "summaries": ["CATEGORIES","ECLASS5","ECLASS7","S2CLASS","VENDORS","MANUFACTURERS","FEATURES","PRICES","PLATFORM_CATEGORIES","ECLASS5SET"],
  "coreArticlesVendorsFilter": ["uuid", ...],
  "blockedEClassVendorsFilters": [
    {
      "vendorIds": ["uuid", ...],
      "eClassVersion": "ECLASS_5_1|ECLASS_7_1|S2CLASS",
      "blockedEClassGroups": [{"eClassGroupCode": 12345, "value": true}]
    }
  ],
  "currency": "EUR",
  "explain": false,
  "eClassesFilter": [123456, ...],
  "eClassesAggregations": [{"id": "agg-id", "eClasses": [123456, ...]}],
  "s2ClassForProductCategories": false
}
```

**Full response body**:

```jsonc
{
  "articles": [{"articleId": "string", "explanation": "string?"}],  // explanation = "N/A" stub when explain=true; null otherwise — §2.2
  "summaries": {
    "vendorSummaries":       [{"vendorId": "uuid", "count": 0}],
    "manufacturerSummaries": [{"name": "string",  "count": 0}],
    "featureSummaries":      [{"name": "string",  "count": 0, "values": [{"value": "string", "count": 0}]}],
    "pricesSummary":         [{"min": 0.0, "max": 0.0, "currencyCode": "EUR"}],
    "categoriesSummary": {
      "currentCategoryPathElements": ["..."],
      "sameLevel": [{"categoryPathElements": ["..."], "count": 0}],
      "children":  [{"categoryPathElements": ["..."], "count": 0}]
    },
    "eClass5Categories": {
      "selectedEClassGroup": 31000000,
      "sameLevel": [{"group": 31010000, "count": 0}],
      "children":  [{"group": 31010001, "count": 0}]
    },
    "eClass7Categories": null,
    "s2ClassCategories": null,
    "eClassesAggregations": [{"id": "agg-id", "count": 0}]
  },
  "metadata": {"page": 1, "pageSize": 10, "pageCount": 5, "term": "...", "hitCount": 47}
}
```

**Mode rules** (must replicate exactly):
- `HITS_ONLY` → `summaries` empty/omitted.
- `BOTH` → both populated; aggregations computed over the full filtered hit set, articles paginated.
- `SUMMARIES_ONLY` → `articles` empty, but aggregations computed against the full filtered hit set with effective `pageSize=0`.
- `articleId` format is `{friendlyId}:{base64UrlEncodedArticleNumber}` — **the wire value must match legacy exactly**; downstream code parses it. The indexer is responsible for projecting (or constructing) this composite into Milvus' PK so ftsearch returns and accepts the legacy value verbatim.

### 3.1 Error contract (must match exactly)

```jsonc
// 400 Bad Request body
{
  "message": "Validation failure",
  "details": [{"field": "currency", "message": "must match \"^[A-Z]{3}$\""}],
  "timestamp": "2021-10-25T11:10:57.849+00:00"
}
```

| Trigger              | Status | Message                                       |
| -------------------- | ------ | --------------------------------------------- |
| Bean validation fail | 400    | "Validation failure" + per-field details      |
| Constraint violation | 400    | "Constraint violation" + propertyPath details |
| JSON parse failure   | 400    | exception message                             |
| Anything else        | 500    | "Internal server error"                       |

---

## 4. Behavioral Parity Requirements (the "what must work the same")

### 4.1 Search mode

The only supported value is `STANDARD` (see §2.1). It is served by ftsearch's `classified hybrid` strategy: dense vector + sparse BM25 fused via RRF, with a classifier that detects strict-identifier queries (numeric EANs, SKU-style tokens) and biases the fusion toward the BM25 leg for those.

There is no per-mode routing for the ACL to perform — `searchArticlesBy: STANDARD` goes in, classified hybrid runs, results come back. Identifier-vs-text behavior is a property of the classifier, configured inside ftsearch.

### 4.2 Sorting

The ACL passes `sort` query params to ftsearch unchanged. ftsearch owns the implementation.

| Sort                               | Today (legacy)                             | ftsearch responsibility                                                |
| ---------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------- |
| Relevance (default, sort omitted)  | ES `_score` desc, name asc                 | Use Milvus score; tiebreak on `articleId`                              |
| `name,asc` / `name,desc`           | Nested sort on `offers.name` (root keyword + `sort_normalizer`), mode=Min | Sort using a denormalized scalar in Milvus or re-sort top-K internally |
| `price,asc` / `price,desc`         | Nested sort on `prices.price`, mode=Min    | Per row, take `min(price)` over `prices` filtered by request `currency` × `sourcePriceListIds` (same scope as the priceFilter); over-fetch and post-sort top-K |
| `articleId,asc` / `articleId,desc` | Native field sort                          | Sort by Milvus PK / id                                                 |

For sorts ftsearch implements via re-sort, it must over-fetch from Milvus (e.g. `k = pageSize × pages_to_cover`) and paginate after sorting. The ACL never sees this — it just receives the already-paginated page from ftsearch.

### 4.3 Filtering — every filter must work

The ACL forwards filters as structured fields on the ftsearch request. ftsearch translates them into Milvus expressions and any auxiliary lookups.

| Filter                                                                                  | Source field today (legacy)                                                         | ftsearch implementation hint                                                                          |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `selectedArticleSources.closedCatalogVersionIds`                                        | nested `offers.catalogVersionIds`                                                   | Milvus expr filter (`array_contains_any(catalog_version_ids, [...])`) — already present in collection |
| `catalogVersionIdsOrderedByPreference`                                                  | same, but order matters for which offer wins                                        | Post-process / pick min by preference                                                                 |
| `sourcePriceListIds`                                                                    | nested `prices.priceListId` term filter                                             | Filter the row's `prices` JSON array by `sourcePriceListId ∈ request.sourcePriceListIds`              |
| `articleIdsFilter`                                                                      | terms on `articleId`                                                                | Milvus expr `id in [...]`                                                                             |
| `vendorIdsFilter`                                                                       | terms on `vendorId`                                                                 | Need vendor field in collection (currently only manufacturerName)                                     |
| `manufacturersFilter`                                                                   | terms on nested `offers.manufacturerName`                                           | Milvus expr on `manufacturerName` — already present                                                   |
| `maxDeliveryTime`                                                                       | numeric range                                                                       | Add to collection (denormalized scalar)                                                               |
| `requiredFeatures`                                                                      | nested terms on `features.name`/`values`                                            | Per `{name, values:[…]}` group, `array_contains_any(features, ["name=v1","name=v2",…])`; AND across groups (see §7 for the `name=value` token shape) |
| `priceFilter`                                                                           | nested numeric range on `prices.price` after currency × priceList × priority scope  | Resolve from `prices` array: filter by request `currency` × `sourcePriceListIds`, take the highest-priority entry, range-check against `priceFilter.min/max`. Decode `min`/`max` (int64) into decimals using `priceFilter.currencyCode`'s fraction-digits (EUR→2, JPY→0); `priceFilter.currencyCode` does **not** participate in matching `prices.currency` — top-level `currency` does (§3, legacy parity). |
| `accessoriesForArticleNumber` / `sparePartsForArticleNumber` / `similarToArticleNumber` | nested `offers.accessoryFor`/`sparePartFor`/`similarTo` term                        | Add relationship arrays to collection                                                                 |
| `currentCategoryPathElements`                                                           | path prefix match                                                                   | Milvus `category_l1..l5` already there but separator is `¦`; needs translation                        |
| `currentEClass5Code` / `currentEClass7Code` / `currentS2ClassCode`                      | terms on eClass group fields                                                        | Add eClass fields to collection                                                                       |
| `coreSortimentOnly`                                                                     | terms on `enabledCoreArticleMarkerSources` minus `disabledCoreArticleMarkerSources` | Add core-marker fields to collection                                                                  |
| `closedMarketplaceOnly`                                                                 | switches the CV-list source: `closedCatalogVersionIds` when true, otherwise `catalogVersionIdsOrderedByPreference` | When true, intersect `catalog_version_ids` with `selectedArticleSources.closedCatalogVersionIds`; empty list ⇒ match nothing (legacy `OfferFilterBuilder`). No row-level flag exists. |
| `coreArticlesVendorsFilter`                                                             | vendor + core marker combo                                                          | Same as above                                                                                         |
| `blockedEClassVendorsFilters`                                                           | inverse filter (vendorId × eClassGroup)                                             | Negative Milvus expr once eClass + vendor are in the collection                                       |

**Net consequence**: The Milvus collection schema is **insufficient** for parity. ftsearch must extend the collection to include the filterable scalar fields (§7). All parity-critical data is row-local in Milvus; auxiliary lookups at query time are out of scope (§6).

### 4.4 Aggregations / facets

All 10 summary kinds must be computable. The ACL forwards the requested `summaries` list; ftsearch produces the counts. Milvus' `group_by` is limited; viable strategies per summary:

| Summary                           | Strategy in ftsearch                                                                                                        |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `VENDORS`                         | Hit-set vendor IDs → Milvus `group_by_field=vendor_id`, or post-aggregate in FastAPI                                        |
| `MANUFACTURERS`                   | Same on `manufacturer_name`                                                                                                 |
| `FEATURES`                        | Requires features in collection; count per `name=value` token                                                               |
| `PRICES`                          | For each hit, filter the row's `prices` array by request `currency` × `sourcePriceListIds`, then `MIN`/`MAX` of `price` over the union, grouped by `currency`. In practice this returns one entry per request currency (legacy filter pins currency); the array shape on the wire is preserved. |
| `CATEGORIES`                      | Hierarchical; needs `category_l1..l5`. Compute `sameLevel` and `children` from `currentCategoryPathElements` parent prefix. |
| `ECLASS5` / `ECLASS7` / `S2CLASS` | Same hierarchical pattern on eClass codes                                                                                   |
| `PLATFORM_CATEGORIES`             | Alias of `CATEGORIES` (or `S2CLASS` if `s2ClassForProductCategories: true`)                                                 |
| `ECLASS5SET`                      | For each request-supplied `eClassesAggregations[]` entry, count hits whose eClass5 ∈ entry.eClasses                         |

ES today computes hits + summaries in one round-trip; ftsearch will need 2–3 internal Milvus calls but should still expose a single `/search` call to the ACL.

**`SUMMARIES_ONLY` and `BOTH` modes**: aggregations are computed over the *full filtered hit set*, not just one page. ftsearch must use a high `k` (e.g. `k = max-page-size × page-budget`, capped reasonably) and aggregate over all returned IDs before returning. `SUMMARIES_ONLY` skips article hydration entirely (returns empty `articles[]`).

### 4.5 German-language search behavior

`settings-articles.json` defines >1000 lines of analyzers: `german_stemmed`, `german_text`, `german_text_decompounded`, `german_strict`, `ean_search`, plus n-gram analyzers for article numbers. `useful-cub-58` (the fine-tuned E5 variant) absorbs most of this for free-text query — but **identifier-style queries** (article numbers, EANs, vendor SKUs) bypass the embedding. ftsearch today applies `query.lower()` only.

With `STANDARD` as the only mode, identifier-vs-text routing happens entirely inside ftsearch's classifier — there is no caller-driven mode that pre-selects the path.

**Required additions in ftsearch (FastAPI layer)**:
- Tighten the `is_strict_identifier` classifier in `hybrid.py` so it reliably catches numeric EANs and SKU-style tokens; extend the denylist for German-specific generic tokens (e.g. "Bohrer", "Schrauben") to avoid misclassification.
- For identifier-classified queries, apply `pattern_replace` + n-gram tokenization equivalent to legacy's `article_number_normalized_*` analyzer before BM25, so partial-match behavior carries over.
- For text-classified queries, decompound German compounds before passing to the BM25 leg, so "Akkubohrmaschine" matches articles indexed as "Akku Bohrmaschine".

### 4.6 Relevance/quality contract

Today (legacy): `DFS_QUERY_THEN_FETCH` for cross-shard scoring consistency, `track_total_hits=true` for accurate `hitCount`, optional `explain` payload per hit.

Required from ftsearch (the ACL just passes through):
- **Accurate hit count.** `metadata.hitCount` is consumed by frontend pagination. Milvus' top-K returns no total; ftsearch must run a separate `count(*)`-style query (Milvus `query` with output_fields=[id] + scalar filter) or over-fetch up to a safety cap.
- **`explain: true`** must produce a non-null `explanation` string per article. Per §2.2 the ACL fills this with the literal `"N/A"`; ftsearch is not involved.
- **DFS-equivalent stability** — Milvus single-node has no shard scoring drift, so this is automatic.

### 4.7 Performance & resilience

The ACL adds one network hop on top of legacy search; ftsearch carries the heavy lifting.

| Aspect           | Current (legacy)                                                           | ACL                                                       | ftsearch                                                                  |
| ---------------- | -------------------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------- |
| Retries          | Spring Retry, max 5, 500ms base, 1.5× multiplier, max 5s delay             | Retry on transient ftsearch failures with same policy     | Internal retries on Milvus + auxiliary calls                              |
| Timeouts         | Implicit Spring MVC                                                        | Explicit ftsearch call timeout                            | Explicit per-call timeouts; total request budget ≤ existing p99           |
| Caching          | None app-side; ES caches                                                   | None (stateless mapper)                                   | Optional caching for category/eClass hierarchies and identifier lookups   |
| Tracing          | Sleuth, 100% sampling, baggage `userId`/`companyId`/`customerOciSessionId` | Accept & forward W3C traceparent + baggage to ftsearch    | Propagate to Milvus calls                                                 |
| Metrics          | Micrometer p50/p70/p95/p99 + Prometheus                                    | ACL exposes its own RED metrics                           | Already has prometheus-fastapi-instrumentator                             |
| Health           | `/actuator/health` on :9090                                                | `/healthz` (and Spring-compatible actuator if needed)     | `/healthz`                                                                |
| Refresh          | ES 30s                                                                     | n/a                                                       | `Bounded` consistency (§9 #8) — matches legacy's seconds-of-staleness tolerance |

### 4.8 Index-versioning / zero-downtime contract

Spring app reads `@indexProperties.getName()` so deploys can swing the index alias. Milvus equivalent: collection aliasing (`MilvusClient.alter_alias`). ftsearch's config must accept `MILVUS_COLLECTION_ALIAS` and never embed a hard collection name. The ACL is unaware of the Milvus collection name.

---

## 5. Gap Analysis — what is missing today

What ftsearch offers today vs. what's needed for parity. Unless noted, the gap is closed in ftsearch (FastAPI + Milvus collection); ACL gaps are called out explicitly.

| Capability                                                                      | ftsearch today                           | Gap to fill                                                    |
| ------------------------------------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------- |
| Hybrid relevance ranking                                                        | ✅ dense + BM25 + RRF + classifier       | None for `STANDARD` mode                                       |
| Article-by-articleId fetch                                                      | ✅ via Milvus query                      | Indexer projects legacy composite into Milvus PK (no deviation) |
| Filter by `manufacturerName`                                                    | ✅ field exists                          | Wire into ftsearch request schema                              |
| Filter by category L1–L5                                                        | ✅ field exists, `¦` separator           | ftsearch translates path into prefix match                     |
| Filter by `vendorId`                                                            | ❌ field absent                          | Add to Milvus collection schema                                |
| Search by EAN / article number / vendor SKU (formerly dedicated modes)          | ⚠️ via BM25 codes collection             | Routed by classified hybrid's strict-identifier classifier (§4.5) |
| Filter by price                                                                 | ❌ no price field                        | Project the legacy nested `prices` array into a row-local JSON field; resolve currency × priceList × priority at query time |
| Filter by features                                                              | ❌                                       | Add `features ARRAY<VARCHAR>` of `name=value` tokens to collection (§7) |
| Filter by delivery time / core sortiment / closed marketplace / blocked vendors | ❌                                       | Add fields to collection                                       |
| Filter by eClass5 / eClass7 / S2Class                                           | ❌                                       | Add fields to collection                                       |
| Search by customer article number (formerly `CUSTOMER_ARTICLE_NUMBER` mode)     | ❌                                       | Dropped — feature loss accepted with the mode removal (§2.1)   |
| Multi-currency price aggregation                                                | ❌                                       | Aggregate min/max within the row's `prices` array filtered by request `currency` × `sourcePriceListIds`; group by currency on the wire (one entry per request in practice) |
| All `summaries.*`                                                               | ❌                                       | Implement in ftsearch FastAPI layer                            |
| Sorting by name / price                                                         | ❌                                       | Implement in ftsearch (denormalize or post-sort top-K)         |
| Sort `name` `price` `articleId` (asc/desc)                                      | ❌                                       | Same                                                           |
| `explain: true` payload                                                         | ⚠️ has `_debug` per hit if `debug=1`     | ACL stubs `"N/A"` per hit (§2.2); ftsearch unchanged           |
| `searchMode: SUMMARIES_ONLY` (page-size-zero behavior)                          | ❌                                       | ftsearch honors mode; aggregations over full filtered hit set  |
| Validation errors as `{message, details, timestamp}`                            | ❌ FastAPI default 422                   | Custom error handler in **ACL** (legacy envelope)              |
| Catalog version scoping (request-scoped, not tenant-authenticated)              | ⚠️ field present, not enforced           | ftsearch enforces `selectedArticleSources.*` in Milvus expr    |
| German analyzer parity for identifier queries                                   | ⚠️ `query.lower()` only                  | ftsearch pre-tokenizes identifier formats                      |
| Retries with exponential backoff                                                | ❌                                       | ACL retries on ftsearch transient failures                     |
| `articleId` composite format                                                    | ❌ today's `id` is opaque per-collection | Indexer projects legacy `{friendlyId}:{base64Url(articleNumber)}` into Milvus' PK so the wire value matches legacy verbatim |

---

## 6. Indexing Pipeline & Data Sources

**MongoDB is the source of truth.** At query time, ftsearch reads from Milvus only — no external lookups, no joins, no MongoDB calls on the hot path. The ACL is stateless and consumes only ftsearch. All parity-critical data is denormalized into Milvus collection scalars by an indexing pipeline.

**Indexing pipeline:**
- **Incremental updates** — A Kafka topic publishes change notifications carrying record IDs. The indexer consumes these IDs, fetches the current record from MongoDB, projects the parity-critical fields, and upserts the corresponding Milvus row.
- **Bulk (re)import** — A full rebuild reads all relevant records from MongoDB and writes them into a fresh Milvus collection (used for first-time hydration, schema migrations, and zero-downtime reindex per §4.8).

**Data the indexer must project from MongoDB into Milvus** (the exact field set is in §7):

1. **Article metadata** — for sort-by-name, sort-by-articleId, feature aggregation.
2. **Prices** — full legacy nested-prices set: `(price, currency, priority, sourcePriceListId)` per nested entry, many per article. Projected verbatim into the row's `prices` JSON array (§7). The price selected for filter/sort/aggregation is resolved at query time from the request's `currency` × `sourcePriceListIds` × `PricingType.priority`, mirroring the legacy ES nested filter — there is no single denormalized scalar to project.
3. ~~**Customer article numbers**~~ — not projected. The `CUSTOMER_ARTICLE_NUMBER` mode is dropped (§2.1); the lookup it depended on is not carried over.
4. **Catalog version metadata** — to know which catalogs are "closed" and to enforce visibility.
5. **eClass / S2Class hierarchy** — for hierarchical aggregations and filtering.
6. **Vendor / manufacturer registry** — for vendor-side filters and core-sortiment markers.

---

## 7. Required New Milvus Collection Schema

Per §6, every query-time lookup must be served from Milvus. Expand the dense `offers` collection so the indexer can project all parity-critical fields into row-local scalars:

```text
id                            STRING (legacy composite `{friendlyId}:{base64Url(articleNumber)}` — projected by the indexer; wire value matches legacy)
offer_embedding               FLOAT_VECTOR(dim) — exists
name                          STRING — exists
manufacturerName              STRING — exists
ean                           STRING — exists
article_number                STRING — exists
catalog_version_ids           ARRAY<STRING> — exists
category_l1..l5               STRING (¦-separated paths) — exists

# NEW
vendor_id                     STRING
prices                        JSON — full legacy nested-prices array projected verbatim:
                                     [{"price": float, "currency": "EUR", "priority": int, "sourcePriceListId": "uuid"}, ...]
                                     ftsearch resolves the request's currency × sourcePriceListIds × priority at query time
                                     (identical to legacy ES nested filter + SortMode.Min). Single scalars don't suffice:
                                     an article carries multiple prices across currencies and price lists, and the chosen
                                     price depends on per-request scope.
delivery_time_days_max        INT
core_marker_enabled_sources   ARRAY<STRING>
core_marker_disabled_sources  ARRAY<STRING>
eclass5_code                  ARRAY<INT> — every level of the legacy hierarchy (root → leaf), mirroring
                                     ES `offers.eclass51Groups`. Filters use `array_contains[_any]`
                                     so a `terms` query at any level matches (single INT collapsed
                                     the hierarchy to one undefined-ordering scalar — silent recall bug).
eclass7_code                  ARRAY<INT> — same shape; mirrors ES `offers.eclass71Groups`.
s2class_code                  ARRAY<INT> — same shape; mirrors ES `offers.s2classGroups`.
features                      ARRAY<VARCHAR> of "name=value" tokens, separator `=`.
                                     Indexer must reject (or escape) values that contain `=` so the
                                     separator stays unambiguous; chosen over JSON because Milvus
                                     ARRAY membership expr (`array_contains_any`) is mature and
                                     matches `requiredFeatures`'s "AND across names, OR within values"
                                     shape directly, and FEATURES aggregation post-processes tokens
                                     in FastAPI either way (Milvus `group_by_field` doesn't traverse
                                     JSON paths or array elements per-key/per-value).
relationship_accessory_for    ARRAY<STRING>
relationship_spare_part_for   ARRAY<STRING>
relationship_similar_to       ARRAY<STRING>
```

(`closedMarketplaceOnly` does not need a row-level flag — legacy filters
by intersecting `catalog_version_ids` with the request's
`closedCatalogVersionIds` list, see §4.3.)

This puts every parity-critical filter into the search call with no external lookups. Aggregations beyond hierarchical category/eClass can run as Milvus `query` (no vector) with `group_by_field` on the filtered set.

---

## 8. Translation Examples (cookbook)

These show ACL → ftsearch translations. The mechanics inside ftsearch (which Milvus call, expr filters, post-sort) are illustrative — ftsearch owns the implementation; the ACL just builds the request.

**A.** Free-text query `"Bohrmaschine"` → ACL forwards `{"query": "Bohrmaschine", "searchArticlesBy": "STANDARD", ...}`. ftsearch's classifier reads it as text, runs hybrid (dense + BM25), fuses via RRF. Map response `hits[]._id` → `articleId`; `_score` is dropped. If the request had `explain: true`, the ACL fills `explanation = "N/A"` on each hit (§2.2).

**B.** Identifier-style query `"4006381333931"` → same request shape (`searchArticlesBy: STANDARD`). ftsearch's classifier flags it as a numeric identifier and biases fusion toward the BM25 codes leg. No caller-side mode change needed.

**C.** `currentCategoryPathElements=["Werkzeug","Akkubohrer"]` → ACL forwards as a structured field. ftsearch builds Milvus expr `array_contains(category_l2, "Werkzeug¦Akkubohrer")` after the collection-schema migration. (`category_l{N}` stores the joined prefix at depth N as ARRAY<VARCHAR>; Milvus 2.6 rejects `LIKE` on array fields, and exact element-membership is the right operator anyway since the depth disambiguates which `category_l{N}` to filter on.)

**D.** Sort `name,asc`, page 5 of pageSize 50 → ACL forwards `sort=name,asc`, `page=5`, `pageSize=50`. ftsearch over-fetches (`k=500`), looks up names from its metadata source, sorts, and returns slice `[200:250]`.

**E.** `summaries: ["VENDORS"]`, `searchMode: SUMMARIES_ONLY` → ACL forwards. ftsearch runs Milvus query with `k=trackTotalHitsCap` (e.g. 10000), groups by `vendor_id`, returns `summaries.vendorSummaries` and empty `articles[]`.

---

## 9. Decisions log

These were originally open questions; all are now resolved. Each links to where the decision is encoded in the spec.

1. **Milvus schema vs. auxiliary stores** — Extend the Milvus collection (§7). All parity-critical fields are denormalized into row-local scalars by the indexer; ftsearch performs no auxiliary lookups at query time (§6).
2. **`searchMode: SUMMARIES_ONLY`** — Kept. All three modes (`HITS_ONLY`, `SUMMARIES_ONLY`, `BOTH`) supported; cheap to implement on top of the same Milvus call ftsearch already runs.
3. **`explain: true`** — Schema unchanged. The ACL returns the literal string `"N/A"` for every hit's `explanation` when `explain=true` (§2.2). The only consumer (`ExplanationButton.tsx`) keeps working — the downloaded `.txt` just contains `"N/A"`. Revisit if a useful breakdown is wanted later.
4. **`articleId` value** — Wire format must match legacy exactly: `{friendlyId}:{base64UrlEncodedArticleNumber}`. The indexer projects (or constructs, when MongoDB doesn't store it directly) this composite into Milvus' PK; ftsearch returns and accepts the legacy value verbatim, the ACL passes it through. Not a deviation.
5. **Customer-article-number search** — Dropped along with the `CUSTOMER_ARTICLE_NUMBER` mode (§2.1). No replacement; the tenant-scoped customer-number lookup is not carried over to ftsearch.
6. **Locale** — German only for v1.
7. **Authentication** — None on either hop; both endpoints are `security: []` (§3). Both ACL and ftsearch are internal services not exposed to the public network, so authenticating per-request is unnecessary. Catalog visibility is scoped by the request's `selectedArticleSources` fields, not by an authenticated tenant identity.
8. **Milvus consistency level** — `Bounded`. Matches legacy ES's seconds-of-staleness tolerance (legacy refresh is 30s); no caller depends on read-your-writes from search. Indexer's Kafka-driven upserts (§6) become visible within Milvus' graceful period.

---

## 10. Acceptance criteria — API contract compliance

**Result equality vs. legacy is explicitly not a goal.** Replacing Elasticsearch with hybrid Milvus is *meant* to change the result set — that's the entire motivation for the new service. Acceptance is about the contract: every supported feature behaves correctly on its own terms, and every dropped feature is rejected cleanly.

The ACL/ftsearch stack is acceptance-ready when:

- ✅ Every request shape accepted by the new (narrowed) OpenAPI returns HTTP 200 and a body that validates against the response schema.
- ✅ Each filter in §4.3 demonstrably narrows the hit set (covered by per-filter functional tests on a controlled fixture, not by comparison to legacy).
- ✅ Each sort in §4.2 produces the documented ordering on the same fixture.
- ✅ Each `summaries.*` kind in §4.4 is computed over the full filtered hit set, not just one page; counts are internally consistent (sum of group counts ≤ `metadata.hitCount`).
- ✅ `metadata.hitCount` is the count of the full filtered set (not page-size-bounded); ftsearch's safety cap on top-K is documented and tested.
- ✅ `articles[].articleId` is emitted in the legacy `{friendlyId}:{base64UrlEncodedArticleNumber}` format on every response (checked directly on the wire payload — no legacy comparison needed).
- ✅ The same `articleId` round-trips through `articleIdsFilter` and ID-based fetches.
- ✅ Malformed input (bean-validation, constraint, JSON-parse) returns HTTP 400 in the legacy `{message, details, timestamp}` envelope; 5xx returns "Internal server error" (§3.1 table).
- ✅ Every request shape dropped by §2 returns HTTP 400 in the legacy envelope with a field-level message:
  - `searchArticlesBy ∉ {STANDARD}` (§2.1): `ALL_ATTRIBUTES`, `ARTICLE_NUMBER`, `CUSTOMER_ARTICLE_NUMBER`, `VENDOR_ARTICLE_NUMBER`, `EAN`, `TEST_PROFILE_01..20`.
- ✅ When `explain: true`, every `articles[].explanation` is the literal string `"N/A"` (§2.2); when `explain: false` or omitted, `explanation` is null/absent.
- ✅ Tracing baggage (`userId`, `companyId`, `customerOciSessionId`, W3C traceparent) flows next-gen → ACL → ftsearch → Milvus on every call.
- ✅ Retry / timeout policy (§4.7) is in place and exercised by injected ftsearch failures.

Captured production traffic is useful as an *input corpus* (does the new stack handle the diversity of real requests without throwing?) and as a relevance-quality dataset for offline evaluation, but **not** as a regression oracle for response equality.

---

The investigation is in `/home/mgerer/.claude/projects/-home-mgerer-next-gen/memory/` (project + reference memories) for future sessions. Want me to drill into any section — for example, sketch the OpenAPI for the ACL, prototype the request/response mapper, or audit `frontend/` for `explanation` consumers — before we move on?
