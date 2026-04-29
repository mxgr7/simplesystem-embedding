# A2 — Legacy → ftsearch request mapper

**Category**: ACL (new FastAPI service in this repo)
**Depends on**: A1 (skeleton + OpenAPI), F2..F5 (the ftsearch contract this maps to)
**Unblocks**: A5, A6

References: spec §3 (legacy request contract), §4.x (forwarding rules), §8 (cookbook).

## Status

✅ **Done** — commit `3e5e9b3`. `acl/models.py` has `LegacySearchRequest` matching the OpenAPI; `acl/mapping/request.py:map_request` is the pure mapper translating legacy → ftsearch (rename `queryString` → `query`, drop `searchArticlesBy` and `explain`, move page/pageSize/sort from body to query string, preserve currency two-roles). `acl/clients/ftsearch.py` is the async httpx client with default 4s timeout. The `/article-features/search` endpoint now does the full round-trip; upstream errors wrap in the legacy envelope. 13 mapper unit tests + 7 integration tests with `httpx.MockTransport`.

A subsequent A6 run surfaced one mapper bug fixed in commit `17ddc62`: ftsearch's `SelectedArticleSources` rejects three legacy customer-article-number fields (the §2.1 enum collapse retired them). The mapper now strips them before forwarding.

**Legacy reference** (next-gen): request shape `api-spec/specs/article-search/query-search-api.yaml` `SearchParams` (lines 113-237). `searchArticlesBy` enum at `article/search/query/src/main/java/com/simplesystem/nextgen/article/search/query/api/ArticleSearchOperations.java:91-117` (only `STANDARD` survives §2.1).

## Scope

Translate every legacy request the ACL accepts into a single ftsearch call. The ACL is "thin": it is not a query planner, it does not split requests across services, it does not own filters or sorts of its own. It builds the ftsearch request and forwards.

## In scope

- Pydantic models for the legacy request shape (matching `acl/openapi.yaml` from A1).
- A mapper module (e.g. `acl/mapping/request.py`) producing the ftsearch request DTO from F2:
  - **Pass-through filters**: `articleIdsFilter`, `vendorIdsFilter`, `manufacturersFilter`, `maxDeliveryTime`, `requiredFeatures`, `currentCategoryPathElements`, `currentEClass5Code`, `currentEClass7Code`, `currentS2ClassCode`, `coreSortimentOnly`, `closedMarketplaceOnly`, `coreArticlesVendorsFilter`, `blockedEClassVendorsFilters`, `eClassesFilter`, `accessoriesForArticleNumber`, `sparePartsForArticleNumber`, `similarToArticleNumber`, `selectedArticleSources` (all sub-fields).
  - **Currency two-roles** (§3): top-level `currency` is forwarded as the matching currency; `priceFilter.currencyCode` is forwarded alongside as the bound-decoding currency. The mapper does NOT collapse them or assert equality.
  - **Search mode**: forward as-is (`HITS_ONLY`, `SUMMARIES_ONLY`, `BOTH`).
  - **Search-articles-by**: only `STANDARD` is accepted (per §2.1); any other value is rejected by A4 before reaching the mapper.
  - **Sort**: forward `sort` query params unchanged.
  - **Pagination**: forward `page` (default 1) and `pageSize` (default 10, max 500).
  - **Summaries**: forward the `summaries[]` list and the `eClassesAggregations[]` payload (used by `ECLASS5SET`).
  - **Explain**: do NOT forward to ftsearch. The explain stub is filled in by the response mapper (A3) per §2.2.
  - **`s2ClassForProductCategories`**: forward (it changes which tree `PLATFORM_CATEGORIES` aliases to, per §4.4).
- HTTP client to ftsearch: persistent `httpx.AsyncClient` configured at app startup; wired to the ftsearch base URL via env (`FTSEARCH_BASE_URL`).
- Pure-function design: the mapper takes the legacy DTO + request-context (tracing baggage) and returns an ftsearch request DTO + outbound headers; no global state, no I/O.
- Unit tests covering each filter's pass-through, the two currency roles, and the explain-not-forwarded rule.

## Out of scope

- Parsing the response — A3.
- Returning legacy-shaped errors — A4.
- Retries / timeouts — A5.
- Acceptance tests — A6.

## Deliverables

- `acl/mapping/request.py` (mapper) + Pydantic models.
- `acl/clients/ftsearch.py` (httpx-based client).
- Wiring inside `acl/main.py` so a real request hits the mapper and posts to ftsearch.
- Unit tests for the mapper.

## Acceptance

- Every legacy filter that exists in §3 has a corresponding ftsearch field on the wire.
- The currency two-roles split is preserved end-to-end.
- A request with `searchMode: SUMMARIES_ONLY` produces an ftsearch call with the same mode set.
- The mapper is pure: same input → same output, no hidden state.
- ftsearch is reachable from the ACL and round-trips a smoke request.
