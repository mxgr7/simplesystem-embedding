# F2 — ftsearch HTTP contract (request/response DTO + OpenAPI)

**Category**: ftsearch (`./search-api/`)
**Depends on**: —
**Unblocks**: F3, F4, F5, A1, A2, A3

References: spec §1, §3, §4 (defines the new internal contract that the ACL will call).

## Scope

Define and document the new ftsearch search contract. Today `/{collection}/_search` accepts `{query, index}` only; we need a richer request that can carry filters, sort, pagination, summaries, and currency context, and a response that carries the data the ACL needs to assemble the legacy envelope (articles, summaries blocks, accurate hit count).

This packet does NOT implement filter/sort/aggregation behaviour — it only locks the wire contract so F3..F5 and A2/A3 can build against a stable schema.

## In scope

- New Pydantic models in `search-api/main.py` (or split out into `search-api/models.py`) for the search request and response. Request fields cover everything F3..F5 will need:
  - `query: str | None`
  - `searchMode: HITS_ONLY | SUMMARIES_ONLY | BOTH`
  - `selectedArticleSources` (catalog scoping fields)
  - filters: `articleIdsFilter`, `vendorIdsFilter`, `manufacturersFilter`, `maxDeliveryTime`, `requiredFeatures`, `priceFilter` (with both `currency` roles per §3), `currentCategoryPathElements`, `currentEClass5Code` / `currentEClass7Code` / `currentS2ClassCode`, `coreSortimentOnly`, `closedMarketplaceOnly`, `coreArticlesVendorsFilter`, `blockedEClassVendorsFilters`, `eClassesFilter`, `accessoriesForArticleNumber` / `sparePartsForArticleNumber` / `similarToArticleNumber`
  - `summaries`: list of summary kinds requested
  - `eClassesAggregations`: payload used by `ECLASS5SET`
  - `currency` (top-level, drives matching)
  - `sort`: list (`name,asc`, `price,desc`, `articleId,asc`, …)
  - `page`, `pageSize` (`pageSize` capped at 500)
  - `s2ClassForProductCategories: bool`
- Response model:
  - `articles: [{articleId: str, score: float | null}]`
  - `summaries: { vendorSummaries, manufacturerSummaries, featureSummaries, pricesSummary, categoriesSummary, eClass5Categories, eClass7Categories, s2ClassCategories, eClassesAggregations }`
  - `metadata: { hitCount, page, pageSize, pageCount, term }`
- Update `search-api/openapi.yaml` to document the new shape; bump version. Keep the existing examples for backwards-compatible behaviour where applicable.
- Decide whether the new request shape replaces the old `{query, index}` body or sits next to it (recommendation: replace — the old shape's only consumer today is the playground; update the playground in the same PR or keep it pinned to a deprecated alias).
- Stub handler that accepts the new shape, validates it, and returns an empty response with `hitCount: 0`. F3..F5 fill in the behaviour.

## Out of scope

- Behaviour behind any of the new fields — they are accepted and validated only.
- Anything in the ACL — A1 onwards.

## Deliverables

- New / extended Pydantic models.
- `search-api/main.py` route accepts and validates the new request, returns a well-shaped empty response.
- `search-api/openapi.yaml` updated with full request/response schemas and at least one realistic example per searchMode.
- Light unit tests covering schema validation: rejects unknown searchMode, rejects `pageSize > 500`, accepts a fully-populated request.

## Acceptance

- Calling the endpoint with a full-shape request returns a 200 + valid empty response per the new schema.
- OpenAPI doc round-trips through any standard validator.
- Schema diff vs. today's contract is reviewed by the ACL author (A1) before merge — the contract is now frozen for downstream packets.

## Resolved decisions

- `pageSize=500` is the legacy cap (`query/src/main/resources/application.yml:22`); ftsearch enforces it server-side (defence in depth).
- Route stays `/{collection}/_search` (alias-resolved). No versioned route.
