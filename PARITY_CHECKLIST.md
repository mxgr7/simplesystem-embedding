# ACL vs Legacy Parity Checklist

Legacy oracle: `localhost:8081` (article-search-query)
ACL under test: `localhost:8018` (acl → ftsearch on :8001, Milvus offers_v8/articles_v8)

Data: `/data/datasets/dev/mongo-exports/` loaded into both ES and Milvus.

**Test suite**: `scripts/parity_test.py` — 89 tests, **72 pass**, 17 fail.

---

## Passing tests (72/89)

### Browse & pagination
- 1a basic browse, 1b browse all (pageSize=500)
- 4a pages 1–3, 4b pageSize=0

### Search modes
- HITS_ONLY, SUMMARIES_ONLY, BOTH

### Sorting (page-1 article set)
- articleId asc/desc, name asc

### Filters
- empty CVIDs, empty sourcePriceListIds
- vendorIdsFilter (4 single-vendor tests), two vendors
- manufacturersFilter (DICK, Aristo, Schneider Electric, empty)
- maxDeliveryTime (1, 2, 5, 10)
- requiredFeatures (4 variants)
- priceFilter (4 ranges)
- category L1/L2 (4 tests), currentEClass5Code (4 tests)
- eClassesFilter (3 tests), s2ClassForProductCategories
- currentS2ClassCode, articleIdsFilter
- blockedEClass per-vendor, blockedEClass global, blockedEClass with exception
- vendor + maxDeliveryTime, vendor + category

### Summary deep checks
- 10a category summary L1, 10b eClass5 summaries (24M, 24220000)
- 10c s2class summary 21000000
- 6a eClassesAggregations, 6b eClassesAgg combined

### Error handling & edge cases
- missing required fields, explain true/false
- searchArticlesBy variants (STANDARD, ARTICLE_NUMBER, CUSTOMER_ARTICLE_NUMBER)
- articleId format round-trip, SUMMARIES_ONLY metadata, nonexistent CVID

### Text search + sort
- 12a query=DICK sort=name,asc, 12b query=DICK sort=price,asc

---

## Failing tests (17/89) — categorised

### Sort tiebreak mismatches (12 tests)
**Tests**: 3 name,desc; 3 price,asc; 3 price,desc; 4c last page;
5a closedMarketplaceOnly; 5o accessoriesFor; 5p sparePartsFor; 5q similarTo;
5t2 closedMarketplace+manufacturer; 14 sort order (3 deep)

**Root cause**: Within equal sort keys (same name or same price), legacy ES
uses insertion order (non-deterministic across reindexes); ftsearch uses
deterministic friendlyId-based tiebreak. Full article sets are identical —
only the page boundary cut differs.

**Verified**: closedMarketplaceOnly (5a, 5t2) and relationship filters
(5o, 5p, 5q) return identical article sets when fetched with large pageSize.

**Status**: Accepted deviation — documented in spec §2. Not fixable without
matching ES doc insertion order, which is non-deterministic.

### Invalid currency status code (1 test)
**Test**: 9b invalid currency

**Mismatch**: legacy=500, ACL=400.

**Root cause**: Legacy crashes (unhandled exception → 500) on invalid
currency; ACL validates and returns 400. ACL behavior is objectively correct.

**Status**: Accepted deviation — legacy bug.

### Text search summaries (4 tests)
**Tests**: 11 query=DICK, Briefablage, Splint, Schneider

**Root cause**: ANN-based search returns ~200 candidates regardless of
relevance. Summary counts (vendor, manufacturer, feature, price) are computed
over these ~200 candidates, not over the BM25-precise result set that legacy
ES uses. Known architecture limitation.

**Status**: Accepted — spec §2 documents text search summary divergence.

---

## Code fixes applied (cumulative)

1. **SUMMARIES_ONLY hitCount=0** — `search-api/main.py`
2. **isEmpty check in closedMarketplace** — `search-api/filters.py`
3. **featureSummaries cap at 100** — `search-api/aggregations.py`
4. **categoriesSummary null when no path** — `search-api/aggregations.py`
5. **eclass/s2class summary null when no selection** — `search-api/aggregations.py`
6. **friendlyId base62 no-pad** — `acl/mapping/response.py`
7. **eClass hierarchy expansion** — `indexer/projection.py`
8. **S2CLASS derivation** — `indexer/s2class_mapper.py`
9. **eClassesFilter always uses s2class_code** — `search-api/filters.py`
10. **Per-offer result emission** — `search-api/routing.py`
11. **Accept searchArticlesBy legacy values** — `acl/models.py`
12. **Accept pageSize=0** — `acl/app.py`
13. **Missing required fields → 500** — `acl/app.py`
14. **Include empty manufacturer names** — `search-api/aggregations.py`
15. **Default sort: relevance + name tiebreak** — `search-api/sorting.py`
16. **articleId sort → relevance sort** — `search-api/sorting.py` (legacy has no articleId case)
17. **Summary scoping to materialised hashes** — `search-api/routing.py`
18. **Disjunctive faceting: categories** — `search-api/routing.py`, `search-api/aggregations.py`
19. **Disjunctive faceting: s2class** — `search-api/routing.py`, `search-api/filters.py`
20. **Disjunctive faceting: vendors** — `search-api/routing.py`, `search-api/filters.py`
21. **Disjunctive faceting: manufacturers** — `search-api/routing.py`, `search-api/filters.py`
22. **eClass depth fix for 8-digit codes** — `search-api/aggregations.py` (_eclass_depth, _eclass_parent)
23. **blockedEClass global no-op** — `search-api/filters.py` (empty vendorIds = no filter)
24. **eClassesAggregations: s2class + per-offer counting** — `search-api/aggregations.py`
25. **eclass/s2class summary per-offer counting** — `search-api/aggregations.py`
26. **Sort tiebreak: friendlyId-format sort key** — `search-api/sorting.py`

## Accepted deviations

| Issue | Category | Status |
|---|---|---|
| Text search hitCount ~200 | Architecture (ANN) | ACCEPTED — known limitation |
| Text search summary counts differ | Architecture (ANN) | ACCEPTED — known limitation |
| Sort tiebreak within equal values | ES insertion order vs deterministic ID | ACCEPTED — §2 |
| Invalid currency: 500 vs 400 | Legacy bug | ACCEPTED — ACL is correct |
| closedMarketplaceOnly page boundary | Sort tiebreak (same article sets) | ACCEPTED — §2 |
| Relationship filter page boundary | Sort tiebreak (same article sets) | ACCEPTED — §2 |
