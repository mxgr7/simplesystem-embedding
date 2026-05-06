# ACL vs Legacy Parity Checklist

Legacy oracle: `localhost:8081` (article-search-query)
ACL under test: `localhost:8018` (acl → ftsearch on :8001, Milvus offers_v8/articles_v8)

Data: `/data/datasets/dev/mongo-exports/` loaded into both ES and Milvus.

**Test suite**: `scripts/parity_test.py` — 89 tests, **68 pass**, 21 fail.

---

## Passing tests (68/89)

### Browse & pagination
- 1a basic browse, 1b browse all (pageSize=500)
- 4a pages 1–3, 4b pageSize=0

### Search modes
- HITS_ONLY, SUMMARIES_ONLY, BOTH

### Sorting (page-1 article set)
- articleId asc/desc, name asc

### Filters
- closedMarketplaceOnly, empty CVIDs, empty sourcePriceListIds
- vendorIdsFilter (4 single-vendor tests), two vendors
- manufacturersFilter (DICK, Aristo, Schneider Electric, empty)
- maxDeliveryTime (1, 2, 5, 10)
- requiredFeatures (4 variants)
- priceFilter (4 ranges)
- category L1/L2 (4 tests), currentEClass5Code (4 tests)
- eClassesFilter (3 tests), s2ClassForProductCategories
- currentS2ClassCode, articleIdsFilter
- blockedEClass per-vendor, blockedEClass with exception
- vendor + maxDeliveryTime, vendor + category

### Summary deep checks
- 10a category summary L1, 10b eClass5 summaries (24M, 24220000)

### Error handling & edge cases
- missing required fields, explain true/false
- searchArticlesBy variants (STANDARD, ARTICLE_NUMBER, CUSTOMER_ARTICLE_NUMBER)
- articleId format round-trip, SUMMARIES_ONLY metadata, nonexistent CVID

### Text search + sort
- 12a query=DICK sort=name,asc, 12b query=DICK sort=price,asc

---

## Failing tests (21/89) — categorised

### Sort tiebreak mismatches (6 tests)
**Tests**: 3 name,desc; 3 price,asc; 3 price,desc; 4c last page; 14 sort order (3 deep)

**Root cause**: Within equal sort keys (same name or same price), legacy ES
uses insertion order (non-deterministic across reindexes); ftsearch uses
string-comparison of offer IDs. Articles differ on the page boundary where
tiebreak changes the cut.

**Status**: Accepted deviation — documented in spec §2. Not fixable without
matching ES doc insertion order, which is non-deterministic.

### Relationship filter article differences (3 tests)
**Tests**: 5o accessoriesFor, 5p sparePartsFor, 5q similarTo

**Root cause**: F9 dedup topology maps relationship article IDs differently
from legacy ES nested docs. Some articles appear in ACL but not legacy and
vice versa — likely an indexing/projection issue with how relationship target
IDs are resolved.

**Status**: Investigate indexer relationship projection.

### closedMarketplaceOnly article differences (2 tests)
**Tests**: 5a closedMarketplaceOnly=true, 5t2 closedMarketplace + manufacturer

**Root cause**: Similar to relationship filters — some articles in ACL but
not legacy and vice versa. Likely a closed-catalog scoping difference in the
offer probe vs legacy ES query.

**Status**: Investigate offer probe scoping for closed marketplace.

### blockedEClass global hitCount (1 test)
**Test**: 5s2 blockedEClass global

**Mismatch**: legacy=297, ACL=276 (21 fewer). The per-vendor blocked eclass
test passes, but the global (no vendor_ids) version doesn't. The ACL's
article-hash-based dedup may resolve fewer articles than legacy's nested
offer approach.

**Status**: Investigate — may be an indexing gap.

### eClassesAggregations counts (2 tests)
**Tests**: 6a eClassesAggregations, 6b eClassesAgg combined

**Mismatch**: agg1 (eClass 21M) L=21 vs A=10; agg2 (24M) L=215 vs A=221;
agg3 (21042101) L=14 vs A=8.

**Root cause**: eClassesAggregations counts are scoped to the materialised
article set from the offer probe. Legacy ES counts across all documents.
The difference comes from the article-hash dedup topology having fewer
distinct articles with eclass 21M (10 vs 21 in legacy).

**Status**: Data/topology difference — investigate if indexer misses articles.

### Invalid currency status code (1 test)
**Test**: 9b invalid currency

**Mismatch**: legacy=500, ACL=400.

**Root cause**: Legacy crashes (unhandled exception → 500) on invalid
currency; ACL validates and returns 400. ACL behavior is objectively correct.

**Status**: Accepted deviation — legacy bug.

### s2class summary count (1 test)
**Test**: 10c s2class summary 21000000

**Mismatch**: sameLevel group 21000000: L=21 vs A=14.

**Root cause**: Count scoped to disjunctive hash set from offer probe (290
hashes), which doesn't include all articles that legacy's ES query covers.
Same data/topology gap as eClassesAggregations.

**Status**: Same root cause as eClassesAggregations counts.

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

## Accepted deviations

| Issue | Category | Status |
|---|---|---|
| Text search hitCount ~200 | Architecture (ANN) | ACCEPTED — known limitation |
| Text search summary counts differ | Architecture (ANN) | ACCEPTED — known limitation |
| Sort tiebreak within equal values | ES insertion order vs ID comparison | ACCEPTED — §2 |
| Invalid currency: 500 vs 400 | Legacy bug | ACCEPTED — ACL is correct |
