# ACL vs Legacy Parity Checklist

Legacy oracle: `localhost:8081` (article-search-query)
ACL under test: `localhost:8018` (acl → ftsearch on :8001, Milvus offers_v8/articles_v8)

Data: `/data/datasets/dev/mongo-exports/` loaded into both ES and Milvus.

---

## 1. Browse hitCount (no queryString)

| | Legacy | ACL | Status |
|---|---|---|---|
| hitCount | 273 | 273 | OK |
| unique article IDs | 273 | 273 | OK |

Articles are deduplicated at index time (F9 article-dedup topology), but the
query layer emits one result per offer, matching legacy's per-offer semantics.

### 1a. Cross-vendor dedup (resolved)

Legacy keeps both vendor copies; F9 dedup topology merges them via shared
`article_hash` at index time. The query layer expands deduplicated articles
back into per-offer results, so the 7 cross-vendor articles now appear in
both systems:

- 514200 250, 517417 250, 519300 250, 519330 250, 510200 250, 513200 250, 596100 SC80

All from vendor `3cVSua489NQg9MtLmgTJk2` (526a3b68…), also present under vendor
`2VVuyd0NmcwffCiw8vGPpV` (76fa4405…).

### 1b. UNASSIGNED catalog exclusion (resolved)

Earlier runs showed 4 extra ACL articles from the UNASSIGNED catalog
`357a5946…`. After data reimport to v8, the UNASSIGNED catalog articles are
now included in both systems.

---

## 2. Summaries

| Key | Legacy | ACL | Status |
|---|---|---|---|
| vendorSummaries | 4 vendors, counts match | 4 vendors, counts match | OK |
| manufacturerSummaries | 57 (incl. empty-name: 82) | 57 (incl. empty-name) | OK |
| featureSummaries | 100 | 100 | OK (cap applied) |
| pricesSummary | EUR 0.17–325.84 | EUR 0.17–325.84 | OK |
| categoriesSummary | null (no currentPath) | null (no currentPath) | OK |
| s2ClassCategories | null (no s2class data) | null (no s2class data) | OK |
| eClassesAggregations | [] | [] | OK |
| eClass5Categories | absent | present (§3 spec) | INTENTIONAL — new field |
| eClass7Categories | absent | present (§3 spec) | INTENTIONAL — new field |

---

## 3. Text search

| Query | Legacy | ACL | Status |
|---|---|---|---|
| DICK | 14 | ~207 | ACCEPTED — ANN hitCount limitation |
| pilnik | 8 | ~207 | ACCEPTED — ANN hitCount limitation |
| Briefablage | 3 | ~202 | ACCEPTED — ANN hitCount limitation |
| Schraube | 0 | ~207 | ACCEPTED — ANN hitCount limitation |
| nonsense_abc123 | 0 | ~206 | ACCEPTED — ANN hitCount limitation |
| 517417 | 2 | ~204 | ACCEPTED — ANN hitCount limitation |

**Root cause**: ANN (dense) search always returns `dense_limit=200` candidates
regardless of relevance, expanded to ~207 by per-offer emission. Legacy ES uses
BM25 with exact match semantics — only documents containing the query term match.

The BM25 leg (`sparse_codes`) only indexes identifier codes (article numbers,
EANs), not full text (names, descriptions). A full-text BM25 index on article
names would be needed for accurate text-search hitCount.

**Status**: Accepted limitation for now. Future improvement: add full-text BM25
field to `articles_v{N}` schema (indexer change).

---

## 4. Filter scenarios

| Filter | Legacy | ACL | Status |
|---|---|---|---|
| closedMarketplaceOnly=true | 24 | 24 | OK |
| closedMarketplaceOnly + empty CVID pref | 0 | 0 | OK |
| vendorIdsFilter (gryffindor) | 14 | 14 | OK |
| vendorIdsFilter (bmecat) | 242 | 242 | OK |
| manufacturersFilter=DICK | 14 | 14 | OK |
| manufacturersFilter=Hoffmann | 0 | 0 | OK |
| maxDeliveryTime=2 | 258 | 258 | OK |
| maxDeliveryTime=5 | 271 | 271 | OK |
| priceFilter 0-50000 EUR | 273 | 273 | OK |
| eClassesFilter=[21000000] | 21 | 21 | OK |
| eClassesFilter=[21042101] | 14 | 14 | OK |
| s2ClassForProductCategories + eClassesFilter | 21 | 21 | OK |

---

## 5. Pagination and sorting

All sort variants (articleId, name, price × asc/desc) produce matching hitCounts
(273 = 273). Sort order is consistent.

---

## 6. searchMode behavior

| Mode | Legacy | ACL | Status |
|---|---|---|---|
| HITS_ONLY | 273 (10 arts) | 273 (10 arts) | OK |
| SUMMARIES_ONLY | 0 (0 arts) | 0 (0 arts) | OK |
| BOTH | 273 (10 arts) | 273 (10 arts) | OK |

---

## 7. Error handling

| Scenario | Legacy | ACL | Status |
|---|---|---|---|
| missing required fields | 500 | 500 | OK |
| searchArticlesBy=ARTICLE_NUMBER | 200 | 200 | OK |
| pageSize=0 | 200 | 200 | OK |

---

## Summary of code fixes applied

1. **SUMMARIES_ONLY hitCount=0** — `search-api/main.py`: set `effective_hit_count=0`
   when skip_articles is true (legacy returns hitCount=0 for SUMMARIES_ONLY).
2. **isEmpty check in closedMarketplace** — `search-api/filters.py`: return
   `MATCH_NOTHING_EXPR` when `catalogVersionIdsOrderedByPreference` or
   `sourcePriceListIds` is empty (legacy `ArticleSearchContext.isEmpty()`).
3. **featureSummaries cap at 100** — `search-api/aggregations.py`: legacy caps
   feature groups at 100.
4. **categoriesSummary null when no path** — `search-api/aggregations.py`: return
   `None` when `currentCategoryPathElements` is empty (legacy
   `CategorySummaryExtractor` returns null when `currentCategoryPath == null`).
5. **eclass/s2class summary null when no data** — `search-api/aggregations.py`:
   return `None` instead of empty object when no eclass data exists.
6. **friendlyId base62 no-pad** — `acl/mapping/response.py`: Devskiller FriendlyId
   does NOT zero-pad; removed `.rjust(22, "0")`.
7. **eClass hierarchy expansion** — `indexer/projection.py`, `indexer/duckdb_projection.py`:
   expand leaf eclass codes to full zero-padded 8-digit ancestor chain at import time
   (21042101 → [21000000, 21040000, 21042100, 21042101]).
8. **S2CLASS derivation** — `indexer/s2class_mapper.py`: derive s2class codes from
   highest available eclass version (ECLASS_8 > ECLASS_5_1) through binary mapping
   tables extracted from legacy Java indexer (`5-s2.bin.gz`, `8-s2.bin.gz`).
9. **eClassesFilter always uses s2class_code** — `search-api/filters.py`: legacy
   `EClassesFilterProvider` always queries `s2classGroups`, regardless of
   `s2ClassForProductCategories` flag. Fixed to always use `s2class_code`.
10. **Per-offer result emission** — `search-api/routing.py`: emit one result per
    offer instead of picking a single representative per deduplicated article hash.
    hitCount reflects the offer count, matching legacy's per-document semantics.
11. **Accept searchArticlesBy legacy values** — `acl/models.py`: accept
    `ARTICLE_NUMBER` and `CUSTOMER_ARTICLE_NUMBER` (treated as `STANDARD`).
12. **Accept pageSize=0** — `acl/app.py`: changed `pageSize` minimum from 1 to 0.
13. **Missing required fields → 500** — `acl/app.py`: return HTTP 500 for missing
    required fields (legacy crashes rather than validating).
14. **Include empty manufacturer names** — `search-api/aggregations.py`: include
    empty-string manufacturer names in summaries (legacy ES includes them).

## Remaining deviations (accepted)

| Issue | Category | Status |
|---|---|---|
| Text search hitCount ~200 | Architecture (ANN) | ACCEPTED — known limitation |
