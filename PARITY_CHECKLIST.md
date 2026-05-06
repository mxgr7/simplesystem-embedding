# ACL vs Legacy Parity Checklist

Legacy oracle: `localhost:8081` (article-search-query)
ACL under test: `localhost:8018` (acl → ftsearch on :8001, Milvus offers_v8/articles_v8)

Data: `/data/datasets/dev/mongo-exports/` loaded into both ES and Milvus.

---

## 1. Browse hitCount (no queryString)

| | Legacy | ACL | Status |
|---|---|---|---|
| hitCount | 273 | 266 | EXPECTED — F9 dedup (§2.4) |
| unique article IDs | 273 | 266 | same gap |

### 1a. Cross-vendor dedup (7 articles)

Legacy keeps both vendor copies; F9 dedup topology merges them via shared
`article_hash`. Exactly 7 articles are legacy-only:

- 514200 250, 517417 250, 519300 250, 519330 250, 510200 250, 513200 250, 596100 SC80

All from vendor `3cVSua489NQg9MtLmgTJk2` (526a3b68…), also present under vendor
`2VVuyd0NmcwffCiw8vGPpV` (76fa4405…). F9 dedup collapses them.

**Status**: EXPECTED per spec §2.4 — F9 dedup is by design.

### 1b. UNASSIGNED catalog exclusion (resolved)

Earlier runs showed 4 extra ACL articles from the UNASSIGNED catalog
`357a5946…`. After data reimport to v8, the UNASSIGNED catalog articles are
now included in both systems. The gap is purely the 7 dedup articles above.

---

## 2. Summaries

| Key | Legacy | ACL | Status |
|---|---|---|---|
| vendorSummaries | 4 vendors, counts match | 4 vendors, counts match | OK |
| manufacturerSummaries | 57 (incl. empty-name: 82) | 56 | EXPECTED — dedup + empty-name |
| featureSummaries | 100 | 100 | OK (cap applied) |
| pricesSummary | EUR 0.17–325.84 | EUR 0.17–325.84 | OK |
| categoriesSummary | null (no currentPath) | null (no currentPath) | OK |
| s2ClassCategories | null (no s2class data) | null (no s2class data) | OK |
| eClassesAggregations | [] | [] | OK |
| eClass5Categories | absent | present (§3 spec) | INTENTIONAL — new field |
| eClass7Categories | absent | present (§3 spec) | INTENTIONAL — new field |

**manufacturerSummaries diff**: Legacy counts 57 manufacturers including 82 articles
with empty manufacturer name. ACL counts 56 (skips empty names). DICK count
differs (14→8) due to dedup; TYROLIT (2→1) same cause.

---

## 3. Text search

| Query | Legacy | ACL | Status |
|---|---|---|---|
| DICK | 14 | 200 | MISMATCH — hitCount |
| pilnik | 8 | 200 | MISMATCH — hitCount |
| Briefablage | 3 | 200 | MISMATCH — hitCount |
| Schraube | 0 | 200 | MISMATCH — hitCount |
| nonsense_abc123 | 0 | 200 | MISMATCH — hitCount |
| 517417 | 2 | 200 | MISMATCH — hitCount |

**Root cause**: ANN (dense) search always returns `dense_limit=200` candidates
regardless of relevance. Legacy ES uses BM25 with exact match semantics — only
documents containing the query term match. Our system has no concept of "no match"
in ANN; RRF scores taper smoothly without a clear cutoff.

The BM25 leg (`sparse_codes`) only indexes identifier codes (article numbers,
EANs), not full text (names, descriptions). A full-text BM25 index on article
names would be needed for accurate text-search hitCount.

**Action**: Add full-text BM25 field to `articles_v{N}` schema (indexer change).
Document as deviation in spec §2 until resolved.

---

## 4. Filter scenarios

| Filter | Legacy | ACL | Status |
|---|---|---|---|
| closedMarketplaceOnly=true | 24 | 24 | OK |
| closedMarketplaceOnly + empty CVID pref | 0 | 0 | OK |
| vendorIdsFilter (gryffindor) | 14 | 14 | OK |
| vendorIdsFilter (bmecat) | 242 | 242 | OK |
| manufacturersFilter=DICK | 14 | 8 | EXPECTED — dedup |
| manufacturersFilter=Hoffmann | 0 | 0 | OK |
| maxDeliveryTime=2 | 258 | 251 | EXPECTED — dedup (diff=7) |
| maxDeliveryTime=5 | 271 | 264 | EXPECTED — dedup (diff=7) |
| priceFilter 0-50000 EUR | 273 | 266 | EXPECTED — dedup (diff=7) |
| eClassesFilter=[21000000] | 21 | 0 | DATA — hierarchy not expanded |
| eClassesFilter=[21042101] | 14 | 8 | EXPECTED — dedup (diff=6) |
| s2ClassForProductCategories + eClassesFilter | 21 | 0 | DATA — hierarchy not expanded |

**eClassesFilter hierarchy gap**: `eclass5_code` in articles_v8 stores only leaf
codes (e.g. 21042101). Legacy ES stores the full root→leaf chain
(21, 2104, 210421, 21042101) so a filter on parent code 21000000 matches.
Fix requires indexer to expand the hierarchy during import.

---

## 5. Pagination and sorting

All sort variants (articleId, name, price × asc/desc) show the same 273 vs 266
gap as the base browse — the 7-article dedup difference. Sort order within the
shared 266 articles is consistent.

---

## 6. searchMode behavior

| Mode | Legacy | ACL | Status |
|---|---|---|---|
| HITS_ONLY | 273 (10 arts) | 266 (10 arts) | EXPECTED — dedup |
| SUMMARIES_ONLY | 0 (0 arts) | 0 (0 arts) | OK |
| BOTH | 273 (10 arts) | 266 (10 arts) | EXPECTED — dedup |

---

## 7. Error handling

| Scenario | Legacy | ACL | Status |
|---|---|---|---|
| missing required fields | 500 | 400 | INTENTIONAL — stricter validation |
| searchArticlesBy=ARTICLE_NUMBER | 200 | 400 | INTENTIONAL — §2.1 drops non-STANDARD |
| pageSize=0 | 200 | 400 | INTENTIONAL — spec alignment |

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

## Remaining issues

| Issue | Category | Fix location |
|---|---|---|
| 273 vs 266 base browse | F9 dedup (expected) | spec §2.4 |
| Text search hitCount=200 | Architecture | indexer: add full-text BM25 field |
| eClassesFilter parent codes → 0 | Data pipeline | indexer: hierarchy expansion |
| manufacturerSummaries 57 vs 56 | F9 dedup + empty-name | expected |
| Empty manufacturer counted in legacy | Behavioral diff | acceptable — legacy quirk |
