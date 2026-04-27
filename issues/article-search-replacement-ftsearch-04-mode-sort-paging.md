# F4 — searchMode + sorting + pagination + accurate hitCount

**Category**: ftsearch (`./search-api/`)
**Depends on**: F1, F2, F3
**Unblocks**: F5, A2, A3, A6

References: spec §3 (mode rules), §4.2 (sort), §4.6 (hitCount, explain).

## Scope

Honour `searchMode`, `sort`, and `page`/`pageSize` in the new ftsearch request, and produce an accurate `hitCount` over the full filtered set. This is what makes the contract feel like a real search API rather than a top-K wrapper.

## In scope

- **searchMode**:
  - `HITS_ONLY`: return paginated articles, summaries empty/omitted.
  - `BOTH`: return paginated articles AND summaries computed over the full filtered hit set (summary computation is F5; this packet wires the mode flag through).
  - `SUMMARIES_ONLY`: skip article hydration, return empty `articles[]`, but still compute summaries over the full filtered hit set.
- **Sorting** (per §4.2 table):
  - Relevance (default — sort omitted): existing hybrid score, tiebreak on `articleId` ascending.
  - `articleId,asc|desc`: native PK sort via Milvus query (no vector).
  - `name,asc|desc`: over-fetch and re-sort by `name` in Python, with deterministic tiebreak on `articleId`.
  - `price,asc|desc`: over-fetch, resolve each row's price via the F3 price-resolution module under request `currency` × `sourcePriceListIds` × priority, post-sort, paginate.
  - Multi-key sort: spec implies single-key but accepts a list — handle by applying the first sort key only, or apply lexicographically; pick one and document.
- **Pagination**: `page` (1-indexed), `pageSize` (default 10, max 500). For non-relevance sorts, over-fetch must cover at least `page × pageSize` rows; cap the total k to a configurable safety bound and document the bound.
- **Accurate `hitCount`**: Milvus top-K search returns no total. Implement via a separate `count(*)`-style pass — Milvus `query` with the same filter expr and `output_fields=[id]` plus a hard upper bound — OR over-fetch up to that bound and count. Bound is configurable via env, document the value.
- Response wiring:
  - `metadata.hitCount` populated for all modes.
  - `metadata.page` / `metadata.pageSize` echo the request.
  - `metadata.pageCount` computed from `hitCount` and `pageSize`.
  - `metadata.term` echoes the request `query` (or empty string when no query).
  - Each `articles[]` entry carries `articleId` and `score` (hybrid score for relevance sort, null/zero for non-relevance sort).
- Sort + filter interaction: the F3 filter translator runs before the sort; filtered-out rows do not participate in the over-fetch.

## Out of scope

- Summaries computation — F5 (this packet wires the mode flag, not the aggregation).
- `explain: true` payload — handled in the ACL (A3) per §2.2.

## Deliverables

- Mode-dispatch logic in `search-api/main.py` / `hybrid.py`.
- Sort module that knows how to over-fetch + re-sort for each non-native sort.
- Count module producing `hitCount`.
- Tests covering each `(mode, sort)` combination on a fixture, asserting:
  - Article ordering matches the documented sort.
  - `hitCount` equals the count of filtered rows independent of `pageSize`.
  - `SUMMARIES_ONLY` returns `articles=[]` but accurate `hitCount`.

## Acceptance

- §10 acceptance lines for sort and hitCount pass against a fixture.
- The safety cap on top-K is documented and exercised by a test that sets it low.
- Sort is deterministic across repeated calls with no underlying writes.

## Open questions for this packet

- Multi-key sort: legacy ES handles it via multi-sort lists; minimum viable here is "first key wins". Confirm whether stable secondary sort is needed (recommendation: tiebreak on `articleId` always, ignore secondary sort keys for now and call out as a future deviation if it bites).
- Hit-count safety cap: pick a number — recommendation is `max(10000, pageSize × 20)` with explicit env override.
