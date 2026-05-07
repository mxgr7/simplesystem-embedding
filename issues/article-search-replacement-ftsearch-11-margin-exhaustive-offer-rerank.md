# F11 — Margin-exhaustive offer-first semantic rerank

**Category**: ftsearch (`./search-api/`) + Milvus schema/projection (`./indexer/`)
**Depends on**: F3 (filter translator), F4 (sort/paging/count contract), F8 (price-scope envelopes), F9 (split `articles_v{N}` + `offers_v{N}` topology), F10 (article-side CV envelope, optional but complementary)
**Unblocks**: legacy-style article dedup + exact correlated offer filters under semantic search without ACL-side re-pagination
**Refines**: F9 Path B. This packet replaces the bounded-probe Path B behaviour for semantic searches where an exact offer-side filter is present and the semantic margin window can be exhausted.

References: F9 correction (one `offers_v{N}` row per source offer), spec §2.3 (relevance-pool bounds), spec §4.3 (filtering), F8 (price resolver / envelopes).

## Status

⬜ **Draft implementation plan** — written after validating the local `articles_v7` / `offers_v7` import and observing that ACL-side de-duplication is too late: it can avoid backend duplicate-key errors, but it cannot make pagination/counts match legacy. The robust fix belongs in ftsearch before pagination.

## Background — the specific gap

Legacy Elasticsearch avoids offer/catalog-version duplicates structurally:

```text
one ES root document = one legacy ArticleId = vendorId + articleNumber
nested offers[]      = catalog-version / source variants
```

A nested query can require predicates to match the same offer element, while Elasticsearch still returns the parent article document once.

F9's Milvus topology intentionally flattened offers because Milvus has no ES-style nested filtering:

```text
articles_v{N}: one row per article_hash, vectors + sparse codes
 offers_v{N}: one row per source offer, exact offer scalars
```

That preserves same-offer predicate correctness on `offers_v{N}`, but introduces a retrieval problem: dense/BM25 ranking happens on `articles_v{N}` by `article_hash`, while legacy response identity and Portal loading are by `(vendor_id, article_number)`. Multiple offer rows and even multiple `article_hash` rows can collapse to the same legacy article. If this collapse happens in ACL after ftsearch pagination, pages can underfill.

We also must not solve this by putting all offer filter fields into arrays on `articles_v{N}` and applying filters there. That causes the classic lost-correlation false positive:

```text
article row:
  manufacturers = [X, Y]
  prices        = [20, 5]

query:
  manufacturer = X AND price < 10

false positive:
  one offer has manufacturer X but price 20
  another offer has price 5 but manufacturer Y
```

The design below treats article-side filtering as candidate generation only, and makes `offers_v{N}` the exact verifier.

## Goal

For semantic search requests with offer-side/correlated filters, produce a complete, precise, legacy-article-level result window:

- exact same-offer filtering via `offers_v{N}`;
- semantic ranking via `articles_v{N}` dense vectors;
- de-duplication by legacy article key `(vendor_id, article_number)` before pagination;
- deterministic representative offer selection;
- exact `hitCount` / `pageCount` **within the exhausted semantic margin window**;
- explicit clipped metadata when any configured maximum prevents exhausting the exact or semantic window.

## Non-goals

- Do not implement ACL-side overfetch/re-pagination as the primary fix. ACL may keep defensive de-duplication, but normal responses should already be legacy-unique.
- Do not claim an infinite/global semantic total. Counts are exact for the completed semantic margin window. If the margin window is clipped, counts are clipped.
- Do not emulate SQL `DISTINCT` in Milvus. Grouping/collapse is application-side after exact offer verification.
- Do not remove the existing F9 Path A / Path B router immediately. F11 should land behind a feature flag and replace only the cases it can complete safely.

## Definitions

- **legacy article key**: `(vendor_id, article_number)`. This is the identity legacy ES returns as `ArticleId`.
- **offer key**: `(vendor_id, article_number, catalog_version_id)`; the current `offers_v{N}.id` encodes this.
- **article hash**: `article_hash`, the F9 embedding dedup key and `articles_v{N}` primary key.
- **exact offer expression**: the Milvus/Python predicate over `offers_v{N}` that preserves same-row offer correlation.
- **semantic margin window**: all dense article hits whose score passes the configured margin rule, retrieved up to a configured maximum. Counts/pages are exact only if this window is exhausted.

## Core algorithm

For a request with `queryString` and any exact offer-side predicate:

```text
Pass 1: exact offer prefilter
  query/iterate offers_v{N} with exact offer filters
  collect eligible article_hashes

Pass 2: margin-exhaustive semantic retrieval
  dense-search articles_v{N}
    filter = article_hash IN eligible_hashes
  retrieve every hit inside the semantic margin, up to MAX

Pass 3: exact offer rehydrate + legacy collapse
  query/iterate offers_v{N}
    filter = exact offer filters AND article_hash IN semantic_hashes
  apply precise Python post-passes (price priority/range, etc.)
  group by (vendor_id, article_number)
  select representative offer
  sort, count, paginate
```

The important property is that Pass 3 repeats the exact offer filter. Pass 2 only ranks candidate hashes; it is never trusted for final predicate correctness.

## Correctness argument

Assuming Pass 1 and Pass 2 both complete without clipping:

1. Every returned offer row in Pass 3 satisfies the exact offer predicate because Pass 3 applies the predicate on `offers_v{N}` rows.
2. Every returned legacy article has at least one exact matching offer row because groups are built only from Pass 3 rows.
3. No article outside the semantic margin can appear because Pass 3 is constrained to Pass 2's semantic hash set.
4. No duplicate legacy article can appear because pagination operates on groups keyed by `(vendor_id, article_number)`, not on offer IDs or article hashes.
5. `hitCount = number of grouped legacy article keys` is exact for the margin window because the full verified set is materialized before pagination.

If Pass 1 or Pass 2 clips, the same logic still gives precise returned rows, but not complete recall/counts. The response must set clipped metadata.

## Detailed implementation plan

### 1. Schema/projection prerequisites

`offers_v{N}` must contain every field needed for exact same-offer verification. The current corrected F9 offer schema already has many of them, but not all legacy nested-offer fields are guaranteed to be present in every collection version.

Add/copy to future `offers_v{N+1}` as needed:

- `vendor_id` — already present;
- `article_number` — already present;
- `catalog_version_id` — already present;
- `article_hash` — already present;
- `prices` / price envelope fields — already present;
- `price_list_ids`, `currencies`, `{ccy}_price_min`, `{ccy}_price_max` — already present from F8;
- `delivery_time_days_max` — already present;
- `features` — already present;
- relationship arrays — already present;
- core marker source arrays — already present;
- **`manufacturerName`** — required if `manufacturersFilter` is to be exact instead of article-side superset;
- **category path fields** (`category_l1..category_l5`) if category filters must preserve same-offer semantics;
- **eClass/S2 class arrays** if eClass filters must preserve same-offer semantics.

Also add a convenience output field:

- `legacy_article_key VARCHAR`, e.g. `{vendor_uuid}:{raw_article_number}` or another stable non-wire internal key.

This is not required for correctness because `vendor_id` + `article_number` are enough, but it simplifies grouping and test assertions. Do not expose this field on the ftsearch wire.

### 2. Filter split revision

Today `build_article_expr` owns some filters because their fields live only on `articles_v{N}`. Under F11, correlated predicates should move to the exact offer path whenever their fields exist on `offers_v{N+1}`.

Introduce a new builder, or extend the existing split with explicit modes:

```text
build_offer_exact_expr(req)
  same-row offer predicates; used in Pass 1 and Pass 3

build_article_superset_expr(req)
  safe article-side prefilters only; may produce false positives but never false negatives
```

Rules:

- Exact offer predicates include CV scope, vendor, article IDs, delivery, required features, relationships, core-only/core-vendor logic, price-list/price-envelope prefilters, manufacturer/category/eClass once those fields exist on offers.
- Precise price evaluation still needs Python verification via `prices.py` because `prices` is a structured list and priority resolution is not fully expressible as a Milvus scalar predicate.
- Negative/exclusion predicates must not be pushed to article-side superset filtering unless proven safe. They can create false negatives under flattened arrays.
- If a correlated filter field is not yet present on `offers_v{N}`, this path must either decline routing or mark the request unsupported for F11 and use the existing F9 path.

### 3. Pass 1 — exact offer prefilter

Implement an iterator helper in `search-api/routing.py` or a new `search-api/exhaustive.py` module:

```text
scan_matching_offers(req, offer_exact_expr, max_rows, max_hashes)
```

Behaviour:

- Use `MilvusClient.query_iterator` where available; otherwise chunk by deterministic partitions or fall back to capped `query()` with clipped metadata.
- Output only small fields:
  - `article_hash`
  - `vendor_id`
  - `article_number` / `legacy_article_key`
  - fields needed for price post-pass if price filters cannot be prefiltered tightly enough.
- Apply Python post-passes that are part of exact filtering but not expressible in Milvus, especially precise price-list/currency/priority/min/max resolution.
- Collect:
  - `eligible_hashes: set[str]`
  - optionally `eligible_legacy_keys: set[str]` for filter-universe diagnostics
  - `offer_prefilter_rows_scanned`
  - `offer_prefilter_rows_kept`
  - `offer_prefilter_clipped`

Caps/env vars:

```text
EXACT_OFFER_SCAN_MAX_ROWS      default TBD by benchmark
EXACT_OFFER_SCAN_MAX_HASHES    default TBD by benchmark
EXACT_OFFER_SCAN_TIMEOUT_MS    default TBD
```

If the scan hits a cap:

- Do not claim exact counts.
- Either fall back to the current F9 path or continue with `recallClipped=true`, depending on an env flag.

### 4. Pass 2 — dense retrieval constrained to eligible hashes

Implement:

```text
dense_margin_search_articles(query, eligible_hashes, article_superset_expr, margin, max_candidates)
```

Filter:

```text
article_hash IN eligible_hashes
AND article_superset_expr   # only if guaranteed no false negatives
```

Retrieval strategy:

1. Start with a limit large enough to cover the requested page window:
   ```text
   needed = page * pageSize
   limit = max(needed * INITIAL_OVERSAMPLE, DENSE_POOL)
   ```
2. Search `articles_v{N}` and record top score.
3. Apply margin rule, e.g. one of:
   ```text
   score >= top_score * SEMANTIC_SCORE_FLOOR
   score >= top_score - SEMANTIC_SCORE_DELTA
   ```
4. Increase limit geometrically until either:
   - the last returned hit is below the margin, meaning the margin window is exhausted; or
   - `SEMANTIC_WINDOW_MAX` / Milvus result-window cap is reached.

Required env vars:

```text
SEMANTIC_SCORE_FLOOR          e.g. 0.20 initially, align with existing relevance floor if possible
SEMANTIC_SCORE_DELTA          optional alternative to floor
SEMANTIC_WINDOW_MAX           hard cap on article hashes retrieved
SEMANTIC_WINDOW_GROWTH_FACTOR e.g. 2
```

Output:

- ordered `semantic_hashes`
- `score_by_hash`
- `semantic_window_exhausted: bool`
- `semantic_window_clipped: bool`
- debug timings and candidate counts

Important Milvus limits:

- A huge `article_hash IN [...]` filter can become too large/slow. Add a separate cap:
  ```text
  ARTICLE_HASH_FILTER_MAX
  ```
- If `eligible_hashes` exceeds that cap, do not build a huge expression blindly. Route to existing F9 fallback or a future article-first/backfill path.

### 5. Pass 3 — exact rehydrate, collapse, representative selection

Implement:

```text
rehydrate_verified_offers(semantic_hashes, offer_exact_expr)
```

Milvus filter:

```text
(offer_exact_expr) AND article_hash IN semantic_hashes
```

Output fields:

- `id`
- `article_hash`
- `vendor_id`
- `article_number` / `legacy_article_key`
- `catalog_version_id`
- `prices`
- fields needed for sorting/summary/representative selection

Post-pass:

- precise price resolution via existing `prices.resolve_price` / `passes_price_filter` logic;
- per-vendor blocked eClass logic if still split-incompatible;
- any other predicate not safely expressible in Milvus.

Grouping:

```text
group key = (vendor_id, article_number)
```

When multiple `article_hash` values map to the same legacy key:

- use the highest semantic score among the group's hashes as the group's relevance score;
- preserve the winning hash for article metadata/summaries;
- retain all exact matching offers under the group for representative selection.

Representative offer selection:

- For default relevance sort:
  1. group relevance score desc;
  2. choose an offer in the group by catalog-version preference order from `selectedArticleSources.catalogVersionIdsOrderedByPreference`;
  3. stable tie-break by legacy article ID / offer ID.
- For `price,asc|desc`:
  - choose cheapest/most-expensive resolved price among exact matching offers;
  - stable tie-break by legacy article ID / offer ID.
- For `name,asc|desc`:
  - use the selected article/offer name used by legacy sorting semantics; if multiple offer names exist, use the same nested sort mode as legacy (`SortMode.Min` for name) or document the deviation.
- For `articleId,asc|desc`:
  - sort by legacy article ID, not offer ID.

Note: legacy Portal ultimately resolves the display offer in backend Mongo via `OfferSelector`, using catalog-version preference order. ftsearch's representative offer mainly exists to produce one stable row and score; ACL maps it back to legacy `ArticleId`.

### 6. Sorting, counts, pagination

After Pass 3 grouping:

```text
verified_groups = all collapsed legacy article groups in semantic margin
sorted_groups   = sort according to request
hitCount        = len(sorted_groups) if not clipped
page_slice      = sorted_groups[offset:offset + pageSize]
```

Metadata:

- If Pass 1 and Pass 2 are complete:
  - `hitCount` exact for the semantic margin window;
  - `pageCount = ceil(hitCount / pageSize)`.
- If any pass clipped:
  - set `recallClipped=true`;
  - set `hitCountClipped=true`;
  - `hitCount` is a lower bound or configured cap, matching current clipped-count semantics.

Add debug fields under `_debug` / route debug:

```text
route = path_b_margin_exhaustive
offer_prefilter_rows_scanned
offer_prefilter_rows_kept
distinct_prefilter_hashes
semantic_hashes_returned
semantic_window_exhausted
semantic_window_clipped
verified_offer_rows
verified_legacy_groups
```

### 7. Routing integration

Add a new route label:

```text
path_b_margin_exhaustive
```

Routing conditions for first implementation:

```text
USE_MARGIN_EXHAUSTIVE_PATH=1
AND queryString is non-empty
AND exact offer filters are present
AND all correlated request filters are supported by offers_v{N}
AND pass-1 eligible hash count <= ARTICLE_HASH_FILTER_MAX
```

If any condition fails, fall back to existing F9 routing.

Browse/no-query requests can use a simpler exact-offer path later:

```text
offers exact scan -> group by legacy key -> sort/page
```

but that is out of the first semantic-rerank slice unless needed for acceptance.

### 8. Summaries / aggregations

For `searchMode=BOTH`, summaries should be computed from the same verified group universe used for hits, not from raw offer rows before semantic ranking.

First implementation options:

1. Reuse existing `_compute_summaries` by passing the verified semantic hashes / legacy keys where possible.
2. Add a dedicated summary computation over `verified_groups` for manufacturer/vendor/features/prices/category/eClass.

Acceptance should pin that summaries are scoped to:

```text
exact offer filter ∩ semantic margin window ∩ legacy article collapse
```

If summaries are deferred, gate F11 to `HITS_ONLY` initially and document the temporary route fallback for `BOTH` / `SUMMARIES_ONLY`.

## Implementation slices

### PR 1 — route skeleton + synthetic unit tests

- Add `path_b_margin_exhaustive` dispatcher branch behind `USE_MARGIN_EXHAUSTIVE_PATH`.
- Add pure-Python helpers for:
  - legacy key construction;
  - grouping by legacy key;
  - representative selection;
  - margin-window clipping decisions.
- Unit tests with mocked Milvus client:
  - duplicate catalog versions collapse before pagination;
  - multiple hashes for same legacy key collapse;
  - clipped margin sets clipped flags;
  - precise pass-3 verification drops a pass-2 false positive.

### PR 2 — offer schema/projection support

- Add missing correlated fields to `offers_v{N+1}` schema/projection, at minimum `manufacturerName` for the documented false-positive class.
- Add `legacy_article_key` if chosen.
- Update `indexer/projection.py`, DuckDB projection, Arrow schema, schema tests, projection tests, docs.

### PR 3 — real Milvus implementation

- Implement Pass 1 with `query_iterator` and caps.
- Implement Pass 2 dense margin retrieval with geometric limit growth and clipping.
- Implement Pass 3 exact rehydrate with query iterator / batched hash filters.
- Wire route/debug/metrics.
- Add live-Milvus integration tests on fixture collections.

### PR 4 — summaries and acceptance hardening

- Scope summaries to verified legacy groups.
- Add red-team tests for:
  - manufacturer + price correlation false positive;
  - category/eClass correlation if those fields are moved to offers;
  - page 1 and page 2 full after collapse when enough verified groups exist;
  - exact `hitCount` / `pageCount` when not clipped;
  - lower-bound/clipped metadata when caps fire.

## Acceptance criteria

- **No ACL pagination shrinkage.** For fixture data with multiple matching catalog versions per article, ftsearch itself returns at most one row per legacy `(vendor_id, article_number)` before ACL mapping. ACL defensive dedup removes zero rows in the normal path.
- **Correlated filter correctness.** A fixture with:
  ```text
  offer A: manufacturer=X, price=20
  offer B: manufacturer=Y, price=5
  query: manufacturer=X AND price<10
  ```
  returns no article. A matching same-row offer returns the article.
- **Multiple-hash collapse.** If two `article_hash` rows map to the same legacy article and both are inside the semantic margin, the response contains one legacy article, with the higher relevance score.
- **Exact count when complete.** When Pass 1 and Pass 2 do not clip, `metadata.hitCount` equals the number of verified legacy groups and `pageCount` equals `ceil(hitCount / pageSize)`.
- **Clipped count when incomplete.** If Pass 1 scan cap, hash-filter cap, or Pass 2 semantic-window cap fires, the response sets `recallClipped=true` and `hitCountClipped=true`.
- **Legacy offer preference.** Representative offer selection follows catalog-version preference order for relevance/name/articleId sorts, and price selection follows resolved price for price sorts.
- **Performance guardrails.** Requests that exceed configured exact-scan/hash-filter/margin caps do not build unbounded `IN` expressions and do not time out silently; they fall back or return clipped metadata.

## Open questions

- **Margin rule.** Use relative score floor, absolute delta, or Milvus range-search parameters if available for the index type? Needs benchmark on real query logs.
- **Milvus result-window ceiling.** The local default `proxy.maxResultWindow` is 16 384 for query/search windows. Production settings and whether search pagination/offset can safely exceed this need verification.
- **Large hash filters.** For `eligible_hashes` above `ARTICLE_HASH_FILTER_MAX`, should the router fall back to existing F9 Path A/backfill, return clipped, or introduce a temporary side collection / partitioned searches? Milvus has no server-side join, so this is the main scale constraint.
- **BM25 / strict identifier queries.** This packet is phrased for dense semantic search. Identifier/BM25 retrieval needs either the same margin-window abstraction over `sparse_codes` or a separate exact-code route.
- **Summaries semantics.** Decide whether summaries over a semantic margin window are acceptable for legacy parity, or whether `BOTH` should fall back until the summary universe is fully specified.
- **Which filters must move to offers.** Manufacturer is the obvious example. Category and eClass are nested under legacy offers too; decide whether hash-level equality is sufficient or whether exact same-offer semantics require copying them to `offers_v{N+1}`.

## Rollout

1. Ship behind `USE_MARGIN_EXHAUSTIVE_PATH=0` by default.
2. Enable in local/dev against `articles_v7` / `offers_v7` only for supported filter shapes.
3. Compare against legacy ES and current F9 for a captured query set:
   - result count;
   - duplicate removal count;
   - first-page overlap;
   - clipped rate;
   - latency distribution.
4. Enable for production only after clipped-rate and latency SLOs are acceptable, and after schema fields required for exact offer filtering are present in the active `offers_v{N}` alias.
