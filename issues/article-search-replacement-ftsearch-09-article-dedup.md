# F9 — Article-level dedup topology (split `articles_v{N}` + `offers_v{N}`)

**Category**: ftsearch (`./search-api/`) + indexer (`./indexer/`) + Milvus schema
**Depends on**: F1 (schema), F3 (filter translator), I1 (projection)
**Unblocks**: F4 (sort-by-price browse path), F8 (envelope columns split across the two collections)
**Refines**: F1, F3, I1 (Phase A landed; Phase B absorbed into this packet's PR2). **Absorbs F6** — the German tokenizer/analyzer that F6 would have configured on the legacy `offers_codes` BM25 path lands directly on `articles_v{N}.sparse_codes` here; F6 as a separate packet should be marked superseded once F9 lands. This packet reshapes existing output without changing the ftsearch wire contract.

References: spec §6 (indexing pipeline), §7 (schema), §4.3 (filtering), §2.3 (relevance-pool bound on non-relevance sorts).

## Background — the gap this packet closes

Production scale: **510M raw MongoDB offers → 159M articles** (after `(vendorId, articleNumber)` aggregation, which I1 Phase B already does, mirroring the legacy ES doc shape) → **130M unique embeddings** (after further deduplication by hashing the embedded fields: `name + manufacturerName + categories + eclass codes`). The article→embedding step is a 1.22× dedup; the offer→article step (which I1 Phase B owns, not F9) is a 3.21× dedup.

**F9 is not justified by storage savings versus the current 159M-article topology** (the embedded-field-hash dedup is only 1.22×, ~13 GB HNSW RAM saved, ~8 hours TEI cycle saved). F9 is justified because the current 159M-article topology **cannot express correlated per-offer filters** — where a single request constraint applies to *the same offer* across multiple fields (catalog × price-list × price-range × core-marker). The article-level union semantics in today's collection answer "does any offer satisfy X *and* does any offer satisfy Y" rather than "does *the same* offer satisfy X *and* Y" (see "Why two collections" below for the Milvus capability survey that pins this constraint).

The only alternative that would support correlated per-offer filtering on the current schema is to **flatten back to one row per offer (510M rows, 510M embeddings)**. F9's split delivers the same per-offer correlation capability at **130M embeddings** instead of 510M.

| Resource                  | 510M-offer flat (correlation alt) | F9 split (130M embeddings) | Saved vs. 510M-flat |
| ------------------------- | ---------------------------------- | --------------------------- | ------------------- |
| Vector storage (fp16×128) | ~130 GB                            | ~33 GB                      | ~97 GB              |
| HNSW (M=16) RAM           | ~227 GB                            | ~57 GB                      | ~170 GB             |
| TEI bulk reindex @ 1k/s   | ~142 hours                         | ~36 hours                   | ~106 hours/cycle    |

The packet's value is **enabling correlated per-offer filtering at 130M-embedding cost** rather than 510M-embedding cost. Versus the no-correlation status quo (159M articles, ~40 GB / ~70 GB / ~44 hours), the savings are modest (~13 GB RAM, ~8 hours TEI).

## Why two collections (Milvus 2.6 capability survey)

The temptation is one deduplicated collection with a `connected_offers ARRAY<STRUCT>` field. Milvus 2.6.4+ has [Array of Structs](https://milvus.io/docs/array-of-structs.md) but the doc is explicit:

- Line 112: *"The scalar fields in the Array of Structs field do not support indexes."*
- Lines 118–120: *"You cannot use an Array of Structs or any fields within its Struct element in filtering expressions within searches and queries."*

The feature is purpose-built for ColBERT/MaxSim multi-vector retrieval, not nested-document filtering.

[JSON path indexing](https://milvus.io/docs/json-indexing.md) and [JSON Shredding](https://milvus.io/docs/json-shredding.md) cover indexed access into nested *objects*, but [shredding doc line 242](https://milvus.io/docs/json-shredding.md): *"JSON shredding does not cover queries on keys inside arrays, so you need a JSON index to accelerate those."* The [JSON operators](https://milvus.io/docs/json-operators.md) (`json_contains` / `_all` / `_any`) cover exact-element membership only — no `[*]` wildcard, no per-element correlated predicates, no ranges inside an array element. F8's price-scope pre-filter (vendor + catalog + price-range, all on the *same* offer) is the killer requirement.

Conclusion: a two-collection split is the only shape that preserves correlated per-offer filtering. Future readers — don't redo this survey.

## Topology

```
articles_v{N}   ─ PK              article_hash  VARCHAR(32)
                ─ offer_embedding  FLOAT16_VECTOR(128)  HNSW + COSINE
                ─ sparse_codes     SPARSE_FLOAT_VECTOR  BM25
                                   built from name + manufacturer +
                                   distinct EANs across offers +
                                   distinct article_numbers across offers
                ─ name, manufacturerName               (article-level scalars)
                ─ category_l1..l5                       INVERTED
                ─ eclass5_code, eclass7_code, s2class_code   ARRAY<INT32>, INVERTED
                ─ {ccy}_price_min, {ccy}_price_max     FLOAT, STL_SORT
                                   per-currency envelope across all the
                                   article's offers; lower/upper bound on
                                   any request-scoped resolved price.
                                   Enables sort-by-price browse path
                                   (F4) without ANN.
                ─ NO per-offer scalars

offers_v{N}     ─ PK              id  VARCHAR(256)
                                   legacy `{friendlyId}:{base64Url(articleNumber)}`
                ─ article_hash    VARCHAR(32), INVERTED   (join key)
                ─ vendor_id, catalog_version_ids, prices,
                  delivery_time_days_max, core_marker_*, relationship_*,
                  ean, article_number, features          (per-offer scalars)
                ─ price_list_ids, currencies,
                  {ccy}_price_min, {ccy}_price_max       (F8's envelope columns,
                                   per-offer scope — keep here for Path B)
                ─ NO vectors, NO sparse codes
```

`offers_codes` (today's separate BM25 collection) is retired — its content is article-level identifier text and folds into `articles_v{N}.sparse_codes`. The decoupled-clients pattern in `search-api/main.py` collapses to one Milvus client.

## Hash function and embedded-field set

- **Hash**: SHA-256 truncated to the first 16 bytes, hex-encoded (32 chars). Collision probability at 10⁸ articles ≈ 10⁻²⁰ — negligible. Halves IN-clause wire cost vs full sha256.
- **Embedded fields** (the inputs to TEI — same set as today's `EmbedClient` corpus):
  - `name`
  - `manufacturerName`
  - `category_l1..l5` (joined with the canonical `¦` separator)
  - `eclass5_code`, `eclass7_code`, `s2class_code` (sorted to canonicalise array order)
- **Not in the hash**: vendor, catalog versions, prices, EANs, article numbers, delivery time, features, relationships, core markers. Two offers of the same article can differ on any of these and still share an embedding.
- Hash is computed in `indexer/projection.py` from the canonicalised tuple; emit it on the offer row (offers-side join key) and use it as the PK on the article row (idempotent upsert by hash).

## Routing layer (search-api)

The dispatcher in `search-api/main.py` picks one of two paths based on filter shape; F3's `build_milvus_expr` splits into two scope-specific builders.

```python
build_article_expr  ← category, eclass, s2class
build_offer_expr    ← vendor, catalog, core_marker, relationships,
                      ean, article_number, features, delivery_time,
                      price-scope (F8 columns)
```

### Path A — vector-first (used when `offer_expr` is empty, or as Path B overflow fallback)
```
1. ANN(articles_v{N}, vec, expr=article_expr, limit=k_oversample)
   → top hashes, relevance-ordered
2. query(offers_v{N},
         filter="article_hash IN {hashes} AND <offer_expr>",
         filter_params={"hashes": [...]},
         output_fields=[id, article_hash, prices, ...],
         limit=len(hashes) * worst_case_offers_per_article)
3. group offers by article_hash; pick representative per article
4. price post-pass (existing prices.py)
5. paginate
```
When `offer_expr` is empty, step 2's filter degenerates to the IN-clause only — the resolve step just attaches offers for response shaping and price post-pass, no per-offer constraint to enforce. When this path runs as the Path B overflow fallback, step 2 *does* apply `offer_expr` — articles whose offers all fail the filter drop, and the page may underfill (the recall cliff documented below).

### Path B — filter-first (selective per-offer filters)
```
1. query(offers_v{N},
         filter=offer_expr,
         output_fields=[article_hash, id, prices, ...],
         limit=PATH_B_HASH_LIMIT + 1)        # bounded probe
2. if len(distinct hashes) > PATH_B_HASH_LIMIT: fall back to Path A
3. ANN(articles_v{N}, vec,
       filter="article_hash IN {hashes}",
       filter_params={"hashes": [...]})
4. re-attach offers per hash; price post-pass; paginate
```

`PATH_B_HASH_LIMIT = 25_000` (env-tunable). Selected from a measured IN-clause cost curve on the existing `offers` collection (159M rows, Milvus 2.6.15, VARCHAR PK with implicit index — same shape as `articles_v{N}.article_hash` will be). p50 latency: 1k=41 ms, 5k=113 ms, 10k=200 ms, **25k=436 ms**, 50k=777 ms, 100k=1473 ms. Templated `filter_params` saves only 1–3 % over string-formatted IN at every N; the cost is bitset construction + HNSW walk constrained to the bitset, not parser overhead. Speedup attempts (parallel batched search across separate gRPC channels, brute-force via `query()` + numpy top-K) all plateau or regress at this scale; the only real escape past ~25 k on CPU would be a GPU index. So 25 k is a hardware ceiling, not a guess. Drop to `10_000` if the per-request search budget is closer to 200 ms p95.

### Routing rule (correctness gate, not heuristic)

Path A and Path B are **not recall-equivalent** under selective offer filters. Worked example: a filter matching ~50 k offers / ~15 k unique hashes — Path A's top-200 ANN over the full 130M corpus contains only ~50 articles whose offers pass the filter (most top-200 ANN hits have no qualifying offer); Path B's ANN constrained to the 15 k matching hashes returns 200, all surviving by construction. Choosing the wrong path silently under-recalls; the bounded probe only catches the Path B → Path A direction (probe overflow), never the Path A under-recall direction.

The routing rule is therefore deterministic, not a cost heuristic:

```
if offer_expr is empty:
    Path A      # resolve step just attaches offers; no per-offer filter to enforce
else:
    Path B      # try; on probe overflow → Path A fallback (with resolve step that filters)
```

Probe overflow → Path A fallback **accepts under-recall** for selective-but-not-tight filters (those matching > 25 k unique hashes but < ~5 % of articles). Documented as a deviation in spec §2.4 and surfaced via a per-request `recall_clipped: true` metadata flag so callers and telemetry can distinguish.

### Sort=price + queryString
Per spec §2.3 (relevance-pool bound), Path A's `k_oversample` is capped at `RELEVANCE_POOL_MAX`. The article-level `{ccy}_price_min/max` columns on `articles_v{N}` are *not* used here — relevance-bounded ANN dominates and per-article exact resolution happens in step 3.

### Sort=price browse (no queryString)
No ANN. Goes against `articles_v{N}` ordered by `{ccy}_price_min ASC` (STL_SORT) with an over-fetch + refine loop using exact priority-resolved prices on the candidate offers. Article-level envelope columns are the enabling primitive; F4 owns the implementation.

## In scope

### Schema (`scripts/`)
- New `scripts/create_articles_collection.py` building `articles_v{N}` per the topology above. Vector index HNSW M=16 efConstruction=200 COSINE (matches today's defaults). Sparse index BM25. Scalar indexes per the field list. STL_SORT on every `{ccy}_price_min/max`.
- `scripts/create_offers_collection.py` revised: drop `offer_embedding`, add `article_hash VARCHAR(32) INVERTED`. Per-offer envelope columns from F8 stay (Path B uses them).
- Paired alias workflow in `scripts/MILVUS_ALIAS_WORKFLOW.md`: `articles` and `offers` aliases swung as one transaction in I3.

### Indexer (`indexer/projection.py`, `indexer/bulk.py`)
- Hash computation in `projection.py` from the canonicalised embedded-field tuple. Stable, deterministic, version-pinned (a `HASH_VERSION` constant — bump on field-set changes).
- TEI cache keyed by hash: only embed on first occurrence per hash; reuse the fp16 vector for subsequent offers sharing the hash. Cache scoped per bulk run. At projected scale (130M unique embeddings × ~350–400B per Python dict entry — 32-char str + 256B fp16 + dict bucket overhead) the in-memory footprint is ~17–19 GB. **Use a Redis sidecar from day one** rather than an in-memory dict; the dict tips over at scale and forces a mid-flight re-architecture.
- Two row streams per source `(offer, pricings, markers)` record:
  - Article row: idempotent upsert into `articles_v{N}` keyed by hash. Duplicates overwrite identical content — safe.
  - Offer row: insert into `offers_v{N}` keyed by `id`, carrying `article_hash`.
- Article-level envelope columns (`{ccy}_price_min/max`) computed at *the articles upsert layer*: aggregate per hash across the bulk run, emit final envelope at end of stream. **Streaming envelope updates (I2) are explicitly out of scope for F9** — see "Out of scope" below for the constraints I2 must satisfy.

### Filter translator (`search-api/filters.py`)
- Split `build_milvus_expr` → `build_article_expr` + `build_offer_expr`. The article-level scope: categories, eclass, s2class, articleIdsFilter (translates to `article_hash IN [...]` after hash-resolving the ids). The offer-level scope: everything else.
- `articleIdsFilter` requires a small map step: legacy `articleId` is the offer-level id, but in the article collection we filter by hash. Two options: (a) pre-query offers to resolve ids → hashes (one extra query), (b) compute the same hash from the request-side article ids if the embedded-field tuple is reachable. (a) is correct and simple; (b) is faster but constrains the request shape. Default to (a); revisit if the extra round-trip shows up in profiles.

### Routing (`search-api/main.py`, `search-api/hybrid.py`)
- New `route()` step before `run_search`: inspect the parsed request, decide Path A vs Path B, dispatch.
- Single `MilvusClient` (the decoupled `dense_client` / `codes_client` pair collapses; both legs hit `articles_v{N}`).
- RRF fusion runs over `article_hash`; the existing `_intersect_with_filter` shim in `hybrid.py:203` is removed for article-level filters and only re-appears as the `article_hash IN {hashes} AND offer_expr` resolve step.
- `PATH_B_HASH_LIMIT` env var (default 25_000).

### Codes folding (F6 absorption)
- BM25 sparse field on `articles_v{N}` is built from the union of identifier strings across the article's offers: `name + " " + manufacturerName + " " + " ".join(distinct EANs) + " " + " ".join(distinct article_numbers)`. The German tokenizer/analyzer config from F6 applies here.
- `offers_codes` collection retired. Migration: don't write to it during the new pipeline; drop it after the alias swing (I3) is complete.

### Acceptance — recall and parity
- For every test query in `tests/test_search_filters_integration_real.py`, results from the dedup'd topology are recall-equivalent to the pre-dedup baseline (same article appearing as the legacy `(vendorId, articleNumber)` group's representative offer).
- The bounded probe in Path B never returns more than `PATH_B_HASH_LIMIT` hashes; fallback to Path A is exercised by a synthetic permissive-filter test.
- Every `(vendorId, articleNumber)` group from the I1 sample maps to a stable `article_hash` (idempotent across re-runs).
- A row whose embedded fields change in MongoDB triggers a *new* article row with a new hash; the old hash's article row stays until garbage-collected by I3.

## Out of scope

- **Article-level garbage collection.** When all offers referencing a hash are deleted/updated away, the article row becomes orphaned. Detection and cleanup belongs in I2/I3 — schedule as a periodic sweep or a refcount maintained on the offers side.
- **Cross-region replication.** Out of scope for this packet; same constraint as today's single-collection topology.
- **Hash field-set evolution.** Changing the embedded-field set or canonicalisation breaks all existing hashes — handled via a `HASH_VERSION` bump that triggers a full rebuild through the existing alias-swing playbook (I3). No live migration path.
- **Streaming envelope updates (Kafka incremental → `articles_v{N}.{ccy}_price_min/max`).** Owned by I2. Eager per-event recompute is **not viable** because production write traffic comes in bursts (vendor mass price uploads); a 1M-event burst over an article fan-out of ~3.3 sibling reads/event = ~4M Milvus operations would saturate the cluster for hours. I2 must implement async coalescing — per-hash write buffer with a flush window (à la the legacy ES `BulkProcessor`), so each affected article re-aggregates once per window rather than per event. F9 establishes the column shape and the bulk write path; I2 picks the buffer/flush parameters and writes the streaming reducer.
- **Sort-by-price browse staleness between bulk reindexes.** Until I2's streaming envelope writer lands, `articles_v{N}.{ccy}_price_min/max` is refreshed only at bulk reindex cycles. Articles whose offers have been touched since the last reindex appear at their *previous* envelope position in sort-by-price browse (no queryString). For a daily reindex cadence and the burst sizes observed in production this affects a small sliver of catalog at any moment — tolerable until I2 lands. Document in spec §2.4.

## Deliverables

- `scripts/create_articles_collection.py` and revised `scripts/create_offers_collection.py`.
- `scripts/MILVUS_ALIAS_WORKFLOW.md` updated for paired-alias swings.
- `indexer/projection.py` with hash computation, Redis-backed TEI cache hook, two-stream emission, and unit tests against `tests/fixtures/mongo_sample/sample_200.json` asserting hash stability across re-runs (the 200-doc sample is too small to validate the production dedup ratio — that measurement happens against a larger sample as part of bulk validation).
- `indexer/bulk.py` orchestrator for the new pipeline. **Absorbs I1 Phase B work in full** — don't ship Phase B as a separate I1 packet; the article-aggregation step that I1 Phase B was scoped for is the same step that emits the article-row half of F9's two-stream projection.
- `search-api/filters.py` split with unit tests covering each scope.
- `search-api/main.py` routing layer + `search-api/hybrid.py` consolidation (single Milvus client, hash-level RRF).
- Integration tests on a fixture-loaded pair of collections demonstrating Path A and Path B paths produce identical hits for representative queries; bounded-probe fallback exercised.
- Doc updates: spec §6 / §7 cross-reference (this packet links to F9 as the canonical design); F1 status banner pointing to F9; F8 packet revised so envelope columns split between the two collections.

## Acceptance

- **Recall parity** vs the pre-dedup single-collection baseline on every fixture query *that does not require correlated per-offer filtering* (the new capability that Path B enables has no pre-dedup equivalent to compare against).
- **Embedded-field-hash dedup ratio measured**: at production scale the projection produces ~130M unique embeddings from ~159M articles (1.22× hash dedup); within tolerance on the 200-doc sample the ratio is too small to measure meaningfully — validate at bulk-run scale instead.
- **Reindex-time win measured against the 510M-flat alternative**: TEI calls per bulk run are bounded by the unique-embedding count (~130M) regardless of how many offers map to each hash, vs ~510M calls if the schema were flattened to one row per offer for correlated filtering. Demonstrate the cache hit-rate matches the measured dedup ratio.
- **Path B probe bound enforced**: probe never returns more than `PATH_B_HASH_LIMIT` hashes; overflow → Path A fallback exercised by a synthetic permissive-filter test that documents the under-recall behavior.
- **Recall cliff documented but bounded**: a synthetic test with a filter matching ~50 k unique hashes (above the limit) confirms Path A fallback returns < `pageSize` results when the filter is selective enough that ANN top-k doesn't overlap the matching set; metadata flag `recall_clipped: true` is set on these responses.
- **Alias swing atomic**: a partial failure during the paired swing leaves the system on the previous (consistent) pair, never on a mixed pair.
- **F8 recall parity** preserved end-to-end through the new topology (Path B's per-offer envelope filtering still fires on `offers_v{N}`; Path A's resolve step still applies the precise priority-resolved price filter; article-level envelope on `articles_v{N}` only used for sort-by-price browse, not for filtering).

## PR cleavage plan

The packet is too large for one PR — schema + indexer + filter splitter + routing + client consolidation in one diff makes failures hard to bisect. Cleavage:

| PR | Surface | Scope | Verifiable on its own |
| -- | ------- | ----- | --------------------- |
| 1  | Schema  | `scripts/create_articles_collection.py` (new); revised `scripts/create_offers_collection.py` (drop `offer_embedding`, add `article_hash VARCHAR(32) INVERTED`); paired alias workflow added to `scripts/MILVUS_ALIAS_WORKFLOW.md` | Collections create cleanly; alias swing dry-runs documented; no production traffic affected |
| 2  | Indexer (absorbs I1 Phase B) | `indexer/projection.py` — hash function + canonicalised embedded-field tuple + two-stream emission + Redis-backed TEI cache hook + per-currency envelope aggregation; `indexer/bulk.py` orchestrator with `(vendorId, articleNumber)` article-aggregation grouping; `scripts/indexer_bulk.py` CLI | Runs against `tests/fixtures/mongo_sample/sample_200.json`; asserts hash stability across re-runs and envelope correctness. No search-path change yet. |
| 3  | Search-api (behind feature flag) | `search-api/filters.py` split (`build_article_expr` + `build_offer_expr`); `search-api/hybrid.py` collapses `dense_client`/`codes_client` into one and runs RRF over `article_hash`; `search-api/main.py` routing layer (Path A / Path B with `PATH_B_HASH_LIMIT=25_000` env-tunable, deterministic rule per "Routing rule" above); `articleIdsFilter` round-trip resolution; F6 BM25 codes folded into `articles_v{N}.sparse_codes`. Feature flag `USE_DEDUP_TOPOLOGY=false` keeps current path; `=true` hits the new collections | Each path serves the integration fixture identically with flag on or off; bounded-probe fallback exercised; recall cliff metadata flag emitted on synthetic over-limit request |
| 4  | Cleanup | Retire `offers_codes` collection; drop `_intersect_with_filter` shim; remove `USE_DEDUP_TOPOLOGY` flag | Only after PR3 has soaked in production |

Sequencing: PR1 and PR2 can develop in parallel (PR2 needs PR1's collections only at integration-test time). PR3 depends on PR1 + PR2. PR4 only after soak.

I1 Phase B is **fully absorbed by PR2** — do not ship Phase B as a separate I1 packet; the article-aggregation grouping that I1 Phase B was scoped for is the same grouping that emits F9's article-row stream.

## Open questions

- **Representative-offer selection for the response.** The wire contract returns `articleId` (per spec §3); each article hash has many offers. Default: pick the offer that ranked first under the requested sort (cheapest matching offer for `sort=price`, lowest articleId for `sort=articleId`, first-by-relevance — i.e. arbitrary but stable — for default). Document the per-sort tiebreak when implementing PR3.
- **`articleIdsFilter` resolution.** Default (a) per "Filter translator" above: pre-query offers for hash mapping, one extra round-trip. Measure on representative requests during PR3 implementation; if it dominates, evaluate (b) hash-from-request reconstruction.

## Resolved decisions (recorded)

- **Routing rule**: deterministic, derived from `offer_expr` presence — not a cost heuristic. Probe-overflow falls back to Path A and accepts under-recall, surfaced via `recall_clipped: true` metadata. See "Routing rule" section above.
- **`PATH_B_HASH_LIMIT = 25_000`**: hardware ceiling on this Milvus + CPU configuration (~430 ms p95 IN-clause), validated by IN-clause cost benchmark. Not env-tunable above this without GPU index.
- **Streaming envelope finalisation**: out of scope; deferred to I2 with the explicit constraint that I2 must use async coalescing rather than per-event eager recompute (production write traffic comes in bursts). See "Out of scope" above.
- **TEI cache topology**: Redis sidecar from day one, not in-memory dict. The dict tips over at 17–19 GB scale.
- **`articleIdsFilter` resolution.** Default (a) above: pre-query offers for hash mapping. Measure the round-trip cost on representative requests; if it dominates, evaluate (b) hash-from-request reconstruction.
