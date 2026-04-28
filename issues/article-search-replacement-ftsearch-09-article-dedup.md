# F9 — Article-level dedup topology (split `articles_v{N}` + `offers_v{N}`)

**Category**: ftsearch (`./search-api/`) + indexer (`./indexer/`) + Milvus schema
**Depends on**: F1 (schema), F3 (filter translator), I1 (projection)
**Unblocks**: F4 (sort-by-price browse path), F8 (envelope columns split across the two collections)
**Refines**: F1, F3, F6, I1 — all already landed (F6 partially); this packet reshapes their output without changing the ftsearch wire contract.

References: spec §6 (indexing pipeline), §7 (schema), §4.3 (filtering), §2.3 (relevance-pool bound on non-relevance sorts).

## Background — the gap this packet closes

Today's F1 schema mirrors one Milvus row per `(vendorId, articleNumber)` group (the legacy ES doc shape). At production scale that's ~159M rows projected from ~515M raw MongoDB offers. Each row carries an `offer_embedding` and a `sparse_codes` BM25 vector — even though many rows share the same embedded-field content (name, manufacturerName, categories, eclass codes) and therefore *would produce the same embedding*.

Measured dedup ratio across the embedded-field hash: **~3.33× (5 : 1.5)**. Absolute savings at the 159M row scale:

| Resource                 | Per-offer (today) | Per-article (dedup) | Saved          |
| ------------------------ | ----------------- | ------------------- | -------------- |
| Vector storage (fp16×128)| ~40 GB            | ~12 GB              | ~28 GB         |
| HNSW (M=16) RAM          | ~70 GB            | ~20 GB              | ~50 GB         |
| TEI bulk reindex @ 1k/s  | ~44 hours         | ~13 hours           | ~31 hours/cycle|
| ANN p95 latency          | baseline          | -10–15%             | modest         |

The reindex-time win pays for the packet on its own; the ANN-RAM win is a multiplier as the catalogue grows.

The simpler "indexer-side embedding cache only" path captures the TEI-time win without touching the search topology, but loses the storage and ANN-size wins. Decision: go straight to the full split (this packet) rather than ship the cache as a stepping stone.

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

### Path A — vector-first (default)
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

`PATH_B_HASH_LIMIT = 25_000` (env-tunable). Selected from the Milvus IN-clause cost curve documented in [discussion #47136](https://github.com/milvus-io/milvus/discussions/47136) and the [filter-templating doc](https://milvus.io/docs/filtering-templating.md): below 25k the IN-clause parse cost stays under ~50 ms with `filter_params` (templated), well within the search latency budget; above 25k Path A's full-corpus ANN + per-offer post-pass is faster.

Routing trigger heuristic: presence of `vendor_id`, `catalog_version_ids`, `core_marker_*`, or a tight price-scope in the request → Path B (likely selective). Otherwise Path A. The bounded probe in step 1 is the safety net — wrong calls fall back cleanly.

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
- TEI cache keyed by hash: only embed on first occurrence per hash; reuse the fp16 vector for subsequent offers sharing the hash. Cache scoped per bulk run; can be in-memory dict at projected scale (~48M unique embeddings × 256B = 12 GB — fits comfortably on the indexer host) or a Redis sidecar if memory is tight.
- Two row streams per source `(offer, pricings, markers)` record:
  - Article row: idempotent upsert into `articles_v{N}` keyed by hash. Duplicates overwrite identical content — safe.
  - Offer row: insert into `offers_v{N}` keyed by `id`, carrying `article_hash`.
- Article-level envelope columns (`{ccy}_price_min/max`) computed at *the articles upsert layer*: aggregate across all offers seen for the hash so far. This requires a finalisation pass at the end of bulk reindex (or a streaming reduce); document the chosen approach.

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

## Deliverables

- `scripts/create_articles_collection.py` and revised `scripts/create_offers_collection.py`.
- `scripts/MILVUS_ALIAS_WORKFLOW.md` updated for paired-alias swings.
- `indexer/projection.py` with hash computation, TEI cache hook, two-stream emission, and unit tests against `tests/fixtures/mongo_sample/sample_200.json` asserting hash stability and dedup ratio matches the measured 3.33× (within tolerance for the small sample).
- `indexer/bulk.py` orchestrator for the new pipeline (Phase B work in I1 picks this up).
- `search-api/filters.py` split with unit tests covering each scope.
- `search-api/main.py` routing layer + `search-api/hybrid.py` consolidation (single Milvus client, hash-level RRF).
- Integration tests on a fixture-loaded pair of collections demonstrating Path A and Path B paths produce identical hits for representative queries; bounded-probe fallback exercised.
- Doc updates: spec §6 / §7 cross-reference (this packet links to F9 as the canonical design); F1 status banner pointing to F9; F8 packet revised so envelope columns split between the two collections.

## Acceptance

- **Recall parity** vs the pre-dedup single-collection baseline on every fixture query.
- **Storage win measured**: vector + HNSW size on `articles_v{N}` is within 5 % of the predicted 3.33× reduction vs `offers_v{N}`-with-embedding.
- **Reindex time win measured**: TEI calls per bulk run reduced by ≥ 60 % vs the current pipeline (the cache hits the 70 % theoretical max minus first-occurrence overhead).
- **Bounded probe correctness**: Path B fallback to Path A exercised; no request returns a result set that depends on which path served it (paths are equivalent on results, distinct only on path-cost).
- **Alias swing atomic**: a partial failure during the paired swing leaves the system on the previous (consistent) pair, never on a mixed pair.
- **F8 recall parity** preserved end-to-end through the new topology (Path B's envelope filtering still fires; Path A's resolve step still applies the precise priority-resolved price filter).

## Open questions

- **Representative-offer selection for the response.** The wire contract returns `articleId` (per spec §3); each article hash has many offers. Default: pick the offer that ranked first under the requested sort (cheapest matching offer for `sort=price`, lowest articleId for `sort=articleId`, first-by-relevance — i.e. arbitrary but stable — for default). Document the per-sort tiebreak.
- **Path A vs Path B routing heuristic.** The "presence of selective per-offer filter" trigger is heuristic; on adversarial filter shapes it may pick Path A when Path B would be cheaper. Telemetry per-path latency for tuning. Could later evolve into a cost-based dispatcher if signals warrant.
- **Article-level envelope finalisation during streaming reindex.** Bulk: aggregate per hash, emit at end. Incremental (I2): every offer change potentially shifts a per-currency min/max — recompute by reading all offers for the hash on each update? Cache the per-currency min/max on the article row and recompute on demand? Trade-off between read-amplification (recompute on every offer write) and stale-envelope risk (recompute lazily). Default: recompute eagerly on each offer write — the envelope is small and reads dominate writes.
- **`articleIdsFilter` resolution.** Default (a) above: pre-query offers for hash mapping. Measure the round-trip cost on representative requests; if it dominates, evaluate (b) hash-from-request reconstruction.
