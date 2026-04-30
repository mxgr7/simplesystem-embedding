# Legacy Search Replacement — Implementation Plan

Sequenced breakdown of `article-search-replacement-spec.md`. Each packet is sized to land in roughly one PR, with explicit dependencies so independent packets can run in parallel.

Three categories, in build order:

1. **ftsearch** (`./search-api/`) — the new search service.
2. **ACL** — new FastAPI service in this repo, in front of ftsearch.
3. **Indexer** — new pipeline projecting MongoDB → Milvus.

---

Status legend: ✅ done · 🟡 partial · ⬜ not started.

## Category 1 — ftsearch (`./search-api/`)

| #  | Status | Packet                                                          | Depends on   | File                                                          |
| -- | ------ | --------------------------------------------------------------- | ------------ | ------------------------------------------------------------- |
| F1 | ✅ `549516a` | Milvus collection schema extension + id format + alias plumbing | —            | `article-search-replacement-ftsearch-01-milvus-schema.md`     |
| F2 | ✅ `fbcd80c` | ftsearch HTTP contract (request/response DTO + OpenAPI)         | —            | `article-search-replacement-ftsearch-02-http-contract.md`     |
| F3 | ✅ `ad3a361` + `d0cb6f4` | Scalar filtering + price-resolution module                      | F1, F2       | `article-search-replacement-ftsearch-03-filtering.md`         |
| F4 | ✅ `ecc7d7d` | searchMode + sorting + pagination + accurate hitCount           | F1, F2, F3   | `article-search-replacement-ftsearch-04-mode-sort-paging.md`  |
| F5 | ✅ `7e01f4f` | Aggregations / summaries                                        | F1, F2, F3   | `article-search-replacement-ftsearch-05-aggregations.md`      |
| F6 | ✅ (absorbed by F9 PR2) | German identifier tokenization (BM25 analyzer on `articles_v{N}.sparse_codes`) | —            | `article-search-replacement-ftsearch-06-german-identifiers.md`|
| F7 | ✅ `9c9e073` + `4057488` + `6323a86` + `b4198a9` | Operational glue (Bounded consistency, retries, timeouts, RED metrics, W3C tracing) | F2..F5       | `article-search-replacement-ftsearch-07-operational.md`       |
| F8 | ✅ `96fdd1f` | Price-scope pre-filter columns (price-list + per-currency envelope) | F1, F3, I1, F9 | `article-search-replacement-ftsearch-08-price-scope-prefilter.md` |
| F9 | ✅ PR1 `350c09c` · PR2 `78f844b` · PR3 `ffc89ff` · PR2b `e4e5b70` + `8e665b2` + `a4fc8a9` | Article-level dedup topology (split `articles_v{N}` + `offers_v{N+1}`) + DuckDB-native projection | F1, F3, I1 | `article-search-replacement-ftsearch-09-article-dedup.md` |

## Category 2 — ACL (new FastAPI service in this repo)

| #  | Status | Packet                                                            | Depends on        | File                                                           |
| -- | ------ | ----------------------------------------------------------------- | ----------------- | -------------------------------------------------------------- |
| A1 | ✅ `3aaac3d` | ACL skeleton + narrowed legacy OpenAPI                            | F2                | `article-search-replacement-acl-01-skeleton-openapi.md`        |
| A2 | ✅ `3e5e9b3` | Legacy → ftsearch request mapper                                  | A1, F2..F5        | `article-search-replacement-acl-02-request-mapper.md`          |
| A3 | ✅ `17e7638` | ftsearch → legacy response mapper                                 | A1, F2..F5        | `article-search-replacement-acl-03-response-mapper.md`         |
| A4 | ✅ `3c9ee10` | Legacy error envelopes + dropped-enum rejection                   | A1                | `article-search-replacement-acl-04-error-contract.md`          |
| A5 | ✅ `a73c2d6` | Resilience + observability (retries, timeouts, baggage, metrics)  | A2, A3            | `article-search-replacement-acl-05-resilience-observability.md`|
| A6 | 🟡 happy-path `17ddc62` + `52a47ed` | Acceptance test suite (§10) — happy-path landed; per-filter / per-sort / per-aggregation expansion deferred | A1..A5, F1..F7, I1| `article-search-replacement-acl-06-acceptance-suite.md`        |

## Category 3 — Indexer (new pipeline)

| #  | Status | Packet                                                       | Depends on  | File                                                       |
| -- | ------ | ------------------------------------------------------------ | ----------- | ---------------------------------------------------------- |
| I1 | ✅ Phase A `81db037`; Phase B absorbed by F9 PR2b (`58a862e`, `04494e2`, `37546ba`) | Bulk rebuild + canonical MongoDB → Milvus projection module  | F1          | `article-search-replacement-indexer-01-bulk-rebuild.md`    |
| I2 | ⬜ design ready | Kafka-driven incremental upserter (consumer + async envelope coalescer + orphan GC) — design landed; implementation blocked on Kafka infra | I1, F9      | `article-search-replacement-indexer-02-incremental.md`     |
| I3 | 🟡 alias-swing CLI `0ab059f` + design ready | Zero-downtime reindex orchestration (alias swap) — paired-swing CLI shipped; full orchestration design landed (state machine, dual-write via second consumer, validation gates, recovery, runbook outline). Implementation pending I2 | F1, I1, I2  | `article-search-replacement-indexer-03-zero-downtime.md`   |

---

## Suggested execution order across categories

- **Phase 1 — foundation** (✅ landed): F1 ‖ F2; F3; I1 Phase A. F6 is absorbed by F9 and no longer ships standalone.
- **Phase 2 — topology pivot** (✅ landed): F9 (PR1 schema split, PR2 indexer two-stream emit, PR3 dispatcher routing, PR2b DuckDB-native projection + bulk-insert sink). Absorbed I1 Phase B (article-aggregation grouping) and F6 (German tokenizer on `articles_v{N}.sparse_codes`); unblocked F4's sort-by-price browse path and F8's envelope-column placement.
- **Phase 3 — ftsearch capability build-out (on new topology)** (✅ landed): F4, F5, F8.
- **Phase 4 — ACL + operational** (✅ landed): A1 → (A2 ‖ A3 ‖ A4) → A5; F7 ‖ I3-partial in parallel. I2 deferred (no Kafka infra yet).
- **Phase 5 — acceptance** (🟡 in progress): A6 happy-path landed; per-filter / per-sort / per-aggregation expansion + PostHog-captured-traffic smoke deferred (mechanical work, see `ARTICLE_SEARCH_REPLACEMENT_STATUS.md`).

The longest critical path was F1 → F3 → **F9** → F4 → F5 → A2/A3 → A6. All landed. Remaining open work is **I2** (Kafka incremental — design ready, implementation blocked on infra) and the I3 + A6 expansion that I2 unblocks; see `ARTICLE_SEARCH_REPLACEMENT_STATUS.md` for the bird's-eye view + smaller follow-ups (full-catalog GPU TEI run, legacy collection drop).

---

## Cross-cutting findings (added during execution)

- **I3 first-cutover dependency** — Milvus disallows an alias whose name matches an existing collection, so the legacy `offers` collection (159M rows) must be renamed or dropped before the `offers` alias can be created. Steady-state swings (v{N} → v{N+1}) are unaffected. Procedure documented in `scripts/MILVUS_ALIAS_WORKFLOW.md` and folded into I3's design pass (`article-search-replacement-indexer-03-zero-downtime.md`) as a deployment precondition.
- **Shared fixtures live under `tests/fixtures/`** — `offers_schema_smoke.json` (8 hand-crafted Milvus-shape rows covering every §7 field + edge cases) and `mongo_sample/sample_200.json` (200 real `prod.offers` docs joined to matching `pricings` + `coreArticleMarkers`). F3..F5 and I1 should reuse these rather than re-source.
- **`closed_catalog` is a phantom column** (corrected during F3) — legacy `OfferFilterBuilder` consumes `closedMarketplaceOnly` as a switch over which CV-list to intersect (`closedCatalogVersionIds` vs `catalogVersionIdsOrderedByPreference`), not as a row-level boolean. F1 schema + F3 filter were both built on the misread; both fixed in `d0cb6f4`. Spec §4.3, §7, F1, F3, I1 packet docs all updated. ftsearch design choice (per the same fix): make CV scoping optional in ftsearch; the ACL is the layer that re-adds always-intersect for legacy parity.
- **Article-aggregation gap** (surfaced during I1A vs prod-ES comparison) — legacy collapses N MongoDB `OfferDocument` rows for the same `(vendorId, articleNumber)` into one ES doc with `offers[]` of length N and union'd `prices[]` / `catalogVersionIds`. Phase A's projection treats one MongoDB row as one Milvus row; the 200-doc sample doesn't bite because `$sample:200` over millions yields 0 dupe keys, but production fidelity needs a grouping step (`db.offers.aggregate([{$group: {_id: {vendorId, articleNumber}, ...}}])` — uses the existing `(vendorId, articleNumber)` compound index). To resolve in I1 Phase B.
- **eClass hierarchy gap** (surfaced during I1A vs prod-ES comparison; **resolved**) — ES stores full hierarchy as `offers.eclass51Groups: ["23000000","23110000","23110100","23110101"]` (root→leaf, multi-valued keyword array). Original F1 schema + I1 projection collapsed to a single leaf int, breaking parent-level recall (and silently matching only when the leaf landed at index 0 of the legacy `Set<Integer>`). **Fix**: promoted `eclass{5,7}_code` and `s2class_code` to `ARRAY<INT32>`; projection copies the array verbatim; F3 emits `array_contains[_any]`. Spec §7, F1, F3, I1 packet docs all updated. Operators must drop and recreate `offers_v{N}` for the schema change to take effect.
- **Naming nit: `priceListId` vs `sourcePriceListId`** — legacy ES uses `priceListId` inside the prices array; spec §7 + our projection use `sourcePriceListId`. Internal-only, no external consumer affected. AGENTS.md flags it; cross-referencing only.
- **Streaming envelope coalescing + cutover dual-write architecture** (design landed in I2 + I3 packets) — F9's article-side per-currency envelope columns aggregate over every offer in the hash group; per-event eager recompute on a 1M-event burst (vendor mass price upload) would issue ~5M Milvus operations and saturate the cluster for 30 minutes. I2's design pass selects async coalescing with a per-hash dirty buffer + 5s flush window (~25× speedup, staleness inside spec §9 #8 tolerance) and accepts an in-memory buffer with documented degraded-recovery semantics (envelope-only loss on crash; offer rows always current). For I3's cutover, the original "dual-write config flag on the consumer" approach was rejected in favour of launching a **second I2 consumer instance** with its own consumer group writing to the staging pair — same cost shape, simpler code, independent observability per consumer, bounded crash blast radius. Both designs documented in `article-search-replacement-indexer-02-incremental.md` and `article-search-replacement-indexer-03-zero-downtime.md`; implementation gated on Kafka infra provisioning.

- **Article-level dedup topology** (✅ landed — F9 across PRs 1/2/3/2b) — production scale: 510M raw MongoDB offers → 159M articles after `(vendorId, articleNumber)` aggregation (the offer→article 3.21× step that I1 Phase B owned) → 130M unique embeddings after embedded-field-hash dedup (the article→hash 1.22× step that F9 owns). **The packet was not justified by storage savings vs the prior 159M-article topology** (~13 GB HNSW RAM, ~8 hours TEI saved — modest). It is justified because the prior article-level union semantics could not express **correlated per-offer filters** (catalog × price-list × price-range applying to the same offer), and the only alternative that does — flattening to one row per offer — would pay 510M embeddings. F9's split delivers the same correlated-filtering capability at 130M embeddings (97 GB storage / 170 GB HNSW RAM / 106 hours TEI saved per cycle vs the 510M-flat alternative). Storage shape: `articles_v{N}` (vector + BM25 codes + article-level scalars + article-level per-currency envelope) and `offers_v{N+1}` (per-offer scalars including F8's per-offer envelope, `article_hash` join key, no vectors). Routing rule is **deterministic by `offer_expr` presence**, not heuristic: empty → Path A; present → Path B with bounded probe at 25 k hashes (hardware ceiling validated by IN-clause cost benchmark — 25 k = ~430 ms p95 on Milvus 2.6 + CPU; templated `filter_params` saves only 1–3 % over string IN; parallel-batched search plateaus at ~880 ms for 100 k, brute-force via `query()` is much worse, only GPU index would lift the ceiling). Probe overflow → Path A fallback with under-recall accepted, surfaced as `metadata.recall_clipped: true`; documented as a deviation in spec §2.4. F9 ships bulk envelope only — streaming envelope updates (I2) must use async coalescing because production write traffic comes in bursts; eager per-event recompute would saturate the cluster. I1 Phase B was fully absorbed by F9 PR2 (no separate I1 packet shipped). Today's `offers_codes` BM25 folded into `articles_v{N}.sparse_codes` (F6 absorption). Canonical design lives in `article-search-replacement-ftsearch-09-article-dedup.md`. Spec §2.3 captures the relevance-pool bound on non-relevance sorts (independent decision; needed regardless of dedup).
