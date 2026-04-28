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
| F4 | ⬜ | searchMode + sorting + pagination + accurate hitCount           | F1, F2, F3   | `article-search-replacement-ftsearch-04-mode-sort-paging.md`  |
| F5 | ⬜ | Aggregations / summaries                                        | F1, F2, F3   | `article-search-replacement-ftsearch-05-aggregations.md`      |
| F6 | ⬜ (absorbed by F9) | German identifier tokenization + classifier hardening           | —            | `article-search-replacement-ftsearch-06-german-identifiers.md`|
| F7 | ⬜ | Operational glue (Bounded consistency, tracing, retries)        | F2..F5       | `article-search-replacement-ftsearch-07-operational.md`       |
| F8 | ⬜ | Price-scope pre-filter columns (price-list + per-currency envelope) | F1, F3, I1, F9 | `article-search-replacement-ftsearch-08-price-scope-prefilter.md` |
| F9 | ⬜ | Article-level dedup topology (split `articles_v{N}` + `offers_v{N}`) | F1, F3, I1 | `article-search-replacement-ftsearch-09-article-dedup.md` |

## Category 2 — ACL (new FastAPI service in this repo)

| #  | Status | Packet                                                            | Depends on        | File                                                           |
| -- | ------ | ----------------------------------------------------------------- | ----------------- | -------------------------------------------------------------- |
| A1 | ⬜ | ACL skeleton + narrowed legacy OpenAPI                            | F2                | `article-search-replacement-acl-01-skeleton-openapi.md`        |
| A2 | ⬜ | Legacy → ftsearch request mapper                                  | A1, F2..F5        | `article-search-replacement-acl-02-request-mapper.md`          |
| A3 | ⬜ | ftsearch → legacy response mapper                                 | A1, F2..F5        | `article-search-replacement-acl-03-response-mapper.md`         |
| A4 | ⬜ | Legacy error envelopes + dropped-enum rejection                   | A1                | `article-search-replacement-acl-04-error-contract.md`          |
| A5 | ⬜ | Resilience + observability (retries, timeouts, baggage, metrics)  | A2, A3            | `article-search-replacement-acl-05-resilience-observability.md`|
| A6 | ⬜ | Acceptance test suite (§10)                                       | A1..A5, F1..F7, I1| `article-search-replacement-acl-06-acceptance-suite.md`        |

## Category 3 — Indexer (new pipeline)

| #  | Status | Packet                                                       | Depends on  | File                                                       |
| -- | ------ | ------------------------------------------------------------ | ----------- | ---------------------------------------------------------- |
| I1 | 🟡 Phase A `81db037`; Phase B absorbed by F9 PR2 | Bulk rebuild + canonical MongoDB → Milvus projection module  | F1          | `article-search-replacement-indexer-01-bulk-rebuild.md`    |
| I2 | ⬜ | Kafka-driven incremental upserter                            | I1          | `article-search-replacement-indexer-02-incremental.md`     |
| I3 | ⬜ | Zero-downtime reindex orchestration (alias swap)             | F1, I1      | `article-search-replacement-indexer-03-zero-downtime.md`   |

---

## Suggested execution order across categories

- **Phase 1 — foundation** (✅ landed): F1 ‖ F2; F3; I1 Phase A. F6 is absorbed by F9 and no longer ships standalone.
- **Phase 2 — topology pivot**: F9. Reshapes the storage topology before further capability work lands on top of the to-be-replaced single-collection shape. Absorbs I1 Phase B (article-aggregation grouping) and F6 (German tokenizer on `articles_v{N}.sparse_codes`); unblocks F4's sort-by-price browse path and F8's envelope-column placement.
- **Phase 3 — ftsearch capability build-out (on new topology)**: F4 ‖ F5 ‖ F8. A1 can start in parallel since it depends only on F2 frozen.
- **Phase 4 — ACL + operational**: A1 → (A2 ‖ A3 ‖ A4) → A5; in parallel F7 ‖ I2 ‖ I3. I2 owns the streaming envelope writer that F9 deferred (async coalescing per the F9 packet).
- **Phase 5 — acceptance**: A6 closes out against the full stack.

The longest critical path is F1 → F3 → **F9** → F4 → F5 → A2/A3 → A6. F9's promotion lengthens the critical path by one packet but avoids rework on F4/F5/F8 against the soon-to-be-replaced single-collection topology, and lets F8 ship its envelope columns split correctly across the two collections from the start.

---

## Cross-cutting findings (added during execution)

- **I3 first-cutover dependency** — Milvus disallows an alias whose name matches an existing collection, so the legacy `offers` collection (159M rows) must be renamed or dropped before the `offers` alias can be created. Steady-state swings (v{N} → v{N+1}) are unaffected. Procedure documented in `scripts/MILVUS_ALIAS_WORKFLOW.md`; I3 should fold this into its cutover playbook.
- **Shared fixtures live under `tests/fixtures/`** — `offers_schema_smoke.json` (8 hand-crafted Milvus-shape rows covering every §7 field + edge cases) and `mongo_sample/sample_200.json` (200 real `prod.offers` docs joined to matching `pricings` + `coreArticleMarkers`). F3..F5 and I1 should reuse these rather than re-source.
- **`closed_catalog` is a phantom column** (corrected during F3) — legacy `OfferFilterBuilder` consumes `closedMarketplaceOnly` as a switch over which CV-list to intersect (`closedCatalogVersionIds` vs `catalogVersionIdsOrderedByPreference`), not as a row-level boolean. F1 schema + F3 filter were both built on the misread; both fixed in `d0cb6f4`. Spec §4.3, §7, F1, F3, I1 packet docs all updated. ftsearch design choice (per the same fix): make CV scoping optional in ftsearch; the ACL is the layer that re-adds always-intersect for legacy parity.
- **Article-aggregation gap** (surfaced during I1A vs prod-ES comparison) — legacy collapses N MongoDB `OfferDocument` rows for the same `(vendorId, articleNumber)` into one ES doc with `offers[]` of length N and union'd `prices[]` / `catalogVersionIds`. Phase A's projection treats one MongoDB row as one Milvus row; the 200-doc sample doesn't bite because `$sample:200` over millions yields 0 dupe keys, but production fidelity needs a grouping step (`db.offers.aggregate([{$group: {_id: {vendorId, articleNumber}, ...}}])` — uses the existing `(vendorId, articleNumber)` compound index). To resolve in I1 Phase B.
- **eClass hierarchy gap** (surfaced during I1A vs prod-ES comparison; **resolved**) — ES stores full hierarchy as `offers.eclass51Groups: ["23000000","23110000","23110100","23110101"]` (root→leaf, multi-valued keyword array). Original F1 schema + I1 projection collapsed to a single leaf int, breaking parent-level recall (and silently matching only when the leaf landed at index 0 of the legacy `Set<Integer>`). **Fix**: promoted `eclass{5,7}_code` and `s2class_code` to `ARRAY<INT32>`; projection copies the array verbatim; F3 emits `array_contains[_any]`. Spec §7, F1, F3, I1 packet docs all updated. Operators must drop and recreate `offers_v{N}` for the schema change to take effect.
- **Naming nit: `priceListId` vs `sourcePriceListId`** — legacy ES uses `priceListId` inside the prices array; spec §7 + our projection use `sourcePriceListId`. Internal-only, no external consumer affected. AGENTS.md flags it; cross-referencing only.
- **Article-level dedup topology** (planned, not yet started — F9) — production scale: 510M raw MongoDB offers → 159M articles after `(vendorId, articleNumber)` aggregation (the offer→article 3.21× step that I1 Phase B owns) → 130M unique embeddings after embedded-field-hash dedup (the article→hash 1.22× step that F9 owns). **The packet is not justified by storage savings vs the current 159M-article topology** (~13 GB HNSW RAM, ~8 hours TEI saved — modest). It is justified because today's article-level union semantics cannot express **correlated per-offer filters** (catalog × price-list × price-range applying to the same offer), and the only alternative that does — flattening to one row per offer — would pay 510M embeddings. F9's split delivers the same correlated-filtering capability at 130M embeddings (97 GB storage / 170 GB HNSW RAM / 106 hours TEI saved per cycle vs the 510M-flat alternative). Decision: split storage into `articles_v{N}` (vector + BM25 codes + article-level scalars + article-level per-currency envelope) and `offers_v{N}` (per-offer scalars including F8's per-offer envelope, `article_hash` join key, no vectors). Routing rule is **deterministic by `offer_expr` presence**, not heuristic: empty → Path A; present → Path B with bounded probe at 25 k hashes (hardware ceiling validated by IN-clause cost benchmark — 25 k = ~430 ms p95 on Milvus 2.6 + CPU; templated `filter_params` saves only 1–3 % over string IN; parallel-batched search plateaus at ~880 ms for 100 k, brute-force via `query()` is much worse, only GPU index would lift the ceiling). Probe overflow → Path A fallback with under-recall accepted, surfaced as `metadata.recall_clipped: true`; documented as a deviation in spec §2.4. F9 ships bulk envelope only — streaming envelope updates (I2) must use async coalescing because production write traffic comes in bursts; eager per-event recompute would saturate the cluster. **I1 Phase B is fully absorbed by F9 PR2** — do not ship Phase B as a separate I1 packet. Folds today's `offers_codes` BM25 into `articles_v{N}.sparse_codes` (F6 absorption). Canonical design lives in `article-search-replacement-ftsearch-09-article-dedup.md`. Spec §2.3 captures the relevance-pool bound on non-relevance sorts (independent decision; needed regardless of dedup).
