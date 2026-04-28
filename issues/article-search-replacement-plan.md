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
| F6 | ⬜ | German identifier tokenization + classifier hardening           | —            | `article-search-replacement-ftsearch-06-german-identifiers.md`|
| F7 | ⬜ | Operational glue (Bounded consistency, tracing, retries)        | F2..F5       | `article-search-replacement-ftsearch-07-operational.md`       |
| F8 | ⬜ | Price-scope pre-filter columns (price-list + per-currency envelope) | F1, F3, I1 | `article-search-replacement-ftsearch-08-price-scope-prefilter.md` |

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
| I1 | 🟡 Phase A `81db037` | Bulk rebuild + canonical MongoDB → Milvus projection module  | F1          | `article-search-replacement-indexer-01-bulk-rebuild.md`    |
| I2 | ⬜ | Kafka-driven incremental upserter                            | I1          | `article-search-replacement-indexer-02-incremental.md`     |
| I3 | ⬜ | Zero-downtime reindex orchestration (alias swap)             | F1, I1      | `article-search-replacement-indexer-03-zero-downtime.md`   |

---

## Suggested execution order across categories

- **Phase 1 — unblock everything**: F1 ‖ F2 ‖ F6. Schema, contract, and the classifier work that has no dependencies on either.
- **Phase 2 — ftsearch capability build-out**: F3 → F4 ‖ F5; I1 starts as soon as F1 lands so capability work can be exercised on populated data.
- **Phase 3 — ACL build-out**: A1 → (A2 ‖ A3 ‖ A4); requires F2 frozen.
- **Phase 4 — operational + reindex**: F7 ‖ A5 ‖ I2 ‖ I3. F8 lands here too — it requires F1/F3/I1 frozen and pairs naturally with the next I3 alias swing.
- **Phase 5 — acceptance**: A6 closes out against the full stack.

The longest critical path is F1 → F3 → F4 → F5 → A2/A3 → A6.

---

## Cross-cutting findings (added during execution)

- **I3 first-cutover dependency** — Milvus disallows an alias whose name matches an existing collection, so the legacy `offers` collection (159M rows) must be renamed or dropped before the `offers` alias can be created. Steady-state swings (v{N} → v{N+1}) are unaffected. Procedure documented in `scripts/MILVUS_ALIAS_WORKFLOW.md`; I3 should fold this into its cutover playbook.
- **Shared fixtures live under `tests/fixtures/`** — `offers_schema_smoke.json` (8 hand-crafted Milvus-shape rows covering every §7 field + edge cases) and `mongo_sample/sample_200.json` (200 real `prod.offers` docs joined to matching `pricings` + `coreArticleMarkers`). F3..F5 and I1 should reuse these rather than re-source.
- **`closed_catalog` is a phantom column** (corrected during F3) — legacy `OfferFilterBuilder` consumes `closedMarketplaceOnly` as a switch over which CV-list to intersect (`closedCatalogVersionIds` vs `catalogVersionIdsOrderedByPreference`), not as a row-level boolean. F1 schema + F3 filter were both built on the misread; both fixed in `d0cb6f4`. Spec §4.3, §7, F1, F3, I1 packet docs all updated. ftsearch design choice (per the same fix): make CV scoping optional in ftsearch; the ACL is the layer that re-adds always-intersect for legacy parity.
- **Article-aggregation gap** (surfaced during I1A vs prod-ES comparison) — legacy collapses N MongoDB `OfferDocument` rows for the same `(vendorId, articleNumber)` into one ES doc with `offers[]` of length N and union'd `prices[]` / `catalogVersionIds`. Phase A's projection treats one MongoDB row as one Milvus row; the 200-doc sample doesn't bite because `$sample:200` over millions yields 0 dupe keys, but production fidelity needs a grouping step (`db.offers.aggregate([{$group: {_id: {vendorId, articleNumber}, ...}}])` — uses the existing `(vendorId, articleNumber)` compound index). To resolve in I1 Phase B.
- **eClass hierarchy gap** (surfaced during I1A vs prod-ES comparison) — ES stores full hierarchy as `offers.eclass51Groups: ["23000000","23110000","23110100","23110101"]` (root→leaf, multi-valued keyword array). Spec §7 + F1 schema collapse to a single leaf int (`eclass5_code INT`). Filtering on a parent code via `eClassesFilter=[23110100]` matches in legacy but misses in our schema. Either promote the column to `ARRAY<INT>` projecting full hierarchy, or record as a deviation in spec §2. Open decision.
- **Naming nit: `priceListId` vs `sourcePriceListId`** — legacy ES uses `priceListId` inside the prices array; spec §7 + our projection use `sourcePriceListId`. Internal-only, no external consumer affected. AGENTS.md flags it; cross-referencing only.
