# Legacy Search Replacement — Implementation Plan

Sequenced breakdown of `article-search-replacement-spec.md`. Each packet is sized to land in roughly one PR, with explicit dependencies so independent packets can run in parallel.

Three categories, in build order:

1. **ftsearch** (`./search-api/`) — the new search service.
2. **ACL** — new FastAPI service in this repo, in front of ftsearch.
3. **Indexer** — new pipeline projecting MongoDB → Milvus.

---

## Category 1 — ftsearch (`./search-api/`)

| #  | Packet                                                          | Depends on   | File                                                          |
| -- | --------------------------------------------------------------- | ------------ | ------------------------------------------------------------- |
| F1 | Milvus collection schema extension + id format + alias plumbing | —            | `article-search-replacement-ftsearch-01-milvus-schema.md`     |
| F2 | ftsearch HTTP contract (request/response DTO + OpenAPI)         | —            | `article-search-replacement-ftsearch-02-http-contract.md`     |
| F3 | Scalar filtering + price-resolution module                      | F1, F2       | `article-search-replacement-ftsearch-03-filtering.md`         |
| F4 | searchMode + sorting + pagination + accurate hitCount           | F1, F2, F3   | `article-search-replacement-ftsearch-04-mode-sort-paging.md`  |
| F5 | Aggregations / summaries                                        | F1, F2, F3   | `article-search-replacement-ftsearch-05-aggregations.md`      |
| F6 | German identifier tokenization + classifier hardening           | —            | `article-search-replacement-ftsearch-06-german-identifiers.md`|
| F7 | Operational glue (Bounded consistency, tracing, retries)        | F2..F5       | `article-search-replacement-ftsearch-07-operational.md`       |
| F8 | Price-scope pre-filter columns (price-list + per-currency envelope) | F1, F3, I1 | `article-search-replacement-ftsearch-08-price-scope-prefilter.md` |

## Category 2 — ACL (new FastAPI service in this repo)

| #  | Packet                                                            | Depends on        | File                                                           |
| -- | ----------------------------------------------------------------- | ----------------- | -------------------------------------------------------------- |
| A1 | ACL skeleton + narrowed legacy OpenAPI                            | F2                | `article-search-replacement-acl-01-skeleton-openapi.md`        |
| A2 | Legacy → ftsearch request mapper                                  | A1, F2..F5        | `article-search-replacement-acl-02-request-mapper.md`          |
| A3 | ftsearch → legacy response mapper                                 | A1, F2..F5        | `article-search-replacement-acl-03-response-mapper.md`         |
| A4 | Legacy error envelopes + dropped-enum rejection                   | A1                | `article-search-replacement-acl-04-error-contract.md`          |
| A5 | Resilience + observability (retries, timeouts, baggage, metrics)  | A2, A3            | `article-search-replacement-acl-05-resilience-observability.md`|
| A6 | Acceptance test suite (§10)                                       | A1..A5, F1..F7, I1| `article-search-replacement-acl-06-acceptance-suite.md`        |

## Category 3 — Indexer (new pipeline)

| #  | Packet                                                       | Depends on  | File                                                       |
| -- | ------------------------------------------------------------ | ----------- | ---------------------------------------------------------- |
| I1 | Bulk rebuild + canonical MongoDB → Milvus projection module  | F1          | `article-search-replacement-indexer-01-bulk-rebuild.md`    |
| I2 | Kafka-driven incremental upserter                            | I1          | `article-search-replacement-indexer-02-incremental.md`     |
| I3 | Zero-downtime reindex orchestration (alias swap)             | F1, I1      | `article-search-replacement-indexer-03-zero-downtime.md`   |

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
