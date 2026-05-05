# F10 — CV-scope article-side envelope

**Category**: ftsearch (`./search-api/`) + indexer (`./indexer/`) + Milvus schema
**Depends on**: F1 (schema), F3 (filter expr — always-on `_closed_marketplace`), F8 (envelope-column pattern), F9 (article-dedup topology), I1 (projection), I2 (streaming envelope coalescing)
**Unblocks**: revives Path A under always-on legacy CV parity; lifts the bounded-probe recall ceiling for CV-only requests
**Refines**: F1 schema, F3 filter expr, I1 projection, I2 streaming envelopes, F9 routing.

References: spec §4.3 (`closedMarketplaceOnly`), §7 (collection schema), F3 (`_closed_marketplace` always-on switch), F8 (envelope column pattern this packet copies), F9 (Path A / Path B dispatcher).

## Background — the gap this packet closes

F3's always-on legacy CV parity (the iteration that ran F3's `_closed_marketplace` through the legacy `OfferFilterBuilder` switch) plugs the recall hole, but it makes Path A unreachable in production:

- Article-side (`articles_v{N}`) carries no CV scope — the only CV column lives on `offers_v{N}`.
- Every parity-compliant request now has at least one offer-side filter (the always-on CV intersection), so `build_offer_expr` is never `None`.
- Path A is gated on `offer_expr is None` (F9 routing rule). With CV always non-None, every search lands on Path B.

Path B's bounded probe (`PATH_B_HASH_LIMIT = 25 000` distinct hashes; benchmark ceiling for the `article_hash IN [...]` clause on the article ANN) was sized for narrow offer-side filters — vendor, price band, delivery — that legitimately cut the candidate set to a few thousand articles. CV scope is the opposite shape: a large tenant's contracted CV set covers most of the catalog, so the offer probe regularly hits the cap and the dispatcher falls back with `recallClipped: true`. For "browse with CV scope but no other offer-side filter," that's a strict regression vs. the pre-parity-fix behaviour.

The fix is the same envelope pattern F8 used for prices: project a superset of the offer-side scope onto the article side, so the article ANN can apply the CV pre-filter directly via Milvus's HNSW + bitset path. No probe, no cap, no fallback.

## Approach — article-side CV envelope

One new column on `articles_v{N}`:

- **`catalog_version_ids ARRAY<VARCHAR>`** — union of every `catalog_version_id` carried by an offer that maps to this article hash. Indexed `INVERTED`. Pre-filter clause emitted by `build_article_expr`:
  ```
  array_contains_any(catalog_version_ids, [<active CV list>])
  ```
  with the same flag-switch + match-nothing semantics as the offer-side `_closed_marketplace`.

The envelope is **broader** than per-offer CV (it answers "does this article have *any* offer in the requested scope" — Path B's offer probe still verifies which specific offers qualify when other offer-side filters are present), so it cannot drop a hit Path B would have kept. Recall preserved exactly.

Offer-side `_closed_marketplace` stays in `build_offer_expr`. In Path B it bounds the probe to offers actually matching the requested CVs (probe-budget efficiency); in Path A it doesn't run because the offer collection isn't queried. Both sides apply the same CV switch logic — the article side is the always-on filter for the ANN, the offer side is the per-offer narrowing for the probe.

## Why Milvus handles this correctly

Milvus's HNSW + scalar filter uses bitset pre-filtering: build a bitmap of rows passing the predicate, then run the graph traversal restricting candidates to set bits.

- **CV filter is low-selectivity** — a tenant's contracted CV set typically covers most articles, so the bitset is dense.
- **Low-selectivity bitsets are HNSW's happy path** — most graph nodes pass the bit check, traversal proceeds essentially as native HNSW. `ef_search` doesn't need to expand; recall holds.
- **No fixed cap.** The bitset is full-cardinality; whether 25 k or 25 M articles pass, the search runs against the actual filtered set.

This is the opposite of bounded probe's failure mode. Bounded probe caps at `PATH_B_HASH_LIMIT` distinct hashes and silently fallbacks (`recallClipped: true`) when the offer probe overflows — popular tenants with broad CV scope are exactly the ones that overflow. Bitset pre-filter has no such cap.

The degenerate case for bitset filtering is **high-selectivity** filters (matching ≪ 5 % of rows), where graph traversal becomes inefficient because most candidates fail the bit check. Milvus auto-falls-back to brute-force scan in that regime. CV scope filters are never high-selectivity in production traffic — they always cover broad portions of the catalog.

## In scope

### Schema (`indexer/collection_specs.py:build_articles_schema`)

- Add `catalog_version_ids ARRAY<VARCHAR>(64)`, `max_capacity` sized to cover the per-article CV-count distribution. The envelope unions every CV any of the article's offers belongs to; popular generic articles (a Bosch drill bit) may appear under dozens of CVs. **Measure on `mongo_sample/sample_200.json` to bound — start with `max_capacity=128` and expand if sampled p99 exceeds it.**
- Add to `ARTICLE_SCALAR_INDEX_FIELDS`: `INVERTED` index. The bitset pre-filter cost is dominated by index lookup throughput; without an inverted index, Milvus scans every row's array per query.
- Bump `articles_v{N}` to a new `N` per `scripts/MILVUS_ALIAS_WORKFLOW.md`. Column-add — no row migration; rebuild fresh via I1.

### Indexer (`indexer/projection.py`, `indexer/test_loader.py`)

- Extend the article-side projection (the `group_by_hash` aggregation) to derive `catalog_version_ids` as the deduplicated union of `catalog_version_id` values across offers in the hash group. One line in the aggregator alongside the existing per-currency price envelope.
- Mirror in fixtures (`tests/fixtures/offers_schema_smoke.json` if it carries article-side rows; `tests/fixtures/mongo_sample/sample_200.json` projection) so existing smoke tests cover the new column.

### Streaming consumer (I2)

- Add `catalog_version_ids` to the per-hash dirty-buffer flush. Same async-coalescing pattern that already maintains the per-currency price envelopes — when an offer is added, removed, or has its `catalog_version_id` changed, mark the hash dirty and recompute the union on flush.
- Staleness window stays bounded by the existing 5 s flush window (spec §9 #8). For offers whose CV-change event predates a flush, a search may briefly miss them — same semantics as price-envelope staleness, no new failure mode.

### Filter translator (`search-api/filters.py`)

- New atom `_closed_marketplace_article(req)` mirroring the offer-side switch:
  - Flag true → `array_contains_any(catalog_version_ids, [closed_catalog_version_ids])`.
  - Flag false → `array_contains_any(catalog_version_ids, [catalog_version_ids_ordered_by_preference])`.
  - Empty active list → `MATCH_NOTHING_EXPR` (already short-circuited at the dispatch layer).
- Wire into `build_article_expr`. Existing offer-side `_closed_marketplace` stays in `build_offer_expr` — both sides emit the same scope filter against their respective columns.
- `build_milvus_expr` (legacy single-collection path, `USE_DEDUP_TOPOLOGY=false`) is unaffected — its `_closed_marketplace` already operates against the offer-collection's `catalog_version_ids` and stays as-is.

### Routing (`search-api/routing.py`)

- Path A becomes reachable for any request whose only offer-side filter is the CV intersection — the article-side `_closed_marketplace_article` carries it, `build_offer_expr` returns `None`, dispatcher routes to Path A. No code change to the routing rule itself; it now selects the right path organically.
- Path B unchanged for requests with additional offer-side filters (vendor, price, delivery, features, sourcePriceListIds). The article-side CV envelope still runs as part of the ANN's pre-filter, narrowing the candidate set before the probe-driven hash constraint applies.
- `_dispatch_summaries_only` (F5 SUMMARIES_ONLY path) inherits the same routing — summary aggregations over a CV-scoped browse no longer pay the probe-overflow tax.

### Documentation

- Update `spec.md` §4.3 (`closedMarketplaceOnly` row) to reflect always-on intersection on both `articles_v{N}` and `offers_v{N}`.
- Update `spec.md` §7 (schema) with the new article-side column.
- Update `ftsearch-01-milvus-schema.md` with the column + index.
- Update `ftsearch-03-filtering.md` to point at this packet for the article-side atom; offer-side atom note unchanged.
- Update `ftsearch-09-article-dedup.md`: Path A is reachable again post-F10; CV envelope joins prices in the article-side envelope catalogue.
- Update `indexer-01-bulk-rebuild.md` and `indexer-02-incremental.md` projection / streaming notes.

## Out of scope

- **Per-tier CV envelopes** (e.g., `closed_catalog_version_ids ARRAY` separate from `open_catalog_version_ids ARRAY`). Tighter bitset, more storage. The single union covers the vast majority of traffic; defer until telemetry says otherwise.
- **Removing offer-side `_closed_marketplace`.** It still bounds the Path B probe budget. Keep it.
- **Removing Path A.** This packet revives it, doesn't replace it. The dispatcher's selection rule (`offer_expr is None`) is unchanged.
- **Schema migration of the `offers_v{N}` `catalog_version_ids` shape** (the F9 correction note's "scalar `catalog_version_id`" decision). Orthogonal to this packet — F10 builds the article-side union from whatever shape the offer side carries.

## Deliverables

- Schema migration script + new `articles_v{N}` collection with the `catalog_version_ids` column and inverted index.
- `indexer/projection.py` + `indexer/test_loader.py` change with unit tests over `mongo_sample/sample_200.json` asserting the article-side `catalog_version_ids` is the deduplicated union of `catalog_version_id` across the hash group's offers.
- `search-api/filters.py` `_closed_marketplace_article` atom + `build_article_expr` wiring, with unit tests covering each branch (flag on / flag off / empty active list / both lists empty).
- I2 streaming consumer change + recovery test — offer add / remove / CV change updates the article envelope within the 5 s flush window.
- Integration tests on a fixture-loaded paired collection demonstrating: (a) recall parity vs. F9-only path for CV-only requests; (b) Path A is taken for CV-only requests post-F10; (c) `recallClipped: false` on requests that previously overflowed the probe.
- Documentation updates per the list above.

## Acceptance

- **Recall parity.** For every test query in the integration fixture, a CV-scoped request post-F10 returns the same article set as the same request via F9 Path B (or pre-parity-fix F3) with no probe limit. No hit dropped, no hit reordered.
- **Path A reactivation.** A CV-only request (no vendor / price / delivery / etc. filter) routes through Path A; `dispatch_dedup` returns `route="path_a"` and the offer collection is not queried in the per-request flow.
- **Probe overflow eliminated for CV-only browse.** A synthetic test with a tenant whose CV scope previously triggered probe overflow returns `recallClipped: false` post-F10.
- **Indexer round-trip.** `indexer/projection.py` produces the same offer-side rows as before (byte-identical), plus the new article-side column derived from them. Re-running I1 against a fixture twice yields identical envelope contents (deterministic union).
- **Streaming staleness.** I2 round-trip test: an offer add / CV change is reflected in the article envelope within ≤ 5 s (existing flush bound). A search before the flush returns the pre-change result; after the flush, the post-change result. No correctness drift, only the documented staleness window.
- **Inverted index applied.** A sanity check via `MilvusClient.describe_index` confirms `INVERTED` on `articles_v{N}.catalog_version_ids`.

## Open questions for this packet

- **`max_capacity` for the article-side array.** Needs measurement on `mongo_sample/sample_200.json` (and ideally a wider prod-scale projection) to bound the CV-count-per-article distribution. The 159 M-article ES corpus has the data; pick `max_capacity` at p99.5 with comfortable headroom.
- **Inverted-index tuning for low-selectivity arrays.** Milvus 2.6's `INVERTED` on `ARRAY<VARCHAR>` is well-supported but the bitset-build cost grows with the request CV-list size. For a 1 000-CV request, that's 1 000 inverted-index lookups per query before the HNSW search. Worth a microbenchmark against a representative prod-shape collection before committing the column shape.
- **Backfill of the new column on the live `articles_v{N}`.** Column-add on Milvus requires `articles_v{N+1}` and an alias swing per I3. Schedule with the next bulk reindex rather than as a standalone migration.
- **Should `_closed_marketplace_article` and offer-side `_closed_marketplace` share an implementation?** Both emit the same `array_contains_any(...)` shape against different field names. Trivially DRY-able with a `field_name` parameter; defer until both have unit-test coverage so the shared helper has a stable contract.
