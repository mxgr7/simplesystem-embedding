# F8 — Price-scope pre-filter columns (price-list scope + per-currency range envelope)

**Category**: ftsearch (`./search-api/`) + indexer (`./indexer/`) + Milvus schema
**Depends on**: F1 (schema), F3 (price-resolution module), I1 (projection), F9 (article-dedup topology — envelope columns split across the two collections)
**Unblocks**: —
**Refines**: F1 schema, F3 filter expr, I1 projection.

References: spec §7 (collection schema), §4.3 (filter table — `sourcePriceListIds`, `priceFilter`, top-level `currency`), §3 (currency two-roles), F9 (this packet's columns are duplicated across `articles_v{N}` and `offers_v{N}` per the F9 split — see "Column placement" below).

## Column placement under F9 topology

F9 splits storage into `articles_v{N}` (vector + article-level scalars) and `offers_v{N}` (per-offer scalars). F8's envelope columns split correspondingly:

- **`articles_v{N}.{ccy}_price_min/max`** — per-currency envelope across *all the article's offers*. Used only by the sort-by-price browse path (no queryString) for ordered scan, not for filtering. Refreshed at bulk reindex; streaming updates owned by I2.
- **`offers_v{N}.{ccy}_price_min/max`, `offers_v{N}.price_list_ids`, `offers_v{N}.currencies`** — per-offer scope. Path B's probe filter consults these to narrow the matching-hash set before the article-collection ANN. This is where F8's recall/latency win for narrow-scope traffic actually lands.

The schema/projection/filter sections below describe the per-offer side (the Path B probe). The article-side envelope is described in F9's "Topology" section and `scripts/create_articles_collection.py`.

## Background — the gap this packet closes

F3 ships price filtering as a pure post-Milvus pass: ANN runs with no price-related scalar predicate, the page is over-fetched (default `PRICE_FILTER_OVERFETCH_N = 10`), and `search-api/prices.py` resolves `currency × sourcePriceListIds × priority` and applies `priceFilter.min/max` in Python. Spec §7 explicitly takes this position (line 389: *"there is no single denormalized scalar to project"*) and §8 maps `priceFilter` to a JSON post-pass.

That works for typical queries but degrades silently at the recall edge:

- **Narrow price-list scope.** A customer whose contracted `sourcePriceListIds` cover only a small share of the catalogue: ANN spends its top-k budget on articles with no qualifying price entry. The post-pass discards them and the page starves. The fixed `N=10` over-fetch is a band-aid that fails at the long tail.
- **Narrow currency scope.** Non-EUR queries (CHF/HUF/PLN/GBP/CZK/CNY together account for ~2% of price entries on prod). The post-pass throws away ~98% of ANN hits and the page collapses.
- **Narrow `priceFilter.min/max` band.** A tight range like `[4990, 5010]` EUR can drop most of the over-fetched candidates regardless of how loose the price-list scope is.

In all three cases the precise filter cannot be pushed into Milvus (same-element JSON constraints aren't expressible). But a **conservative envelope** can: a pre-filter that is guaranteed to be a *superset* of the precise filter loses no recall while letting the bitset prune ANN before the graph walk.

## Approach — superset envelopes

Three new columns / column families, each populated from the same `prices` JSON the indexer already builds:

1. **`price_list_ids ARRAY<VARCHAR>`** — union of every `prices[].sourcePriceListId` on the article. Pre-filter clause: `array_contains_any(price_list_ids, [request.sourcePriceListIds])`. Drops articles with no contracted price entry at all.
2. **`currencies ARRAY<VARCHAR>`** — union of every `prices[].currency` on the article (small set in practice: 1–7 symbols). Pre-filter clause: `array_contains(currencies, request.currency)`. Drops articles with no entry in the request currency.
3. **Per-currency `(min, max)` envelope** — for each currency the catalogue carries, two FLOAT columns spanning *every* price the article has in that currency, irrespective of priceList or priority. Pre-filter clause:
   ```
   {ccy}_price_min <= decode(priceFilter.max, currencyCode)
   AND {ccy}_price_max >= decode(priceFilter.min, currencyCode)
   ```
   Drops articles whose entire range in that currency is outside the requested band. Selected dynamically per request: only the currency named in top-level `currency` is consulted.

The envelope is **broader** than the precise filter (it ignores priceList scope and priority selection), so it cannot drop a hit that the post-pass would have kept — recall preserved exactly. False positives (envelope passes, precise pass rejects) flow into the existing `prices.py` resolution and are dropped there. The post-pass workload shrinks because the pre-filter has already removed the obvious misses.

## In scope

### Schema (`scripts/create_offers_collection.py`)

- Add `price_list_ids ARRAY<VARCHAR>(64)`, `max_capacity` sized to cover the per-article price-list count distribution (median 4, max ~470 from sampled prod data — set `max_capacity=512`).
- Add `currencies ARRAY<VARCHAR>(8)`, `max_capacity=8` (catalogue carries 7 currencies today).
- For each currency in the catalogue, add `{ccy}_price_min FLOAT` and `{ccy}_price_max FLOAT`. Initial set: `eur_price_min/max`, `chf_price_min/max`, `huf_price_min/max`, `pln_price_min/max`, `gbp_price_min/max`, `czk_price_min/max`, `cny_price_min/max`. Convention: **column-name list is data-driven, not hard-coded** — pull from a config so adding a new currency is one line. NaN sentinel for "no price in this currency on this row" so the range predicate naturally excludes it.
- Scalar indexes: `INVERTED` on `price_list_ids` and `currencies`; `STL_SORT` on every `*_price_min` / `*_price_max` (range queries are the hot path).
- Bump `offers_v{N}` to a new `N` per the alias-swing playbook in `scripts/MILVUS_ALIAS_WORKFLOW.md`. This is a column-add — no live row migration, the new collection is built fresh by I1.

### Indexer (`indexer/projection.py`)

- Extend `project()` to derive the three new column families in the same pass that already builds the `prices` JSON column:
  - `price_list_ids` = `sorted(set(p["sourcePriceListId"] for p in prices))`.
  - `currencies` = `sorted(set(p["currency"] for p in prices))`.
  - For each currency listed in the new config: `min(p.price for p in prices if p.currency == ccy)` and `max(...)`. Emit NaN when the currency is absent on the article.
- Mirror in `tests/fixtures/offers_schema_smoke.json` and `tests/fixtures/mongo_sample/sample_200.json` so existing smoke tests cover the new columns.

### Filter translator (`search-api/filters.py`)

- New clauses, AND-composed with the existing expr:
  - When `selectedArticleSources.sourcePriceListIds` is non-empty: `array_contains_any(price_list_ids, [...])`.
  - When top-level `currency` is set: `array_contains(currencies, currency)`.
  - When `priceFilter` is set with `min` and/or `max`: `{ccy}_price_min <= decoded_max AND {ccy}_price_max >= decoded_min`, where `{ccy}` is `top-level currency.lower()` (NOT `priceFilter.currencyCode` — currency two-roles split per §3 still holds; `priceFilter.currencyCode` only decodes bound integers into decimals).
  - Each new clause is **additive only** — emit it when the corresponding request field is set, omit it otherwise. Existing F3 clauses unchanged.

### Price-resolution post-pass (`search-api/prices.py`)

- The post-pass remains in place — it's still required to pick the priority-resolved entry and apply the precise `min/max` against that single price. The new pre-filter shrinks the page it operates on; it does not replace it.
- Drop the now-redundant initial scope check **only if** there's a measurable win. Default: leave the precise pass untouched and treat the pre-filter as a pure recall/latency optimization.

### Documentation

- Update `spec.md` §7 (schema), §8.B/§8.C (filter translation table), §3.2 (clarify that `priceFilter` is now part-pre-filter / part-post-pass).
- Update `ftsearch-01-milvus-schema.md` with the new columns and indexes.
- Update `ftsearch-03-filtering.md` to reference F8 for the pre-filter clauses, leaving the price-resolution module unchanged.
- Update `indexer-01-bulk-rebuild.md` projection notes.

## Out of scope

- **Per-priority envelope columns** (`{ccy}_p{1..4}_min/max`). Tighter envelope, more columns. Defer until telemetry from the per-currency envelope shows the post-pass is still the latency bottleneck on real traffic.
- **Iterator-based recall fallback.** `MilvusClient.search_iterator` for pathological narrow queries that even the envelope can't tighten. Possible future packet; not needed if the per-currency envelope handles 99% of traffic, which the prod data distribution suggests it will.
- **Adaptive `top_k`.** Bumping over-fetch when the request hints at a narrow scope. Cheap optimization but orthogonal to this packet.

## Deliverables

- Schema migration script + new `offers_v{N}` collection.
- `indexer/projection.py` with the new column projections, plus unit tests over `mongo_sample/sample_200.json` asserting:
  - `price_list_ids` is the deduplicated union of `prices[].sourcePriceListId`.
  - `currencies` is the deduplicated union of `prices[].currency`.
  - For each currency-min/max column: equals `min`/`max` over filtered prices, NaN when absent.
- `search-api/filters.py` clause emission with unit tests covering each new clause in isolation and AND-composition with existing clauses.
- Integration test on a fixture-loaded collection demonstrating that a tight `priceFilter` returns the same hits as the F3 post-pass-only path (recall parity) but with `top_k = pageSize` instead of `pageSize × N` (latency win).
- README/spec updates per the documentation list above.

## Acceptance

- **Recall parity.** For every test query in `tests/test_search_filters_integration_real.py`, the result set with F8 pre-filters enabled is identical to the result set with F8 disabled (post-pass-only F3 path). No hit dropped, no hit reordered.
- **Pre-filter activation.** Logging or a per-request metric records that the pre-filter clauses were emitted on requests carrying `sourcePriceListIds`, top-level `currency`, or `priceFilter` — and that the over-fetch factor stayed at `1×` (no inflation needed) on the same requests.
- **Narrow-scope cases recover.** A synthetic test with a customer scoped to a single rare price-list — the F3-only path starves the page; the F8 path returns a full `pageSize`.
- **Indexer round-trip.** `indexer/projection.py` produces the same `prices` JSON column as before (byte-identical), plus the new columns derived from it. The price-resolution post-pass on the new collection produces the same per-row decisions as on a collection without the new columns.
- **NaN handling.** A row with no price in a given currency has `{ccy}_price_min = {ccy}_price_max = NaN`; the range predicate excludes it (Milvus comparison against NaN is false), as desired.

## Open questions for this packet

- **Currency column set is data-driven** — confirm the config source. Today the seven currencies are observable from the live ES aggregation; whether the projection reads them from a hard-coded list, a config file, or queries Milvus for existing columns at startup is an implementation choice.
- **Should the precise post-pass scope check be dropped?** Default: keep it for defence-in-depth. Re-evaluate after measuring the false-positive rate against real traffic.
- **Backfill of the new columns into the live `offers_v{N}`** — column-add on an existing Milvus collection requires a fresh `offers_v{N+1}` and an alias swing per I3. Schedule this with the next bulk reindex rather than as a standalone migration.
