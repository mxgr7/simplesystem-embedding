# F1 — Milvus collection schema extension + id format + alias plumbing

**Category**: ftsearch (`./search-api/`)
**Depends on**: —
**Unblocks**: F3, F4, F5, I1, I3

References: spec §4.8, §6, §7, §9 #4.

## Scope

Bring the dense `offers` Milvus collection schema up to parity-readiness: add every scalar field §7 calls out, widen the primary key to carry the legacy `articleId` composite, and stand up the alias-based deployment plumbing required by §4.8.

This packet only defines and creates the schema. Population is I1; consumption is F3..F5.

## In scope

- New collection schema with all of §7:
  - `vendor_id` (see open question on cardinality)
  - `prices JSON` — full legacy nested-prices array, projected verbatim
  - `delivery_time_days_max INT`
  - `core_marker_enabled_sources ARRAY<STRING>`, `core_marker_disabled_sources ARRAY<STRING>`
  - `eclass5_code INT`, `eclass7_code INT`, `s2class_code INT`
  - `features ARRAY<VARCHAR>` of `name=value` tokens (separator `=`)
  - `relationship_accessory_for ARRAY<STRING>`, `relationship_spare_part_for ARRAY<STRING>`, `relationship_similar_to ARRAY<STRING>`
  - `closed_catalog BOOL`
  - retain `name`, `manufacturerName`, `ean`, `article_number`, `catalog_version_ids`, `category_l1..l5`, `offer_embedding`
- Widen `id` to comfortably hold `{friendlyId}:{base64Url(articleNumber)}` (current cap is 64; pick ≥ 256 and document the choice).
- Scalar indexes on the fields that filtering will hit on the hot path: `vendor_id`, `eclass5_code`, `eclass7_code`, `s2class_code`, `closed_catalog`, `delivery_time_days_max`. Confirm Milvus index types per field type.
- Schema-creation script under `scripts/` that creates the collection (default name `offers_v2`, configurable) and registers a Milvus alias (default `offers`) pointing at it (`MilvusClient.alter_alias`).
- ftsearch (`search-api/main.py`) keeps the path-param contract (`/{collection}/_search`) — alias resolution is an operator concern, not an API one. Document this explicitly.
- Operational notes: how to bring up a new collection, how to register the alias, what to do during a swap.

## Out of scope

- Populating the collection — I1.
- Filtering on the new fields — F3.
- Aggregating on the new fields — F5.
- Old-collection retirement / dual-read — I3.

## Deliverables

- `scripts/create_offers_collection.py` (or extension to `milvus_import.py`) producing schema + scalar indexes + alias.
- README/notes covering the alias workflow.
- ftsearch unchanged otherwise; existing endpoint still answers traffic against the alias.

## Acceptance

- Script run against an empty Milvus produces the expected collection with all §7 fields, expected indexes, and a registered alias.
- A representative legacy `articleId` (≥ 80 chars) inserts and round-trips through the PK without truncation.
- Existing `/{collection}/_search` traffic against the alias name continues to behave identically to today.
- All scalar filter expressions that F3 will rely on (e.g. `vendor_id == "x"`, `eclass5_code in [...]`, `array_contains_any(features, [...])`) parse and execute against an empty collection without error.

## Open questions for this packet

- `vendor_id` cardinality: legacy ES uses a single vendor field, but this repo's existing schema has `vendor_ids ARRAY`. Confirm one-vendor-per-row vs many; if many, keep as ARRAY and adjust §4.3 filter wording in the spec.
- `prices JSON` query patterns: confirm the Milvus version in use supports the JSON path expressions ftsearch will need at filter time, or fall back to fetching the JSON and resolving in Python (F3/F4 will design accordingly — flag any constraints found here).
