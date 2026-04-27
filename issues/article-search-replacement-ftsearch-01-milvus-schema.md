# F1 — Milvus collection schema extension + id format + alias plumbing

**Category**: ftsearch (`./search-api/`)
**Depends on**: —
**Unblocks**: F3, F4, F5, I1, I3

References: spec §4.8, §6, §7, §9 #4.

**Legacy reference** (next-gen): `article/search/commons/src/main/java/com/simplesystem/nextgen/article/search/commons/domain/SearchArticleDocument.java` (current ES document shape), `…/domain/CategoryPath.java` (`¦` U+00A6 separator with `|` U+007C as the in-element escape).

**Milvus deployment**: `milvusdb/milvus:v2.6.15` (see `/home/mgerer/milvus/docker-compose.yml`). VARCHAR up to 65535, JSON path expressions in `expr`, `group_by_field` for search, `MilvusClient.alter_alias` — all available on this version.

## Scope

Bring the dense `offers` Milvus collection schema up to parity-readiness: add every scalar field §7 calls out, widen the primary key to carry the legacy `articleId` composite, and stand up the alias-based deployment plumbing required by §4.8.

This packet only defines and creates the schema. Population is I1; consumption is F3..F5.

## In scope

- New collection schema with all of §7:
  - `vendor_id VARCHAR(64)` — single UUID per offer. Legacy is single (`SearchArticleDocument.java:31`); the existing experimental `vendor_ids ARRAY` form in `scripts/milvus_import.py` is to be replaced.
  - `prices JSON` — full legacy nested-prices array, projected verbatim
  - `delivery_time_days_max INT`
  - `core_marker_enabled_sources ARRAY<STRING>`, `core_marker_disabled_sources ARRAY<STRING>`
  - `eclass5_code INT`, `eclass7_code INT`, `s2class_code INT`
  - `features ARRAY<VARCHAR>` of `name=value` tokens (separator `=`)
  - `relationship_accessory_for ARRAY<STRING>`, `relationship_spare_part_for ARRAY<STRING>`, `relationship_similar_to ARRAY<STRING>`
  - `closed_catalog BOOL`
  - retain `name`, `manufacturerName`, `ean`, `article_number`, `catalog_version_ids`, `category_l1..l5`, `offer_embedding`
- Widen `id` to `VARCHAR(256)`. Current cap is 64 (`scripts/milvus_import.py:83`); 256 leaves ample headroom for `{friendlyId}:{base64Url(articleNumber)}` (≥ 80 chars in practice). Milvus 2.6.15 allows up to 65535 — no hard limit hit.
- Scalar indexes on the fields that filtering will hit on the hot path: `vendor_id`, `eclass5_code`, `eclass7_code`, `s2class_code`, `closed_catalog`, `delivery_time_days_max`. Confirm Milvus index types per field type.
- Schema-creation script under `scripts/` that creates the collection and registers a Milvus alias (default `offers`) pointing at it (`MilvusClient.alter_alias`). **Naming convention**: versioned constants `offers_v{N}` (e.g. `offers_v2`, `offers_v3`); operator picks `N = current+1` when triggering reindex (I3 takes the name as a CLI argument).
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

(none — `vendor_id` cardinality locked to single, Milvus 2.6.15 supports JSON-path expressions and `group_by_field`.)
