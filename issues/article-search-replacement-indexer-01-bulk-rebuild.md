# I1 — Bulk rebuild + canonical MongoDB → Milvus projection module

**Category**: Indexer (new pipeline)
**Depends on**: F1 (collection schema)
**Unblocks**: I2, I3, F3..F5 testing on populated data, A6

References: spec §6, §7, §9 #4.

## Scope

Project every parity-critical field from MongoDB into the new Milvus collection schema (F1), and run a full reimport that hydrates the collection from scratch. The projection module is the canonical mapper used by both this packet and I2 (incremental Kafka).

This packet establishes the indexer surface in the repo. Today the repo has bulk-import scripts (`scripts/milvus_*import*.py`) but they only project the existing fields — this packet replaces that with a full mapper.

## In scope

- **Projection module** (`indexer/projection.py` or similar): pure function `mongo_record → milvus_row`:
  - **PK / `id`**: construct `{friendlyId}:{base64Url(articleNumber)}` (§9 #4). If MongoDB stores `friendlyId` directly use it; otherwise derive per legacy code (lift from next-gen).
  - **`offer_embedding`**: produce via the existing TEI embedder (reuse `search-api/embed_client.py` patterns or call the embedder directly from the indexer — pick one and document).
  - **`name`, `manufacturerName`, `ean`, `article_number`**: straight projections.
  - **`vendor_id`** (or `vendor_ids`): per F1's resolved cardinality.
  - **`catalog_version_ids`**: as today.
  - **`category_l1..l5`**: `¦`-separated path encoding (matches the existing convention).
  - **`prices`**: project the legacy nested `prices` array verbatim into JSON: `[{"price": float, "currency": "EUR", "priority": int, "sourcePriceListId": "uuid"}, ...]`. Do NOT collapse — ftsearch resolves at query time (§7).
  - **`delivery_time_days_max`**, **`closed_catalog`**: straight projections.
  - **`core_marker_enabled_sources`**, **`core_marker_disabled_sources`**: array projections.
  - **`eclass5_code`**, **`eclass7_code`**, **`s2class_code`**: integer projections.
  - **`features`**: tokenise into `name=value` strings (§7). The indexer must reject or escape `=` inside values so the separator stays unambiguous; pick one (recommendation: reject loud, log the offender, drop the feature) and document.
  - **`relationship_accessory_for`**, **`relationship_spare_part_for`**, **`relationship_similar_to`**: array projections.
- **Bulk pipeline** (`indexer/bulk.py` and a CLI under `scripts/`):
  - Read all relevant records from MongoDB (use existing connection logic if any; otherwise add it cleanly).
  - Project via the module above.
  - Write into a fresh Milvus collection (the alias-target name from F1 — e.g. `offers_v2`).
  - Batch sized to amortise embedding + Milvus insert; document the chosen size.
  - Resumable / restartable on failure (checkpoint file or use the Milvus PK as the truth source).
  - Validation pass at the end: row count ≥ expected, sampled-row spot-checks pass.
- Drop / archive the old `scripts/milvus_*import*.py` if they are now superseded — confirm with the team first; flag any callers.

## Out of scope

- Kafka-driven incremental updates — I2.
- Alias swap orchestration — I3 (this packet writes to the new collection name; I3 swaps the alias).
- Field-level changes that aren't in §7.

## Deliverables

- `indexer/projection.py` with full unit tests.
- `indexer/bulk.py` + a CLI entry-point (`scripts/indexer_bulk.py` or via `pyproject.toml`).
- Fixture-based unit tests for the projection: known MongoDB doc → expected Milvus row.
- An end-to-end smoke that runs the bulk pipeline against a tiny MongoDB fixture and asserts the resulting Milvus collection has the expected count and shape.
- Operational notes: how to run, expected runtime at production volume, how to resume.

## Acceptance

- Projection unit tests pass on a representative MongoDB fixture.
- Running the bulk pipeline against a small fixture produces a Milvus collection that F3..F5 can query correctly.
- A representative legacy `articleId` (≥ 80 chars) lands in the PK without truncation and is queryable verbatim.
- A row with multiple prices across currencies and price lists round-trips through the JSON `prices` field unchanged.
- A row with feature values containing `=` is handled per the documented policy (reject + log).

## Open questions for this packet

- MongoDB connection details: confirm where credentials and connection strings come from (env? secret manager?); align with existing repo conventions.
- Embedder reuse: ftsearch currently calls a TEI service via `embed_client.py`. The indexer can call the same service or run its own; pick one. Recommendation: the same service to avoid embedding-model drift.
- `friendlyId` source: confirm whether MongoDB stores it or it must be derived from another field. If derived, lift the derivation from next-gen so behaviour matches.
