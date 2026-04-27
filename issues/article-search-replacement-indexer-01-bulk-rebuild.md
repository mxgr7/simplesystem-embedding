# I1 — Bulk rebuild + canonical MongoDB → Milvus projection module

**Category**: Indexer (new pipeline)
**Depends on**: F1 (collection schema)
**Unblocks**: I2, I3, F3..F5 testing on populated data, A6

References: spec §6, §7, §9 #4.

**Legacy reference** (next-gen):
- `articleId` encode/decode: `article/search/commons/src/main/java/com/simplesystem/nextgen/article/search/commons/domain/ArticleId.java:22-37`. Encoding: `FriendlyId.toFriendlyId(vendorId) + ":" + BaseEncoding.base64Url().omitPadding().encode(articleNumber.getBytes())`. Decoding reverses. **`friendlyId` is derived on the fly from the `vendorId` UUID — not stored** in MongoDB. Port the `FriendlyId` derivation directly.
- Document shape: `commons/.../domain/SearchArticleDocument.java` (single `vendorId UUID` per offer).
- Category path: `commons/.../domain/CategoryPath.java:16-40` — separator `¦` (U+00A6); replace `¦` inside elements with `|` (U+007C) at encode time.
- Legacy indexer reads MongoDB **indirectly** via Kafka events (`indexer/application/src/main/resources/application.yml`, MongoDB env vars `MONGODB_PORTAL_URI_V2`, `MONGODB_PORTAL_DATABASE` lines 10-14). For our **bulk rebuild** we need a direct MongoDB scan; reuse the same env-var conventions.

**TEI embedder** (existing): service defined in `playground-app/compose.yaml` — image `ghcr.io/huggingface/text-embeddings-inference:cpu-1.8`, model mounted at `/model` from `${TEI_MODEL_DIR:-/data/tei-models/useful-cub-58-st}`, mean pooling, port 8080. For heavy reimport runs, a dedicated GPU instance will be provisioned; the indexer must accept an alternative `EMBED_URL` env at runtime to point at it.

## Scope

Project every parity-critical field from MongoDB into the new Milvus collection schema (F1), and run a full reimport that hydrates the collection from scratch. The projection module is the canonical mapper used by both this packet and I2 (incremental Kafka).

This packet establishes the indexer surface in the repo. Today the repo has bulk-import scripts (`scripts/milvus_*import*.py`) but they only project the existing fields — this packet replaces that with a full mapper.

## In scope

- **Projection module** (`indexer/projection.py` or similar): pure function `mongo_record → milvus_row`:
  - **PK / `id`**: construct `{friendlyId}:{base64Url(articleNumber)}` (§9 #4). `friendlyId` is **derived from the `vendorId` UUID** — port `FriendlyId.toFriendlyId` from next-gen `commons/.../FriendlyId.java`. The base64Url uses URL-safe alphabet, no padding.
  - **`offer_embedding`**: produce via the existing TEI service (`EMBED_URL` env). Reuse `search-api/embed_client.py` patterns to avoid embedding-model drift between indexer and query path.
  - **`name`, `manufacturerName`, `ean`, `article_number`**: straight projections.
  - **`vendor_id`**: single UUID (per F1's locked schema; legacy is single `UUID vendorId`).
  - **`catalog_version_ids`**: as today.
  - **`category_l1..l5`**: `¦` (U+00A6) separator; if a path element itself contains `¦`, replace it with `|` (U+007C) before joining (per legacy `CategoryPath.java`).
  - **`prices`**: project the legacy nested `prices` array verbatim into JSON: `[{"price": float, "currency": "EUR", "priority": int, "sourcePriceListId": "uuid"}, ...]`. Do NOT collapse — ftsearch resolves at query time (§7).
  - **`delivery_time_days_max`**, **`closed_catalog`**: straight projections.
  - **`core_marker_enabled_sources`**, **`core_marker_disabled_sources`**: array projections.
  - **`eclass5_code`**, **`eclass7_code`**, **`s2class_code`**: integer projections.
  - **`features`**: tokenise into `name=value` strings (§7). Legacy stores features structurally (`Set<OfferFeature>{name, values}`) and never serialises through `=`, so there's no precedent. **Policy: reject + log + drop** the offending feature when a value contains `=`. The rest of the row still indexes.
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

(none — MongoDB env vars resolved (`MONGODB_PORTAL_URI_V2`, `MONGODB_PORTAL_DATABASE`); embedder reuse confirmed; `friendlyId` derivation lifted from next-gen `FriendlyId`.)
