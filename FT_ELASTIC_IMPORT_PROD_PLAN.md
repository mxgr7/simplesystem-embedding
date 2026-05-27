# Plan: prod → semantic-enriched index pipeline

## Context

We need to produce a new ES index that mirrors `prod-article-index-v1` plus the §2.1 semantic additions: nested `embeddings` (one fp32 128-d vector per unique embedding, with `inputHash`), `embeddingModelVersion`, and the denormalised `catalogVersionIds` / `priceKeys` top-level scalars. The new index is named `prod-article-index-v1-semantic-<YYYYMMDDhhmmss>` (timestamp stamped at create time) and ultimately replaces the read alias via swap.

Why a Mongo roundtrip is unavoidable: the ES source documents do not carry the offer `description` field, but the renderer takes it as an input (`configs/data/default.yaml` template: `Description: {{ clean_description }}`). Description must come from Mongo. As in v5, Mongo data feeds the embeddings *via the cache* — the importer never reads Mongo directly.

What we are *not* doing:

- **No in-cluster clone of the source.** v5 used `stg-articles-v1-clone-20260516` as operational insulation against scan-vs-search contention on staging; on prod we accept the load directly. PIT gives the consistency guarantee, no clone needed for correctness.
- **No on-disk dump of the source ES index.** We stream from `prod-article-index-v1` via sliced PIT, same as v5's PIT loop. The only on-disk source artifact is the Mongo export — that's where the description data we don't already have in ES lives.

Operational consequences worth being aware of:
- The import puts sustained read load on prod ES for the duration of the run (32 sliced PIT readers). Run off-peak.
- PIT keep-alive pins segments that were visible at PIT open for ~12h; live writes during that window can't merge those segments away until PIT closes.
- Reruns are non-deterministic by design — each one captures fresher prod data.

---

## Phase 1 — Mongo offers export

**Why:** materialise the description-carrying fields the renderer needs, plus the vendor/article-number/hash join material that Phase 2 turns into the importer's lookup.

**New script: `scripts/dump_mongo_offers.py`** — no existing in-repo tool produces these dumps (v5's were done out-of-band). PyMongo against `MONGODB_URI` from `.env` (prod, *not* the v5-era `MONGODB_NONPROD_URI`), projects only the fields the renderer needs: `articleNumber`, `vendorId`, and `offer.offerParams.{name, manufacturerName, description, categoryPaths, ean, manufacturerArticleNumber, manufacturerArticleType}`. Shards output per `vendorId`.

- Output: `/data/datasets/mongo_offers_export_<YYYYMMDD>/vendor_<vendorId>.json.gz` (newline-delimited gzipped JSON — exactly the format `prewarm_v2_missing.py` already expects).
- Resumable per vendor: write to `vendor_<id>.json.gz.tmp` then atomic rename. Skip vendors whose final file already exists.
- Sanity verification: `gzip -dc vendor_*.json.gz | wc -l` ≈ prod `db.offers.estimatedDocumentCount()` (within Mongo sync noise).

---

## Phase 2 — Prewarm KVRocks

### 2.1 Bring up KVRocks

`docker compose -f dev/kvrocks/compose.yaml up -d`. KVRocks listens on `localhost:6666`, RocksDB-backed at `/data/kvrocks-data`. Speaks the Redis protocol → the existing Python `redis` client works unchanged. Fresh container = empty keyspace; resumability still works (a partial prewarm leaves the rest "missing", next run picks up where it left off via the `EXISTS`-filter pass).

### 2.2 Build the (vendor, article_number) → article_hashes lookup parquet

**Why:** the Phase 3 importer joins source docs to embedding hashes via this lookup (`index_embeddings_to_es_v5_mp.py:48`'s `DEFAULT_PARQUET = ".../article_hashes_v2/**/*.parquet"`). Must be materialised before Phase 3.

**New script: `scripts/build_article_hashes_lookup.py`** — same DuckDB query that `prewarm_v2_missing.py:build_render_inputs` already does for `render_inputs.parquet`, but groups by `(vendor_id, article_number)` instead of by `article_hash`, emitting the list of hashes per pair. Single small DuckDB run over the gzipped exports from Phase 1.

- Output: `/data/datasets/mongo_offers_export_<YYYYMMDD>/article_hashes_v2/bucket=NN.parquet` (16 buckets, matches the convention used by `offers_codes_staging` etc).

### 2.3 Run prewarm against KVRocks

`scripts/prewarm_v2_missing.py` runs as-is with two CLI overrides:

```
uv run python scripts/prewarm_v2_missing.py \
    --input-glob '/data/datasets/mongo_offers_export_<YYYYMMDD>/vendor_*.json.gz' \
    --redis-host localhost --redis-port 6666 \
    --tei-url <local TEI URL>
```

Phase-1 of the script builds `render_inputs.parquet` (per-unique-hash render-input table — separate artifact from 2.2's lookup). Phase-2 streams the parquet, `EXISTS`-filters against KVRocks, dispatches missing hashes to 64 fork workers that render via Jinja, POST to TEI, write fp16 to `tei:v2:<hash>`.

End state: `redis-cli -p 6666 DBSIZE` ≈ count of distinct `article_hash` values in `render_inputs.parquet`. Misses in Phase 3 should be ~zero.

---

## Phase 3 — Create target index, stream-enrich, write

### 3.1 Create the target index

Generate the timestamped name once and reuse it:

```
DST="prod-article-index-v1-semantic-$(date -u +%Y%m%d%H%M%S)"
uv run python scripts/setup_es_v5.py --dst "$DST" --settings-src prod-article-index-v1
```

`setup_es_v5.py` is reusable as-is. Mappings from `target_mapping.json`; analyzers/normalizers/shard count cloned from `prod-article-index-v1`; replicas=0, refresh=-1, async translog (per `BULK_IMPORT_TUNING.md`).

### 3.2 Stream from prod and enrich → target

`scripts/index_embeddings_to_es_v5_mp.py` runs essentially as-is with three CLI overrides:

```
uv run python scripts/index_embeddings_to_es_v5_mp.py \
    --src prod-article-index-v1 \
    --dst "$DST" \
    --redis redis://localhost:6666/0 \
    --parquet '/data/datasets/mongo_offers_export_<YYYYMMDD>/article_hashes_v2/**/*.parquet'
```

What it does (unchanged from v5, recapped because it's load-bearing):
- 32 PIT slices, one fork worker per slice → 32-way parallel read from `prod-article-index-v1`.
- Parent builds the (vendor, artno) → hashes lookup once from the parquet as fork-COW-shared numpy buffers; workers read it lock-free. Don't reintroduce per-worker dicts — see [[feedback_fork_gc_disable_leak]].
- Per article: `denormalize()` computes `catalogVersionIds` / `priceKeys`, `mget_vectors()` pulls fp16 from KVRocks, bulk indexes into `$DST` with 429 / transient-transport backoff. Explicit `_id` so retries are idempotent.
- `gc.freeze()` in parent + `gc.enable()` per worker preserves COW sharing.

### 3.3 Post-import finalize

Per `BULK_IMPORT_TUNING.md`:

- `POST /$DST/_refresh`
- `POST /$DST/_forcemerge?max_num_segments=1` (read-mostly index; merge to one segment for the §2.1.4 bench shape)
- `PUT /$DST/_settings {"index": {"number_of_replicas": 1, "refresh_interval": "1s"}}`

Verify pre-merge `_count` == post-merge `_count` (the v5 success criterion).

### 3.4 Pre-cutover warmup — load-bearing

Per `FT_ELASTIC_IMPORT.md` §2.1.6: a cold HNSW index has been measured at ~196 s on the first kNN query, converging to steady state over ~3-4 queries. **Alias flip must be gated on warmth, not on completion of the import.** Run a representative query pass (sample of `reports/hnsw_eval/queries.parquet`, or `tools/test18-latency-sweep.sh` if available) against `$DST` until p99 latency is in §2.1.4 territory. Warm replicas too — they serve searches.

### 3.5 Alias swap

Atomic:
```
POST /_aliases { "actions": [
  {"remove": {"index": "<current-concrete>", "alias": "<serving-alias>"}},
  {"add":    {"index": "$DST",                "alias": "<serving-alias>"}}
]}
```
The serving alias is whatever the consumer reads; confirm with the consumer before flipping.

---

## Files to create

| Path | Purpose | Reuses |
|---|---|---|
| `scripts/dump_mongo_offers.py` | Phase 1 — PyMongo → vendor-sharded JSON.gz | `.env` MONGODB_PROD_URI |
| `scripts/build_article_hashes_lookup.py` | Phase 2.2 — DuckDB → article_hashes_v2 parquet | `prewarm_v2_missing.py:build_render_inputs` SQL |

## Files to reuse unchanged

- `scripts/prewarm_v2_missing.py` — Phase 2.3, just two CLI overrides (port 6666, input-glob)
- `scripts/setup_es_v5.py` — Phase 3.1, pass `--dst $DST` and `--settings-src prod-article-index-v1`
- `scripts/index_embeddings_to_es_v5_mp.py` — Phase 3.2, three CLI overrides (`--src`, `--dst`, `--redis`)
- `target_mapping.json` — the §2.1 target mapping
- `dev/kvrocks/compose.yaml` — Phase 2.1
- `configs/data/default.yaml` — rendering template, pinned by HASH_VERSION=v2
- Runbooks: `BULK_IMPORT_TUNING.md`, `FT_ELASTIC_IMPORT.md` §2.1.3 / §2.1.6

## Files NOT involved (adjacent, out of scope)

- `embedding-service/` — the online serving path. Shares the KVRocks keyspace (`tei:v2:<hash>`), so prewarmed entries serve it for free, but it is not part of the batch pipeline.

---

## Verification

End-of-Phase-1:
- `gzip -dc /data/datasets/mongo_offers_export_<DATE>/vendor_*.json.gz | wc -l` ≈ prod `db.offers.estimatedDocumentCount()`.

End-of-Phase-2:
- `redis-cli -p 6666 DBSIZE` ≈ `duckdb -c "SELECT COUNT(DISTINCT article_hash) FROM read_parquet('.../render_inputs.parquet')"`.
- `duckdb -c "SELECT COUNT(*) FROM read_parquet('.../article_hashes_v2/**/*.parquet')"` non-zero and ≈ unique (vendor, articleNumber) pair count.

End-of-Phase-3:
- Importer's reported `arts:` total ≈ `GET /prod-article-index-v1/_count` minus `orphan:` skips (use `_count`, not `_cat docs.count` — see [[project_article_index]]).
- `redis misses == 0` (prewarm covered everything by construction).
- Pre-merge `_count` == post-merge `_count` == importer `arts` total.
- 10 random `articleId`s spot-checked on `$DST`: `embeddings[]` length matches `article_hashes_v2` lookup; `catalogVersionIds` is the deduped union of `offers[].catalogVersionIds`; `priceKeys` is the deduped union of `"{priceListId}|{currency}"` over `prices[]`.
- Warmup pass: p99 kNN latency in §2.1.4 territory before alias swap.

---

## Decisions locked in

1. **Target index name:** `prod-article-index-v1-semantic-<YYYYMMDDhhmmss>` (UTC at create time).
2. **ES source:** live alias `prod-article-index-v1` via sliced PIT. No clone, no on-disk dump.
3. **Mongo source:** prod (`MONGODB_PROD_URI` from `.env`, db `prod`, collection `offers`).
4. **Mongo dump shape:** per-vendor `vendor_<vendorId>.json.gz`, newline-delimited.
5. **`article_hashes_v2` parquet bucketing:** 16 buckets.
6. **Cache backend:** KVRocks on `localhost:6666`, keyspace `tei:v2:<hash>` (fp16, 256 B per 128-d vector).

---

## Smoke run (two-vendor end-to-end)

Before launching the full prod run, do an e2e dry-run restricted to two pre-selected small vendors. The point is to exercise *the same code paths* as the full run end to end; the only differences are scope filters and a distinct target index name. Both `dump_mongo_offers.py` and `index_embeddings_to_es_v5_mp.py` take `--vendor-ids` for this; the prewarm + lookup builder are naturally constrained by the input glob and need no change.

**Vendors:**
- `f508ac53-86b2-4b97-bd26-4789a3a40a1b`
- `0928e639-fc5a-4138-8c29-9201e8eba09c`

**Runbook:**

```bash
TS=$(date -u +%Y%m%d%H%M%S)
DST="prod-article-index-v1-semantic-${TS}-smoke"
VENDORS="f508ac53-86b2-4b97-bd26-4789a3a40a1b,0928e639-fc5a-4138-8c29-9201e8eba09c"
EXPORT_DIR="/data/datasets/mongo_offers_export_${TS%??????}"

# Phase 1 — vendor-restricted Mongo dump
uv run python scripts/dump_mongo_offers.py --vendor-ids "$VENDORS" --out-dir "$EXPORT_DIR"

# Phase 2.1 — KVRocks
docker compose -f dev/kvrocks/compose.yaml up -d

# Phase 2.2 — article_hashes_v2 lookup
uv run python scripts/build_article_hashes_lookup.py \
    --input-glob "$EXPORT_DIR/vendor_*.json.gz" \
    --out-dir   "$EXPORT_DIR/article_hashes_v2"

# Phase 2.3 — prewarm (naturally restricted to the two vendors)
uv run python scripts/prewarm_v2_missing.py \
    --input-glob "$EXPORT_DIR/vendor_*.json.gz" \
    --render-inputs-parquet "$EXPORT_DIR/render_inputs.parquet" \
    --redis-host localhost --redis-port 6666 \
    --tei-url <local TEI URL>

# Phase 3.1 — smoke target (note -smoke suffix to keep it distinct)
uv run python scripts/setup_es_v5.py --dst "$DST" --settings-src prod-article-index-v1

# Phase 3.2 — importer restricted to same two vendors
uv run python scripts/index_embeddings_to_es_v5_mp.py \
    --src prod-article-index-v1 --dst "$DST" \
    --redis redis://localhost:6666/0 \
    --parquet "$EXPORT_DIR/article_hashes_v2/**/*.parquet" \
    --vendor-ids "$VENDORS"

# Phase 3.3 — finalize (skip replicas restore for smoke)
curl -XPOST "$ES/$DST/_refresh"
curl -XPOST "$ES/$DST/_forcemerge?max_num_segments=1"

# Spot-check: per-vendor count parity
for V in ${VENDORS//,/ }; do
  EXPECTED=$(curl -s "$ES/prod-article-index-v1/_count" -H 'Content-Type: application/json' \
               -d "{\"query\":{\"term\":{\"vendorId\":\"$V\"}}}" | jq .count)
  ACTUAL=$(curl -s "$ES/$DST/_count" -H 'Content-Type: application/json' \
               -d "{\"query\":{\"term\":{\"vendorId\":\"$V\"}}}" | jq .count)
  echo "$V  prod=$EXPECTED  smoke=$ACTUAL"
done

# Field-derivation spot check (~3 articles per vendor): _source must have
# embeddings[] with vector + inputHash, embeddingModelVersion,
# catalogVersionIds == sorted unique union over offers[].catalogVersionIds,
# priceKeys == sorted unique set of "{priceListId}|{currency}" over prices[].

# Smoke runs skip Phase 3.4 (warmup) and Phase 3.5 (alias swap).

# Cleanup once smoke passes:
curl -XDELETE "$ES/$DST"
# Keep $EXPORT_DIR if you want to rerun the smoke; the full run will produce a
# fresh dated EXPORT_DIR for all 2.6k vendors.
```

