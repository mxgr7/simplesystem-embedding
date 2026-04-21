# Progressive Field Import Study — Autonomous Execution Plan

Goal: measure the marginal impact of each field on **import time**, **Milvus RSS after load**, and **MinIO/volume disk usage**. Re-run the import from scratch for each field-set using the streaming importer.

## Host & data context

- Host: Hetzner dedicated, 48 cores, 184 GB RAM, 512 GB NVMe at `/mnt/HC_Volume_105463954`.
- Milvus 2.6.15 standalone via `~/milvus/docker-compose.yml`. Service name in compose: `standalone` (container `milvus-standalone`). MinIO container: `milvus-minio`.
- Collection name: `offers`.
- Source parquet: pre-flattened at `/mnt/HC_Volume_105463954/simplesystem/data/offers_flat.parquet/bucket=NN.parquet` (16 buckets). Schema includes `id`, `offer_embedding`, 7 filter arrays, 7 text fields, `n`, plus original nested columns.
- Aggressive GC env vars MUST be set on `standalone` service in `docker-compose.yml` (already done): `DATACOORD_GC_INTERVAL=60`, `DATACOORD_GC_DROPTOLERANCE=60`, `DATACOORD_GC_MISSINGTOLERANCE=60`.

## Dataset choice

**Default: `bucket=00.parquet` only** (9,954,348 rows). Each iteration ~15–25 min, full sweep ~4 h.

Optional full-scale (all 16 buckets, 159.3M rows) — each iteration ~2–3 h, full sweep ~24–36 h. Use this only after the small-scale sweep identifies fields worth measuring at full scale.

Rationale: relative costs per field scale predictably. A 1-bucket sweep gives signal quickly; full-scale can follow for specific fields.

## Field progression (13 iterations)

Ordered small → large to surface the cheap majority first and isolate the big contributors.

| Iter | Adds | Type | Est. impact |
|---|---|---|---|
| 0 | `id`, `offer_embedding` | VARCHAR PK + FLOAT16_VECTOR(128) | **baseline** |
| 1 | `n` | INT64 | trivial |
| 2 | `ean` | VARCHAR(32), avg 12 B | trivial |
| 3 | `article_number` | VARCHAR(64), avg 9 B | trivial |
| 4 | `manufacturerName` | VARCHAR(128), avg 10 B | trivial |
| 5 | `manufacturerArticleNumber` | VARCHAR(128), avg 10 B | trivial |
| 6 | `manufacturerArticleType` | VARCHAR(512), avg 2 B | trivial |
| 7 | `name` | VARCHAR(256), avg 57 B | small |
| 8 | `vendor_ids` | ARRAY<VARCHAR(64)>, max_cap 32 + INVERTED | small-medium |
| 9 | `category_l1` … `l5` (5 arrays, added together) | ARRAY<VARCHAR>, max_cap 64 + INVERTED each | medium |
| 10 | `catalog_version_ids` | ARRAY<VARCHAR(64)>, max_cap 2048 + INVERTED | **large** (known problem field) |
| 11 | `description` | VARCHAR(65535), avg 499 B | **very large** |

If iteration 10 or 11 pushes disk < 30 GB free or RSS > 170 GB during any step, mark as "too big for config" and stop the sweep; don't force through.

## Per-iteration procedure

Each iteration is a complete end-to-end cycle. No state carries over.

### 1. Cleanup phase (~3–5 min)

```
# Stop any running python clients
pkill -f 'milvus_import|progressive' || true

# Drop collection if present
python -c "from pymilvus import connections, utility; connections.connect(host='localhost', port='19530'); \
  [utility.drop_collection(c) for c in utility.list_collections()]"

# Restart Milvus to reset Go heap
docker restart milvus-standalone
until curl -sf http://localhost:9091/healthz > /dev/null; do sleep 3; done

# Wait up to 3 min for GC to remove dropped collection data from MinIO.
# Poll MinIO logical size; accept when it stabilizes or drops below 2 GB.
# (The 1-minute gc.interval guarantees collection is gone within ~2 min of drop.)
```

**Success criterion:** `mc du local/a-bucket/files` stable or < 2 GB.

### 2. Create collection with iteration's field subset

Always include: `id` (VARCHAR(64), PK), `offer_embedding` (FLOAT16_VECTOR(128)).
Add only the fields for this iteration's cumulative set.

Use the schema definitions from `scripts/milvus_import.py:build_collection()` as the source of truth for field max_length / max_capacity.

```python
# Example iter=8 (base + n + 6 small text + name + vendor_ids)
fields = [
    FieldSchema("id", VARCHAR, max_length=64, is_primary=True),
    FieldSchema("offer_embedding", FLOAT16_VECTOR, dim=128),
    FieldSchema("n", INT64),
    FieldSchema("ean", VARCHAR, max_length=32),
    # ... etc
    FieldSchema("vendor_ids", ARRAY, element_type=VARCHAR, max_capacity=32, max_length=64),
]
```

No mmap. Default settings.

### 3. Streaming import (~5–15 min for 1 bucket)

Read parquet, project only the columns this iteration cares about, insert in batches.

- Batch size: **50_000 rows** (matches `milvus_import.py` default; under gRPC 64 MB limit even for wide schemas).
- Use `pa.ParquetFile.iter_batches(batch_size=50_000, columns=[<iter fields>])`.
- For embeddings: `np.stack(batch.column("offer_embedding").to_numpy(zero_copy_only=False))` yields (n, 128) fp16.
- Description: pre-truncate rows exceeding 65535 bytes using the byte-safe `pc.if_else(pc.greater(binary_length, 65535), utf8_slice_codeunits(col, 0, 16383), col)` pattern from `milvus_bulk_import.py:convert_batch`.
- Measure: `insert_wall_s`, `rows_per_s`.

### 4. Finalize indexes (~5–30 min)

```python
col.flush()
col.create_index("offer_embedding",
    index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 4096}})
wait_index_finished("offer_embedding")  # poll state=="Finished", not pending_index_rows

for field in scalar_array_fields_in_this_iter:
    col.create_index(field, index_params={"index_type": "INVERTED"}, index_name=field)
    wait_index_finished(field)
```

- Reuse `wait_index_finished` from `scripts/milvus_import.py:197`.

### 5. Wait for compaction to settle (~2–15 min)

Poll every 30 s until ALL fields show `state=="Finished"` AND `pending_index_rows == 0` for **two consecutive checks**. Cap wait at 30 min; if it hasn't settled, record `compaction_settled=false` and proceed.

### 6. Load (~30 s – 5 min)

```python
t0 = time.time()
col.load()          # load ALL fields in collection (no load_fields arg)
load_s = time.time() - t0
```

Capture `load_wall_s`.

### 7. Measure

Record into a CSV row (one row per iteration):

```
iter, field_added, insert_wall_s, rows_per_s, flush_s, vec_idx_s,
scalar_idx_total_s, compaction_s, load_wall_s,
baseline_rss_gb, loaded_rss_gb, rss_delta_gb,
minio_logical_gb, insert_log_gb, index_files_gb,
volume_used_gb, volume_free_gb,
segment_count, num_entities, compaction_settled
```

- `baseline_rss_gb`: RSS right before `col.load()`
- `loaded_rss_gb`: RSS 30 s after `col.load()` returns (allow delegator to settle)
- MinIO sizes: `docker exec milvus-minio mc du --depth=1 local/a-bucket/files`
- Volume: `df -B1 /mnt/HC_Volume_105463954`

### 8. Teardown for next iteration

Drop collection. Iteration 0's cleanup phase is equivalent; letting the next iter's Step 1 handle it is fine.

## Abort conditions (hard stops)

Halt the entire sweep, not just the current iteration, if any of:

1. **Volume free < 30 GB** at any checkpoint.
2. **Loaded RSS > 170 GB** (within 90% of 184 GB cap with buffer).
3. **Iteration wall time > 2× expected** (e.g. > 50 min for 1-bucket).
4. **Milvus container not healthy** for > 3 min after restart.
5. **Any unhandled exception** from pymilvus or Docker.

On abort: write final state (docker stats, df, mc du) to log, mark sweep as incomplete, exit non-zero.

## Deliverables

1. **`scripts/progressive_import.py`** — one Python orchestrator that runs the whole sweep autonomously. Argparse flags: `--data-dir`, `--bucket` (default `bucket=00.parquet`), `--iters <range>` (optional, for resuming), `--csv-out`, `--log-out`.
2. **`/tmp/progressive_import_<timestamp>.csv`** — results table (see schema in Step 7).
3. **`/tmp/progressive_import_<timestamp>.log`** — detailed per-step log.
4. **`PROGRESSIVE_IMPORT_RESULTS.md`** (written at end of sweep) — summary table + one-paragraph analysis per field noting notable costs. Written to repo root.

## Pre-flight checks (before starting iteration 0)

1. `docker compose ps` shows all 3 containers healthy.
2. `df -h /mnt/HC_Volume_105463954` shows ≥ 150 GB free.
3. `pymilvus.utility.list_collections()` returns empty or only `offers`.
4. Aggressive GC env vars are set (grep docker-compose.yml for `DATACOORD_GC_INTERVAL`).
5. Source parquet bucket file exists and is readable.

If any check fails, fix before starting. Do not start the sweep in a dirty state.

## Non-goals

- Not benchmarking query latency. (That's `scripts/milvus_bench.py`, separate.)
- Not evaluating mmap tradeoffs. (Earlier investigation covered that; this study is about in-RAM cost.)
- Not producing production config recommendations — just the per-field cost table.

## Notes from prior investigation

- Go runtime on Milvus retains heap between loads; **restart between iterations** is non-negotiable.
- `loading_progress` goes 0% → ~10% (preimport) → 40% (import) → 65%/70% → 100% (compaction). A stuck 65% = OOM-by-threshold, not real stuckness.
- `col.create_index` returns immediately with `state="Finished"` but `pending_index_rows != 0` — keep polling segments until stable.
- Collection-level mmap property has no effect on existing indexes; would require per-index alter. Not used in this study.
- `insert_log` holds raw columnar data required at query time (for output fields). It cannot be excluded from RSS when `col.load()` runs on a field; only `mmap` can disk-back it.
