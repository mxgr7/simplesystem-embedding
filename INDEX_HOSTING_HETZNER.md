# Hetzner Milvus Import Summary

Production import of 159M offer embeddings into Milvus on a Hetzner dedicated box. Streaming insert + IVF_FLAT. Runs from April 2026. For the full plan and context see `INDEX_HOSTING.md`.

## Machine

- **Hetzner dedicated**, 48 cores, 184 GB RAM
- **Storage**: 512 GB NVMe data volume at `/mnt/HC_Volume_105463954`
- **Milvus**: v2.6.15 standalone, Docker Compose (etcd 3.5.25, MinIO 2024-12-18). Not memory-capped — host RAM is the ceiling.
- Compose file: `~/milvus/docker-compose.yml`

## Collection schema

The unique key is `id` (32-char hex).

```python
FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True)
FieldSchema(name="offer_embedding", dtype=DataType.FLOAT16_VECTOR, dim=128)
```

Importer reads only the two needed columns from the wide source: `iter_batches(columns=["id", "offer_embedding"])`.

Index: IVF_FLAT, nlist=4096, metric COSINE.

### Filterable variant

Adds 7 `ARRAY<VARCHAR>` scalar fields to match the app's ES layout. `catalog_version_id` is globally unique across vendors, so independent arrays are sufficient — no pair-composite field.

```python
FieldSchema("vendor_ids",          ARRAY<VARCHAR>, max_capacity=32,   max_length=64)
FieldSchema("catalog_version_ids", ARRAY<VARCHAR>, max_capacity=2048, max_length=64)
FieldSchema("category_l1..l5",     ARRAY<VARCHAR>, max_capacity=64,   max_length=256..1280)
```

Category arrays mirror `categoryPaths.upToLevelN`: each level stores the joined first-N breadcrumb elements. Separator is `U+00A6` (¦); literal ¦ inside element names is replaced with `|` (U+007C) before joining, matching `CategoryPath.asStringPath()`. All 7 scalar arrays get INVERTED indexes; vector index unchanged.

Caps came from scanning the full bucket=00 (9.95M rows): max 22 vendors/row, **260 catalog_version_ids/row**, 20 category entries/row at L2–L4. Caps were sized with 2–4× headroom.

## Import timings

### Single bucket (bucket=00, 9,954,348 rows)

| Phase | Time |
|---|---|
| Streaming insert (batch=100k) | **2.5 min** (66.0k rows/s) |
| Flush | 1.2 s |
| IVF_FLAT build | <10 s |
| Load | 11.3 s |
| **Total** | **~3 min** |

### Full 16 buckets (159,275,274 rows)

| Phase | Time |
|---|---|
| Streaming insert b00–b15 | **39.3 min** (sustained 67.6k rows/s) |
| Flush | 1.7 s |
| IVF_FLAT `create_index` (blocking) | **~69 min** |
| `col.load()` | 106.6 s |
| **Total start-to-queryable** | **~110 min (1h 50min)** |

Streaming throughput was essentially flat across buckets (69.4k → 65.5k rows/s, ~6% drop). On 48 cores the background IVF builder doesn't contend meaningfully with the insert path.

### Memory during import

Peak Milvus RSS during the full 110-min run: **~4.3 GB**. The script creates the collection fresh and doesn't call `col.load()` until after flush + index build, so sealed segments never accumulate in the query node RSS during streaming. After `col.load()` RSS jumped to ~105 GB and settled at **~91 GB post-compaction**.

### Disk (MinIO segment store, 512 GB volume)

| Snapshot | MinIO | Total volume |
|---|---|---|
| Pre-import | ~3 GB | 175 GB |
| After flush / start of index build | ~145 GB | 304 GB |
| After `col.load()` | ~220 GB | 379 GB |
| Steady-state expected post-GC | ~80 GB | — |

### Post-compaction (settled ~45 min after load)

| Metric | At load | Settled |
|---|---|---|
| Segment count | 405 | **68** |
| avg rows/segment | 393k | **2.34M** |
| `pending_index_rows` | 154M | **0** |
| Milvus RSS | 105 GB | **91 GB** |

2.34M rows × 1/4096 ≈ 572 vectors/cell — healthy for kmeans.

### Filterable single bucket (bucket=00, 9,954,348 rows)

| Phase | Time |
|---|---|
| Streaming insert (batch=50k) | **16.1 min** (10,310 rows/s) |
| Flush | 6.3 s |
| IVF_FLAT build | <1 s (state=Finished at return) |
| Scalar INVERTED × 7 | <1 s each |
| Load | 25.9 s |
| **Total** | **24.3 min** |

Sustained insert was **~6.5× slower** than the vector-only run (67k → 10k rows/s). Bottleneck: Python-side flattening of `categoryPaths` and `vendor_listings` — per-row `set()` builds and dict accesses under the GIL. Server-side writes were not the limit.

Mitigation: `scripts/prepare_flat_buckets.py` uses DuckDB to precompute the flat columns into a parallel parquet tree. Bucket=00 preflatten: 139 s with 4 DuckDB threads (159k rows/s on flatten alone) → 2.96 GB output per bucket. `milvus_import.py` auto-detects `vendor_ids` in the input schema and skips the Python flatten path. Expected recovery: 2–3× import throughput.

gRPC gotcha: the filterable payload pushes a 100k-row batch past the default 64 MB gRPC message limit (69 MB observed). Default `--batch-size` lowered to 50k (~35 MB); stays well under the ceiling.

### Filterable post-compaction (bucket=00)

| Metric | At load | Settled |
|---|---|---|
| Segment count | — | **9** |
| `pending_index_rows` | 5.85M | **0** |
| Drain time | — | **3.0 min** |

Dramatically faster than the 159M full run (3 min vs ~45 min) — 1/16 the data and far fewer segments to merge.

## Benchmark (post-compaction, 159M × 128 fp16)

Single client, 64 random gaussian queries, 20 trials/cell after 3-query warmup, post-`--prime`.

### Search latency, IVF_FLAT

| nprobe | min | **p50** | p95 | p99 |
|---|---|---|---|---|
| 4 | 5.4 ms | **5.7 ms** | 6.4 ms | 6.6 ms |
| 16 | 7.5 ms | **8.1 ms** | 9.3 ms | 9.3 ms |
| 64 | 14.5 ms | **16.5 ms** | 17.6 ms | 17.7 ms |
| 256 | 46.0 ms | **50.2 ms** | 56.6 ms | 57.0 ms |

**16.5 ms p50 at nprobe=64 is interactive-serving territory.** nprobe=16 at 8.1 ms is the default sweet spot if recall is sufficient.

Pre-compaction the same queries ran 2.8–5.6× slower (47 ms p50 at nprobe=64). **Always rebench after compaction settles.**

### Batched latency (nprobe=64, limit=10)

| batch | total p50 | per-query p50 | speedup |
|---|---|---|---|
| 1 | 16.2 ms | 16.2 ms | 1.0× |
| 5 | 55.9 ms | 11.2 ms | 1.4× |
| 20 | 201.5 ms | 10.1 ms | 1.6× |

Batching amortization is modest post-compaction — the per-query baseline is already small.

### Recall@10 vs nprobe=256 reference

| nprobe | recall@10 |
|---|---|
| 4 | 0.373 |
| 16 | 0.620 |
| 64 | 0.825 |
| 256 | 1.000 (ref) |

Numbers are artificially low: queries are random gaussian (not on the data manifold) and reference is nprobe=256, not FLAT. **Measure against real workload queries + a FLAT ground truth before committing to a production nprobe.**

### Priming

Cold first query ~45 ms → stable ~16 ms after ~20 queries at nprobe=64. **Prime before accepting production traffic.** The bench script's `--prime` flag is sufficient.

## Gotchas worth remembering

1. **Poll `state == "Finished"`, not `pending_index_rows == 0`.** `pending_index_rows` drifts due to compaction and never reliably reaches 0 during an import. `scripts/milvus_import.py` uses state-based polling.
2. **`create_index` is blocking in pymilvus 2.6.12** — synchronously waits for the initial IVF build (~69 min here). Watch `utility.index_building_progress` from a second process if you need progress.
3. **Don't `col.load()` during streaming inserts.** Keep the collection unloaded until after flush + index; otherwise sealed segments accumulate in query-node RSS.
4. **Default gRPC message limit is 64 MB.** A 100k-row batch on the filterable schema (id + emb + 7 ARRAY<VARCHAR>) serializes to ~69 MB and hits `RESOURCE_EXHAUSTED`. Use `--batch-size 50000` (default in `milvus_import.py`).
5. **Python flatten is the ceiling on filterable ingest.** Per-row set building under the GIL caps throughput at ~10k rows/s. Preflatten with `scripts/prepare_flat_buckets.py` (DuckDB) to push the work off the hot path.

## Scripts

- `scripts/milvus_import.py` — streaming importer. `--bucket NAME` for single bucket; omit for all 16. `--index-type {IVF_FLAT,IVF_PQ,FLAT,HNSW}` (default IVF_FLAT). Auto-detects pre-flattened input (checks for `vendor_ids` column) and skips Python flatten.
- `scripts/prepare_flat_buckets.py` — DuckDB preflatten for the filterable schema. Emits `bucket=NN.parquet` with `id`, `offer_embedding`, `vendor_ids`, `catalog_version_ids`, `category_l1..l5`. `--skip-existing` for resumability.
- `scripts/milvus_verify.py` — stats + random query + self-hit sanity check (uses `id` as PK).
- `scripts/milvus_bench.py` — latency/recall bench with `--prime` warmup.

## Monthly cost

Assumed usage schedule: Mon–Fri 07:00–20:00 CET ≈ **282 hrs/month** (13 hrs × 5 days × 52/12, ~39% of a 720-hr reference month).

Both providers bill compute hourly. Hetzner Cloud caps at a monthly rate; AWS has no cap but offers Savings Plans. Realizing schedule savings requires tearing down outside the window — on AWS stop/start is enough, on Hetzner Cloud you must **delete** the server (paused servers still bill) and rebuild from a snapshot. Volumes bill continuously on both.

| Platform | Compute | Storage | Monthly total |
|---|---|---|---|
| Hetzner Cloud CCX63 equivalent (48 vCPU dedicated / 192 GB), 24/7 | ~€259 (cap) | 512 GB volume ~€22 | **~€280 (~$300)** |
| Hetzner same, scheduled (delete/rebuild nightly + weekends) | 282/720 × €259 ≈ €101 | volume €22 + snapshot ~€3 | **~€126 (~$135)** |
| AWS eu-central-1 `m6i.12xlarge` + 500 GB gp3, scheduled on-demand | 282 hrs × $2.70 ≈ $762 | ~$48 | **~$810** |

A 1yr AWS Compute Savings Plan (no upfront, ~28% off compute) brings AWS to **~$600/month**.

**Hetzner on the same schedule is ~6× cheaper** than AWS on-demand; left running 24/7 it's still ~2.7× cheaper. The tradeoff is operational: the delete/rebuild pattern means ~10 min to restore from snapshot at morning startup plus `col.load()` + prime, and no managed failover.

## Recommendations for next re-import

1. Plan **~110–140 min end-to-end** at 159M; add ~45 min for compaction to settle before clean bench numbers.
2. **Rebench after compaction every time** — pre-compaction numbers are misleadingly slow.
3. **Adding filter fields** (`vendor_ids`, `catalog_version_ids`, `category_terms` as `ARRAY<VARCHAR>`) will raise loaded RSS from ~91 GB to ~145–160 GB. Fits on 184 GB with narrow margin; Milvus per-field mmap is the escape hatch if OOM looms.
4. **Priming is necessary**, not optional — cold is ~2.8× warm p50.
5. **Use nprobe=16 unless recall requires more.** Verify recall against real queries first.
