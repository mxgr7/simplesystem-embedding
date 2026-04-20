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

## Scripts

- `scripts/milvus_import.py` — streaming importer. `--bucket NAME` for single bucket; omit for all 16. `--index-type {IVF_FLAT,IVF_PQ,FLAT,HNSW}` (default IVF_FLAT).
- `scripts/milvus_verify.py` — stats + random query + self-hit sanity check (uses `id` as PK).
- `scripts/milvus_bench.py` — latency/recall bench with `--prime` warmup.

## Monthly cost

Assumed usage schedule: Mon–Fri 07:00–20:00 CET ≈ **282 hrs/month** (13 hrs × 5 days × 52/12). Hetzner is flat-rate regardless of utilization; AWS is billed hourly, so a weekday-only schedule cuts ~60% off a 24/7 bill.

| Platform | Compute | Storage | Monthly total |
|---|---|---|---|
| Hetzner Cloud CCX63 equivalent (48 vCPU dedicated / 192 GB) | ~€259 | 512 GB volume ~€22 | **~€280 (~$300)** |
| AWS eu-central-1 `m6i.12xlarge` + 500 GB gp3, on-demand | 282 hrs × $2.70 ≈ $762 | ~$48 | **~$810** |

A 1yr Compute Savings Plan (no upfront, ~28% off compute) brings AWS to **~$600/month**.

**Hetzner is ~2.7× cheaper** than AWS on-demand for this hardware shape, even after the weekday-only AWS schedule. The tradeoff is operational: bare/dedicated-CPU cloud has no stop/start, slower provisioning, and no managed snapshot/failover story — AWS buys you those at the price premium.

## Recommendations for next re-import

1. Plan **~110–140 min end-to-end** at 159M; add ~45 min for compaction to settle before clean bench numbers.
2. **Rebench after compaction every time** — pre-compaction numbers are misleadingly slow.
3. **Adding filter fields** (`vendor_ids`, `catalog_version_ids`, `category_terms` as `ARRAY<VARCHAR>`) will raise loaded RSS from ~91 GB to ~145–160 GB. Fits on 184 GB with narrow margin; Milvus per-field mmap is the escape hatch if OOM looms.
4. **Priming is necessary**, not optional — cold is ~2.8× warm p50.
5. **Use nprobe=16 unless recall requires more.** Verify recall against real queries first.
