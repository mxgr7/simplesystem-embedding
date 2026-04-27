# Milvus HNSW Re-index — 2026-04-20 23:29:58

159M × 128d fp16 offer embeddings reindexed from IVF_FLAT to HNSW on Hetzner box
(48 cores, 184 GB RAM). Switched because IVF_FLAT upcasts fp16 → fp32 internally
(via knowhere `KNOWHERE_MOCK_REGISTER` — all IVF variants materialize fp16 as
fp32 at build and search). HNSW is the only CPU index family in Knowhere that
keeps fp16 natively end-to-end.

Original IVF_FLAT load pinned ~160 GB RSS on 159M vectors (fp32 raw + fp32 index
kept in memory). HNSW with mmap'd scalar indexes settles well under that.

## Build

- Index: `HNSW(M=16, efConstruction=360, metric=COSINE)`
- rows: 159,275,274
- build wall time: **~108 min** (steady ~1.6M rows/min)
- peak RSS during build: ~3 GB (HNSW builder is disk-streaming)
- on-disk index size (MinIO, all indexes): 52GiB	18313 objects	a-bucket/files/index_files

## Load

- `load_fields=["id", "offer_embedding"]`
- vector index (HNSW): resident in RAM (fp16 native)
- 7 scalar INVERTED indexes + their fields: `mmap.enabled=true` (on both field and index)
- load wall: **741.9 s** (12.4 min)
- RSS trajectory: 0 → peak **~142 GB** (mid-load) → settled ~97 GB
- min free disk during load: 30 GB
- threshold bump applied: `queryCoord.overloadedMemoryThresholdPercentage: 90 → 95`

## Post-load state (at bench time)

- RSS: **94.9 GB**
- free disk: 30 GB
- collection state: Loaded

## Latency (single-query, limit=10, 20 trials after warmup)

| ef | min | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| 16 | 13.7 | **15.5** | 16.5 | 17.2 | 17.2 |
| 64 | 16.4 | **19.2** | 21.8 | 21.8 | 21.8 |
| 256 | 14.2 | **17.7** | 23.5 | 24.9 | 24.9 |

Values in ms. Queries are random unit-norm fp16 (off-manifold), not real traffic.

## Recall@10 vs ef=512 pseudo-reference (same index, higher ef)

| ef | recall@10 |
|---|---|
| 16 | 0.273 |
| 64 | 0.564 |
| 256 | 0.889 |

Recall is a *self-consistency* measure against ef=512 on the same HNSW
index. It is NOT absolute recall vs FLAT ground truth. Measure against real
workload queries + a FLAT reference before tuning production ef.

## Gotchas remembered from today

1. **fp16 → fp32 upcast in all IVF_\* indexes** is a Knowhere compile-time
   decision (`KNOWHERE_MOCK_REGISTER`). Use HNSW to keep fp16 in memory.
2. **`load_fields` does NOT restrict scalar INVERTED index loading.** Scalar
   indexes warm per-segment regardless. Only `mmap.enabled=true` (on both field
   AND index) pushes them to disk.
3. `overloadedMemoryThresholdPercentage: 90` is a Milvus-internal cap that reads
   host memUsage from cgroup. It considers all process memory, including Go
   heap slack from prior builds/loads. After large builds, RESTART Milvus before
   loading to reset the Go heap (`col.release()` doesn't return memory to OS).
4. IVF_SQ8 is NOT a memory fix for fp16: still upcasts to fp32 + keeps both raw
   and SQ8 codes (~100 GB for 159M × 128). IVF_RABITQ is the proper
   memory-constrained alternative if HNSW RAM is too high.

## Final state

- RSS (post-bench): 95.2 GB
- free disk: 30 GB
- HNSW collection loaded, queryable, ready for real-workload benchmarking
