# Milvus Full Bulk Import — 2026-04-21 06:02 → 08:18 UTC

Fresh bulk import of 159,275,274 × 128d fp16 offer embeddings + 16 scalar/array/text
fields into a single-node Milvus 2.6.15 on the Hetzner box (48 cores, 184 GB RAM,
902 GB NVMe). Inline HNSW + INVERTED index build during the `do_bulk_insert`
pipeline, no post-hoc create_index, all non-indexed fields `mmap.enabled=true`.

Orchestrator: `scripts/milvus_bulk_import.py` (16-worker ProcessPool staging,
pipelined `do_bulk_insert` submission). Full log:
`logs/full_run_20260421_060247.log`.

## Summary

```
rows:              159,275,274
stage+submit:        405.2s   (6.8 min — pipelined, 16 workers, disk-read bound)
ingest wait:        7451.9s   (124.2 min — PreImport → Import → Sort → IndexBuilding)
flush:                 0.8s
vector index:          0.0s   (built inline; verify-only pass — HNSW M=16, efC=360)
idx vendor_ids         0.0s   (built inline; INVERTED)
idx catalog_version    0.0s   (built inline; INVERTED)
load:                290.2s   (4.8 min)
total wall:         8150.1s   (135.8 min / 2h 15m 50s)
num_entities:      159,275,274 ✓
```

Post-load steady state: **RSS 125.3 GB, MinIO 244.4 GB, 224 segments, no active
compaction**.

## Schema and index plan

17 fields; 4 RAM-resident, 13 mmap-backed (pushed to disk, returnable via
`output_fields` but not held in RAM).

| field | type | indexed | mmap |
|---|---|---|---|
| `id` | VARCHAR(64) PK | — | no |
| `offer_embedding` | FLOAT16_VECTOR(128) | **HNSW(M=16, efC=360, COSINE)** | no |
| `vendor_ids` | ARRAY<VARCHAR>(cap=32, len=64) | **INVERTED** | no |
| `catalog_version_ids` | ARRAY<VARCHAR>(cap=2048, len=64) | **INVERTED** | no |
| `category_l1..l5` | 5× ARRAY<VARCHAR>(cap=64) | — | **yes** |
| `name` | VARCHAR(256) | — | **yes** |
| `manufacturerName` | VARCHAR(128) | — | **yes** |
| `manufacturerArticleNumber` | VARCHAR(128) | — | **yes** |
| `manufacturerArticleType` | VARCHAR(512) | — | **yes** |
| `description` | VARCHAR(65535) | — | **yes** |
| `ean` | VARCHAR(32) | — | **yes** |
| `article_number` | VARCHAR(64) | — | **yes** |
| `n` | INT64 | — | **yes** |

Note `mmap_enabled=True` is set at **field schema creation time** (pymilvus
`FieldSchema(..., mmap_enabled=True)` → `params.mmap_enabled`). Field-level
mmap is sufficient for unindexed fields; indexed fields kept RAM-resident by
leaving mmap unset.

## The critical sequencing rule (aborted run 3)

**Indexes must be defined BEFORE the first `do_bulk_insert` call.** A prior run
defined HNSW post-flush and discovered:

- `create_index(HNSW)` on an already-ingested+flushed collection succeeds
  metadata-wise (`col.indexes` lists it) but **schedules no build task**.
- `index_building_progress` returns
  `{state: "Finished", indexed_rows: 0, pending_index_rows: 159M}` —
  a race state where "Finished" means "no task registered" not "index ready".
- `col.load()` then falls back to an in-memory **IVF_FLAT_CC interim index**
  (fp32 — 2× RAM of HNSW fp16), not HNSW. Queries would use the wrong index.

Fix: move all `create_index` calls immediately after `build_collection()`. The
bulk-insert job pipeline then auto-builds them as part of its `IndexBuilding`
state (`jobState=IndexBuilding` observable in Milvus's import_checker logs and
per-task `GetImportProgress` responses). Verified via single-bucket run that
completed cleanly with `indexed_rows == 9,954,348`.

A second guard lives in the client: `wait_index_finished()` now rejects
`state=Finished` when `indexed_rows < total_rows`, so future accidents can't
silently pass.

## Phase timeline (derived from client log + Milvus stats)

```
 0:00 → 6:45    stage+submit     16× parquet convert (fp16 list → uint8 list) + S3 upload
 6:45 → 11:45   PreImport        all 16 parsing parquet row-group meta + row counts (~5 min parallel)
11:45 → 17:00   Import           raw binlog writes; MinIO 84 GB → 120 GB
17:00 → 32:00   Sort             per-segment sort-compaction; tail-bucket Sorts took ~30 min (slot contention)
19:00 → 126:30  IndexBuilding    HNSW + INVERTEDs per segment, ~47 cores saturated, load 80-95
19:00 → 126:30  Completions      first at ~19:00 wall, last at ~126:30, 5-6 min avg gap mid-run
131:00          flush + verify   0.8s flush, instant index-state verify
131:00 → 135:50 load             290s to read segment/index files + mmap scalars into page cache
```

All 16 buckets completed in-order by `do_bulk_insert` submission — no reordering.

## CPU / load / RSS behavior

| phase | Milvus CPU | load avg | Milvus RSS |
|---|---|---|---|
| stage+submit | ~7% (just receiving) | 40 | 186 MiB |
| PreImport (parallel, I/O-bound) | 150-450% | 15-30 | 1.7 GiB |
| Import + first Sort wave | 2400-4500% | 22-93 | 14-25 GiB (peaks when segments buffered) |
| Sort drain + IndexBuilding peak | 4500-4700% | 66-97 | **3-5 GiB** (streaming HNSW builder) |
| load | 1000% → 500 → idle | 15 → 7 | climbed 90 → 125 GiB as segments pulled |
| steady | 7-11% | 0-2 | **125.3 GiB** |

Observations:
- **PreImport ceiling was ~16 parallel tasks** (one per file, `taskSlot=1` each)
  bounded by per-task single-threaded parquet reading — not slot capacity.
- **IndexBuilding parallelized at the segment level, not the job level**: 16
  bulk-insert jobs reported `IndexBuilding:12` simultaneously while ~47 cores
  ran concurrent per-segment HNSW builds. Client-visible per-job progress
  parked at 80-89% across 10+ jobs because jobs only flip to Completed when
  their *last* segment indexes. Then completions arrived in ~5-min bursts.
- HNSW build kept Milvus RSS tight (3-5 GiB) — Knowhere's HNSW builder streams
  from MinIO rather than buffering the whole segment.

## Post-load state

```
$ docker stats --no-stream
milvus-standalone   125.3 GiB / 184.3 GiB   11% CPU
milvus-minio          1.4 GiB                 <1% CPU
milvus-etcd          64.0 MiB                 1% CPU

$ utility.load_state('offers')                        → Loaded
$ utility.loading_progress('offers')                  → 100%
$ utility.get_query_segment_info('offers')            → 224 segments
    median rows/seg 708,867; range 488,457 – 1,021,233
```

### MinIO storage breakdown

```
    53.50 GB   7,762 objs  files/index_files      (HNSW + INVERTED + stats)
   103.94 GB     672 objs  files/insert_log       (raw columnar binlogs)
     0.48 GB     224 objs  files/stats_log        (per-segment stats)
     0.00 GB      53 objs  files/wp               (write-path metadata)
    86.46 GB      16 objs  bulk_offers/           (staged input parquet — can delete)
  ─────────────────────────
   244.38 GB   8,727 objs  TOTAL
```

`index_files` at 53.5 GB matches APRIL_20's 52 GiB HNSW footprint — the vector
index dominates. `insert_log` at 104 GB is the per-field binlog data
(embedding fp16 bytes + all scalar fields).

### Why RSS is 125 GiB (vs APRIL_20_hnsw_reindex.md's 94.9 GiB)

This run's field footprint is larger than APRIL_20's. APRIL_20 was a re-index
of a collection with only {id, offer_embedding, 7 scalar arrays}. This run
adds 5 category arrays, 7 text fields (including `description` up to 65 KB),
and `n` — 13 additional mmap'd fields whose first-query reads pull pages into
the kernel page cache.

Expected trajectory: 125 GiB is the warm-cache high-water mark. Under memory
pressure the kernel will evict mmap'd scalar pages back to disk. The
RAM-resident budget is:

- HNSW index (fp16, 159M × 128): ~53 GB
- `id` VARCHAR PK + stats: ~3-5 GB
- `vendor_ids` + `catalog_version_ids` arrays + INVERTEDs: ~10-20 GB
- Milvus Go heap + buffers: ~3-5 GB
- **Baseline resident:** ~70-80 GB

The remaining ~45 GB in the current measurement is mmapped scalar field pages
from the warmup queries and load-time reads. This will fluctuate with query
workload, not grow unboundedly.

## Smoke test results (post-load, warm cache)

### HNSW search latency — 20 trials, random unit-norm fp16 vectors, limit=10

| ef | min | **p50** | p95 | max |
|---|---|---|---|---|
| 16  | 14.9 | **16.0** | 17.2 | 23.4 |
| 64  | 17.0 | **18.7** | 19.5 | 20.3 |
| 256 | 15.6 | **18.6** | 25.0 | 27.2 |

All ms. Essentially identical to APRIL_20 (15.5 / 19.2 / 17.7 ms p50) —
confirms HNSW index integrity and that the mmap footprint isn't affecting
vector search throughput.

### Filtered HNSW (INVERTED scalar filter + HNSW search)

```
expr: array_contains(vendor_ids, '<uuid>')       (single-vendor filter)
  ef= 64   p50=32.1ms   p95=35.1ms
  ef=256   p50=32.0ms   p95=32.8ms

expr: array_contains(catalog_version_ids, '<uuid>')
  ef= 64   p50=20.5ms   p95=30.7ms
  ef=256   p50=20.4ms   p95=21.0ms
```

Filter adds ~2-15 ms on top of unfiltered HNSW. INVERTED index resident in
RAM, filter predicate evaluated before graph traversal.

### Pure-INVERTED scalar lookup (no vector)

```
array_contains(vendor_ids, '<uuid>')            → 16,384 matches in 275 ms
array_contains(catalog_version_ids, '<uuid>')   → 16,384 matches in 324 ms
```

(Limit capped at 16384 — the sampled vendor/catalog has more matching rows;
this is read throughput, not match density.)

## What worked / what to keep

1. **Indexes-before-bulk-insert.** Every other option tested was a footgun.
2. **Pipelined stage+submit.** Overlapping the tail of staging with the head
   of bulk_insert submission saved ~3-4 min vs `stage_all` then batch-submit.
3. **Field-level mmap via `FieldSchema(mmap_enabled=True)`.** Simpler than
   `alter_collection_field` post-creation, and works correctly at
   `create_collection` time.
4. **Process-pool staging with 16 workers** — disk-read saturated the source
   NVMe; bumping workers further would've just thrashed.
5. **Validation run first.** Single-bucket (`--bucket 'bucket=00.parquet'`)
   caught the post-hoc-index bug before committing to the 2-hour full run.

## Gotchas logged for future

- `utility.index_building_progress` returning `state=Finished` with
  `indexed_rows=0` on a non-empty collection is NOT success — it's the
  "no build task scheduled" response. Always check `indexed == total`.
- Sort-compaction happens per-segment INSIDE the bulk-insert pipeline (it's
  the "Sorting" state, not an external compaction). It dominates the ~30-min
  tail for late-submitted buckets because of slot FIFO draining.
- Milvus's per-bucket progress reporting is coarse (10/40/70/80-89/100) and
  the 80-89% band silently tracks inner `imported_rows / total_rows` during
  IndexBuilding. Don't interpret a long 80% plateau as stalled.
- bulk-insert pipeline `Completed` state in Milvus 2.6 includes
  IndexBuilding — `wait_for_jobs` returning means indexes are already built
  on those segments. The post-hoc verify pass in the script is for safety,
  not work.

## Reproduce

```
# Pre-flight: stack up, volumes empty, collection dropped
cd /home/max/milvus && docker compose up -d
cd /home/max/simplesystem-embedding
nohup uv run --no-project --with pymilvus --with pyarrow --with numpy --with boto3 \
  python scripts/milvus_bulk_import.py \
  > logs/full_run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Single-bucket validation first:

```
uv run ... python scripts/milvus_bulk_import.py --bucket 'bucket=00.parquet'
```
