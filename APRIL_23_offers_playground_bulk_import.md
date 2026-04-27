# Playground Collection Bulk Import — 2026-04-23 15:03 → 15:17 UTC

Fresh bulk import of 18,330,690 × 128d fp16 offer embeddings + 8 scalar fields
into a new `offers_playground` collection in the same single-node Milvus
2.6.15 that already hosts the 159M-row `offers` collection. Inline HNSW +
INVERTED index build during the `do_bulk_insert` pipeline, all non-PK,
non-vector fields `mmap.enabled=true` (data), and both INVERTED indexes
additionally built with index-level `mmap.enabled=true`.

Orchestrator: `scripts/milvus_bulk_import_playground.py` (9-worker
ProcessPool staging, pipelined `do_bulk_insert` submission). Mirrors the
APRIL_21 importer structure with three material schema differences noted
below.

Logs:
- validation (data_8 only): `logs/playground_validate_20260423_150011.log`
- full run: `logs/playground_full_20260423_150345.log`

## Summary

```
rows:              18,330,690
stage+submit:         59.4s   (9 workers, all 9 jobs submitted inside 1s)
ingest wait:         761.6s   (12.7 min — PreImport → Import → Sort → IndexBuilding)
flush:                 0.7s
vector index:          0.0s   (verify-only; HNSW M=16 efC=360)
idx playground_vendorId   0.0s (built inline; INVERTED mmap)
idx playground_articleId  0.0s (built inline; INVERTED mmap)
load:                 11.5s
total wall:          836.2s   (13.9 min)
num_entities:      18,330,690 ✓
```

Post-load Milvus RSS: **82.73 GiB** — this includes *both* `offers` and
`offers_playground` loaded simultaneously. Attributable to
`offers_playground` on MinIO: **~11.4 GB** (≈6 GB HNSW + 5.4 GB binlogs),
proportional to 18.3M rows being ~12% of `offers`' 159M.

## Source dataset

`/data/datasets/offers_playground_elastic_with_categories.parquet/`
- 9 files `data_0.parquet` … `data_8.parquet`, 18 GB on disk
- embeddings stored as `list<float>` (fp32) — **converted fp32 → fp16 at
  stage time** (APRIL_21's source was already fp16; extra `.astype(fp16)`
  step in `convert_batch`)
- two row populations:

  | population | rows | share | fields populated |
  |---|---|---|---|
  | "enriched" | 17,484,763 | 95.4% | all fields |
  | "sparse"   |    845,927 |  4.6% | playground_* + name/article/ean/mfgName + embedding only |

  The sparse population is entirely contained in `data_8.parquet` (the
  smallest file at 845,927 rows, exactly the sparse count). This made it
  a perfect single-file validation target — 100% null coverage on the
  nullable fields.

## Schema and index plan

10 fields total. Only the PK and vector are RAM-resident; everything else
(including INVERTED index storage) is disk-backed via mmap.

| # | field | type | nullable | index | field mmap | index mmap |
|---|---|---|---|---|---|---|
| 1 | `playground_id` | VARCHAR(96) PK | no | — | no | — |
| 2 | `offer_embedding` | FLOAT16_VECTOR(128) | no | HNSW M=16 efC=360 COSINE | no | no |
| 3 | `playground_vendorId` | VARCHAR(64) | no | INVERTED | **yes** | **yes** |
| 4 | `playground_articleId` | VARCHAR(96) | no | INVERTED | **yes** | **yes** |
| 5 | `name` | VARCHAR(256) | no | — | yes | — |
| 6 | `manufacturerName` | VARCHAR(128) | no | — | yes | — |
| 7 | `manufacturerArticleNumber` | VARCHAR(128) | **yes** | — | yes | — |
| 8 | `manufacturerArticleType` | VARCHAR(512) | **yes** | — | yes | — |
| 9 | `ean` | VARCHAR(32) | no | — | yes | — |
| 10 | `article_number` | VARCHAR(64) | no | — | yes | — |

### Three material differences from APRIL_21's `offers` schema

1. **Nullable VARCHAR fields.** `manufacturerArticleNumber` and
   `manufacturerArticleType` declare `nullable=True` to preserve parquet
   nulls through `bulk_insert`. Milvus 2.6 honors these against `is null`
   / `is not null` expressions, verified end-to-end against the full set
   (845,927 nulls + 17,484,763 not-null = 18,330,690 exactly).

2. **INVERTED index mmap.** APRIL_21 kept its INVERTED indexes
   (`vendor_ids`, `catalog_version_ids`) RAM-resident. Here both indexed
   fields have `params={"mmap.enabled": "true"}` passed to
   `create_index`, pushing the posting lists to disk. Still 5.2 ms p50
   for equality lookups post-warmup (see smoke tests).

3. **Dropped fields.** No categories (`category_l1..l5`), no
   `vendor_ids`/`catalog_version_ids` arrays, no `description`, no
   `playground_keywords`, no `n`. Deliberate scope cut — playground use
   case only needs vector search + vendor/article ID filtering + a few
   display fields. The 65,535-byte VARCHAR cap on `description` (which
   would have required either truncation, external storage, or raising
   `proxy.maxVarCharLength` — source max was 191,009 chars) became moot.

## Timeline — full run

```
 0:00 → 0:59    stage+submit     9× parquet convert (fp32→fp16→uint8) + MinIO upload
 0:09           data_8.parquet submitted first (smallest)
 0:59           all 9 jobs submitted (within 1s of each other)
 0:59 → 1:20    data_8 PreImport/Import/Sort/IndexBuilding (80s — matches validation)
 1:20           data_8 Completed (first)
 1:20 → 11:41   remaining 8 jobs in parallel IndexBuilding
 2:30           RSS snapshot: 75.6 GiB, 8 jobs at 70-80%
 4:30           data_0 Completed (270s total)
 7:00           data_7, data_3, data_4 Completed within 10s (421-431s)
 8:57           RSS snapshot: 75.2 GiB, 4 jobs at 80% plateau
10:11           data_2 Completed (611s)
10:21           data_1 Completed (621s)
12:31           data_6 Completed (751s)
12:41           data_5 Completed (761s) — all 9 done
12:42           flush 0.7s, index verify instant
13:54           col.load() complete — 11.5s
```

Validation run timeline (single-file `--file data_8.parquet`):
stage+submit 9.2s → ingest 80.0s → flush 0.7s → load 7.1s → **99.2s total**.

## Smoke tests (post-load, warm cache)

### HNSW vector search — 20 trials, random unit-norm fp16 vectors, limit=10

| ef  | p50 | p95 | max |
|-----|-----|-----|-----|
| 16  | 1.7 | 1.9 | 2.4 |
| 64  | 1.9 | 2.0 | 2.1 |
| 256 | 2.7 | 2.9 | 3.0 |

All ms. Roughly 8× faster than APRIL_21's 159M-row collection (16–19 ms p50
at the same ef values) — scales with collection size. The HNSW index
resident in RAM is the dominant factor; the mmap'd scalar fields and
INVERTED indexes do not appear in the vector-search hot path.

### INVERTED scalar filter (data + index both mmap'd)

```
expr: playground_vendorId == '<uuid>'   (count 41,618 matches)
  count(*)                         5.2 ms
  filtered vector search (ef=64)   5.2 ms, 10 hits, all correct vendor
```

### Null semantics end-to-end

```
manufacturerArticleNumber is null       →    845,927   (10.4 ms)
manufacturerArticleNumber is not null   → 17,484,763   (12.0 ms)
                                          ──────────
                                          18,330,690 ✓
```

Exactly the sparse/enriched split predicted from the source-side parquet
scan.

## Collection segmentation

```
$ utility.get_query_segment_info('offers_playground')  → 17 segments
    min/median/max rows/seg: 845,927 / 1,092,000 / 1,226,613
```

One segment per source file plus a couple of splits for the larger
enriched files.

## MinIO storage (pre-cleanup)

Shared across both collections:

```
    86.46 GB     16 objs  bulk_offers/             ← APRIL_21 staging (stale)
     5.07 GB      9 objs  bulk_offers_playground/  ← this run's staging
    59.48 GB   9288 objs  files/index_files        (+5.98 GB vs APRIL_21 baseline)
   109.29 GB    723 objs  files/insert_log         (+5.35 GB)
     0.54 GB    241 objs  files/stats_log          (+0.06 GB)
     0.00 GB    159 objs  files/wp
  ─────────────────────
   260.83 GB  10436 objs  TOTAL
```

Both `bulk_offers/` and `bulk_offers_playground/` deleted after verification
(91.5 GB recovered). Final MinIO: **169.30 GB, 10,409 objs**.

## What worked / what to keep

1. **APRIL_21's playbook transfers.** The three non-negotiables from the
   prior run — indexes-before-bulk-insert, pipelined stage+submit,
   field-level mmap at schema creation — applied unchanged and hit clean
   on the first full run. No aborts. No tuning.

2. **Single-file validation on the sparse-only file.** `data_8.parquet`
   happens to contain exactly the 845,927 sparse rows with 100% null
   coverage on the two nullable fields. Made it the best possible test
   for the `nullable=True` + parquet-null + Milvus-bulk-insert path
   before committing 12.7 min of full-run ingest time.

3. **fp32 → fp16 on stage.** One-line `.astype(np.float16)` added before
   the existing `.view(np.uint8)` byte-reinterpret. No schema impact —
   Milvus still sees FLOAT16_VECTOR with raw bytes as `list<uint8>`.

4. **9 workers for 9 files.** All 9 `do_bulk_insert` jobs submitted inside
   1 second of each other. IndexBuilding parallelized across ~9 segment
   batches simultaneously, tail bucket completed 12:41 after first at
   1:20 — similar shape to APRIL_21's 16-job tail but compressed.

## Gotchas logged for future

- **`mmap.enabled` vs `mmap_enabled`.** Field-schema mmap uses
  `FieldSchema(..., mmap_enabled=True)` (Python keyword). Index-level
  mmap uses `create_index(..., index_params={"params":
  {"mmap.enabled": "true"}})` — server-side param name with a dot, value
  a string literal `"true"`. These are two different switches on two
  different storage areas and both are needed if you want both raw
  column and index posting lists on disk.

- **Description length cap.** Source has descriptions up to 191,009
  chars. Milvus VARCHAR hard cap is 65,535 bytes. Options are (a) raise
  `proxy.maxVarCharLength` in milvus.yaml (server restart), (b) declare
  with `enable_analyzer=True` to route through the internal Text path
  (2 MB cap, but field becomes a BM25/search primitive), or (c) external
  storage. We chose (d) — drop the field. Future imports that need full
  descriptions should pick (a).

- **Sparse-row population is a real concern for filtered search UX.**
  The 845,927 sparse rows have valid embeddings but null
  `manufacturerArticleNumber`/`manufacturerArticleType`. Unfiltered
  vector search includes them; filters like
  `manufacturerArticleType == "X"` silently exclude them. App-level
  awareness needed.

## Reproduce

```
# Preflight
docker ps | grep milvus          # stack up
uv run --no-project --with pymilvus python -c \
  "from pymilvus import connections, utility; \
   connections.connect(alias='default', host='localhost', port='19530'); \
   print(utility.list_collections())"

# Validation (single sparse-only file, ~100s)
nohup uv run --no-project --with pymilvus --with pyarrow --with numpy --with boto3 \
  python scripts/milvus_bulk_import_playground.py --file 'data_8.parquet' \
  > logs/playground_validate_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full run (~14 min)
nohup uv run --no-project --with pymilvus --with pyarrow --with numpy --with boto3 \
  python scripts/milvus_bulk_import_playground.py \
  > logs/playground_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Post-run cleanup of staging prefixes (safe; milvus ingest already consumed them)
docker exec milvus-minio mc alias set localroot http://localhost:9000 minioadmin minioadmin
docker exec milvus-minio mc rm --recursive --force localroot/a-bucket/bulk_offers_playground/
```

Commit: `97c2f09 Add bulk importer for offers_playground collection`.
