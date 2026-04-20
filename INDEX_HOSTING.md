# Milvus Hosting Plan: 159M Vector Index

## Overview

Production deployment of Milvus for ~159M offer embeddings (128-dim float16) using Docker Compose on a dedicated Hetzner server.

**Data summary:**
- **159,275,274 records** across 16 parquet buckets (~10M rows each, ~2.7 GB each)
- Schema: `row_number` (int64), `id` (varchar), `offer_embedding` (128-dim **float16**, stored as `list<halffloat>`)
- Source data: ~40 GB parquet on disk
- Embeddings are trained and stored in fp16 — **use `FLOAT16_VECTOR` in Milvus**, not `FLOAT_VECTOR`. Converting to fp32 doubles wire traffic, disk, and RAM for zero precision gain.

This plan was validated end-to-end on a local Apple Silicon Mac (M-series, 128 GB RAM, 16 cores) running Docker via OrbStack. See the "Validation run notes" section at the bottom for concrete numbers and gotchas.

## Hardware Requirements

### Memory (critical constraint)

Milvus loads the full collection into RAM for search. With `FLOAT16_VECTOR` the raw vector footprint is 159M × 128 × 2 bytes = **~38 GB** — exactly half of the fp32 equivalent.

| Index Type            | Vectors in RAM (fp16) | Total RAM Needed | Recall@10 | Build cost |
|-----------------------|-----------------------|------------------|-----------|------------|
| **FLAT** (brute force)    | ~38 GB            | **50–60 GB**     | 100%      | **zero** — no training, no clustering |
| IVF_FLAT (nlist=4096) | ~38 GB                | **50–60 GB**     | 95–99%    | kmeans training on each segment |
| IVF_PQ (m=16)         | ~2.5 GB               | **12–18 GB**     | 90–95%    | kmeans + product quantization training |
| HNSW (M=16)           | ~58 GB                | **75–90 GB**     | 97–99%    | graph build per segment |

Add ~10–15 GB for Milvus process overhead, etcd, MinIO, OS.

**FLAT is much more attractive at fp16 than the old fp32 math suggested** — the full index fits in 60 GB of RAM, has zero build cost, and gives exact search. Brute-force latency on 159M fp16 vectors with 16 cores is 1–3 seconds per query (SIMD'd by Milvus). For offline analysis, batch scoring, or a warm-up deployment where you haven't committed to a training recipe, FLAT is the right default. Switch to IVF_PQ or HNSW only if you need sub-100ms queries under concurrent load.

### Disk

| Component                            | Size         |
|--------------------------------------|-------------|
| Milvus segments, FLAT/IVF_FLAT (fp16) | ~40 GB     |
| Milvus segments, IVF_PQ (fp16)       | ~5 GB       |
| etcd + WAL + temp                    | ~10 GB      |
| Docker images                        | ~5 GB       |
| Source parquet (for import)          | ~40 GB      |
| **Staged parquet for bulk insert**   | **~40 GB**  |
| **Total (FLAT / IVF_FLAT)**          | **~135 GB** |
| **Total (IVF_PQ)**                   | **~100 GB** |

NVMe strongly preferred for import speed and segment loading. **The bulk insert path temporarily doubles the parquet footprint** because files have to be rewritten with a Milvus-compatible schema and staged into MinIO before they're ingested — budget for ~80 GB of parquet on disk during the import window, then the staging copy can be deleted.

### CPU

Milvus parallelizes search across cores. 8–16 cores is adequate; 32+ for high concurrent QPS.

## Hetzner Recommendations

### Recommended: AX162-R (dedicated, best value for sustained use)

- AMD Ryzen 9 7950X3D, 16 cores / 32 threads
- **128 GB DDR5 ECC RAM** — fits any index type at fp16 with >60 GB headroom
- 2× 1.92 TB NVMe SSD
- ~€82/mo
- Best price/performance for a long-running deployment

### Alternative: CCX63 (cloud, more headroom)

- 48 dedicated AMD EPYC vCPUs
- **192 GB RAM** — comfortable margin for IVF_FLAT or HNSW
- 360 GB NVMe
- ~€270/mo
- Easier to provision/deprovision, better for experimentation

### Budget: CCX53 or EX44 (cloud/dedicated, 128 / 64 GB)

- CCX53: 32 vCPU, **128 GB RAM**, 240 GB NVMe — ~€180/mo (comfortable for any index type at fp16)
- EX44: Intel i5-13500, **64 GB RAM**, 2× 512 GB NVMe — ~€44/mo (FLAT/IVF_FLAT tight, IVF_PQ roomy; HNSW out)

### Summary

With fp16 vectors the RAM requirement collapses — **even a 64 GB box can host FLAT or IVF_FLAT** for 159M vectors, leaving the EX44 as a real option for any index type.

| Server   | Type      | RAM    | Cores | Disk      | Price   | Index Types at fp16               |
|----------|-----------|--------|-------|-----------|---------|-----------------------------------|
| AX162-R  | Dedicated | 128 GB | 16/32 | 2× 1.9 TB | €82/mo  | FLAT, IVF_FLAT, IVF_PQ, HNSW      |
| CCX63    | Cloud     | 192 GB | 48    | 360 GB    | €270/mo | all of the above + headroom       |
| CCX53    | Cloud     | 128 GB | 32    | 240 GB    | €180/mo | FLAT, IVF_FLAT, IVF_PQ, HNSW      |
| EX44     | Dedicated | 64 GB  | 14/20 | 2× 512 GB | €44/mo  | FLAT, IVF_FLAT, IVF_PQ (tight)    |

## Deployment Plan (Docker Compose)

### Step 1: Provision server and install Docker

```bash
apt-get update && apt-get install -y docker.io docker-compose-plugin
```

### Step 2: Deploy Milvus Standalone

Use the official Milvus docker-compose (includes etcd + MinIO + Milvus):

```bash
mkdir -p ~/milvus && cd ~/milvus
curl -L https://github.com/milvus-io/milvus/releases/download/v2.6.14/milvus-standalone-docker-compose.yml \
  -o docker-compose.yml
```

Adjust `docker-compose.yml` for this workload:

```yaml
services:
  standalone:
    environment:
      - QUERY_NODE_CACHE_MEMORY_LIMIT=100gb    # generous for 128 GB machine
      - DATA_COORD_SEGMENT_MAX_SIZE=1024        # 1 GB segments
      - DATA_NODE_INSERT_BUF_SIZE=67108864      # 64 MB insert buffers
    deploy:
      resources:
        limits:
          memory: 120g
    volumes:
      - ./volumes/milvus:/var/lib/milvus
  etcd:
    command: >
      etcd
      --quota-backend-bytes=4294967296
      --auto-compaction-mode=revision
      --auto-compaction-retention=1000
    volumes:
      - ./volumes/etcd:/etcd
  minio:
    volumes:
      - ./volumes/minio:/minio_data
```

Start:

```bash
docker compose up -d
# Wait for ready
while ! nc -z localhost 19530; do sleep 1; done
echo "Milvus ready on port 19530"
```

### Step 3: Transfer data to server

```bash
# From the current machine:
rsync -avP /home/max/workspaces/simplesystem/data/offers_embedded.parquet/ \
  user@hetzner-server:/data/offers_embedded.parquet/
```

### Step 4: Create collection and import data

**Use bulk insert, not streaming inserts.** Streaming via `client.insert()` saturates Milvus server-side CPU at ~45–60k rows/s regardless of how many client threads you throw at it — the bottleneck is per-segment index build and WAL write running concurrently with ingestion. For 159M rows that's ~60 minutes wall clock.

Bulk insert via MinIO is ~3x faster (~15 min) because Milvus ingests parquet files directly from object storage into sealed segments, skipping the growing-segment + WAL path entirely.

**The catch**: Milvus v2.6's parquet reader expects `FLOAT16_VECTOR` columns as `list<uint8>` (raw fp16 bytes, 2 × dim bytes per row), *not* the `list<halffloat>` your source parquet almost certainly has. You must rewrite the vector column before upload. It's a mechanical view: `fp16_ndarray.view(np.uint8)` is zero-copy.

Reference implementation in this repo:

- `scripts/milvus_bulk_import.py` — rewrites every `bucket=NN.parquet` to the uint8 schema, uploads to MinIO in parallel, submits one `do_bulk_insert` job per file, polls until completion, creates a FLAT index, loads the collection.
- `scripts/milvus_verify.py` — stats + random query + self-hit sanity check.

The critical code paths:

```python
# 1. Parquet schema rewrite (list<halffloat>[128] -> list<uint8>[256])
emb_obj = batch.column("offer_embedding").to_numpy(zero_copy_only=False)
emb_2d = np.stack(emb_obj)            # (n, 128) float16
emb_u8 = emb_2d.view(np.uint8)        # (n, 256) uint8, zero-copy

n, width = emb_u8.shape
flat = pa.array(emb_u8.reshape(-1), type=pa.uint8())
offsets = pa.array(np.arange(0, n * width + 1, width, dtype=np.int32))
new_emb = pa.ListArray.from_arrays(offsets, flat)

new_batch = pa.RecordBatch.from_arrays(
    [batch.column("row_number"), batch.column("id"), new_emb],
    names=["row_number", "id", "offer_embedding"],
)

# 2. Milvus collection schema
from pymilvus import (
    Collection, CollectionSchema, FieldSchema, DataType,
    connections, utility,
)

schema = CollectionSchema([
    FieldSchema(name="row_number", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="offer_embedding", dtype=DataType.FLOAT16_VECTOR, dim=128),
])
col = Collection("offers", schema=schema)

# 3. Upload staged parquet files to Milvus's MinIO bucket (default: "a-bucket")
import boto3
from botocore.client import Config
s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9010",  # see MinIO port note below
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)
s3.upload_file(staged_path, "a-bucket", f"bulk_offers/{name}")

# 4. Submit bulk insert jobs (one per file, all run concurrently in Milvus)
job_id = utility.do_bulk_insert(
    collection_name="offers",
    files=[f"bulk_offers/{name}"],
)
state = utility.get_bulk_insert_state(job_id)  # poll until Completed or Failed

# 5. FLAT index (zero build cost)
col.create_index(
    field_name="offer_embedding",
    index_params={"index_type": "FLAT", "metric_type": "COSINE"},
)
col.load()
```

**Why you need to expose MinIO to the host**: the stock `milvus-standalone-docker-compose.yml` doesn't map MinIO ports externally. To upload from a host-side Python client you need to add `ports: ["9010:9000", "9011:9001"]` to the `minio` service (non-default host ports avoid collisions with other MinIO instances on dev machines). The `minio:9000` internal hostname used by Milvus is untouched.

**Expected wall time** for 159M records on a 16-core box:
- Parquet rewrite + upload: **3–5 min** (CPU-bound on the conversion; use 8–12 parallel workers)
- Bulk insert jobs: **10–15 min** (Milvus processes 4 segments concurrently, each ~2.5 GB)
- Flush + FLAT index create + load: **<1 min**
- **Total: ~15–20 min**

Streaming insert fallback: if for some reason bulk insert is unavailable (e.g. MinIO not reachable from the importer), use `scripts/milvus_import.py` which uses the columnar `Collection.insert([col1, col2, col3])` API with `FLOAT16_VECTOR` and a 2D fp16 ndarray passed through zero-conversion. Expect ~45–60 min wall time and Milvus pinned at 100% CPU throughout.

### Step 5: Verify

```python
from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("http://localhost:19530")
stats = client.get_collection_stats("offers")
print(stats)  # Should show ~159,275,274 entities

# Query vectors for a FLOAT16_VECTOR field must be fp16 (ndarray with dtype=np.float16)
q = np.random.randn(128).astype(np.float32)
q /= np.linalg.norm(q)
q_f16 = q.astype(np.float16)

results = client.search(
    collection_name="offers",
    data=[q_f16],
    limit=10,
    search_params={"metric_type": "COSINE", "params": {}},  # FLAT: no nprobe
    output_fields=["row_number", "id"],
)
for hit in results[0]:
    print(hit["entity"]["row_number"], hit["distance"])
```

**Gotcha — fp16 return format**: `client.query(..., output_fields=["offer_embedding"])` returns the fp16 vector as a **single-element list containing raw bytes**, i.e. `[b"\x9f/..."]`, not a numpy array and not a list of floats. To decode for a self-hit sanity check:

```python
def decode_f16(v):
    if isinstance(v, list) and len(v) == 1:
        v = v[0]
    if isinstance(v, (bytes, bytearray)):
        return np.frombuffer(v, dtype=np.float16)
    return np.asarray(v, dtype=np.float16)
```

See `scripts/milvus_verify.py` for the end-to-end sanity check (stats, random query, self-hit test that pulls a row's own vector and confirms it's the top-1 match with score 1.0).

### Step 6: Expose for production

Milvus listens on port 19530 (gRPC) and 9091 (metrics). For production:

- Restrict port 19530 to application servers via firewall / Hetzner network
- Monitor via Prometheus scraping :9091/metrics
- Key metrics: `milvus_proxy_search_latency_bucket`, `milvus_querynode_loaded_segment_total`

## Index Type Decision Guide

| Priority                        | Choose       | Why                                                          |
|---------------------------------|--------------|--------------------------------------------------------------|
| Exact recall, simplest setup    | **FLAT**     | 100% recall, zero build time, ~55 GB RAM at fp16, 1–3s/query |
| 95–99% recall, fast search      | IVF_FLAT     | ~55 GB RAM at fp16, sub-100ms with `nprobe=64`              |
| Lowest latency                  | HNSW         | ~1–5ms search, ~75–90 GB RAM at fp16                        |
| Tight budget / small memory     | IVF_PQ       | ~15 GB RAM, 90–95% recall                                   |
| GPU acceleration                | GPU_IVF_FLAT | Sub-ms search, needs GPU instance                           |

**Start with FLAT.** It's zero-effort to build, gives exact results, and the fp16 footprint fits on any machine in the options table above. Only upgrade to IVF_PQ or HNSW once you've measured search latency against actual workload and found brute force too slow.

## Validation run notes

The plan above was validated end-to-end on a 128 GB Apple Silicon Mac (16 cores, OrbStack Docker) in April 2026. Key observations:

**Actual timings (159,275,274 rows):**
- Streaming insert with columnar `Collection.insert` + `FLOAT16_VECTOR`: 45–60k rows/s, ETA ~60 min. Milvus pinned at 1600% CPU throughout; client-side at <10% CPU. Adding client-side parallelism does not help — the bottleneck is Milvus-internal.
- Streaming insert with row-oriented `MilvusClient.insert` + `FLOAT_VECTOR` + fp32 cast: 42–45k rows/s, ETA ~60 min. The fp32 cast was pure waste.
- **Bulk insert via MinIO**: 3.5 min to stage all 16 buckets (rewrite fp16 column to uint8 + upload, 4 parallel workers), 13.5 min for Milvus to process all bulk jobs, <1 min to flush + create FLAT index + load. **Total: ~17 min.** Would shave ~2 min by using 8–12 stage workers instead of 4.

**OrbStack / Apple Silicon specifics:**
- Milvus v2.6.14 has native `linux/arm64` images for the `milvusdb/milvus`, `etcd`, and `minio` containers. Pin `platform: linux/arm64` on each service in docker-compose.yml to avoid Rosetta emulation.
- OrbStack's VM size caps at the host's physical CPU count (16 on a 16-core machine) and `memory_mib` from `orb config show`. Docker's `Total Memory` reports the *currently ballooned* size, not the cap — don't confuse the two. Our 128 GB host showed 98 GiB at idle and grew to 121 GiB under Milvus load, both within the 124 GiB (126976 MiB) cap.
- `docker compose down -v` removes the bind-mounted `volumes/` directories. Recreate them with `mkdir -p volumes/{milvus,etcd,minio}` before bringing the stack back up.

**MinIO port collisions:**
Dev machines often already have something bound to 9000/9001 (another MinIO instance, a local S3 gateway). Our stock compose had to be adjusted to either drop the host port mappings entirely (if you don't need host-side access to this MinIO) or remap to `9010:9000` / `9011:9001` (required for the bulk-insert staging upload). On a dedicated Hetzner server this is a non-issue — ports are free.

**pymilvus quirks hit during validation:**
1. `FLOAT16_VECTOR` parquet schema for bulk insert must be `list<uint8>` (or `list<item: uint8>`), not `list<halffloat>`. The wrong schema produces `error: schema not equal, err=field 'offer_embedding' type mis-match, expect arrow type 'list<item: uint8, nullable>'`.
2. `client.query(..., output_fields=["offer_embedding"])` on a `FLOAT16_VECTOR` field returns `[bytes]` (single-element list), not a numpy array. Decode with `np.frombuffer(v[0], dtype=np.float16)`. See `scripts/milvus_verify.py`.
3. `MilvusClient.insert(data=...)` only accepts row-oriented `list[dict]`. For columnar inserts use the older `Collection.insert([col1, col2, col3])` API — meaningfully faster because it skips per-row dict construction, though the win is modest when Milvus is already CPU-bound.
4. Bucket file sanity check: our source had eight `.parquet.<hash>` partial-download sidecars at one point. The importer should match `^bucket=\d{2}\.parquet$` explicitly, not just `.endswith(".parquet")`, to avoid picking up temp files.

**Measured search latency (FLAT, 159M × 128 fp16, single client, no concurrent load, 20 trials per cell after 3-query warmup):**

| Setup                        | min     | **p50**     | p95     |
|------------------------------|---------|-------------|---------|
| 1 query, limit=5             | 1859 ms | **2415 ms** | 4103 ms |
| 1 query, limit=10            | 2193 ms | **2486 ms** | 4811 ms |
| 1 query, limit=100           | 2139 ms | **2541 ms** | 4241 ms |
| 5 queries batched, limit=5   | 2771 ms | **3338 ms** | 6727 ms |
| 5 queries batched, limit=100 | 3072 ms | **3670 ms** | 5575 ms |

At p50 ≈ 2.4 s on 159M × 256 B = ~40 GB scanned per query, Milvus achieves ~17 GB/s effective memory throughput through the OrbStack VM — reasonable for a Docker container on Apple Silicon's unified memory. A bare-metal Hetzner AX162-R with DDR5 should roughly double that and land in the 1–1.5 s range per query for the same workload.

Behaviors worth knowing before you design around these numbers:

- **`limit` (top-k) barely affects latency.** top-5 vs top-100 differ by <100 ms because FLAT scans every vector regardless; only the final top-k selection varies, and that's a rounding error next to 159M comparisons.
- **Batching amortizes extremely well.** 5 queries in ~3.3 s ≈ 660 ms per query — a 3.6× throughput improvement over serial single queries. Milvus reuses each memory scan to score against all query vectors in the batch at once, so after the first query in a batch the rest are effectively free of memory bandwidth cost. For offline scoring or bulk probe workloads, batch 20–100 queries at a time.
- **Tail latency is ~2× median.** p95 at ~4.1 s vs p50 at ~2.4 s on a shared dev machine. The tail is dominated by background noise (Milvus segment management, OrbStack, macOS scheduling). Expect meaningfully tighter p95 on a dedicated Hetzner box, but **do not** count on FLAT for any SLA tighter than "eventually".
- **Cold and warm are identical.** Five random fresh query vectors without warmup landed at 2.5–3.2 s, matching the warm distribution. FLAT doesn't benefit from query-side caching — every search pays the full scan cost. This also means the first query after `load_collection()` is not anomalously slow (assuming segments are resident), which makes it easy to reason about production p50.

**Acceptable for batch scoring, offline evaluation, and qualitative probes. Not acceptable for interactive serving.** If you need sub-100 ms queries under concurrent load, rebuild on top of FLAT with IVF_PQ (`nprobe=64`) for ~50–150 ms per query at 90–95% recall, or IVF_FLAT for ~100–300 ms per query at 95–99% recall. Both pay a one-time index build cost you skipped with FLAT, but can be done in place without re-importing.

### Bucket=00 IVF_FLAT streaming run (April 2026)

A follow-up run on the same 128 GB OrbStack host imported a single bucket (`bucket=00.parquet`, **9,954,348 rows**) into a fresh `offers` collection using the streaming `Collection.insert` path and built an IVF_FLAT index. Purpose was to validate the streaming+IVF_FLAT path end-to-end at a smaller scale before committing to a full 159M import with a non-FLAT index. Script: `~/milvus-offers/import_bucket00.py`.

**Wall-clock timings (9.95M rows):**

| Phase                         | Time      |
|-------------------------------|-----------|
| Streaming insert (50k batches) | **5.6 min** (sustained 29.4k rows/s) |
| Flush                         | 1.9 s     |
| IVF_FLAT build (nlist=4096)   | **15.2 min** |
| Load into memory              | 7.2 s     |
| **Total**                     | **~21 min** |

**Insert throughput was ~2× slower than the earlier FLAT streaming baseline (29k vs 45–60k rows/s).** Same machine, same pymilvus, same `Collection.insert` columnar path, same fp16 ndarray input — only the index differs (FLAT was created *after* insert in both runs, so index type shouldn't matter during insert itself). The delta is either per-run variance or cache/competition from other processes on the host; worth re-measuring if streaming speed matters.

**Gotcha — the source parquet has a lying physical type.** The `offer_embedding` column declares `list<float>` (Arrow `float` = float32), but every value round-trips through `fp16` with **zero loss** across a 128k-element sample. The embeddings are fp16 data widened to fp32 on write, so `astype(np.float16)` on insert is *actually* lossless — not "lossy but acceptable". Verify with `np.array_equal(arr, arr.astype(np.float16).astype(np.float32))` before trusting the bucket.

**Gotcha — `Collection.insert` wants `List[np.ndarray]` for `FLOAT16_VECTOR`, not raw bytes.** Passing `[row.tobytes() for row in emb_f16]` fails with `ParamError: Wrong type for vector field: embedding, expect=List<np.ndarray(dtype='float16')>, got=List<class 'bytes'>`. Correct form: `[np.ascontiguousarray(row) for row in emb_f16]` where `emb_f16` is a `(batch, dim)` fp16 ndarray. The pymilvus error message spells the expected type out — trust it.

**Gotcha — `utility.wait_for_index_building_complete` hangs on stale `pending_index_rows`.** At the end of the IVF_FLAT build, milvus reported `state=Finished, indexed_rows == total_rows` within a few minutes. But `pending_index_rows` stayed at **~5.5M for another ~10 minutes** while milvus re-indexed segments that background compaction had merged. The wait helper kept spinning — it's clearly polling `pending_index_rows` rather than `state`. If you want to cut the tail off, skip the helper and check `state == "Finished"` yourself, then call `col.load()`. The re-indexing of compacted segments happens asynchronously and doesn't block searches.

  This matters for the 159M import: if per-bucket compaction tails scale roughly linearly, the IVF_FLAT build tail across 16 buckets could add **tens of minutes of dead wait time** past the point where the index is actually usable. Use bulk insert (which bypasses the growing-segment → compaction path) if you can, or don't block on the helper.

**Shebang and buffering gotchas (not milvus-specific but bit this run):**
- `#!/usr/bin/env -S uv run --with "pyarrow>=17" ...` fails to parse — `env -S` preserves the literal quotes and uv reads `"pyarrow>=17"` (with quotes) as a package name. Use `--with pyarrow` unquoted.
- Python `print()` output via `nohup script.py > file 2>&1 &` is **block-buffered** (not line-buffered) because stdout is not a TTY, so the log stays empty for minutes. Fix with `PYTHONUNBUFFERED=1` in the environment or `python3 -u`.

### Full 16-bucket IVF_FLAT streaming run (April 2026)

Follow-up to the bucket=00 run above: **all 16 buckets** streamed into the same `offers` collection on the same 128 GB OrbStack host, with the IVF_FLAT index from the bucket=00 run kept in place so that newly-sealed segments would be background-indexed as inserts arrived. Done because the bulk-insert path had been ruled out for this run, so the goal was to validate "streaming + IVF_FLAT, end-to-end, at full scale" and characterize the cost honestly.

Scripts (in `~/milvus-offers/`): `import_all_buckets.py` (resumable per-bucket streamer with state file), `watch_index.py` (post-import poller for index/compaction drain), `verify.py` (load + self-hit + latency).

**Wall-clock timings (159,275,274 rows total, 9.95M per bucket):**

| Phase | Time | Note |
|---|---|---|
| Streaming insert, b01–b15 (15 buckets) | **~1h 50min** | 50k batches, sustained ~22k rows/s |
| Index build, in-progress during inserts + final drain | **~2h 38min** after script end | overlapped partly with inserts; this is the *additional* tail |
| Load collection into memory | **87.6 s** | once everything was drained |
| **Total wall time, start to queryable** | **~4.5 hours** | |

**Streaming throughput drifted down as the collection grew** under the existing IVF_FLAT index:

| Bucket | Time | Sustained throughput |
|---|---|---|
| b00 (initial run, no pre-existing index) | 5.6 min | **29.4k rows/s** |
| b01 | 6.4 min | 26.2k rows/s |
| b02 | 6.8 min | 24.6k rows/s |
| b03 | 8.8 min | 18.9k rows/s |
| b04–b06 | 8.2–8.5 min | 19.6–20.7k rows/s |
| b07–b15 | 6.8–8.3 min | 19.7–24.6k rows/s |

The drop from 29k → ~20k is **CPU contention from the background IVF_FLAT builder** running concurrently with inserts. Milvus pinned at 1600–1800% CPU (essentially saturating all 16 cores via hyperthreads) for almost the entire import. Adding more client-side parallelism would not have helped.

**Memory was the scary part — and the single biggest lesson.** During the import, Milvus RSS grew steadily despite only ~3 GB/bucket of actual vector+id data:

| Snapshot | RSS | Mostly |
|---|---|---|
| After b00 import, settled, loaded | 6.3 GB | loaded sealed segments + fixed overhead |
| Mid-b04 (~38M entities) | 34 GB | + growing segments + compaction scratch + WAL |
| Mid-b06 (~55M entities) | 44 GB | same trajectory |
| **Mid-b12 (~113M entities)** | **80.5 GB** | trending toward the 115 GB cgroup limit |
| **After `col.release()`** (mid-import) | **4.9 GB** | all loaded sealed segments evicted |
| Post-release b13–b15 import | peaked ~10 GB | minimal overhead now |
| Post-import idle, unloaded | 2.7 GB | pure datacoord/etcd/coordinator |
| After final `col.load()` | ~57 GB | all segments back in RAM (vectors+ids+metadata) |

**The single most important takeaway from this run**: ***never keep the collection loaded during a long streaming import***. The loaded sealed segments are mmap'd into Milvus's RSS, which grows linearly with stored data. We hit 80 GB at bucket 12 and were headed for OOM at the 115 GB cgroup limit. Releasing the collection mid-import dropped RSS from **84 GB → 4.9 GB instantly**, freeing ~80 GB of headroom, and the remaining 3 buckets imported uneventfully. The streaming path doesn't need the collection loaded — `col.insert()` writes through the growing-segment path, which is independent of whether sealed segments are queryable. Concrete fix: after the *first* bucket loads the collection (because bucket=00 was a separate run that called `col.load()` at the end), call `col.release()` before starting the multi-bucket streamer. Or have `import_all_buckets.py` call `col.release()` at startup unconditionally.

**Disk usage (MinIO segment store):**

| Snapshot | MinIO size | Notes |
|---|---|---|
| After b00 | 19 GB | small because compaction hadn't run aggressively yet |
| Mid-b04 | 59 GB | |
| Mid-b06 | 84 GB | |
| Post-import, after compaction settled | **253 GB** | includes ~418 dropped pre-compaction segment artifacts awaiting MinIO GC |

253 GB at 159M rows is ~1.6 KB/row, dominated by the dropped intermediate compaction artifacts. After full GC the steady-state should drop to roughly 60–80 GB (raw fp16 vectors + indexes + bookkeeping). Plan for **~300 GB of MinIO disk during a streaming import of this size**, dropping to ~80 GB long-term.

**Compaction drain is the long pole.** When the importer finished writing the last bucket, the index reported only **57.2% indexed (91M / 159M)**. The remaining 68M came from compaction churn: Milvus continuously merged small sealed segments into bigger ones, and each merge produced a brand-new segment that needed its IVF_FLAT index rebuilt from scratch. The drain pattern looked like this:

| t (after script exit) | indexed | pending | state |
|---|---|---|---|
| 0 min | 91.1M | 83.3M | InProgress |
| 30 min | ~110M | ~70M | InProgress |
| 60 min | ~127M | ~80M (oscillating) | InProgress |
| 90 min | 159.3M (= total) | 38.9M | **Finished** ← but still draining |
| 120 min | 159.3M | ~28M | Finished |
| 150 min | 159.3M | ~6M | Finished |
| **158 min** | **159.3M** | **0** | **Finished — drain complete** |

Confirms the bucket=00 prediction: `wait_for_index_building_complete` would have spun on stale `pending_index_rows` for the entire 158 min. We polled `state == Finished AND pending == 0` ourselves via `watch_index.py`. **The state went to `Finished` ~88 min before pending actually drained**, because compaction kept producing new pending work. If your application can tolerate searches against partially-indexed segments, you can `col.load()` and start serving as soon as `state=Finished` the first time — the remaining "pending" is just compaction polishing that doesn't break correctness.

**Final segment shape after compaction settled:**

- 85 sorted L1 Flushed segments (the live data) — average **~1.87M rows/segment**
- 297 sorted L1 Dropped + 121 unsorted L1 Dropped = 418 dropped intermediate compaction artifacts awaiting MinIO GC
- 0 unsorted L1 Flushed (all original streaming-insert segments fully consumed by compaction)
- 0 growing segments

The 1.87M rows/segment is **below** Milvus's nominal 1 GB / ~4M-row segment target, which means compaction stopped short of full consolidation. That's OK — segment count will continue to drift down slowly during normal operation. For nlist=4096 sizing, 1.87M rows × 1/4096 ≈ 460 vectors per cell, which is healthy (kmeans needs ≥40).

**Search latency, IVF_FLAT, single client, no warmup, 3 trials per nprobe** (compare to FLAT table at line 312 above):

| nprobe | min | **median** | max | vs FLAT p50 |
|---|---|---|---|---|
| 16  |  71 ms | **76 ms**  |  76 ms | **32× faster** |
| 64  | 123 ms | **190 ms** | 653 ms | **13× faster** |
| 128 | 284 ms | **307 ms** | 473 ms | **8× faster** |
| 256 | 306 ms | **393 ms** | 436 ms | 6× faster |

FLAT baseline from earlier runs: p50 ~2400 ms. **IVF_FLAT at nprobe=64 is 13× faster than FLAT** on the same machine, which is exactly the speedup the index was supposed to buy. nprobe=16 is 32× faster but worth verifying recall on real query workloads before shipping. p99 was not measured (only 3 trials per nprobe) but the visible tail at nprobe=64 (123→653 ms, 5× spread) suggests OrbStack/macOS scheduling noise dominates as it did in the FLAT runs — expect tighter tails on bare metal.

**Self-hit sanity check passed** at all nprobe values: pulled one row's vector, searched with it, top-1 was the same id with `distance=1.0000`, top-5 was *identical* across nprobe=16/64/128 — meaning the kmeans clustering placed the row's true neighborhood inside the 16 most-probed cells, and increasing nprobe didn't change which neighbors won. Strong signal that nlist=4096 is *not* undersized for our segment shape and that nprobe=16 is a viable starting point if latency matters more than tail recall.

**Resume / kill-cleanly mechanics that worked.** `import_all_buckets.py` writes a state file (`~/milvus-offers/import_state.json`) recording which buckets are complete. State save happens *after* each bucket's `col.flush()`. Kill-between-buckets is safe; kill-mid-bucket leaves the in-flight rows already inserted but with the bucket marked incomplete, so a resume would re-stream the bucket and produce duplicates (because `Collection.insert` does not upsert by PK). The pattern that worked for the mid-import release/restart:

1. Watchdog: `while ! grep -q "\[state\] saved: 13/16" /tmp/milvus-import-all.log; do sleep 0.2; done; kill -TERM <pid>`
2. The race window between the state-saved log line and the next bucket's first `col.insert()` is microseconds in Python plus ~100 ms to open the next parquet file. With 0.2s polling, the SIGTERM lands inside that window in practice (verified: zero rows of bucket 13 were inserted before the kill).
3. After kill: `col.release()` to free the loaded segments → restart the same script → it picks up at b13 from the state file.

**If running this again, what we'd change:**

1. **`col.release()` before starting the streamer.** The single biggest fix. Avoids the OOM-near-miss. One line.
2. **Batch size 100k–200k instead of 50k.** Fewer round-trips, slightly better insert throughput. We saw no signs of memory pressure from larger batches.
3. **A SIGTERM handler in the streamer** that sets a "stop after current bucket" flag. Removes the race in the watchdog approach and makes `kill -TERM` always safe.
4. **Don't bother polling `wait_for_index_building_complete`.** It's broken for this workload. Poll `state == "Finished"` directly and accept that compaction will continue in the background.
5. **For bigger collections (≫160M)**: at this rate of compaction churn, the streaming-insert path becomes increasingly painful. Bulk insert via MinIO (per the earlier section) remains the right answer for production-scale imports. This run is a useful fallback validation for when bulk insert is unavailable, not the recommended path.
