# Milvus Hosting Plan: 159M Vector Index

## Overview

Production deployment of Milvus for ~159M offer embeddings (128-dim float16) using Docker Compose on a dedicated Hetzner server.

**Data summary:**
- ~149.4M records across 15 healthy parquet buckets (bucket=15 is corrupted, needs regeneration)
- Schema: `row_number` (int64), `id` (varchar), `offer_embedding` (128-dim float16)
- Source data: ~40 GB parquet on disk

## Hardware Requirements

### Memory (critical constraint)

Milvus loads the full collection into RAM for search. Raw vector size: 159M × 128 × 4 bytes (float32) = **~76 GB**.

| Index Type       | Vectors in RAM | Total RAM Needed | Recall@10 |
|------------------|---------------|------------------|-----------|
| IVF_FLAT (nlist=4096) | ~76 GB     | **100–120 GB**   | 95–99%    |
| IVF_PQ (m=16)    | ~2.5 GB       | **16–20 GB**     | 90–95%    |
| HNSW (M=16)      | ~116 GB       | **140–160 GB**   | 97–99%    |

Add ~10–15 GB for Milvus process overhead, etcd, MinIO, OS.

### Disk

| Component                | Size         |
|--------------------------|-------------|
| Milvus segments (IVF_FLAT) | 85–100 GB |
| Milvus segments (IVF_PQ)  | ~10 GB     |
| etcd + WAL + temp        | ~10 GB      |
| Docker images            | ~5 GB       |
| Source parquet (for import) | ~40 GB    |
| **Total (IVF_FLAT)**     | **~150 GB** |
| **Total (IVF_PQ)**       | **~65 GB**  |

NVMe strongly preferred for import speed and segment loading.

### CPU

Milvus parallelizes search across cores. 8–16 cores is adequate; 32+ for high concurrent QPS.

## Hetzner Recommendations

### Recommended: AX162-R (dedicated, best value for sustained use)

- AMD Ryzen 9 7950X3D, 16 cores / 32 threads
- **128 GB DDR5 ECC RAM** — fits IVF_FLAT with ~20 GB headroom
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

- CCX53: 32 vCPU, **128 GB RAM**, 240 GB NVMe — ~€180/mo (IVF_FLAT, tight)
- EX44: Intel i5-13500, **64 GB RAM**, 2× 512 GB NVMe — ~€44/mo (IVF_PQ only)

### Summary

| Server   | Type      | RAM    | Cores | Disk      | Price   | Index Types       |
|----------|-----------|--------|-------|-----------|---------|-------------------|
| AX162-R  | Dedicated | 128 GB | 16/32 | 2× 1.9 TB | €82/mo  | IVF_FLAT, IVF_PQ  |
| CCX63    | Cloud     | 192 GB | 48    | 360 GB    | €270/mo | IVF_FLAT, HNSW    |
| CCX53    | Cloud     | 128 GB | 32    | 240 GB    | €180/mo | IVF_FLAT, IVF_PQ  |
| EX44     | Dedicated | 64 GB  | 14/20 | 2× 512 GB | €44/mo  | IVF_PQ only       |

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

```python
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
import pyarrow.parquet as pq
import numpy as np
import os

URI = "http://localhost:19530"
COLLECTION = "offers"
BATCH_SIZE = 100_000
DATA_DIR = "/data/offers_embedded.parquet"

client = MilvusClient(URI)

# Schema
fields = [
    FieldSchema(name="row_number", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="offer_embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
]
schema = CollectionSchema(fields)

# Index — IVF_FLAT for best recall
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="offer_embedding",
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={"nlist": 4096},
)

client.create_collection(
    collection_name=COLLECTION,
    schema=schema,
    index_params=index_params,
)

# Import all buckets
for bucket_file in sorted(os.listdir(DATA_DIR)):
    if not bucket_file.endswith(".parquet"):
        continue
    path = os.path.join(DATA_DIR, bucket_file)
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        print(f"Skipping {bucket_file}: {e}")
        continue

    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        row_numbers = batch.column("row_number").to_pylist()
        ids = batch.column("id").to_pylist()
        emb = np.vstack(batch.column("offer_embedding").to_numpy()).astype(np.float32)

        data = [
            {
                "row_number": row_numbers[i],
                "id": ids[i],
                "offer_embedding": emb[i].tolist(),
            }
            for i in range(len(row_numbers))
        ]
        client.insert(collection_name=COLLECTION, data=data)
    print(f"Finished {bucket_file}")

# Load collection for search
client.load_collection(COLLECTION)
print("Collection loaded and ready for search")
```

Expected import time: ~30–60 minutes for 149M records.

### Step 5: Verify

```python
stats = client.get_collection_stats("offers")
print(stats)  # Should show ~149M entities

# Test search
import numpy as np
query = np.random.randn(1, 128).astype(np.float32).tolist()
results = client.search(
    collection_name="offers",
    data=query,
    limit=10,
    search_params={"nprobe": 64},
)
print(results)
```

### Step 6: Expose for production

Milvus listens on port 19530 (gRPC) and 9091 (metrics). For production:

- Restrict port 19530 to application servers via firewall / Hetzner network
- Monitor via Prometheus scraping :9091/metrics
- Key metrics: `milvus_proxy_search_latency_bucket`, `milvus_querynode_loaded_segment_total`

## Index Type Decision Guide

| Priority         | Choose       | Why                                     |
|------------------|--------------|-----------------------------------------|
| Best recall      | IVF_FLAT     | 95–99% recall, straightforward          |
| Lowest latency   | HNSW         | ~1–5ms search, needs 160+ GB RAM        |
| Tight budget     | IVF_PQ       | Fits on 64 GB RAM, 90–95% recall        |
| GPU acceleration | GPU_IVF_FLAT | Sub-ms search, needs GPU instance        |
