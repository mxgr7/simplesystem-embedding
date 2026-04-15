"""Bulk import all 16 offer embedding buckets into Milvus via MinIO.

Pipeline per bucket (run in parallel across buckets):
  1. Read source parquet (list<halffloat>).
  2. Rewrite as list<uint8> (raw fp16 bytes) into /tmp staging file.
  3. Upload staging file to milvus MinIO under bulk_offers/.
  4. Delete local staging file.

Then submit one do_bulk_insert job per uploaded file and poll until all
jobs reach Completed/Failed. Finally flush + create FLAT index + load.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.client import Config
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

DATA_DIR = Path("/Users/max/Clients/simplesystem/data/offers_embedded.parquet")
STAGE_DIR = Path("/tmp/milvus_bulk_stage")
COLLECTION = "offers"
DIM = 128
WRITE_BATCH = 500_000
NUM_WORKERS = 4

S3_ENDPOINT = "http://localhost:9010"
S3_BUCKET = "a-bucket"
S3_PREFIX = "bulk_offers"
S3_KEY = "minioadmin"
S3_SECRET = "minioadmin"

BUCKET_RE = re.compile(r"^bucket=\d{2}\.parquet$")


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def target_schema() -> pa.Schema:
    return pa.schema(
        [
            ("row_number", pa.int64()),
            ("id", pa.string()),
            ("offer_embedding", pa.list_(pa.uint8())),
        ]
    )


def convert_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    emb_obj = batch.column("offer_embedding").to_numpy(zero_copy_only=False)
    emb_2d = np.stack(emb_obj)  # (n, 128) fp16
    emb_u8 = emb_2d.view(np.uint8)  # (n, 256) uint8
    n, width = emb_u8.shape

    flat = pa.array(emb_u8.reshape(-1), type=pa.uint8())
    offsets = pa.array(np.arange(0, n * width + 1, width, dtype=np.int32))
    new_emb = pa.ListArray.from_arrays(offsets, flat)

    return pa.RecordBatch.from_arrays(
        [batch.column("row_number"), batch.column("id"), new_emb],
        names=["row_number", "id", "offer_embedding"],
    )


def stage_one(name: str) -> tuple[str, float, float]:
    src = DATA_DIR / name
    tmp = STAGE_DIR / name
    s3 = s3_client()

    t0 = time.time()
    pf = pq.ParquetFile(src)
    schema = target_schema()
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for batch in pf.iter_batches(batch_size=WRITE_BATCH):
            writer.write_batch(convert_batch(batch))
    convert_t = time.time() - t0

    t1 = time.time()
    s3.upload_file(str(tmp), S3_BUCKET, f"{S3_PREFIX}/{name}")
    upload_t = time.time() - t1

    size = tmp.stat().st_size
    tmp.unlink()
    print(
        f"  {name}: convert={convert_t:5.1f}s upload={upload_t:5.1f}s "
        f"size={size/1e9:.2f} GB",
        flush=True,
    )
    return name, convert_t, upload_t


def stage_all(names: list[str]) -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Staging {len(names)} buckets through {STAGE_DIR} (workers={NUM_WORKERS})")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for fut in as_completed(ex.submit(stage_one, n) for n in names):
            fut.result()
    print(f"All staging done in {time.time() - t0:.1f}s")
    shutil.rmtree(STAGE_DIR, ignore_errors=True)


def build_collection() -> Collection:
    if utility.has_collection(COLLECTION):
        print(f"Dropping existing collection {COLLECTION!r}")
        utility.drop_collection(COLLECTION)
    schema = CollectionSchema(
        [
            FieldSchema(name="row_number", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="offer_embedding", dtype=DataType.FLOAT16_VECTOR, dim=DIM),
        ]
    )
    col = Collection(COLLECTION, schema=schema)
    print(f"Created collection {COLLECTION!r}")
    return col


def submit_jobs(names: list[str]) -> dict[str, int]:
    jobs: dict[str, int] = {}
    for name in names:
        path = f"{S3_PREFIX}/{name}"
        job_id = utility.do_bulk_insert(collection_name=COLLECTION, files=[path])
        jobs[name] = job_id
        print(f"  submitted {name} -> job {job_id}")
    return jobs


def wait_for_jobs(jobs: dict[str, int]) -> None:
    pending = dict(jobs)
    t0 = time.time()
    while pending:
        next_pending = {}
        for name, job_id in pending.items():
            state = utility.get_bulk_insert_state(job_id)
            if state.state_name == "Completed":
                print(
                    f"  [{int(time.time()-t0):4d}s] {name}: Completed "
                    f"({state.row_count:,} rows)"
                )
            elif state.state_name == "Failed":
                print(f"  [{int(time.time()-t0):4d}s] {name}: FAILED -- {state.infos}")
            else:
                next_pending[name] = job_id
        if next_pending:
            still = ", ".join(
                f"{n}={utility.get_bulk_insert_state(j).progress}%"
                for n, j in next_pending.items()
            )
            print(f"  [{int(time.time()-t0):4d}s] pending: {still}", flush=True)
            time.sleep(10)
        pending = next_pending


def main() -> None:
    bucket_names = sorted(
        p.name for p in DATA_DIR.iterdir() if BUCKET_RE.match(p.name)
    )
    if not bucket_names:
        sys.exit(f"no bucket=NN.parquet under {DATA_DIR}")
    print(f"Found {len(bucket_names)} source buckets")

    connections.connect(host="localhost", port="19530")
    col = build_collection()

    stage_all(bucket_names)

    print("\nSubmitting bulk insert jobs...")
    jobs = submit_jobs(bucket_names)

    print("\nWaiting for jobs to complete...")
    wait_for_jobs(jobs)

    print("\nFlushing...")
    col.flush()

    print("Creating FLAT index...")
    col.create_index(
        field_name="offer_embedding",
        index_params={"index_type": "FLAT", "metric_type": "COSINE"},
    )
    print("Loading collection...")
    col.load()

    print(f"\nDone. num_entities = {col.num_entities:,}")


if __name__ == "__main__":
    main()
