"""Bulk import the playground offers dataset into a new Milvus collection.

Source: /data/datasets/offers_playground_elastic_with_categories.parquet/data_*.parquet
Target collection: `offers_playground` (18.3M rows, 128d fp16 embeddings).

Pipeline per source file:
  1. Read data_N.parquet (playground_id + fp32 embedding + scalar fields).
  2. Convert fp32 list embedding → fp16 → raw bytes as list<uint8> (stride=256).
  3. Pass remaining scalar fields through, preserving nulls on nullable columns.
  4. Upload staging parquet to MinIO, then submit one `do_bulk_insert` per file.

Schema (10 fields) — offer_embedding is the only RAM-resident field; everything
else (field data *and* inverted indexes) is mmap-backed:

  playground_id              VARCHAR(96) PK          (RAM, no mmap)
  offer_embedding            FLOAT16_VECTOR(128)     HNSW M=16 efC=360 COSINE (RAM)
  playground_vendorId        VARCHAR(64)             INVERTED (field+index mmap)
  playground_articleId       VARCHAR(96)             INVERTED (field+index mmap)
  name                       VARCHAR(256)            no index (field mmap)
  manufacturerName           VARCHAR(128)            no index (field mmap)
  manufacturerArticleNumber  VARCHAR(128) nullable   no index (field mmap)
  manufacturerArticleType    VARCHAR(512) nullable   no index (field mmap)
  ean                        VARCHAR(32)             no index (field mmap)
  article_number             VARCHAR(64)             no index (field mmap)

The two nullable fields carry NULL on the 845,927 "sparse" rows (4.6% of input)
which have no enrichment data. All rows are imported — vector search covers the
full corpus; scalar filters naturally miss sparse rows.

See APRIL_21_offers_bulk_import.md for the operational lessons this script inherits (index-before-
bulk-insert, pipelined stage+submit, inline `IndexBuilding` in the job).
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

DEFAULT_DATA_DIR = Path(
    "/data/datasets/offers_playground_elastic_with_categories.parquet"
)
STAGE_DIR = Path("/tmp/milvus_bulk_stage_playground")
COLLECTION = "offers_playground"
DIM = 128
WRITE_BATCH = 500_000
NUM_WORKERS = 9
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 360

S3_ENDPOINT = "http://localhost:9000"
S3_BUCKET = "a-bucket"
S3_PREFIX = "bulk_offers_playground"
S3_KEY = "minioadmin"
S3_SECRET = "minioadmin"

# Scalar fields with an INVERTED index. Per-user decision: mmap both the field
# data AND the index storage for these — this dataset is small and we want to
# keep RAM headroom free for the HNSW vector index.
INDEXED_SCALAR_FIELDS = ["playground_vendorId", "playground_articleId"]

# (field_name, max_length_bytes, nullable). All field data mmap'd. The first
# two (playground_vendorId, playground_articleId) additionally get an INVERTED
# index — the rest are stored unindexed. Max lengths sized from a full-
# dataset scan with generous margin.
TEXT_FIELDS: list[tuple[str, int, bool]] = [
    ("playground_vendorId", 64, False),
    ("playground_articleId", 96, False),
    ("name", 256, False),
    ("manufacturerName", 128, False),
    ("manufacturerArticleNumber", 128, True),
    ("manufacturerArticleType", 512, True),
    ("ean", 32, False),
    ("article_number", 64, False),
]
TEXT_FIELD_NAMES = [name for name, _, _ in TEXT_FIELDS]

# Columns to pull from source parquet. Everything else in the source file is
# ignored (description, playground_keywords, categoryPaths, vendor_listings,
# n, id — all dropped per schema decision).
SOURCE_COLS = ["playground_id", "offer_embedding", *TEXT_FIELD_NAMES]

# Target staged-parquet column order (matches schema field order).
STAGED_COLS = ["playground_id", "offer_embedding", *TEXT_FIELD_NAMES]

SOURCE_RE = re.compile(r"^data_\d+\.parquet$")


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
            ("playground_id", pa.string()),
            ("offer_embedding", pa.list_(pa.uint8())),
            *((name, pa.string()) for name in TEXT_FIELD_NAMES),
        ]
    )


def convert_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    # Source embeddings are fp32 list<float>. Milvus bulk insert expects the
    # fp16 vector as raw bytes (list<uint8>, stride = DIM*2). Convert once.
    emb_obj = batch.column("offer_embedding").to_numpy(zero_copy_only=False)
    emb_f32 = np.stack(emb_obj).astype(np.float32, copy=False)
    emb_f16 = emb_f32.astype(np.float16)
    emb_u8 = emb_f16.view(np.uint8)
    n, width = emb_u8.shape

    flat = pa.array(emb_u8.reshape(-1), type=pa.uint8())
    offsets = pa.array(np.arange(0, n * width + 1, width, dtype=np.int32))
    new_emb = pa.ListArray.from_arrays(offsets, flat)

    arrays = [batch.column("playground_id"), new_emb]
    for name in TEXT_FIELD_NAMES:
        arrays.append(batch.column(name))
    return pa.RecordBatch.from_arrays(arrays, names=STAGED_COLS)


def stage_one(src: Path) -> tuple[str, float, float, int]:
    tmp = STAGE_DIR / src.name
    s3 = s3_client()

    t0 = time.time()
    pf = pq.ParquetFile(src)
    schema = target_schema()
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=1) as writer:
        for batch in pf.iter_batches(batch_size=WRITE_BATCH, columns=SOURCE_COLS):
            writer.write_batch(convert_batch(batch))
    convert_t = time.time() - t0

    t1 = time.time()
    s3.upload_file(str(tmp), S3_BUCKET, f"{S3_PREFIX}/{src.name}")
    upload_t = time.time() - t1

    size = tmp.stat().st_size
    tmp.unlink()
    print(
        f"  {src.name}: convert={convert_t:5.1f}s upload={upload_t:5.1f}s "
        f"size={size/1e9:.2f} GB",
        flush=True,
    )
    return src.name, convert_t, upload_t, size


def stage_and_submit(sources: list[Path], workers: int) -> dict[str, int]:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Staging + submitting {len(sources)} file(s) through {STAGE_DIR} "
          f"(workers={workers}) — pipelined", flush=True)
    jobs: dict[str, int] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(stage_one, s): s for s in sources}
        for fut in as_completed(futs):
            name, _c, _u, _s = fut.result()
            path = f"{S3_PREFIX}/{name}"
            job_id = utility.do_bulk_insert(collection_name=COLLECTION, files=[path])
            jobs[name] = job_id
            print(f"  [{int(time.time()-t0):4d}s] {name}: staged+submitted -> job {job_id}",
                  flush=True)
    print(f"All staging + submission done in {time.time() - t0:.1f}s", flush=True)
    shutil.rmtree(STAGE_DIR, ignore_errors=True)
    return jobs


def build_collection(drop_existing: bool) -> Collection:
    if utility.has_collection(COLLECTION):
        if drop_existing:
            try:
                Collection(COLLECTION).release()
            except Exception:
                pass
            print(f"Dropping existing collection {COLLECTION!r}")
            utility.drop_collection(COLLECTION)
        else:
            print(f"Reusing existing collection {COLLECTION!r}")
            return Collection(COLLECTION)

    # Field-level mmap only for the non-PK, non-vector fields. PK and vector
    # stay resident.
    fields = [
        FieldSchema(name="playground_id", dtype=DataType.VARCHAR, max_length=96,
                    is_primary=True),
        FieldSchema(name="offer_embedding", dtype=DataType.FLOAT16_VECTOR, dim=DIM),
    ]
    for name, max_len, nullable in TEXT_FIELDS:
        fields.append(FieldSchema(
            name=name, dtype=DataType.VARCHAR, max_length=max_len,
            nullable=nullable, mmap_enabled=True,
        ))

    schema = CollectionSchema(
        fields,
        description="Playground offers — fp16 128d embeddings + minimal metadata",
    )
    col = Collection(COLLECTION, schema=schema)
    print(f"Created collection {COLLECTION!r} with {len(fields)} fields")
    for f in fields:
        mmap_flag = f.params.get("mmap_enabled", False)
        nullable = getattr(f, "nullable", False)
        print(f"  - {f.name:<28} {f.dtype.name:<18}  mmap={mmap_flag}  nullable={nullable}")
    return col


def wait_for_jobs(jobs: dict[str, int]) -> int:
    pending = dict(jobs)
    total_rows = 0
    t0 = time.time()
    while pending:
        next_pending = {}
        for name, job_id in pending.items():
            state = utility.get_bulk_insert_state(job_id)
            if state.state_name == "Completed":
                total_rows += state.row_count
                print(
                    f"  [{int(time.time()-t0):4d}s] {name}: Completed "
                    f"({state.row_count:,} rows)"
                )
            elif state.state_name == "Failed":
                raise RuntimeError(f"{name}: bulk insert FAILED -- {state.infos}")
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
    return total_rows


def wait_index_finished(col: Collection, field: str, poll_s: int = 5) -> float:
    """Guard: state=Finished with indexed_rows=0 on a non-empty collection is
    the 'no build task registered' response, NOT success. Only accept
    Finished when indexed_rows == total_rows. See APRIL_21_offers_bulk_import.md for history."""
    t0 = time.time()
    while True:
        progress = utility.index_building_progress(col.name, index_name=field)
        total = progress.get("total_rows", 0)
        indexed = progress.get("indexed_rows", 0)
        pending = progress.get("pending_index_rows", 0)
        state = progress.get("state", "?")
        elapsed = time.time() - t0
        print(
            f"  [{field}] t={elapsed:6.1f}s  state={state}  "
            f"indexed={indexed:,}/{total:,}  pending={pending:,}",
            flush=True,
        )
        if state == "Finished" and (total == 0 or indexed == total):
            return elapsed
        time.sleep(poll_s)


def select_sources(data_dir: Path, filter_name: str | None) -> list[Path]:
    files = [p for p in sorted(data_dir.iterdir()) if SOURCE_RE.match(p.name)]
    if not files:
        raise SystemExit(f"No data_N.parquet files under {data_dir}")
    if filter_name:
        files = [p for p in files if p.name == filter_name]
        if not files:
            raise SystemExit(f"No file matching {filter_name!r} under {data_dir}")
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument(
        "--file",
        default="",
        help="Single source filename (e.g., 'data_0.parquet'). Default: all.",
    )
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", default="19530")
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--no-drop", action="store_true",
                   help=f"Do not drop existing {COLLECTION!r} collection.")
    p.add_argument("--skip-load", action="store_true")
    args = p.parse_args()

    sources = select_sources(args.data_dir, args.file or None)
    print(f"Found {len(sources)} file(s) to import:")
    for s in sources:
        print(f"  - {s}")

    wall = time.time()

    connections.connect(alias="default", host=args.host, port=args.port)
    col = build_collection(drop_existing=not args.no_drop)

    # Indexes MUST be defined before the first bulk-insert job is submitted.
    # Milvus 2.6's bulk-insert pipeline builds them inline; post-hoc
    # create_index is a no-op (state=Finished, indexed_rows=0). APRIL_21_offers_bulk_import.md.
    print(f"\nDefining HNSW index on offer_embedding "
          f"(M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION})...", flush=True)
    col.create_index(
        field_name="offer_embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION},
        },
    )
    for field in INDEXED_SCALAR_FIELDS:
        print(f"Defining INVERTED (mmap) index on {field}...", flush=True)
        col.create_index(
            field_name=field,
            # `mmap.enabled` is the server-side param name for scalar-index
            # mmap. This keeps the INVERTED posting lists on disk.
            index_params={
                "index_type": "INVERTED",
                "params": {"mmap.enabled": "true"},
            },
            index_name=field,
        )

    t0 = time.time()
    jobs = stage_and_submit(sources, args.workers)
    stage_submit_s = time.time() - t0

    print("\nWaiting for bulk-insert jobs to complete (includes IndexBuilding)...",
          flush=True)
    t0 = time.time()
    total_rows = wait_for_jobs(jobs)
    ingest_s = time.time() - t0

    print(f"\nFlushing... (stage+submit: {stage_submit_s:.1f}s, "
          f"ingest wait: {ingest_s:.1f}s, rows: {total_rows:,})", flush=True)
    t0 = time.time()
    col.flush()
    flush_s = time.time() - t0
    print(f"  flush: {flush_s:.1f}s  num_entities={col.num_entities:,}", flush=True)

    print("\nVerifying index build state...", flush=True)
    scalar_timings: dict[str, float] = {}
    vec_s = wait_index_finished(col, "offer_embedding")
    for field in INDEXED_SCALAR_FIELDS:
        scalar_timings[field] = wait_index_finished(col, field)
    vec_progress = utility.index_building_progress(col.name, index_name="offer_embedding")
    assert vec_progress["indexed_rows"] == total_rows, (
        f"HNSW indexed_rows={vec_progress['indexed_rows']:,} != "
        f"total_rows={total_rows:,} — build was a no-op")
    print(f"  offer_embedding: indexed {vec_progress['indexed_rows']:,}/"
          f"{vec_progress['total_rows']:,} ✓", flush=True)

    load_s = 0.0
    if args.skip_load:
        print("\nSkipping col.load() as requested.")
    else:
        print("\nLoading collection...")
        t0 = time.time()
        col.load()
        load_s = time.time() - t0

    total_wall = time.time() - wall
    print("\n=== summary ===")
    print(f"  rows:              {total_rows:,}")
    print(f"  stage+submit:      {stage_submit_s:7.1f}s  (pipelined)")
    print(f"  ingest wait:       {ingest_s:7.1f}s")
    print(f"  flush:             {flush_s:7.1f}s")
    print(f"  vector index:      {vec_s:7.1f}s  "
          f"(HNSW M={HNSW_M} efConstruction={HNSW_EF_CONSTRUCTION})")
    for f, s in scalar_timings.items():
        print(f"  idx {f:<25}{s:7.1f}s")
    print(f"  load:              {load_s:7.1f}s")
    print(f"  total wall:        {total_wall:7.1f}s  ({total_wall/60:.1f} min)")
    print(f"  num_entities:      {col.num_entities:,}")


if __name__ == "__main__":
    main()
