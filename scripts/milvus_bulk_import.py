"""Bulk import pre-flattened offer buckets into Milvus via MinIO.

Pipeline per bucket:
  1. Read `offers_flat.parquet/bucket=NN.parquet` (id + fp16 emb + 7 ARRAY<VARCHAR>).
  2. Rewrite `offer_embedding` (list<halffloat>) as list<uint8> (raw fp16 bytes);
     pass the 7 ARRAY<VARCHAR> filter columns through unchanged.
  3. Upload staging parquet to MinIO at `a-bucket/bulk_offers/bucket=NN.parquet`.
  4. Submit one `do_bulk_insert` per staged file and poll until Completed.

Index creation happens RIGHT AFTER `build_collection`, BEFORE any bulk-insert
is submitted. Milvus 2.6's bulk-insert pipeline auto-builds indexes as part
of its `IndexBuilding` stage — but only for indexes defined at submit time.
Creating indexes post-flush is a no-op: `state=Finished indexed_rows=0`
with no real build task ever scheduled, and `col.load()` then falls back to
an in-memory `IVF_FLAT_CC` interim index (fp32, ~2× RAM of HNSW fp16).

HNSW on `offer_embedding`, INVERTED on `vendor_ids` and `catalog_version_ids`
only. The 5 `category_l*` arrays and all text/int fields are unindexed.

Field mmap: everything except `id`, `offer_embedding`, `vendor_ids`,
`catalog_version_ids` is declared with `mmap_enabled=True`, so after
`col.load()` only those four live in RAM; all other fields are disk-backed
but still returnable via `output_fields`.

Why list<uint8> for FLOAT16_VECTOR: Milvus bulk insert expects the fp16 vector
as raw bytes. The source `offers_flat.parquet` keeps embeddings as list<halffloat>
via pyarrow passthrough; we view() the numpy buffer as uint8 and rebuild the
list array with a fixed offset stride of 256 (128 × 2 bytes).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
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
    "/mnt/HC_Volume_105463954/simplesystem/data/offers_flat.parquet"
)
STAGE_DIR = Path("/tmp/milvus_bulk_stage")
COLLECTION = "offers"
DIM = 128
WRITE_BATCH = 500_000
NUM_WORKERS = 16
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 360

S3_ENDPOINT = "http://localhost:9000"
S3_BUCKET = "a-bucket"
S3_PREFIX = "bulk_offers"
S3_KEY = "minioadmin"
S3_SECRET = "minioadmin"

CAT_FIELDS = [f"category_l{lvl}" for lvl in (1, 2, 3, 4, 5)]
# All ARRAY<VARCHAR> fields present in the schema.
SCALAR_ARRAY_FIELDS = ["vendor_ids", "catalog_version_ids", *CAT_FIELDS]
# Only these get an INVERTED scalar index; the rest are stored unindexed.
INDEXED_SCALAR_FIELDS = ["vendor_ids", "catalog_version_ids"]
# Fields that must stay resident in RAM. Every other field is declared with
# mmap_enabled=True so col.load() disk-backs it.
RAM_RESIDENT_FIELDS = {"id", "offer_embedding", *INDEXED_SCALAR_FIELDS}

# Unindexed descriptive fields — stored + returnable via output_fields, not
# searched. Sized from a bucket=00 scan (max chars): name 130, mfgName 50,
# description 117k (→ cap at Milvus VARCHAR limit), ean 14, article 32,
# mfgArticleNumber 50, mfgArticleType 253.
TEXT_FIELDS = [
    ("name", 256),
    ("manufacturerName", 128),
    ("description", 65_535),
    ("ean", 32),
    ("article_number", 64),
    ("manufacturerArticleNumber", 128),
    ("manufacturerArticleType", 512),
]
TEXT_FIELD_NAMES = [name for name, _ in TEXT_FIELDS]
INT_FIELDS = ["n"]
DESCRIPTION_MAX_BYTES = 65_535

ALL_COLS = [
    "id", "offer_embedding",
    *SCALAR_ARRAY_FIELDS,
    *TEXT_FIELD_NAMES,
    *INT_FIELDS,
]

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
            ("id", pa.string()),
            ("offer_embedding", pa.list_(pa.uint8())),
            ("vendor_ids", pa.list_(pa.string())),
            ("catalog_version_ids", pa.list_(pa.string())),
            ("category_l1", pa.list_(pa.string())),
            ("category_l2", pa.list_(pa.string())),
            ("category_l3", pa.list_(pa.string())),
            ("category_l4", pa.list_(pa.string())),
            ("category_l5", pa.list_(pa.string())),
            *((name, pa.string()) for name in TEXT_FIELD_NAMES),
            ("n", pa.int64()),
        ]
    )


def convert_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    emb_obj = batch.column("offer_embedding").to_numpy(zero_copy_only=False)
    emb_2d = np.stack(emb_obj)           # (n, 128) fp16
    emb_u8 = emb_2d.view(np.uint8)       # (n, 256) uint8
    n, width = emb_u8.shape

    flat = pa.array(emb_u8.reshape(-1), type=pa.uint8())
    offsets = pa.array(np.arange(0, n * width + 1, width, dtype=np.int32))
    new_emb = pa.ListArray.from_arrays(offsets, flat)

    arrays = [batch.column("id"), new_emb]
    for f in SCALAR_ARRAY_FIELDS:
        arrays.append(batch.column(f))
    for f in TEXT_FIELD_NAMES:
        col = batch.column(f)
        if f == "description":
            # Milvus VARCHAR max_length is BYTES, not codepoints. Only
            # truncate rows whose UTF-8 byte length exceeds the cap;
            # truncate those to DESCRIPTION_MAX_BYTES // 4 codepoints,
            # which is a safe upper bound (UTF-8 ≤ 4 bytes/codepoint).
            byte_len = pc.binary_length(pc.cast(col, pa.binary()))
            over = pc.greater(byte_len, DESCRIPTION_MAX_BYTES)
            truncated = pc.utf8_slice_codeunits(col, 0, DESCRIPTION_MAX_BYTES // 4)
            col = pc.if_else(over, truncated, col)
        arrays.append(col)
    arrays.append(batch.column("n"))
    return pa.RecordBatch.from_arrays(arrays, names=ALL_COLS)


def stage_one(src: Path) -> tuple[str, float, float, int]:
    tmp = STAGE_DIR / src.name
    s3 = s3_client()

    t0 = time.time()
    pf = pq.ParquetFile(src)
    schema = target_schema()
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=1) as writer:
        for batch in pf.iter_batches(batch_size=WRITE_BATCH, columns=ALL_COLS):
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
    """Stage each bucket (convert → upload to MinIO) and submit its Milvus
    bulk-insert job the instant the upload completes. Overlaps the tail of
    staging with the head of ingestion — bucket 0's segments can be flushing
    while bucket 15 is still being converted.
    """
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Staging + submitting {len(sources)} bucket(s) through {STAGE_DIR} "
          f"(workers={workers}) — pipelined", flush=True)
    jobs: dict[str, int] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(stage_one, s): s for s in sources}
        for fut in as_completed(futs):
            name, _convert_t, _upload_t, _size = fut.result()
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

    def mm(field_name: str) -> dict:
        return {} if field_name in RAM_RESIDENT_FIELDS else {"mmap_enabled": True}

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True,
                    **mm("id")),
        FieldSchema(name="offer_embedding", dtype=DataType.FLOAT16_VECTOR, dim=DIM,
                    **mm("offer_embedding")),
        FieldSchema(name="vendor_ids", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=32, max_length=64,
                    **mm("vendor_ids")),
        FieldSchema(name="catalog_version_ids", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=2048, max_length=64,
                    **mm("catalog_version_ids")),
        FieldSchema(name="category_l1", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=64, max_length=256,
                    **mm("category_l1")),
        FieldSchema(name="category_l2", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=64, max_length=640,
                    **mm("category_l2")),
        FieldSchema(name="category_l3", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=64, max_length=768,
                    **mm("category_l3")),
        FieldSchema(name="category_l4", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=64, max_length=1024,
                    **mm("category_l4")),
        FieldSchema(name="category_l5", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_capacity=64, max_length=1280,
                    **mm("category_l5")),
    ]
    for name, max_len in TEXT_FIELDS:
        fields.append(FieldSchema(name=name, dtype=DataType.VARCHAR,
                                  max_length=max_len, **mm(name)))
    fields.append(FieldSchema(name="n", dtype=DataType.INT64, **mm("n")))
    schema = CollectionSchema(fields, description="Offer embeddings (fp16, 128d) + filter fields + text metadata")
    col = Collection(COLLECTION, schema=schema)
    print(f"Created collection {COLLECTION!r} with {len(fields)} fields")
    for f in fields:
        mmap_flag = f.params.get("mmap_enabled", False)
        print(f"  - {f.name:<28} {f.dtype.name:<18}  mmap={mmap_flag}")
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
    """Poll until the index build for `field` is truly finished.

    Guards against the pymilvus race where `state=Finished` is returned with
    `indexed_rows=0` on a non-empty collection — which means the server has
    no pending build task registered yet, NOT that the build is complete.
    We accept "Finished" only if indexed_rows == total_rows (or total is 0).
    """
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


def select_buckets(data_dir: Path, filter_name: str | None) -> list[Path]:
    files = [p for p in sorted(data_dir.iterdir()) if BUCKET_RE.match(p.name)]
    if not files:
        raise SystemExit(f"No bucket=NN.parquet files under {data_dir}")
    if filter_name:
        files = [p for p in files if p.name == filter_name]
        if not files:
            raise SystemExit(f"No bucket matching {filter_name!r} under {data_dir}")
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument(
        "--bucket",
        default="",
        help="Single bucket filename (e.g., 'bucket=00.parquet'). Default: all.",
    )
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", default="19530")
    p.add_argument("--workers", type=int, default=NUM_WORKERS,
                   help=f"Parallel stage workers (default {NUM_WORKERS})")
    p.add_argument("--no-drop", action="store_true",
                   help="Do not drop existing 'offers' collection.")
    p.add_argument(
        "--skip-load", action="store_true",
        help="Skip col.load() at the end (useful to manually control "
             "which fields get loaded into RAM afterward).",
    )
    p.add_argument(
        "--load-fields", default="",
        help="Comma-separated field names to pass to col.load(load_fields=...). "
             "Default: load everything. Use this to exclude bulky text "
             "fields like 'description' from RAM.",
    )
    args = p.parse_args()

    sources = select_buckets(args.data_dir, args.bucket or None)
    print(f"Found {len(sources)} bucket(s) to import:")
    for s in sources:
        print(f"  - {s}")

    wall = time.time()

    # Connect + build collection FIRST so bulk-insert jobs can fire the moment
    # each staged file hits MinIO.
    connections.connect(alias="default", host=args.host, port=args.port)
    col = build_collection(drop_existing=not args.no_drop)

    # Indexes MUST be defined before the first bulk-insert job is submitted.
    # Milvus's bulk-insert pipeline builds them inline during its IndexBuilding
    # stage; post-hoc create_index on sealed-and-completed segments is a no-op.
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
        print(f"Defining INVERTED index on {field}...", flush=True)
        col.create_index(
            field_name=field,
            index_params={"index_type": "INVERTED"},
            index_name=field,
        )

    t0 = time.time()
    jobs = stage_and_submit(sources, args.workers)
    stage_submit_s = time.time() - t0

    # wait_for_jobs returns once every bulk-insert reaches Completed, which
    # in Milvus 2.6 includes PreImport → Import → Sort → IndexBuilding →
    # Completed. So by the time this returns, HNSW + both INVERTEDs are
    # already built for every imported segment.
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

    # Verify indexes really built (catches silent no-op).
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
        if args.load_fields:
            fields = [f.strip() for f in args.load_fields.split(",") if f.strip()]
            print(f"  load_fields={fields}")
            col.load(load_fields=fields)
        else:
            col.load()
        load_s = time.time() - t0

    total_wall = time.time() - wall
    print("\n=== summary ===")
    print(f"  rows:              {total_rows:,}")
    print(f"  stage+submit:      {stage_submit_s:7.1f}s  (pipelined)")
    print(f"  ingest wait:       {ingest_s:7.1f}s  ({total_rows/max(ingest_s,0.1):,.0f} rows/s post-submit)")
    print(f"  flush:             {flush_s:7.1f}s")
    print(f"  vector index:      {vec_s:7.1f}s  "
          f"(HNSW M={HNSW_M} efConstruction={HNSW_EF_CONSTRUCTION})")
    for f, s in scalar_timings.items():
        print(f"  idx {f:<22}{s:7.1f}s")
    print(f"  load:              {load_s:7.1f}s")
    print(f"  total wall:        {total_wall:7.1f}s  ({total_wall/60:.1f} min)")
    print(f"  num_entities:      {col.num_entities:,}")


if __name__ == "__main__":
    main()
