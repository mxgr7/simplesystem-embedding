"""F9 bulk-insert path — parquet → MinIO/S3 → `do_bulk_insert`.

Sibling to the upsert path in `indexer/bulk.py`. Same DuckDB pipeline
upstream; the difference is the Milvus sink:

  - upsert path:   `MilvusClient.upsert(data=batch)` per batch.
                   Slow (~800 rows/sec) but per-row idempotent and
                   queryable immediately.
  - bulk-insert:   stage chunked parquets to MinIO, submit one
                   `do_bulk_insert` per chunk (parallel via thread
                   pool). Throughput ~50–100K rows/sec aggregate; rows
                   visible only after the server finishes its
                   `PreImport → Import → Sort → IndexBuilding` pipeline.

At F9 production scale (159M articles + 510M offers) the upsert path
is ~10 days of Milvus-side work; bulk-insert collapses that to ~2h.
The two paths share the same row-emission code — we wrap the
DuckDB-fed `_iter_relation_dicts` stream into parquet writers per
collection rather than into per-batch Milvus calls.

Pipelining: `stream_chunks_to_milvus` produces chunk parquets one at
a time and hands each to a `ThreadPoolExecutor` for upload + submit.
Chunk N+1 stages while chunk N is still uploading or being ingested
server-side, so the wall time is roughly `max(stage, upload+submit)`
rather than their sum. Mirrors the pipelining trick from
`scripts/milvus_bulk_import.py:stage_and_submit`.

Encoding rules (pinned by Milvus 2.6 bulk-insert format):
  - `FLOAT16_VECTOR` columns → `LIST<UINT8>` of raw fp16 bytes
    (2 × dim bytes per row). Same trick as
    `scripts/milvus_bulk_import.py`.
  - `FLOAT_VECTOR` columns → `LIST<FLOAT>` (e.g. the 2-d
    `_placeholder_vector` on offers).
  - `JSON` columns (`prices`, `customer_article_numbers`) → STRING
    holding the JSON-serialised value.
  - `ARRAY<INT32>` columns (eclass codes) → `LIST<INT32>`.
  - `ARRAY<VARCHAR>` columns → `LIST<STRING>`.
  - `BM25` `sparse_codes` is *server-computed* from `text_codes` via
    the schema's BM25 function — it is NOT included in the parquet.

Indexes MUST be defined on the collection BEFORE the first
`do_bulk_insert` is submitted. Milvus's bulk-insert pipeline builds
indexes inline during its `IndexBuilding` stage; post-flush
`create_index` on already-sealed bulk-insert segments is a silent
no-op (`state=Finished indexed_rows=0`). The
`scripts/create_*_collection.py` builders register all indexes at
collection-create time, so this is satisfied as long as those
scripts ran first — which the orchestrator's existence check
enforces.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.client import Config
from pymilvus import connections, utility

log = logging.getLogger(__name__)

# Mirror of `indexer.duckdb_projection.CATALOG_CURRENCIES`.
CATALOG_CURRENCIES = ("eur", "chf", "huf", "pln", "gbp", "czk", "cny")
DIM = 128  # `articles_v{N}.offer_embedding` dim, per create_articles_collection.py


@dataclass
class BulkInsertConfig:
    """All knobs for the bulk-insert sink. Defaults match the local
    docker-compose MinIO at `localhost:9000` (see `playground-app/compose.yaml`).

    Chunking + parallelism: parquet output is split at `chunk_rows`
    per file. `upload_workers` threads upload + submit chunks
    concurrently, so chunk N+1's upload can overlap with chunk N's
    server-side `do_bulk_insert` pipeline. At production scale the
    serial path is upload-bound (single-stream MinIO at ~200 MB/s);
    `upload_workers=4` saturates a 1 GbE link without exhausting MinIO.
    Tune up if MinIO is on a 10 GbE link."""
    s3_endpoint: str = "http://localhost:9000"
    s3_bucket: str = "a-bucket"
    s3_prefix: str = "f9_indexer"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_region: str = "us-east-1"
    stage_dir: Path = Path("/tmp/f9_indexer_stage")
    parquet_compression: str = "zstd"
    parquet_compression_level: int = 1
    write_batch_rows: int = 100_000
    poll_interval_s: float = 5.0
    chunk_rows: int = 1_000_000
    upload_workers: int = 4


@dataclass
class BulkInsertStats:
    """Per-collection metrics surfaced to the orchestrator's end-of-run
    summary."""
    rows_written: int = 0
    parquet_bytes: int = 0
    write_seconds: float = 0.0
    upload_seconds: float = 0.0
    bulk_insert_seconds: float = 0.0


def _s3_client(cfg: BulkInsertConfig):
    return boto3.client(
        "s3",
        endpoint_url=cfg.s3_endpoint,
        aws_access_key_id=cfg.s3_access_key,
        aws_secret_access_key=cfg.s3_secret_key,
        config=Config(signature_version="s3v4"),
        region_name=cfg.s3_region,
    )


# ---------- schemas -------------------------------------------------------

def articles_parquet_schema() -> pa.Schema:
    """Articles collection parquet schema — see encoding rules in the
    module docstring. Field order matches `create_articles_collection.py`
    for readability; bulk-insert is column-name addressable so order
    doesn't affect correctness."""
    return pa.schema([
        ("article_hash", pa.string()),
        # FLOAT16_VECTOR(128) → list<uint8> of 256 raw fp16 bytes.
        ("offer_embedding", pa.list_(pa.uint8())),
        ("text_codes", pa.string()),
        ("name", pa.string()),
        ("manufacturerName", pa.string()),
        ("category_l1", pa.list_(pa.string())),
        ("category_l2", pa.list_(pa.string())),
        ("category_l3", pa.list_(pa.string())),
        ("category_l4", pa.list_(pa.string())),
        ("category_l5", pa.list_(pa.string())),
        ("eclass5_code", pa.list_(pa.int32())),
        ("eclass7_code", pa.list_(pa.int32())),
        ("s2class_code", pa.list_(pa.int32())),
        # JSON column → string containing the JSON value.
        ("customer_article_numbers", pa.string()),
        *[(f"{c}_price_{s}", pa.float32()) for c in CATALOG_CURRENCIES for s in ("min", "max")],
    ])


def offers_parquet_schema() -> pa.Schema:
    """Offers collection parquet schema — F9 split + F8 envelope."""
    return pa.schema([
        ("id", pa.string()),
        # FLOAT_VECTOR(2) — 2-d placeholder; ordinary list<float>.
        ("_placeholder_vector", pa.list_(pa.float32())),
        ("article_hash", pa.string()),
        ("ean", pa.string()),
        ("article_number", pa.string()),
        ("vendor_id", pa.string()),
        ("catalog_version_ids", pa.list_(pa.string())),
        # JSON column.
        ("prices", pa.string()),
        ("delivery_time_days_max", pa.int32()),
        ("core_marker_enabled_sources", pa.list_(pa.string())),
        ("core_marker_disabled_sources", pa.list_(pa.string())),
        ("features", pa.list_(pa.string())),
        ("relationship_accessory_for", pa.list_(pa.string())),
        ("relationship_spare_part_for", pa.list_(pa.string())),
        ("relationship_similar_to", pa.list_(pa.string())),
        ("price_list_ids", pa.list_(pa.string())),
        ("currencies", pa.list_(pa.string())),
        *[(f"{c}_price_{s}", pa.float32()) for c in CATALOG_CURRENCIES for s in ("min", "max")],
    ])


# ---------- batch encoders -----------------------------------------------

def _encode_fp16_to_uint8_list(vectors: list[np.ndarray]) -> pa.Array:
    """Stack a list of fp16 vectors and view as raw bytes, then build a
    list<uint8> array with one entry per vector. Mirrors
    `scripts/milvus_bulk_import.py:convert_batch`."""
    arr = np.stack(vectors)              # (n, dim) fp16
    if arr.dtype != np.float16:
        arr = arr.astype(np.float16)
    n, dim = arr.shape
    flat_bytes = arr.view(np.uint8)      # (n, 2*dim) uint8
    width = flat_bytes.shape[1]
    flat = pa.array(flat_bytes.reshape(-1), type=pa.uint8())
    offsets = pa.array(np.arange(0, n * width + 1, width, dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, flat)


def _articles_batch_to_arrow(batch: list[dict]) -> pa.RecordBatch:
    """Convert one orchestrator-emitted article batch (DuckDB row dicts +
    `offer_embedding` numpy arrays) into a parquet RecordBatch."""
    n = len(batch)
    cols: dict[str, pa.Array] = {}
    cols["article_hash"] = pa.array([r["article_hash"] for r in batch], type=pa.string())
    cols["offer_embedding"] = _encode_fp16_to_uint8_list([r["offer_embedding"] for r in batch])
    cols["text_codes"] = pa.array([r["text_codes"] for r in batch], type=pa.string())
    cols["name"] = pa.array([r["name"] for r in batch], type=pa.string())
    cols["manufacturerName"] = pa.array([r["manufacturerName"] for r in batch], type=pa.string())
    for d in range(1, 6):
        col = f"category_l{d}"
        cols[col] = pa.array([r.get(col) or [] for r in batch], type=pa.list_(pa.string()))
    for f in ("eclass5_code", "eclass7_code", "s2class_code"):
        cols[f] = pa.array([r.get(f) or [] for r in batch], type=pa.list_(pa.int32()))
    # JSON column → string-encode each row's value (or null marker).
    cols["customer_article_numbers"] = pa.array(
        [json.dumps(r.get("customer_article_numbers") or [], ensure_ascii=False) for r in batch],
        type=pa.string(),
    )
    for c in CATALOG_CURRENCIES:
        for s in ("min", "max"):
            field = f"{c}_price_{s}"
            cols[field] = pa.array([float(r[field]) for r in batch], type=pa.float32())
    schema = articles_parquet_schema()
    return pa.RecordBatch.from_arrays(
        [cols[name] for name in schema.names], schema=schema,
    )


def _offers_batch_to_arrow(batch: list[dict]) -> pa.RecordBatch:
    """Convert one orchestrator-emitted offer batch into a parquet
    RecordBatch."""
    cols: dict[str, pa.Array] = {}
    cols["id"] = pa.array([r["id"] for r in batch], type=pa.string())
    cols["_placeholder_vector"] = pa.array(
        [list(r["_placeholder_vector"]) for r in batch], type=pa.list_(pa.float32()),
    )
    cols["article_hash"] = pa.array([r["article_hash"] for r in batch], type=pa.string())
    cols["ean"] = pa.array([r["ean"] for r in batch], type=pa.string())
    cols["article_number"] = pa.array([r["article_number"] for r in batch], type=pa.string())
    cols["vendor_id"] = pa.array([r["vendor_id"] for r in batch], type=pa.string())
    cols["catalog_version_ids"] = pa.array(
        [r.get("catalog_version_ids") or [] for r in batch], type=pa.list_(pa.string()),
    )
    cols["prices"] = pa.array(
        [json.dumps(r.get("prices") or [], ensure_ascii=False) for r in batch],
        type=pa.string(),
    )
    cols["delivery_time_days_max"] = pa.array(
        [int(r.get("delivery_time_days_max") or 0) for r in batch], type=pa.int32(),
    )
    for f in ("core_marker_enabled_sources", "core_marker_disabled_sources",
              "features",
              "relationship_accessory_for", "relationship_spare_part_for", "relationship_similar_to",
              "price_list_ids", "currencies"):
        cols[f] = pa.array([r.get(f) or [] for r in batch], type=pa.list_(pa.string()))
    for c in CATALOG_CURRENCIES:
        for s in ("min", "max"):
            field = f"{c}_price_{s}"
            cols[field] = pa.array([float(r[field]) for r in batch], type=pa.float32())
    schema = offers_parquet_schema()
    return pa.RecordBatch.from_arrays(
        [cols[name] for name in schema.names], schema=schema,
    )


# ---------- writers -------------------------------------------------------

# Type alias for a "row converter" — turns one orchestrator batch
# (list of DuckDB dicts) into one parquet RecordBatch with the right
# column types. Lets `_write_chunked` stay schema-agnostic.
from typing import Callable

_BatchConverter = Callable[[list[dict]], pa.RecordBatch]


def _write_chunked(
    batches: Iterator[list[dict]],
    *,
    stage_dir: Path,
    name_template: str,
    schema: pa.Schema,
    convert: _BatchConverter,
    chunk_rows: int,
    compression: str,
    compression_level: int,
) -> Iterator[tuple[Path, int, int]]:
    """Stream parquet output as a sequence of `(path, rows, bytes)` chunks.

    A new chunk file opens whenever the in-flight writer reaches
    `chunk_rows`. The last chunk may be smaller. Yields each chunk the
    moment it closes — callers can pipeline upload/submit while the
    next chunk stages.

    `name_template` should contain a `{idx:04d}` placeholder, e.g.
    `articles.{idx:04d}.parquet`."""
    chunk_idx = 0
    rows_in_chunk = 0
    writer: pq.ParquetWriter | None = None
    path: Path | None = None

    def _open() -> tuple[Path, pq.ParquetWriter]:
        p = stage_dir / name_template.format(idx=chunk_idx)
        return p, pq.ParquetWriter(
            p, schema,
            compression=compression,
            compression_level=compression_level,
        )

    for batch in batches:
        if not batch:
            continue
        if writer is None:
            path, writer = _open()
            rows_in_chunk = 0
        writer.write_batch(convert(batch))
        rows_in_chunk += len(batch)
        if rows_in_chunk >= chunk_rows:
            writer.close()
            assert path is not None
            yield path, rows_in_chunk, path.stat().st_size
            chunk_idx += 1
            writer, path, rows_in_chunk = None, None, 0

    if writer is not None and path is not None:
        writer.close()
        yield path, rows_in_chunk, path.stat().st_size


def write_articles_parquet(
    batches: Iterator[list[dict]],
    *,
    stage_dir: Path,
    chunk_rows: int,
    compression: str,
    compression_level: int,
) -> Iterator[tuple[Path, int, int]]:
    """Articles parquet writer. Yields one `(path, rows, bytes)` per
    chunk as it closes."""
    yield from _write_chunked(
        batches,
        stage_dir=stage_dir,
        name_template="articles.{idx:04d}.parquet",
        schema=articles_parquet_schema(),
        convert=_articles_batch_to_arrow,
        chunk_rows=chunk_rows,
        compression=compression,
        compression_level=compression_level,
    )


def write_offers_parquet(
    batches: Iterator[list[dict]],
    *,
    stage_dir: Path,
    chunk_rows: int,
    compression: str,
    compression_level: int,
) -> Iterator[tuple[Path, int, int]]:
    """Offers parquet writer. Yields one `(path, rows, bytes)` per
    chunk as it closes."""
    yield from _write_chunked(
        batches,
        stage_dir=stage_dir,
        name_template="offers.{idx:04d}.parquet",
        schema=offers_parquet_schema(),
        convert=_offers_batch_to_arrow,
        chunk_rows=chunk_rows,
        compression=compression,
        compression_level=compression_level,
    )


# ---------- MinIO upload + Milvus submit ----------------------------------

def upload_to_s3(local: Path, *, cfg: BulkInsertConfig, key: str) -> None:
    """Upload a single parquet file to MinIO/S3 under the configured
    bucket + key. `key` is relative to the bucket root."""
    client = _s3_client(cfg)
    client.upload_file(str(local), cfg.s3_bucket, key)


def _ensure_milvus_connection(milvus_uri: str) -> None:
    """`do_bulk_insert` lives on the legacy `pymilvus.utility` API which
    uses the global `connections` registry. `connections.connect` is
    idempotent on the same alias, so it's safe to call repeatedly from
    background threads."""
    if "://" in milvus_uri:
        _scheme, _, hostport = milvus_uri.partition("://")
    else:
        hostport = milvus_uri
    host, _, port = hostport.partition(":")
    connections.connect(alias="default", host=host or "localhost", port=port or "19530")


def stream_chunks_to_milvus(
    chunks: Iterator[tuple[Path, int, int]],
    *,
    milvus_uri: str,
    collection: str,
    cfg: BulkInsertConfig,
    on_chunk_done: Callable[[Path, int, int, int], None] | None = None,
) -> tuple[int, BulkInsertStats]:
    """Pipelined upload + `do_bulk_insert` over a chunk-yielding writer.

    For each `(path, rows, bytes)` produced by the writer:
      1. Submit the chunk to a worker thread that uploads it to MinIO,
         then calls `utility.do_bulk_insert(collection, files=[key])`.
      2. Collect job IDs as workers complete.
      3. Continue consuming the writer in the main thread (it stages
         the next chunk's parquet while prior chunks are in-flight).

    After the writer drains, poll the server until every job reaches
    Completed (or Failed → raise). Per-chunk parquet files are deleted
    immediately after upload to keep `stage_dir` bounded.

    `on_chunk_done(path, rows, bytes, job_id)` is invoked once per chunk
    after its bulk_insert is submitted — used by the orchestrator for
    progress logging."""
    _ensure_milvus_connection(milvus_uri)
    pool = ThreadPoolExecutor(max_workers=cfg.upload_workers)
    upload_futures: list[Future[tuple[int, int]]] = []
    stats = BulkInsertStats()

    def _upload_and_submit(path: Path, rows: int, byte_count: int) -> tuple[int, int]:
        s3_key = f"{cfg.s3_prefix}/{path.name}"
        t0 = time.time()
        upload_to_s3(path, cfg=cfg, key=s3_key)
        upload_t = time.time() - t0
        path.unlink(missing_ok=True)
        # `do_bulk_insert` returns immediately with a job id; the actual
        # ingest runs on the Milvus side and we poll for it below. Each
        # call needs a connection on this thread's connection-pool slot.
        _ensure_milvus_connection(milvus_uri)
        job_id = utility.do_bulk_insert(collection_name=collection, files=[s3_key])
        log.info(
            "  %s: uploaded %.2f GB in %.1fs → job %d",
            path.name, byte_count / 1e9, upload_t, job_id,
        )
        if on_chunk_done is not None:
            on_chunk_done(path, rows, byte_count, job_id)
        return job_id, rows

    # Drain the writer; spawn an upload+submit task for each chunk.
    write_t0 = time.time()
    for path, rows, bytes_ in chunks:
        stats.rows_written += rows
        stats.parquet_bytes += bytes_
        upload_futures.append(pool.submit(_upload_and_submit, path, rows, bytes_))
    stats.write_seconds = time.time() - write_t0

    # Wait for every upload+submit to complete. Job IDs collected here.
    job_ids: list[int] = []
    upload_t0 = time.time()
    for fut in as_completed(upload_futures):
        job_id, _rows = fut.result()
        job_ids.append(job_id)
    stats.upload_seconds = time.time() - upload_t0
    pool.shutdown(wait=True)

    # Poll all jobs to completion. Each chunk's IndexBuilding stage
    # runs in parallel server-side, so this is roughly bounded by the
    # slowest single chunk's pipeline + cluster contention.
    insert_t0 = time.time()
    pending = list(job_ids)
    total_rows = 0
    while pending:
        next_pending: list[int] = []
        for job_id in pending:
            state = utility.get_bulk_insert_state(job_id)
            if state.state_name == "Completed":
                total_rows += state.row_count
                log.info("  job %d Completed (%d rows)", job_id, state.row_count)
            elif state.state_name == "Failed":
                raise RuntimeError(f"bulk_insert job {job_id} FAILED: {state.infos}")
            else:
                next_pending.append(job_id)
        if next_pending:
            still = ", ".join(
                f"{j}={utility.get_bulk_insert_state(j).progress}%"
                for j in next_pending
            )
            log.info("  bulk_insert pending: %s", still)
            time.sleep(cfg.poll_interval_s)
        pending = next_pending
    stats.bulk_insert_seconds = time.time() - insert_t0
    return total_rows, stats


def submit_and_wait_bulk_insert(
    *,
    milvus_uri: str,
    collection: str,
    s3_keys: list[str],
    cfg: BulkInsertConfig,
) -> tuple[int, float]:
    """Submit one or more pre-uploaded parquet files to `do_bulk_insert`
    and wait for every job to complete. Returns
    `(total_rows, wall_seconds)`.

    Used for already-staged-and-uploaded chunks; the new chunked
    orchestrator (`stream_chunks_to_milvus`) handles upload + submit
    itself."""
    if not s3_keys:
        return 0, 0.0
    # `MilvusClient(uri=...)` accepts URIs like "http://host:19530";
    # `connections.connect` wants split host/port. Parse minimally.
    if "://" in milvus_uri:
        _scheme, _, hostport = milvus_uri.partition("://")
    else:
        hostport = milvus_uri
    host, _, port = hostport.partition(":")
    connections.connect(alias="default", host=host or "localhost", port=port or "19530")

    job_ids: list[int] = []
    for key in s3_keys:
        job_id = utility.do_bulk_insert(collection_name=collection, files=[key])
        job_ids.append(job_id)
        log.info("  submitted bulk_insert job %d for %s", job_id, key)

    t0 = time.time()
    pending = list(job_ids)
    total_rows = 0
    while pending:
        next_pending: list[int] = []
        for job_id in pending:
            state = utility.get_bulk_insert_state(job_id)
            if state.state_name == "Completed":
                total_rows += state.row_count
                log.info("  job %d Completed (%d rows)", job_id, state.row_count)
            elif state.state_name == "Failed":
                raise RuntimeError(f"bulk_insert job {job_id} FAILED: {state.infos}")
            else:
                next_pending.append(job_id)
        if next_pending:
            still = ", ".join(
                f"{j}={utility.get_bulk_insert_state(j).progress}%"
                for j in next_pending
            )
            log.info("  bulk_insert pending: %s", still)
            time.sleep(cfg.poll_interval_s)
        pending = next_pending
    return total_rows, time.time() - t0


__all__ = [
    "BulkInsertConfig",
    "BulkInsertStats",
    "articles_parquet_schema",
    "offers_parquet_schema",
    "write_articles_parquet",
    "write_offers_parquet",
    "upload_to_s3",
    "stream_chunks_to_milvus",
    "submit_and_wait_bulk_insert",
]
