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
    # Retry tunables. boto3 retries S3 ops itself via its `retries`
    # config (we set it on every client). do_bulk_insert is wrapped
    # by `_do_bulk_insert_with_retry`. Both use exponential backoff
    # capped at `retry_max_backoff_s`.
    retry_attempts: int = 5
    retry_initial_backoff_s: float = 2.0
    retry_max_backoff_s: float = 60.0
    # Resume support. If `checkpoint_path` is set, the orchestrator:
    #   1. On startup: reads the file (if present), skips DuckDB rows
    #      already done, continues chunk numbering from the last index.
    #   2. After each chunk's bulk_insert succeeds: updates the file
    #      atomically (write-temp + rename).
    # Default `None` = no resume; partial runs are wasted on failure.
    # For production runs always set this to a stable path under
    # `stage_dir` so a node restart picks up where it left off.
    checkpoint_path: Path | None = None


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
        # `retries.mode='standard'` gives 5xx + ConnectTimeout +
        # ReadTimeout retries with exponential backoff out of the box.
        # `max_attempts` includes the initial call, so attempts=5 means
        # 1 try + 4 retries.
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": cfg.retry_attempts, "mode": "standard"},
        ),
        region_name=cfg.s3_region,
    )


# Milvus exception messages we treat as PERMANENT — retrying these is
# pointless and just delays surfacing the real problem to the operator.
# Lowercased substring match; expand as we encounter new permanent
# failure shapes in production.
_PERMANENT_MILVUS_ERRORS = (
    "validation",
    "schema",
    "not exist",
    "not found",
    "permission",
    "unauthenticated",
    "field not found",
    "duplicate",
)


def _is_permanent_milvus_error(exc: BaseException) -> bool:
    """Return True for errors that won't go away by waiting. Anything
    else is treated as transient and retried."""
    msg = str(exc).lower()
    return any(s in msg for s in _PERMANENT_MILVUS_ERRORS)


def _do_bulk_insert_with_retry(
    *,
    collection: str,
    files: list[str],
    cfg: BulkInsertConfig,
) -> int:
    """Wrap `utility.do_bulk_insert` with exponential-backoff retries on
    transient failures. Permanent failures (validation, schema, perms)
    raise immediately so operators see the real cause.

    Boto3 retries S3 ops itself; this wrapper is for the gRPC call into
    Milvus. Returns the job_id."""
    last_exc: Exception | None = None
    for attempt in range(cfg.retry_attempts):
        try:
            return utility.do_bulk_insert(collection_name=collection, files=files)
        except Exception as e:
            if _is_permanent_milvus_error(e):
                log.error("do_bulk_insert: permanent failure (%s) — not retrying", e)
                raise
            last_exc = e
            if attempt == cfg.retry_attempts - 1:
                log.error("do_bulk_insert: exhausted %d attempts — final error: %s",
                          cfg.retry_attempts, e)
                raise
            wait = min(
                cfg.retry_initial_backoff_s * (2 ** attempt),
                cfg.retry_max_backoff_s,
            )
            log.warning(
                "do_bulk_insert attempt %d/%d failed (%s) — retrying in %.1fs",
                attempt + 1, cfg.retry_attempts, e, wait,
            )
            time.sleep(wait)
    # Unreachable — the loop either returns or raises.
    raise RuntimeError("unreachable") from last_exc


# ---------- checkpoint ----------------------------------------------------

# Increment if checkpoint format changes incompatibly.
_CHECKPOINT_VERSION = 1


def _empty_checkpoint() -> dict:
    """Initial state when no checkpoint file exists yet — fresh run."""
    return {
        "version": _CHECKPOINT_VERSION,
        "articles": {"rows_done": 0, "chunks_done": 0},
        "offers":   {"rows_done": 0, "chunks_done": 0},
    }


def load_checkpoint(path: Path | None) -> dict:
    """Load checkpoint state from disk, or return the empty initial
    state if `path` is None or the file doesn't exist. Raises if the
    file exists but isn't parseable — operators should investigate
    rather than silently lose track of completed chunks."""
    if path is None or not path.exists():
        return _empty_checkpoint()
    state = json.loads(path.read_text())
    if state.get("version") != _CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint at {path} has version {state.get('version')!r}; "
            f"this build expects {_CHECKPOINT_VERSION}. Delete the file "
            f"to start fresh, or downgrade."
        )
    # Tolerate a checkpoint that only mentions one stream.
    state.setdefault("articles", {"rows_done": 0, "chunks_done": 0})
    state.setdefault("offers",   {"rows_done": 0, "chunks_done": 0})
    return state


def save_checkpoint(path: Path | None, state: dict) -> None:
    """Atomically persist checkpoint state via temp + rename. No-op when
    `path` is None (resume disabled)."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


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
    starting_chunk_idx: int = 0,
) -> Iterator[tuple[int, Path, int, int]]:
    """Stream parquet output as a sequence of `(chunk_idx, path, rows, bytes)`
    chunks.

    A new chunk file opens whenever the in-flight writer reaches
    `chunk_rows`. The last chunk may be smaller. Yields each chunk the
    moment it closes — callers can pipeline upload/submit while the
    next chunk stages.

    `name_template` should contain a `{idx:04d}` placeholder, e.g.
    `articles.{idx:04d}.parquet`. `starting_chunk_idx` lets a resume
    continue file numbering past the chunks already in the bucket so
    the new files don't collide with old ones."""
    chunk_idx = starting_chunk_idx
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
            yield chunk_idx, path, rows_in_chunk, path.stat().st_size
            chunk_idx += 1
            writer, path, rows_in_chunk = None, None, 0

    if writer is not None and path is not None:
        writer.close()
        yield chunk_idx, path, rows_in_chunk, path.stat().st_size


def write_articles_parquet(
    batches: Iterator[list[dict]],
    *,
    stage_dir: Path,
    chunk_rows: int,
    compression: str,
    compression_level: int,
    starting_chunk_idx: int = 0,
) -> Iterator[tuple[int, Path, int, int]]:
    """Articles parquet writer. Yields `(chunk_idx, path, rows, bytes)`
    per chunk as it closes."""
    yield from _write_chunked(
        batches,
        stage_dir=stage_dir,
        name_template="articles.{idx:04d}.parquet",
        schema=articles_parquet_schema(),
        convert=_articles_batch_to_arrow,
        chunk_rows=chunk_rows,
        compression=compression,
        compression_level=compression_level,
        starting_chunk_idx=starting_chunk_idx,
    )


def write_offers_parquet(
    batches: Iterator[list[dict]],
    *,
    stage_dir: Path,
    chunk_rows: int,
    compression: str,
    compression_level: int,
    starting_chunk_idx: int = 0,
) -> Iterator[tuple[int, Path, int, int]]:
    """Offers parquet writer."""
    yield from _write_chunked(
        batches,
        stage_dir=stage_dir,
        name_template="offers.{idx:04d}.parquet",
        schema=offers_parquet_schema(),
        convert=_offers_batch_to_arrow,
        chunk_rows=chunk_rows,
        compression=compression,
        compression_level=compression_level,
        starting_chunk_idx=starting_chunk_idx,
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
    chunks: Iterator[tuple[int, Path, int, int]],
    *,
    milvus_uri: str,
    collection: str,
    cfg: BulkInsertConfig,
    on_chunk_completed: Callable[[int, int], None] | None = None,
) -> tuple[int, BulkInsertStats]:
    """Pipelined upload + `do_bulk_insert` over a chunk-yielding writer.

    For each `(chunk_idx, path, rows, bytes)` produced by the writer:
      1. Submit the chunk to a worker thread that uploads it to MinIO
         (boto3 retries internally) then calls `do_bulk_insert` (with
         retry via `_do_bulk_insert_with_retry`).
      2. Track the in-flight (chunk_idx, job_id, rows) tuple.
      3. As jobs complete, fire `on_chunk_completed(chunk_idx, rows)`
         so the orchestrator can persist a checkpoint.

    Per-chunk parquet files are deleted immediately after upload so the
    local stage dir holds at most `chunk_rows × upload_workers` rows
    worth of data at any moment.

    `on_chunk_completed` is invoked synchronously from the polling loop
    in submission order — checkpoint writes serialise here, so a
    completion of chunk N+1 won't be persisted until chunk N has been
    persisted, which keeps the stored `chunks_done` monotonic and safe
    to use as a resume offset (`rows_done = chunks_done × chunk_rows`)."""
    _ensure_milvus_connection(milvus_uri)
    pool = ThreadPoolExecutor(max_workers=cfg.upload_workers)
    # Each future returns (chunk_idx, job_id, rows) once its parquet is
    # uploaded + bulk_insert is submitted. Polling happens in the main
    # thread so we don't double-poll the same job.
    upload_futures: list[Future[tuple[int, int, int]]] = []
    stats = BulkInsertStats()

    def _upload_and_submit(
        chunk_idx: int, path: Path, rows: int, byte_count: int,
    ) -> tuple[int, int, int]:
        s3_key = f"{cfg.s3_prefix}/{path.name}"
        t0 = time.time()
        upload_to_s3(path, cfg=cfg, key=s3_key)
        upload_t = time.time() - t0
        path.unlink(missing_ok=True)
        _ensure_milvus_connection(milvus_uri)
        job_id = _do_bulk_insert_with_retry(
            collection=collection, files=[s3_key], cfg=cfg,
        )
        log.info(
            "  %s: uploaded %.2f GB in %.1fs → job %d",
            path.name, byte_count / 1e9, upload_t, job_id,
        )
        return chunk_idx, job_id, rows

    # Drain the writer; spawn an upload+submit task for each chunk.
    write_t0 = time.time()
    for chunk_idx, path, rows, bytes_ in chunks:
        stats.rows_written += rows
        stats.parquet_bytes += bytes_
        upload_futures.append(pool.submit(_upload_and_submit, chunk_idx, path, rows, bytes_))
    stats.write_seconds = time.time() - write_t0

    # Collect (chunk_idx, job_id, rows) per chunk that successfully
    # submitted. If any worker raised (e.g. retry-exhausted upload or
    # do_bulk_insert error), we keep going so we can poll already-
    # submitted chunks before propagating the exception. Without this
    # we'd lose track of jobs that are running server-side, leaving
    # the checkpoint stale and the next resume re-importing duplicates.
    submitted: list[tuple[int, int, int]] = []
    submit_exceptions: list[BaseException] = []
    upload_t0 = time.time()
    for fut in as_completed(upload_futures):
        try:
            submitted.append(fut.result())
        except BaseException as e:
            submit_exceptions.append(e)
    stats.upload_seconds = time.time() - upload_t0
    pool.shutdown(wait=True)

    # Poll all submitted jobs to completion. Fire `on_chunk_completed`
    # only for the *contiguous* prefix of completed chunks — out-of-order
    # server-side completions (chunk 5 done while chunk 3 still importing)
    # must NOT advance `chunks_done` past chunk 3, because the checkpoint
    # is consumed as a single-integer resume offset.
    insert_t0 = time.time()
    submitted.sort(key=lambda t: t[0])
    expected_next_idx = submitted[0][0] if submitted else 0
    completed_rows_by_idx: dict[int, int] = {}
    pending: list[tuple[int, int, int]] = list(submitted)
    total_rows = 0
    poll_failures: list[str] = []
    while pending:
        next_pending: list[tuple[int, int, int]] = []
        for chunk_idx, job_id, rows in pending:
            state = utility.get_bulk_insert_state(job_id)
            if state.state_name == "Completed":
                total_rows += state.row_count
                completed_rows_by_idx[chunk_idx] = rows
                log.info("  chunk %d (job %d) Completed (%d rows)",
                         chunk_idx, job_id, state.row_count)
            elif state.state_name == "Failed":
                # Track but don't raise here — we want to flush the
                # contiguous-prefix checkpoint for any earlier chunks
                # that did succeed before bailing.
                poll_failures.append(
                    f"chunk {chunk_idx} (job {job_id}) FAILED: {state.infos}"
                )
                log.error("  %s", poll_failures[-1])
            else:
                next_pending.append((chunk_idx, job_id, rows))

        # Drain contiguous-prefix completions into the checkpoint.
        while expected_next_idx in completed_rows_by_idx:
            rows = completed_rows_by_idx.pop(expected_next_idx)
            if on_chunk_completed is not None:
                on_chunk_completed(expected_next_idx, rows)
            expected_next_idx += 1

        if next_pending:
            still = ", ".join(
                f"chunk {ci}={utility.get_bulk_insert_state(ji).progress}%"
                for ci, ji, _ in next_pending
            )
            log.info("  bulk_insert pending: %s", still)
            time.sleep(cfg.poll_interval_s)
        pending = next_pending
    stats.bulk_insert_seconds = time.time() - insert_t0

    # Surface any deferred failures (poll-side or submit-side) now that
    # we've safely persisted the prefix of completed chunks.
    if poll_failures:
        raise RuntimeError("; ".join(poll_failures))
    if submit_exceptions:
        # Re-raise the first to preserve the traceback; secondary errors
        # are typically the same cascading-network issue and one example
        # is enough for the operator.
        raise submit_exceptions[0]
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
    "load_checkpoint",
    "save_checkpoint",
]
