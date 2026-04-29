"""F9 production bulk indexer (MongoDB Atlas snapshot → Milvus).

Top-level orchestrator that wires the DuckDB-native projection +
aggregation (`indexer.duckdb_projection`) to TEI batched embedding with
a Redis cache (`indexer.tei_cache`) and paired Milvus writes (upsert
or bulk-insert).

Pipeline:

    raw S3 / local                DuckDB                 TEI + Redis             Milvus
    ──────────────                ──────                 ───────────             ──────
    offers/*.json.gz   ─┐
    pricings/*.json.gz ─┤  load_raw_collections
    markers/*.json.gz  ─┼─►  +  4-way JOIN on (vendorId, articleNumber)
    cans/*.json.gz     ─┘  +  per-row projection
                              +  per-hash aggregation     ┐
                                  ▼                         │
                              `articles` table  ──── batch  ┼─► embed_articles ─► sink
                              `offer_rows` table ── batch  ─┴───────────────────► sink

Two sink modes — pick via `sink_mode` ("upsert" or "bulk_insert"):

  * "upsert" (default): per-batch `MilvusClient.upsert(data=...)`. Slow
    but per-row idempotent and queryable immediately. Right for smoke
    runs and small collections.
  * "bulk_insert": stage all rows to parquet + MinIO/S3, submit
    `do_bulk_insert` jobs. ~50–100K rows/sec vs ~800/sec for upsert.
    Right for production-scale reindex (159M articles + 510M offers).
    See `indexer.bulk_insert`.

The DuckDB stage stages everything to disk-spill (`duckdb` temp dir),
so a 510M-offer / 158 GB dataset fits a 1 TB NVMe scratch box. Article
+ offer rows are materialised so the upsert phase can iterate without
re-running the join. The TEI cache is hash-keyed in Redis so reruns
skip the GPU for any embedding already produced — `HASH_VERSION` (in
`indexer/projection.py`) prefixes the keys, so a hash-version bump
naturally invalidates the cache.

Failure modes + how the indexer handles them:

  - DuckDB temp-dir overflow → bump `--duckdb-temp-dir-limit` (env-tunable)
    or point `--duckdb-temp-dir` at a larger volume.
  - TEI degraded / partial failures → the orchestrator surfaces the
    HTTP error; restart picks up from Redis cache (any successfully
    embedded hash is cached).
  - Milvus upsert failure → log + raise. Re-run is idempotent at the
    article level (PK = article_hash, upsert overwrites identical
    content) and at the offer level (PK = `{vendor_uuid}:{b64url(article_number)}`).

Per F9, the alias swing happens AFTER this script completes — see
`scripts/MILVUS_ALIAS_WORKFLOW.md` "Paired alias swing" for the
operator playbook. This script writes to versioned collection names
(`articles_v{N}`, `offers_v{N+1}`) directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterator

import duckdb
import redis
from pymilvus import MilvusClient

from indexer.bulk_insert import (
    BulkInsertConfig,
    BulkInsertStats,
    load_checkpoint,
    save_checkpoint,
    stream_chunks_to_milvus,
    write_articles_parquet,
    write_offers_parquet,
)
from indexer.duckdb_projection import (
    _build_articles_sql,
    _build_offers_sql,
    init_macros,
    load_raw_collections,
)
from indexer.tei_cache import TEICache, TEICacheStats

log = logging.getLogger(__name__)


@dataclass
class BulkRunStats:
    """Surfaced to the CLI for end-of-run reporting + parsing by ops
    tooling. `tei` is forwarded from `TEICache.stats` post-run."""
    raw_offer_count: int = 0
    raw_pricing_count: int = 0
    raw_marker_count: int = 0
    raw_can_count: int = 0
    article_count: int = 0
    offer_row_count: int = 0
    article_upsert_seconds: float = 0.0
    offer_upsert_seconds: float = 0.0
    duckdb_seconds: float = 0.0
    total_seconds: float = 0.0
    tei: TEICacheStats = field(default_factory=TEICacheStats)
    # Populated only on `sink_mode='bulk_insert'`. The upsert path
    # leaves these zeroed out — the upsert wall time is in
    # `article_upsert_seconds` / `offer_upsert_seconds` for both modes
    # so the end-of-run summary stays comparable.
    articles_bulk_insert: BulkInsertStats = field(default_factory=BulkInsertStats)
    offers_bulk_insert: BulkInsertStats = field(default_factory=BulkInsertStats)


def _connect_duckdb(
    *,
    temp_dir: str | None,
    temp_dir_limit_gb: int,
    s3_region: str | None,
    needs_s3: bool,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection sized for the bulk run. Disk-spill is
    pinned to `temp_dir` (default = DuckDB's per-connection temp dir)
    with a hard cap so a runaway query doesn't fill the host root FS.
    The httpfs extension + S3 credential_chain secret are installed
    only when at least one glob is `s3://...` — local-only runs skip the
    network setup entirely."""
    con = duckdb.connect()
    if temp_dir:
        con.execute(f"SET temp_directory = '{temp_dir}'")
    con.execute(f"SET max_temp_directory_size = '{temp_dir_limit_gb}GB'")
    if needs_s3:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        # `credential_chain` mirrors boto3's lookup order (env → shared
        # config → instance metadata). Set AWS_PROFILE on the host or
        # rely on instance role; this script doesn't accept inline
        # secrets to avoid leaking them into shell history / process list.
        region_clause = f", REGION '{s3_region}'" if s3_region else ""
        con.execute(
            "CREATE OR REPLACE SECRET s3_secret "
            f"(TYPE S3, PROVIDER credential_chain{region_clause})"
        )
    init_macros(con)
    return con


def _iter_relation_dicts(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    batch_size: int,
    *,
    offset: int = 0,
) -> Iterator[list[dict]]:
    """Stream a DuckDB table as batches of plain dicts. `fetchmany`
    pulls one chunk at a time so the orchestrator never holds more
    than `batch_size` rows in Python at once — important at the
    130M-article scale where holding everything would OOM Python long
    before Milvus pushed back.

    `offset` skips that many rows from the start of the scan — used by
    the bulk-insert resume path to pick up past chunks already
    successfully ingested. DuckDB's table scan is deterministic on a
    static materialised table (no concurrent writers), so the same
    offset returns the same suffix across runs.
    """
    if offset > 0:
        cur = con.execute(f"SELECT * FROM {table_name} OFFSET {offset}")
    else:
        cur = con.execute(f"SELECT * FROM {table_name}")
    columns = [d[0] for d in cur.description]
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            return
        yield [dict(zip(columns, r)) for r in rows]


def _materialise_streams(con: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    """Run the raw → article + raw → offer SQL chains, materialising
    both result sets as DuckDB tables. Returns the row counts.

    Both streams share the upstream `flat → projected → finalized →
    with_hash` chain. CREATE TABLE forces materialisation up to that
    point; downstream reads against the two tables don't re-run the
    join. DuckDB will reuse the same vectorised aggregate state for
    article + offer streams."""
    log.info("Building articles table from raw collections (DuckDB JOIN + aggregate)…")
    con.execute(f"CREATE OR REPLACE TABLE articles AS {_build_articles_sql(source='raw')}")
    article_count = con.execute("SELECT count(*) FROM articles").fetchone()[0]
    log.info("  articles materialised: %d rows", article_count)

    log.info("Building offer_rows table from raw collections (DuckDB JOIN)…")
    con.execute(f"CREATE OR REPLACE TABLE offer_rows AS {_build_offers_sql(source='raw')}")
    offer_count = con.execute("SELECT count(*) FROM offer_rows").fetchone()[0]
    log.info("  offer_rows materialised: %d rows", offer_count)

    return article_count, offer_count


def _upsert_articles(
    *,
    con: duckdb.DuckDBPyConnection,
    milvus: MilvusClient,
    collection: str,
    cache: TEICache,
    batch_size: int,
) -> float:
    """Stream the materialised `articles` table → batched TEI lookup →
    Milvus upsert. Each batch:

      1. Pull `batch_size` rows from DuckDB.
      2. Hand the dicts to `cache.embed_articles` — Redis lookup +
         TEI call for misses, returns `{article_hash → fp16 vector}`.
      3. Attach `offer_embedding` to each row, upsert into Milvus.

    Order matters across the paired upsert: per F9 alias-swing
    protocol (`scripts/MILVUS_ALIAS_WORKFLOW.md`), articles must land
    before offers so a partial failure leaves the system with usable
    article rows missing some offers (rather than orphaned offers
    pointing at non-existent hashes)."""
    t0 = time.time()
    total = 0
    for batch in _iter_relation_dicts(con, "articles", batch_size):
        embeddings = cache.embed_articles(batch)
        for row in batch:
            row["offer_embedding"] = embeddings[row["article_hash"]]
        milvus.upsert(collection_name=collection, data=batch)
        total += len(batch)
        log.info(
            "  articles upserted: %d  (cache hits=%d misses=%d)",
            total, cache.stats.hits, cache.stats.misses,
        )
    return time.time() - t0


def _upsert_offers(
    *,
    con: duckdb.DuckDBPyConnection,
    milvus: MilvusClient,
    collection: str,
    batch_size: int,
) -> float:
    """Stream the materialised `offer_rows` table → Milvus upsert. No
    embedding step — `offers_v{N}` carries only the placeholder vector
    (the F9 dense vector lives on `articles_v{N}`)."""
    t0 = time.time()
    total = 0
    for batch in _iter_relation_dicts(con, "offer_rows", batch_size):
        milvus.upsert(collection_name=collection, data=batch)
        total += len(batch)
        log.info("  offers upserted: %d", total)
    return time.time() - t0


def _bulk_insert_articles(
    *,
    con: duckdb.DuckDBPyConnection,
    milvus_uri: str,
    collection: str,
    cache: TEICache,
    batch_size: int,
    cfg: BulkInsertConfig,
    checkpoint: dict,
) -> tuple[float, BulkInsertStats]:
    """Bulk-insert variant of `_upsert_articles`. Streams DuckDB → TEI
    cache → fp16 vectors → chunked parquets in `stage_dir`, with each
    chunk pipelined into upload + `do_bulk_insert` via a thread pool
    (`stream_chunks_to_milvus`). Chunk N+1 stages while chunk N is
    still uploading or being ingested server-side.

    Resume: `checkpoint['articles']['rows_done']` is used as a DuckDB
    OFFSET so a re-run skips rows already ingested. The chunk file
    names continue past the indices already in the bucket
    (`starting_chunk_idx = checkpoint['articles']['chunks_done']`) so
    new files don't collide with old ones still in MinIO. After each
    chunk's bulk_insert Completes, the checkpoint is updated atomically.

    Returns `(wall_seconds, stats)`."""
    cfg.stage_dir.mkdir(parents=True, exist_ok=True)
    rows_done = checkpoint["articles"]["rows_done"]
    chunks_done = checkpoint["articles"]["chunks_done"]
    if rows_done > 0:
        log.info(
            "  articles: resuming from row %d (chunk %d) per checkpoint",
            rows_done, chunks_done,
        )

    def _embed_and_emit() -> Iterator[list[dict]]:
        """Inner generator: pull a batch from DuckDB, attach embeddings,
        yield for the chunked parquet writer."""
        for batch in _iter_relation_dicts(con, "articles", batch_size, offset=rows_done):
            embeddings = cache.embed_articles(batch)
            for row in batch:
                row["offer_embedding"] = embeddings[row["article_hash"]]
            log.info(
                "  articles staged: +%d  (cache hits=%d misses=%d)",
                len(batch), cache.stats.hits, cache.stats.misses,
            )
            yield batch

    def _on_chunk_completed(chunk_idx: int, rows: int) -> None:
        checkpoint["articles"]["chunks_done"] = chunk_idx + 1
        checkpoint["articles"]["rows_done"] = checkpoint["articles"].get("rows_done", 0) + rows
        save_checkpoint(cfg.checkpoint_path, checkpoint)

    t0 = time.time()
    chunks = write_articles_parquet(
        _embed_and_emit(),
        stage_dir=cfg.stage_dir,
        chunk_rows=cfg.chunk_rows,
        compression=cfg.parquet_compression,
        compression_level=cfg.parquet_compression_level,
        starting_chunk_idx=chunks_done,
    )
    rows_imported, stats = stream_chunks_to_milvus(
        chunks, milvus_uri=milvus_uri, collection=collection, cfg=cfg,
        on_chunk_completed=_on_chunk_completed,
    )
    log.info("  articles bulk_insert imported %d rows total", rows_imported)
    return time.time() - t0, stats


def _bulk_insert_offers(
    *,
    con: duckdb.DuckDBPyConnection,
    milvus_uri: str,
    collection: str,
    batch_size: int,
    cfg: BulkInsertConfig,
    checkpoint: dict,
) -> tuple[float, BulkInsertStats]:
    """Bulk-insert variant of `_upsert_offers`. No embedding step —
    offer rows already carry their `_placeholder_vector` from the SQL.
    Resume semantics mirror `_bulk_insert_articles`."""
    cfg.stage_dir.mkdir(parents=True, exist_ok=True)
    rows_done = checkpoint["offers"]["rows_done"]
    chunks_done = checkpoint["offers"]["chunks_done"]
    if rows_done > 0:
        log.info(
            "  offers: resuming from row %d (chunk %d) per checkpoint",
            rows_done, chunks_done,
        )

    def _emit() -> Iterator[list[dict]]:
        for batch in _iter_relation_dicts(con, "offer_rows", batch_size, offset=rows_done):
            log.info("  offers staged: +%d", len(batch))
            yield batch

    def _on_chunk_completed(chunk_idx: int, rows: int) -> None:
        checkpoint["offers"]["chunks_done"] = chunk_idx + 1
        checkpoint["offers"]["rows_done"] = checkpoint["offers"].get("rows_done", 0) + rows
        save_checkpoint(cfg.checkpoint_path, checkpoint)

    t0 = time.time()
    chunks = write_offers_parquet(
        _emit(),
        stage_dir=cfg.stage_dir,
        chunk_rows=cfg.chunk_rows,
        compression=cfg.parquet_compression,
        compression_level=cfg.parquet_compression_level,
        starting_chunk_idx=chunks_done,
    )
    rows_imported, stats = stream_chunks_to_milvus(
        chunks, milvus_uri=milvus_uri, collection=collection, cfg=cfg,
        on_chunk_completed=_on_chunk_completed,
    )
    log.info("  offers bulk_insert imported %d rows total", rows_imported)
    return time.time() - t0, stats


def run_bulk_indexer(
    *,
    # Source — globs / S3 prefixes for each collection
    offers_glob: str,
    pricings_glob: str,
    markers_glob: str,
    cans_glob: str,
    # Sinks — Milvus collections (versioned, NOT aliases per F9 contract)
    milvus_uri: str,
    articles_collection: str,
    offers_collection: str,
    # Embedding service
    tei_url: str,
    redis_url: str,
    # Tunables
    article_batch_size: int = 1000,
    offer_batch_size: int = 5000,
    tei_batch_size: int = 64,
    duckdb_temp_dir: str | None = None,
    duckdb_temp_dir_limit_gb: int = 500,
    s3_region: str | None = "eu-central-1",
    # Sink — "upsert" (default, per-row) or "bulk_insert" (parquet → MinIO →
    # `do_bulk_insert`). Production-scale runs need bulk_insert; smoke runs
    # leave it on upsert for queryable-immediately semantics.
    sink_mode: str = "upsert",
    bulk_insert: BulkInsertConfig | None = None,
) -> BulkRunStats:
    """End-to-end bulk indexer entry point.

    The CLI (`scripts/indexer_bulk.py`) is a thin argparse wrapper
    around this — call this from a notebook or a parent driver if you
    want programmatic control over batch sizes."""
    if sink_mode not in ("upsert", "bulk_insert"):
        raise ValueError(f"sink_mode must be 'upsert' or 'bulk_insert', got {sink_mode!r}")
    if sink_mode == "bulk_insert" and bulk_insert is None:
        bulk_insert = BulkInsertConfig()
    wall_t0 = time.time()
    stats = BulkRunStats()

    needs_s3 = any(g.startswith("s3://") for g in (offers_glob, pricings_glob, markers_glob, cans_glob))
    con = _connect_duckdb(
        temp_dir=duckdb_temp_dir,
        temp_dir_limit_gb=duckdb_temp_dir_limit_gb,
        s3_region=s3_region,
        needs_s3=needs_s3,
    )

    log.info("Loading raw collections from %s", "S3" if needs_s3 else "local")
    log.info("  offers:   %s", offers_glob)
    log.info("  pricings: %s", pricings_glob)
    log.info("  markers:  %s", markers_glob)
    log.info("  cans:     %s", cans_glob)

    duck_t0 = time.time()
    load_raw_collections(
        con,
        offers_glob=offers_glob,
        pricings_glob=pricings_glob,
        markers_glob=markers_glob,
        cans_glob=cans_glob,
    )
    stats.raw_offer_count = con.execute("SELECT count(*) FROM raw_offers").fetchone()[0]
    stats.raw_pricing_count = con.execute("SELECT count(*) FROM raw_pricings").fetchone()[0]
    stats.raw_marker_count = con.execute("SELECT count(*) FROM raw_markers").fetchone()[0]
    stats.raw_can_count = con.execute("SELECT count(*) FROM raw_cans").fetchone()[0]
    log.info(
        "Loaded: offers=%d pricings=%d markers=%d cans=%d",
        stats.raw_offer_count, stats.raw_pricing_count,
        stats.raw_marker_count, stats.raw_can_count,
    )

    stats.article_count, stats.offer_row_count = _materialise_streams(con)
    stats.duckdb_seconds = time.time() - duck_t0

    log.info("Connecting to Milvus at %s", milvus_uri)
    milvus = MilvusClient(uri=milvus_uri)
    if not milvus.has_collection(articles_collection):
        raise RuntimeError(
            f"articles collection {articles_collection!r} does not exist. "
            f"Run scripts/create_articles_collection.py first."
        )
    if not milvus.has_collection(offers_collection):
        raise RuntimeError(
            f"offers collection {offers_collection!r} does not exist. "
            f"Run scripts/create_offers_collection.py first."
        )

    log.info("Connecting to Redis at %s", redis_url)
    redis_client = redis.Redis.from_url(redis_url)
    redis_client.ping()  # fail-fast if Redis is unreachable

    # Load resume state for the bulk-insert path. `checkpoint_path=None`
    # gives the empty initial state — no resume, fresh run.
    checkpoint = (
        load_checkpoint(bulk_insert.checkpoint_path)
        if sink_mode == "bulk_insert" and bulk_insert is not None
        else None
    )
    if checkpoint and (
        checkpoint["articles"]["rows_done"] > 0
        or checkpoint["offers"]["rows_done"] > 0
    ):
        log.info(
            "Resuming from checkpoint: articles=%d/%d rows, offers=%d/%d rows",
            checkpoint["articles"]["rows_done"], stats.article_count,
            checkpoint["offers"]["rows_done"],   stats.offer_row_count,
        )

    log.info("Phase 1: writing %d article rows via %s (TEI batch=%d)…",
             stats.article_count, sink_mode, tei_batch_size)
    with TEICache(
        tei_url=tei_url,
        redis_client=redis_client,
        tei_batch_size=tei_batch_size,
    ) as cache:
        if sink_mode == "upsert":
            stats.article_upsert_seconds = _upsert_articles(
                con=con,
                milvus=milvus,
                collection=articles_collection,
                cache=cache,
                batch_size=article_batch_size,
            )
        else:
            stats.article_upsert_seconds, stats.articles_bulk_insert = _bulk_insert_articles(
                con=con,
                milvus_uri=milvus_uri,
                collection=articles_collection,
                cache=cache,
                batch_size=article_batch_size,
                cfg=bulk_insert,    # type: ignore[arg-type]
                checkpoint=checkpoint,    # type: ignore[arg-type]
            )
        stats.tei = cache.stats

    log.info("Phase 2: writing %d offer rows via %s (no embedding)…",
             stats.offer_row_count, sink_mode)
    if sink_mode == "upsert":
        stats.offer_upsert_seconds = _upsert_offers(
            con=con,
            milvus=milvus,
            collection=offers_collection,
            batch_size=offer_batch_size,
        )
    else:
        stats.offer_upsert_seconds, stats.offers_bulk_insert = _bulk_insert_offers(
            con=con,
            milvus_uri=milvus_uri,
            collection=offers_collection,
            batch_size=offer_batch_size,
            cfg=bulk_insert,    # type: ignore[arg-type]
            checkpoint=checkpoint,    # type: ignore[arg-type]
        )

    stats.total_seconds = time.time() - wall_t0
    log.info(
        "Bulk run complete: articles=%d (%.0fs) offers=%d (%.0fs) duckdb=%.0fs total=%.0fs",
        stats.article_count, stats.article_upsert_seconds,
        stats.offer_row_count, stats.offer_upsert_seconds,
        stats.duckdb_seconds, stats.total_seconds,
    )
    log.info(
        "TEI cache: hits=%d misses=%d tei_calls=%d bytes_written=%.1f MB",
        stats.tei.hits, stats.tei.misses, stats.tei.tei_calls,
        stats.tei.bytes_written / 1e6,
    )
    return stats


__all__ = ["run_bulk_indexer", "BulkRunStats"]
