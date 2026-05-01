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
from pathlib import Path
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
from indexer.collection_specs import (
    apply_indexes_and_load,
    build_articles_index_params,
    build_offers_index_params,
    release_and_drop_indexes,
)
from indexer.duckdb_projection import (
    RAW_CAN_COLUMNS,
    RAW_MARKER_COLUMNS,
    RAW_OFFER_COLUMNS,
    RAW_PRICING_COLUMNS,
    _build_articles_sql,
    _build_offers_sql,
    init_macros,
    load_raw_collections,
    materialise_grouped_tables,
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
    # Index lifecycle wall time, only populated on the bulk_insert path
    # when `index_cycle=True`. `drop` covers release_collection + drop_index
    # for both collections; `rebuild` covers create_index + index-state
    # polling + load_collection. Together these collapse the per-segment
    # inline IndexBuilding cost (~20s per chunk on a loaded+indexed
    # collection) into a single post-flush pass.
    index_drop_seconds: float = 0.0
    index_rebuild_seconds: float = 0.0


def _connect_duckdb(
    *,
    temp_dir: str | None,
    temp_dir_limit_gb: int,
    memory_limit_gb: int | None,
    threads: int,
    s3_region: str | None,
    needs_s3: bool,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection sized for the bulk run. Disk-spill is
    pinned to `temp_dir` (default = DuckDB's per-connection temp dir)
    with a hard cap so a runaway query doesn't fill the host root FS.
    `memory_limit_gb` caps DuckDB's RSS — without it, multi-shard JSON
    loads can balloon past host RAM and OOM the kernel. The httpfs
    extension + S3 credential_chain secret are installed only when at
    least one glob is `s3://...` — local-only runs skip the network
    setup entirely."""
    con = duckdb.connect()
    if temp_dir:
        con.execute(f"SET temp_directory = '{temp_dir}'")
    con.execute(f"SET max_temp_directory_size = '{temp_dir_limit_gb}GB'")
    if memory_limit_gb is not None:
        con.execute(f"SET memory_limit = '{memory_limit_gb}GB'")
    if threads > 0:
        con.execute(f"SET threads = {threads}")
    # `preserve_insertion_order=false` per DuckDB's OOM hint — order is
    # irrelevant for our materialised tables (we group + re-sort downstream),
    # and dropping the constraint cuts JOIN intermediate memory substantially.
    con.execute("SET preserve_insertion_order = false")
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
    """Single-shot materialise (no chunking). Use this when the data fits
    in memory; otherwise call `_materialise_chunk_streams` per chunk.

    Three-step pipeline (vs the old single-CTAS-per-stream form which
    OOM'd at 100+ GB on the inline pricings aggregator):

      1. `materialise_grouped_tables`: GROUP-BY pricings/markers/cans
         once into per-article LIST tables. Each runs as its own CTAS
         so DuckDB releases the aggregator hash-table memory between
         steps. Each `raw_*` source is dropped after its grouped table
         is built — saves ~30 GB once `raw_pricings` is no longer needed.

      2. `articles`: hash-JOIN raw_offers against the 3 grouped tables
         (just lookups, no aggregator state) → with_hash → group again
         per article_hash. Memory budget = articles aggregator only.

      3. `offer_rows`: hash-JOIN raw_offers against the 3 grouped tables
         → projected → with_hash → one row per offer. Streaming output,
         minimal aggregator memory."""
    p_n, m_n, c_n = materialise_grouped_tables(con)

    log.info("Building articles table from grouped tables (hash JOIN + group_by_hash)…")
    con.execute(f"CREATE OR REPLACE TABLE articles AS {_build_articles_sql(source='raw_grouped')}")
    article_count = con.execute("SELECT count(*) FROM articles").fetchone()[0]
    log.info("  articles materialised: %d rows", article_count)

    log.info("Building offer_rows table from grouped tables (hash JOIN)…")
    con.execute(f"CREATE OR REPLACE TABLE offer_rows AS {_build_offers_sql(source='raw_grouped')}")
    offer_count = con.execute("SELECT count(*) FROM offer_rows").fetchone()[0]
    log.info("  offer_rows materialised: %d rows", offer_count)

    return article_count, offer_count


def _materialise_chunk_streams(
    con: duckdb.DuckDBPyConnection,
    *,
    n_chunks: int,
    chunk_idx: int,
    offers_glob: str,
    pricings_glob: str,
    markers_glob: str,
    cans_glob: str,
) -> tuple[int, int]:
    """Build chunk-local articles + offer_rows tables for chunk `chunk_idx`
    of `n_chunks`. Each chunk processes 1/N of the (vendor, articleNumber)
    keyspace, partitioned via `hash() % n_chunks == chunk_idx`.

    Reads source data directly from the per-collection globs (parquet via
    `read_parquet` or gzipped JSON via `read_json`). When the source is
    parquet, DuckDB pushes the chunk filter down so each chunk only
    materialises ~1/N of the rows — no full-collection load required.

    Why this exists: a single-shot materialise hits OOM in the pricings
    GROUP BY (1.2B → 159M aggregator state). Chunking reduces per-step
    GROUP BY input proportionally to ~75M per chunk at N=16.

    Each chunk overwrites the global `pricings_grouped` / `markers_grouped`
    / `cans_grouped` / `raw_offers` / `articles` / `offer_rows` tables.
    Phase 1/2 read from these and stream to Milvus before the next
    chunk's CTAS replaces them. Milvus's article_hash + (vendor, article)
    PKs absorb any duplicate writes safely (impossible by construction
    here since hash partitions are disjoint, but cheap)."""
    chunk_filter = (
        f'hash(vendorId."$binary".base64 || articleNumber) '
        f'% {n_chunks} = {chunk_idx}'
    )

    def _src(glob: str, columns_for_json: dict) -> str:
        """Render the FROM clause for either parquet (predicate pushdown)
        or JSON (must declare columns)."""
        if ".parquet" in glob:
            return f"read_parquet('{glob}')"
        # JSON path uses explicit columns — not commonly used in chunked
        # mode, but supported for parity.
        cols_sql = ", ".join(f"{k}: '{v}'" for k, v in columns_for_json.items())
        return (
            f"read_json('{glob}', format='newline_delimited', "
            f"maximum_object_size={256 * 1024 * 1024}, columns={{{cols_sql}}})"
        )

    # Fast path: detect pre-aggregated pricings_grouped parquet next to
    # the pricings source. If `pricings.parquet/` lives at <root>/, look
    # for <root>/pricings_grouped.parquet/chunk_{K:04d}.parquet — produced
    # one-time by `build_pricings_grouped.py`. Skip the runtime GROUP BY
    # entirely (saves ~50s per chunk × N chunks).
    pricings_grouped_path = None
    if ".parquet" in pricings_glob:
        p = Path(pricings_glob.split('*')[0].rstrip('/'))
        base = p if p.name == 'pricings.parquet' else (
            p.parent if p.parent.name == 'pricings.parquet' else None
        )
        if base is not None:
            cand = base.parent / 'pricings_grouped.parquet' / f'chunk_{chunk_idx:04d}.parquet'
            if cand.exists():
                pricings_grouped_path = cand

    if pricings_grouped_path is not None:
        log.info("[chunk %d/%d] loading pre-aggregated pricings_grouped from %s…",
                 chunk_idx + 1, n_chunks, pricings_grouped_path.name)
        con.execute(f"""
            CREATE OR REPLACE TABLE pricings_grouped AS
            SELECT * FROM read_parquet('{pricings_grouped_path}')
        """)
    else:
        log.info("[chunk %d/%d] building chunk-local pricings_grouped…", chunk_idx + 1, n_chunks)
        con.execute(f"""
            CREATE OR REPLACE TABLE pricings_grouped AS
            SELECT
                vendorId."$binary".base64 AS vk,
                articleNumber AS ak,
                list(struct_pack(
                    articleNumber := articleNumber,
                    vendorId := vendorId,
                    pricingDetails := pricingDetails
                )) AS pricings_list
            FROM {_src(pricings_glob, RAW_PRICING_COLUMNS)}
            WHERE {chunk_filter}
            GROUP BY vendorId."$binary".base64, articleNumber
        """)
    p_n = con.execute("SELECT count(*) FROM pricings_grouped").fetchone()[0]
    log.info("[chunk %d/%d] pricings_grouped: %d rows", chunk_idx + 1, n_chunks, p_n)

    log.info("[chunk %d/%d] building chunk-local markers_grouped…", chunk_idx + 1, n_chunks)
    con.execute(f"""
        CREATE OR REPLACE TABLE markers_grouped AS
        SELECT
            vendorId."$binary".base64 AS vk,
            articleNumber AS ak,
            list(struct_pack(
                articleNumber := articleNumber,
                vendorId := vendorId,
                coreArticleListSourceId := coreArticleListSourceId,
                coreArticleMarker := coreArticleMarker
            )) AS markers_list
        FROM {_src(markers_glob, RAW_MARKER_COLUMNS)}
        WHERE {chunk_filter}
        GROUP BY vendorId."$binary".base64, articleNumber
    """)

    log.info("[chunk %d/%d] building chunk-local cans_grouped…", chunk_idx + 1, n_chunks)
    con.execute(f"""
        CREATE OR REPLACE TABLE cans_grouped AS
        SELECT
            vendorId."$binary".base64 AS vk,
            articleNumber AS ak,
            list(struct_pack(
                articleNumber := articleNumber,
                vendorId := vendorId,
                customerArticleNumbersListVersionId := customerArticleNumbersListVersionId,
                customerArticleNumber := customerArticleNumber
            )) AS cans_list
        FROM {_src(cans_glob, RAW_CAN_COLUMNS)}
        WHERE {chunk_filter}
        GROUP BY vendorId."$binary".base64, articleNumber
    """)

    # Fast path #2: detect pre-projected offer_projected.parquet next to
    # the offers source. If found, skip raw_offers + the heavy projected
    # CTE work entirely — just load the precomputed offer-derived columns.
    offer_projected_path = None
    if ".parquet" in offers_glob:
        op = Path(offers_glob.split('*')[0].rstrip('/'))
        base = op if op.name == 'offers.parquet' else (
            op.parent if op.parent.name == 'offers.parquet' else None
        )
        if base is not None:
            cand = base.parent / 'offer_projected.parquet' / f'chunk_{chunk_idx:04d}.parquet'
            if cand.exists():
                offer_projected_path = cand

    if offer_projected_path is not None:
        log.info("[chunk %d/%d] loading pre-projected offer_projected from %s…",
                 chunk_idx + 1, n_chunks, offer_projected_path.name)
        con.execute(f"""
            CREATE OR REPLACE TABLE offer_projected AS
            SELECT * FROM read_parquet('{offer_projected_path}')
        """)
        # raw_offers still needs to exist as a table because some downstream
        # CTEs reference it indirectly (and to keep parity tests honest).
        # Build it from the chunk's offer parquet — fast.
        con.execute(f"""
            CREATE OR REPLACE TABLE raw_offers AS
            SELECT * FROM {_src(offers_glob, RAW_OFFER_COLUMNS)}
            WHERE {chunk_filter}
        """)

        log.info("[chunk %d/%d] building articles table (op_grouped path)…",
                 chunk_idx + 1, n_chunks)
        con.execute(f"CREATE OR REPLACE TABLE articles AS {_build_articles_sql(source='op_grouped')}")
        article_count = con.execute("SELECT count(*) FROM articles").fetchone()[0]
        log.info("[chunk %d/%d] articles: %d rows", chunk_idx + 1, n_chunks, article_count)

        log.info("[chunk %d/%d] building offer_rows table (op_grouped path)…",
                 chunk_idx + 1, n_chunks)
        con.execute(f"CREATE OR REPLACE TABLE offer_rows AS {_build_offers_sql(source='op_grouped')}")
        offer_count = con.execute("SELECT count(*) FROM offer_rows").fetchone()[0]
        log.info("[chunk %d/%d] offer_rows: %d rows", chunk_idx + 1, n_chunks, offer_count)
    else:
        log.info("[chunk %d/%d] loading chunk-local raw_offers…", chunk_idx + 1, n_chunks)
        con.execute(f"""
            CREATE OR REPLACE TABLE raw_offers AS
            SELECT * FROM {_src(offers_glob, RAW_OFFER_COLUMNS)}
            WHERE {chunk_filter}
        """)

        log.info("[chunk %d/%d] building articles table…", chunk_idx + 1, n_chunks)
        con.execute(f"CREATE OR REPLACE TABLE articles AS {_build_articles_sql(source='raw_grouped')}")
        article_count = con.execute("SELECT count(*) FROM articles").fetchone()[0]
        log.info("[chunk %d/%d] articles: %d rows", chunk_idx + 1, n_chunks, article_count)

        log.info("[chunk %d/%d] building offer_rows table…", chunk_idx + 1, n_chunks)
        con.execute(f"CREATE OR REPLACE TABLE offer_rows AS {_build_offers_sql(source='raw_grouped')}")
        offer_count = con.execute("SELECT count(*) FROM offer_rows").fetchone()[0]
        log.info("[chunk %d/%d] offer_rows: %d rows", chunk_idx + 1, n_chunks, offer_count)

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
    tei_batch_size: int = 4096,
    tei_concurrency: int = 8,
    duckdb_temp_dir: str | None = None,
    duckdb_temp_dir_limit_gb: int = 500,
    duckdb_memory_limit_gb: int | None = None,
    duckdb_threads: int = 0,
    n_chunks: int = 1,
    s3_region: str | None = "eu-central-1",
    # Sink — "upsert" (default, per-row) or "bulk_insert" (parquet → MinIO →
    # `do_bulk_insert`). Production-scale runs need bulk_insert; smoke runs
    # leave it on upsert for queryable-immediately semantics.
    sink_mode: str = "upsert",
    bulk_insert: BulkInsertConfig | None = None,
    # When True (and `sink_mode='bulk_insert'`), the orchestrator
    # releases both collections and drops every index BEFORE Phase 1,
    # then re-creates indexes + reloads AFTER the post-import flush.
    # This collapses per-chunk inline IndexBuilding cost into a single
    # post-import pass.
    #
    # Default False because the win only materialises at production
    # scale (1M+ rows per chunk where inline HNSW build genuinely costs
    # seconds per chunk). At smoke-test scale (≤100K rows / 1 chunk) the
    # ~50s drop + rebuild overhead is pure regression — Milvus's per-
    # chunk state-machine floor (~15s) dominates over per-chunk
    # IndexBuilding cost on small data, so removing the inline build
    # saves nothing while the rebuild adds a fixed cost. Empirical:
    # shard-0.0 (26K articles + 27K offers) total wall went 47s → 95s
    # with index_cycle=True vs False on Milvus 2.6.15.
    #
    # Recommended on for full-catalog reindexes; recommended off (the
    # default) for smoke / single-shard runs. The CLI flag is
    # `--index-cycle` to opt in.
    index_cycle: bool = False,
    # Vector-index recipe for the articles collection's `offer_embedding`
    # rebuild. Must match the index originally created by
    # `create_articles_collection.py` so the post-rebuild collection
    # behaves identically. Ignored when `index_cycle=False`.
    vector_index: str = "HNSW",
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
        memory_limit_gb=duckdb_memory_limit_gb,
        threads=duckdb_threads,
        s3_region=s3_region,
        needs_s3=needs_s3,
    )

    log.info("Loading raw collections from %s", "S3" if needs_s3 else "local")
    log.info("  offers:   %s", offers_glob)
    log.info("  pricings: %s", pricings_glob)
    log.info("  markers:  %s", markers_glob)
    log.info("  cans:     %s", cans_glob)

    duck_t0 = time.time()
    if n_chunks <= 1:
        # Single-shot: pre-load all 4 raw_* tables once, then materialise.
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
    else:
        # Chunked path: each chunk reads its 1/N partition directly from
        # parquet (predicate pushdown). Skip the global load entirely —
        # avoids the 17-min full-table copy that's pure overhead when we
        # only need 1/N of each table per chunk.
        log.info(
            "Chunked mode (n_chunks=%d): skipping global load; chunks read "
            "directly from source per-partition.",
            n_chunks,
        )

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
    # gives the empty initial state — no resume, fresh run. Note:
    # checkpoint resume is not supported with `n_chunks > 1` — chunks
    # rewrite the same articles/offer_rows tables, so a partial checkpoint
    # would point at rows from the wrong chunk. Pass an empty checkpoint
    # so per-chunk Phase 1/2 sees `rows_done=0` (no resume) without
    # crashing on subscript.
    if sink_mode == "bulk_insert" and bulk_insert is not None:
        if n_chunks > 1:
            checkpoint = {
                "articles": {"rows_done": 0, "chunks_done": 0},
                "offers":   {"rows_done": 0, "chunks_done": 0},
            }
        else:
            checkpoint = load_checkpoint(bulk_insert.checkpoint_path)
    else:
        checkpoint = None
    if checkpoint and (
        checkpoint["articles"]["rows_done"] > 0
        or checkpoint["offers"]["rows_done"] > 0
    ):
        log.info(
            "Resuming from checkpoint: articles=%d, offers=%d rows so far",
            checkpoint["articles"]["rows_done"],
            checkpoint["offers"]["rows_done"],
        )

    # Drop indexes + release before bulk_insert so each chunk's server-
    # side IndexBuilding stage becomes a no-op. The single post-flush
    # rebuild below is significantly cheaper than per-chunk index
    # maintenance, especially for the small-chunk shape (one chunk per
    # shard at the standard 1M-row chunk size means the per-chunk floor
    # is the dominant cost). Skipped on the upsert path — upsert needs
    # the loaded indexed collection to accept writes.
    if sink_mode == "bulk_insert" and index_cycle:
        drop_t0 = time.time()
        log.info("Index cycle: releasing + dropping indexes on %s and %s before bulk_insert…",
                 articles_collection, offers_collection)
        a_dropped = release_and_drop_indexes(milvus, articles_collection)
        o_dropped = release_and_drop_indexes(milvus, offers_collection)
        stats.index_drop_seconds = time.time() - drop_t0
        log.info("  dropped %d index(es) on %s, %d on %s in %.1fs",
                 len(a_dropped), articles_collection,
                 len(o_dropped), offers_collection,
                 stats.index_drop_seconds)

    # Materialise + write loop. For n_chunks==1, single shot using the
    # original `_materialise_streams`. For n_chunks > 1, partition by
    # `hash(vendor||article) % n_chunks` and run materialise + Phase 1 +
    # Phase 2 once per chunk. Each chunk's working set is 1/n_chunks of
    # the full data, fits well below the OOM threshold.
    cache_ctx = TEICache(
        tei_url=tei_url,
        redis_client=redis_client,
        tei_batch_size=tei_batch_size,
        tei_concurrency=tei_concurrency,
    )
    with cache_ctx as cache:
        for chunk_idx in range(max(1, n_chunks)):
            if n_chunks > 1:
                ca, co = _materialise_chunk_streams(
                    con, n_chunks=n_chunks, chunk_idx=chunk_idx,
                    offers_glob=offers_glob,
                    pricings_glob=pricings_glob,
                    markers_glob=markers_glob,
                    cans_glob=cans_glob,
                )
                # Per-chunk fresh checkpoint — each chunk's articles/offer_rows
                # are a fresh independent table, so rows_done must reset to 0
                # or _bulk_insert_articles would skip the new data thinking
                # the prior chunk's rows were "already done".
                checkpoint = {
                    "articles": {"rows_done": 0, "chunks_done": 0},
                    "offers":   {"rows_done": 0, "chunks_done": 0},
                }
            else:
                ca, co = _materialise_streams(con)
            stats.article_count += ca
            stats.offer_row_count += co

            log.info("Phase 1: writing %d article rows via %s (TEI batch=%d × conc=%d)…",
                     ca, sink_mode, tei_batch_size, tei_concurrency)
            if sink_mode == "upsert":
                stats.article_upsert_seconds += _upsert_articles(
                    con=con,
                    milvus=milvus,
                    collection=articles_collection,
                    cache=cache,
                    batch_size=article_batch_size,
                )
            else:
                t1, bi_a = _bulk_insert_articles(
                    con=con,
                    milvus_uri=milvus_uri,
                    collection=articles_collection,
                    cache=cache,
                    batch_size=article_batch_size,
                    cfg=bulk_insert,    # type: ignore[arg-type]
                    checkpoint=checkpoint,    # type: ignore[arg-type]
                )
                stats.article_upsert_seconds += t1
                stats.articles_bulk_insert = bi_a  # last chunk wins; sums tracked in metrics

            log.info("Phase 2: writing %d offer rows via %s (no embedding)…", co, sink_mode)
            if sink_mode == "upsert":
                stats.offer_upsert_seconds += _upsert_offers(
                    con=con,
                    milvus=milvus,
                    collection=offers_collection,
                    batch_size=offer_batch_size,
                )
            else:
                t2, bi_o = _bulk_insert_offers(
                    con=con,
                    milvus_uri=milvus_uri,
                    collection=offers_collection,
                    batch_size=offer_batch_size,
                    cfg=bulk_insert,    # type: ignore[arg-type]
                    checkpoint=checkpoint,    # type: ignore[arg-type]
                )
                stats.offer_upsert_seconds += t2
                stats.offers_bulk_insert = bi_o
        stats.tei = cache.stats

    stats.duckdb_seconds = time.time() - duck_t0

    # Flush so `get_collection_stats` reflects the bulk-loaded rows
    # immediately. Without this, operators running scripts/swing_aliases.py
    # right after the indexer hit row_count=0 (sealed-segment count
    # only) and the swing's row-count validation rejects the target.
    log.info("Flushing both collections so post-run stats reflect new rows…")
    milvus.flush(articles_collection)
    milvus.flush(offers_collection)

    # Re-apply the index recipe to both collections + reload. Symmetric
    # with the pre-Phase-1 release_and_drop_indexes — only fires when
    # the cycle was actually requested AND the bulk_insert path produced
    # data, so a partial run that crashed mid-Phase-1 still has its
    # indexes dropped (operator can re-run with --skip-index-cycle to
    # hand-rebuild, or just rerun the full cycle).
    if sink_mode == "bulk_insert" and index_cycle:
        rebuild_t0 = time.time()
        log.info("Index cycle: rebuilding indexes + reloading both collections…")
        apply_indexes_and_load(
            milvus, articles_collection,
            index_params=build_articles_index_params(milvus, vector_index),
        )
        apply_indexes_and_load(
            milvus, offers_collection,
            index_params=build_offers_index_params(milvus),
        )
        stats.index_rebuild_seconds = time.time() - rebuild_t0
        log.info("  rebuild + load complete in %.1fs", stats.index_rebuild_seconds)

    stats.total_seconds = time.time() - wall_t0
    log.info(
        "Bulk run complete: articles=%d (%.0fs) offers=%d (%.0fs) duckdb=%.0fs "
        "index_drop=%.1fs index_rebuild=%.1fs total=%.0fs",
        stats.article_count, stats.article_upsert_seconds,
        stats.offer_row_count, stats.offer_upsert_seconds,
        stats.duckdb_seconds,
        stats.index_drop_seconds, stats.index_rebuild_seconds,
        stats.total_seconds,
    )
    log.info(
        "TEI cache: hits=%d misses=%d tei_calls=%d bytes_written=%.1f MB",
        stats.tei.hits, stats.tei.misses, stats.tei.tei_calls,
        stats.tei.bytes_written / 1e6,
    )
    return stats


__all__ = ["run_bulk_indexer", "BulkRunStats"]
