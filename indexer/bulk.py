"""F9 production bulk indexer (MongoDB Atlas snapshot → Milvus).

Top-level orchestrator that wires the DuckDB-native projection +
aggregation (`indexer.duckdb_projection`) to TEI batched embedding with
a Redis cache (`indexer.tei_cache`) and paired Milvus upserts.

Pipeline:

    raw S3 / local                DuckDB                 TEI + Redis             Milvus
    ──────────────                ──────                 ───────────             ──────
    offers/*.json.gz   ─┐
    pricings/*.json.gz ─┤  load_raw_collections
    markers/*.json.gz  ─┼─►  +  4-way JOIN on (vendorId, articleNumber)
    cans/*.json.gz     ─┘  +  per-row projection
                              +  per-hash aggregation     ┐
                                  ▼                         │
                              `articles` table  ──── batch  ┼─► embed_articles ─► upsert
                              `offer_rows` table ── batch  ─┴───────────────────► upsert

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
) -> Iterator[list[dict]]:
    """Stream a DuckDB table as batches of plain dicts. `fetchmany`
    pulls one chunk at a time so the orchestrator never holds more
    than `batch_size` rows in Python at once — important at the
    130M-article scale where holding everything would OOM Python long
    before Milvus pushed back."""
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
) -> BulkRunStats:
    """End-to-end bulk indexer entry point.

    The CLI (`scripts/indexer_bulk.py`) is a thin argparse wrapper
    around this — call this from a notebook or a parent driver if you
    want programmatic control over batch sizes."""
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

    log.info("Phase 1: upserting %d article rows (TEI batch=%d)…",
             stats.article_count, tei_batch_size)
    with TEICache(
        tei_url=tei_url,
        redis_client=redis_client,
        tei_batch_size=tei_batch_size,
    ) as cache:
        stats.article_upsert_seconds = _upsert_articles(
            con=con,
            milvus=milvus,
            collection=articles_collection,
            cache=cache,
            batch_size=article_batch_size,
        )
        stats.tei = cache.stats

    log.info("Phase 2: upserting %d offer rows (no embedding)…", stats.offer_row_count)
    stats.offer_upsert_seconds = _upsert_offers(
        con=con,
        milvus=milvus,
        collection=offers_collection,
        batch_size=offer_batch_size,
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
