# Bulk indexer operator runbook

Operating `scripts/indexer_bulk.py` — the F9 production bulk reindex
pipeline (Mongo Atlas snapshot → DuckDB → TEI/Redis → Milvus).

## Quick reference

| Concern | Where |
| --- | --- |
| Code | `indexer/{bulk.py,bulk_insert.py,duckdb_projection.py,tei_cache.py,embedding_text.py,projection.py}` |
| CLI | `scripts/indexer_bulk.py` |
| Schema setup | `scripts/create_{articles,offers}_collection.py` |
| Alias swing | `scripts/swing_aliases.py` |
| Test surface | `tests/test_indexer_bulk_smoke.py`, `tests/test_bulk_insert.py`, `tests/test_duckdb_*.py` |

## Pre-flight

The indexer assumes:

  - **Milvus** at `--milvus-uri` (default `http://localhost:19530`),
    with the target `articles_v{N}` and `offers_v{N+1}` collections
    already created. Use the per-collection scripts:
    ```
    uv run python scripts/create_articles_collection.py --version 4 --no-alias
    uv run python scripts/create_offers_collection.py   --version 5 --no-alias
    ```
  - **TEI service** at `--tei-url` (default `http://localhost:8080`),
    serving the `useful-cub-58-st` model (32-token max input — the
    indexer sends `truncate=true` automatically).
  - **Redis** at `--redis-url` (default `redis://localhost:6379/0`),
    used for the hash-keyed embedding cache. Default
    `playground-app/compose.yaml` config has RDB snapshots disabled
    (cache is reproducible from the corpus, no need for durability).
  - **Source data** at `--s3-base s3://...` or `--local-cache /path`,
    laid out as `{base}/{collection}/atlas-fkxrb3-shard-N.M.json.gz`
    where `{collection}` ∈ {`offers`, `pricings`, `coreArticleMarkers`,
    `customerArticleNumbers`}.
  - For S3 source: `AWS_PROFILE` env or instance role with read
    access to the snapshot bucket. The DuckDB `httpfs` extension uses
    `credential_chain` (boto3-style lookup).
  - For `--sink-mode bulk_insert`: a MinIO/S3 bucket reachable from
    Milvus's `proxy.httpfs` config. Default `--bulk-insert-s3-endpoint`
    (`http://localhost:9000`) + `--bulk-insert-s3-bucket` (`a-bucket`)
    match the local docker-compose Milvus.

## Sink modes

```sh
# Default: per-batch upsert via MilvusClient.upsert. Slow (~800 rows/sec)
# but per-row idempotent + queryable immediately. Right for smoke runs.
--sink-mode upsert

# Production: parquet → MinIO → utility.do_bulk_insert with chunked
# pipelining and exponential-backoff retry. ~50–100K rows/sec.
--sink-mode bulk_insert
```

The bulk_insert mode chunks the parquet output (default 1M rows per
chunk) and pipelines upload + submit through a thread pool (default 4
workers). Per-chunk parquet files are deleted immediately after upload
so peak local stage usage stays ≈ `chunk_rows × upload_workers` rows
worth of data (~400 MB for defaults).

**Recommendation**: always set `--bulk-insert-checkpoint /path/to/checkpoint.json`
on production runs. After each chunk's bulk_insert completes, the
state is atomically written; on crash, restart with the same flag and
the orchestrator skips already-ingested rows. Without it, a partial
run is wasted on failure.

### Optional index cycle (`--index-cycle`)

Off by default. When enabled, the orchestrator releases both
collections and drops every index BEFORE the first `do_bulk_insert`,
then re-creates the indexes + reloads after the post-import flush.
The recipe used for the rebuild lives in
`indexer/collection_specs.py` — same single source of truth that
`scripts/create_*_collection.py` use, so the post-rebuild collection
is byte-identical in shape to a fresh collection.

The intent is to collapse per-chunk inline IndexBuilding cost into a
single post-import pass. **Empirically (shard 0.0, ~26K articles +
~27K offers, Milvus 2.6.15), this REGRESSES the smoke-test wall time
by ~50s** — the ~12s drop + ~36s rebuild are pure overhead because
the per-chunk Milvus state-machine floor (~15s) dominates over the
inline IndexBuilding cost on small data, so removing inline
IndexBuilding saves nothing. The win only materialises when chunks
are large enough that inline HNSW build genuinely costs seconds per
chunk (production scale: 1M+ rows per chunk).

  - **When to enable**: full-catalog reindexes (e.g. against
    production S3 snapshot, ~130 chunks at 1M rows each).
  - **When to leave off**: smoke runs, single-shard pulls, anything
    where chunk count is single-digit.
  - **Side effect**: when on, collections are *not queryable* between
    Phase 1 start and the post-flush rebuild. Always run against the
    versioned `*_v{N+1}` collection (per the F9 alias workflow) so
    the live alias keeps pointing at the previous version.
  - **`--vector-index HNSW|IVF_FLAT|FLAT`** chooses the rebuild
    recipe for `articles.offer_embedding`. Must match what
    `create_articles_collection.py --vector-index <X>` originally
    created. Default `HNSW`.

### `--bulk-insert-poll-interval-s`

Default `1.0`. Polling cadence for `do_bulk_insert` state. The
smoke-run wall time was historically inflated by ~5–10s of
detect-rounding because the prior 5.0s default missed
state-transition timing. Tightening to 1.0s removes that without
meaningfully increasing RPC load on RootCoord. Bump to 5.0+ for
production runs where each chunk takes minutes — the per-chunk
detection lag is then a rounding error.

## Standard local-cache run

```sh
uv run python scripts/indexer_bulk.py \
    --local-cache ~/s3-cache \
    --offers-glob 'atlas-fkxrb3-shard-0.*.json.gz' \
    --milvus-uri http://localhost:19530 \
    --articles-collection articles_v4 \
    --offers-collection offers_v5 \
    --tei-url http://localhost:8080 \
    --redis-url redis://localhost:6379/0 \
    --sink-mode bulk_insert \
    --bulk-insert-checkpoint /tmp/f9_indexer_checkpoint.json
```

## Standard S3 run

```sh
AWS_PROFILE=simplesystem uv run python scripts/indexer_bulk.py \
    --s3-base s3://mongo-atlas-snapshot-for-lab/exported_snapshots/.../prod \
    --milvus-uri http://milvus.internal:19530 \
    --articles-collection articles_v4 \
    --offers-collection offers_v5 \
    --tei-url http://tei.internal:8080 \
    --redis-url redis://redis.internal:6379/0 \
    --sink-mode bulk_insert \
    --bulk-insert-s3-endpoint http://minio.internal:9000 \
    --bulk-insert-checkpoint /var/run/f9_indexer/checkpoint.json \
    --duckdb-temp-dir /mnt/scratch/duckdb \
    --duckdb-temp-dir-limit-gb 800
```

## Sizing

At full production scale (~510M offers → ~159M articles → ~130M unique
embeddings per the F9 dedup ratio):

| Stage | Estimate | Bottleneck |
| --- | --- | --- |
| DuckDB JOIN + project | 1–2 h | NVMe scratch I/O |
| TEI (CPU, current) | ~30 days | model throughput |
| TEI (single GPU) | ~12 h | model throughput |
| Milvus upsert | ~10 days | per-row gRPC overhead |
| Milvus bulk_insert | ~2 h | parquet upload + ingest pipeline |

`--sink-mode bulk_insert` + a GPU TEI box are required to make a full
reindex feasible in a working day. CPU TEI is fine for smoke runs.

## Tunables

```
--article-batch-size      Default 1000. Article rows per Milvus
                          upsert + TEI cache lookup. Lower if memory
                          pressure shows up; higher won't help once
                          TEI is the bottleneck.
--offer-batch-size        Default 5000. Offer rows per upsert (no
                          embedding step, can go bigger).
--tei-batch-size          Default 4096. Texts per TEI HTTP call.
                          Must be ≤ TEI server's --max-client-batch-size
                          (TEI returns 422 otherwise — symptom: every
                          embed call fails immediately at Phase 1 start).
--tei-concurrency         Default 8. Parallel TEI HTTP calls per embed
                          phase. Bound by server's --max-concurrent-requests;
                          sweet spot is typically 4-16 even when the server
                          allows more (returns diminish past the point
                          where the GPU is fully fed).
--bulk-insert-chunk-rows  Default 1M. Larger chunks = fewer pipeline-
                          overhead hits per row but worse pipelining
                          parallelism.
--bulk-insert-upload-workers
                          Default 4. ThreadPool workers for parallel
                          upload + submit. Tune up if MinIO is on a
                          10 GbE link.
--duckdb-temp-dir         Default per-connection temp. Point at a
                          large NVMe for production.
--duckdb-temp-dir-limit-gb
                          Default 500. Bump for full-catalog runs.
```

## Failure modes

### TEI 413 / "must have less than N tokens"

The model's `max_input_length` is too small for the rendered text.
The indexer already passes `truncate=true` per request — this should
be transparent. If it surfaces, the TEI service was started with
`auto_truncate=false` AND the per-request `truncate` field is being
ignored — investigate the TEI version.

### Redis "MISCONF Redis is configured to save RDB snapshots, but it's
currently unable to persist to disk"

Disk is full where Redis writes its RDB. Either free disk OR disable
RDB on the running instance:

```sh
docker exec <redis-container> redis-cli CONFIG SET save ""
docker exec <redis-container> redis-cli CONFIG SET stop-writes-on-bgsave-error no
```

The default `playground-app/compose.yaml` already disables RDB; this
only fires on environments with the legacy `save 300 1000` config.

### `pymilvus.exceptions.MilvusException: duplicate primary keys are
not allowed in the same batch`

The Atlas snapshot has duplicate `(vendorId, articleNumber)` tuples.
The DuckDB SQL already dedupes via `row_number() PARTITION BY id`, so
this should not surface from the indexer itself. If it does, check
the SQL and the source data integrity.

### Bulk-insert chunk fails after retries

`scripts/indexer_bulk.py` raises with the failed chunk's `job_id` +
the Milvus-side error message. With `--bulk-insert-checkpoint` set,
restart picks up from the last successfully-completed chunk —
duplicates in the failed chunk's window are absorbed by Milvus PK
uniqueness on subsequent re-import.

### DuckDB temp-dir full

Bump `--duckdb-temp-dir-limit-gb` or point `--duckdb-temp-dir` at a
larger volume. The 4-way JOIN at full catalog scale spills 400-800 GB
intermediates to disk.

## Resume contract

`--bulk-insert-checkpoint` writes `{articles, offers}.{rows_done,
chunks_done}` after each chunk's `do_bulk_insert` Completes. On
restart:

  - DuckDB OFFSET past `rows_done` rows for each stream.
  - Chunk numbering continues at `chunks_done` so new parquet files
    don't collide with already-uploaded ones.
  - TEI cache hits absorb the cost of re-rendering article texts for
    rows that were embedded in a prior run.

**Resume guarantee**: never drop chunks. **MAY** re-submit a chunk in
the small window where it succeeded server-side but the orchestrator
crashed before persisting the checkpoint. Milvus PK uniqueness
(`article_hash` on articles, `id` on offers) absorbs the duplicate
during compaction.

If a clean rerun is needed instead of resume, delete the checkpoint
file before restarting.

## After a successful run

Check counts and sample a row:

```sh
uv run python -c "
from pymilvus import MilvusClient
c = MilvusClient(uri='http://localhost:19530')
print('articles:', c.get_collection_stats('articles_v4'))
print('offers:  ', c.get_collection_stats('offers_v5'))
"
```

Then swing the public aliases atomically (per F9 paired-swing
protocol — articles first, offers last):

```sh
uv run python scripts/swing_aliases.py \
    --articles-target articles_v4 \
    --offers-target offers_v5 \
    --milvus-uri http://localhost:19530
```

This validates row counts + samples 200 random offer hashes against
the target articles collection (catches join-key drift cheaply) before
moving any alias.
