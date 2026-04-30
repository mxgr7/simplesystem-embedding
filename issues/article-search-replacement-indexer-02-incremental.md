# I2 — Kafka-driven incremental upserter

**Category**: Indexer (new pipeline)
**Depends on**: I1 (projection module + bulk pipeline), F9 (article-level dedup topology + envelope columns).
**Unblocks**: I3 full (zero-downtime reindex + dual-write window), production deploy.
**Blocked on (operational)**: Kafka infra + topic provisioning at the target environment.

References: spec §6, §9 #8 (Bounded consistency tolerance), §2.4 (sort-by-price browse staleness), F9 packet "Out of scope" (the streaming envelope constraint).

**Legacy reference** (next-gen): `article/search/indexer/application/src/main/resources/application.yml`. Confirmed values:
- Topic: **`${ENVIRONMENT_NAME}.portal.marketplace.facts.articles.changed`** (line 119).
- Dead-letter pattern: **`${ENVIRONMENT_NAME}.article.search-indexer-%s.facts.articles.failed`** (line 127).
- Consumer group: **`article-indexer`** (line 48) — propose `article-search-indexer` (or similar suffix) for the new pipeline so it composes alongside without overlapping legacy offsets. Confirm with infra before first deploy.

## Scope

Stand up the Kafka consumer that turns MongoDB record-id change notifications into Milvus upserts using the F9 two-stream topology + I1 projection module. After the bulk run hydrates the collections, this packet keeps both `articles_v{N}` and `offers_v{N+1}` fresh.

Two distinct write paths the design must reconcile:

  - **Offer rows** are written eagerly per Kafka message — idempotent upsert keyed by `id`, fast, no fan-out.
  - **Article rows + per-currency envelope** are written via async coalescing — the F9 envelope columns (`{ccy}_price_min/max`, `text_codes`, `customer_article_numbers`) aggregate over every offer that hashes to the same article. Per-event eager recompute is **not viable** at production burst rates (see "Coalescing budget" below). I2's central design choice is the buffer/flush parameters that bound the article-side write load while keeping staleness inside spec §9 #8's seconds-tolerance.

## In scope

- **Kafka consumer** (`indexer/incremental.py`):
  - Subscribe to `${ENVIRONMENT_NAME}.portal.marketplace.facts.articles.changed`.
  - Consumer group: distinct from legacy `article-indexer` (proposed `article-search-indexer`); commit offsets **after** successful Milvus offer-row upsert + dirty-article enqueue (envelope flush is decoupled — see "Coalescer crash semantics").
  - Idempotent at the message level: re-delivering the same record id results in the same Milvus row.
  - Per-message flow:
    1. Fetch the current MongoDB record by id.
    2. Project via the I1 module (`indexer.projection.project`). Compute `article_hash`.
    3. Read the existing offer row's article_hash (if present) — needed for orphan tracking when an offer migrates between hashes.
    4. Upsert offer row into `offers_v{N+1}` (`MilvusClient.upsert`, NOT delete-then-insert).
    5. Mark the new article_hash dirty (and the old one too, if the offer moved hashes).
  - Handle deletes: if the MongoDB record has been deleted, delete the corresponding offer row + mark the now-stale article_hash dirty (its envelope shrinks; may become orphaned).
- **Async envelope coalescer** (`indexer/coalescer.py`):
  - Per-hash dirty buffer in-process. Append-only set keyed by `article_hash`.
  - Flush trigger: whichever fires first
    - Time: `--flush-interval-s` (default 5s).
    - Size: `--max-dirty-hashes` (default 100_000).
    - Memory: `--max-dirty-bytes` (default 256 MB — coarse, computed from per-entry size estimate).
  - Flush procedure per dirty hash:
    1. Query `offers_v{N+1}` for all offer rows with that `article_hash` (one batched IN over the dirty set).
    2. Group by hash, run `indexer.projection.aggregate_article` over each group → article row.
    3. If the group is empty (last offer was deleted), enqueue for the orphan sweep (see "Article GC") instead of writing.
    4. For non-empty groups: check if `article_hash` already exists in `articles_v{N}` and the embedded-field tuple matches → skip TEI (cache hit), otherwise call TEI.
    5. Bulk upsert article rows.
- **Embedding cost control**: TEI cache (Redis, same backing store as the bulk pipeline) keyed by `article_hash`. The hash IS the embedded-field fingerprint, so a cache hit is correct by construction. Document the cache key prefix `tei:v{HASH_VERSION}:`.
- **Article-level garbage collection** (per F9 "Out of scope" note): periodic sweep, not refcount.
  - Cron job (separate process or scheduled task in the same container): every 24h, sample article hashes with no offer-side referrer and batch-delete from `articles_v{N}`.
  - Implementation: stream `article_hash` from `articles_v{N}`, page-by-page; for each page, query `offers_v{N+1}` for the same hashes; delete orphans. ~10 minutes at production scale (130M article rows, 100k hashes/page, ~1300 round-trips).
  - Refcount alternative was considered and rejected: a multi-writer offer-side counter would need either a transactional update or a CRDT, and Milvus offers neither. Sweep is dead-simple and the staleness window (24h until orphan removal) doesn't affect search correctness — orphan article rows are silently filtered out at query time because no offer references them.
- **Backpressure / retries**:
  - Transient Milvus or MongoDB failures: same retry policy as F7 / A5 (max 5 attempts, 500ms base, 1.5× multiplier, max 5s delay, 5s budget).
  - Permanent failures (record no longer exists; projection rejects the row; schema mismatch): send to dead-letter, do NOT block the consumer.
  - Cap concurrent in-flight upserts so a backlog burst doesn't OOM the indexer (`--max-inflight-upserts`, default 16).
- **Dead-letter sink**: Kafka topic following the legacy pattern `${ENVIRONMENT_NAME}.article.search-indexer-{group_suffix}.facts.articles.failed`. Payload: `{record_id, last_error, retry_count, original_offset, timestamp}`.
- **Observability**: Prometheus metrics on a sidecar HTTP port (9091) — see "Observability" below for the full set.
- Smoke / integration test: produce a fixture record-id batch onto an in-memory or test broker, observe Milvus rows materialise.

## Design

### Process architecture

```
                ┌─────────────────────────────────────────────┐
                │  indexer/incremental.py (single process)    │
                │                                             │
   Kafka ──►──► │  consumer loop (asyncio + aiokafka)         │
                │     ├──► project + offer-row upsert (eager) │──►── Milvus offers_v{N+1}
                │     └──► mark hash dirty (in-mem set)       │
                │                                             │
                │  flush loop (asyncio task, 5s interval)     │
                │     ├──► drain dirty set                    │
                │     ├──► batched re-aggregate per hash      │──►── Milvus articles_v{N}
                │     ├──► TEI cache lookup (Redis)           │──►── TEI (on miss only)
                │     └──► bulk upsert article rows           │
                │                                             │
                │  GC sweep (cron, separate process or APScheduler)
                │     └──► find hashes with no offer ref      │──►── delete from articles_v{N}
                │                                             │
                │  metrics endpoint (port 9091)               │
                └─────────────────────────────────────────────┘
```

Single process keeps the dirty-set in memory cheap (no IPC or Redis round-trip per dirty mark). Asyncio gives the consumer + flush loop native cooperative concurrency without a thread pool.

**Library choice**: `aiokafka` (asyncio-native, mature, fewer GIL surprises than `confluent-kafka` with thread bridges). Same `httpx.AsyncClient` for TEI as the search-api side.

### Coalescing budget — why these defaults

F9 cites the worst-case burst: **1M change events in a tight window** (vendor mass price upload). Per-event eager recompute would issue:

  - 1M offer-row upserts (cheap, single PK, ~500ns each gRPC = 500ms in batched form)
  - For each event, ~3.3 sibling offer reads (the article fan-out — 1.22 hash dedup × 2.7 offers/article average) → 3.3M read round-trips
  - 1M article-row upserts with re-aggregation
  - = ~5M Milvus operations, ~30 minutes at sustained ~3k ops/sec, blocking the pipeline.

With coalescing at `--flush-interval-s 5` + `--max-dirty-hashes 100_000`:

  - Same 1M events → 1M offer upserts (unavoidable, but cheap)
  - Dirty set unique hashes ≤ unique article count in the burst. At 1.22× dedup, 1M events touch ~820K unique articles. Two flush windows of 100K + smaller tail.
  - Per flush window: 100K hash IN-clause for offer rows (~430ms p95 by F9's IN-clause benchmark, well below `PATH_B_HASH_LIMIT=25k` because we batch in 25k chunks → 4 round-trips per flush) + 100K aggregate computes (in-process, ~5s for ~3M offer rows) + 100K article-row bulk upsert (~1s).
  - Total: ~10 windows × 7s = ~70s end-to-end vs 30 min eager. **~25× speedup**, search staleness bounded at ~70s for the worst-case burst (well within Bounded-consistency tolerance).

| Knob | Default | Why this default |
| --- | --- | --- |
| `--flush-interval-s` | 5 | Spec §9 #8 says "seconds" of staleness; 5s leaves room for one flush + ingest visible-by-Bounded latency. |
| `--max-dirty-hashes` | 100_000 | Each hash → ~3.3 offer reads → ~330k Milvus reads per flush; 4× chunks of 25k stays within `PATH_B_HASH_LIMIT`. |
| `--max-dirty-bytes` | 256 MB | Hard cap before flush regardless of count; protects from a pathological burst with very long `text_codes`. |
| `--max-inflight-upserts` | 16 | Cap on parallel offer-row upserts; tuning floor based on Milvus client connection-pool default. |
| `--gc-interval-h` | 24 | Orphan articles are search-invisible (no offers reference them), so 24h staleness is correctness-irrelevant; cheaper than hourly. |

All three flush triggers fire whichever-first; this gives bounded staleness AND bounded memory AND bounded round-trips at flush time.

### Coalescer crash semantics

Dirty buffer is **in-memory only by design**. Trade-off:

  - **Loss case**: process crashes between Kafka commit and the next envelope flush → up to 5s of dirty-hash marks lost. Article-side envelope columns drift from offer-side reality until the next bulk reindex.
  - **Why not WAL**: a Redis-backed WAL adds a per-message round-trip (offer rows are eagerly written; we'd need to write a Redis entry per event), eliminating the asyncio-only deployment simplicity.
  - **Why this is OK**: per F9 §2.4, the system already runs envelope-stale between bulk reindexes (daily cadence). I2 narrows the staleness from ~24h to ~5s in the no-crash path; a process crash returns to the F9 baseline for one bulk-cycle window. Search remains correct (offer rows are current); only **sort-by-price browse without queryString** is affected (F4 sort path; queryString-based searches don't use envelope).
  - **Recovery in extremis**: if a crash window matters, operator can trigger a partial bulk reindex (filter by `last_modified > T-window`) to recompute envelopes for recently-changed articles. Document the procedure as a runbook addendum.

If staleness during crash windows becomes a real problem post-deploy, add a Redis-backed WAL as a follow-up — the dirty-set entries are tiny (just `article_hash` strings), so the WAL cost is bounded.

### Article-hash migration handling

When an offer's projected `article_hash` changes (e.g. vendor renames a product, ean changes), the consumer must:

  1. Read the offer row's old `article_hash` (one query before the upsert).
  2. Upsert the offer row with the new `article_hash`.
  3. Mark **both** old and new hashes as dirty.
  4. The next flush re-aggregates both:
     - Old hash → either still has remaining offers (envelope tightens) or empty group (queued for GC).
     - New hash → either already exists (envelope grows + cache-hit on TEI) or new article row (cache miss → TEI call).

This is the same code path as a normal offer change; the only addition is the pre-upsert read for the old hash. Cost: +1 Milvus query per consumed message (~500ns gRPC, well within budget).

### Schema observation: write-once vs upsert

Per F9, `articles_v{N}` is keyed by `article_hash`. Two writers (the bulk indexer and the streaming flush) can race on the same hash. Milvus `upsert` is last-writer-wins on collision; this is correct because both writers compute the **same** content for the same hash (the embedded fields are by definition the hash key). The TEI vector is also identical (TEI is deterministic given the same input, modulo numerical noise; both call the same model). So upsert collisions are no-ops by content.

The envelope columns DO change between writes — but that's the whole point of the streaming flush. The bulk writer establishes the floor; streaming maintains it.

### Cutover dual-write (I3 dependency)

I2 **does not own** the cutover orchestration; that's I3. But I2 must run cleanly alongside the legacy indexer during the dual-write window:

  - Different consumer group → different offsets, both consume every message.
  - Different target collections (legacy `offers` + `offers_codes` vs new `articles_v{N}` + `offers_v{N+1}`) → no write conflicts.
  - I3 alias-swing playbook drains legacy after the swing soaks (per `MILVUS_ALIAS_WORKFLOW.md`).

The only cross-version requirement: both indexers must be running on the same Kafka topic + DLT pattern, otherwise messages are missed. Document as a deployment precondition for I3.

## Out of scope

- Bulk reimport — I1 / F9 PR2b.
- Alias swap orchestration — I3.
- Schema migrations — I3 (the consumer assumes the schema is what F1/F9 declared).
- WAL for the dirty buffer (see "Coalescer crash semantics" — added later only if profiling shows the need).
- Cross-region replication.

## Deliverables

- `indexer/incremental.py` — async Kafka consumer + flush loop + offer/article write paths.
- `indexer/coalescer.py` — `DirtyHashBuffer` class (size + memory + time-trigger flush) and the per-flush re-aggregation routine. Pure async, no Kafka coupling.
- `indexer/orphan_sweep.py` — periodic GC sweep for article rows with no offer referrer.
- `scripts/indexer_incremental.py` — CLI entry-point with all knobs from "Coalescing budget".
- `Dockerfile` for the incremental indexer (mirror of search-api/Dockerfile shape).
- `playground-app/compose.yaml` service entry for local development against a Kafka broker (Redpanda or Kafka in compose).
- Tests:
  - `tests/test_coalescer.py` — unit tests for dirty buffer flush triggers (size, time, memory), hash-migration two-mark behavior.
  - `tests/test_incremental_consumer.py` — integration tests with `aiokafka.helpers.create_ssl_context` mocked + in-process Milvus stub for happy path, dead-letter, retry-then-success, embed-skip-on-cache-hit.
  - `tests/test_orphan_sweep.py` — fixture pair with intentional orphan article rows, sweep removes them.
  - `tests/test_incremental_smoke.py` — Redpanda compose service + live Milvus, end-to-end produce → upsert → envelope flush observable in `articles_v{N}` within 5s.

## Acceptance

- Happy path: a produced record-id arrives in `offers_v{N+1}` within ~1s; the corresponding article row's envelope is updated within `--flush-interval-s` (5s default).
- Embed skip: a republished record-id with unchanged embedded fields does NOT re-call TEI (verify via TEI mock invocation count).
- Delete: a produced delete removes the offer row immediately + marks the article hash dirty; the flush either updates the envelope (if other offers remain) or queues the article for GC (if no offers remain).
- Hash migration: an offer whose embedded fields change writes a new article row + updates the offer's `article_hash` link + marks the old hash dirty; old article eventually GC'd.
- Coalescing under burst: synthetic 100K-event burst over 50K unique articles produces ≤ 50K article-row upserts (one per unique dirty hash), measured by Milvus write metrics.
- Retries mask transient failures; permanent failures land in the dead letter without blocking the consumer.
- Lag metric is exposed and accurate (within one fetch interval).
- Crash recovery: kill the consumer mid-flush, restart; offer rows up to the last commit are present; dirty-set entries from the killed flush are absent from the envelope but eventually catch up via the next bulk reindex.
- GC sweep removes orphan article rows within 24h of the last offer being deleted.

## Observability

Prometheus metrics on `:9091/metrics`:

| Metric | Type | Labels | Why |
| --- | --- | --- | --- |
| `indexer_messages_consumed_total` | Counter | `topic`, `partition` | Throughput + per-partition fanout |
| `indexer_consumer_lag` | Gauge | `topic`, `partition` | Alerting when lag > threshold |
| `indexer_offer_upsert_duration_seconds` | Histogram | — | Per-message Milvus latency |
| `indexer_offer_upsert_errors_total` | Counter | `error_kind` | Transient/permanent split |
| `indexer_dirty_hashes_buffered` | Gauge | — | Coalescer pressure |
| `indexer_dirty_bytes_buffered` | Gauge | — | Memory pressure |
| `indexer_envelope_flush_duration_seconds` | Histogram | `trigger` (time/size/memory) | Per-flush cost |
| `indexer_envelope_flushes_total` | Counter | `outcome` | Success/failure rate |
| `indexer_envelope_articles_written_total` | Counter | — | Article-side write throughput |
| `indexer_embed_cache_hits_total` | Counter | — | TEI cache effectiveness |
| `indexer_embed_calls_total` | Counter | — | Actual TEI invocation count |
| `indexer_dead_letter_total` | Counter | `reason` | DLT volume + cause |
| `indexer_orphan_articles_swept_total` | Counter | — | GC effectiveness |
| `indexer_orphan_sweep_duration_seconds` | Histogram | — | GC cost |
| `indexer_hash_migrations_total` | Counter | — | How often offers move between articles (signal for embedded-field churn) |

Suggested dashboards:
  - Consumer health: lag + messages/s + offer-upsert error rate.
  - Coalescer health: dirty buffer size + flush duration + articles-per-flush.
  - Embedding cost: cache hit rate (target ≥ 80% in steady state, lower during catalog churn).
  - GC: weekly sweep duration + orphans removed.

Alerts:
  - Consumer lag > 5min sustained.
  - Dirty buffer > 80K entries for > 1min (approaching `--max-dirty-hashes`).
  - DLT rate > 10/min sustained.
  - Envelope flush duration > 30s p95.

## Open questions for this packet

- **HTTP surface**: a tiny FastAPI sidecar serving `/healthz` + `/metrics` + a manual replay endpoint? Recommendation: yes for `/healthz` + `/metrics` (k8s readiness, prom scrape); skip the manual replay until ops asks (replay implies operator-facing semantics that need their own doc).
- **MongoDB read on every message**: §6 says "fetches the current MongoDB record from id." At burst rate, this is potentially 1M MongoDB reads/min. Consider two optimisations once measured: (a) batch MongoDB reads at the consumer level (Mongo `find({_id: {$in: [...]}}` for the next N messages), (b) skip the MongoDB read when the Kafka payload already carries the projected fields. Per the legacy yml, the topic carries record IDs only; (a) is the realistic path. Defer until profile shows it.
- **GC sweep concurrency**: should the sweep pause while the bulk reindexer runs? Bulk writers and sweep writers don't conflict at the row level (sweep deletes are by `article_hash IN [...]`; bulk writes are by upsert), but both compete for Milvus capacity. Recommend a `--gc-pause-on-bulk-flag` env flag that sweep checks (set by I3's reindex orchestrator).
- **Hash version bump**: when `HASH_VERSION` changes (rare; embedded-field set evolves), every existing article hash invalidates. The right path is a full bulk rebuild via the alias-swing playbook, not a streaming migration — the streaming consumer would generate ~130M new hashes + ~130M old orphans, equivalent to a full reindex but slower and with a multi-day-degraded window. Document as a deployment precondition: bump `HASH_VERSION` only as part of an I3 reindex cycle.
