# I2 — Kafka-driven incremental upserter

**Category**: Indexer (new pipeline)
**Depends on**: I1 (projection module + bulk pipeline)
**Unblocks**: production deploy (incremental updates after first hydration)

References: spec §6, §9 #8 (Bounded consistency tolerance).

## Scope

Stand up the Kafka consumer that turns MongoDB record-id change notifications into Milvus upserts using the projection module from I1. After I1's bulk run hydrates the collection, this packet keeps it fresh.

## In scope

- **Kafka consumer** (`indexer/incremental.py`):
  - Subscribe to the change-notification topic (name + cluster details from infra; document where they're configured).
  - Consumer group identity + offset commit policy (commit after successful Milvus upsert, not before).
  - Idempotent at the message level: re-delivering the same record id results in the same Milvus row.
  - Per-message flow:
    1. Fetch the current MongoDB record by id.
    2. Project via the I1 module.
    3. Upsert into Milvus (use `MilvusClient.upsert`, NOT delete-then-insert).
  - Handle deletes: if the MongoDB record has been deleted, delete the corresponding Milvus row.
- **Embedding cost control**: do not re-embed if the projected text content is unchanged. Hash the embedding inputs; skip the TEI call when the hash matches what's already in Milvus (read the row first, compare, skip or re-embed). Document the hash field added to the projection.
- **Backpressure / retries**:
  - Transient Milvus or MongoDB failures: retry with the same policy as F7 / A5 (max 5 attempts, 500ms base, 1.5× multiplier, max 5s delay).
  - Permanent failures (e.g. record no longer exists; projection rejects the row): log + send to a dead-letter sink, do NOT block the consumer.
  - Cap concurrent in-flight upserts so a backlog burst doesn't OOM the indexer.
- **Observability**: Prometheus metrics for messages consumed, upserts succeeded / failed, embed-skipped count, dead-letter rate, lag.
- Smoke / integration test: produce a fixture record-id batch onto an in-memory or test broker, observe Milvus rows materialise.

## Out of scope

- Bulk reimport — I1.
- Alias swap orchestration — I3.
- Schema migrations — I3 (the consumer assumes the schema is what F1 declared).

## Deliverables

- `indexer/incremental.py` + entry-point.
- Embedding-skip hash machinery in the projection (extends I1's module).
- Dead-letter sink (file? topic? log? — pick one and document).
- Metrics + dashboards (or at least a documented PromQL set).
- Tests covering: upsert, delete, dedupe-by-hash skip, retry-then-success, retry exhaustion → dead letter.

## Acceptance

- A produced record-id arrives in Milvus within Bounded-consistency-friendly latency (§9 #8 — seconds).
- A republished record-id with unchanged content does not re-embed.
- A produced delete removes the row from Milvus.
- Retries mask transient failures; permanent failures land in the dead letter without blocking.
- Lag metric is exposed and accurate.

## Open questions for this packet

- Topic name / cluster: confirm with infra.
- Consumer group identity: confirm naming convention so it composes with existing observability.
- Whether the consumer should serve any HTTP surface (e.g. a `/healthz`, a manual replay endpoint). Recommendation: tiny FastAPI sidecar serving healthz + a controlled replay endpoint, but only if ops asks.
