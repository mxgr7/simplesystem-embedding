# I3 — Zero-downtime reindex orchestration (alias swap)

**Category**: Indexer (new pipeline)
**Depends on**: F1 (alias plumbing), F9 (paired collection topology + paired alias swing), I1 (bulk rebuild), I2 (incremental — to keep the new collection fresh during the dual-write window).
**Unblocks**: production schema migrations after the first cutover.

References: spec §4.8, §6 (the bulk path is "first-time hydration, schema migrations, and zero-downtime reindex per §4.8"), F9 paired-swing protocol.

**Locked operational tunables**:
- Soak window: **7 days** (configurable via `SOAK_DAYS`).
- Dual-write window: **bulk-start → alias-swap + 24 hours**, then dual-write stops. Rollback >24h after swap requires bulk re-import (I1), not an alias flip — document this prominently in the runbook.
- New collection naming: paired **`articles_v{N}` + `offers_v{N+1}`** per F9 (N = current+1; integer alignment is part of the F9 contract).

## Status

🟡 **Partial — paired alias-swing CLI landed; full reindex orchestration deferred (depends on I2 implementation).**

Landed in commit `0ab059f`:
  - `scripts/swing_aliases.py` — single-command paired swap of `articles` + `offers` aliases per the F9 paired-swing protocol. Articles first, offers last (consumer-side last). Auto-rollback on second-swing failure. Explicit `--rollback-to` flag for post-cutover incidents.
  - Pre-flight validation: existence + row count + `--join-key-sample` random offers verify their `article_hash` resolves in the target articles collection (catches join-key drift between the two streams cheaply).
  - 5 e2e tests against live Milvus: dry-run leaves aliases untouched, happy-path swing, min-rows rejection, join-key drift rejection, explicit rollback.
  - Operator runbook in `scripts/SWING_ALIASES_RUNBOOK.md`.
  - `indexer/bulk.py` flushes both collections at end of run so `get_collection_stats` returns current counts immediately — avoids a spurious "below --min-rows" rejection when the swing follows the indexer.

Deferred (depend on I2 incremental implementation):
  - Reindex orchestration script (`scripts/indexer_reindex.py`) covering the full pipeline: create collections → run I1/F9-PR2b bulk → launch second I2 consumer for dual-write → validate → swing → drain dual-write window → drop old after soak.
  - Operator runbook for the full reindex cycle (separate doc, this packet).
  - Dual-write deployment knob (covered by I2 design — second consumer instance, not a config flag on the existing one).

## Scope

End-to-end orchestration of a paired-collection reindex with no observable disruption to ftsearch traffic. The alias mechanics are already shipped; this packet wraps them with the create → bulk → dual-write → validate → swing → drain → drop pipeline.

## Design

### Cutover model — second consumer, not a dual-write flag

The original I3 sketch proposed a dual-write knob on the I2 consumer ("also write to <new collection name>"). I2's design pass rejects this approach. Instead, dual-write is achieved by **launching a second I2 consumer instance** targeting the new pair:

```
                                    ┌──────── I2 consumer A ────► articles_v4 + offers_v5  (current/live)
   Kafka topic ─►─ portal.facts ─►──┤
                                    └──────── I2 consumer B ────► articles_v5 + offers_v6  (new/staging)
                          │                                       (started by indexer_reindex.py)
                          │
                          ▼
                  legacy article-indexer (third consumer group, runs until full cutover from legacy)
```

Each consumer has its own consumer group (`article-search-indexer-current`, `article-search-indexer-staging-v5`), so Kafka delivers every message to both. They write to disjoint Milvus collections so there's no write-side contention. After the alias swing, consumer A (now writing to a no-longer-aliased pair) gets shut down once the dual-write drain window closes.

Trade-off vs the original "config flag" approach:
  - **+** Simpler consumer code (no per-message branching, no "which collections to write to today" config).
  - **+** Honest separation: each consumer's offsets, lag, retries are independent and observable.
  - **+** Crash-blast-radius is bounded: a bug introduced in the staging consumer can't corrupt the live collection.
  - **−** 2× resource cost during the dual-write window (~24h–48h depending on bulk run + soak start).
  - **−** Slightly more deployment plumbing (k8s Deployment + ServiceAccount per consumer) vs a single config bump.

Resource cost is the right trade. A reindex is rare (schema migration cadence ≤ monthly) and the cost is bounded.

### Orchestration state machine

```
┌─────────────┐
│  init       │  parse args; sanity-check Milvus + Kafka reachable
└──────┬──────┘
       ▼
┌─────────────────────┐
│  create_collections │  run scripts/create_{articles,offers}_collection.py --no-alias
└──────┬──────────────┘
       ▼
┌──────────────────┐
│  bulk_indexer    │  run scripts/indexer_bulk.py (bulk_insert sink, with --bulk-insert-checkpoint)
└──────┬───────────┘
       ▼
┌──────────────────────────────┐
│  start_dual_write_consumer   │  k8s apply: second I2 consumer with the new pair as targets,
│                              │  consumer-group suffix tied to staging version
└──────┬───────────────────────┘
       ▼
┌─────────────────┐
│  catchup_wait   │  wait until staging-consumer lag drops below threshold (Kafka MetricsReporter)
└──────┬──────────┘
       ▼
┌─────────────────┐
│  validate       │  row counts + sampled join-key check + content-parity smoke (see below)
└──────┬──────────┘
       ▼
┌─────────────────┐
│  swing_aliases  │  delegate to scripts/swing_aliases.py with the target pair
└──────┬──────────┘
       ▼
┌──────────────────┐
│  post_swing_soak │  wait 24h for dual-write window; monitor staging-consumer lag + RED metrics on ftsearch
└──────┬───────────┘
       ▼
┌──────────────────────────────┐
│  stop_legacy_consumer        │  k8s apply: scale the now-no-longer-aliased consumer to 0
└──────┬───────────────────────┘
       ▼
┌──────────────────┐
│  long_soak       │  wait SOAK_DAYS=7 with old collections still readable for emergency rollback
└──────┬───────────┘
       ▼
┌──────────────────┐
│  drop_old        │  drop old collection pair (post 7-day soak)
└──────────────────┘
```

Each state is **idempotent** and **resumable**: the script writes a JSON state file (`/var/run/indexer_reindex_state.json`) after each transition. On crash, restart with `--resume` re-enters at the last completed state and continues. No state can be skipped.

### State details

**`create_collections`**: Wraps the existing per-collection scripts. Uses `--no-alias` because aliases stay on the old pair until swing. Versioned names (`articles_v{N}` + `offers_v{N+1}`) are derived from `--target-version N`. Idempotent: skips if collections already exist.

**`bulk_indexer`**: Wraps `scripts/indexer_bulk.py` with `--sink-mode bulk_insert` and `--bulk-insert-checkpoint`. Resumable via the existing checkpoint contract. Failure during bulk → leave state at `bulk_indexer`; on resume, the bulk script picks up from its own checkpoint.

**`start_dual_write_consumer`**: Generates a k8s manifest for the I2 consumer Deployment (or whatever the deploy target is — Nomad/ECS/etc.). Apply via `kubectl apply` (or equivalent). Waits for the Deployment to report `Ready`. Sets the `KAFKA_GROUP_SUFFIX=staging-v{N}` env so the consumer group is unique.

**`catchup_wait`**: Polls the staging consumer's lag metric (`indexer_consumer_lag` from I2's metrics endpoint, or Kafka `consumer-group describe` directly). Waits until lag < `--catchup-lag-threshold` (default 1000 messages) for 60s sustained. This proves the staging consumer is keeping up with live write traffic.

**`validate`**: Three gates, each with a clear pass/fail:

  1. **Row count**: target collections must contain ≥ `--min-row-fraction` × source-collection row count (default 0.95 — accounts for hash-dedup ratio + bulk-vs-streaming-since-bulk drift).
  2. **Join-key consistency**: same check `swing_aliases.py` runs (random sample of offers, verify each `article_hash` resolves in target articles).
  3. **Content-parity smoke**: replay a sample of `tests/test_acl_acceptance_e2e.py`-shaped queries against both old and new collections; assert top-K hits overlap ≥ `--parity-threshold` (default 0.9). Catches cases where the new bulk projection introduced a regression. The 10% slack absorbs the inevitable drift between bulk + streaming-since-bulk; tighter thresholds would block valid cutovers.

  Validation failure → exit non-zero, leave state at `validate`. Operator inspects, fixes, re-runs (validation re-runs deterministically).

**`swing_aliases`**: Delegates to `scripts/swing_aliases.py --articles-target articles_v{N} --offers-target offers_v{N+1}`. The swing script's auto-rollback on second-swing failure is preserved.

**`post_swing_soak`**: Waits 24h while watching:
  - ftsearch RED metrics (5xx rate, p99 latency).
  - Staging consumer lag (must stay < threshold).
  - Old-pair consumer lag (must keep draining at the same rate as before swing — sudden drop signals the consumer is stuck).
  - Operator can shorten with `--soak-hours N` for emergency cutovers.

**`stop_legacy_consumer`**: Scales the old consumer Deployment to 0 replicas. The old collections continue to exist but stop accumulating writes.

**`long_soak`**: 7-day wait. The script can be killed and re-run with `--resume` at any time during this window — the wait is just `time.sleep` + state-file write, no Milvus interaction.

**`drop_old`**: `MilvusClient.drop_collection` on both old collection names. Final state.

### Recovery from each step

| Step crashed mid-execution | Recovery |
| --- | --- |
| `create_collections` | Resume re-runs (idempotent) |
| `bulk_indexer` | Resume picks up from `--bulk-insert-checkpoint`; bulk script's own resume contract |
| `start_dual_write_consumer` | Resume re-applies the k8s manifest; `kubectl apply` is idempotent |
| `catchup_wait` | Resume re-polls; passes immediately if already caught up |
| `validate` | Resume re-runs all three gates (cheap) |
| `swing_aliases` | If first swing succeeded but second failed: `swing_aliases.py` already auto-rolled back. Resume re-runs the full swing pair. If both swings succeeded but the orchestrator died before writing state: resume sees aliases already on target, skips swing, continues to `post_swing_soak` |
| `post_swing_soak` | Resume re-enters wait; the 24h timer restarts (conservative — accept the cost over a less-correct partial-window count) |
| `stop_legacy_consumer` | Resume re-applies scale-to-0 (idempotent) |
| `long_soak` | Resume re-enters wait at remaining time (state file records the soak start timestamp) |
| `drop_old` | If only one collection dropped before crash, resume drops the other. Both already dropped → no-op |

### Rollback

Two distinct rollback paths depending on when the issue surfaces:

**Inside the 24h dual-write window (`post_swing_soak`)**:

  1. `swing_aliases.py --rollback-to articles_v{N-1},offers_v{N}` — flips aliases back to the prior pair.
  2. Stop the staging consumer (was preparing to become legacy after window close).
  3. Old consumer is still running and current; system is back to pre-cutover state.
  4. Investigate the failure cause; re-run reindex from scratch with the fix.

**Outside the dual-write window (>24h post-swing)**:

  1. The new collections may have diverged from any feasible "old state" — old collections stopped receiving writes 24h+ ago.
  2. Rollback requires a fresh bulk reindex onto **another** new pair (`articles_v{N+1}` + `offers_v{N+2}`) using the prior-known-good schema/projection version.
  3. Re-run reindex from scratch with the rollback target.

This is why the 24h dual-write window is a hard lock — outside it, rollback ≡ forward fix.

### CLI shape

```
scripts/indexer_reindex.py \
    --target-version 5 \
    --milvus-uri http://milvus.internal:19530 \
    --kafka-bootstrap-servers kafka.internal:9092 \
    --bulk-insert-checkpoint /var/run/reindex_v5/bulk_checkpoint.json \
    --state-file /var/run/reindex_v5/state.json \
    --soak-hours 24 \
    --catchup-lag-threshold 1000 \
    --min-row-fraction 0.95 \
    --parity-threshold 0.9 \
    --consumer-deployment-template /etc/indexer/deploy_template.yaml
```

`--resume` flag (no value) → load state file, re-enter at last completed state.

`--dry-run` flag → run validate-only (gates 1-3) against an already-prepared pair, no creates/swings.

`--target-state STATE` → run only up to state STATE and stop. Useful for staged rollouts (e.g. `--target-state validate` lets the operator inspect before swinging).

## Out of scope

- The first hydration — that's just I1/F9-PR2b + a one-time alias creation (`scripts/create_*_collection.py --alias`).
- Multi-region / multi-cluster failover.
- Automatic rollback on post-swing RED-metric anomalies. The operator decides; the script just exposes the metrics + the rollback command.

## Deliverables

- `scripts/indexer_reindex.py` — orchestration script with state machine, resume, target-state stop, dry-run.
- Per-state implementation modules in `indexer/reindex/` (one file per state for testability).
- `scripts/INDEXER_REINDEX_RUNBOOK.md` — operator-facing runbook covering the same shape as `indexer/RUNBOOK.md` and `scripts/SWING_ALIASES_RUNBOOK.md`.
- k8s/Deployment template for the staging consumer (or whatever the deploy target shape is).
- Tests:
  - `tests/test_reindex_state_machine.py` — state transitions, resume after crash at each state, invalid-state-skip rejection.
  - `tests/test_reindex_validate.py` — validation gate logic against fixture pair.
  - Smoke test: end-to-end reindex from `articles_v4` + `offers_v5` to `articles_v5` + `offers_v6` against live Milvus + Kafka, asserts post-cutover ftsearch traffic returns identical results from the new pair.

## Acceptance

- A staged reindex on a non-prod environment swings traffic with no `search-api` 5xx and no observable disruption to ACL latency.
- A forced rollback inside the 24h window (revert the alias) works without data loss on the old collection.
- Each validation gate has a clear failure mode; the script exits non-zero with a parseable error message + a documented re-run path.
- Resume after crash at each state produces the same outcome as a clean run.
- The runbook is readable cold by an operator who didn't write the script.
- Total reindex wall-clock at production scale (510M offers → 130M unique embeddings) is bounded by the bulk pipeline (~12h GPU TEI per F9 cost model) + ≤ 24h dual-write soak. The 7-day long-soak is just `sleep` time, not active work.

## Open questions for this packet

- **Deploy target plumbing**: the design assumes k8s Deployments. If the prod target is Nomad / ECS / bare metal, the `start_dual_write_consumer` and `stop_legacy_consumer` steps need a different shim. Defer the choice until ops confirms the target — abstract the consumer-launch behind a `ConsumerLifecycle` protocol with a default `KubectlConsumerLifecycle` impl.
- **State file location**: `/var/run/...` works for a long-running operator-side process (e.g. a tmux session on a bastion host). For a CI-driven reindex, the state needs to live somewhere durable across job invocations — S3 backend? Defer until the operational shape is clearer.
- **Whether ftsearch should ever be aware of an in-progress reindex** (e.g. a `/healthz` flag): no — the alias hides it entirely, per the original packet's recommendation. Keep it that way.
- **Catchup lag threshold tuning**: 1000 messages is a starting guess. Production traffic at peak may be 10K msg/s, so 1000 = ~100ms of backlog. Tune after first non-prod cutover; expose via `--catchup-lag-threshold`.
