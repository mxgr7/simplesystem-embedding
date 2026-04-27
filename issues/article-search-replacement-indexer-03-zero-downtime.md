# I3 — Zero-downtime reindex orchestration (alias swap)

**Category**: Indexer (new pipeline)
**Depends on**: F1 (alias plumbing), I1 (bulk rebuild), I2 (incremental — to keep the new collection fresh during the cutover)
**Unblocks**: production schema migrations after the first cutover

References: spec §4.8, §6 (the bulk path is "first-time hydration, schema migrations, and zero-downtime reindex per §4.8").

## Scope

The runbook + thin tooling for swinging the Milvus alias to a freshly-rebuilt collection without dropping search traffic. This is the legacy `@indexProperties.getName()` mechanism rebuilt around `MilvusClient.alter_alias`.

## In scope

- **Reindex orchestration script** (`scripts/indexer_reindex.py` or `indexer/reindex.py`):
  - Accept a target collection name (e.g. `offers_v3`).
  - Step 1: create the new collection from F1's schema script.
  - Step 2: run I1's bulk pipeline against the new collection.
  - Step 3: instruct I2 (or a parallel consumer) to also write to the new collection until the alias swap (dual-write window). Document this — it may require a config flag on the I2 consumer.
  - Step 4: validation pass — count, sampled-row spot-checks, sanity searches, optional comparison to the live collection.
  - Step 5: alias swap via `MilvusClient.alter_alias`.
  - Step 6: drain the dual-write window; confirm the old collection is no longer being written.
  - Step 7 (deferred): drop the old collection after a soak.
- **Validation gates**: each step gated on a clear pass/fail, with a documented rollback. The alias swap is reversible — keep the old collection until soak completes.
- **Operator docs**: a runbook in `INDEX_HOSTING.md` (or a new doc) covering:
  - When to use this (schema change, bad data, failed bulk).
  - Pre-flight checks.
  - Estimated time per step at production volume.
  - Rollback procedure (reverse the alias swap, drop the failed collection).
- Integration with I2: a config knob telling the consumer "also write to <new collection name>" so the new collection stays fresh while it's being validated. Keep this knob temporary; document removal after cutover.

## Out of scope

- The first hydration — that's just I1 + a one-time alias creation.
- Multi-region / multi-cluster failover.

## Deliverables

- Orchestration script with step-by-step gates.
- Operator runbook.
- Dual-write knob on the I2 consumer (small extension).
- A dry-run mode that performs validation without alias swap.

## Acceptance

- A staged reindex on a non-prod environment swings traffic with no `search-api` 5xx and no observable disruption to ACL latency.
- A forced rollback (revert the alias) works without data loss on the old collection.
- Validation gates each have a clear failure mode and a documented rollback.
- The runbook is readable cold by an operator who didn't write the script.

## Open questions for this packet

- Soak window: how long do we keep the old collection around after a successful swap? Recommendation: 7 days, configurable.
- Dual-write window: how long is the consumer expected to write to both collections? Long enough that any read still served from the old collection has a corresponding row in the new one. In practice: from "I1 bulk start" until "alias swap + Bounded-consistency window".
- Whether ftsearch should ever be aware of an in-progress reindex (e.g. health endpoint reflecting it). Recommendation: no — the alias hides it entirely.
