# PRD: Hard-Negative Mining For Retrieval Training

## Summary

Add hard-negative mining so training batches include offers that are difficult for the current model, not only random negatives or same-query labeled negatives.

## Problem

Current negative construction is limited to:

- same-query labeled negatives
- synthetic cross-query negatives sampled from other offers

These negatives are often too easy. Retrieval models usually improve when trained against confusing near-miss items.

## Goal

Improve retrieval fine-tuning by supplying harder negatives during training, especially for product search where many offers are semantically close.

## Non-Goals

- online distributed ANN infrastructure
- full reranker training
- changing relevance labels themselves

## Requirements

1. The system must support mining hard negatives from the current corpus.
2. Hard negatives must exclude known positives for the anchor query.
3. Mining should support at least one offline workflow that can be reused across training runs.
4. Training batches must be able to mix hard negatives with existing same-query or random negatives.
5. The origin of negatives should be visible in batch stats or debug output.
6. Documentation must explain how hard negatives are generated and refreshed.

## Product Decisions

1. Start with offline mining rather than fully online in-loop mining.
2. Prefer a simple reproducible workflow over a highly dynamic system.
3. Negative provenance should be explicit, such as `same_query`, `random_cross_query`, `hard_negative`.

## Candidate Approach

1. Build embeddings for offers and queries using a baseline or current checkpoint.
2. Retrieve top candidate offers per query from the catalog.
3. Remove known positives.
4. Keep the highest-ranked remaining offers as hard negatives.
5. Persist mined negatives in a parquet or sidecar dataset used by the datamodule.

## Acceptance Criteria

1. A reproducible mining workflow exists and outputs hard negatives per query.
2. Training can consume mined negatives without breaking existing modes.
3. Batch metrics or logs distinguish hard negatives from random negatives.
4. Tests cover positive exclusion and data loading of mined negatives.

## Risks

1. Poor mining quality can introduce false negatives.
2. Stale negatives may become less useful as the model improves.
3. Mining adds data pipeline complexity and compute cost.

## Open Questions

1. Should the first version mine from the full catalog or only the train split?
2. How many hard negatives per query should be stored?
3. When should mining be refreshed during iterative training?
