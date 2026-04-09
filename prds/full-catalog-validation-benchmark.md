# PRD: Full-Catalog Validation Benchmark

## Summary

Replace the current pairwise validation proxy with validation that measures retrieval quality by ranking each validation query against the full validation catalog, or a clearly documented close approximation when exhaustive scoring is too expensive. Use that retrieval metric for model selection and checkpointing.

## Problem

The current training pipeline validates on labeled query-offer pairs and computes ranking metrics only within those labeled rows. This is not the same task as product retrieval against a full catalog.

As a result:

- validation metrics can look good while real catalog retrieval remains poor
- checkpoint selection is based on a proxy objective
- fine-tuning decisions are hard to trust for dense retrieval models such as `microsoft/mdeberta-v3-base`

## Goal

Make validation reflect the actual retrieval objective used by `embedding-catalog-benchmark`, so that training, checkpointing, and offline benchmarking are aligned.

## Non-Goals

- redesigning the training loss stack
- adding ANN-based validation as the primary validation path
- changing data labeling semantics beyond what is needed for evaluation parity

## Users

- engineers fine-tuning embedding models in this repo
- anyone comparing checkpoints against zero-shot baselines

## Success Criteria

- validation metrics correlate with `embedding-catalog-benchmark` results on the same checkpoint
- the best checkpoint chosen during training is competitive on the exhaustive benchmark
- validation output makes it obvious which metric is used for checkpoint selection

## Current State

Relevant code paths:

- training entrypoint: `src/embedding_train/train.py`
- validation metric aggregation: `src/embedding_train/model.py`
- pairwise metrics helpers: `src/embedding_train/metrics.py`
- exhaustive benchmark reference: `src/embedding_train/catalog_benchmark.py`

Current behavior:

- `validation_step` stores scores only for labeled query-offer pairs
- `on_validation_epoch_end` computes metrics over those rows
- `ModelCheckpoint` monitors `val/by_batch/exact_mrr`

This differs from the benchmark, which deduplicates queries and catalog offers and scores each query against the full catalog.

## Requirements

### Functional Requirements

1. Validation must build a deduplicated validation query set and validation catalog from the validation rows.
2. Validation must encode all validation queries and all validation offers with the same model and text rendering logic used for inference.
3. Validation must score each validation query against the full validation catalog with exact dot-product or cosine similarity.
4. Validation must compute retrieval metrics using the same label semantics as `embedding-catalog-benchmark`.
5. Validation must expose at least these metrics:
   - `val/full_catalog/mrr`
   - `val/full_catalog/ndcg_at_5`
   - `val/full_catalog/ndcg_at_10`
   - `val/full_catalog/recall_at_10`
   - `val/full_catalog/recall_at_100`
6. Training checkpoint selection must monitor a full-catalog validation metric instead of the current pairwise proxy.
7. The selected checkpoint metric must be configurable, with a sensible default.
8. Validation must support the same relevant-label configuration as the benchmark, defaulting to `Exact` for binary retrieval metrics.
9. Validation must handle repeated judgments for the same query-offer pair by keeping the highest-gain label, matching benchmark behavior.
10. Validation must clearly report how many queries were evaluated, how many were eligible, and the validation catalog size.

### Approximation Requirements

If exhaustive validation is too slow for some runs, the system may support an optional approximation mode, but:

1. exhaustive validation must remain the default for correctness
2. approximation mode must be explicitly named and never silently replace exhaustive validation
3. approximation mode must document what is approximated, such as catalog subsampling
4. checkpoint selection for serious experiments should still default to exhaustive validation

### Non-Functional Requirements

1. Validation should reuse existing benchmark logic where practical instead of duplicating metric behavior.
2. Implementation should avoid material divergence between training-time validation and `embedding-catalog-benchmark`.
3. The validation path must work on CPU, CUDA, and MPS, subject to existing model support.
4. Metric names must distinguish pairwise proxy metrics from full-catalog metrics during any migration period.
5. The implementation should keep memory usage bounded by batching query scoring against the catalog.

## Product Decisions

1. The source of truth for retrieval metric semantics should be the exhaustive benchmark behavior.
2. Full-catalog validation metrics should be logged separately from batch-aligned loss metrics.
3. Default checkpoint selection metric should be `val/full_catalog/ndcg_at_5` or another explicitly chosen catalog metric, not `val/by_batch/exact_mrr`.
4. If legacy pairwise validation metrics remain, they should be treated as diagnostic only.

## Proposed Scope

### In Scope

- add full-catalog validation data aggregation during validation epochs
- compute exhaustive validation embeddings and rankings
- log full-catalog retrieval metrics
- switch checkpoint monitoring to a full-catalog metric
- add tests covering metric parity with the benchmark logic
- document the new validation behavior

### Out of Scope

- adding online hard-negative mining
- changing default model architecture
- introducing cross-encoder reranking

## UX / CLI Expectations

Training output and logs should make it easy to answer:

- what catalog retrieval metric is used for checkpointing
- how large the validation catalog is
- whether the run used exhaustive or approximate validation

Suggested config surface:

- `trainer.validation_mode: full_catalog | pairwise_proxy`
- `trainer.validation_metric: ndcg_at_5 | mrr | recall_at_10 | recall_at_100`
- `trainer.validation_similarity: dot | cosine`
- optional approximation controls only if needed

## Acceptance Criteria

1. A training run logs full-catalog validation metrics at each validation interval.
2. Those metrics are computed by ranking each validation query against the deduplicated validation catalog.
3. Re-running `embedding-catalog-benchmark` on the same validation subset yields matching metrics within expected numerical tolerance.
4. `ModelCheckpoint` monitors a full-catalog validation metric by default.
5. Logs and checkpoint filenames identify the selected full-catalog metric.
6. Tests cover:
   - deduplication of queries and offers
   - repeated label resolution by highest gain
   - metric parity with benchmark helpers
   - checkpoint monitor wiring

## Risks

1. Validation may become substantially slower, especially with larger catalogs.
2. Memory pressure may increase if catalog embeddings or score matrices are not batched carefully.
3. More realistic validation may expose that current training defaults are weak, causing apparent metric regressions.

## Open Questions

1. Should full-catalog validation score against only the validation catalog, or against train+validation offers while only evaluating validation queries?
2. Which metric should be the default checkpoint monitor: `ndcg_at_5`, `mrr`, or another retrieval metric?
3. Do we want to keep pairwise proxy metrics after rollout for debugging, or remove them entirely?

## Milestones

1. Extract or share benchmark aggregation/scoring logic for training-time validation.
2. Implement full-catalog validation metrics in the Lightning module.
3. Switch checkpoint monitoring to the new metric.
4. Add tests and documentation.
