# PRD: Retrieval-Oriented Default Training Configuration

## Summary

Change the default training setup from pairwise BCE on random pairs to a retrieval-oriented configuration that uses structured batches and a contrastive objective.

## Problem

The current defaults are:

- `loss_type: bce`
- `train_batching_mode: random_pairs`

This is workable for pair scoring, but it is weakly aligned with dense retrieval, where relative ranking against negatives matters more than independent binary decisions.

## Goal

Make the default training configuration better suited for dense retrieval fine-tuning, especially for models such as `microsoft/mdeberta-v3-base`.

## Non-Goals

- redesigning the full loss framework
- introducing cross-encoder training
- guaranteeing one default works best for every model family

## Requirements

1. Default config must use a retrieval-oriented loss, preferably `contrastive`.
2. Default config must use a query-structured batching mode, preferably `anchor_query` or `random_query_pool`.
3. Defaults must ensure each batch contains at least one positive and meaningful negatives for the anchor query.
4. Batch statistics should continue to expose positive, same-query-negative, and cross-query-negative counts.
5. Documentation must explain why the defaults changed and when to override them.
6. Legacy BCE training must remain available as an explicit opt-in.

## Product Decisions

1. Retrieval quality takes priority over preserving the current pairwise-classification default.
2. The default should favor stronger retrieval learning even if training becomes slightly slower.
3. The system should keep configuration simple rather than exposing many new knobs initially.

## Suggested Default Shape

- `model.loss_type: contrastive`
- `data.train_batching_mode: anchor_query`
- keep `n_pos_samples_per_query` and `n_neg_samples_per_query` configurable
- preserve `triplet` and `bce` as alternative modes

## Acceptance Criteria

1. A fresh training run uses retrieval-oriented defaults without extra overrides.
2. Batch construction consistently produces query-grouped batches suitable for contrastive learning.
3. Documentation and config files reflect the new defaults.
4. Tests cover the new default config path.

## Risks

1. Some datasets may not have enough positives per query for the new defaults.
2. Users relying on BCE behavior may need explicit config changes.
3. Contrastive training may be more sensitive to batch composition.

## Open Questions

1. Should `anchor_query` or `random_query_pool` be the default?
2. Should `triplet` remain a first-class recommended alternative or mainly diagnostic?
