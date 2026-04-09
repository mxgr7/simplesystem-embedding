# PRD: Retrieval-Safe Validation Splitting

## Summary

Make `offer_connected_component` the preferred validation splitting strategy for retrieval experiments so the model is evaluated on more realistic separation between train and validation offers.

## Problem

Simple query-id splitting can still leave overlapping offers across train and validation. That weakens validation because the model may be evaluated on offers it effectively saw during training through other queries.

## Goal

Reduce train-validation leakage for retrieval experiments by preferring splits that separate connected query-offer components.

## Non-Goals

- solving every possible leakage path in the raw data
- introducing full dataset versioning or lineage tooling

## Requirements

1. Retrieval-focused configs must prefer `offer_connected_component` over `query_id`.
2. Dataset statistics must clearly report whether offers are shared between train and validation.
3. Documentation must explain the tradeoff between split realism and split size.
4. The default should remain overridable for small or highly connected datasets.
5. Tests must cover the split behavior and shared-offer counts.

## Product Decisions

1. Retrieval realism is more important than preserving the simplest split mode as default for retrieval runs.
2. `query_id` splitting should remain available as a fallback when component splitting is too restrictive.
3. Training output should make split mode visible and easy to audit.

## Candidate Change

- change retrieval-oriented configs to default to `data.val_split_mode: offer_connected_component`
- keep `query_id` as an explicit override
- fail clearly when a dataset is so connected that the requested validation fraction empties the train split

## Acceptance Criteria

1. Retrieval configs default to `offer_connected_component`.
2. Run metadata shows split mode and shared offer counts.
3. Tests confirm that connected offers stay within one split.
4. Documentation explains when to override to `query_id`.

## Risks

1. Highly connected datasets may produce small or unstable validation splits.
2. Users may see lower validation metrics after leakage is reduced.
3. Component splitting can be harder to explain than query-id splitting.

## Open Questions

1. Should `offer_connected_component` become the global default or only the retrieval preset default?
2. Should the system warn when `query_id` split leaves shared offers between train and validation?
