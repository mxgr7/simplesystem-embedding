# PRD: Graded Relevance Training Labels

## Summary

Align training label handling more closely with the benchmark's graded relevance model instead of treating only `Exact` as positive and everything else as negative.

## Problem

Training currently collapses labels to a binary target driven by `positive_label`, which defaults to `Exact`. That means `Substitute` and `Complement` are treated the same as `Irrelevant` during training, even though evaluation gives them non-zero gain in `nDCG`.

## Goal

Allow training to represent graded relevance or at least configurable positive/semi-positive semantics, improving alignment between optimization and evaluation.

## Non-Goals

- changing the benchmark gain definitions by default
- introducing a fully new learning-to-rank framework in the first version

## Requirements

1. Training must support configurable label semantics beyond a single positive label.
2. The system must support at least these modes:
   - `exact_only`
   - `exact_plus_substitute`
   - graded relevance
3. Data preparation and loss computation must preserve enough label information to support the chosen mode.
4. The chosen label mode must be logged in training metadata.
5. Documentation must explain how training label semantics differ from benchmark metrics.

## Product Decisions

1. Start with configurable semantics before introducing more complex ranking losses.
2. Preserve current `exact_only` behavior as an explicit mode for comparability.
3. Favor small extensions to the data model rather than a full rewrite.

## Candidate Modes

### Mode 1: Exact Only

- `Exact` positive
- everything else negative

### Mode 2: Exact Plus Substitute

- `Exact` and `Substitute` positive
- `Complement` and `Irrelevant` negative

### Mode 3: Graded

- pass graded targets or weights into the selected loss path
- exact implementation may vary by loss family

## Acceptance Criteria

1. Config supports selecting training label semantics.
2. The datamodule and renderer preserve the necessary raw label signal.
3. At least one non-binary label mode is trainable end to end.
4. Tests verify mapping behavior for `Exact`, `Substitute`, `Complement`, and `Irrelevant`.

## Risks

1. Treating substitutes as positives may improve recall while hurting exactness.
2. False assumptions about label meaning could degrade business relevance.
3. Some loss functions may need adaptation for truly graded targets.

## Open Questions

1. Is `Substitute` truly desirable as a positive retrieval target for the main product experience?
2. Should `Complement` ever contribute to retrieval training, or only to secondary evaluation?
3. Which loss types should support graded targets first?
