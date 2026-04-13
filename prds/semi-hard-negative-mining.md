# PRD: Semi-Hard Negative Mining For Retrieval Training

## Summary

Add semi-hard negative mining so training can be supplied with negatives that are harder than random but softer than the top-ranked hard negatives, avoiding the false-negative and instability issues of pure hard mining.

## Problem

The existing hard negative mining workflow keeps the highest-ranked non-positive offers per query. In practice the very top of that list tends to contain:

- unlabeled true positives that look like false negatives to the loss
- near-duplicate offers that are effectively the same item
- extremely confusing items that destabilize contrastive training

Training against only these items can stall learning or pull the model in the wrong direction. At the same time, fully random negatives are too easy to provide useful gradient.

## Goal

Let the mining workflow produce a middle tier of negatives — confusing enough to matter, but not drawn from the top of the ranked list — and let training consume them alongside positives, same-query negatives, and random negatives.

## Non-Goals

- online in-loop semi-hard selection based on current batch distances
- triplet-margin-aware selection tied to a specific loss
- replacing the existing hard negative workflow

## Requirements

1. The mining workflow must support selecting negatives from a configurable rank band rather than only the top of the ranked list.
2. Semi-hard negatives must exclude known positives for the anchor query, matching hard negative behavior.
3. Semi-hard negatives must be persisted in the same reusable sidecar format the datamodule already understands, or in a parallel file that can be loaded together with hard negatives.
4. Training batches must be able to mix semi-hard negatives with existing negative sources without breaking current modes.
5. The origin of each negative must remain visible in batch stats, with semi-hard negatives counted distinctly from hard negatives and random cross-query negatives.
6. Documentation must explain the difference between hard and semi-hard mining and when to prefer each.

## Product Decisions

1. Reuse the existing offline mining entry point rather than introducing a separate CLI where possible.
2. Prefer a simple rank-band definition (e.g., keep ranks `[start, end)` after positive exclusion) over loss-aware score thresholds for the first version.
3. Provenance must be explicit: `hard_negative` and `semi_hard_negative` are distinct labels, not merged.

## Candidate Approach

1. Extend the mining CLI with `--rank-start` and `--rank-end` arguments (or an equivalent band specification) that control which slice of retrieved candidates is kept after positive exclusion.
2. Tag each mined row with a provenance column so downstream code can distinguish hard from semi-hard rows in a single parquet, or write them to separate files.
3. Extend the datamodule to load a semi-hard sidecar alongside the hard negative sidecar, or to respect a provenance column when loading a combined file.
4. Extend batch construction and batch stats to account for the new provenance class.

## Acceptance Criteria

1. A reproducible mining invocation exists that outputs semi-hard negatives per query from a configurable rank band.
2. Training can consume semi-hard negatives on their own, together with hard negatives, or with neither, without breaking existing modes.
3. Batch metrics or logs report semi-hard negative counts separately from hard and random negatives.
4. Tests cover rank-band selection, positive exclusion for semi-hard rows, and data loading of the new provenance class.

## Risks

1. A rank band that is too shallow degenerates into hard negative mining; one that is too deep degenerates into random negatives.
2. Maintaining two parallel mined sidecars increases pipeline complexity and the chance of staleness.
3. Additional provenance classes add surface area to batch builders and their tests.

## Open Questions

1. Should semi-hard rows live in the same parquet as hard negatives with a provenance column, or in a separate file?
2. What default rank band produces useful signal for the current product search corpus?
3. Should the same mining invocation emit both hard and semi-hard rows in one pass to save compute, or should they stay as independent runs for simplicity?
