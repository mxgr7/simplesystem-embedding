# PRD: Model-Specific Text Templates

## Summary

Support model-specific query and offer templates so retrieval models such as E5 can use their required prefixes without forcing the same formatting onto DeBERTa and other backbones.

## Problem

The repo currently uses E5-style templates as defaults:

- `query: ...`
- `passage: ...`

Those prefixes are appropriate for E5, but they should not be assumed to be optimal for DeBERTa fine-tuning.

## Goal

Make text rendering behavior explicit and model-aware, while keeping overrides simple.

## Non-Goals

- automatic prompt search
- maintaining handcrafted templates for every public encoder on Hugging Face

## Requirements

1. The system must support different default templates for different model families.
2. E5 models must continue to use E5-compatible prefixes by default.
3. DeBERTa runs must be able to use neutral templates by default, without `query:` and `passage:` prefixes unless explicitly requested.
4. Explicit CLI or config overrides must continue to win over model-derived defaults.
5. Documentation must explain template behavior and model-specific expectations.
6. Tests must cover template resolution for at least E5 and DeBERTa.

## Product Decisions

1. Template selection should be deterministic and easy to inspect.
2. Model-specific defaults should be based on simple family detection or named presets, not hidden heuristics.
3. Neutral templates should remain available for experimentation.

## Candidate Approach

Introduce a small template preset layer such as:

- `template_preset: auto | e5 | neutral`

Where:

- `auto` chooses based on model name
- `e5` applies `query:` and `passage:` conventions
- `neutral` uses plain query and offer text

## Acceptance Criteria

1. Training and inference resolve model-appropriate templates by default.
2. DeBERTa runs do not inherit E5 prefixes unless explicitly configured.
3. Existing E5 workflows continue to behave as before.
4. Tests verify preset resolution and override precedence.

## Risks

1. Incorrect model-family detection could silently degrade quality.
2. More configuration surface may confuse users unless documented clearly.

## Open Questions

1. Should model-specific template logic live in config loading or rendering?
2. Do we want explicit presets only, or an `auto` mode as the default?
