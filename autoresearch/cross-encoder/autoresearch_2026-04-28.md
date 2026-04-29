# cross-encoder autoresearch — handoff (apr28-ce, 2026-04-28→29)

Audience: the next autoresearch session. Build on this; don't relearn it.

Branch: `autoresearch/apr28-ce`. Champion commit: **`c3400bf`** (focal_gamma=2.0).
Per-run log: `results.tsv`. Notes: `NOTES.md`. Data
patterns: `data-insights.md`. MLflow runs in experiment
`cross-encoder-autoresearch` (id 7) renamed `NN-keep|disc-{knob}` for scanability.

## Final keep state (commit `c3400bf`)

```yaml
model:
  model_name: deepset/gelectra-large
  head_dropout: 0.1
  label_smoothing: 0.0
  focal_gamma: 2.0
  use_class_weights: false
  gradient_checkpointing: false
  compile: true
data:
  max_pair_length: 512
  batch_size: 32
optimizer:
  lr: 1.5e-5
  weight_decay: 0.01
  scheduler: constant_with_warmup
  warmup_ratio: 0.33
trainer:
  precision: bf16-mixed
  max_time: 00:00:20:00
  val_check_interval: 1.0
```

**val_micro_f1 = 0.9204, val_macro_f1 = 0.7575, peak VRAM ≈ 23 GB.**

Per-class F1 at step 6050 (the saved best step):
- Exact 0.968 / Complement 0.745 / Irrelevant 0.708 / **Substitute 0.584**

For comparison: trivial-Exact baseline ≈ 0.825 micro; Wu et al. 2022 best
single CE on English ESCI ≈ 0.759 micro; E-CARE 2025 DeBERTa-v3-large CE on
ESCI ≈ macro 0.59. We are at the top of the published cross-encoder
frontier — further gains will be hard.

## Keep chain

| commit  | micro  | macro  | knob (incremental change)             |
|---------|--------|--------|---------------------------------------|
| 3570c02 | 0.8797 | 0.6860 | data-path fix + setup                 |
| a1f0bd9 | 0.9088 | 0.7143 | `use_class_weights=false`             |
| 0b0a561 | 0.9083 | 0.7204 | `lr=2e-5`                             |
| 7837228 | 0.9151 | 0.7298 | `gelectra-base` → `gelectra-large`    |
| 47ca58f | 0.9157 | 0.7366 | `max_pair=512`                        |
| 0e8893c | 0.9193 | 0.7512 | `lr=2e-5` → `lr=1.5e-5`               |
| **c3400bf** | **0.9204** | **0.7575** | **focal_gamma=2.0**         |

Total budget consumed: 27 distinct experiments + 2 long diagnostics. The
biggest single jump was `gelectra-base → gelectra-large` (+0.007/+0.009).
The biggest macro-only jump was `lr=2e-5 → 1.5e-5` after L=512 (+0.015 macro).

## Settled findings — do NOT retest

These have been measured definitively under the keep config; resist the
temptation to re-litigate them in a 20-min budget without a fundamentally
different setup.

1. **`use_class_weights=true` is net-harmful at bs=32.** Inverse-freq weights
   give 3.7-4.3× gradient on minority classes; at small batch this is too
   noisy and the optimizer fights itself. With cw=off, all four per-class
   F1 scores improved or stayed flat. This is unambiguous (+0.029 / +0.028
   in one move).
2. **lr=3e-5 is a divergence cliff.** Both at gelectra-base and
   gelectra-large, lr=3e-5 with constant_with_warmup + warmup_ratio=0.33
   collapses to "always Exact" (micro=0.826, macro=0.226). Stay ≤ 2e-5
   unless you also redo the warmup or add aggressive decay.
3. **bs=64 hurts.** At bs=64 the model sees ~30% more examples in 20 min
   but loses 0.006 micro / 0.012 macro. The bs=32 gradient noise is
   load-bearing. Don't increase batch without compensating elsewhere.
4. **The tokenizer is not the bottleneck.** Three independent tests all lost:
   - `mdeberta-v3-base` (smaller + SentencePiece): −0.011/−0.047
   - `xlm-roberta-large` (larger + SentencePiece): −0.009/−0.036
   - `space_digits=true` on gelectra (force single-digit tokens): −0.009/−0.035
   The German-specialized pretraining of gelectra dominates any
   tokenizer-fragmentation benefit. Note: `mdeberta-v3-large` does **not
   exist on HuggingFace** — Microsoft only published v3-base for the
   multilingual line.
5. **Standard regularization slows the epoch-1 peak more than it tames
   epoch-2 overfit.** Tested: `label_smoothing=0.1`, `head_dropout=0.2`,
   `weight_decay=0.05`, `cosine + max_epochs=3`. All lost on macro by
   0.004-0.012. The "ep1 peak vs ep2 stability" tradeoff is unwinnable
   inside a 20-min budget.
6. **Lexical features concat'd to [CLS] hurt** (both raw and projected
   through Linear(3,16)+GELU). The substring feature is value=0 for ~88% of
   Substitute pairs (where query is rarely in offer.name) and dominates the
   classifier's read of the feature vector. The cross-encoder already learns
   this implicitly via attention. The KDD-Cup teams' wins from features
   came from *group/candidate-set* features, which we did not implement.
7. **EAN placeholder cleanup (`'00000000'` → blank) does nothing meaningful.**
   Removes ~6 tokens of noise on 19% of rows but micro within noise, macro
   −0.012. The model already learns to ignore the placeholder.
8. **`val_check_interval=0.5` causes a counter-intuitive macro regression.**
   Finer cadence finds a higher-micro step that is past the macro peak;
   ModelCheckpoint (which selects on micro) saves the worse-macro step. If
   you change this, also change the checkpoint criterion.
9. **`focal_gamma=3.0` overshoots the focal sweet spot.** Substitute F1
   dropped 0.59→0.55 vs γ=2 — too much focus on residual hard examples
   starves the easier classes. γ=2.0 is the local optimum on this data.

## Open ideas worth trying — ranked by expected leverage

These have NOT been tried and are expected to be in the +0.002 to +0.015
range based on literature precedent or our diagnostic data.

1. **Group/context features over the candidate list** *(highest leverage,
   substantial pipeline change)*. Wu et al. 2022 §3.2.5 reported +0.008
   micro on ESCI Task 2 from group statistics: per-query min/median/max of
   the cross-encoder's own probabilities, brand-frequency-in-group,
   "is there an Exact in this group" indicator. Substitute is contextual
   on what *else* is in the candidate set. Requires running the encoder on
   all rows of a query first, then re-running with these features. Allowed
   per the program (multi-task auxiliary heads / data pipeline rework). The
   program memory-rule constraint says inference-time features must be
   computable from (query, offer); group features over the *batch* of
   candidates qualify only if the re-ranker sees the full candidate list at
   serve time — confirm before building.
2. **R-Drop adversarial / consistency regularization**. Forward pass twice
   with different dropout masks, add KL-divergence between logits. Cited
   as a +0.5-1pt micro winner in multiple KDD-Cup ESCI top-10 papers.
   Caveat: doubles per-step compute → halves step count. **Will probably
   only help under an extended budget** (see next section).
3. **Synthetic Substitute augmentation**. For each Exact pair, mutate one
   article-number digit / dimension in the offer to produce a synthetic
   Substitute. Targets the dominant failure mode (digit near-misses, see
   `data-insights.md`). Data engineering work; risk of
   distribution shift if the perturbations don't match real Substitute
   patterns.
4. **Rationale distillation from an LLM** on hard Substitute pairs (Ahemad
   et al., COLING-Industry 2025: +2.4% ROC-AUC on three ESCI subsets).
   Highest published trick; expensive (LLM inference cost, prompt design).
5. **focal_gamma=1.5 or 1.0** to fine-search the focal optimum. We tested
   {0, 2, 3}; the curve looks parabolic with peak near 2 but a finer search
   is cheap and might give +0.001-0.003 macro. Likely at the noise floor;
   only worth running if you've replicated the keep with a different seed
   first (see methodology below).
6. **Two-stage fine-tuning**: epoch 1 at lr=2e-5 (current discard but had
   a steep slope), then epoch 2+ at lr=5e-6 for refinement. Mimics a manual
   cosine. Requires either resume-from-checkpoint with a new optimizer or a
   step-wise scheduler — non-trivial but not too bad.

## What to consider if the time-budget restriction is lifted

The 20-min budget systematically rewards configs that converge fast in
epoch 1 and overfit in epoch 2. Several of our discards lost specifically
because they reduced the ep1 peak in exchange for a flatter ep2+ trajectory.
With 40-60 min budget, these become plausible winners — but only if you
**pre-commit to the longer budget for an entire ablation set** so you
compare apples to apples (we tried a one-off 40-min resume on exp 23
and it confused the analysis).

Configs that should be re-tested at 40-60 min budget:

| 20-min discard           | reason it lost                       | why a longer budget might flip it |
|--------------------------|--------------------------------------|------------------------------------|
| `cosine + max_epochs=3`  | decay was too aggressive; lower ep1  | with max_epochs=5-6 + 40 min wall, decay matches the actual run length, lr stays high enough early but decays through ep2/3 |
| `weight_decay=0.05`      | reduced ep1 peak by 0.005            | ep2+ overfit gets meaningfully suppressed; macro might recover |
| `label_smoothing=0.1`    | macro -0.006 at ep1                  | softens overconfidence on Exact, helps minority recall under longer training |
| `head_dropout=0.2`       | net wash at 20 min                   | similar reasoning to label_smoothing |
| `lr=1e-5` + L=512        | still climbing at end of 20 min      | given another 1-2 epochs, may surpass focal=2 keep — its ep1→ep2 slope was the steepest of any "still-climbing" run |
| `R-Drop` (untried)       | doubles compute per step             | with 60 min you fit a full 2 epochs of doubled forward passes; literature says +0.5-1pt micro |
| `bs=64`                  | fewer steps per minute               | the bigger batch's smoother gradient may pay off when there are enough steps to converge |

Configs that probably **still won't win** at extended budget:
- `lr=2e-5 + focal=2` (peaks at step 9827, **confirmed** by 60-min resume diagnostic)
- `mdeberta-v3-base` / `xlm-roberta-large` — already saturated at ep1 with lower
  ceiling than gelectra; no slope at ep2 to climb further
- `space_digits=true` — distribution shift from pretraining is too large

Other extended-budget directions:
- **`max_pair_length=768`** captures p99 of pair lengths (1.5% truncation
  vs 4.7% at 512). At 20 min the throughput cost was forbidding; at 40 min
  it's worth a clean test.
- **`xlm-roberta-xl` (3.5B params)** doesn't fit a meaningful number of
  epochs at 20 min. With 60 min and gradient-checkpointing it might fit
  one full epoch. Monolingual gelectra-large should still win on this
  German-mostly data, but worth confirming.
- **Cosine + max_epochs aligned with the actual budget** — the trick is
  that `estimated_stepping_batches = max_epochs × steps_per_epoch`, so
  cosine decays over the *nominal* total. Set `max_epochs` so this matches
  your wall-clock budget for meaningful decay.

## Methodology gotchas (the next session should internalize)

1. **The ep1 peak is the peak.** Across 8 keep + discard runs we measured
   trajectories on, **every** gelectra-large + max_pair=512 run peaked at
   step 6050 (= end of epoch 1) and declined through epoch 2. The "still
   climbing at end of budget" signal turned out to be one-step noise in
   exp 23 (resumed for 40 min more and confirmed it had peaked already).
   Don't extrapolate from one-step trends.
2. **Default to 20 min for screening, but pre-commit to extended-budget
   for any regularizer / scheduler / R-Drop test.** See table above. Mixing
   budgets in the same chain is dangerous.
3. **`val_check_interval` interacts with `ModelCheckpoint(monitor='micro_f1')`
   in a non-obvious way.** Finer cadence can find a higher-micro step that
   is past the macro peak. If you change cadence, consider also tracking
   macro for the saved-checkpoint criterion.
4. **Hydra cannot parse `=` in a checkpoint path.** Lightning's saved
   checkpoint filenames contain `step=NNN-...=0.NNNN.ckpt` which break
   Hydra overrides. Workaround: symlink to a safe-name path before passing
   to `trainer.resume_from_checkpoint=`.
5. **The runtime occasionally kills long-running background bash tasks
   silently** (observed on exp 26 and exp 28, both ~30 min in, no SIGKILL
   trace, no OOM). MLflow has the partial val checkpoints but the run
   stays in `RUNNING` status forever. **Mitigation:** mark these as
   `KILLED` via the MLflow REST API after the fact; if the partial result
   is meaningfully below or above the keep, use it as a coarse signal but
   don't trust the trajectory it implies. Both of our deaths happened on
   experiments that involved code changes to `model.py` + `data.py`; could
   be worth sandboxing differently.
6. **Don't commit `results.tsv`, `NOTES.md`, or
   `data-insights.md`** — those are user-reviewed artifacts
   per the program. Same for `vram.log`, `run.log`, and the `explore_*.py`
   scripts (handy but not part of the experiment).

## Critical data insights (don't re-derive)

Full details in `data-insights.md`. The actionable ones:

- **Dataset is 80% Exact**, val split slightly more Exact-heavy
  (82.6%). Trivial-always-Exact baseline = 0.825 micro on val.
- **Substitute is 5.7% of data and the worst class.** Our model lifted its
  F1 from 0.517 → 0.584 across the full keep chain — the bulk of the macro
  gain is here.
- **Substitute pairs have the longest text** (p90 pair length = 439 tokens
  vs 385 for other classes). The truncated tail contains the exact spec
  details (dimensions, materials, model variants) that distinguish them
  from `Exact`. **`max_pair_length=512` is load-bearing at gelectra-large**.
- **65% of queries are mono-label; 59% are all-Exact.** Only ~17-20% of
  queries supply minority-class contrast within their candidate group.
  This is an argument for **group features** (#1 in the open ideas above) —
  the model has limited intra-query contrastive signal during training.
- **Naïve "predict majority class within root_category" hits accuracy
  0.873.** Of our 0.083 lift over trivial-Exact, ~0.048 is just category
  leverage; only ~0.035 is text-reading skill. The marginal text-reading
  win is what differentiates Exact from Substitute / Complement / Irrelevant.
- **Failure-mode pattern from CPU inference on the keep checkpoint**: the
  dominant Substitute→Exact failures are **digit/spec near-misses** (e.g.,
  query `M10x80` vs offer `M10x16 KLEMMHEBEL`, query `OR 126,6×3,53 N` vs
  offer with `EPDM` material). These are character-level distinctions —
  but tokenizer interventions all failed (see settled finding #4), so the
  remaining headroom likely lives in *group features* and *synthetic data
  perturbations*, not tokenizer changes.
- **Zero duplicate `(query_id, offer_id_b64)` pairs**. No label noise from
  duplicates; no need to dedupe.

## Code state at handoff

- `src/cross_encoder_train/model.py` — focal_loss code is in `compute_loss`
  (read `focal_gamma` from cfg). `forward()` is single-input (no lexical
  features). The `compile=True` path stores state_dict keys as
  `encoder._orig_mod.X` (relevant for any future inference scripts —
  `explore_failures.py` already handles this).
- `src/cross_encoder_train/data.py` — vanilla. No featurizer code remains
  on the keep branch.
- All `explore_*.py` diagnostic scripts have been deleted at handoff. They
  produced the analyses that live in `data-insights.md` —
  re-derive from the parquet if needed. Topics covered:
  - Token-length distribution by class (`max_pair_length` rationale)
  - Truncated-tail content for long Substitute pairs
  - Class distribution / per-query label structure / duplicate check
  - Field presence and template-prefix token overhead
  - Category-majority baseline (0.873 accuracy)
  - Query digit/length features and digit-overlap with offer name
  - CPU inference for confusion-matrix / qualitative failure cases
- `rename_mlflow_runs.py` (also deleted) gave MLflow runs the
  `NN-{keep|disc}-{knob}` convention. The MLflow runs themselves are
  already renamed; only re-build this if you start a new experiment chain.

## Best-deployable artifact

If you need a model **right now**: checkpoint
`checkpoints/valuable-finch-654/best-step=6051-val_cls_micro_f1=0.9204.ckpt`
(referred to internally as the focal=2 keep). Saved in bf16-mixed.

To re-train cleanly from `main`:
1. `git checkout autoresearch/apr28-ce` (or cherry-pick the 7 keep commits onto a fresh branch).
2. `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH uv run cross-encoder-train trainer.max_time=00:00:20:00 trainer.max_epochs=1000`
3. The saved best step will be at step 6050±1 with micro 0.9200±0.001 and macro 0.755±0.005 (replication noise).
