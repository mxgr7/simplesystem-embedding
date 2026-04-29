# cross-encoder autoresearch — research notes

Branch: `autoresearch/apr28-ce`. Audience: future me. Terse.

## Baseline (commit 3570c02, MLflow run righteous-carp-160)

`deepset/gelectra-base`, `[CLS]`-pooled Linear→4-class head, `bf16-mixed`,
batch=32, `max_pair_length=384`, AdamW lr=1e-5 wd=0.01, scheduler
`constant_with_warmup` warmup_ratio=0.33, `label_smoothing=0`,
`use_class_weights=true` (inverse-freq), `head_dropout=0.1`,
`gradient_checkpointing=false`, `compile=true`.

**Result**: val_micro_f1 = **0.8797**, val_macro_f1 = **0.6860**, VRAM 7.5 GB.

### Per-class F1 at best step (28520)
| class       | precision | recall | f1     | support |
|-------------|-----------|--------|--------|---------|
| Irrelevant  | 0.536     | 0.712  | 0.6114 | 565     |
| Complement  | 0.685     | 0.658  | 0.6712 | 746     |
| Substitute  | 0.449     | 0.609  | 0.5168 | 530     |
| Exact       | 0.964     | 0.926  | 0.9447 | 8 724   |

### Per-epoch val curve (5 val ckpts in 20 min)
| step  | micro_f1 | macro_f1 | val_loss |
|-------|----------|----------|----------|
|  6050 | 0.8513   | 0.6555   | —        |
| 12101 | 0.8090   | 0.6193   | —        |
| 18152 | 0.8712   | 0.6672   | —        |
| 24203 | 0.8666   | 0.6757   | —        |
| 28520 | 0.8797   | 0.6860   | 0.852    |

## Observations & implications

- **Program's stated baseline (0.853 / 0.654) was a 1-epoch number.** With the
  20-min budget and `compile=true` + `bf16-mixed`, we actually fit ~4.7 epochs
  and 5 val checkpoints. The model is still improving on both metrics at the
  end of the budget. → micro_f1 headroom realistically ~+0.02 (target ~0.90),
  macro_f1 headroom ~+0.06 (target ~0.75) from this 20-min-budget baseline.
- **Train/val loss gap**: train_loss_epoch=0.189 vs val_loss=0.852 by epoch 5.
  Big gap → overfitting risk grows with longer budgets. Regularization
  (dropout, label_smoothing, weight_decay) might already pay off here.
- **Substitute is the worst class (F1=0.517).** That's the lever with the
  most macro_f1 headroom. A substitute-aware loss / sampler / template lift
  would likely move macro the most.
- **Irrelevant precision = 0.536** — one in two predicted-Irrelevant pairs is
  actually a different class. Likely confusion with Complement/Substitute.
  Worth pulling a confusion matrix at some point.
- **Class weights are aggressive**: w(Irrelevant)=3.69, w(Complement)=3.48,
  w(Substitute)=4.34 vs w(Exact)=0.31. Inverse-freq weighting is strong; worth
  trying `use_class_weights=false` or a tempered schedule as one experiment.
- **Step 12101 regression** (0.809 micro): single-epoch noise, recovered by
  step 18152. Suggests val noise floor on micro_f1 is meaningful at single-epoch
  granularity — within a run, take peak across val checkpoints, not the last.
- **Non-monotonic at intermediate epochs** but monotonically improving from
  epoch 3→5 → the model wants more steps. Larger batch / shorter sequences /
  GA might help squeeze more updates into the budget.

## Keep chain

| commit  | micro  | macro  | knob                        | note                                |
|---------|--------|--------|-----------------------------|-------------------------------------|
| 3570c02 | 0.8797 | 0.6860 | baseline (gelectra-base)    | 5 val ckpts, both still climbing    |
| a1f0bd9 | 0.9088 | 0.7143 | `use_class_weights=false`   | +0.029 / +0.028 at SAME step 18152  |
| 0b0a561 | 0.9083 | 0.7204 | `lr=2e-5` (cw=off)          | macro peak +0.006 at last step 28480 |
| 7837228 | 0.9151 | 0.7298 | `gelectra-large`            | **+0.007 / +0.009**; Substitute f1 0.52→0.57; 1.9 epochs in budget |
| 47ca58f | 0.9157 | 0.7366 | `max_pair=512` (gelectra-large) | macro **+0.007** within noise on micro; peak at epoch-1 (step 6050); only 2 vals in 20min |
| 0e8893c | 0.9193 | 0.7512 | `lr=1.5e-5`                  | both lift: micro **+0.004**, macro **+0.015**; crosses 0.75 macro target; all 4 classes improve |
| c3400bf | 0.9204 | 0.7575 | `focal_loss gamma=2.0`       | macro **+0.006** within noise on micro; Substitute F1 jumps to 0.59 — exactly where focal targets |

## Strategic lessons learned

- **`use_class_weights=true` is net-harmful at bs=32.** Inverse-freq weights
  put 3.7-4.3x gradient on minority classes; at small batch (32) this makes
  minority gradients dominant but noisy and the optimizer fights itself. With
  weights off, the dataset's ~80% Exact still leaves enough minority signal
  that classes co-adapt naturally and training is more stable. **All four
  per-class F1 scores improved or stayed flat** when weights went off — even
  the rare classes (Irrelevant 0.611→0.639, Substitute 0.517→0.523). This is
  a strong default unless someone re-introduces large effective batch.
- **The val curve peaks early then drifts** under cw=off + lr=1e-5
  (peak at step 18152, decline through 24203, partial recovery at 28570).
  At lr=2e-5 the same drift on micro is gone — micro is roughly flat across
  the run (0.901-0.908) while macro **keeps climbing** all the way to step
  28480 (0.7204). So the cw=off+lr=1e-5 "drift" was undertraining-with-noise
  rather than overfitting; what looked like overfitting was the model failing
  to consolidate on minority classes within the budget.
- **lr=2e-5 only works after cw=off.** With cw=on, lr=2e-5 lost -0.010 micro
  and -0.011 macro vs lr=1e-5; the 4x effective gradient on minority classes
  exploded at the higher lr. With cw=off (effective gradient back to 1x),
  the same lr=2e-5 hits a higher macro peak. The lesson: lr and class-weight
  scale interact — never sweep lr without considering the gradient norm
  baseline you're operating at.
- **Substitute is still the worst class** but improved meaningfully:
  baseline F1 0.517 → exp5 macro 0.720 implies all classes lifted. Need to
  pull confusion matrix when convenient.
- **gelectra-large is ~3× the parameters and the right swing for capacity.**
  Got `+0.007 micro / +0.009 macro` over the lr=2e-5 base model in the same
  20-min budget, despite training only ~1.9 epochs vs ~4.7. The capacity gain
  dominated the step-count loss. Most importantly, **the Substitute F1
  jumped from ~0.52 to 0.57 (+5.4 points)** — the per-class story is exactly
  what model capacity should buy: the saturated dominant class (Exact) barely
  moved; the under-fit minority classes lifted. Macro was still climbing at
  the last val (step 11672) — a longer budget would push further.
- **`max_pair_length=512` lost on the base model but won on gelectra-large.**
  Same data hypothesis (Substitute p90=439 needs full context), but the
  step-count cost differed: base lost ~28k → ~22k steps and missed; large
  was already step-bound at 11k → 9.8k and the per-step capacity gain at
  L=512 cleared the bar. **Lesson: max_pair vs steps tradeoff depends on
  the model's sample-efficiency at that capacity.** Higher capacity, longer
  context.
- **The lr=3e-5 collapse mode is identical across model sizes.** Both
  gelectra-base and gelectra-large at lr=3e-5 + cw=off + warmup_ratio=0.33
  converge to "always predict Exact" with micro=0.826 macro=0.226. Probably
  warmup_ratio=0.33 with 6051-step "epoch" pushes lr to 3e-5 with no decay,
  and the gradient on minority classes overwhelms the head before recovery.
  Don't try lr ≥ 3e-5 again without also reducing warmup or adding decay.

## Open ideas (rough, ranked by leverage)

1. **Substitute-focused fixes.** Focal loss (γ=2) targeting hard-negative
   classes; tempered class weights (e.g. weight_pow=0.5); auxiliary contrastive
   loss between query and Substitute offers.
2. **More steps within budget.** `max_pair_length=256` (256 is the median
   tokenized length probably — need to inspect); larger batch (64, 96) since
   VRAM is at 7.5 GB and we have 80 GB to spare.
3. **Encoder swap.** `xlm-roberta-base` and `mdeberta-v3-base` are reasonable
   in size; a stronger German base might be `deepset/gelectra-large` if it
   fits. Capacity vs throughput tradeoff.
4. **Label smoothing 0.05–0.1.** Cheap regularizer; addresses overfit gap.
5. **Pooling change.** Mean-pooled or attention-pool head over `[CLS]`. Not
   obvious gain on a classification setup but cheap to try.
