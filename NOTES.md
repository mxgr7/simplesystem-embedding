# Next experiments

Context: `model.triplet_negative_selection` now toggles between `semi_hard`
(default) and `hardest` in `in_batch_triplet_loss`. We need to validate that
the new default is actually better before committing to it for the next
production run, and understand how it interacts with batch size, LR, and the
offline-mined semi-hard pool.

## 1. Baseline vs semi-hard triplet (A/B)

Goal: confirm the new default beats the old behavior on the same budget.

Run two matched runs, identical seed, config, data, steps:

```bash
embedding-train model.loss_type=triplet model.triplet_negative_selection=hardest
embedding-train model.loss_type=triplet model.triplet_negative_selection=semi_hard
```

Compare at matched `records_seen`:
- `val/ndcg_at_5`, `val/exact_mrr`, `val/exact_recall_at_10`
- `train/loss` trajectory shape (semi-hard should be smoother, lower variance)
- Collapse check: any run where `train/loss` drops near 0 in the first epoch
  while val metrics stagnate is the failure mode semi-hard is meant to avoid —
  expect to see it in `hardest`, not in `semi_hard`.

Decision rule: keep `semi_hard` as default if it matches or beats `hardest` on
val; revert otherwise.

## 2. Semi-hard vs contrastive on the same data

Goal: sanity-check that the semi-hard triplet is in the same league as the
current `contrastive` default. If it isn't, the toggle work is moot.

```bash
embedding-train model.loss_type=contrastive
embedding-train model.loss_type=triplet model.triplet_negative_selection=semi_hard
```

Match LR, batch size, steps, seed. Report val metrics side by side. Contrastive
has been the workhorse, so treat it as the bar to clear.

## 3. Batch-size sensitivity for semi-hard

Goal: check whether semi-hard's reliance on "enough candidates below positive"
introduces a floor on useful batch size.

Sweep batch size at fixed steps and LR:

```bash
embedding-train model.loss_type=triplet data.batch_size=32
embedding-train model.loss_type=triplet data.batch_size=64
embedding-train model.loss_type=triplet data.batch_size=128
```

Log and inspect:
- `train/by_batch/batch_semi_hard_negative_share` — how often the mined pool
  is actually drawn from
- Frequency of the semi-hard fallback path (all negatives above positive).
  **Gap:** we currently don't log this. If this experiment matters, add a
  `triplet_semi_hard_fallback_share` metric in `in_batch_triplet_loss` (count
  rows where `has_semi_hard` is False, divide by `valid_rows.sum()`) before
  running the sweep.

Hypothesis: fallback share drops monotonically with batch size. If it's >30%
at the chosen production batch size, semi-hard is effectively acting like
`hardest` for a meaningful slice of the batch.

## 4. LR sensitivity

Goal: does semi-hard need a different LR than contrastive/hardest?

Small sweep on top of the winner from experiment 1:

```bash
embedding-train model.loss_type=triplet optimizer.lr=1.0e-5
embedding-train model.loss_type=triplet optimizer.lr=2.0e-5   # current default
embedding-train model.loss_type=triplet optimizer.lr=4.0e-5
```

Watch for plateau in `train/loss`: if semi-hard plateaus earlier at the
default LR, the 4e-5 run should pull ahead.

## 5. Interaction with offline-mined (semi-)hard negatives

Goal: the offline mining pipeline (`embedding-mine-hard-negatives`) produces
both hard and semi-hard rows, and `AnchorQueryBatchBuilder` already seeds
batches with them. In-batch semi-hard triplet selection is a second filter on
top of that. Do they compose or fight?

Four-cell matrix:

| mined negatives | in-batch selection | expected behavior |
|---|---|---|
| none              | `hardest`   | legacy |
| none              | `semi_hard` | current change, unaugmented |
| `hard_negative`   | `semi_hard` | hard pool + softened filter |
| `semi_hard_negative` | `semi_hard` | full semi-hard stack |

Run each with matched steps/batch/LR. Key question: does the offline
semi-hard pool already provide enough of the stability benefit that the
in-batch filter stops mattering? If yes, we could simplify back to `hardest`
in-batch selection when a mined semi-hard pool is configured.

## 6. Strict FaceNet band (stretch)

Goal: the current implementation uses the loose `sim < pos_sim` definition.
FaceNet's strict band is `pos_sim - margin < sim < pos_sim`. Try it as an
opt-in if experiments 1–5 show semi-hard winning clearly.

Implementation sketch: extend `negative_selection` with a `facenet_band` mode
that ANDs the existing `below_positive_mask` with `sim > pos_sim - margin`,
keeping the same pool-hardest fallback. Not worth the complexity unless the
loose variant already looks promising.

## 7. Follow-ups from `adorable-mole-653` (contrastive, 10-epoch SHN)

Context: `adorable-mole-653` (run_id `e7809027a96b47e2bc33fda1778d9cb8`) is the
current SOTA on the full-catalog eval — MRR 0.816, nDCG@10 0.740, Recall@10
0.931. It's `intfloat/multilingual-e5-base` + contrastive loss, bs=256,
lr=1e-5, warmup_ratio=0.33 per-epoch, 10 epochs, with SHN pool
`semi_hard_negatives-placid-snake-749.parquet`. Two observations from its
curves drive the next ideas:

- Val metrics peaked at epoch ~7.5 (MRR 0.819) and the last three evals
  oscillate slightly below the peak. Train loss ticked up on the final epoch
  (0.590 → 0.611). Mild late-epoch plateau / hint of overfit.
- `batch_semi_hard_negative_share ≈ 0.040` — only ~10 of ~246 in-batch
  negatives per step come from the curated SHN pool. The lift over
  `secretive-squid-635` (same SHN file, 6 epochs, MRR 0.761) looks like it
  comes mostly from the longer schedule + per-epoch warmup, *not* from the
  mined pool doing heavy lifting at its current ratio.

### 7a. Shorter schedule at matched quality

Hypothesis: 8 epochs captures ~99% of the 10-epoch quality at 20% less
compute. Re-run adorable-mole-653's config with `trainer.max_epochs=8` and
compare peak + final val metrics. If the gap is <0.005 MRR, make 8 the new
default and stop burning the extra two epochs.

### 7b. Warmup ratio sweep

`warmup_ratio=0.33` (per-epoch, set in `7eb28a2`) is the suspected lever
behind the jump from `secretive-squid-635` (0.761) to `adorable-mole-653`
(0.816), but we've only tested one value. Sweep {0.1, 0.2, 0.33, 0.5} at
fixed 8 epochs / bs=256 / lr=1e-5. Watch early-epoch val curves — the point
is whether a shorter warmup gets us to the same peak faster, or a longer one
delays the plateau.

### 7c. Re-mine SHNs from the current checkpoint

The SHN pool was mined from `placid-snake-749` (MRR 0.760), which is ~6 MRR
points behind the current encoder. That pool's "semi-hard" rows are almost
certainly *easy* for adorable-mole-653 — which would explain why the in-batch
SHN share is so low and why the contribution of the mined pool looks muted.

Action: run `embedding-mine-hard-negatives` with adorable-mole-653's
checkpoint as the scoring model, produce a fresh
`semi_hard_negatives-adorable-mole-653.parquet`, and re-train with the new
pool at the 8-epoch schedule. This is the single change I'd expect to move
the needle most.

### 7d. Raise the in-batch SHN share

Orthogonal to 7c: `data.n_neg_samples_per_query=2` means each query seeds at
most 2 curated SHN rows per batch. Try `n_neg_samples_per_query ∈ {4, 8}`
and confirm the `batch_semi_hard_negative_share_epoch` metric actually rises
(it's log-gated on `data.log_batch_stats`). Only worth running *after* 7c —
flooding stale negatives is pointless.

### 7e. Light regularization pass (only if 7a–7d don't resolve the plateau)

The final-epoch train-loss uptick is small but real. If shorter schedule +
fresh SHNs don't already flatten it, try one of: `optimizer.weight_decay=0.1`
(up from 0.01), or `optimizer.lr=5e-6` for the last 2 epochs via
`override_lr_on_resume`. Don't do both at once. Low priority — the plateau
is ~0.003 MRR, which is inside run-to-run noise.

### Order for these

1. 7c (re-mine SHNs) — biggest expected lift, unblocks 7d.
2. 7a (8 vs 10 epochs) — cheap, immediately actionable, reduces the cost of
   every subsequent experiment.
3. 7d (SHN share) on top of 7c's pool.
4. 7b (warmup sweep) once the data-side levers are settled.
5. 7e only if a plateau is still visible after the above.

## Logging / tooling gaps to close first

- Add `triplet_semi_hard_fallback_share` to the loss so experiment 3 can be
  evaluated without re-running with instrumentation.
- Consider logging the mean gap `pos_sim - selected_neg_sim` per batch — a
  direct signal for how aggressively the filter is biting. Also cheap.

## Order of operations

1. Add the fallback-share metric (prerequisite for 3 and honest analysis of 1).
2. Experiment 1 (A/B on the toggle) — this is the call on whether the default
   change was correct. Block the merge/release on it if needed.
3. Experiment 2 (sanity vs contrastive).
4. Experiment 5 (composition with mined pool) — informs whether we keep both
   layers or simplify.
5. Experiments 3 and 4 only if 1 is inconclusive or we see the failure modes
   they're designed to catch.
6. Experiment 6 only if there's a clear reason to tighten further.
