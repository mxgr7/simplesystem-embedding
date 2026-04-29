# cross-encoder baselines on ESCI-merged dataset (2026-04-29)

Initial baselines after switching from `queries_offers_labeled.parquet` to
`queries_offers_esci/queries_offers_merged_labeled.parquet`. Establishes the
floor and ceiling for the next ablation chain on this dataset.

Pipeline changes shipped in commit `9034a24`:
- `data.path` repointed at the new parquet directory
- `column_mapping.offer_id` remapped (`offer_id_b64` → `offer_id`)
- New opt-in `data.split_column` honors pre-assigned `train` / `val` / `test`
  rows; `test` is excluded from training and val loops
- `_split_records_by_assignment` added to `CrossEncoderDataModule`

The model code is unchanged. All baselines below use the same gelectra-large
keep config (lr=1.5e-5, focal_gamma=2.0, max_pair=512, bs=32, bf16-mixed,
constant_with_warmup, warmup_ratio=0.33, head_dropout=0.1).

## Dataset shape (vs old)

| | Old | New | Δ |
|---|---|---|---|
| Total rows | 204,182 | 775,728 (775,753 − 25 null labels) | 3.80× |
| Train rows | random ~95% | 624,501 (pre-assigned) | 3.06× |
| Val rows | random ~5% | 76,048 (pre-assigned) | 7.4× |
| Test rows | — | 75,179 (held out) | new |
| Unique queries | 21,083 | 19,944 | −5% |
| Unique offers | 179,415 | 740,143 | 4.12× |
| Query overlap with old | — | 0 / 21,083 | disjoint |
| Median candidates / query | 7 | 30 | 4.3× |
| Mono-label queries | 64.9% | 10.9% | −54pp |

**Label distribution** (val):

| Class | Old val | New val | Δpp |
|---|---|---|---|
| Exact | 82.6% | 51.8% | −30.8 |
| Substitute | 5.7% | 12.7% | +7.0 |
| Complement | 7.0% | 1.8% | −5.2 |
| Irrelevant | 4.7% | 33.7% | +29.0 |

Trivial-Exact baseline drops from 0.825 micro to 0.518 micro.

## Baselines (new val, 76,048 rows)

| # | Baseline | micro F1 | macro F1 | Exact | Substitute | Complement | Irrelevant | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | Always-Exact | 0.5175 | 0.1705 | 0.682 | 0.000 | 0.000 | 0.000 | sanity floor |
| 2 | Category-majority | 0.6102 | 0.3223 | 0.702 | 0.009 | 0.001 | 0.576 | 5,638 distinct root_categories on train; 99.7% val coverage |
| 3 | Zero-shot transfer (old keep) | 0.7315 | 0.6214 | 0.833 | 0.501 | 0.511 | 0.641 | `valuable-finch-654/best-step=6051` evaluated on new val |
| 4 | **Naive retrain (1 epoch)** | **0.8851** | **0.7802** | **0.927** | **0.712** | **0.585** | **0.896** | new champion |

Ranking metrics (champion):
- ndcg@1 = 0.937, ndcg@5 = 0.924, ndcg@10 = 0.927
- exact_mrr = 0.953
- exact_recall@5 = 0.982, exact_recall@10 = 0.992

Per-class precision/recall (champion):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Exact | 0.919 | 0.936 | 0.927 | 39,356 |
| Substitute | 0.759 | 0.671 | 0.712 | 9,660 |
| Complement | 0.638 | 0.541 | 0.585 | 1,394 |
| Irrelevant | 0.886 | 0.907 | 0.896 | 25,638 |

**Run identifiers**:
- Zero-shot eval: MLflow run `2acd8c7b4e7a4eb8b6a6ba05d7272cef` (`00-baseline-zeroshot-old-keep`)
- Naive retrain: MLflow run `0ac8299883d5453e9267b061cc9bd368` (`01-baseline-naive-retrain-1ep`)
- Champion checkpoint: `checkpoints/01-baseline-naive-retrain-1ep/best-step=19516-val_cls_micro_f1=0.8851.ckpt`
- Wall-clock: 41.6 min for 1 epoch (19,516 steps at 7.95 steps/s on RTX with bf16-mixed)

## Comparison with old champion

| Metric | Old champion (`c3400bf`, old val) | Naive retrain (new val) | Δ |
|---|---|---|---|
| micro F1 | 0.9204 | 0.8851 | −0.035 |
| macro F1 | 0.7575 | 0.7802 | +0.023 |
| F1 Exact | 0.968 | 0.927 | −0.041 |
| F1 Substitute | 0.584 | 0.712 | +0.128 |
| F1 Complement | 0.745 | 0.585 | −0.160 |
| F1 Irrelevant | 0.708 | 0.896 | +0.188 |

Cross-val comparison is approximate (different val sets), but the per-class
trajectory is what matters:
- **Substitute jumped +0.128** thanks to 2.3× more support and richer
  intra-query contrast — the previously worst class is no longer the floor.
- **Irrelevant jumped +0.188** because it went from a 6.7% background class
  to a 33.7% plurality with abundant training signal.
- **Complement dropped −0.160** because prevalence collapsed to 1.6%; this
  is the new bottleneck.
- **Exact dropped −0.041** because the old model was Exact-overconfident at
  80% prior; with 51% prior the precision-recall balance is healthier
  (P=0.92, R=0.94 vs old P~0.99 R~0.95).

## What changes for the next ablation chain

Settled findings from `autoresearch_2026-04-28.md` that need re-litigation:

1. **`use_class_weights=true`** — old finding (cw=false wins by +0.029
   micro / +0.028 macro at bs=32) was on 80/7/6/7. Inverse-freq weights
   are now `[0.72, 15.6, 1.93, 0.49]` (Comp at 15.6×, vs 4.3× before).
   Worth a fresh test specifically targeted at lifting Complement.
2. **`focal_gamma=2.0`** — local optimum was tuned to extreme imbalance.
   Sweep γ ∈ {1.0, 1.5, 2.0, 2.5} now that distribution is more balanced.
3. **"ep1 peak is the peak"** — was specific to a 6,050-step epoch on
   80%-Exact data. With ~19,500-step epochs and rich intra-query contrast,
   ep2/3 may genuinely improve. `train_loss_epoch=0.128` vs
   `val_loss=0.130` at end of ep1 says the model is *not* yet overfitting.
4. **`bs=64 hurts`** — gradient-noise math is different at 3× more data.
   May now be net-positive given enough steps to converge.

Findings expected to still hold:
- Tokenizer choice (gelectra-large dominates — pretraining advantage is
  dataset-independent)
- `max_pair_length=512` covers p99 (truncation 0% on new data per EDA)
- `mdeberta-v3-base` / `xlm-roberta-large` likely still lose

New leverage that didn't exist before:
- **Intra-query contrast jumps from ~17-20% → 89%** of queries supply
  minority-class contrast. Pairwise/listwise ranking losses (margin loss,
  RankNet) become viable on this data; were not on old.
- **Group features** (#1 ranked open idea on old data, motivated by
  *missing* this signal) may have lower expected gain now since the data
  itself provides the contrast — but still untested.

## Proposed next experiments (priority order)

1. **`use_class_weights=true`** at the new champion config → measure macro
   lift, especially on Complement.
2. **2-epoch run** of the champion config — confirms or refutes "ep1 is
   the peak" on the new data scale.
3. **focal_gamma sweep** {1.0, 1.5, 2.5} (γ=2 is the current keep, γ=3
   was previously shown to overshoot).
4. **bs=64 retest** — only after #2 (to know how many steps a full epoch
   takes at the larger batch).

Budget: epoch parity (1 epoch = ~42 min wall-clock at bs=32) instead of
the old 20-min wall-clock screening, since the budget framework no longer
maps to a clean fraction of an epoch.
