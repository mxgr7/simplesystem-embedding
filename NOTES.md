# embedding/main — research notes

Guide for continuing past the `autoresearch/apr13` plateau. Audience is me,
later. Terse; assumes familiarity with SHN, RQP vs `anchor_query`, and
contrastive-vs-triplet.

## Current state

**Champion**: `valuable-cod-204` (commit `87fe454`), **nDCG@5 = 0.7682**.
Plateaued after ~7 consecutive non-improvements following a second (regressive)
SHN re-mine.

**Baseline config**: `intfloat/multilingual-e5-base`, contrastive,
`train_batching_mode: random_query_pool`, `precision: bf16-mixed`, bs=512,
lr=1e-5, warmup_ratio=0.33, n_pos=2, n_neg=4, compile=true,
max_offer_length=256, `output_dim=null` (no projection), SHN pool re-mined
from `defiant-mink-296`.

**Regime**: 20-min wall clock, single H100 80GB, ~2500 steps. Noise floor
±0.003 nDCG@5. All claims below are relative to this regime.

**Artifacts**:
- Checkpoint: `checkpoints/valuable-cod-204/best-step=2444-val_full_catalog_ndcg_at_5=0.7682.ckpt`
- SHN pool: `../../data/semi_hard_negatives-defiant-mink-296.parquet`
- Per-run log: `results.tsv` (30 rows from the apr13 loop, 2026-04-13 →
  2026-04-14). Metric numbers there are `val/full_catalog/ndcg_at_5` from
  MLflow, not from the best-checkpoint filename.

**Prior SOTA for reference**: `adorable-mole-653` at **MRR 0.816**, but trained
10 epochs under `anchor_query`. Not directly comparable to the 20-min budget
regime, and its ceiling was likely capped by the same `anchor_query`
starvation the apr13 loop uncovered.

## Next experiments (ranked by expected leverage)

All inherit the baseline config unless noted.

1. **Larger base encoder** (`intfloat/multilingual-e5-large`, 568M). Biggest
   untested single swing. Probably needs bs=256 and gets ~1200 steps per
   20-min budget vs ~2500 on e5-base, so the capacity ceiling has to outweigh
   fewer steps.
2. **ANCE-style in-training SHN refresh.** Warm-start the trainee from
   `valuable-cod-204` and re-mine the pool on step 0 so miner and trainee
   geometries match at training start. One-shot manual re-mining already gave
   +0.005 and plateaued — ANCE's value is the *recurring* refresh while the
   trainee geometry drifts. Needs a mining hook inside the training loop plus
   a cadence decision (cost vs staleness). High leverage but infra work.
3. **Pretokenize the training set** at datamodule setup time and cache the
   tokenized tensors. `num_workers=16` was a noise no-op so the DataLoader
   pipeline isn't currently pinned, but pretokenizing removes the tokenizer
   from the hot path entirely. Likely a few percent more steps per budget;
   cheap code change.
4. **Wider SHN rank band at mine time** (e.g. `--rank-start 10 --rank-end 80`,
   `--max-negatives-per-query 20`). Keeps rank 0-10 excluded (the noisy
   false-positive band) but increases pool diversity. Cheap — one mine + one
   training run.
5. **Dropout bump** in the encoder (try 0.15–0.2 vs the pretrained 0.1
   default). Untested regularizer. `weight_decay=0.1` was a no-op so
   wd-based regularization is off the table, but dropout is a different axis.

Triplet revisit: only on a qualitatively different setup (new base model, new
dataset, new batching format). Not another tuning knob on this setup. See
dead-ends section for why.

## Strategic lessons (why the baseline looks like it does)

- **`random_query_pool` >> `anchor_query` (+0.055 nDCG@5).** Single biggest
  lever found in the apr13 loop. `anchor_query` starves the in-batch
  contrastive softmax of cross-query negatives because each batch is
  dominated by one query's rows. RQP shuffles all query records into a pool
  and samples uniformly, so every batch contains many queries and the softmax
  has many genuine negatives per row. Whatever sets the upstream default to
  `anchor_query` is silently leaving ~0.05+ nDCG@5 on the floor — worth
  making RQP the default unless there's a specific reason to block-sample
  by query.
- **`bf16-mixed` is free throughput (+0.016).** ~43% more steps at identical
  quality, no stability issues. Biggest single compute-time lever. Default.
- **bs=512 + n_neg=4 is the 80 GB sweet spot.** bs=768/1024 OOM with bf16 +
  compile + e5-base + max_offer_length=256 (bs=512 reports ≈55 GB resident).
  bs=640 is noise — the extra VRAM buys nothing. n_neg=6 regressed −0.011
  (fewer unique queries per batch → less diverse negatives). n_pos=2 is the
  floor: n_pos=1 dropped −0.038 because contrastive needs ≥2 positives per
  query so each anchor row has a same-query positive to lock onto. n_pos=3
  dropped −0.018 from the same "too few unique queries per batch" failure
  mode as n_neg=6.
- **The 20-min budget is throughput-bound, not tuning-bound.** Half the
  apr13 gains came from throughput (bf16, bs). LR / wd / scheduler /
  similarity-scale / projection-head / num_workers are all at or below the
  noise floor. Don't re-sweep without a qualitative setup change.
- **Warmup matters, LR doesn't (much).** `warmup_steps=0` regressed −0.028,
  but LR sweep {5e-6, 1.5e-5, 2e-5} all sat at or below noise vs 1e-5. The
  linear scheduler is near-flat because `max_epochs=1000` stretches the
  schedule over ~378k nominal steps while we only run ~2600, so it barely
  decays over a 20-min run.
- **One SHN re-mine helps, a treadmill doesn't.** Re-mining once from a
  stronger trainee (`defiant-mink-296` → champion pool) gave +0.005. A second
  iterative re-mine from `valuable-cod-204` itself hurt by −0.006: the pool
  geometry converged to the trainee's and negatives became too easy.
  Corollary: the miner must be at least as strong as the miner behind the
  current pool — re-mining from `big-stoat-600` (0.7089) on top of
  `placid-snake-749` also hurt, for the same reason. Without a
  qualitatively stronger trainee, there is probably ≤0.01 nDCG@5 left in
  the mining direction on this model.
- **max_offer_length=256 is load-bearing.** max_offer_length=192 regressed
  −0.007 — offer texts are long enough that the truncation cost outweighs
  the extra steps per budget.

## How we got here (keep chain)

apr13 autoresearch loop, `a87e2c0` → `87fe454`, +0.086 nDCG@5 total.

| commit    | nDCG@5 | change | knob                                       |
|-----------|--------|--------|--------------------------------------------|
| `a87e2c0` | 0.6823 | baseline | prior config, fresh run                  |
| `f02164f` | 0.6980 | +0.016 | `precision=bf16-mixed`                     |
| `11e50a6` | 0.7054 | +0.007 | `batch_size=512`                           |
| `213463a` | 0.7089 | +0.004 | `n_neg_samples_per_query=4`                |
| `c6131a0` | 0.7637 | +0.055 | `train_batching_mode=random_query_pool`    |
| `87fe454` | 0.7682 | +0.005 | SHN pool re-mined from `defiant-mink-296`  |

## Dead ends — don't rerun without a qualitative setup change

### Hyperparameters tested and rejected (e5-base + RQP + bf16-mixed + 20-min)

| knob                                  | result        | note                                              |
|---------------------------------------|---------------|---------------------------------------------------|
| lr ∈ {5e-6, 1.5e-5, 2e-5}             | noise         | 1e-5 is optimum; scheduler barely decays          |
| lr=2e-5 + RQP                         | noise         | RQP does not change the LR optimum                |
| `weight_decay=0.1`                    | noise (+0.0001) | no benefit from stronger regularization         |
| `warmup_steps=0`                      | −0.028        | warmup matters despite near-flat scheduler math   |
| `similarity_scale=30`                 | slightly worse | 20 (τ=0.05) is near-optimal                      |
| `gradient_checkpointing` + bs=1024    | −0.038        | GC slowdown (~950 vs ~2650 steps) kills it        |
| `batch_size=640`                      | noise          | between keep (512) and OOM (768)                 |
| `batch_size ≥ 768`                    | OOM            | naive-scaling ceiling on 80 GB                   |
| `max_offer_length=192`                | −0.007         | truncation cost > throughput gain                |
| `output_dim=256` (projection head)    | −0.015         | learned projection on mean-pooled e5 hurts       |
| `num_workers=16`                      | noise          | GPU not data-bound; 8 workers are enough         |
| `compile=false`                       | noise (−0.002) | compile marginally positive                      |
| `n_pos=1`                             | −0.038         | breaks same-query positive requirement           |
| `n_pos=3` (with n_neg=4)              | −0.018         | fewer unique queries → repetitive in-batch negs  |
| `n_neg=6`                             | −0.011         | same failure mode as n_pos=3                     |

### Data / negatives

- **Hard negatives (rank 0-20) mixed with semi-hard (20-60)**: −0.003 (at
  noise). Rank 0-10 is contaminated by unlabeled positives and
  near-duplicates that destabilize the softmax. Stick with semi-hard only.
- **Second iterative SHN re-mine** from the champion trainee: −0.006. Pool
  geometry converges to the trainee's and negatives become easy. See the
  "one re-mine helps" lesson above — the next re-mine has to come from a
  *qualitatively stronger* trainee (new model, new data), not another
  iteration on the current one.

### Triplet loss — parked

Three triplet runs across two neg-selection modes and two pool-freshness
conditions all peak in the MRR 0.72–0.75 band and regress within one epoch
of the peak. Contrastive on the identical config hits MRR 0.816 smoothly.
Gap is ~0.08 MRR, one-directional.

| run                                | mode        | pool                          | budget         | MRR    | note                 |
|------------------------------------|-------------|-------------------------------|----------------|--------|----------------------|
| `respected-bass-254` (`680d0bfe…`) | `semi_hard` | stale `placid-snake-749`      | 5 ep (killed)  | 0.7478 | regressed ep 5       |
| `nosy-smelt-449`   (`2a7cebd9…`)   | `semi_hard` | fresh `adorable-mole-653`     | 4 ep (killed)  | 0.7329 | regressed ep 4       |
| `beautiful-bear-447` (`eae167e0…`) | `hardest`   | fresh `adorable-mole-653`     | 3 ep (killed)  | 0.7163 | loss locked at margin |

Observations that shaped the call:

- Fresh pool fixed the data confound
  (`train/triplet_semi_hard_fallback_share = 0.0` on the fresh pool vs
  meaningful fallback on the stale one). Retrieval still regressed, so stale
  pool was never the root cause.
- `hardest` mode's train loss parks at the margin (~0.20 with margin=0.2)
  because the hardest negative on a hard-mined pool usually sits at or above
  the positive. Loss signal is useless in that mode; retrieval still tracks
  `semi_hard` within 0.003 MRR but collapses one epoch earlier.
- Val loss decouples from `val/full_catalog/*` on every triplet run — val
  loss keeps improving (or stays parked) while retrieval regresses.
- `train/triplet_valid_anchor_count` is only 2–20 per batch of 256. Triplet
  structurally uses far less of each batch than contrastive's in-batch
  negatives do.

Triplet implementation stays in the tree — it's tested and cheap to keep.
**Revisit only** on a different base model, a different dataset, or a
qualitatively different batching format. Not another tuning knob on this
setup.
