# Next experiments

Triplet is parked (see below). Contrastive is the loss. The biggest
unexplored lever now is **capacity / warm-starting**: larger base encoder,
or ANCE-style in-training re-mine from a warm-started trainee. Throughput
knobs (precision, batch size, batching mode) are largely dialed in — see
`autoresearch/apr13 run` at the bottom for the full landscape.

**What the 20-min-budget autoresearch loop settled**:

- **Use `train_batching_mode: random_query_pool`, not `anchor_query`.**
  This was the single biggest win (+0.055 nDCG@5). `anchor_query` starves
  in-batch contrastive of cross-query negatives because each batch is
  dominated by one query's rows. The prior SOTA `adorable-mole-653` was
  trained under `anchor_query` at 10 epochs — it probably left real
  headroom on the floor that `random_query_pool` would have recovered.
- **Use `precision: bf16-mixed`.** ~43% more steps at identical quality,
  no stability issues, fills the 20-min budget better.
- **bs=512 + n_neg_samples_per_query=4** is the sweet spot on 80 GB.
  bs=768 OOMs; bs=640 is noise; n_neg=6 starves diversity.
- **LR, weight_decay, scheduler, warmup_steps, num_workers, compile,
  similarity_scale, max_offer_length, projection head** are all at or
  below the ±0.003 noise floor. Don't re-tune them without a reason.
- **Re-mining the SHN pool helps exactly once.** One re-mine from a
  stronger trainee (+0.005) is the sweet spot; a second iterative
  re-mine regressed because the pool converges to the trainee's
  geometry and the negatives become easy. The miner must be at least
  as strong as the miner behind the current pool or the new pool is
  just easier.
- **Hard negatives (rank 0-20) are noisy.** Mixing them with semi-hard
  (20-60) didn't help — unlabeled positives/near-duplicates at the top
  destabilize. Stick with semi-hard only.

## Recent runs

| run | loss | SHN pool | budget | nDCG@5 | MRR | notes |
|---|---|---|---|---|---|---|
| `valuable-cod-204` (`87fe454`) | contrastive / RQP | `defiant-mink-296` re-mine | 20 min | **0.7682** | — | autoresearch/apr13 champion; e5-base, bs=512, bf16-mixed, n_pos=2 n_neg=4 |
| `defiant-mink-296` (`c6131a0`) | contrastive / RQP | `placid-snake-749` | 20 min | 0.7637 | — | pivot run: `random_query_pool` unlocked +0.055 over anchor_query |
| `adorable-mole-653` (`e7809027…`) | contrastive / anchor_query | `placid-snake-749` | 10 epochs | — | **0.816** | prior SOTA; not directly comparable (longer budget, different batching) |
| `respected-bass-254` (`680d0bfe…`) | triplet/`semi_hard` | stale `placid-snake-749` | 5 ep (killed) | — | 0.7478 | regressed ep 5 |
| `nosy-smelt-449` (`2a7cebd9…`) | triplet/`semi_hard` | fresh `adorable-mole-653` | 4 ep (killed) | — | 0.7329 | regressed ep 4 |
| `beautiful-bear-447` (`eae167e0…`) | triplet/`hardest` | fresh `adorable-mole-653` | 3 ep (killed) | — | 0.7163 | loss locked at margin |

Current config baseline: `intfloat/multilingual-e5-base`, bs=512, lr=1e-5,
warmup_ratio=0.33, precision=bf16-mixed, `train_batching_mode:
random_query_pool`, n_pos=2, n_neg=4.

## Triplet is parked

Three triplet runs across two negative-selection modes and two pool-freshness
conditions all peak in the 0.72–0.75 MRR band and regress within one epoch of
the peak. Contrastive with identical config hits 0.816 smoothly. Gap is
~0.08 MRR and one-directional.

Observations that shaped the call:

- Fresh pool fixed the data confound (`train/triplet_semi_hard_fallback_share
  = 0.0` on the new pool vs meaningful fallback on the stale one). Retrieval
  still regressed, so stale pool was never the root cause.
- `hardest` mode's train loss parks at the margin (~0.20 with margin=0.2)
  because the hardest negative on a hard-mined pool usually sits at or above
  the positive. Loss signal is useless in that mode; retrieval still tracks
  semi-hard within 0.003 MRR but collapses one epoch earlier.
- Val loss decouples from `val/full_catalog/*` on every triplet run — val
  loss keeps improving (or stays parked) while retrieval regresses.
- `train/triplet_valid_anchor_count` is only 2–20 per batch of 256. Triplet
  structurally uses far less of each batch than contrastive's in-batch
  negatives do.

The triplet implementation stays in the tree — it's tested and cheap to keep.
Revisit only on a different base model, a different dataset, or a
qualitatively different batching format (not another tuning knob on this
setup).

## Experiments

Inherit the current baseline (e5-base, bs=512, lr=1e-5, warmup_ratio=0.33,
bf16-mixed, `random_query_pool`, n_pos=2 n_neg=4) unless noted. Ordered
by expected leverage.

- **Larger base encoder** (`intfloat/multilingual-e5-large`, 568 M). Probably
  needs bs=256 and will get ~1200 steps/20 min vs ~2500 on e5-base. Untested.
  Next-biggest single swing.
- **ANCE-style in-training re-mine**: warm-start the trainee from the
  current champion (`valuable-cod-204` / `87fe454`, 0.7682) and re-mine
  the SHN pool from the warm-started initial state so miner-and-trainee
  geometries match on step 0. Manual single-iteration re-mining already
  gave +0.005 and then plateaued — ANCE's value is in *recurring* refresh
  while the geometry drifts. Needs a mining hook inside the training loop
  and a cadence decision (cost vs staleness). Still high-leverage.
- **Pretokenize the training set** at datamodule setup time and cache the
  tokenized tensors instead of tokenizing per batch. `num_workers=16` was a
  noise no-op in the autoresearch run (so the DataLoader pipeline isn't
  pinned), but pretokenizing removes the tokenizer from the hot path
  entirely and could yield a few extra percent of steps.
- **Wider SHN rank band at mine time** (e.g. `--rank-start 10 --rank-end 80`
  and/or `--max-negatives-per-query 20`). Keeps rank-0-10 (the noisy band)
  excluded but increases pool diversity. Cheap to test — one mine + one
  training run.
- **Dropout bump** inside the encoder (try 0.15–0.2 vs pretrained default).
  Untested in the autoresearch loop. weight_decay=0.1 was a no-op so
  regularization-via-wd is off the table, but dropout is a different knob.
- **Triplet revisit**: only if a new base model or dataset changes the
  structural picture below.

### Things not worth re-testing on this setup

The autoresearch loop showed these are dead ends for a 20-min budget on
e5-base + RQP + bf16-mixed. Don't rerun without a qualitative change in
setup:

- LR sweep {5e-6, 1.5e-5, 2e-5} — all at or below noise vs 1e-5
- `weight_decay=0.1` — noise
- `warmup_steps=0` — destabilizes (-0.028)
- `similarity_scale=30` — slightly worse
- `gradient_checkpointing=true + bs=1024` — GC slowdown eats the bigger-batch gain
- `batch_size` > 512 — OOM (bs=640 is noise, 768/1024 OOM)
- `max_offer_length=192` — truncation cost > throughput gain
- `output_dim=256` projection head — hurts by 0.015
- `num_workers=16` — noise (data loading is not the bottleneck)
- `compile=false` — marginally worse
- `n_pos=1` or `n_pos=3` — hurts (2 is the sweet spot)
- `n_neg=6` — hurts (4 is the sweet spot)
- Adding hard negatives (rank 0-20) alongside semi-hard — noisy, unlabeled
  near-positives at the top destabilize
- A *second* iterative SHN re-mine — pool converges to trainee geometry

## autoresearch/apr13 run (2026-04-13 → 2026-04-14)

Autonomous 20-min-budget experiment loop on `autoresearch/apr13`. Goal: push
`val/full_catalog/ndcg_at_5` as high as possible within a fixed wall-clock
training budget from `e5-base`. All runs 20 min, single H100 80GB, unless
noted. Noise floor measured at ±0.003 nDCG@5.

**Config baseline at run start** (commit `a87e2c0`): e5-base, contrastive,
bs=256, lr=1e-5, warmup_ratio=0.33, precision=32-true, anchor_query
batching, n_pos=2, n_neg=2, SHN pool `placid-snake-749`, compile=true,
output_dim=null (no projection).

**Final champion** (commit `87fe454`): nDCG@5 = **0.7682** (+0.086 vs
baseline 0.6823). The keep chain was:

| commit | nDCG@5 | change | note |
|---|---|---|---|
| `a87e2c0` | 0.6823 | baseline | fresh run of prior config |
| `f02164f` | 0.6980 | +0.016 | **precision=bf16-mixed** — ~43% more steps in budget |
| `11e50a6` | 0.7054 | +0.007 | **batch_size=512** — more in-batch negatives |
| `213463a` | 0.7089 | +0.004 | **n_neg_samples_per_query=4** — more SHN per anchor group |
| `c6131a0` | 0.7637 | +0.055 | **train_batching_mode=random_query_pool** — dominant win |
| `87fe454` | 0.7682 | +0.005 | **SHN pool re-mined from defiant-mink-296** (intermediate champion) |

### What worked

1. **bf16-mixed is free throughput**. Biggest compute-time lever. ~+0.016
   nDCG@5 at identical hyperparams just from fitting more steps into the
   budget. No stability issues. Should be the default.
2. **random_query_pool >> anchor_query** (+0.055). By far the biggest single
   jump. `anchor_query` mode forces each batch to be dominated by one query's
   positives/negatives, which starves the in-batch contrastive loss of
   diverse cross-query negatives. `random_query_pool` shuffles all query
   records into one big pool and samples uniformly — every batch contains
   many queries so the contrastive softmax has many genuine negatives per
   row. Root cause of why the prior SOTA could only be reached at 10 epochs
   with `anchor_query` was probably this. At the 20-min budget the gap is
   enormous.
3. **Modest in-batch knobs stack** — bs=256→512 (+0.007) and n_neg=2→4
   (+0.004) both added small wins by giving contrastive more negatives per
   step. bs=768/1024 OOM on 80GB at fp32 activations.
4. **Re-mining SHN from a stronger trainee helps, but with strong diminishing
   returns**. First re-mine (from `defiant-mink-296`, 0.7637) gave +0.005.
   Second iterative re-mine (from `valuable-cod-204`, 0.7682) *hurt* by
   -0.006 — pool geometry converged to the trainee's and negatives became
   too easy. One re-mine per phase is the sweet spot, not a treadmill.
   Also: the initial attempt to re-mine from `big-stoat-600` (0.7089)
   *hurt* because it was weaker than whatever mined `placid-snake-749`, so
   the new pool was effectively easier. The miner must be at least as
   strong as the miner behind the current pool.

### What didn't (and why)

- **LR sweep** (`5e-6 / 1.5e-5 / 2e-5`): all worse or noise. lr=1e-5 is the
  sweet spot for this model+batching. Surprisingly insensitive — the linear
  scheduler is essentially flat because `max_epochs=1000` stretches the
  total schedule over ~378k steps while we only run ~2600, so the scheduler
  barely decays.
- **warmup_steps=0**: −0.028. Warmup actually matters, despite the apparent
  near-no-op math. Keep warmup_ratio=0.33.
- **weight_decay=0.1**: noise (+0.0001). Fine-tuning this model for 20 min
  doesn't benefit from stronger regularization.
- **similarity_scale 20→30**: slightly worse. 20 (= τ 0.05) is near-optimal.
- **gradient_checkpointing + bs=1024**: −0.038 vs champion. GC slowed
  throughput enough that only ~950 steps fit in 20 min vs ~2650; the larger
  batch did not compensate.
- **batch_size=640**: noise. Sits between bs=512 (keep) and bs=768 (OOM);
  fewer total steps offset the larger-batch gain.
- **batch_size=768 / 1024**: OOM at 80GB. Upper bound of naive batch
  scaling with this model in bf16-mixed.
- **max_offer_length 256→192**: −0.007. Faster per step but the truncation
  cost outweighed the extra steps. Offer texts are long enough that 256 is
  load-bearing.
- **output_dim=256 projection head**: −0.015. A learned projection on top
  of mean-pooled e5 hurts consistently. Drop.
- **Hard+semi-hard negatives combined** (rank 0-20 added alongside 20-60):
  −0.003 (at noise floor). The rank 0-20 band is contaminated by unlabeled
  positives and near-duplicates (confirms NOTES.md caveat). Stick with
  semi-hard only.
- **n_pos=1 n_neg=3**: −0.038. Contrastive needs ≥2 positives per query
  sampled into each batch to give each anchor row a same-query positive to
  lock onto. Don't drop below 2.
- **n_pos=3 n_neg=4**: −0.018. Too few unique queries per batch — the
  in-batch negatives become repetitive and contrastive signal flattens.
- **n_neg=6**: −0.011. Same failure mode as n_pos=3 — fewer queries per
  batch → less diverse negatives.
- **num_workers=16**: noise. GPU was not data-bound; 8 workers are enough.
- **compile=false**: noise (−0.002). torch.compile is marginally positive;
  the steady-state speedup slightly beats the ~60s compile-time cost.
- **lr=2e-5 + random_query_pool**: noise vs lr=1e-5. RQP does not change
  the LR optimum.

### New strategic observations

- **Batching mode is the biggest lever I found, and it wasn't on the
  original NOTES.md `Experiments` list.** Whatever sets default
  `train_batching_mode: anchor_query` is silently leaving ~0.05+ nDCG@5
  on the floor. Worth making `random_query_pool` the default unless
  there's a specific reason to block-sample by query.
- **The 20-min budget is binding; precision and throughput matter more
  than tuning knobs.** Half of all improvements came from throughput
  (bf16-mixed, bs). Tuning knobs (lr, wd, scheduler) are at or below the
  noise floor in this regime.
- **Re-mining works but is not a treadmill.** One re-mine from a stronger
  trainee helped. A second iterative re-mine hurt. I don't think there's
  more than ~+0.01 left in the mining direction at this model strength
  — the pool converges to "easy for this model" as soon as the trainee
  catches up to the miner.
- **VRAM ceiling at bs=768 is sharp.** With bf16-mixed + compile + e5-base
  + max_offer_length=256, 80GB is fully used at bs=512 (≈55 GB reported)
  and OOMs at bs=768. Any further batch growth has to come from shorter
  sequences, gradient checkpointing (rejected, too slow), or a smaller
  encoder.

### Plateau & suggested next directions

Champion plateaued at 0.7682 with ~7 consecutive non-improvements after the
second SHN re-mine. Things I did *not* test that I'd try next:

- **Larger base encoder** (`multilingual-e5-large`, 568M). At bs=256 it
  probably fits; training is ~2× slower so ~1200 steps per budget. Higher
  capacity ceiling may outweigh fewer steps.
- **Pretokenize the dataset at setup time** to eliminate per-batch tokenizer
  latency. Cheap code change, possibly +5–10% throughput → more steps.
- **Wider SHN rank band** (e.g. `--rank-start 10 --rank-end 80`) for a more
  diverse pool while keeping the false-positive problem contained (still
  avoids rank 0-10).
- **`max_negatives_per_query=20`** at mine time to double pool diversity,
  then let RQP sample across the richer pool.
- **Dropout bump** in the encoder (pretrained e5 uses 0.1 by default; try
  0.2) — untested regularizer.
- **Warm-start subsequent 20-min runs from the current champion
  checkpoint** (ANCE-style in-training refresh). This is the high-leverage
  item in the old NOTES.md but needs a mining hook in the training loop.

### Artifacts

- Champion checkpoint: `checkpoints/valuable-cod-204/best-step=2444-val_full_catalog_ndcg_at_5=0.7682.ckpt`
- Champion SHN pool: `../../data/semi_hard_negatives-defiant-mink-296.parquet`
- Full per-run log: `results.tsv` (30 rows, ~24 distinct experiments +
  crashes + one infra commit). All metric numbers in `results.tsv` are
  `val/full_catalog/ndcg_at_5` from the MLflow run, not from the best
  checkpoint filename (they usually match to 4 decimals but the MLflow
  one is the ground-truth-per-program.md).
