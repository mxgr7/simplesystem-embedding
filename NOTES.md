# embedding/main — research notes

Guide for continuing past the `autoresearch/apr13` plateau. Audience is me,
later. Terse; assumes familiarity with SHN, RQP vs `anchor_query`, and
contrastive-vs-triplet.

## Current state

**Apr14 champion**: `angry-calf-984` (commit `6247f3b`), **nDCG@5 = 0.7382,
MRR = 0.8062**. Trained under the new `output_dim=128` pin added in commit
`205caff`, so not directly comparable to the apr13 SOTA.

**Apr13 champion for reference**: `valuable-cod-204` (commit `87fe454`),
nDCG@5 = 0.7682. Trained with `output_dim=null` (no projection). Pinning
`output_dim=128` forced a 768→128 projection head onto the apr14 runs and
cost ≈0.05 nDCG@5 out of the gate; apr14 recovered +0.0197 of that.

**Apr14 baseline config** (what the apr14 champion inherits):
`intfloat/multilingual-e5-base`, contrastive,
`train_batching_mode: random_query_pool`, `precision: bf16-mixed`, bs=512,
lr=1e-5 (encoder), warmup_ratio=0.33, n_pos=2, n_neg=4, compile=true,
max_offer_length=256, `output_dim=128` (linear projection), SHN pool
`defiant-mink-296`. Plus the two apr14 lifts below.

**Apr14 lifts (total +0.0197 nDCG@5 over the apr14 baseline 0.7185)**:
1. `f45780f` / `b916dcc` — **30x LR for the projection head** (via a second
   optimizer param group). +0.014. The randomly-init 768→128 Linear needs to
   catch up to the pretrained encoder; uniform lr=1e-5 under-trains it. Swept
   10x (+0.009), 30x (+0.014, peak), 50x (equal to 30x within noise), 100x
   (regresses back to 10x-level). 50x and 30x are on a plateau; 30x kept as
   the simpler value.
2. `6247f3b` — **Auxiliary 768-d contrastive loss on the pre-projection
   pooled embedding, α=0.5**. +0.005. Same encoder forward pass produces both
   128-d projected and 768-d pooled embeddings; both feed an in-batch
   contrastive loss with `total = loss_128 + α·loss_768`. Preserves the
   full-dim geometry while the bottleneck trains. α sweep: 0.3 (−0.004),
   **0.5 peak**, 0.7 (not completed), 1.0 (−0.007 — 768-d loss dominates and
   under-trains the 128-d head).

**Regime**: 20-min wall clock, single H100 80GB, ~2150 steps at bs=512.
Noise floor ±0.003 nDCG@5.

**Artifacts**:
- Apr14 champion checkpoint: `checkpoints/angry-calf-984/best-step=2177-val_full_catalog_ndcg_at_5=0.7382.ckpt`
- SHN pool: `../../data/semi_hard_negatives-defiant-mink-296.parquet` (untouched from apr13)
- Per-run log: `results.tsv` (apr13 rows 1-30, apr14 rows 31+). Metric
  numbers are `val/full_catalog/ndcg_at_5` from MLflow, not from the
  best-checkpoint filename. MRR is in the description column where logged.

**Prior-era reference**: `adorable-mole-653` at **MRR 0.816**, 10 epochs
under `anchor_query`, pre-apr13 loop. Not directly comparable to either the
apr13 or apr14 20-min budget regime.

## Next experiments (ranked by expected leverage, apr14-updated)

All inherit the apr14 champion config unless noted.

1. **Domain-adaptive pretraining on `offers_distinct.parquet`.** MLM (or
   SimCSE-style unsupervised contrastive) over the deduped offer text dump
   before the supervised fine-tune. The e5-base checkpoint has never seen
   the product distribution. With the aux 768-d loss now wired up, a
   domain-adapted base should compound — the aux loss teaches the encoder
   to use its full dim against our task, and domain adaptation teaches it
   the distribution shape in the first place. Start with 1-2 shards as a
   sanity check. Highest infra cost, biggest potential.
2. **e5-large at longer per-run budget (30-40 min @ bs=256).** The 20-min
   apr14 attempt (`f88f189`, 0.7178) stopped at 1379 steps — clearly
   under-fit. The capacity gain never landed. At 30-40 min that run would
   hit ≈2500 steps, enough to exercise the 568M-param encoder. Combined
   with the aux 768-d loss this is the single most likely lift. Breaks
   20-min comparability so use it as a probe, not a loop iteration.
3. **ANCE-style in-training SHN refresh.** Recurring pool refresh while the
   trainee geometry drifts. apr13's one-shot re-mine from `defiant-mink-296`
   gave +0.005 then plateaued; apr13's iterative re-mine from the champion
   itself hurt (-0.006). ANCE's value is cadence and freshness synchronized
   with trainee state. Needs a mining hook inside the Lightning loop.
4. **Warm-start cascade from the apr14 champion.** Fix the `encoder._orig_mod.`
   state-dict prefix mismatch (stripping the prefix in
   `checkpoint_connector`'s load path, or pre-patching the checkpoint
   file), then chain 2-3 warm-starts. Cheap to implement once the key
   rewrite is in place. Tests whether the champion is step-starved vs at a
   local optimum. apr14 tried this once (commit `defiant-rat-166`) and
   failed: with `compile=true` the 20-min budget was entirely consumed by
   torch.compile restoration; with `compile=false` the state_dict keys
   mismatched. Both fixable.
5. **768→128 distillation from a dedicated 768-d teacher.** The
   `teacher_checkpoint` / `distill_temperature` / `kl_distillation_loss`
   code path is already wired. Run one probe with `output_dim=null` to
   produce a 768-d teacher (requires a temporary unpin), then distill to
   a 128-d student with KL on similarity matrices. The aux-loss result
   already proves 768-d signal helps the 128-d head; an explicit teacher
   should be strictly more informative.
6. **Cosine LR schedule with explicit `total_steps≈2200`.** Current
   `constant_with_warmup` + `max_epochs=1000` stretches the schedule over
   ~378k nominal steps and therefore barely decays over the 2150 actual
   steps. A cosine tuned to the real step count lets the LR actually
   tighten in the final epochs. Cheap one-run probe.
7. **Wider SHN rank band at mine time** (e.g. `--rank-start 10 --rank-end 80
   --max-negatives-per-query 20`). Keeps rank 0-10 excluded but broadens
   pool diversity. Cheap if the miner is strong enough — currently the
   apr14 champion (0.7382) is weaker than the apr13 `defiant-mink-296` that
   produced the current pool, so mining from apr14 is likely to degrade
   the pool. Wait until (1), (2), or (5) lifts the trainee above the
   current pool's miner.
8. **Pretokenize the training set.** NOTES-sanctioned small throughput win
   (~few % more steps). Cheap if (1)-(7) are exhausted.

Triplet revisit: still parked, same reasons as apr13. See dead-ends.

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

## Apr14 strategic lessons (under the `output_dim=128` pin)

- **The projection head needs its own LR group.** Uniform lr=1e-5 trained
  the randomly-initialized 768→128 Linear far too slowly next to the
  pretrained encoder. Scaling the projection's LR 30× unlocked +0.014
  nDCG@5. Optimum is a plateau across ~30-50×; 10× under-shoots by 0.005,
  100× over-shoots by the same amount. Implementation: a second AdamW
  param group at 30×base_lr containing every parameter of
  `self.projection` (and, if added later, any module downstream of the
  encoder such as `self.pre_projection_norm`).
- **Auxiliary full-dim contrastive loss works as a bottleneck compensator
  (+0.005).** Running a second in-batch contrastive loss on the
  L2-normalized pre-projection 768-d pool, added to the primary 128-d
  loss at α=0.5, lifts nDCG@5 by 0.005. Cost: one extra similarity matrix
  per batch, shared encoder forward pass, negligible throughput hit. α
  sweep is sharply peaked: α=0.3 gives +0.001, α=0.5 gives +0.005, α=0.7
  untested, α=1.0 regresses −0.002. The interpretation: the encoder gets
  two gradient signals — the 768-d one keeps its full geometry intact,
  the 128-d one trains the projection — and at α=0.5 they reinforce
  rather than compete. Only the 128-d head is used at eval/inference.
- **The apr14 lifts stack (projection LR + aux loss = +0.0197).** Neither
  was tried in apr13 because apr13 ran with `output_dim=null` and had no
  projection head. Both are consequences of the `output_dim=128` pin and
  would not have been discovered without it.
- **Throughput-bound vs tuning-bound — still true at 128-d.** Every lever
  that cost steps without a matching capacity gain regressed: e5-large at
  bs=256 lost because 1379 steps < champion's 2150; warmup_ratio=0.1
  reached its peak 300 steps earlier then regressed. The apr14 wins both
  came from parameter-efficient changes that kept the step count flat.
- **MRR moves less than nDCG@5 on these lifts.** apr14 baseline
  MRR=0.7985, champion MRR=0.8062: +0.008 MRR for +0.020 nDCG@5. The
  bottleneck penalty and its compensation both live mostly in the rank-2
  to rank-5 tail, not at rank 1. This is further evidence the query
  distribution is ID-heavy (see dead-end section below): rank-1 is
  dominated by exact ID token matches which both the apr13 SOTA and the
  apr14 champion already solve well.

## How we got here (keep chain)

apr13 autoresearch loop, `a87e2c0` → `87fe454`, +0.086 nDCG@5 total on the
`output_dim=null` config.

| commit    | nDCG@5 | change | knob                                       |
|-----------|--------|--------|--------------------------------------------|
| `a87e2c0` | 0.6823 | baseline | prior config, fresh run                  |
| `f02164f` | 0.6980 | +0.016 | `precision=bf16-mixed`                     |
| `11e50a6` | 0.7054 | +0.007 | `batch_size=512`                           |
| `213463a` | 0.7089 | +0.004 | `n_neg_samples_per_query=4`                |
| `c6131a0` | 0.7637 | +0.055 | `train_batching_mode=random_query_pool`    |
| `87fe454` | 0.7682 | +0.005 | SHN pool re-mined from `defiant-mink-296`  |

apr14 autoresearch loop (2026-04-14), under the new `output_dim=128` pin
from commit `205caff`, `af12696` → `6247f3b`, +0.0197 nDCG@5 total over
the apr14 baseline. The apr14 champion is still ~0.030 below the apr13
SOTA because the 128-d projection costs ~0.05 out of the gate and apr14
recovered ~0.020 of that.

| commit    | nDCG@5 | MRR    | change | knob                                                    |
|-----------|--------|--------|--------|---------------------------------------------------------|
| `af12696` | 0.7185 | 0.7985 | baseline | apr14 fresh start, output_dim=128 projection pinned   |
| `f45780f` | 0.7272 | n/a    | +0.009 | 10× LR for projection head params (2nd optimizer group) |
| `b916dcc` | 0.7329 | 0.8049 | +0.006 | 30× LR for projection head (from 10×)                  |
| `6247f3b` | 0.7382 | 0.8062 | +0.005 | auxiliary 768-d contrastive loss on pre-projection pool, α=0.5 |

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

### Apr14 dead ends (under `output_dim=128` + 30x projection LR)

| knob                                              | result          | note                                                         |
|---------------------------------------------------|-----------------|--------------------------------------------------------------|
| orthogonal init for projection (`nn.init.orthogonal_`) | noise (−0.003) | random init is fine once the LR is right                    |
| 2-layer MLP projection (Linear→GELU→Linear)       | −0.012          | extra head capacity hurts; the bottleneck wants to be linear |
| `similarity_scale=15` (softer)                    | −0.019          | 20 is still optimal under the projection pin                 |
| encoder `lr=2e-5` with projection LR 30× still on | noise           | LR split doesn't reopen the encoder LR axis                  |
| natural-sentence offer_template (ID fields kept)  | −0.006          | current "Field: value" format exposes ID tokens cleanly     |
| random offer field dropout (0.2 per optional field at collate) | noise      | regularizers don't help at throughput-bound budget          |
| `warmup_ratio=0.1`                                | noise, unstable  | peaks at step 1830 then regresses                           |
| e5-large @ bs=256                                 | −0.015          | 1379 steps in 20 min — capacity gain lost to throughput      |
| pre-projection LayerNorm (head group)             | noise (−0.002)   | LN on already-well-scaled mean pool buys nothing             |
| `aux_raw_loss_weight` α=0.3                       | −0.004 vs α=0.5 | too little 768-d signal to pull the encoder                  |
| `aux_raw_loss_weight` α=1.0                       | −0.007 vs α=0.5 | 768-d loss dominates and under-trains the 128-d head         |
| warm-start cascade                                | blocked         | `encoder._orig_mod.` state-dict prefix mismatches compile=false loader; compile=true restore consumes entire 20-min budget. Fixable (strip prefix in load path) — left as future work. |

### Data / negatives

- **Hard negatives (rank 0-20) mixed with semi-hard (20-60)**: −0.003 (at
  noise). Rank 0-10 is contaminated by unlabeled positives and
  near-duplicates that destabilize the softmax. Stick with semi-hard only.
- **Second iterative SHN re-mine** from the champion trainee: −0.006. Pool
  geometry converges to the trainee's and negatives become easy. See the
  "one re-mine helps" lesson above — the next re-mine has to come from a
  *qualitatively stronger* trainee (new model, new data), not another
  iteration on the current one.

### Text rendering

- **EAN / article_number / manufacturer_article_number are load-bearing
  offer fields**, not noise. Dropping them from `offer_template` regressed
  −0.056 nDCG@5 (0.7329 → 0.6773, apr14 commit `ad2187c`). Many queries in
  the labeled parquet appear to be literal article numbers or EANs — the
  encoder matches them as token sequences, so removing them kills that
  retrieval path entirely. Any future offer_template change must keep all
  three ID fields present, even though they look like junk numeric strings.

### Metric choice — MRR is probably a better fit than nDCG@5 (not yet acted on)

The ID-field finding implies the query distribution is ID-heavy, which makes
this a known-item retrieval task: one query → one specific offer. That is
exactly the regime MRR was designed for, and where nDCG@5's graded-gain
machinery is mostly wasted. Tradeoffs:

- **MRR pros**: tighter signal on ID queries (rank-1 vs rank-10 is a 10x
  MRR delta; nDCG@5 truncates both to 0 past rank 5); likely a wider
  run-to-run spread at fixed noise floor, which would sharpen keep/discard
  calls; matches the likely production KPI if the UI shows a single "best
  match" per query.
- **MRR cons**: binary relevance, so a Substitute at rank 1 with the Exact
  at rank 2 looks like MRR 0.5 even if the Substitute is user-acceptable;
  brittle to unlabeled duplicate offers.
- **Capacity angle**: we're training a 128-d encoder to memorize literal
  EANs/article numbers, which is a fight embeddings are structurally bad
  at. In a production system the right design is hybrid retrieval: route
  ID-shaped queries to an exact-lookup / BM25 path and let the dense
  embedding specialize in semantic queries. That would also free 128-d
  capacity for the semantic half. Inside this eval harness, though, both
  paths collapse into the same dense dot-product metric, so hybrid gains
  are invisible and we're forced to make the embedding carry both roles.

**Status**: `val/full_catalog/mrr` is already logged to MLflow alongside
`val/full_catalog/ndcg_at_5`, so we can cross-check MRR as a secondary
signal without changing anything. `validation_metric` is pinned by
`program.md` so the loop keeps optimizing nDCG@5 for now — switching the
monitor would break comparability with the apr13 `results.tsv` rows.

**Next actions (queued, not yet run)**:
1. Quantify the ID-query fraction: regex over `queries_offers_labeled.parquet`
   val split for digit-only / alnum-only query shapes. If it's a sizable
   chunk (say >20%), the effective nDCG@5 ceiling is materially lower than
   it looks, and future lifts should be evaluated separately on the
   ID-query subset vs the semantic subset.
2. Cross-check MRR alongside nDCG@5 when logging future `results.tsv` rows
   — include the MRR in the description column so we can see whether a
   change is moving the rank-1 head or just reshuffling the tail.
3. If the ID-query fraction is large and we want a bigger lift, push back
   to the human about carving out a split eval (semantic-only nDCG@5 +
   ID-query exact MRR) or unpinning `validation_metric` to let us monitor
   MRR directly. Not a unilateral call.

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
