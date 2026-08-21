# splade autoresearch — research notes

Branch: autoresearch/splade-jul22. Audience: future me. Terse.
Pre-seeded 2026-07-21 by the setup session with everything already known —
READ THIS BEFORE THE FIRST EXPERIMENT; none of it needs re-testing.

## Session log (newest at top)

- 2026-07-22 ~11:00 **SESSION ENDED** (Fable-5 usage limit reached mid-loop;
  Max chose to close the session here). Coordinator captured the final
  in-flight experiment whose eval was lost with the agent:
  **s7_lr3e5 (lr 1.5e-5 -> 3e-5) is the SESSION-FINAL KEEP / chain best**:
  seg .9590 / gold .9046 (final@1016), vs b15_root .9583/.8956 = seg +.0007
  gold +.0090 mean +.0048; gold R@10 .5010 (all-time high in the whole
  program). Takeaway: at the shortened 15-min budget the hotter LR helps —
  consistent with the s6 finding (short budget wants warm LR through the whole
  run; annealing kills sparsification). ckpt: vastai0:~/ar_splade/checkpoints/
  s7_lr3e5/final-step=1016.ckpt.
  FINAL CHAIN: base_raw .9591/.8921 -> k1_fdrop30 .9609/.8971 -> b15_root
  (k2 config, 15min) .9583/.8956 -> s7_lr3e5 .9590/.9046. Best serving profile
  = k2/b15 lineage (FLOPS ~1.9, qnnz ~30-31). Untried at session end:
  distillation with KL/normalized margins (additive MarginMSE fights FLOPS —
  the raw form is dead), dropout-prob sweep, gbert-large, hard-neg mining,
  lr sweep beyond 3e-5, and the promoted "full-length candidates"
  (k2 config @ 8ep, s7 lr @ full-length). Two teacher score files staged on
  the box for whoever resumes (see the ~03:20 entry).

- 2026-07-22 ~09:10 **REGIME CHANGE (Max directive via coordinator)**:
  1) Chain RE-ROOTED on the k2 config — desc0 parquet + max_offer_length
     256 + batch 512 + accum 1 + field_dropout .3 + bucketing — overriding
     the R@100-only keep verdict (R@10 +.009/+.008, FLOPS 1.89/1.47,
     qnnz 30, 5.8 ep in 45 min). k2 45-min numbers: seg .9628/.4422 gold
     .8962/.5018.
  2) Budget 45 -> 15 MIN per run (run_remote enforces 900s cap + kill at
     2100s; commit 48da6a0). ~1,000 steps ≈ 1.9 ep of k2 config per run.
  3) New 15-min keep-chain root = b15_root (k2 config, 15 min, both
     evals) — pending. All subsequent deltas vs that row.
  4) FLOPS/qnnz are mid-schedule at 15 min (~3-4 expected): guardrail ≤6
     applies, but do NOT compare FLOPS across budget generations (45-min
     rows and 8-ep rows are different λ-schedule points).
  k3_desc0_384 (seq384/b384/accum1) OOM'd (17GiB alloc) — 384/384 dead at
  accum1; if wanted later use b256@384 or accum2.

- 2026-07-22 ~07:30: **DESCRIPTION RETIRED** (coordinator desc0 ablation,
  other box): the eval harness ALWAYS blanks description at doc-render
  ("deploy: desc-free") — every number ever produced is desc-free-served.
  desc0_raw (desc blanked at train, 8ep) ties fold_raw (.9643/.8989 vs
  .9635/.8999); desc-ON serving +.001 noise. => never spend screens on
  description content. New lever: splade_train_raw_desc0.parquet +
  data.max_offer_length=256 (desc-free renders p50 206, >256 = 32%,
  >384 = 9.6%) + data.batch_size=512 accum=1 (FULL in-batch pool — kills
  the halved-negatives confound). run_remote now allows accum override
  (7f85f4b). k2_desc0_fd30 launched (keep-track: desc0+seq256+b512+
  accum1+fdrop30; fallback if it regresses: seq384/b384).
- 2026-07-22 ~07:10: **k1_fdrop30_raw = KEEP, new chain best**: seg
  .9609/.4332 gold .8971/.4935 (vs root +.0018/+.0050, mean +.0034).
  fdrop30+bucketing on raw, final@1662. soup_k1s5 (k1+s5): seg .9600
  gold .9000, mean vs k1 +.0010 < bar — not kept; BUT gold R@10 .5028 =
  best anywhere (soup remains the deploy-mix candidate).
  s5_fdrop30 screen: seg .9570 (+.0034 vs s1) — graduated.

- 2026-07-22 ~05:50: **MarginMSE distillation = DEAD IN-BUDGET.**
  s3 (w=.05 scale=20): FLOPS_w 68, qnnz 127, seg .8993 (-.054).
  s4 (w=.02 scale=5): FLOPS_w 33, seg .9369 (-.017). Mechanism: MSE on raw
  dot margins pushes student scores up/apart without bound → activation
  inflation the FLOPS reg can't hold at 45-min λ. For full-length owners:
  needs margin normalization (e.g. per-batch z-scored margins, clipped
  targets, or listwise KL over CE probs) — NOT plain additive MarginMSE.
  Both ce score files verified loading (285,103 pairs, 100% coverage).
- 2026-07-22 ~04:00: s1_bucket = Phase-0 KEEP: 1632 steps vs 1200 (+36%
  usable, incl. final-ckpt capture), seg .9536 flat vs .9528 baseline,
  FLOPS_w 2.37 vs 4.42. length_bucketing=true now config default
  (commit a23afe7). New screen reference = s1 .9536/.4240.
  base_raw keep root (final@1320): seg .9591/.4276 gold .8921/.4866.

- 2026-07-22 ~03:20: **CE distillation UNBLOCKED — two score files**:
  1) coordinator's: /home/max/simplesystem-embedding/data/ce_scores_fold.parquet
     (ce_full_v3 teacher, bge-reranker-v2-m3 2026-07-17, native unfolded
     pipe-render + segment hints; mean .66 p25 .25 median .96; 100% raw_fold
     coverage; caveats: v1-era labels, scores its own gold train pairs
     optimistically/no OOF).
  2) mine: /home/max/ar_splade/data/ce_scores_foldtrain.parquet (soup gelectra
     CE v1.0-2026-04-29 via scripts/score_ce_pairs.py on UNFOLDED
     splade_train_raw.parquet — key sets identical to raw_fold, verified
     285,103/285,103; scored in 80s @3.5k/s).
  Using (1) for screens; teacher A/B is a cheap later screen.
  s3_distill first launch CRASHED: margin_mse_teacher_scale not in
  model=splade struct (only in splade_distill.yaml) — keys now exposed in
  splade.yaml (commit 4262547). ce_scores_path enters prepare-cache key →
  first distill run re-prepares (~5min).
- 2026-07-22 ~03:00: s2_mask100 (train-time stoplist, IDS100) seg
  .9538/.4246 FLOPS 2.47 — FLAT vs s1_bucket (.9536/2.37). No in-budget
  win; mask may still matter at full length/deployment. Parked.

- 2026-07-22 early session:
  - Harness port REPRODUCED soup_fold exactly (seg .9588/.4293, gold
    .9040/.4925); seg eval 10m00, gold 9m00 (was 15-25m). Added FAST=1
    (skip mask configs) + eval_remote 3rd arg best|final|last (auto
    prefers final).
  - **In-budget baselines**: base_b50 (45min, step1200, best ckpt) seg
    .9528/.4206 FLOPS_w 4.42 — SCREEN BASELINE. base_raw step1320
    internal .8223 — keep-root evals pending. 45-min models are ~2x
    denser (FLOPS 4.4 vs 2.2-2.7 at 8ep); guardrail 6.0 still met.
  - **Tail-loss fix**: max_time stop can discard training since last val
    ckpt (base_b50 lost 134 steps; base_raw DID val at stop — behavior
    inconsistent). train.py now saves final-step=N.ckpt after fit
    (weights only); eval_remote auto-prefers it.
  - Step rate ~29-31 opt steps/min (batch 256x2, seq 512); val overhead
    small (~2min/45min). Biggest training lever = padding waste:
    LengthBucketedBatchSampler (window=16 batches) — screening as
    s1_bucket.
  - Stoplist ids from soup_fold evals (top-150 doc-activation-freq of
    BOTH pools, 144 common). Eval-time mask costs recall (mask100 seg
    -.010) — lever is TRAIN-time. IDS100 for model.stopword_mask_ids:
    566,818,136,232,128,2161,853,15462,125,143,105,255,10083,1452,12438,30891,4024,30881,8603,4080,30885,292,1693,5559,6157,21020,3149,3483,4805,737,227,1320,976,275,616,247,1483,30911,830,289,8991,30886,30916,2610,6004,164,7421,820,30950,3708,560,11664,7977,13080,4869,493,179,1999,14761,1535,2454,30908,1231,5436,223,7838,353,8549,792,698,6120,15624,2251,4482,6359,2039,1749,2719,4866,19375,5650,30940,30930,342,2519,176,6834,262,288,190,7308,30943,15241,10748,30935,12019,18321,1129,12498,1429

- 2026-07-21/22 session start (agent):
  - fdrop bakeoff landed (box ~/fdrop/fdrop_results.txt): fdrop_b50 seg .9548
    gold .9120 (beats fold_b50 on BOTH); fdrop_raw .9602/.9032 (seg -3.3m,
    gold +3.3m vs fold_raw); soup_fdrop .9593/.9080 — beats soup_fold mean
    +0.0022. p=0.3 field dropout = best known full-run recipe for b50/soup.
    p-sweep screen is queued (prepare cache shared across p).
  - Ported main-tree fork-pool parallel eval harness (commit 52cf74c);
    reproduction check vs soup_fold pending first GPU-free window.
  - **CE distillation lever BLOCKED as-is**: ce_scores*.parquet key on
    (uuid5 query_id, md5-ish offer_id_b64) — a different id space. Bridge via
    queries_offers_labeled.parquet (has query_term + item_id in fold format)
    joins 204k CE rows, but only 0.5% of fold-train rows covered (1,391 rows;
    59/9447 queries have a covered pos+neg margin pair). mined_pairs_for_ce
    lacks item_id. Real fallback: re-score fold train pairs with the CE
    teacher on-box (~285k pairs, cheap GPU job) — parked; needs a look at
    which CE ckpt is the prod teacher + budget etiquette.
  - base_b50 in-budget screen baseline launched (run base_b50).

## State of the art (the baselines you must beat)

fold_raw / fold_b50 / soup_fold (see results.tsv) — trained 2026-07-21:
kitchen-sink doc fields (name, ean, artnos, customer artnos, brand, vendor,
leaf categories, s2class labels, keywords, features, description; template in
configs/data/splade_sink.yaml) + symmetric text folding (both query and doc
lowercase+diacritic-stripped) + uniform soup. Checkpoints on vastai0:
`~/simplesystem-embedding/checkpoints/{fold_raw,fold_b50}/best-*.ckpt`,
`checkpoints/soup_fold.ckpt`.

## Established findings (2026-07-21 session — do not rediscover)

- **Folding was worth +8-9pt gold R@100** (sink_raw unfolded .8171 → fold .90+).
  Symmetry is load-bearing; eval rows are folded. Never unfold.
- **Kitchen-sink fields beat the name/ids-only render** (+2pt seg, +2.7 gold
  over the previous generation). description carries most tokens; 15% of
  renders exceed the 512-token cap (description is last in template, so the
  cut eats prose).
- **Warm start from data/v1a_best.ckpt is mandatory** — gbert-base from
  scratch on these sets degenerates (docnnz ~4k, R@100 0.16).
- **raw vs b50 blend**: raw wins seg, b50 wins gold (small but consistent);
  soup captures both. Soups have beaten members in every generation.
- **batch 512 @ seq 512 OOMs the H100** ([B,seq,vocab] logits ≈32 GiB);
  256×2 accum works but HALVES the in-batch negative pool vs the older
  bo_* generation — an untested confound, see levers.
- **Backbone dropout is implicitly 0.1** (HF config default; never swept).
- **Deployed top-256 vectors waste ~1/3 nnz on stopwords/punctuation** —
  `model.stopword_mask_ids` exists; a masked run measured FLOPS 2.3 at flat
  recall on the older generation. Not yet tried on the fold generation.
- **Internal val (val_full_catalog_ndcg_at_5) drifts from the harness** —
  directional only. fold_raw internal 0.8304 / sink_raw 0.7732.
- **field dropout (p=0.3, enrichment fields only)**: implemented + verified in
  this branch (collate-path re-render, val untouched, cache shared across p).
  The fdrop bakeoff results (fold parquets) landed after this scaffold — check
  `vastai0:~/fdrop/fdrop_results.txt` before re-running that experiment.
- Eval harness trap: `--dist` MUST be the `_fold` distractors file
  (eval_remote.sh does this; the script default crashes on missing columns).
- Data quality known-issues (cheap levers): ~0.4% of descriptions are
  pipe-separated attribute dumps; 770 offers have raw HTML tags inside
  features_text (the cleaner only strips description HTML); some vendors'
  keywords contain CSS debris (rare, 0.02%).

## Timing facts (Phase-0 inputs, measured 2026-07-21)

- Full 8-epoch raw_fold run: ~2h10m wall on the H100 (≈4,100 steps at
  batch 256×2 accum, ~31 steps/min). b50_fold ≈ 60% of that. A 45-min run
  gets ~1,300-1,400 steps ≈ 2.5 epochs of b50 — enough for direction, and why
  throughput work compounds.
- Dataset prepare (cold): renders+tokenizes 285k rows single-process inside
  the trainer, several minutes; cached afterwards (.cache/prepared_dataset in
  the run cwd — ~/ar_splade keeps its own cache, so the FIRST run pays it).
- Eval harness: seg ~15-20 min, gold ~25 min; GPU sits at 0% for long CPU
  phases (render+tokenize of 39k/150k docs, single-process). Doc pools are
  frozen → pre-tokenization cache is the obvious big win (likely →3-5 min
  GPU-bound evals). Encode itself: 36.5k docs in 129s at seq 512 (measured on
  the v2.1 eval), ~150k docs ≈ 9 min.
- Render length distribution: p50 299 / p90 619 / p95 801 tokens (pre-cap) —
  pad-to-longest on shuffled batches wastes ~40% of compute; length-bucketing
  is a real training-speed lever.
- Full-catalog validation during training is the other hidden cost (runs
  every epoch; encodes the whole val catalog at encode_batch_size 32).
- encode-server throughput reference: H100 ceiling ~6k docs/s at seq 256 with
  bf16 + pad-to-longest + GPU top-k (see splade-inference-throughput memory) —
  the eval encode path is far from that today.

## Untried levers (ranked guesses, cheapest screens first)

1. **Stoplist mask on the fold generation** (model.stopword_mask_ids) — frees
   ~85/256 doc slots; expect flat-to-plus recall, lower FLOPS.
2. **MarginMSE/KL distillation from the CE teacher**
   (data.ce_scores_path=/home/max/simplesystem-embedding/data/ce_scores*.parquet,
   loss_weights.margin_mse>0) — the CE is the strongest relevance signal in
   the stack; SPLADE-v3-style recipe. Teacher scores cover the OLD gold pairs
   — check join coverage vs the fold parquets first (query_id+offer_id_b64).
3. **Dropout sweep** (model config hidden/attention dropout 0.0/0.1/0.2 —
   needs a small build_encoder change to pass overrides).
4. **gbert-large warm start** (checkpoints/splade-large-* on box; large gold-ft
   scored 0.8024 internal on the old harness). VRAM at seq 512: mind batch.
5. **Restore full in-batch negatives**: batch 384 or 512 with
   max_offer_length 384 instead of 512 (renders p90 ≈ 620 tokens — measure the
   truncation cost first), or contrastive across accumulation steps.
6. **Self-mined hard negatives** from the train pool with fold_raw
   (data.semi_hard_negatives_path) — the current runs use in-batch only.
7. **Continued training from soup_fold** (short low-lr runs on raw_fold).
8. **Field-order / template surgery** (e.g. keywords before categories;
   drop category_paths L1 duplicate — category_leaf_text supersedes it).
9. **n_pos/n_neg sampling shape** (2/4 never swept), triplet_margin,
   similarity_scale.
10. **Query-side augmentation** (restore umlauts/case variants of train
    queries as extra rows — must NOT touch eval; derive from train parquet
    only).

## Keep chain (final — full session lineage)

Full-length 8-epoch reference models (context, trained pre-session; NOT the
in-budget comparison point):

| model | seg R@100 | gold R@100 | seg R@10 | gold R@10 | FLOPS_w | qnnz |
|---|--:|--:|--:|--:|--:|--:|
| fold_raw (8ep) | .9635 | .8999 | .4316 | .4866 | 2.2 | 44 |
| fold_b50 (8ep) | .9523 | .9101 | .4209 | .4927 | 2.7 | 42 |
| soup_fold (8ep) | .9588 | .9040 | .4293 | .4925 | 2.3 | 39 |

In-budget keep chain (each row = the accepted base the next experiment built on):

| # | model | budget | seg R@100 | gold R@100 | gold R@10 | FLOPS_w | qnnz | knob vs prev | mean Δ |
|--:|---|---|--:|--:|--:|--:|--:|---|--:|
| 0 | base_raw | 45m | .9591 | .8921 | .4866 | 2.86 | 42 | reference recipe, in-budget root | — |
| 1 | k1_fdrop30_raw | 45m | .9609 | .8971 | .4935 | 2.32 | 41 | + field dropout p=.3 + bucketing | +.0034 |
| — | *(Max re-root)* | — | — | — | — | — | — | adopt k2 config: desc0 + seq256 + batch512/accum1 + fdrop30 + bucketing | — |
| 2 | b15_root | 15m | .9583 | .8956 | .4941 | 2.78 | 39 | k2 config, budget 45m→15m (~1.9 ep) | new root |
| 3 | **s7_lr3e5** | 15m | **.9590** | **.9046** | **.5010** | 2.64 | 31 | + lr 1.5e-5 → 3e-5 | **+.0048** |

**s7_lr3e5 = session-final best** (gold R@10 .5010 is the all-time high across the
entire SPLADE program). It's a 15-minute / ~1.9-epoch model that matches the
2-hour fold_raw on seg and beats it on gold — the compounded throughput +
recipe wins (bucketing, desc-free seq-256, restored batch-512 negatives, field
dropout, hot LR) closed the gap to full-length training at ~1/8 the compute.

Rejected / null branches (not in the chain, mechanisms in the log above):
s2_mask100 (train-time stopword mask — flat); s3/s4_distill (additive MarginMSE
blows FLOPS — needs KL/normalized margins); soup_k1s5 (below keep bar, but held
gold R@10 record briefly); k2 as a keep (R@100 tie — Max promoted its *config*
instead); s6_ep2 (full anneal in-budget breaks FLOPS guardrail).
