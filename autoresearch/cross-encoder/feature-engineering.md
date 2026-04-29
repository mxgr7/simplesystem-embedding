# cross-encoder feature engineering — experiment plan (2026-04-29)

What to run, in what order, what to expect, and how to decide. Branch:
`autoresearch/apr29-ce`. Champion to beat: `01-baseline-naive-retrain-1ep`
(commit `1d721e5`) — micro_f1 = **0.8851**, macro_f1 = **0.7802** on the new
ESCI-merged val (76,048 rows, 1,950 queries).

## What's already shipped

- `src/cross_encoder_train/features.py` — `FeatureExtractor` and pure
  `feature_token_names()` consumed by both DataModule (tokenizer add) and
  Model (`resize_token_embeddings`).
- `src/cross_encoder_train/specs.py` — regex registry for production-derived
  spec rules (thread_m, dimensions, mm, voltage, …).
- `configs/data/cross_encoder.yaml` — `features:` block, **`enabled: false`**
  by default. Three slots:
  - `ean` — 3-state (NONE / MATCH / MISMATCH), GTIN checksum.
  - `article` — 5-state (NONE / EXACT / SUBSTRING_ONLY / MISMATCH /
    OFFER_INVALID), validates over `article_number` and
    `manufacturer_article_number`. Already collapses the former
    article+shape+brand slots (commit `6230590`).
  - `spec` — 3-state, currently `enabled: false` in the per-slot config.
- `explore_feature_noise.py` — preflight audit (no GPU).

So the implementation is in place; this doc is purely about **what to run
next** and **how to read the results**.

## Why the prior expected-value calc has shifted

The original feature-engineering motivation (~47% of production queries are
identifier-shape, ~15% spec-shape) was estimated against the **old** dataset
where Exact was 80% of val. On the new dataset:

- Exact prior: 51.8% (down from 82.6%) — identifier features have less
  free leverage on the dominant class.
- Substitute support: 9,660 rows (+2.3×) with rich intra-query contrast —
  this is where ART_MISMATCH should bite (textually-similar offer with a
  different identifier ⇒ Substitute, not Exact).
- Complement F1 = 0.585 with only 1,394 val rows — the new bottleneck.
  **None of the EAN/ART/SPEC features are designed to help Complement** —
  Complement is a semantic, not lexical, distinction. Don't expect macro
  lift to come from there.
- Irrelevant F1 = 0.896 already; ART_MISMATCH could push it higher by
  pulling confused Substitute/Exact predictions over.

**Net headroom estimate**: features will primarily redistribute mass at
the Exact↔Substitute boundary. Plausible lift: **+0.005 to +0.020 macro
F1**, smaller on micro. Bigger lifts are possible only if the gelectra
encoder is substantially under-using identifier matching today (testable —
see exp 0).

## Pre-flight: run the audit first (no GPU, ~1 min)

Before any GPU run, get the actual fire rates and noise rates on the full
training set. The audit answers the policy questions the config can't.

```bash
PYTHONPATH=src uv run python explore_feature_noise.py \
  --config configs/data/cross_encoder.yaml \
  --sample 200000 --enable ean article spec
```

Things to read off the report and pin in this doc as **audit/<slot>**:

- `query_present` rate — fraction of queries where the slot fires.
  - Below ~3%: the token will be `[X_NONE]` for >97% of training pairs;
    the model has very little signal to learn from. Consider dropping
    that slot.
  - Above ~10%: the model has plenty of opportunity to learn.
- `offer_invalid` rate — fraction of offers where the offer-side field
  fails validation. Drives the `on_offer_invalid` policy decision:
  - Low (<5%): both `none` and `mismatch` produce similar token streams —
    pick `none` (safer; doesn't punish offers we just can't validate).
  - High (>15%): `none` masks too many real signals; flip to `mismatch`,
    but expect noise.
- For ART specifically: ratio of `EXACT` to `SUBSTRING_ONLY` to
  `MISMATCH` per label. **Look for the monotonicity**: EXACT should
  concentrate in Exact, MISMATCH should concentrate in Substitute /
  Irrelevant. If the per-label distribution is flat, the feature won't
  separate classes — abort that slot.
- For SPEC: per-class token distribution under both policies. SPEC is
  the noisiest slot (regex over free-text); if MATCH and MISMATCH look
  identical across labels, drop it.

Persist the audit numbers in `data-insights.md` so re-runs don't have to
re-derive them.

## Experiments

Sized for the ~42-min full-epoch budget of the ESCI-merged dataset. The
champion (`1d721e5`) is the head-to-head opponent for every run below.
Use the screen20 preset (commit `8af740a`,
`limit_train_batches=0.49 max_epochs=1`) for fast screening when noted —
it costs ~21 min and is the fairest direct comparison to a screen20
re-baseline.

**Common knobs (don't change unless explicitly noted)**: gelectra-large,
bs=32, max_pair=512, bf16-mixed, lr=1.5e-5, focal_gamma=2.0,
warmup_ratio=0.33, head_dropout=0.1.

### exp 0 — token-presence sanity check (5 min, no GPU)

Before training, instrument the model to print the gradient norm of the
new feature-token embedding rows after one forward+backward pass on a
single batch. **Goal**: confirm the model can move them. If they're
frozen (e.g. due to compile-graph issue or resize-then-freeze bug),
every downstream experiment is dead.

```python
# Pre-training assertion. Add temporarily to train.py or model.py.
# After model construction:
old_size = encoder.config.vocab_size  # before resize
new_ids = list(range(old_size, encoder.embeddings.word_embeddings.num_embeddings))
print("new token ids:", new_ids)
# After one optimizer step, compare embedding rows pre/post.
```

Skip if the pre-training tokenizer log already shows
`features/added_tokens >= 1` and the resize completed.

### exp 1 — features ON, EAN+ART only (full epoch, ~42 min) — **lead exp**

```bash
LD_LIBRARY_PATH=... uv run cross-encoder-train \
  data.features.enabled=true \
  data.features.spec.enabled=false \
  trainer.max_time=00:00:50:00 trainer.max_epochs=1 \
  logger.run_name=02-features-ean-art > run.log 2>&1
```

**Hypothesis**: the strongest, lowest-noise features (EAN with checksum,
ART with EXACT/SUBSTRING distinction) lift macro F1 by redistributing
the Exact↔Substitute boundary.

**Expected**:
- micro F1: −0.003 to +0.010 (could regress if ART_MISMATCH is too
  aggressive on Exact).
- macro F1: +0.005 to +0.020 (Substitute and Irrelevant primarily).
- Per-class targets to watch:
  - Substitute precision **up** (model now distinguishes "looks like
    Exact but identifier disagrees" from real Exact) — current 0.759.
  - Substitute recall: ambiguous; could go either way.
  - Exact recall: should hold (ART_EXACT reinforces matches).
  - Complement F1: ~unchanged (features don't address it).
  - Irrelevant F1: small uplift if ART_MISMATCH catches near-misses.

**Failure modes**:
- Macro flat or down → `on_offer_invalid: none` is masking too much, or
  ART_SUBSTRING_ONLY is dragging Substitute toward Exact. Run **exp 1b**:
  `data.features.article.on_offer_invalid=mismatch`.
- Both metrics regress → revisit token init. The HF default for
  `add_special_tokens` re-inits added rows to N(0, init_range²); if the
  encoder's existing embedding norms are far from that, the new tokens
  start as noise relative to the rest of the input. Switch to
  mean-of-existing init (small code change in model.py) and re-run.

**Look-fors beyond the headline metrics**:
- val confusion matrix (compute post-hoc from `validation_rows` if not
  logged): expect cell (Substitute_pred, Exact_true) to drop, and
  (Exact_pred, Substitute_true) to drop.
- val MRR for Exact (`val/rank/exact_mrr`) — if it drops below 0.953
  while micro is flat, the features are buying micro at the cost of
  ranking. That's a discard.
- VRAM: should be unchanged from baseline (resize adds ~6 KB of params).

### exp 2 — SPEC ablation (full epoch, ~42 min)

Only if exp 1 is `keep`. Otherwise skip; compounding two uncertain
features on top of a regression is wasted budget.

```bash
... data.features.enabled=true \
    data.features.spec.enabled=true \
    logger.run_name=03-features-ean-art-spec
```

**Hypothesis**: SPEC adds orthogonal information (dimensions, voltage,
DIN codes) that EAN/ART can't capture. SPEC_MISMATCH is most relevant
for the Substitute boundary (e.g. "M8 screw" query, "M10 screw" offer
⇒ Substitute, not Irrelevant — the existing token will say MISMATCH
which the model can read as "wrong size, otherwise the same product").

**Expected**: smaller marginal lift than exp 1; SPEC is noisier.
- macro: +0.000 to +0.008 on top of exp 1.
- If SPEC_MATCH and SPEC_MISMATCH have similar per-class distributions
  in the audit, expect ~0.

**Failure modes**:
- Regex false positives — `mm` rule firing on offer descriptions that
  list every available size, leading to spurious MATCHes. Mitigation:
  set `data.features.spec.on_offer_invalid=mismatch` is a **bad** idea
  for SPEC (it amplifies noise); prefer pruning the rule list.
  Particularly suspect: `mm`, `cm`, `volume_l` — these match common
  size mentions in product descriptions. If exp 2 regresses, drop those
  three rules and re-run.

### exp 3 — policy ablation, EAN+ART, on_offer_invalid=mismatch

Only meaningful if the audit shows a non-trivial `offer_invalid` rate
(>5%) for EAN or ART. Otherwise skip — both policies produce nearly
identical token streams.

```bash
... data.features.enabled=true \
    data.features.spec.enabled=false \
    data.features.ean.on_offer_invalid=mismatch \
    data.features.article.on_offer_invalid=mismatch \  # n/a — article has 5 states; this is a no-op
    logger.run_name=04-features-mismatch-policy
```

**Hypothesis**: explicit MISMATCH on noisy offers gives the model a
weaker but useful signal vs the silent NONE fallback. Net effect
depends on whether the noise correlates with class.

**Expected**: ±0.005 swing on both metrics. Honest experiment, not a
high-confidence win.

### exp 4 — features ON + class-weights ON (full epoch, ~42 min)

A combination test, only after exp 1 and the standalone class-weight
experiment from `baselines_new_dataset.md` §"Proposed next experiments".
The two ideas could be additive (different mechanisms — class weights
fix gradient imbalance, features add inductive bias) or could collide
(class weights up-weight Complement gradients ~16×, but features barely
help Complement, so the class-weight hit on Exact precision could
dominate).

Run only if both **standalone** experiments pass (`keep` on at least
one metric without regression on the other).

### exp 5 — article slot only, then EAN slot only (screen20, ~21 min each)

Decomposition study to attribute exp 1's lift (or loss) to a single slot.

```bash
... data.features.enabled=true \
    data.features.ean.enabled=false \
    data.features.article.enabled=true \
    trainer.limit_train_batches=0.49 \
    logger.run_name=05a-features-art-only
```

**Hypothesis**: ART carries most of the signal (higher fire rate, 5
states vs 3, validates against 2 fields). EAN is rarer but very high-
precision when it fires.

**Expected**:
- ART-only: ~80–100% of exp 1's macro lift.
- EAN-only: ~10–30% of exp 1's macro lift.
- If they sum to *more* than exp 1, there's saturation/redundancy.
- If they sum to *less*, they're mildly complementary.

This is diagnostic, not a champion candidate — screen20 budget is
intentional.

## What to validate (regardless of which exp wins)

Before declaring any feature configuration a `keep`, run these checks:

1. **Confusion matrix delta**: which cells moved? Macro lift driven by
   one class only is fragile; macro lift spread across 2+ classes is
   robust.
2. **Ranking metrics**: `val/rank/ndcg_at_5` and `val/rank/exact_mrr`
   should not regress more than 0.002. If micro is up but exact_mrr is
   down, the features are corrupting Exact's calibrated probability —
   discard.
3. **Stat plausibility**: log `features/{slot}/{state}` counters from
   `dataset_stats`. If `ean/match` rate at training time differs by >2×
   from the audit on the val split, there's a data-skew bug.
4. **Token attention**: optional but cheap — pull attention weights from
   the [CLS] head at the new token positions on a few val examples. If
   attention is uniformly tiny, the encoder is ignoring the tokens and
   the lift came from somewhere else (e.g. seed perturbation).
5. **Replicate with seed=43**: any lift <0.005 must be replicated. The
   noise floor for a single seed at this dataset size is around ±0.002
   on micro and ±0.004 on macro (estimate from old data; refresh on new
   if a near-noise winner appears).

## Risks specific to feature tokens

- **Train/serve skew (very high blast radius)**: the saved memory
  `feedback_inference_features.md` warns that retrieval metadata isn't
  available at inference time. EAN/ART/SPEC are derived purely from
  `(query, offer)` text, so they're inference-safe — but only if the
  inference path runs the *same* `FeatureExtractor` against the *same*
  config. Ship-day checklist:
    - The `FeatureExtractor` config must be persisted next to the
      checkpoint (it already lands in MLflow params via
      `save_hyperparameters` — verify by reading back a run's params).
    - The inference renderer must call `FeatureExtractor.extract` and
      prepend tokens with the same separator (` `).
    - Tokens must be in the tokenizer vocab. `add_special_tokens` writes
      them to `tokenizer_config.json`; the inference loader must
      `from_pretrained` from the run's saved tokenizer dir, not the
      base model's.
- **SUBSTRING_ONLY directionality**: the implementation accepts
  `query_cand in offer_id` OR `offer_id in query_cand`. The latter is
  rare but creates an asymmetry — a typo'd query that strictly contains
  the offer's article number gets SUBSTRING_ONLY too. Audit a sample
  of SUBSTRING_ONLY hits to confirm the right semantics.
- **GTIN checksum strictness**: 8/12/13/14 only. Some real EANs in
  German B2B catalogs are 7-digit internal codes; these will be
  silently `OFFER_INVALID`. Acceptable for now (consistent with
  production), but flag in the audit.
- **Compile cache**: `model.compile=true` is on. After
  `resize_token_embeddings`, the compile graph is rebuilt — first epoch
  overhead may be longer. Not a correctness issue.
- **Token init quality**: HF `add_special_tokens` initializes new rows
  with `N(0, σ²)` where σ matches the encoder's `initializer_range`
  (~0.02). The mean of gelectra's existing embeddings has a ~0.05 norm
  — new tokens start ~2σ off. Usually fine for fine-tuning, but if
  exp 1 regresses with no other obvious cause, override the init to
  the mean of existing rows.

## Decision tree (one screen)

```
audit numbers in hand?
  no  -> run audit, write data-insights.md
  yes -> exp 1 (EAN+ART, 42 min)

exp 1 macro lift >= +0.005, micro within noise?
  yes -> keep, branch advances. go exp 2 (add SPEC).
  no  -> exp 1b (policy=mismatch on EAN). still bad? discard, skip ahead
         to exp 5 to attribute the loss.

exp 2 macro lift >= +0.003 on top of exp 1?
  yes -> keep, advance. consider exp 4 (features + class_weights).
  no  -> revert SPEC, keep exp 1 result, skip exp 3.

exp 5 (decomposition) only after exp 1 settled — diagnostic, not a keep.
```

## Out of scope here (separate threads)

- 2-epoch run, focal_gamma sweep, bs=64, use_class_weights — owned by
  `baselines_new_dataset.md` §"Proposed next experiments". Run those
  first if the goal is "best possible champion"; run features first if
  the goal is "validate the feature-token investment".
- Brand dictionary — was dropped in commit `6230590` when the article
  slot absorbed shape+brand. Not on the roadmap unless audit reveals a
  large fraction of brand-shaped queries that ART misses.
- Group features (production-style aggregations across an offer's
  category): explicitly not addressed by this design and not in scope.
