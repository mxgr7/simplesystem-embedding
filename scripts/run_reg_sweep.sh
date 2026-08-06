#!/bin/bash
# Sparsity-regularizer screening sweep: DF-FLOPS and query-side L1.
#
#   run_reg_sweep.sh <box> [arm ...]        # default: every arm, in order
#   run_reg_sweep.sh vastai0 probe ctl      # just those two
#
# WHY
#   ES latency is set by the union of the query's posting lists, not by the dot
#   product. Measured on the live index: prod_soup sits at FLOPS_w 1.28 (SPLADE-
#   v3 SOTA is ~1.2) and STILL visits 85.9% of the corpus. Optimising FLOPS
#   further is not the lever.
#
#   Two mechanisms from the literature, both testable here:
#     l1       query-side L1. FLOPS penalizes a^2, whose gradient vanishes as
#              a -> 0, so it shrinks magnitudes without zeroing dims — our own
#              sweep moved query nnz 71 -> 80.5 while cranking lambda x20. L1
#              has constant gradient and actually reduces nnz. Coverage is
#              exponential in nnz, so this is the highest-leverage knob:
#              post-hoc truncation to 8 tokens already measured 3.11x on ES.
#     df_flops weights each dim's penalty by a logistic of its document
#              frequency (arXiv 2505.15070), attacking breadth not magnitude.
#
# Each arm trains, then evals seg+gold via eval_remote.sh. Progress is teed to
# /workspace/pipeline/reg_sweep.log with per-arm elapsed time.
set -euo pipefail

BOX=${1:?usage: run_reg_sweep.sh <box> [arm ...]}; shift
WT=$(cd "$(dirname "$0")/.." && pwd)
LOG=/workspace/pipeline/reg_sweep_${BOX}.log   # per-box: 4 drivers run concurrently
DATA=/home/max/simplesystem-embedding/data
REMOTE_AR=/home/max/prod_splade

# The locked lr=3e-5 folded base recipe. Do not vary these across arms — the
# whole point is that only the regularizer differs.
BASE=(
  model=splade data=splade_sink
  model.init_checkpoint=$DATA/v1a_best.ckpt
  data.path=$DATA/splade_train_raw_desc0_folde.parquet
  data.max_offer_length=256
  data.field_dropout_p=0.3
  data.length_bucketing=true
  data.batch_size=512
  trainer.accumulate_grad_batches=1
  optimizer.lr=3e-5
  model.flops_warmup_steps=300
  model.fold_vocab_mask=true
  trainer.validation_metric=ndcg_at_5
  trainer.encode_batch_size=32
)

# Lambdas CALIBRATED from probe2 (2026-07-28, lambda=0, warm-started), measured
# with a REAL df_ema. Magnitudes at step 509:
#
#   reg            query    doc     vs flops
#   flops          21.625   438.4    1.000
#   df_flops       21.615   435.4    0.9995 / 0.9932   <- ~1.0x, NOT 0.039x
#   df_paper       21.612   434.3    0.9994 / 0.9908
#   l1             28.285   780.9    1.308  / 1.782
#
# => df_flops needs the SAME lambda as flops; l1 needs ~1/1.3 to ~1/1.8.
#
# THE EARLIER 0.039x WAS AN ARTIFACT, and it invalidated round 1's df arms.
# _update_document_frequency used to early-return unless a df regularizer was
# active, so on the flops-only probe df_ema stayed all zeros and the logged
# df_flops was sigmoid(alpha*(0-beta)) * flops = 0.0392 * flops — a constant
# rescale of FLOPS mistaken for a measurement. Lambdas were then set 25x too
# high; dfl_hard ran at ~260x intended pressure. That, not the activation shape,
# is why round 1 saw doc nnz collapse 223 -> 71.
#
# Why df_flops ~= flops in magnitude: FLOPS sums the SQUARES of mean activations,
# so it is dominated by the highest-magnitude dims — which are exactly the
# high-df dims the weight sends to ~1. The df weighting removes gradient from
# rare dims without shrinking the total.
#
# alpha/beta (and df_half/df_sharp) are FIXED across the df arms so the sweep
# varies pressure only; changing them also changes the regularizer's magnitude
# and would confound the two.
#
# NOTE for next time: every regularizer is now logged every step, so no lambda=0
# probe is needed — read the ratios off the `ctl` arm at its own operating point.
# probe2 sat at offer_avg_nnz 1153 vs ~217 for a regularized arm, so its df_ema
# describes a ~5x denser model than any arm actually occupies.
declare -A ARM
ARM[probe]="model.flops_lambda_q=0 model.flops_lambda_d=0 trainer.max_epochs=1"
ARM[ctl]="model.flops_lambda_q=5e-4 model.flops_lambda_d=3e-4"
# beta=0.08 / alpha=40 calibrated on the measured df of prod_soup query tokens
# (p10=0.011, p50=0.076, p90=0.207) -> 17x weight separation p10->p90.
# alpha=20,beta=0.05 was only 3.1x — indistinguishable from plain FLOPS.
DFA="model.df_alpha=40 model.df_beta=0.08"
ARM[dfl_mild]="model.reg_type_q=df_flops model.reg_type_d=df_flops model.flops_lambda_q=5e-4 model.flops_lambda_d=3e-4 $DFA"
ARM[dfl_mid]="model.reg_type_q=df_flops model.reg_type_d=df_flops model.flops_lambda_q=2e-3 model.flops_lambda_d=1.2e-3 $DFA"
ARM[dfl_hard]="model.reg_type_q=df_flops model.reg_type_d=df_flops model.flops_lambda_q=5e-3 model.flops_lambda_d=3e-3 $DFA"
ARM[ql1_mild]="model.reg_type_q=l1 model.flops_lambda_q=3.8e-4 model.flops_lambda_d=3e-4"
ARM[ql1_hard]="model.reg_type_q=l1 model.flops_lambda_q=1.4e-3 model.flops_lambda_d=3e-4"
ARM[combo]="model.reg_type_q=l1 model.reg_type_d=df_flops model.flops_lambda_q=1.4e-3 model.flops_lambda_d=1.2e-3 $DFA"

# --- round 2 (2026-07-28), driven by the literature review --------------------
# ALL ROUND-1 dfl_* RESULTS ARE VOID — they ran at 26-260x intended pressure on
# the artifact lambdas (see the calibration block above). Their doc-nnz collapse
# was over-regularization, NOT the sigmoid-vs-generalized-logistic shape I first
# blamed. dfl_paper is still worth running to compare activation shapes, but now
# at correctly matched pressure, and the shape hypothesis is untested rather than
# supported. Round-1 ql1_* results stand: L1's calibration never went through the
# broken path (measured 1.31/1.78 vs the 1.4-1.7 assumed).
#
# Round 1 found query-L1 inert at matched pressure and only partly effective at
# 4x (qnnz 15.0 -> 10.6) — which is what Lassance & Clinchant predict for a
# SHARED encoder ("nothing to differentiate between them"). `untied_*` tests
# whether untying unlocks the query lever; untied_ctl is the matched baseline
# that isolates untying from the regularizer. NOTE untied arms build a second
# encoder, which shifts the RNG stream — untied_ctl vs ctl is NOT a seed-matched
# comparison, only a same-recipe one.
#
# Baseline caveat for any "L1 reduced query nnz" claim: on the lambda=0 probe
# query_avg_nnz still fell 117 -> 31, so the contrastive objective alone
# sparsifies queries heavily. The comparison must be against the lambda=0
# trajectory, not against step 0.
#
# encode_batch_size halved on untied arms: 2 encoders, and validation already
# peaked at 79.7/81.5 GB on the tied model.
PAPER="model.df_activation=paper model.df_half=0.10 model.df_sharp=10"
UNTIE="model.untied_encoders=true trainer.encode_batch_size=16"
ARM[dfl_paper]="model.reg_type_q=df_flops model.reg_type_d=df_flops model.flops_lambda_q=5e-4 model.flops_lambda_d=3e-4 $PAPER"
ARM[untied_ctl]="$UNTIE model.flops_lambda_q=5e-4 model.flops_lambda_d=3e-4"
ARM[untied_ql1]="$UNTIE model.reg_type_q=l1 model.flops_lambda_q=1.4e-3 model.flops_lambda_d=3e-4"
ARM[untied_combo]="$UNTIE model.reg_type_q=l1 model.reg_type_d=df_flops model.flops_lambda_q=1.4e-3 model.flops_lambda_d=3e-4 $PAPER"

# THE decisive control: plain FLOPS at the SAME lambda as dfl_mid/dfl_hard.
# Without these, a coverage reduction under df_flops x4/x10 cannot be
# attributed to the df weighting rather than simply to more pressure.
ARM[flops_mid]="model.flops_lambda_q=2e-3 model.flops_lambda_d=1.2e-3"
ARM[flops_hard]="model.flops_lambda_q=5e-3 model.flops_lambda_d=3e-3"

# Seed replicates. Running ctl twice already showed ~3pp coverage / ~0.005 R@100
# run-to-run spread under trainer.deterministic=false; these turn that anecdote
# into an estimate, and test whether the L1 coverage win survives a reseed. The
# L1 result is the only one likely to hold, so it is the one worth replicating.
ARM[ctl_s2]="model.flops_lambda_q=5e-4 model.flops_lambda_d=3e-4 seed=1337"
ARM[ql1_hard_s2]="model.reg_type_q=l1 model.flops_lambda_q=1.4e-3 model.flops_lambda_d=3e-4 seed=1337"

# --- pushing untied_ql1, the only point above the plain-FLOPS frontier --------
# It scores 0.9030 at 110.3M postings where the interpolated FLOPS frontier
# predicts ~0.894 (+0.009, ~2x the noise floor), and it gets there by cutting
# QUERY nnz (10.8 vs 14-15) rather than doc nnz (221, ~unchanged).
#   _s2   replicate: the base result is n=1 and ctl swung 23% in postings by seed
#   _x2   more L1 pressure: coverage is EXPONENTIAL in query nnz (post-hoc
#         truncation maps qnnz 8 -> 54.9% cov, 6 -> 44.7%, 4 -> 32.0%)
#   _dfx  add doc-side pressure: L1 cuts the query count, FLOPS cuts doc df —
#         different terms of postings = sum_{j in q} df_j, so they should
#         multiply. Uses plain FLOPS on the doc side; df_flops is refuted.
ARM[untied_ql1_s2]="$UNTIE model.reg_type_q=l1 model.flops_lambda_q=1.4e-3 model.flops_lambda_d=3e-4 seed=1337"
ARM[untied_ql1_x2]="$UNTIE model.reg_type_q=l1 model.flops_lambda_q=2.8e-3 model.flops_lambda_d=3e-4"
ARM[untied_ql1_dfx]="$UNTIE model.reg_type_q=l1 model.flops_lambda_q=1.4e-3 model.flops_lambda_d=1.2e-3"

#   _frozen  doc encoder held at the warm-start weights, query encoder trained
#            with L1. If this works, the existing 113M-doc index stays valid and
#            deployment is "swap the query encoder" — no ~6.5h re-encode. Also
#            halves training cost. untied_ql1 already gets its entire win from
#            the query side (doc nnz 221, ~unchanged), so the doc-side training
#            may have been contributing nothing anyway.
ARM[untied_ql1_frozen]="$UNTIE model.freeze_doc_encoder=true model.reg_type_q=l1 model.flops_lambda_q=1.4e-3 model.flops_lambda_d=3e-4"

#   _x4        push L1 further still: x2 reached qnnz 9.3 at 96.3M postings for
#              -0.0003 R@100, so the query lever had not yet run out of road.
#   _frozen_x2 frozen doc encoder at the x2 pressure, if _frozen holds up.
ARM[untied_ql1_x4]="$UNTIE model.reg_type_q=l1 model.flops_lambda_q=5.6e-3 model.flops_lambda_d=3e-4"
ARM[untied_ql1_frozen_x2]="$UNTIE model.freeze_doc_encoder=true model.reg_type_q=l1 model.flops_lambda_q=2.8e-3 model.flops_lambda_d=3e-4"

#   _x8  x4 reached qnnz 8.1 / 80.0M postings at R@100 0.9013 (BETTER than ctl),
#        so the query lever still has not saturated. Push once more.
ARM[untied_ql1_x8]="$UNTIE model.reg_type_q=l1 model.flops_lambda_q=1.12e-2 model.flops_lambda_d=3e-4"

# Saturation search. x8 reached qnnz 6.9 / 66.1M postings at R@100 0.9031 —
# still no quality cost and still falling, so keep doubling until the curve
# turns. Base lambda_q for untied_ql1 is 1.4e-3.
ARM[untied_ql1_x16]="$UNTIE model.reg_type_q=l1 model.flops_lambda_q=2.24e-2 model.flops_lambda_d=3e-4"
ARM[untied_ql1_x32]="$UNTIE model.reg_type_q=l1 model.flops_lambda_q=4.48e-2 model.flops_lambda_d=3e-4"
ARM[untied_ql1_x64]="$UNTIE model.reg_type_q=l1 model.flops_lambda_q=8.96e-2 model.flops_lambda_d=3e-4"
ARM[untied_ql1_x128]="$UNTIE model.reg_type_q=l1 model.flops_lambda_q=1.792e-1 model.flops_lambda_d=3e-4"

# --- settling the operating point -------------------------------------------
#   _tied     x32 WITHOUT untying. Untying was only ever tested at base pressure
#             (13% fewer postings, R@100 +0.0029 — both ~1-2x noise). It doubles
#             params to 220M, splade-service cannot load it, and 9 call sites
#             need migrating; if tied matches untied here, ship tied.
#   _s2       x32 and x8 are both n=1 and quality noise is +-0.008, so the
#             x8->x16 step-down may not be real.
#   _frozen_ps  the CORRECTED no-reindex test: freeze the doc encoder at
#             prod_soup (the model the deployed 113M-doc index was actually
#             built from). The earlier frozen arm froze v1a_best, the warm-start
#             backbone, which preserves nothing — doc nnz pinned at 256.0.
ARM[ql1_x32_tied]="model.reg_type_q=l1 model.flops_lambda_q=4.48e-2 model.flops_lambda_d=3e-4"
ARM[untied_ql1_x32_s2]="$UNTIE model.reg_type_q=l1 model.flops_lambda_q=4.48e-2 model.flops_lambda_d=3e-4 seed=1337"
ARM[untied_ql1_x8_s2]="$UNTIE model.reg_type_q=l1 model.flops_lambda_q=1.12e-2 model.flops_lambda_d=3e-4 seed=1337"
ARM[untied_ql1_frozen_ps]="$UNTIE model.freeze_doc_encoder=true model.init_checkpoint=$DATA/prod_soup.ckpt model.reg_type_q=l1 model.flops_lambda_q=1.12e-2 model.flops_lambda_d=3e-4"

# --- CONFIRMATION ROUND (2026-07-29): 8 epochs, the shipping configuration ----
# Everything above is 3-epoch SCREENING. These are the runs a ship decision may
# rest on, so they use the production run length (prod_soup was trained for 8)
# and are named distinctly — reusing a screening arm name silently overwrites
# that arm's row in out/reg_sweep_all_results.json, which already happened once
# when an 8-epoch ctl clobbered the 3-epoch one.
#
# ALL TIED. ql1_x32_tied matched untied x32 exactly on postings (48.1M both) at
# +0.004 R@100, so untying is refuted as a lever: it costs 220M params, blocks
# splade-service from loading the model, and needs 9 call sites migrated.
#
# WHY THE ANCHORS. The whole x8..x128 saturation curve was measured at 3 epochs
# and L1 sparsity is dose x TIME, so the curve moves at 8. Two opposite outcomes
# are live: longer training deepens sparsity (knee slides left, x16 wins) or it
# recovers quality at fixed lambda (knee slides right and x128 — 33.2M postings
# at 3 epochs — could beat x32 outright). One run at each end distinguishes
# them for the cost of a box that is idle anyway while the seeds train.
#
# THE SOUP. s1/s2/s3 are the seed replicates the quality claim needs (the x32
# vs ctl delta is +0.0017, well inside the +-0.008 seed band) AND the soup
# ingredients — same runs, one purchase. Soup them with scripts/soup.py and
# ACCEPT ON POSTINGS, not recall: L1 zeroes seed-specific dims, so weight
# averaging drifts the active set toward the union of the seeds'. At qnnz 5.4 a
# couple of surviving dims is a 40-60% cost regression. Souping was safe for
# prod_soup because FLOPS shrinks magnitudes rather than zeroing dims, so those
# seeds already shared a support; that precedent does NOT transfer to L1.
# Soup one flavor consistently (all final, or all best) — mixing a seed's
# epoch-4 best with another's epoch-8 final weakens the same-basin assumption
# the average relies on.
QL1="model.reg_type_q=l1 model.flops_lambda_d=3e-4"
ARM[e8_x32_s1]="$QL1 model.flops_lambda_q=4.48e-2"
ARM[e8_x32_s2]="$QL1 model.flops_lambda_q=4.48e-2 seed=1337"
ARM[e8_x32_s3]="$QL1 model.flops_lambda_q=4.48e-2 seed=2024"
ARM[e8_x64_s1]="$QL1 model.flops_lambda_q=8.96e-2"
ARM[e8_x128_s1]="$QL1 model.flops_lambda_q=1.792e-1"

# --- soup ingredients, matching prod_soup's OWN recipe -----------------------
# prod_soup is a 4-arm soup over {raw, b50} data mixes x {42, 1337} seeds — it
# varies the DATA MIX, not just the seed, which yields more independent members
# than reseeding alone. The first x32 soup was 2 seeds of raw only, and only
# because seed 42's checkpoint was destroyed by auto-teardown before it could be
# pulled. These three restore parity and then some: with the surviving
# raw/1337 and raw/2024 they make a 5-member soup over both mixes.
#   _s1r  re-runs raw/seed42 (the lost one; same config as e8_x32_s1)
B50="data.path=$DATA/splade_train_b50_desc0_folde.parquet"
ARM[e8_x32_s1r]="$QL1 model.flops_lambda_q=4.48e-2"
ARM[e8_x32_b50_s42]="$QL1 model.flops_lambda_q=4.48e-2 $B50"
ARM[e8_x32_b50_s1337]="$QL1 model.flops_lambda_q=4.48e-2 $B50 seed=1337"

# --- MATCHED-DOSE TEST: is the b50 mix actually better, or just less trained? --
# At the same lambda and same 8 epochs, b50 beat raw by +0.021 R@100 (0.8969 vs
# 0.8758) at +13% postings — and beat the FLOPS control on both axes, which no
# raw arm did. But b50 is the smaller parquet: 2400 optimizer steps vs raw's
# 4112, i.e. 1.713x LESS training. Since L1 sparsity is dose x time (§7), b50's
# edge is confounded with simply having received 1.713x less L1 pressure. Its
# higher query nnz (5.0-5.1 vs raw's 4.8) is exactly what under-dosing looks
# like.
#
# Two independent routes to matched dose; they agree only if the mix is real:
#   _dose   raise lambda by 1.713x (4.48e-2 -> 7.674e-2), keep 8 epochs. Same
#           data exposure, matched lambda*steps.
#   _ep14   keep lambda, train 14 epochs (2400 * 14/8 = 4200 steps ~= raw's
#           4112). Matched steps AND lambda, but more passes over less data —
#           so it trades the lambda confound for an overfitting one. Agreement
#           between _dose and _ep14 is the evidence; either alone is weak.
#   _dose_s2  seed replicate of the decisive arm, since one run cannot separate
#           a real effect from the +-0.003 seed band.
#
# READ IT AS: if _dose lands at raw's recall (~0.876) once postings match ~37M,
# the b50 "win" was dose all along and raw+lambda is the simpler lever. If it
# stays near 0.897 at 37M, the mix is genuinely better and b50 should be the
# production data mix.
ARM[e8_b50_dose]="$QL1 model.flops_lambda_q=7.674e-2 $B50"
ARM[e8_b50_dose_s2]="$QL1 model.flops_lambda_q=7.674e-2 $B50 seed=1337"
ARM[e8_b50_ep14]="$QL1 model.flops_lambda_q=4.48e-2 $B50 trainer.max_epochs=14"

ORDER=(probe ctl ctl_s2 ql1_hard_s2 flops_mid flops_hard dfl_mild dfl_mid dfl_hard ql1_mild ql1_hard combo dfl_paper untied_ctl untied_ql1 untied_combo untied_ql1_s2 untied_ql1_x2 untied_ql1_dfx untied_ql1_frozen untied_ql1_x4 untied_ql1_frozen_x2 untied_ql1_x8 untied_ql1_x16 untied_ql1_x32 untied_ql1_x64 untied_ql1_x128 ql1_x32_tied untied_ql1_x32_s2 untied_ql1_x8_s2 untied_ql1_frozen_ps e8_x32_s1 e8_x32_s2 e8_x32_s3 e8_x64_s1 e8_x128_s1 e8_x32_s1r e8_x32_b50_s42 e8_x32_b50_s1337 e8_b50_dose e8_b50_dose_s2 e8_b50_ep14)
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=("${ORDER[@]}")

EPOCHS=${EPOCHS:-3}   # screening length; winners get the full 8 x 2 seeds

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== reg sweep on $BOX: ${ARMS[*]} (EPOCHS=$EPOCHS) ==="
SWEEP_T0=$(date +%s)
for i in "${!ARMS[@]}"; do
  arm="${ARMS[$i]}"
  overrides="${ARM[$arm]:-}"
  if [ -z "$overrides" ]; then say "unknown arm: $arm (known: ${ORDER[*]})"; exit 1; fi

  # `probe` pins its own max_epochs; everything else takes the sweep default.
  case "$overrides" in *max_epochs*) ep="" ;; *) ep="trainer.max_epochs=$EPOCHS" ;; esac

  T0=$(date +%s)
  MARKER=/tmp/reg_sweep_marker_$arm
  ssh -F /workspace/.ssh/vastai.conf "$BOX" "touch $MARKER" >/dev/null 2>&1 || true
  say "--- arm $((i + 1))/${#ARMS[@]}: $arm ---"
  say "    $overrides $ep"
  # Per-arm failures must NOT abort the sweep: a dropped ssh during one arm's
  # 12-minute eval previously killed the whole run and left the GPU idle for 90
  # minutes. Each stage is guarded so the sweep continues to the next arm.
  if ! "$WT/autoresearch/splade/run_full.sh" "$BOX" "$arm" "${BASE[@]}" $overrides $ep 2>&1 | tee -a "$LOG"; then
    say "    !! launch FAILED for $arm — skipping to next arm"
    continue
  fi

  # run_full.sh returns as soon as the job is detached; wait for it to finish.
  # `|| true` on the poll so a transient ssh failure reads as "still running"
  # rather than terminating the loop (and, under set -e, the whole sweep).
  #
  # pgrep is scoped to THIS arm's run_name (bracket trick keeps it from matching
  # the ssh command itself), so a concurrent job on the box can't make us wait
  # forever or, worse, make us think our own arm is still alive.
  say "    training launched, waiting for completion..."
  sleep 60
  WAITED=0
  MAX_WAIT=$(( 3 * 60 * 60 ))   # 3h: ~4x the longest expected arm
  while [ "$(ssh -F /workspace/.ssh/vastai.conf "$BOX" \
      "pgrep -f 'embedding_train.tra[i]n.*run_name=$arm' >/dev/null && echo RUN || echo DONE" 2>/dev/null || echo RUN)" = RUN ]; do
    sleep 60
    WAITED=$(( WAITED + 60 ))
    if [ "$WAITED" -gt "$MAX_WAIT" ]; then
      say "    !! $arm exceeded ${MAX_WAIT}s — abandoning, NOT evaluating"
      break
    fi
  done
  say "    trained in $(( ($(date +%s) - T0) / 60 ))m"

  # A crashed arm leaves the PREVIOUS run's final-*.ckpt in place, and
  # eval_remote.sh's `ls -t | head -1` would happily evaluate it and publish the
  # numbers under this arm's name. Require the checkpoint to be newer than the
  # marker written just before launch.
  FRESH=$(ssh -F /workspace/.ssh/vastai.conf "$BOX" \
    "find $REMOTE_AR/checkpoints/$arm -name 'final-*.ckpt' -newer $MARKER 2>/dev/null | head -1" \
    2>/dev/null || true)
  if [ -z "$FRESH" ]; then
    say "    !! no final checkpoint newer than launch for $arm — training did not"
    say "       complete. SKIPPING eval so a stale checkpoint is not reported as this arm."
    continue
  fi

  say "    evaluating seg+gold..."
  # BOX must be EXPORTED: eval_remote.sh falls back to its own vastai2 default,
  # so without this every eval in a parallel sweep serialises onto one box.
  if ! BOX="$BOX" "$WT/autoresearch/splade/eval_remote.sh" "$arm" both final 2>&1 | tee -a "$LOG"; then
    say "    !! eval FAILED for $arm — checkpoint kept, re-run eval_remote.sh later"
  fi

  ELAPSED=$(( $(date +%s) - SWEEP_T0 ))
  DONE=$((i + 1)); LEFT=$(( ${#ARMS[@]} - DONE ))
  say "    arm done in $(( ($(date +%s) - T0) / 60 ))m | sweep $DONE/${#ARMS[@]}, ETA $(( LEFT * ELAPSED / DONE / 60 ))m"
done
say "=== sweep complete in $(( ($(date +%s) - SWEEP_T0) / 60 ))m ==="
say "Compare with: uv run --no-project pipeline/splade_df_metrics.py --name <arm>"
