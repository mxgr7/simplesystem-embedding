#!/bin/bash
# Evaluate a run's best checkpoint (or an explicit ckpt path) on the frozen
# fold pools. Usage:
#   eval_remote.sh <run_name_or_remote_ckpt_path> seg|gold|both
# Prints the harness `top256` line(s). Uses the ar_splade staged scripts, so
# eval code == the code you committed. --dist is ALWAYS passed explicitly (the
# harness default points at a non-fold file whose rows lack the sink columns —
# Jinja StrictUndefined kills the render).
set -euo pipefail
TARGET=${1:?usage: eval_remote.sh <run_name|ckpt_path> seg|gold|both}
WHICH=${2:?seg|gold|both}
# BOX/AR/FOLD are overridable so this works across box recycles.
#   BOX  — vastai2 is the 2026-07-28 H100 (driver 595, needs NO libcuda shim;
#          vastai0's 555 driver did). Staged under prod_splade, not ar_splade.
#   FOLD — 'folde' is the ae/oe/ue regime that prod_soup and every current
#          model is trained on; 'fold' is the older a/o/u strip regime. Using
#          the wrong one silently mismatches query normalization and tanks recall.
BOX=${BOX:-vastai2}
SSH="ssh -F /workspace/.ssh/vastai.conf $BOX"
AR=${AR:-/home/max/prod_splade}
MAIN=/home/max/simplesystem-embedding
DATA=$MAIN/data
FOLD=${FOLD:-folde}
PRELOAD=""
[ "$BOX" = "vastai0" ] && PRELOAD="LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1"

# Optional 3rd arg picks the checkpoint flavor for run-name targets:
#   final (default if present) = end-of-fit weights (all trained steps),
#   best = internal-val best (directional metric), last = ModelCheckpoint last.
FLAVOR=${3:-auto}
CKPT=$TARGET
case "$TARGET" in
  /*) : ;;
  *) if [ "$FLAVOR" = auto ]; then
       CKPT=$($SSH "ls -t $AR/checkpoints/$TARGET/final-*.ckpt $AR/checkpoints/$TARGET/best-*.ckpt 2>/dev/null | head -1")
     else
       CKPT=$($SSH "ls -t $AR/checkpoints/$TARGET/$FLAVOR-*.ckpt 2>/dev/null | head -1")
     fi;;
esac
[ -z "$CKPT" ] && { echo "NO CKPT for $TARGET"; exit 1; }
echo "ckpt: $CKPT"

# FAST=1 skips the mask-sweep configs (speed only; top256 metrics identical).
FASTFLAG=""
[ "${FAST:-0}" = "1" ] && FASTFLAG="--skip-mask-configs"

run_eval() { # $1 = gold jsonl, $2 = tag
  # Keep the RAW output. The old form piped straight into grep, so anything the
  # filter missed was gone forever and surfaced only as "NO MATCHING OUTPUT" —
  # which twice cost a rented box to re-diagnose (a missing $AR staging, and a
  # silently failed gold stage). Now the last lines are printed on failure.
  local raw; raw=$(mktemp)
  $SSH "cd $AR && env PYTORCH_ALLOC_CONF=expandable_segments:True \
    $PRELOAD \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR/src \
    $MAIN/.venv/bin/python scripts/splade_flops_stoplist.py \
    --splade-ckpt '$CKPT' --gold $1 \
    --dist $DATA/desc_distractors_${FOLD}.jsonl $FASTFLAG \
    --out $AR/eval_${2}_\$(basename \$(dirname '$CKPT'))_\$(basename '$CKPT' .ckpt | tr -d '=').json 2>&1" \
    > "$raw" 2>&1
  if grep -qE "^top256 " "$raw"; then
    grep -E "^top256 " "$raw" | sed "s/^/[$2] /"
  else
    echo "[$2] EVAL FAILED — last 15 lines of raw output:"
    grep -vE "Welcome to vast.ai|Have fun" "$raw" | tail -15 | sed "s/^/[$2]   /"
  fi
  rm -f "$raw"
}

case "$WHICH" in
  seg)  run_eval $DATA/splade_seg_test_eval_${FOLD}.jsonl seg ;;
  gold) run_eval $DATA/esci_gold_eval_${FOLD}.jsonl gold ;;
  both) run_eval $DATA/splade_seg_test_eval_${FOLD}.jsonl seg || echo "[seg] STAGE FAILED"
        run_eval $DATA/esci_gold_eval_${FOLD}.jsonl gold || echo "[gold] STAGE FAILED" ;;
  *) echo "unknown eval: $WHICH"; exit 1 ;;
esac
