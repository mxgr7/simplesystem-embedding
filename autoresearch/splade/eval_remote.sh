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
SSH="ssh -F /workspace/.ssh/vastai.conf vastai0"
AR=/home/max/ar_splade
MAIN=/home/max/simplesystem-embedding
DATA=$MAIN/data

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
  $SSH "cd $AR && env PYTORCH_ALLOC_CONF=expandable_segments:True \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR/src \
    $MAIN/.venv/bin/python scripts/splade_flops_stoplist.py \
    --splade-ckpt '$CKPT' --gold $1 \
    --dist $DATA/desc_distractors_fold.jsonl $FASTFLAG \
    --out $AR/eval_${2}_\$(basename '$CKPT' .ckpt | tr -d '=').json 2>&1" \
    | grep -E "^top256 |Error|Traceback" | sed "s/^/[$2] /"
}

case "$WHICH" in
  seg)  run_eval $DATA/splade_seg_test_eval_fold.jsonl seg ;;
  gold) run_eval $DATA/esci_gold_eval_fold.jsonl gold ;;
  both) run_eval $DATA/splade_seg_test_eval_fold.jsonl seg
        run_eval $DATA/esci_gold_eval_fold.jsonl gold ;;
  *) echo "unknown eval: $WHICH"; exit 1 ;;
esac
