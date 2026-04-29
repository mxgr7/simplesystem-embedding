#!/usr/bin/env bash
# Evaluate the best checkpoint of a named run with fast-eval (5000 prefixes,
# seed 42 for reproducibility), append the result to suggest_results.tsv.
#
#   scripts/eval_run.sh RUN_NAME [VARIANT_TAG] [NOTES]
set -euo pipefail
RUN_NAME=${1:?run_name required}
VARIANT=${2:-lm}
NOTES=${3:-}

DATA_ROOT=${SUGGEST_DATA_ROOT:-/home/max/workspaces/simplesystem/data/suggest}
CKPT_DIR="$DATA_ROOT/checkpoints/$RUN_NAME"

# Pick the best-* file with the lowest val_loss (encoded in filename).
BEST=$(ls "$CKPT_DIR"/best-* 2>/dev/null | head -1 || true)
if [ -z "$BEST" ]; then
    echo "No best-* checkpoint in $CKPT_DIR" >&2
    exit 1
fi
echo "ckpt: $BEST"

OUT_JSON="$DATA_ROOT/results/${RUN_NAME}_fast.json"
TSV="$DATA_ROOT/results/suggest_results.tsv"

PREFIX_BATCH=${PREFIX_BATCH:-512}
uv run suggest-eval \
    --model lm \
    --lm-ckpt "$BEST" \
    --lm-prefix-batch "$PREFIX_BATCH" \
    --sample-prefixes 5000 \
    --sample-seed 42 \
    --out "$OUT_JSON" 2>&1 | tee "/tmp/eval_${RUN_NAME}.log"

VAL_LOSS=$(echo "$BEST" | sed -n 's/.*val_loss=\([0-9]*\.[0-9]*\)\.ckpt/\1/p')

# Pull metrics from the JSON.
uv run python - <<PY
import json, time
rep = json.load(open("$OUT_JSON"))
o = rep["overall"]
m = rep["_meta"]
row = "\t".join([
    "$RUN_NAME",
    "$VARIANT",
    "$NOTES",
    "-",  # epochs (filled by user / re-eval)
    "${VAL_LOSS:-}",
    f"{o['mrr@10']:.4f}",
    f"{o['recall@1']:.4f}",
    f"{o['recall@5']:.4f}",
    f"{o['recall@10']:.4f}",
    f"{m['elapsed_sec']/60:.1f}",
    "fast-eval n=5000 seed=42",
])
with open("$TSV", "a") as f:
    f.write(row + "\n")
print("appended:", row)
PY
