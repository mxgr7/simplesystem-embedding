#!/bin/bash
# Full-length (NO time cap) SPLADE training launcher for production runs.
#   run_full.sh <box> <run_name> [hydra overrides...]
# <box> = vastai0 | vastai1. Stages worktree src/configs/scripts to
# <box>:~/prod_splade (separate from ~/ar_splade and the read-only main tree),
# refuses to launch if the GPU is busy, runs detached (setsid), prints the
# remote pid, tails the log briefly. Unlike run_remote.sh there is NO 15-min
# budget and NO forced accumulate_grad_batches — pass what you need.
set -euo pipefail
BOX=${1:?usage: run_full.sh <box> <run_name> [overrides...]}; shift
NAME=${1:?usage: run_full.sh <box> <run_name> [overrides...]}; shift
WT=$(cd "$(dirname "$0")/../.." && pwd)
SSH="ssh -F /workspace/.ssh/vastai.conf $BOX"
AR=/home/max/prod_splade
MAIN=/home/max/simplesystem-embedding

# H100 driver shim: vastai0 needs LD_PRELOAD of libcuda; vastai1 (driver 595)
# does not. Set per-box.
PRELOAD=""
[ "$BOX" = "vastai0" ] && PRELOAD="LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1"

if $SSH "pgrep -f 'embedding_train.tra[i]n' >/dev/null || pgrep -f 'splade_flops_stopli[s]t' >/dev/null"; then
  echo "GPU BUSY on $BOX (training or eval running) — not launching. Wait and retry."
  exit 2
fi

rsync -e "ssh -F /workspace/.ssh/vastai.conf" -az --delete \
  "$WT/src" "$WT/configs" "$WT/scripts" $BOX:$AR/

$SSH "mkdir -p $AR/checkpoints && cd $AR && \
  GOT=\$(PYTHONPATH=$AR/src $MAIN/.venv/bin/python -c 'import embedding_train; print(embedding_train.__file__)') && \
  case \"\$GOT\" in $AR/src/*) : ;; *) echo \"ABORT wrong module: \$GOT\"; exit 1;; esac && \
  setsid env PYTORCH_ALLOC_CONF=expandable_segments:True $PRELOAD \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR/src \
    timeout --signal=KILL 21600 \
    $MAIN/.venv/bin/python -m embedding_train.train \
    logger=local \
    trainer.checkpoint_dir=$AR/checkpoints logger.run_name=$NAME \
    $* > $AR/run_$NAME.log 2>&1 < /dev/null & echo REMOTE_PID=\$!"

echo "launched $NAME on $BOX; log: $BOX:$AR/run_$NAME.log"
sleep 45
$SSH "tail -6 $AR/run_$NAME.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
