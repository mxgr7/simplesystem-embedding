#!/bin/bash
# Launch a SPLADE training run on vastai0 from the autoresearch worktree.
#   run_remote.sh <run_name> [hydra overrides...]
# Stages worktree src/configs/scripts to vastai0:~/ar_splade (never touches the
# main tree), refuses to launch if the GPU is busy with a training/eval job,
# runs detached (setsid), prints the remote pid, and tails the log briefly.
set -euo pipefail
NAME=${1:?usage: run_remote.sh <run_name> [overrides...]}; shift
WT=$(cd "$(dirname "$0")/../.." && pwd)
SSH="ssh -F /workspace/.ssh/vastai.conf vastai0"
AR=/home/max/ar_splade
MAIN=/home/max/simplesystem-embedding

# HARD TIME BUDGET: every run trains for at most 45 minutes. Inject the cap if
# absent; refuse any user-supplied max_time above it (format DD:HH:MM:SS).
MAXTIME_ARG=""
for a in "$@"; do
  case "$a" in trainer.max_time=*)
    MAXTIME_ARG="${a#trainer.max_time=}"
    SECS=$(echo "$MAXTIME_ARG" | awk -F: '{print $1*86400+$2*3600+$3*60+$4}')
    if [ "$SECS" -gt 2700 ]; then
      echo "REFUSED: trainer.max_time=$MAXTIME_ARG exceeds the 45-minute budget."
      exit 3
    fi ;;
  esac
done
EXTRA=""
[ -z "$MAXTIME_ARG" ] && EXTRA="trainer.max_time=00:00:45:00"

# accumulate_grad_batches defaults to 2 (H100 VRAM at seq 512); overridable
# for short-seq configs (e.g. seq 256 + batch 512 + accum 1). Hydra errors on
# duplicate overrides, so only inject the default when the caller didn't.
ACCUM="trainer.accumulate_grad_batches=2"
for a in "$@"; do
  case "$a" in trainer.accumulate_grad_batches=*) ACCUM="";; esac
done

if $SSH "pgrep -f 'embedding_train.tra[i]n' >/dev/null || pgrep -f 'splade_flops_stopli[s]t' >/dev/null"; then
  echo "GPU BUSY on vastai0 (training or eval running) — not launching. Wait and retry."
  exit 2
fi

rsync -e "ssh -F /workspace/.ssh/vastai.conf" -az --delete \
  "$WT/src" "$WT/configs" "$WT/scripts" vastai0:$AR/

$SSH "mkdir -p $AR/checkpoints && cd $AR && \
  GOT=\$(PYTHONPATH=$AR/src $MAIN/.venv/bin/python -c 'import embedding_train; print(embedding_train.__file__)') && \
  case \"\$GOT\" in $AR/src/*) : ;; *) echo \"ABORT wrong module: \$GOT\"; exit 1;; esac && \
  setsid env PYTORCH_ALLOC_CONF=expandable_segments:True \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR/src \
    timeout --signal=KILL 5400 \
    $MAIN/.venv/bin/python -m embedding_train.train \
    model=splade data=splade_sink logger=local \
    trainer.checkpoint_dir=$AR/checkpoints logger.run_name=$NAME \
    trainer.encode_batch_size=32 $ACCUM \
    $EXTRA $* > $AR/run_$NAME.log 2>&1 < /dev/null & echo REMOTE_PID=\$!"

echo "launched $NAME; log: vastai0:$AR/run_$NAME.log"
sleep 60
$SSH "tail -3 $AR/run_$NAME.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
