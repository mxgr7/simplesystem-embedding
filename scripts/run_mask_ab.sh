#!/bin/bash
# A/B on the LOCKED lr=3e-5 folded recipe: control (fold_vocab_mask=false) vs
# treatment (fold_vocab_mask=true = cased dims zeroed at train time). Then gold-eval
# both on v2.1. Answers: does baking the cased-vocab mask into training beat the
# post-hoc mask? Runs on vastai1 (no LD_PRELOAD shim needed).
set -u
AR=/home/max/ar_splade; MAIN=/home/max/simplesystem-embedding; DATA=$MAIN/data
PY=$MAIN/.venv/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR/src
cd $AR || exit 1
R=$AR/mask_ab_results.txt; : > $R
COMMON="model=splade data=splade_sink logger=local model.init_checkpoint=$DATA/v1a_best.ckpt \
  data.path=$DATA/splade_train_raw_desc0.parquet data.max_offer_length=256 \
  data.field_dropout_p=0.3 data.length_bucketing=true data.batch_size=512 \
  trainer.accumulate_grad_batches=1 model.flops_lambda_q=5e-4 model.flops_lambda_d=3e-4 \
  model.flops_warmup_steps=300 optimizer.lr=3e-5 trainer.max_epochs=8 \
  trainer.validation_metric=ndcg_at_5 trainer.encode_batch_size=32 trainer.checkpoint_dir=$AR/checkpoints"

train() {  # $1=run_name  $2=fold_vocab_mask
  echo "== TRAIN $1 (fold_vocab_mask=$2) $(date -u +%H:%M:%S) ==" | tee -a $R
  $PY -m embedding_train.train $COMMON logger.run_name=$1 model.fold_vocab_mask=$2 \
    > $AR/train_$1.log 2>&1
  ck=$(ls -t $AR/checkpoints/$1/best-*.ckpt 2>/dev/null | head -1)
  if [ -z "$ck" ]; then echo "  NO CKPT for $1 — tail:" | tee -a $R; tail -8 $AR/train_$1.log | tee -a $R
  else echo "  ckpt=$ck" | tee -a $R; fi
}
evalck() {  # $1=run_name
  ck=$(ls -t $AR/checkpoints/$1/best-*.ckpt 2>/dev/null | head -1)
  [ -z "$ck" ] && return
  echo "== EVAL $1 ==" | tee -a $R
  $PY scripts/splade_cased_mask_eval.py --checkpoint "$ck" --name $1 \
    --eval-jsonl $DATA/esci_gold_eval_v21_sink.jsonl 2>&1 \
    | grep -E "===|NDCG@10|doc  nnz|cased dims|ΔNDCG|mass kept" | tee -a $R
}

train lr3e5_ctl  false
train lr3e5_mask true
evalck lr3e5_ctl
evalck lr3e5_mask
echo "== ALL DONE $(date -u +%H:%M:%S) ==" | tee -a $R
