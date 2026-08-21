#!/bin/bash
# Train the ä→ae German-fold SPLADE variant on the LOCKED lr=3e-5 recipe, then
# gold-eval it on the German-folded v2.1 set. Compare NDCG@10 to the strip-fold
# baseline (fold_raw = 0.7953). vastai1 (no LD_PRELOAD shim needed).
set -u
AR=/home/max/ar_splade; MAIN=/home/max/simplesystem-embedding; DATA=$MAIN/data
PY=$MAIN/.venv/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR/src
cd $AR || exit 1
R=$AR/folde_results.txt; : > $R
COMMON="model=splade data=splade_sink logger=local model.init_checkpoint=$DATA/v1a_best.ckpt \
  data.max_offer_length=256 data.field_dropout_p=0.3 data.length_bucketing=true data.batch_size=512 \
  trainer.accumulate_grad_batches=1 model.flops_lambda_q=5e-4 model.flops_lambda_d=3e-4 \
  model.flops_warmup_steps=300 optimizer.lr=3e-5 trainer.max_epochs=8 \
  trainer.validation_metric=ndcg_at_5 trainer.encode_batch_size=32 trainer.checkpoint_dir=$AR/checkpoints"

echo "== TRAIN folde_lr3e5 $(date -u +%H:%M:%S) ==" | tee -a $R
$PY -m embedding_train.train $COMMON data.path=$DATA/splade_train_raw_desc0_folde.parquet \
  logger.run_name=folde_lr3e5 > $AR/train_folde.log 2>&1
ck=$(ls -t $AR/checkpoints/folde_lr3e5/best-*.ckpt 2>/dev/null | head -1)
if [ -z "$ck" ]; then echo "NO CKPT — tail:" | tee -a $R; tail -8 $AR/train_folde.log | tee -a $R; exit 1; fi
echo "  ckpt=$ck" | tee -a $R
echo "== EVAL folde_lr3e5 on German-folded v2.1 (--no-fold) ==" | tee -a $R
$PY scripts/splade_cased_mask_eval.py --checkpoint "$ck" --name folde_lr3e5 \
  --eval-jsonl $DATA/esci_gold_eval_v21_folde.jsonl --no-fold 2>&1 \
  | grep -E "===|NDCG@10|doc  nnz|cased dims|ΔNDCG|mass kept" | tee -a $R
echo "== DONE $(date -u +%H:%M:%S) ==" | tee -a $R
