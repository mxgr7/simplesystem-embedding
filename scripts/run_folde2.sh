#!/bin/bash
# Stage-2 fine-tune from the German-folded backbone (v1a_folde) — the FULLY CLEAN
# fold model: folded pretraining + folded fine-tune. Identical to folde_lr3e5 in
# every way EXCEPT init_checkpoint (v1a_folde instead of the raw v1a_best), so it
# isolates the effect of re-doing Stage-1 in the fold regime. Compare gold NDCG@10
# to folde_lr3e5 (0.8157, raw backbone) and fold_raw (0.7953, strip).
set -u
AR=/home/max/ar_splade; MAIN=/home/max/simplesystem-embedding; DATA=$MAIN/data
PY=$MAIN/.venv/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR/src
cd $AR || exit 1
# symlink around the '=' in the ckpt filename (hydra-equals-path-trap)
BB=$(ls -t $AR/checkpoints/v1a_folde/best-*.ckpt | head -1)
ln -sf "$BB" $AR/v1a_folde_backbone.ckpt
R=$AR/folde2_results.txt; : > $R
COMMON="model=splade data=splade_sink logger=local \
  data.max_offer_length=256 data.field_dropout_p=0.3 data.length_bucketing=true data.batch_size=512 \
  trainer.accumulate_grad_batches=1 model.flops_lambda_q=5e-4 model.flops_lambda_d=3e-4 \
  model.flops_warmup_steps=300 optimizer.lr=3e-5 trainer.max_epochs=8 \
  trainer.validation_metric=ndcg_at_5 trainer.encode_batch_size=32 trainer.checkpoint_dir=$AR/checkpoints"

echo "== TRAIN folde2_lr3e5 (folded backbone v1a_folde) $(date -u +%H:%M:%S) | bb=$(basename $BB) ==" | tee -a $R
$PY -m embedding_train.train $COMMON \
  data.path=$DATA/splade_train_raw_desc0_folde.parquet \
  model.init_checkpoint=$AR/v1a_folde_backbone.ckpt \
  logger.run_name=folde2_lr3e5 > $AR/train_folde2.log 2>&1
ck=$(ls -t $AR/checkpoints/folde2_lr3e5/best-*.ckpt 2>/dev/null | head -1)
if [ -z "$ck" ]; then echo "NO CKPT — tail:" | tee -a $R; tail -12 $AR/train_folde2.log | tee -a $R; exit 1; fi
echo "  ckpt=$ck" | tee -a $R
echo "== EVAL folde2_lr3e5 on German-folded v2.1 (--no-fold) ==" | tee -a $R
$PY scripts/splade_cased_mask_eval.py --checkpoint "$ck" --name folde2_lr3e5 \
  --eval-jsonl $DATA/esci_gold_eval_v21_folde.jsonl --no-fold 2>&1 \
  | grep -E "===|NDCG@10|doc  nnz|cased dims|ΔNDCG|mass kept" | tee -a $R
echo "== DONE $(date -u +%H:%M:%S) ==" | tee -a $R
