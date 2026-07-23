#!/bin/bash
# Clean Stage-1 re-pretraining in the German-fold (ä→ae) regime: exactly the
# defiant-lynx-807 / v1a_best recipe (from VANILLA gbert-base, base splade config,
# semi-hard negatives, lr 2e-5, flops_d 6e-4, warmup 1500, 8ep) but on German-folded
# data. Produces v1a_folde — the folded backbone. Original v1a_best = 0.6702 val.
set -u
AR=/home/max/ar_splade; MAIN=/home/max/simplesystem-embedding; DATA=$MAIN/data
PY=$MAIN/.venv/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR/src
cd $AR || exit 1
R=$AR/stage1_folde_results.txt; : > $R
echo "== STAGE-1 v1a_folde (a->ae, from vanilla gbert) $(date -u +%H:%M:%S) ==" | tee -a $R
$PY -m embedding_train.train \
  model=splade data=splade logger=local \
  data.path=$DATA/queries_offers_labeled_folde.parquet \
  data.semi_hard_negatives_path=$DATA/semi_hard_negatives-uc58-splade_folde.parquet \
  model.init_checkpoint=null \
  model.flops_lambda_q=5e-4 model.flops_lambda_d=6e-4 model.flops_warmup_steps=1500 \
  optimizer.lr=2e-5 data.batch_size=512 trainer.accumulate_grad_batches=1 \
  data.max_offer_length=256 trainer.max_epochs=8 \
  trainer.validation_metric=ndcg_at_5 trainer.encode_batch_size=32 \
  trainer.checkpoint_dir=$AR/checkpoints logger.run_name=v1a_folde \
  > $AR/train_stage1_folde.log 2>&1
ck=$(ls -t $AR/checkpoints/v1a_folde/best-*.ckpt 2>/dev/null | head -1)
if [ -z "$ck" ]; then echo "NO CKPT — tail:" | tee -a $R; tail -12 $AR/train_stage1_folde.log | tee -a $R; exit 1; fi
echo "  v1a_folde ckpt=$ck" | tee -a $R
echo "== STAGE-1 DONE $(date -u +%H:%M:%S) ==" | tee -a $R
