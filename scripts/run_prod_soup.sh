#!/bin/bash
# PRODUCTION seed-soup (German-fold ä→ae regime). Trains the locked s7_lr3e5 recipe
# on {raw, b50} data mixes × 2 seeds, uniform-soups the 4 arms, and harness-evals
# the soup + members on the folded seg/gold sets. folde_lr3e5 = raw/seed42 already
# exists and is reused as arm 1, so only 3 new runs here.
#   arms: raw/s42 (=folde_lr3e5) + raw/s1337 + b50/s42 + b50/s1337
set -u
AR=/home/max/ar_splade; MAIN=/home/max/simplesystem-embedding; DATA=$MAIN/data
PY=$MAIN/.venv/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR/src
cd $AR || exit 1
R=$AR/prod_soup_results.txt; : > $R
COMMON="model=splade data=splade_sink logger=local model.init_checkpoint=$DATA/v1a_best.ckpt \
  data.max_offer_length=256 data.field_dropout_p=0.3 data.length_bucketing=true data.batch_size=512 \
  trainer.accumulate_grad_batches=1 model.flops_lambda_q=5e-4 model.flops_lambda_d=3e-4 \
  model.flops_warmup_steps=300 optimizer.lr=3e-5 trainer.max_epochs=8 \
  trainer.validation_metric=ndcg_at_5 trainer.encode_batch_size=32 trainer.checkpoint_dir=$AR/checkpoints"

train() {  # $1=run_name  $2=data_parquet  $3=seed
  ck=$(ls -t $AR/checkpoints/$1/best-*.ckpt 2>/dev/null | head -1)
  if [ -n "$ck" ]; then echo "== SKIP $1 (exists) $ck ==" | tee -a $R; return; fi
  echo "== TRAIN $1 (data=$2 seed=$3) $(date -u +%H:%M:%S) ==" | tee -a $R
  $PY -m embedding_train.train $COMMON data.path=$DATA/$2 seed=$3 logger.run_name=$1 \
    > $AR/train_$1.log 2>&1
  ck=$(ls -t $AR/checkpoints/$1/best-*.ckpt 2>/dev/null | head -1)
  if [ -z "$ck" ]; then echo "  NO CKPT $1 — tail:" | tee -a $R; tail -8 $AR/train_$1.log | tee -a $R
  else echo "  $1 ckpt=$ck" | tee -a $R; fi
}
harness() {  # $1=ckpt  $2=label
  [ -z "$1" ] && return
  for gd in "splade_seg_test_eval_folde.jsonl:seg" "esci_gold_eval_folde.jsonl:gold"; do
    g="${gd%%:*}"; tag="${gd##*:}"
    $PY scripts/splade_flops_stoplist.py --splade-ckpt "$1" --gold $DATA/$g \
      --dist $DATA/desc_distractors_folde.jsonl --skip-mask-configs 2>&1 \
      | grep -E "^top256 " | sed "s/^/[$2 $tag] /" | tee -a $R
  done
}

train folde_raw_s1337  splade_train_raw_desc0_folde.parquet  1337
train folde_b50_s42    splade_train_b50_desc0_folde.parquet  42
train folde_b50_s1337  splade_train_b50_desc0_folde.parquet  1337

# collect arm checkpoints (folde_lr3e5 = raw/seed42, pre-existing)
declare -A ARM
ARM[raw_s42]=$(ls -t $AR/checkpoints/folde_lr3e5/best-*.ckpt 2>/dev/null | head -1)
ARM[raw_s1337]=$(ls -t $AR/checkpoints/folde_raw_s1337/best-*.ckpt 2>/dev/null | head -1)
ARM[b50_s42]=$(ls -t $AR/checkpoints/folde_b50_s42/best-*.ckpt 2>/dev/null | head -1)
ARM[b50_s1337]=$(ls -t $AR/checkpoints/folde_b50_s1337/best-*.ckpt 2>/dev/null | head -1)
INGR=""
for k in raw_s42 raw_s1337 b50_s42 b50_s1337; do [ -n "${ARM[$k]}" ] && INGR="$INGR ${ARM[$k]}"; done
echo "== SOUP ingredients:$INGR ==" | tee -a $R
$PY scripts/soup.py $AR/prod_soup.ckpt $INGR 2>&1 | tee -a $R

echo "== HARNESS (top256, folded seg+gold) $(date -u +%H:%M:%S) ==" | tee -a $R
for k in raw_s42 raw_s1337 b50_s42 b50_s1337; do harness "${ARM[$k]}" "$k"; done
harness "$AR/prod_soup.ckpt" "SOUP"

echo "== SOUP gold NDCG@10 (pool metric) ==" | tee -a $R
$PY scripts/splade_cased_mask_eval.py --checkpoint "$AR/prod_soup.ckpt" --name prod_soup \
  --eval-jsonl $DATA/esci_gold_eval_v21_folde.jsonl --no-fold 2>&1 \
  | grep -E "NDCG@10|doc  nnz" | tee -a $R
echo "== PROD SOUP DONE $(date -u +%H:%M:%S) ==" | tee -a $R
