# Embedding Training Baseline

Minimal Hydra + PyTorch Lightning project for fine-tuning a text embedding model on labeled query-offer pairs.

## Stack

- `uv` for environment and dependency management
- `hydra` for configuration
- `lightning` for training orchestration
- `mlflow` for experiment logging
- `transformers` for the pretrained encoder

## Default model

- Shared bi-encoder backbone
- Default pretrained model: `microsoft/mdeberta-v3-base`
- Mean pooling over the last hidden state
- Optional learned projection head via `model.output_dim`; default keeps the encoder hidden size (`768` for `microsoft/mdeberta-v3-base`)
- Selectable BCE, in-batch contrastive, or in-batch triplet loss over cosine similarity
- Binary target: `Exact` vs. everything else

## Data

Default dataset path:

`/Users/max/Clients/simplesystem/data/queries_offers_labeled.parquet`

The input text for each side is rendered with configurable Jinja2 templates.

## Install

```bash
uv sync
```

## Train

```bash
uv run embedding-train
```

Example overrides:

```bash
uv run embedding-train model.model_name=sentence-transformers/all-mpnet-base-v2
uv run embedding-train model.output_dim=256
uv run embedding-train logger.tracking_uri=http://localhost:5000 logger.experiment_name=embeddings
uv run embedding-train data.limit_rows=2048 trainer.max_epochs=1
uv run embedding-train model.loss_type=contrastive
uv run embedding-train model.loss_type=triplet model.triplet_margin=0.2
uv run embedding-train trainer.resume_from_checkpoint=checkpoints/<run-name>/last-step=<step>-val_ndcg_at_5=<score>.ckpt
```

To continue training from an interrupted run, point `trainer.resume_from_checkpoint` at the saved `last-step=...ckpt` checkpoint so Lightning restores optimizer and loop state.

## Infer

Use the offline CLI to stream Parquet input, export embeddings in batches, and write results incrementally.

```bash
uv run embedding-infer --checkpoint checkpoints/best.ckpt --input data/input.parquet --output data/offer_embeddings.parquet
uv run embedding-infer --checkpoint checkpoints/best.ckpt --input data/input.parquet --output data/query_embeddings.parquet --mode query
uv run embedding-infer --checkpoint checkpoints/best.ckpt --input data/input.parquet --output data/pair_scores.parquet --mode pair_score
uv run embedding-infer --checkpoint checkpoints/best.ckpt --input data/input.parquet --output data/offer_embeddings_f16.parquet --embedding-precision float16
uv run embedding-infer --checkpoint checkpoints/best.ckpt --input data/input.parquet --output data/offer_embeddings_binary.parquet --embedding-precision binary
```

Helpful flags:

- `--read-batch-size 4096` to stream larger Parquet chunks
- `--encode-batch-size 256` to control model batch size independently
- `--embedding-precision float32|float16|int8|sign|binary` to control storage precision; `binary` stores packed sign bits
- `--include-text` to persist rendered query and/or offer text alongside outputs
- `--copy-columns query_id,offer_id_b64,label` to keep additional join keys in the output

## Evaluate

Use the offline evaluator to either compare lower-precision pair scoring against the float32 baseline, or evaluate exact-hit retrieval against a built FAISS index.

```bash
uv run embedding-eval --checkpoint checkpoints/best.ckpt --input data/input.parquet --embedding-precision float16
uv run embedding-eval --checkpoint checkpoints/best.ckpt --input data/input.parquet --embedding-precision int8
uv run embedding-eval --checkpoint checkpoints/best.ckpt --input data/input.parquet --embedding-precision binary
uv run embedding-eval --checkpoint checkpoints/best.ckpt --input data/labeled_pairs.parquet --index data/offer-index --top-k 10
```

Notes:

- Pairwise mode reports both graded ranking metrics (`nDCG@k`) and exact-match metrics (`exact_success@1`, `exact_mrr`, `exact_recall@k`), plus delta vs `float32` for lower precisions
- Retrieval mode reports exact-match search metrics like `exact_success@1`, `exact_mrr`, and `exact_recall@k`
- Retrieval mode expects the built index metadata to include `offer_id_b64`; the default index build settings already keep it when present in the input

## Benchmark Catalog Search

Use the exhaustive benchmark CLI when you want to embed the queries and offers from one labeled parquet file, score every query against the full deduplicated catalog exactly, and report ranking metrics.

```bash
uv run embedding-catalog-benchmark --checkpoint checkpoints/best.ckpt --input data/queries_offers_eval.parquet
uv run embedding-catalog-benchmark --checkpoint checkpoints/best.ckpt --input data/queries_offers_eval.parquet --similarity cosine --ks 1,5,10,100
uv run embedding-catalog-benchmark --checkpoint checkpoints/best.ckpt --input data/queries_offers_eval.parquet --relevant-labels Exact,Substitute
```

Notes:

- The benchmark uses exact exhaustive scoring over the whole catalog, not ANN search
- The input file should contain query fields, offer fields, and `label` in the same row format as `queries_offers_eval.parquet`
- `nDCG@k` uses the graded gains from the labeled data: `Exact=1.0`, `Substitute=0.1`, `Complement=0.01`, `Irrelevant=0.0`
- `Recall@k`, `MRR`, and `Precision@k` treat `Exact` as relevant by default; override with `--relevant-labels`
- The catalog is built by deduplicating the offer id column from the same input file

## Build Index

Build a FAISS inner-product index over offer embeddings rendered with the same `offer_template` used for training and evaluation. The command writes an artifact directory containing `index.faiss`, `metadata.parquet`, and `manifest.json`.

```bash
uv run embedding-index-build --input data/offers.parquet --output data/offer-index
uv run embedding-index-build --checkpoint checkpoints/best.ckpt --input data/offers.parquet --output data/offer-index
uv run embedding-index-build --model-name intfloat/multilingual-e5-base --input data/offers.parquet --output data/offer-index --index-type ivf_flat --nlist 4096 --train-sample-size 50000 --nprobe 32
uv run embedding-index-build --checkpoint checkpoints/best.ckpt --input data/offers.parquet --output data/offer-index-pq --index-type ivf_pq --nlist 4096 --train-sample-size 100000 --pq-m 16 --pq-bits 8 --nprobe 32
uv run embedding-index-build --input data/offers.parquet --output data/offer-index-hnsw --index-type hnsw --hnsw-m 32 --ef-construction 200 --ef-search 64
```

Helpful flags:

- `--index-type flat|ivf_flat|ivf_pq|hnsw` to choose exact or approximate ANN search
- `--nlist`, `--train-sample-size`, and `--nprobe` to tune IVF indexes
- `--pq-m` and `--pq-bits` to control `ivf_pq` compression quality and memory usage
- `--hnsw-m`, `--ef-construction`, and `--ef-search` to tune HNSW graph accuracy and build cost
- `--copy-columns offer_id_b64,name` to retain extra offer metadata alongside the index
- `--read-batch-size 4096` and `--encode-batch-size 256` to tune throughput
- `--limit-rows 10000` to build a smaller test index
- `--checkpoint` to use a fine-tuned model, or omit it to use the default base model
- `--model-name` to override the pretrained base model when `--checkpoint` is omitted
- `--overwrite` to replace an existing artifact directory

## Mine Hard Negatives

Mine hard negatives offline from a built FAISS index so training batches can mix them with same-query and random negatives. The command encodes unique queries from a labeled parquet file, searches the index, drops known positives, and writes the top-ranked remaining offers to a parquet sidecar.

```bash
uv run embedding-mine-hard-negatives --index data/offer-index --input data/queries_offers_labeled.parquet --output data/hard_negatives.parquet
uv run embedding-mine-hard-negatives --checkpoint checkpoints/best.ckpt --index data/offer-index --input data/queries_offers_labeled.parquet --output data/hard_negatives.parquet --top-k 50 --max-negatives-per-query 10
```

Wire the output into training by pointing `data.hard_negatives_path` at the parquet file:

```bash
uv run embedding-train data.hard_negatives_path=data/hard_negatives.parquet
```

Notes:

- Mined negatives exclude offers labeled as positives (`data.positive_label`) for the anchor query
- Batch stats count hard negatives separately from synthetic cross-query negatives when `data.log_batch_stats=true`
- Refresh the mined parquet against a newer checkpoint whenever the model has moved far enough that the old negatives are no longer confusing; there is no automatic refresh during training
- `--top-k` controls how many candidates are retrieved per query before positive exclusion; `--max-negatives-per-query` caps how many survivors are kept

## Search Index

Search a built index with either Parquet queries rendered through the same `query_template` used for training and evaluation, or with raw query text.

```bash
uv run embedding-index-search --index data/offer-index --input data/queries.parquet --output data/search_results.parquet --top-k 10
uv run embedding-index-search --index data/offer-index --query-text "hex bolt" --top-k 5
uv run embedding-index-search --checkpoint checkpoints/best.ckpt --index data/offer-index --input data/queries.parquet --output data/search_results.parquet --top-k 10 --nprobe 64
uv run embedding-index-search --model-name intfloat/multilingual-e5-base --index data/offer-index-hnsw --query-text "hex bolt" --top-k 5 --ef-search 128
```

Helpful flags:

- `--nprobe` to override the saved IVF probe count at query time
- `--ef-search` to override the saved HNSW search breadth at query time
- `--copy-columns query_id` to retain query metadata in Parquet search results
- `--output data/search_results.parquet` to persist raw-text searches instead of printing them
- `--limit-rows 1000` to search only part of a Parquet query file
- `--checkpoint` to use a fine-tuned query encoder, or omit it to use the default base model
- `--model-name` to override the pretrained base model when `--checkpoint` is omitted
- `--overwrite` to replace an existing search output Parquet file
