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

Use the offline evaluator to compare lower-precision scoring against the float32 baseline with the same nDCG metrics used during validation.

```bash
uv run embedding-eval --checkpoint checkpoints/best.ckpt --input data/input.parquet --embedding-precision float16
uv run embedding-eval --checkpoint checkpoints/best.ckpt --input data/input.parquet --embedding-precision int8
uv run embedding-eval --checkpoint checkpoints/best.ckpt --input data/input.parquet --embedding-precision binary
```

## Build Index

Build a FAISS inner-product index over offer embeddings rendered with the same `offer_template` used for training and evaluation. The command writes an artifact directory containing `index.faiss`, `metadata.parquet`, and `manifest.json`.

```bash
uv run embedding-index-build --checkpoint checkpoints/best.ckpt --input data/offers.parquet --output data/offer-index
uv run embedding-index-build --checkpoint checkpoints/best.ckpt --input data/offers.parquet --output data/offer-index --index-type ivf_flat --nlist 4096 --train-sample-size 50000 --nprobe 32
uv run embedding-index-build --checkpoint checkpoints/best.ckpt --input data/offers.parquet --output data/offer-index-pq --index-type ivf_pq --nlist 4096 --train-sample-size 100000 --pq-m 16 --pq-bits 8 --nprobe 32
uv run embedding-index-build --checkpoint checkpoints/best.ckpt --input data/offers.parquet --output data/offer-index-hnsw --index-type hnsw --hnsw-m 32 --ef-construction 200 --ef-search 64
```

Helpful flags:

- `--index-type flat|ivf_flat|ivf_pq|hnsw` to choose exact or approximate ANN search
- `--nlist`, `--train-sample-size`, and `--nprobe` to tune IVF indexes
- `--pq-m` and `--pq-bits` to control `ivf_pq` compression quality and memory usage
- `--hnsw-m`, `--ef-construction`, and `--ef-search` to tune HNSW graph accuracy and build cost
- `--copy-columns offer_id_b64,name` to retain extra offer metadata alongside the index
- `--read-batch-size 4096` and `--encode-batch-size 256` to tune throughput
- `--limit-rows 10000` to build a smaller test index
- `--overwrite` to replace an existing artifact directory

## Search Index

Search a built index with either Parquet queries rendered through the same `query_template` used for training and evaluation, or with raw query text.

```bash
uv run embedding-index-search --checkpoint checkpoints/best.ckpt --index data/offer-index --input data/queries.parquet --output data/search_results.parquet --top-k 10
uv run embedding-index-search --checkpoint checkpoints/best.ckpt --index data/offer-index --query-text "hex bolt" --top-k 5
uv run embedding-index-search --checkpoint checkpoints/best.ckpt --index data/offer-index --input data/queries.parquet --output data/search_results.parquet --top-k 10 --nprobe 64
uv run embedding-index-search --checkpoint checkpoints/best.ckpt --index data/offer-index-hnsw --query-text "hex bolt" --top-k 5 --ef-search 128
```

Helpful flags:

- `--nprobe` to override the saved IVF probe count at query time
- `--ef-search` to override the saved HNSW search breadth at query time
- `--copy-columns query_id` to retain query metadata in Parquet search results
- `--output data/search_results.parquet` to persist raw-text searches instead of printing them
- `--limit-rows 1000` to search only part of a Parquet query file
- `--overwrite` to replace an existing search output Parquet file
