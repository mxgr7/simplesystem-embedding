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
```

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
