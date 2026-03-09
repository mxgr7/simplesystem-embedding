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
uv run embedding-train logger.tracking_uri=http://localhost:5000 logger.experiment_name=embeddings
uv run embedding-train data.limit_rows=2048 trainer.max_epochs=1
uv run embedding-train model.loss_type=contrastive
uv run embedding-train model.loss_type=triplet model.triplet_margin=0.2
```
