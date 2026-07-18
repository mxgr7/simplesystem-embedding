"""Export the deterministic validation split to a standalone parquet.

The train/val split (seed + val_fraction + query_id connected components)
is model-independent, so the exported parquet lets embedding-catalog-benchmark
evaluate different checkpoints (dense, SPLADE) on the identical val
queries/catalog that in-training full-catalog validation uses.

Usage:
  uv run python scripts/export_val_split.py \
    --config-overrides model=splade data=splade data.path=/abs/queries_offers_labeled.parquet \
    --output data/val_split.parquet
"""

import argparse

import hydra
import pandas as pd

from embedding_train.config import CONFIG_DIR
from embedding_train.data import EmbeddingDataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-overrides", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with hydra.initialize_config_dir(
        version_base="1.3", config_dir=str(CONFIG_DIR)
    ):
        cfg = hydra.compose(config_name="config", overrides=args.config_overrides)

    datamodule = EmbeddingDataModule(cfg)
    datamodule.setup()

    val_query_ids = {
        record["query_id"] for record in datamodule.val_dataset.records
    }
    frame = pd.read_parquet(cfg.data.path)
    if cfg.data.limit_rows:
        frame = frame.head(int(cfg.data.limit_rows))

    query_id_column = cfg.data.column_mapping.query_id
    val_frame = frame[frame[query_id_column].isin(val_query_ids)]
    val_frame.to_parquet(args.output, index=False)
    print(
        f"wrote {len(val_frame)} rows, {val_frame[query_id_column].nunique()} "
        f"queries to {args.output}"
    )


if __name__ == "__main__":
    main()
