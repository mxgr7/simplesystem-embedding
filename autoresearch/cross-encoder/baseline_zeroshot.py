"""Zero-shot transfer: evaluate the old keep checkpoint on the new dataset's val split.

Loads valuable-finch-654 (focal_gamma=2.0 keep, micro=0.9204 on OLD val)
and runs validation against the NEW datamodule (current configs/cross_encoder.yaml).

Reports val/cls/micro_f1, val/cls/macro_f1, per-class F1, and ranking metrics.
"""

from pathlib import Path

import lightning as L
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf

from cross_encoder_train.data import CrossEncoderDataModule
from cross_encoder_train.model import CrossEncoderModule


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
CKPT = REPO_ROOT / "checkpoints" / "valuable-finch-654" / (
    "best-step=6051-val_cls_micro_f1=0.9204.ckpt"
)


def main():
    torch.set_float32_matmul_precision("high")

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="cross_encoder")

    print("Loading datamodule with NEW dataset config...")
    datamodule = CrossEncoderDataModule(cfg)
    datamodule.setup()
    print(f"Val rows: {datamodule.dataset_stats.get('val_rows')}")

    print(f"Loading checkpoint: {CKPT}")
    model = CrossEncoderModule.load_from_checkpoint(str(CKPT), cfg=cfg)

    logger = MLFlowLogger(
        experiment_name=cfg.logger.experiment_name,
        tracking_uri=cfg.logger.tracking_uri,
        run_name="00-baseline-zeroshot-old-keep",
        log_model=False,
        tags={"baseline": "zeroshot", "source_ckpt": "valuable-finch-654"},
    )

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    metrics = trainer.validate(model, datamodule=datamodule)
    print("\n=== Zero-shot transfer (old keep checkpoint on NEW val) ===")
    print(OmegaConf.to_yaml(OmegaConf.create(metrics[0]), resolve=True))


if __name__ == "__main__":
    main()
