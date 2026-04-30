"""Distillation CLI — train a small student CE against the frozen teacher.

Mirrors `cross_encoder_train.train.fit_from_cfg` but swaps the LightningModule
to `CrossEncoderDistillModule`. Reuses the existing `CrossEncoderDataModule`
(student tokenizer is the same vocab as teacher → tokenized inputs match).

Run:
  TEACHER_CKPT=../checkpoints/cross-encoder/releases/v1.0-2026-04-29/soup.ckpt \\
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \\
  uv run --extra train python -m cross_encoder_train.distill --config-name distill_cross_encoder
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

from cross_encoder_train.data import CrossEncoderDataModule
from cross_encoder_train.distill_module import CrossEncoderDistillModule
from cross_encoder_train.train import (
    build_callbacks,
    build_logger,
    configure_logging,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def fit_distill(cfg):
    load_dotenv()
    configure_logging(cfg.log_level)
    torch.set_float32_matmul_precision("high")
    L.seed_everything(int(cfg.seed), workers=True)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    datamodule = CrossEncoderDataModule(cfg)
    model = CrossEncoderDistillModule(cfg)
    logger = build_logger(cfg)
    _ = logger.experiment
    logging.info(
        "MLflow run artifact URI: %s",
        logger.experiment.get_run(logger.run_id).info.artifact_uri,
    )
    callbacks = build_callbacks(cfg, logger)

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=int(cfg.trainer.max_epochs),
        max_time=getattr(cfg.trainer, "max_time", None),
        precision=cfg.trainer.precision,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        deterministic=bool(cfg.trainer.deterministic),
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        gradient_clip_val=getattr(cfg.trainer, "gradient_clip_val", None),
        enable_checkpointing=bool(getattr(cfg.trainer, "enable_checkpointing", True)),
        default_root_dir=str(Path.cwd()),
        logger=logger,
        callbacks=callbacks,
    )

    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer.fit(model, datamodule=datamodule)

    from lightning.pytorch.callbacks import ModelCheckpoint
    best = ""
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            best = cb.best_model_path
            break
    if best:
        print(f"Best checkpoint: {best}")


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="distill_cross_encoder")
def run(cfg):
    fit_distill(cfg)


def main():
    run()


if __name__ == "__main__":
    main()
