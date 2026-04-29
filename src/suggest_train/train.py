"""Training entry point for the suggest LM (option-a, targets-only).

Hydra config root: ``configs/suggest/``. The CLI delegates to
``configs/suggest/config.yaml``; sub-configs are layered via
``defaults:``. Override anything from the command line, e.g.::

    suggest-train trainer.max_epochs=20 model.n_layers=8

All hyperparameters that affect the model architecture are saved alongside
the checkpoint via ``save_hyperparameters`` so ``infer.py`` can reconstruct
the model without the original config.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import hydra
import lightning as L
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from transformers import get_scheduler

from .data import DATA_ROOT
from .lit_data import LABEL_IGNORE, SuggestLMDataModule
from .model import LMConfig, SuggestLM
from .tokenizer import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "suggest"


class DatasetStatsLogger(Callback):
    def __init__(self) -> None:
        self._logged = False

    def setup(self, trainer, pl_module, stage):  # type: ignore[override]
        if self._logged or trainer.datamodule is None:
            return
        stats = getattr(trainer.datamodule, "dataset_stats", None)
        if not stats:
            return
        if trainer.logger is not None:
            trainer.logger.log_hyperparams(
                {f"dataset/{k}": v for k, v in stats.items()}
            )
        self._logged = True


class SuggestLMModule(L.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        # Load the tokenizer to size the embedding table.
        tokenizer_dir = Path(cfg.data.tokenizer_dir)
        self.tokenizer = Tokenizer.load(tokenizer_dir / "spm.model")

        lm_cfg = LMConfig(
            vocab_size=self.tokenizer.vocab_size,
            n_layers=int(cfg.model.n_layers),
            d_model=int(cfg.model.d_model),
            n_heads=int(cfg.model.n_heads),
            d_ff=int(cfg.model.d_ff) if cfg.model.get("d_ff") else None,
            max_seq_len=int(cfg.model.max_seq_len),
            dropout=float(cfg.model.dropout),
            rope_base=float(cfg.model.rope_base),
            pad_id=self.tokenizer.pad_id,
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            sep_id=self.tokenizer.sep_id,
        )
        self.lm_cfg = lm_cfg
        self.model = SuggestLM(lm_cfg)
        self.records_seen = 0

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)

    def _compute_loss(self, batch):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch["labels"].reshape(-1),
            ignore_index=LABEL_IGNORE,
        )
        return logits, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._compute_loss(batch)
        bs = int(batch["input_ids"].size(0))
        self.records_seen += bs
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=bs,
        )
        with torch.no_grad():
            self.log(
                "train/ppl", torch.exp(loss.detach()), on_step=True,
                on_epoch=True, prog_bar=True, batch_size=bs,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss = self._compute_loss(batch)
        bs = int(batch["input_ids"].size(0))
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=bs, sync_dist=True,
        )
        self.log(
            "val/ppl", torch.exp(loss), on_step=False, on_epoch=True,
            prog_bar=True, batch_size=bs, sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.optimizer.lr),
            betas=tuple(self.cfg.optimizer.betas),
            weight_decay=float(self.cfg.optimizer.weight_decay),
        )
        scheduler_name = str(self.cfg.optimizer.scheduler).lower()
        if scheduler_name == "none":
            return opt
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup = int(self.cfg.optimizer.warmup_steps)
        if total_steps <= 0:
            return opt
        warmup = min(warmup, total_steps)
        sched = get_scheduler(
            scheduler_name,
            optimizer=opt,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }


def _resolve_git_branch() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    branch = result.stdout.strip()
    return branch or None


def _build_logger(cfg: DictConfig) -> MLFlowLogger | None:
    if not bool(cfg.logger.enabled):
        return None
    tags = OmegaConf.to_container(cfg.logger.get("tags") or {}, resolve=True) or {}
    branch = _resolve_git_branch()
    if branch:
        tags["git_branch"] = branch
    return MLFlowLogger(
        experiment_name=cfg.logger.experiment_name,
        tracking_uri=cfg.logger.tracking_uri,
        run_name=cfg.logger.run_name,
        log_model=bool(cfg.logger.log_model),
        tags=tags,
    )


def _build_callbacks(cfg: DictConfig) -> list[Callback]:
    callbacks: list[Callback] = [
        LearningRateMonitor(logging_interval="step"),
        DatasetStatsLogger(),
    ]
    if not bool(cfg.trainer.enable_checkpointing):
        return callbacks
    ckpt_dir = Path(cfg.trainer.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor = cfg.trainer.monitor
    safe_monitor = str(monitor).replace("/", "_")
    cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"best-step={{step}}-{safe_monitor}={{{monitor}:.4f}}",
        monitor=monitor,
        mode=cfg.trainer.monitor_mode,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    cb.CHECKPOINT_NAME_LAST = f"last-step={{step}}-{safe_monitor}={{{monitor}:.4f}}"
    return [cb, *callbacks]


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def run(cfg: DictConfig) -> None:
    load_dotenv()
    logging.basicConfig(level=str(cfg.log_level).upper())
    torch.set_float32_matmul_precision("high")
    L.seed_everything(int(cfg.seed), workers=True)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    datamodule = SuggestLMDataModule(
        targets_dir=Path(cfg.data.targets_dir),
        pairs_dir=Path(cfg.data.get("pairs_dir") or cfg.data.targets_dir),
        tokenizer_dir=Path(cfg.data.tokenizer_dir),
        max_seq_len=int(cfg.model.max_seq_len),
        batch_size=int(cfg.data.batch_size),
        val_batch_size=int(cfg.data.val_batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        train_samples_per_epoch=cfg.data.get("train_samples_per_epoch") or None,
        seed=int(cfg.seed),
        variant=str(cfg.data.get("variant", "a")),
    )

    model = SuggestLMModule(cfg)
    print(f"Model params: {model.model.num_parameters():,}", flush=True)

    logger = _build_logger(cfg)
    if logger is not None:
        _ = logger.experiment

    callbacks = _build_callbacks(cfg)
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=int(cfg.trainer.max_epochs),
        max_time=cfg.trainer.get("max_time"),
        precision=cfg.trainer.precision,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        gradient_clip_val=float(cfg.trainer.gradient_clip_val),
        deterministic=False,
        limit_train_batches=cfg.trainer.get("limit_train_batches"),
        limit_val_batches=cfg.trainer.get("limit_val_batches"),
        val_check_interval=cfg.trainer.get("val_check_interval"),
        enable_checkpointing=bool(cfg.trainer.enable_checkpointing),
        default_root_dir=str(Path.cwd()),
        logger=logger,
        callbacks=callbacks,
    )

    if logger is not None:
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer.fit(model, datamodule=datamodule)

    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
            print(f"Best checkpoint: {cb.best_model_path}", flush=True)


def main() -> None:
    # Force the data root to flow into config defaults if user hasn't set it.
    os.environ.setdefault("SUGGEST_DATA_ROOT", str(DATA_ROOT))
    run()


if __name__ == "__main__":
    main()
