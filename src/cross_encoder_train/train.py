import logging
import subprocess
from pathlib import Path

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf

from cross_encoder_train.data import CrossEncoderDataModule
from cross_encoder_train.model import CrossEncoderModule


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
MLFLOW_RUN_NAME_TAG = "mlflow.runName"


def _resolve_git_branch():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    branch = result.stdout.strip()
    return branch or None


def configure_logging(log_level):
    level_name = str(log_level).strip().upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {log_level}")
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)


def build_logger(cfg):
    tags = OmegaConf.to_container(cfg.logger.tags, resolve=True) or {}
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


def _sanitize_path_component(value):
    value = str(value).strip().replace("/", "-").replace("\\", "-")
    return value or "run"


def _resolve_checkpoint_run_name(logger):
    run = logger.experiment.get_run(logger.run_id)
    run_name = run.data.tags.get(MLFLOW_RUN_NAME_TAG) or logger.run_id
    return _sanitize_path_component(run_name)


class DatasetStatsLogger(Callback):
    def __init__(self):
        self._logged = False

    def setup(self, trainer, pl_module, stage):
        if self._logged or trainer.datamodule is None:
            return
        dataset_stats = getattr(trainer.datamodule, "dataset_stats", None)
        if not dataset_stats:
            return
        trainer.logger.log_hyperparams(
            {f"dataset/{k}": v for k, v in dataset_stats.items()}
        )
        self._logged = True


def build_callbacks(cfg, logger):
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        DatasetStatsLogger(),
    ]

    if not getattr(cfg.trainer, "enable_checkpointing", True):
        return callbacks

    checkpoint_dir = Path(cfg.trainer.checkpoint_dir) / _resolve_checkpoint_run_name(
        logger
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monitor = str(cfg.trainer.monitor_metric)
    safe_monitor = monitor.replace("/", "_")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"best-step={{step}}-{safe_monitor}={{{monitor}:.4f}}",
        monitor=monitor,
        mode=str(cfg.trainer.monitor_mode),
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = (
        f"last-step={{step}}-{safe_monitor}={{{monitor}:.4f}}"
    )

    return [checkpoint_callback, *callbacks]


def fit_from_cfg(cfg):
    load_dotenv()
    configure_logging(cfg.log_level)
    torch.set_float32_matmul_precision("high")
    L.seed_everything(int(cfg.seed), workers=True)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    datamodule = CrossEncoderDataModule(cfg)
    model = CrossEncoderModule(cfg)
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
        enable_checkpointing=bool(
            getattr(cfg.trainer, "enable_checkpointing", True)
        ),
        default_root_dir=str(Path.cwd()),
        logger=logger,
        callbacks=callbacks,
    )

    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    if cfg.trainer.validate_before_training:
        original_finalize = logger.finalize
        logger.finalize = lambda *args, **kwargs: None
        trainer.validate(model, datamodule=datamodule)
        logger.finalize = original_finalize

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=cfg.trainer.resume_from_checkpoint,
    )

    best_model_path = ""
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_model_path = callback.best_model_path
            break

    if best_model_path:
        print(f"Best checkpoint: {best_model_path}")


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="cross_encoder")
def run(cfg):
    fit_from_cfg(cfg)


def main():
    run()


if __name__ == "__main__":
    main()
