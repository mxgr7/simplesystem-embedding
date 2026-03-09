from pathlib import Path

import hydra
import lightning as L
from dotenv import load_dotenv
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf

from embedding_train.data import EmbeddingDataModule
from embedding_train.model import EmbeddingModule


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def build_logger(cfg):
    tags = OmegaConf.to_container(cfg.logger.tags, resolve=True)
    return MLFlowLogger(
        experiment_name=cfg.logger.experiment_name,
        tracking_uri=cfg.logger.tracking_uri,
        run_name=cfg.logger.run_name,
        log_model=bool(cfg.logger.log_model),
        tags=tags,
    )


def build_callbacks(cfg):
    checkpoint_dir = Path(cfg.trainer.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def run(cfg):
    load_dotenv()
    L.seed_everything(int(cfg.seed), workers=True)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    datamodule = EmbeddingDataModule(cfg)
    model = EmbeddingModule(cfg)
    logger = build_logger(cfg)
    callbacks = build_callbacks(cfg)

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=int(cfg.trainer.max_epochs),
        precision=cfg.trainer.precision,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        deterministic=bool(cfg.trainer.deterministic),
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        default_root_dir=str(Path.cwd()),
        logger=logger,
        callbacks=callbacks,
    )

    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer.fit(model, datamodule=datamodule)

    best_model_path = ""
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_model_path = callback.best_model_path
            break

    if best_model_path:
        print(f"Best checkpoint: {best_model_path}")


def main():
    run()


if __name__ == "__main__":
    main()
