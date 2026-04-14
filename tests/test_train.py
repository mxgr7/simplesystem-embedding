import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf

from embedding_train.train import build_callbacks, configure_logging, run


class _LoggerStub:
    def __init__(self, run_name, run_id="run-123"):
        self.run_id = run_id
        self.experiment = self
        self._run = SimpleNamespace(
            data=SimpleNamespace(tags={"mlflow.runName": run_name})
        )

    def get_run(self, run_id):
        self.last_run_id = run_id
        return self._run


class TrainCallbackTests(unittest.TestCase):
    def test_configure_logging_accepts_warning_level(self):
        configure_logging("WARNING")

        self.assertEqual(logging.getLogger().level, logging.WARNING)

    def test_build_callbacks_saves_to_run_name_subdirectory(self):
        with TemporaryDirectory() as tmp_dir:
            cfg = OmegaConf.create(
                {
                    "trainer": {"checkpoint_dir": tmp_dir},
                }
            )
            logger = _LoggerStub("train-run-001")

            callbacks = build_callbacks(cfg, logger)

        checkpoint_callback = next(
            callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
        )

        self.assertEqual(
            Path(checkpoint_callback.dirpath).resolve(),
            (Path(tmp_dir) / "train-run-001").resolve(),
        )
        self.assertEqual(
            checkpoint_callback.filename,
            "best-step={step}-val_full_catalog_ndcg_at_5={val/full_catalog/ndcg_at_5:.4f}",
        )
        self.assertEqual(checkpoint_callback.monitor, "val/full_catalog/ndcg_at_5")
        self.assertEqual(checkpoint_callback.mode, "max")
        self.assertEqual(checkpoint_callback.save_top_k, 1)
        self.assertTrue(checkpoint_callback.save_last)
        self.assertFalse(checkpoint_callback.auto_insert_metric_name)
        self.assertEqual(
            checkpoint_callback.CHECKPOINT_NAME_LAST,
            "last-step={step}-val_full_catalog_ndcg_at_5={val/full_catalog/ndcg_at_5:.4f}",
        )

    def test_build_callbacks_includes_learning_rate_monitor(self):
        with TemporaryDirectory() as tmp_dir:
            cfg = OmegaConf.create(
                {
                    "trainer": {"checkpoint_dir": tmp_dir},
                }
            )
            logger = _LoggerStub("train-run-001")

            callbacks = build_callbacks(cfg, logger)

        lr_monitor = next(
            callback
            for callback in callbacks
            if isinstance(callback, LearningRateMonitor)
        )
        self.assertEqual(lr_monitor.logging_interval, "step")

    def test_build_callbacks_sanitizes_run_name_for_paths(self):
        with TemporaryDirectory() as tmp_dir:
            cfg = OmegaConf.create({"trainer": {"checkpoint_dir": tmp_dir}})
            logger = _LoggerStub("nested/run\\name")

            callbacks = build_callbacks(cfg, logger)

        checkpoint_callback = next(
            callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
        )

        self.assertEqual(
            Path(checkpoint_callback.dirpath).resolve(),
            (Path(tmp_dir) / "nested-run-name").resolve(),
        )

    @patch("embedding_train.train.load_dotenv")
    @patch("embedding_train.train.configure_logging")
    @patch("embedding_train.train.L.seed_everything")
    @patch("embedding_train.train.torch.set_float32_matmul_precision")
    @patch("embedding_train.train.build_callbacks", return_value=[])
    @patch("embedding_train.train.build_logger")
    @patch("embedding_train.train.EmbeddingModule")
    @patch("embedding_train.train.EmbeddingDataModule")
    @patch("embedding_train.train.L.Trainer")
    def test_run_passes_resume_checkpoint_to_trainer_fit(
        self,
        trainer_cls,
        data_module_cls,
        module_cls,
        build_logger,
        build_callbacks,
        set_matmul_precision,
        seed_everything,
        configure_logging,
        load_dotenv,
    ):
        cfg = OmegaConf.create(
            {
                "log_level": "WARNING",
                "seed": 42,
                "trainer": {
                    "accelerator": "cpu",
                    "devices": 1,
                    "max_epochs": 1,
                    "precision": "32-true",
                    "log_every_n_steps": 10,
                    "accumulate_grad_batches": 1,
                    "deterministic": False,
                    "limit_train_batches": 1.0,
                    "limit_val_batches": 1.0,
                    "val_check_interval": 1000,
                    "validate_before_training": False,
                    "resume_from_checkpoint": "checkpoints/run-001/last.ckpt",
                },
            }
        )
        trainer = Mock()
        logger = Mock()
        datamodule = Mock()
        model = Mock()

        trainer_cls.return_value = trainer
        build_logger.return_value = logger
        data_module_cls.return_value = datamodule
        module_cls.return_value = model
        datamodule.dataset_stats = {}

        run.__wrapped__(cfg)

        configure_logging.assert_called_once_with("WARNING")
        trainer.fit.assert_called_once_with(
            model,
            datamodule=datamodule,
            ckpt_path="checkpoints/run-001/last.ckpt",
        )


if __name__ == "__main__":
    unittest.main()
