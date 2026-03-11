import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf

from embedding_train.train import build_callbacks


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
            "best-step={step}-val_ndcg_at_5={val/ndcg_at_5:.4f}",
        )
        self.assertEqual(checkpoint_callback.monitor, "val/ndcg_at_5")
        self.assertEqual(checkpoint_callback.mode, "max")
        self.assertEqual(checkpoint_callback.save_top_k, 3)
        self.assertTrue(checkpoint_callback.save_last)
        self.assertFalse(checkpoint_callback.auto_insert_metric_name)
        self.assertEqual(
            checkpoint_callback.CHECKPOINT_NAME_LAST,
            "last-step={step}-val_ndcg_at_5={val/ndcg_at_5:.4f}",
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

        self.assertTrue(
            any(isinstance(callback, LearningRateMonitor) for callback in callbacks)
        )

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


if __name__ == "__main__":
    unittest.main()
