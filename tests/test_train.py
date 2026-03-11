import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf

from embedding_train.train import build_callbacks


class TrainCallbackTests(unittest.TestCase):
    def test_build_callbacks_configures_checkpoint_names_with_step_and_metric(self):
        with TemporaryDirectory() as tmp_dir:
            cfg = OmegaConf.create({"trainer": {"checkpoint_dir": tmp_dir}})

            callbacks = build_callbacks(cfg)

        checkpoint_callback = next(
            callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
        )

        self.assertEqual(
            Path(checkpoint_callback.dirpath).resolve(), Path(tmp_dir).resolve()
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
            cfg = OmegaConf.create({"trainer": {"checkpoint_dir": tmp_dir}})

            callbacks = build_callbacks(cfg)

        self.assertTrue(
            any(isinstance(callback, LearningRateMonitor) for callback in callbacks)
        )


if __name__ == "__main__":
    unittest.main()
