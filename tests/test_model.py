import unittest

import torch
from omegaconf import OmegaConf

from embedding_train.model import EmbeddingModule


class _LogCaptureModule:
    def __init__(self, log_batch_stats):
        self.cfg = OmegaConf.create({"data": {"log_batch_stats": log_batch_stats}})
        self.logged = []

    def log(self, name, value, **kwargs):
        self.logged.append({"name": name, "value": value, "kwargs": kwargs})


class EmbeddingModuleBatchLoggingTests(unittest.TestCase):
    def test_logs_training_batch_stats_when_present_and_enabled(self):
        module = _LogCaptureModule(log_batch_stats=True)
        batch = {
            "labels": torch.tensor([1.0, 0.0, 0.0]),
            "batch_stats": {
                "positive_count": 1,
                "same_query_negative_count": 1,
                "cross_query_negative_count": 1,
            },
        }

        EmbeddingModule.log_training_batch_stats(module, batch)

        self.assertEqual(
            [entry["name"] for entry in module.logged],
            [
                "train/batch_positive_count",
                "train/batch_same_query_negative_count",
                "train/batch_cross_query_negative_count",
            ],
        )
        self.assertEqual([entry["value"] for entry in module.logged], [1.0, 1.0, 1.0])
        self.assertTrue(all(entry["kwargs"]["on_step"] for entry in module.logged))
        self.assertTrue(all(entry["kwargs"]["on_epoch"] for entry in module.logged))
        self.assertEqual(
            [entry["kwargs"]["batch_size"] for entry in module.logged],
            [3, 3, 3],
        )

    def test_skips_logging_when_disabled(self):
        module = _LogCaptureModule(log_batch_stats=False)
        batch = {
            "labels": torch.tensor([1.0, 0.0]),
            "batch_stats": {
                "positive_count": 1,
                "same_query_negative_count": 1,
                "cross_query_negative_count": 0,
            },
        }

        EmbeddingModule.log_training_batch_stats(module, batch)

        self.assertEqual(module.logged, [])

    def test_skips_logging_when_batch_stats_are_missing(self):
        module = _LogCaptureModule(log_batch_stats=True)
        batch = {"labels": torch.tensor([1.0, 0.0])}

        EmbeddingModule.log_training_batch_stats(module, batch)

        self.assertEqual(module.logged, [])


if __name__ == "__main__":
    unittest.main()
