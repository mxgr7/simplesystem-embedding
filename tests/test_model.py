import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from embedding_train.model import EmbeddingModule, load_embedding_module_from_checkpoint


class _EncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _EncoderStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)

    def forward(self, input_ids=None, attention_mask=None):
        del attention_mask
        return _EncoderOutput(self.embedding(input_ids))


def build_cfg(**overrides):
    cfg = {
        "seed": 42,
        "model": {
            "model_name": "stub-model",
            "pooling": "mean",
            "loss_type": "bce",
            "similarity_scale": 20.0,
            "triplet_margin": 0.2,
            "gradient_checkpointing": False,
        },
        "trainer": {"precision": "32-true"},
        "data": {"log_batch_stats": True},
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


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


class EmbeddingModuleCheckpointTests(unittest.TestCase):
    @patch("embedding_train.model.AutoModel.from_pretrained")
    def test_loads_checkpoint_using_saved_hyperparameters(self, from_pretrained):
        from_pretrained.side_effect = lambda *args, **kwargs: _EncoderStub()
        cfg = build_cfg()
        module = EmbeddingModule(cfg)

        checkpoint = {
            "hyper_parameters": OmegaConf.to_container(cfg, resolve=True),
            "state_dict": module.state_dict(),
        }

        with TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "model.ckpt"
            torch.save(checkpoint, checkpoint_path)

            loaded_module, loaded_cfg = load_embedding_module_from_checkpoint(
                checkpoint_path
            )

        self.assertEqual(loaded_cfg.model.model_name, cfg.model.model_name)
        self.assertEqual(loaded_cfg.model.pooling, cfg.model.pooling)
        self.assertEqual(loaded_cfg.trainer.precision, cfg.trainer.precision)
        self.assertFalse(loaded_module.training)

        for name, tensor in module.state_dict().items():
            self.assertTrue(torch.equal(tensor, loaded_module.state_dict()[name]))

    def test_skips_logging_when_batch_stats_are_missing(self):
        module = _LogCaptureModule(log_batch_stats=True)
        batch = {"labels": torch.tensor([1.0, 0.0])}

        EmbeddingModule.log_training_batch_stats(module, batch)

        self.assertEqual(module.logged, [])


if __name__ == "__main__":
    unittest.main()
