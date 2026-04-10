import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from embedding_train.model import (
    EmbeddingModule,
    load_embedding_module_from_checkpoint,
    resolve_scheduler_name,
    resolve_warmup_steps,
)


class _EncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _EncoderStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)
        self.config = type("Config", (), {"hidden_size": 4})()

    def forward(self, input_ids=None, attention_mask=None):
        del attention_mask
        return _EncoderOutput(self.embedding(input_ids))


def build_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "model": {
                "model_name": "stub-model",
                "output_dim": None,
                "pooling": "mean",
                "loss_type": "contrastive",
                "similarity_scale": 20.0,
                "triplet_margin": 0.2,
                "gradient_checkpointing": False,
            },
            "trainer": {"precision": "32-true"},
            "data": {"log_batch_stats": True},
            "optimizer": {
                "lr": 2.0e-5,
                "weight_decay": 0.01,
                "scheduler": "linear",
                "warmup_ratio": 0.1,
                "warmup_steps": None,
            },
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


class _LogCaptureModule:
    def __init__(self, log_batch_stats, log_every_n_steps=1):
        self.cfg = OmegaConf.create({"data": {"log_batch_stats": log_batch_stats}})
        self.logged = []
        self.records_seen = 12
        self.pending_train_record_metrics = {}
        self.record_metrics = []
        self.logger = None
        self.trainer = SimpleNamespace(log_every_n_steps=log_every_n_steps)

    def log(self, name, value, **kwargs):
        self.logged.append({"name": name, "value": value, "kwargs": kwargs})

    def record_aligned_metric_name(self, name):
        return EmbeddingModule.record_aligned_metric_name(self, name)

    def batch_aligned_metric_name(self, split, name):
        return EmbeddingModule.batch_aligned_metric_name(self, split, name)

    def metric_value_to_float(self, value):
        return EmbeddingModule.metric_value_to_float(self, value)

    def log_metrics_by_records(self, metrics):
        self.record_metrics.append(
            {
                "step": self.records_seen,
                "metrics": {
                    self.record_aligned_metric_name(name): self.metric_value_to_float(
                        value
                    )
                    for name, value in metrics.items()
                },
            }
        )

    def resolve_batch_record_count(self, batch):
        return EmbeddingModule.resolve_batch_record_count(self, batch)

    def should_log_records_on_batch(self, batch_idx):
        return EmbeddingModule.should_log_records_on_batch(self, batch_idx)

    def resolve_log_every_n_steps(self):
        return EmbeddingModule.resolve_log_every_n_steps(self)


class _MLflowLoggerStub:
    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step):
        self.calls.append({"metrics": metrics, "step": step})


class _ValidationLogCaptureModule:
    def __init__(self, validation_rows, records_seen=24, sanity_checking=False):
        self.validation_rows = validation_rows
        self.logged = []
        self.loss_type = "bce"
        self.records_seen = records_seen
        self.validation_loss_total = 1.5
        self.validation_loss_examples = 3
        self.validation_mode = "pairwise_proxy"
        self.logger = _MLflowLoggerStub()
        self.trainer = SimpleNamespace(sanity_checking=sanity_checking)

    def log(self, name, value, **kwargs):
        self.logged.append({"name": name, "value": value, "kwargs": kwargs})

    def record_aligned_metric_name(self, name):
        return EmbeddingModule.record_aligned_metric_name(self, name)

    def batch_aligned_metric_name(self, split, name):
        return EmbeddingModule.batch_aligned_metric_name(self, split, name)

    def metric_value_to_float(self, value):
        return EmbeddingModule.metric_value_to_float(self, value)

    def log_metrics_by_records(self, metrics):
        EmbeddingModule.log_metrics_by_records(self, metrics)

    def is_sanity_checking(self):
        return EmbeddingModule.is_sanity_checking(self)

    def _compute_pairwise_validation_metrics(self):
        return EmbeddingModule._compute_pairwise_validation_metrics(self)


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

        stats_to_log = EmbeddingModule.log_training_batch_stats(module, batch)

        self.assertEqual(
            [entry["name"] for entry in module.logged],
            [
                "train/by_batch/batch_positive_count",
                "train/by_batch/batch_same_query_negative_count",
                "train/by_batch/batch_cross_query_negative_count",
                "train/by_batch/batch_hard_negative_count",
                "train/by_batch/batch_hard_negative_share",
            ],
        )
        self.assertEqual([entry["value"] for entry in module.logged], [1.0, 1.0, 1.0, 0.0, 0.0])
        self.assertTrue(all(entry["kwargs"]["on_step"] for entry in module.logged))
        self.assertTrue(all(entry["kwargs"]["on_epoch"] for entry in module.logged))
        self.assertEqual(
            [entry["kwargs"]["batch_size"] for entry in module.logged],
            [3, 3, 3, 3, 3],
        )
        self.assertEqual(
            stats_to_log,
            {
                "train/by_batch/batch_positive_count": 1,
                "train/by_batch/batch_same_query_negative_count": 1,
                "train/by_batch/batch_cross_query_negative_count": 1,
                "train/by_batch/batch_hard_negative_count": 0,
                "train/by_batch/batch_hard_negative_share": 0.0,
            },
        )
        self.assertEqual(module.record_metrics, [])

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

    def test_logs_record_metrics_after_train_batch_end(self):
        module = _LogCaptureModule(log_batch_stats=True)
        module.records_seen = 0
        module.pending_train_record_metrics = {
            "train/by_batch/loss": torch.tensor(0.25),
            "train/by_batch/batch_positive_count": 1.0,
        }
        batch = {"labels": torch.tensor([1.0, 0.0, 0.0])}

        EmbeddingModule.on_train_batch_end(module, None, batch, 0)

        self.assertEqual(module.records_seen, 3)
        self.assertEqual(module.pending_train_record_metrics, {})
        self.assertEqual(
            module.record_metrics,
            [
                {
                    "step": 3,
                    "metrics": {
                        "train/loss": 0.25,
                        "train/batch_positive_count": 1.0,
                    },
                }
            ],
        )

    def test_skips_record_metrics_until_log_every_n_steps_boundary(self):
        module = _LogCaptureModule(log_batch_stats=True, log_every_n_steps=10)
        module.records_seen = 0
        module.pending_train_record_metrics = {
            "train/by_batch/loss": torch.tensor(0.25),
        }
        batch = {"labels": torch.tensor([1.0, 0.0, 0.0])}

        EmbeddingModule.on_train_batch_end(module, None, batch, 8)

        self.assertEqual(module.records_seen, 3)
        self.assertEqual(module.pending_train_record_metrics, {})
        self.assertEqual(module.record_metrics, [])

    def test_logs_record_metrics_on_log_every_n_steps_boundary(self):
        module = _LogCaptureModule(log_batch_stats=True, log_every_n_steps=10)
        module.records_seen = 27
        module.pending_train_record_metrics = {
            "train/by_batch/loss": torch.tensor(0.25),
        }
        batch = {"labels": torch.tensor([1.0, 0.0, 0.0])}

        EmbeddingModule.on_train_batch_end(module, None, batch, 9)

        self.assertEqual(module.records_seen, 30)
        self.assertEqual(
            module.record_metrics,
            [
                {
                    "step": 30,
                    "metrics": {"train/loss": 0.25},
                }
            ],
        )


class EmbeddingModuleValidationMetricTests(unittest.TestCase):
    def test_logs_exact_metrics_alongside_ndcg(self):
        module = _ValidationLogCaptureModule(
            [
                {"query_id": "q1", "score": 0.9, "raw_label": "Irrelevant"},
                {"query_id": "q1", "score": 0.8, "raw_label": "Exact"},
                {"query_id": "q2", "score": 0.7, "raw_label": "Exact"},
                {"query_id": "q2", "score": 0.6, "raw_label": "Irrelevant"},
                {"query_id": "q3", "score": 0.5, "raw_label": "Substitute"},
            ]
        )

        EmbeddingModule.on_validation_epoch_end(module)

        logged_by_name = {entry["name"]: entry for entry in module.logged}

        self.assertIn("val/by_batch/ndcg_at_1", logged_by_name)
        self.assertIn("val/by_batch/ndcg_at_5", logged_by_name)
        self.assertIn("val/by_batch/exact_success_at_1", logged_by_name)
        self.assertIn("val/by_batch/exact_mrr", logged_by_name)
        self.assertIn("val/by_batch/exact_recall_at_5", logged_by_name)
        self.assertIn("val/by_batch/exact_recall_at_10", logged_by_name)
        self.assertIn("val/by_batch/eligible_queries", logged_by_name)
        self.assertIn("val/by_batch/evaluated_queries", logged_by_name)
        self.assertAlmostEqual(
            logged_by_name["val/by_batch/exact_success_at_1"]["value"],
            0.5,
        )
        self.assertAlmostEqual(logged_by_name["val/by_batch/exact_mrr"]["value"], 0.75)
        self.assertAlmostEqual(
            logged_by_name["val/by_batch/exact_recall_at_5"]["value"], 1.0
        )
        self.assertAlmostEqual(
            logged_by_name["val/by_batch/evaluated_queries"]["value"],
            3.0,
        )
        self.assertTrue(
            logged_by_name["val/by_batch/exact_success_at_1"]["kwargs"]["prog_bar"]
        )
        self.assertTrue(logged_by_name["val/by_batch/exact_mrr"]["kwargs"]["prog_bar"])
        self.assertEqual(len(module.logger.calls), 1)
        self.assertEqual(module.logger.calls[0]["step"], 24)
        self.assertAlmostEqual(
            module.logger.calls[0]["metrics"]["val/loss"],
            0.5,
        )
        self.assertIn("val/exact_mrr", module.logger.calls[0]["metrics"])
        self.assertIn("val/ndcg_at_1", module.logger.calls[0]["metrics"])

    def test_skips_record_aligned_validation_metrics_before_training(self):
        module = _ValidationLogCaptureModule([], records_seen=0, sanity_checking=True)

        EmbeddingModule.on_validation_epoch_end(module)

        self.assertEqual(module.logger.calls, [])


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
        self.assertEqual(loaded_cfg.model.output_dim, cfg.model.output_dim)
        self.assertEqual(loaded_cfg.model.pooling, cfg.model.pooling)
        self.assertEqual(loaded_cfg.trainer.precision, cfg.trainer.precision)
        self.assertFalse(loaded_module.training)

        for name, tensor in module.state_dict().items():
            self.assertTrue(torch.equal(tensor, loaded_module.state_dict()[name]))

    @patch("embedding_train.model.AutoModel.from_pretrained")
    def test_loads_checkpoint_with_projection_parameters(self, from_pretrained):
        from_pretrained.side_effect = lambda *args, **kwargs: _EncoderStub()
        cfg = build_cfg(model={"output_dim": 2})
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

        self.assertEqual(loaded_cfg.model.output_dim, 2)
        self.assertIsInstance(loaded_module.projection, torch.nn.Linear)
        self.assertEqual(loaded_module.projection.out_features, 2)

    @patch("embedding_train.model.AutoModel.from_pretrained")
    def test_persists_records_seen_in_checkpoint_hooks(self, from_pretrained):
        from_pretrained.return_value = _EncoderStub()
        module = EmbeddingModule(build_cfg())
        module.records_seen = 128
        checkpoint = {}

        module.on_save_checkpoint(checkpoint)
        module.records_seen = 0
        module.on_load_checkpoint(checkpoint)

        self.assertEqual(checkpoint["records_seen"], 128)
        self.assertEqual(module.records_seen, 128)

    def test_skips_logging_when_batch_stats_are_missing(self):
        module = _LogCaptureModule(log_batch_stats=True)
        batch = {"labels": torch.tensor([1.0, 0.0])}

        EmbeddingModule.log_training_batch_stats(module, batch)

        self.assertEqual(module.logged, [])


class EmbeddingModuleProjectionTests(unittest.TestCase):
    @patch("embedding_train.model.AutoModel.from_pretrained")
    def test_keeps_encoder_hidden_size_when_output_dim_is_unset(self, from_pretrained):
        from_pretrained.return_value = _EncoderStub()
        module = EmbeddingModule(build_cfg())
        inputs = {
            "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
        }

        embeddings = module.encode(inputs)

        self.assertIsNone(module.projection)
        self.assertEqual(tuple(embeddings.shape), (2, 4))

    @patch("embedding_train.model.AutoModel.from_pretrained")
    def test_projects_embeddings_to_configured_output_dim(self, from_pretrained):
        from_pretrained.return_value = _EncoderStub()
        module = EmbeddingModule(build_cfg(model={"output_dim": 2}))
        inputs = {
            "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
        }

        embeddings = module.encode(inputs)

        self.assertIsInstance(module.projection, torch.nn.Linear)
        self.assertEqual(module.projection.in_features, 4)
        self.assertEqual(module.projection.out_features, 2)
        self.assertEqual(tuple(embeddings.shape), (2, 2))

    @patch("embedding_train.model.AutoModel.from_pretrained")
    def test_rejects_non_positive_output_dim(self, from_pretrained):
        from_pretrained.return_value = _EncoderStub()

        with self.assertRaisesRegex(ValueError, "output_dim must be at least 1"):
            EmbeddingModule(build_cfg(model={"output_dim": 0}))


class OptimizerSchedulerTests(unittest.TestCase):
    def test_resolve_scheduler_name_accepts_none_aliases(self):
        self.assertEqual(resolve_scheduler_name("none"), "none")
        self.assertEqual(resolve_scheduler_name("off"), "none")

    def test_resolve_scheduler_name_rejects_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "Unsupported optimizer scheduler"):
            resolve_scheduler_name("exponential")

    def test_resolve_warmup_steps_uses_ratio_when_steps_are_not_set(self):
        optimizer_cfg = OmegaConf.create({"warmup_ratio": 0.1, "warmup_steps": None})

        self.assertEqual(resolve_warmup_steps(optimizer_cfg, 200), 20)

    def test_resolve_warmup_steps_prefers_explicit_steps(self):
        optimizer_cfg = OmegaConf.create({"warmup_ratio": 0.1, "warmup_steps": 25})

        self.assertEqual(resolve_warmup_steps(optimizer_cfg, 200), 25)

    @patch("embedding_train.model.get_scheduler")
    @patch("embedding_train.model.AutoModel.from_pretrained")
    def test_configure_optimizers_builds_step_scheduler(
        self, from_pretrained, get_scheduler
    ):
        from_pretrained.return_value = _EncoderStub()
        scheduler = object()
        get_scheduler.return_value = scheduler
        module = EmbeddingModule(build_cfg())
        module._trainer = SimpleNamespace(estimated_stepping_batches=1000)

        configured = module.configure_optimizers()

        self.assertIsInstance(configured["optimizer"], torch.optim.AdamW)
        self.assertIs(configured["lr_scheduler"]["scheduler"], scheduler)
        self.assertEqual(configured["lr_scheduler"]["interval"], "step")
        self.assertEqual(configured["lr_scheduler"]["frequency"], 1)
        get_scheduler.assert_called_once()
        self.assertEqual(get_scheduler.call_args.args[0], "linear")
        self.assertEqual(get_scheduler.call_args.kwargs["num_warmup_steps"], 100)
        self.assertEqual(get_scheduler.call_args.kwargs["num_training_steps"], 1000)

    @patch("embedding_train.model.get_scheduler")
    @patch("embedding_train.model.AutoModel.from_pretrained")
    def test_configure_optimizers_returns_bare_optimizer_when_scheduler_disabled(
        self, from_pretrained, get_scheduler
    ):
        from_pretrained.return_value = _EncoderStub()
        module = EmbeddingModule(build_cfg(optimizer={"scheduler": "none"}))
        module._trainer = SimpleNamespace(estimated_stepping_batches=1000)

        configured = module.configure_optimizers()

        self.assertIsInstance(configured, torch.optim.AdamW)
        get_scheduler.assert_not_called()


class EmbeddingModuleRecordLoggingTests(unittest.TestCase):
    def test_logs_mlflow_metrics_with_record_aligned_step(self):
        logger = _MLflowLoggerStub()
        module = _ValidationLogCaptureModule([])
        module.logger = logger
        module.records_seen = 64

        EmbeddingModule.log_metrics_by_records(
            module, {"train/loss": torch.tensor(0.25)}
        )

        self.assertEqual(
            logger.calls,
            [
                {
                    "step": 64,
                    "metrics": {"train/loss": 0.25},
                }
            ],
        )

    def test_skips_mlflow_record_metrics_until_records_are_seen(self):
        logger = _MLflowLoggerStub()
        module = _ValidationLogCaptureModule([], records_seen=0)
        module.logger = logger

        EmbeddingModule.log_metrics_by_records(module, {"train/by_batch/loss": 0.25})

        self.assertEqual(logger.calls, [])


if __name__ == "__main__":
    unittest.main()
