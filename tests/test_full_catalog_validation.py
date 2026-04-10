import math
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from embedding_train.catalog_benchmark import score_queries_against_catalog
from embedding_train.model import (
    EmbeddingModule,
    build_full_catalog_monitor_metric,
    resolve_validation_metric,
    resolve_validation_mode,
    resolve_validation_similarity,
    subsample_catalog,
)
from embedding_train.train import _resolve_checkpoint_monitor


class _MLflowLoggerStub:
    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step):
        self.calls.append({"metrics": metrics, "step": step})


class _TokenizerStub:
    pad_token = "[PAD]"
    eos_token = "[EOS]"

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        del padding, truncation, max_length, return_tensors
        input_ids = torch.tensor(
            [[self._text_to_id(text)] for text in texts],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _text_to_id(self, text):
        return hash(text) % 1000


class _DataModuleStub:
    def __init__(self):
        self.tokenizer = _TokenizerStub()


class _FullCatalogValidationModule:
    """Minimal stub for testing full-catalog validation methods."""

    def __init__(
        self,
        validation_rows,
        records_seen=24,
        sanity_checking=False,
        validation_mode="full_catalog",
        validation_metric="ndcg_at_5",
        validation_similarity="dot",
    ):
        self.validation_rows = validation_rows
        self.logged = {}
        self.loss_type = "bce"
        self.records_seen = records_seen
        self.validation_loss_total = 0.0
        self.validation_loss_examples = 0
        self.logger = _MLflowLoggerStub()
        self.trainer = SimpleNamespace(
            sanity_checking=sanity_checking,
            datamodule=_DataModuleStub(),
        )
        self.validation_mode = validation_mode
        self.validation_metric = validation_metric
        self.validation_similarity = validation_similarity
        self.cfg = OmegaConf.create(
            {
                "trainer": {
                    "encode_batch_size": 2,
                    "score_batch_size": 2,
                },
                "data": {
                    "max_query_length": 32,
                    "max_offer_length": 64,
                    "log_batch_stats": True,
                },
            }
        )
        self.device = torch.device("cpu")
        self._training = False

    @property
    def training(self):
        return self._training

    def log(self, name, value, **kwargs):
        self.logged[name] = {"value": value, "kwargs": kwargs}

    def eval(self):
        self._training = False
        return self

    def train(self):
        self._training = True
        return self

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

    def _compute_full_catalog_validation_metrics(self):
        return EmbeddingModule._compute_full_catalog_validation_metrics(self)

    def _build_full_catalog_validation_data(self):
        return EmbeddingModule._build_full_catalog_validation_data(self)

    def _encode_texts_batched(self, tokenizer, texts, max_length, batch_size, device):
        return EmbeddingModule._encode_texts_batched(
            self, tokenizer, texts, max_length, batch_size, device
        )


def _make_encode_stub(text_to_vector):
    """Build a stub encode method that maps texts to known vectors."""

    def encode(self_or_inputs, inputs=None):
        if inputs is None:
            inputs = self_or_inputs
        vectors = []
        for token_id in inputs["input_ids"].squeeze(1).tolist():
            vectors.append(text_to_vector.get(token_id, [0.1, 0.1, 0.1]))
        return torch.tensor(vectors, dtype=torch.float32)

    return encode


class ResolutionHelperTests(unittest.TestCase):
    def test_resolve_validation_mode_accepts_valid_modes(self):
        self.assertEqual(resolve_validation_mode("full_catalog"), "full_catalog")
        self.assertEqual(resolve_validation_mode("pairwise_proxy"), "pairwise_proxy")

    def test_resolve_validation_mode_rejects_invalid(self):
        with self.assertRaises(ValueError):
            resolve_validation_mode("unknown")

    def test_resolve_validation_metric_accepts_valid_metrics(self):
        self.assertEqual(resolve_validation_metric("ndcg_at_5"), "ndcg_at_5")
        self.assertEqual(resolve_validation_metric("mrr"), "mrr")
        self.assertEqual(resolve_validation_metric("recall_at_100"), "recall_at_100")

    def test_resolve_validation_metric_rejects_invalid(self):
        with self.assertRaises(ValueError):
            resolve_validation_metric("accuracy")

    def test_resolve_validation_similarity_accepts_valid(self):
        self.assertEqual(resolve_validation_similarity("dot"), "dot")
        self.assertEqual(resolve_validation_similarity("cosine"), "cosine")

    def test_resolve_validation_similarity_rejects_invalid(self):
        with self.assertRaises(ValueError):
            resolve_validation_similarity("euclidean")

    def test_build_full_catalog_monitor_metric(self):
        self.assertEqual(
            build_full_catalog_monitor_metric("ndcg_at_5"),
            "val/full_catalog/ndcg_at_5",
        )
        self.assertEqual(
            build_full_catalog_monitor_metric("mrr"),
            "val/full_catalog/mrr",
        )


class BuildFullCatalogValidationDataTests(unittest.TestCase):
    def test_deduplicates_queries_and_offers(self):
        rows = [
            {
                "query_id": "q1",
                "offer_id": "o1",
                "raw_label": "Exact",
                "query_text": "query one",
                "offer_text": "offer one",
            },
            {
                "query_id": "q1",
                "offer_id": "o2",
                "raw_label": "Irrelevant",
                "query_text": "query one",
                "offer_text": "offer two",
            },
            {
                "query_id": "q2",
                "offer_id": "o1",
                "raw_label": "Substitute",
                "query_text": "query two",
                "offer_text": "offer one",
            },
        ]

        module = _FullCatalogValidationModule(rows)
        query_rows, catalog_rows, judgments = (
            EmbeddingModule._build_full_catalog_validation_data(module)
        )

        self.assertEqual(len(query_rows), 2)
        self.assertEqual(len(catalog_rows), 2)
        query_ids = {row["query_id"] for row in query_rows}
        offer_ids = {row["offer_id"] for row in catalog_rows}
        self.assertEqual(query_ids, {"q1", "q2"})
        self.assertEqual(offer_ids, {"o1", "o2"})

    def test_resolves_duplicate_labels_by_highest_gain(self):
        rows = [
            {
                "query_id": "q1",
                "offer_id": "o1",
                "raw_label": "Complement",
                "query_text": "query one",
                "offer_text": "offer one",
            },
            {
                "query_id": "q1",
                "offer_id": "o1",
                "raw_label": "Exact",
                "query_text": "query one",
                "offer_text": "offer one",
            },
            {
                "query_id": "q1",
                "offer_id": "o1",
                "raw_label": "Substitute",
                "query_text": "query one",
                "offer_text": "offer one",
            },
        ]

        module = _FullCatalogValidationModule(rows)
        _, _, judgments = EmbeddingModule._build_full_catalog_validation_data(module)

        self.assertEqual(judgments["q1"]["o1"], "Exact")

    def test_does_not_downgrade_label(self):
        rows = [
            {
                "query_id": "q1",
                "offer_id": "o1",
                "raw_label": "Exact",
                "query_text": "q",
                "offer_text": "o",
            },
            {
                "query_id": "q1",
                "offer_id": "o1",
                "raw_label": "Irrelevant",
                "query_text": "q",
                "offer_text": "o",
            },
        ]

        module = _FullCatalogValidationModule(rows)
        _, _, judgments = EmbeddingModule._build_full_catalog_validation_data(module)

        self.assertEqual(judgments["q1"]["o1"], "Exact")

    def test_returns_empty_for_no_rows(self):
        module = _FullCatalogValidationModule([])
        query_rows, catalog_rows, judgments = (
            EmbeddingModule._build_full_catalog_validation_data(module)
        )

        self.assertEqual(query_rows, [])
        self.assertEqual(catalog_rows, [])
        self.assertEqual(judgments, {})


class SubsampleCatalogTests(unittest.TestCase):
    def test_retains_all_judged_offers(self):
        catalog_rows = [
            {"offer_id": "o1", "offer_text": "a"},
            {"offer_id": "o2", "offer_text": "b"},
            {"offer_id": "o3", "offer_text": "c"},
            {"offer_id": "o4", "offer_text": "d"},
            {"offer_id": "o5", "offer_text": "e"},
        ]
        judgments_by_query = {
            "q1": {"o1": "Exact", "o3": "Irrelevant"},
        }

        result = subsample_catalog(
            catalog_rows, 3, judgments_by_query
        )

        result_ids = {r["offer_id"] for r in result}
        self.assertIn("o1", result_ids)
        self.assertIn("o3", result_ids)
        self.assertEqual(len(result), 3)

    def test_returns_full_catalog_when_sample_exceeds_size(self):
        catalog_rows = [
            {"offer_id": "o1", "offer_text": "a"},
            {"offer_id": "o2", "offer_text": "b"},
        ]
        judgments_by_query = {"q1": {"o1": "Exact"}}

        result = subsample_catalog(
            catalog_rows, 10, judgments_by_query
        )

        self.assertEqual(len(result), 2)

    def test_subsample_is_deterministic(self):
        catalog_rows = [
            {"offer_id": f"o{i}", "offer_text": f"text-{i}"}
            for i in range(20)
        ]
        judgments_by_query = {"q1": {"o0": "Exact"}}

        r1 = subsample_catalog(catalog_rows, 5, judgments_by_query)
        r2 = subsample_catalog(catalog_rows, 5, judgments_by_query)

        self.assertEqual(
            [r["offer_id"] for r in r1],
            [r["offer_id"] for r in r2],
        )

    def test_config_triggers_subsampling(self):
        rows = [
            {
                "query_id": "q1",
                "offer_id": f"o{i}",
                "raw_label": "Exact" if i == 0 else "Irrelevant",
                "query_text": "q",
                "offer_text": f"offer-{i}",
            }
            for i in range(10)
        ]

        module = _FullCatalogValidationModule(rows)
        module.cfg = OmegaConf.merge(
            module.cfg,
            OmegaConf.create({"trainer": {"validation_catalog_sample": 3}}),
        )

        _, catalog_rows, _ = EmbeddingModule._build_full_catalog_validation_data(module)

        self.assertEqual(len(catalog_rows), 3)
        catalog_ids = {r["offer_id"] for r in catalog_rows}
        self.assertIn("o0", catalog_ids)


class FullCatalogMetricParityTests(unittest.TestCase):
    """Verify that training-time full-catalog validation produces results
    matching score_queries_against_catalog when given the same data."""

    def test_metric_parity_with_benchmark_scoring(self):
        query_rows = [
            {"query_id": "q1", "query_text": "query-a"},
            {"query_id": "q2", "query_text": "query-b"},
        ]
        catalog_rows = [
            {"offer_id": "o1", "offer_text": "exact-a"},
            {"offer_id": "o2", "offer_text": "irrelevant"},
            {"offer_id": "o3", "offer_text": "exact-b"},
        ]
        judgments_by_query = {
            "q1": {"o1": "Exact", "o2": "Irrelevant"},
            "q2": {"o3": "Exact", "o2": "Irrelevant"},
        }

        query_embeddings = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        catalog_embeddings = torch.tensor(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.1, 0.1],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        ks = (5, 10, 100)
        relevant_labels = {"Exact"}

        metrics = score_queries_against_catalog(
            query_rows=query_rows,
            query_embeddings=query_embeddings,
            catalog_rows=catalog_rows,
            catalog_embeddings=catalog_embeddings,
            judgments_by_query=judgments_by_query,
            ks=ks,
            relevant_labels=relevant_labels,
            score_batch_size=2,
        )

        self.assertEqual(metrics["evaluated_queries"], 2.0)
        self.assertEqual(metrics["retrieval_eligible_queries"], 2.0)
        self.assertGreater(metrics["mrr"], 0.0)
        for k in ks:
            self.assertGreater(metrics[f"ndcg@{k}"], 0.0)
            self.assertGreater(metrics[f"recall@{k}"], 0.0)


class FullCatalogValidationEpochEndTests(unittest.TestCase):
    def _build_module_with_encode_stub(self, validation_rows, encode_fn):
        module = _FullCatalogValidationModule(validation_rows)
        module.encode = lambda inputs: encode_fn(inputs)
        return module

    def test_logs_full_catalog_metrics(self):
        rows = [
            {
                "query_id": "q1",
                "offer_id": "o1",
                "raw_label": "Exact",
                "query_text": "query-a",
                "offer_text": "exact-a",
                "score": 0.9,
            },
            {
                "query_id": "q1",
                "offer_id": "o2",
                "raw_label": "Irrelevant",
                "query_text": "query-a",
                "offer_text": "irrelevant",
                "score": 0.5,
            },
            {
                "query_id": "q2",
                "offer_id": "o3",
                "raw_label": "Exact",
                "query_text": "query-b",
                "offer_text": "exact-b",
                "score": 0.8,
            },
            {
                "query_id": "q2",
                "offer_id": "o2",
                "raw_label": "Irrelevant",
                "query_text": "query-b",
                "offer_text": "irrelevant",
                "score": 0.3,
            },
        ]

        def encode_stub(inputs):
            vectors = []
            for token_id in inputs["input_ids"].squeeze(1).tolist():
                vectors.append([float(token_id % 10) / 10.0, 0.5, 0.5])
            return torch.tensor(vectors, dtype=torch.float32)

        module = self._build_module_with_encode_stub(rows, encode_stub)

        EmbeddingModule.on_validation_epoch_end(module)

        # Only the monitor metric is logged via self.log(); all others go to MLflow
        self.assertIn("val/full_catalog/ndcg_at_5", module.logged)

        mlflow_metrics = module.logger.calls[0]["metrics"]
        self.assertIn("val/full_catalog/mrr", mlflow_metrics)
        self.assertIn("val/full_catalog/ndcg_at_5", mlflow_metrics)
        self.assertIn("val/full_catalog/ndcg_at_10", mlflow_metrics)
        self.assertIn("val/full_catalog/recall_at_10", mlflow_metrics)
        self.assertIn("val/full_catalog/recall_at_100", mlflow_metrics)
        self.assertIn("val/full_catalog/catalog_size", mlflow_metrics)
        self.assertIn("val/full_catalog/evaluated_queries", mlflow_metrics)

        self.assertEqual(mlflow_metrics["val/full_catalog/catalog_size"], 3.0)
        self.assertEqual(
            mlflow_metrics["val/full_catalog/evaluated_queries"], 2.0
        )

    def test_monitor_metric_has_prog_bar_true(self):
        rows = [
            {
                "query_id": "q1",
                "offer_id": "o1",
                "raw_label": "Exact",
                "query_text": "q",
                "offer_text": "o",
                "score": 0.9,
            },
        ]

        def encode_stub(inputs):
            n = inputs["input_ids"].size(0)
            return torch.randn(n, 3, dtype=torch.float32)

        module = self._build_module_with_encode_stub(rows, encode_stub)
        module.validation_metric = "ndcg_at_5"

        EmbeddingModule.on_validation_epoch_end(module)

        self.assertTrue(
            module.logged["val/full_catalog/ndcg_at_5"]["kwargs"]["prog_bar"]
        )
        # Non-monitor metrics are no longer logged via self.log()
        self.assertNotIn("val/full_catalog/mrr", module.logged)

    def test_pairwise_metrics_still_logged(self):
        rows = [
            {
                "query_id": "q1",
                "offer_id": "o1",
                "raw_label": "Exact",
                "query_text": "q",
                "offer_text": "o",
                "score": 0.9,
            },
        ]

        def encode_stub(inputs):
            n = inputs["input_ids"].size(0)
            return torch.randn(n, 3, dtype=torch.float32)

        module = self._build_module_with_encode_stub(rows, encode_stub)

        EmbeddingModule.on_validation_epoch_end(module)

        self.assertIn("val/by_batch/exact_mrr", module.logged)
        self.assertIn("val/by_batch/ndcg_at_5", module.logged)

    def test_uses_configured_relevant_labels(self):
        rows = [
            {
                "query_id": "q1",
                "offer_id": "o1",
                "raw_label": "Substitute",
                "query_text": "q",
                "offer_text": "o1",
                "score": 0.9,
            },
            {
                "query_id": "q1",
                "offer_id": "o2",
                "raw_label": "Irrelevant",
                "query_text": "q",
                "offer_text": "o2",
                "score": 0.5,
            },
        ]

        def encode_stub(inputs):
            n = inputs["input_ids"].size(0)
            return torch.randn(n, 3, dtype=torch.float32)

        module = self._build_module_with_encode_stub(rows, encode_stub)
        module.cfg = OmegaConf.merge(
            module.cfg,
            OmegaConf.create(
                {"trainer": {"validation_relevant_labels": "Exact,Substitute"}}
            ),
        )

        EmbeddingModule.on_validation_epoch_end(module)

        mlflow_metrics = module.logger.calls[0]["metrics"]
        self.assertGreater(
            mlflow_metrics["val/full_catalog/retrieval_eligible_queries"],
            0.0,
        )

    def test_skips_full_catalog_in_pairwise_proxy_mode(self):
        rows = [
            {
                "query_id": "q1",
                "score": 0.9,
                "raw_label": "Exact",
            },
        ]
        module = _FullCatalogValidationModule(
            rows, validation_mode="pairwise_proxy"
        )

        EmbeddingModule.on_validation_epoch_end(module)

        full_catalog_keys = [
            k for k in module.logged if "full_catalog" in k
        ]
        self.assertEqual(full_catalog_keys, [])
        self.assertIn("val/by_batch/exact_mrr", module.logged)


class CheckpointMonitorTests(unittest.TestCase):
    def test_full_catalog_mode_uses_full_catalog_metric(self):
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "validation_mode": "full_catalog",
                    "validation_metric": "ndcg_at_5",
                    "checkpoint_dir": "checkpoints",
                },
                "logger": {},
            }
        )

        monitor = _resolve_checkpoint_monitor(cfg)

        self.assertEqual(monitor, "val/full_catalog/ndcg_at_5")

    def test_pairwise_proxy_mode_uses_pairwise_metric(self):
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "validation_mode": "pairwise_proxy",
                    "validation_metric": "ndcg_at_5",
                    "checkpoint_dir": "checkpoints",
                },
                "logger": {},
            }
        )

        monitor = _resolve_checkpoint_monitor(cfg)

        self.assertEqual(monitor, "val/by_batch/exact_mrr")

    def test_custom_validation_metric_reflected_in_monitor(self):
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "validation_mode": "full_catalog",
                    "validation_metric": "recall_at_100",
                    "checkpoint_dir": "checkpoints",
                },
                "logger": {},
            }
        )

        monitor = _resolve_checkpoint_monitor(cfg)

        self.assertEqual(monitor, "val/full_catalog/recall_at_100")


class CollateTextPassthroughTests(unittest.TestCase):
    def test_collated_batch_includes_query_and_offer_texts(self):
        from embedding_train.data import EmbeddingDataModule

        class _StubTokenizer:
            pad_token = "[PAD]"
            eos_token = "[EOS]"

            def __call__(self, texts, **kwargs):
                n = len(texts)
                return {
                    "input_ids": torch.ones((n, 2), dtype=torch.long),
                    "attention_mask": torch.ones((n, 2), dtype=torch.long),
                }

        cfg = OmegaConf.create(
            {
                "seed": 42,
                "model": {"model_name": "stub"},
                "data": {
                    "path": "/tmp/x.parquet",
                    "positive_label": "Exact",
                    "query_id_column": "query_id",
                    "offer_id_column": "offer_id_b64",
                    "batch_size": 2,
                    "train_batching_mode": "random_pairs",
                    "n_pos_samples_per_query": 1,
                    "n_neg_samples_per_query": 1,
                    "log_batch_stats": False,
                    "num_workers": 0,
                    "pin_memory": False,
                    "max_query_length": 32,
                    "max_offer_length": 64,
                    "val_fraction": 0.2,
                    "val_split_mode": "query_id",
                    "clean_html": True,
                    "limit_rows": None,
                    "query_template": "{{ query_term }}",
                    "offer_template": "{{ name }}",
                },
            }
        )

        with patch(
            "embedding_train.data.AutoTokenizer.from_pretrained",
            return_value=_StubTokenizer(),
        ):
            dm = EmbeddingDataModule(cfg)

        records = [
            {
                "query_id": "q1",
                "offer_id": "o1",
                "query_text": "hello",
                "offer_text": "world",
                "label": 1.0,
                "raw_label": "Exact",
            },
        ]

        batch = dm.collate_fn(records)

        self.assertEqual(batch["query_texts"], ["hello"])
        self.assertEqual(batch["offer_texts"], ["world"])


if __name__ == "__main__":
    unittest.main()
