import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from embedding_train.losses import (
    flops_regularizer,
    margin_mse_loss,
    quadratic_warmup,
)
from embedding_train.model import (
    load_embedding_module_from_checkpoint,
    resolve_architecture,
)
from embedding_train.splade_model import SpladeModule


VOCAB_SIZE = 32
SPECIAL_TOKEN_IDS = [0, 1, 2]


class _MlmOutput:
    def __init__(self, logits):
        self.logits = logits


class _MlmEncoderStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB_SIZE, 4)
        self.head = torch.nn.Linear(4, VOCAB_SIZE)
        self.config = type(
            "Config", (), {"hidden_size": 4, "vocab_size": VOCAB_SIZE}
        )()

    def forward(self, input_ids=None, attention_mask=None):
        del attention_mask
        return _MlmOutput(self.head(self.embedding(input_ids)))


def _tokenizer_stub(*args, **kwargs):
    del args, kwargs
    return SimpleNamespace(all_special_ids=list(SPECIAL_TOKEN_IDS))


def build_splade_cfg(**model_overrides):
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "model": {
                "model_name": "stub-mlm",
                "architecture": "splade",
                "output_dim": None,
                "pooling": "splade_max",
                "loss_type": "contrastive",
                "similarity_scale": 1.0,
                "triplet_margin": 0.2,
                "gradient_checkpointing": False,
                "teacher_checkpoint": None,
                "distill_temperature": 0.05,
                "compile": False,
                "flops_lambda_q": 0.0,
                "flops_lambda_d": 0.0,
                "flops_warmup_steps": 0,
                "loss_weights": {"margin_mse": 0.0, "kl": 0.0},
            },
            "trainer": {"precision": "32-true"},
            "data": {"log_batch_stats": False},
            "optimizer": {
                "lr": 2.0e-5,
                "weight_decay": 0.01,
                "scheduler": "linear",
                "warmup_ratio": 0.1,
                "warmup_steps": None,
            },
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create({"model": model_overrides}))


def build_splade_module(**model_overrides):
    with patch(
        "embedding_train.splade_model.AutoModelForMaskedLM.from_pretrained",
        side_effect=lambda *args, **kwargs: _MlmEncoderStub(),
    ), patch(
        "embedding_train.splade_model.load_fast_tokenizer",
        side_effect=_tokenizer_stub,
    ):
        module = SpladeModule(build_splade_cfg(**model_overrides))
    module.log = lambda *args, **kwargs: None
    return module


class ResolveArchitectureTests(unittest.TestCase):
    def test_defaults_to_dense_when_missing(self):
        self.assertEqual(resolve_architecture(None), "dense")

    def test_accepts_dense_and_splade(self):
        self.assertEqual(resolve_architecture("dense"), "dense")
        self.assertEqual(resolve_architecture("splade"), "splade")

    def test_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            resolve_architecture("colbert")


class SpladeEncodeTests(unittest.TestCase):
    def test_encode_returns_vocab_sized_nonnegative_representations(self):
        module = build_splade_module()
        inputs = {
            "input_ids": torch.tensor([[3, 4, 5], [6, 7, 8]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        }

        representations = module.encode(inputs)

        self.assertEqual(representations.shape, (2, VOCAB_SIZE))
        self.assertTrue((representations >= 0).all())

    def test_padding_positions_do_not_contribute(self):
        module = build_splade_module()
        short_inputs = {
            "input_ids": torch.tensor([[3, 4]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        padded_inputs = {
            "input_ids": torch.tensor([[3, 4, 9, 10]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        }

        short_reps = module.encode(short_inputs)
        padded_reps = module.encode(padded_inputs)

        torch.testing.assert_close(short_reps, padded_reps)

    def test_special_token_dimensions_are_zeroed(self):
        module = build_splade_module()
        inputs = {
            "input_ids": torch.tensor([[3, 4, 5]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }

        representations = module.encode(inputs)

        self.assertTrue(
            (representations[:, SPECIAL_TOKEN_IDS] == 0).all()
        )

    def test_rejects_projection_head(self):
        with self.assertRaises(ValueError):
            build_splade_module(output_dim=128)


class FlopsRegularizerTests(unittest.TestCase):
    def test_matches_hand_computed_value(self):
        representations = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
        # mean |activation| per dim: [2.0, 0.0]; sum of squares: 4.0
        self.assertAlmostEqual(
            flops_regularizer(representations).item(), 4.0, places=6
        )

    def test_quadratic_warmup_endpoints(self):
        self.assertEqual(quadratic_warmup(0, 100, 0.5), 0.0)
        self.assertAlmostEqual(quadratic_warmup(50, 100, 0.5), 0.125)
        self.assertEqual(quadratic_warmup(100, 100, 0.5), 0.5)
        self.assertEqual(quadratic_warmup(250, 100, 0.5), 0.5)
        self.assertEqual(quadratic_warmup(7, 0, 0.5), 0.5)


class MarginMseLossTests(unittest.TestCase):
    def test_matches_hand_computed_value(self):
        scores = torch.tensor([5.0, 2.0, 9.0])
        labels = torch.tensor([1.0, 0.0, 1.0])
        query_ids = ["q1", "q1", "q2"]
        teacher_scores = torch.tensor([4.0, 3.0, 1.0])

        # Only q1 has a (pos, neg) pair: student margin 3, teacher margin 1.
        loss = margin_mse_loss(scores, query_ids, labels, teacher_scores)

        self.assertAlmostEqual(loss.item(), 4.0, places=6)

    def test_ignores_rows_without_teacher_scores(self):
        scores = torch.tensor([5.0, 2.0, 4.0])
        labels = torch.tensor([1.0, 0.0, 0.0])
        query_ids = ["q1", "q1", "q1"]
        teacher_scores = torch.tensor([4.0, float("nan"), 3.0])

        # The NaN-teacher negative is dropped: margin pair is rows (0, 2).
        loss = margin_mse_loss(scores, query_ids, labels, teacher_scores)

        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_returns_zero_without_pairs(self):
        scores = torch.tensor([5.0, 2.0])
        labels = torch.tensor([1.0, 1.0])
        query_ids = ["q1", "q2"]
        teacher_scores = torch.tensor([4.0, 3.0])

        loss = margin_mse_loss(scores, query_ids, labels, teacher_scores)

        self.assertEqual(loss.item(), 0.0)


class SpladeComputeLossTests(unittest.TestCase):
    def _build_batch(self, module):
        inputs = {
            "input_ids": torch.tensor([[3, 4], [5, 6], [7, 8], [9, 10]]),
            "attention_mask": torch.ones(4, 2, dtype=torch.long),
        }
        query_embeddings = module.encode(inputs)
        offer_embeddings = module.encode(inputs)
        scores = (query_embeddings * offer_embeddings).sum(dim=1)
        batch = {
            "query_inputs": inputs,
            "offer_inputs": inputs,
            "query_ids": ["q1", "q1", "q2", "q2"],
            "labels": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        }
        return batch, query_embeddings, offer_embeddings, scores

    def test_flops_terms_are_added_to_ranking_loss(self):
        module = build_splade_module()
        batch, query_embeddings, offer_embeddings, scores = self._build_batch(
            module
        )

        plain_loss = module.compute_loss(
            batch, query_embeddings, offer_embeddings, scores
        )

        module.flops_lambda_q = 0.1
        module.flops_lambda_d = 0.1
        regularized_loss = module.compute_loss(
            batch, query_embeddings, offer_embeddings, scores
        )

        unique_queries = query_embeddings[[0, 2]]
        expected = (
            plain_loss
            + 0.1 * flops_regularizer(unique_queries)
            + 0.1 * flops_regularizer(offer_embeddings)
        )
        torch.testing.assert_close(regularized_loss, expected)

    def test_margin_mse_requires_ce_scores(self):
        module = build_splade_module()
        module.margin_mse_weight = 0.05
        batch, query_embeddings, offer_embeddings, scores = self._build_batch(
            module
        )

        with self.assertRaises(RuntimeError):
            module.compute_loss(
                batch, query_embeddings, offer_embeddings, scores
            )


class SpladeCheckpointRoundTripTests(unittest.TestCase):
    def test_loader_dispatches_to_splade_module(self):
        module = build_splade_module()
        checkpoint = {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
        }

        with TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "splade.ckpt"
            torch.save(checkpoint, checkpoint_path)

            with patch(
                "embedding_train.splade_model.AutoModelForMaskedLM.from_pretrained",
                side_effect=lambda *args, **kwargs: _MlmEncoderStub(),
            ), patch(
                "embedding_train.splade_model.load_fast_tokenizer",
                side_effect=_tokenizer_stub,
            ):
                loaded, cfg = load_embedding_module_from_checkpoint(
                    checkpoint_path
                )

        self.assertIsInstance(loaded, SpladeModule)
        self.assertEqual(cfg.model.architecture, "splade")

        inputs = {
            "input_ids": torch.tensor([[3, 4, 5]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        with torch.no_grad():
            torch.testing.assert_close(
                loaded.encode(inputs), module.encode(inputs)
            )


class WarmStartTests(unittest.TestCase):
    def test_build_module_warm_starts_from_init_checkpoint(self):
        from embedding_train.train import build_module

        source = build_splade_module()
        checkpoint = {
            "state_dict": source.state_dict(),
            "hyper_parameters": dict(source.hparams),
        }

        with TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "warm.ckpt"
            torch.save(checkpoint, checkpoint_path)

            cfg = build_splade_cfg(init_checkpoint=str(checkpoint_path))
            with patch(
                "embedding_train.splade_model.AutoModelForMaskedLM.from_pretrained",
                side_effect=lambda *args, **kwargs: _MlmEncoderStub(),
            ), patch(
                "embedding_train.splade_model.load_fast_tokenizer",
                side_effect=_tokenizer_stub,
            ):
                warmed = build_module(cfg)

        inputs = {
            "input_ids": torch.tensor([[3, 4, 5]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        with torch.no_grad():
            torch.testing.assert_close(
                warmed.encode(inputs), source.encode(inputs)
            )


if __name__ == "__main__":
    unittest.main()
