"""Guard the warm-start path against new SpladeModule buffers.

train.py loads init_checkpoint with a STRICT load_state_dict. Any buffer or
parameter added to SpladeModule that lands in state_dict will raise here — and
because training SPLADE from a vanilla backbone degenerates, a broken warm start
does not fail loudly, it just produces a bad model after a full training run.

The regularizer work added a `df_ema` buffer, which is why this exists. It is
registered persistent=False specifically so this test passes.

Skips cleanly when the backbone checkpoint or the HF cache is unavailable.
"""

import os
import unittest

import torch
from omegaconf import OmegaConf

_CKPT_CANDIDATES = (
    "/home/max/simplesystem-embedding/data/v1a_best.ckpt",          # GPU box
    "/workspace/pipeline/out/box_backup/checkpoints/defiant-lynx-807/"
    "best-step=4880-val_full_catalog_ndcg_at_5=0.6702.ckpt",        # workspace
)
CKPT = next((p for p in _CKPT_CANDIDATES if os.path.exists(p)), _CKPT_CANDIDATES[0])
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--deepset--gbert-base")


def _cfg(**model_overrides):
    base = OmegaConf.load("configs/model/splade.yaml")
    base.merge_with(OmegaConf.create(model_overrides))
    return OmegaConf.create(
        {
            "model": base,
            "data": OmegaConf.load("configs/data/splade_sink.yaml"),
            "trainer": OmegaConf.load("configs/trainer/default.yaml"),
            "optimizer": OmegaConf.load("configs/optimizer/adamw.yaml"),
        }
    )


@unittest.skipUnless(
    os.path.exists(CKPT) and os.path.isdir(HF_CACHE),
    "needs the v1a_best backbone checkpoint and a cached deepset/gbert-base",
)
class WarmStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        cls.state_dict = torch.load(CKPT, map_location="cpu", weights_only=False)[
            "state_dict"
        ]

    def test_checkpoint_has_no_regularizer_buffers(self):
        self.assertNotIn("df_ema", self.state_dict)
        self.assertNotIn("special_token_vocab_mask", self.state_dict)

    def test_df_ema_absent_from_module_state_dict(self):
        from embedding_train.splade_model import SpladeModule

        module = SpladeModule(_cfg(reg_type_q="l1", reg_type_d="df_flops"))
        self.assertTrue(hasattr(module, "df_ema"))
        self.assertNotIn("df_ema", module.state_dict())

    def test_strict_warm_start_still_loads(self):
        from embedding_train.splade_model import SpladeModule
        from embedding_train.model import _align_encoder_compile_prefix

        for reg_q, reg_d in (("flops", "flops"), ("l1", "df_flops")):
            with self.subTest(reg_type_q=reg_q, reg_type_d=reg_d):
                module = SpladeModule(_cfg(reg_type_q=reg_q, reg_type_d=reg_d))
                aligned = _align_encoder_compile_prefix(self.state_dict, module)
                module.load_state_dict(aligned)  # strict — must not raise



@unittest.skipUnless(os.path.isdir(HF_CACHE), "needs a cached deepset/gbert-base")
class UntiedEncoderTests(unittest.TestCase):
    """Untied encoders had zero coverage and shipped a step-0 crash.

    `encode()` raises on an untied model when the caller omits is_query (so the
    nine unmigrated call sites fail loudly rather than encoding queries with the
    document encoder). `forward()` itself omitted it, so every untied arm died on
    the first batch — and surfaced only as a misleading "eval FAILED".
    """

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    def _batch(self, rows=2, length=4):
        return {
            "input_ids": torch.randint(1, 100, (rows, length)),
            "attention_mask": torch.ones(rows, length, dtype=torch.long),
        }

    def test_forward_runs_on_untied_model(self):
        from embedding_train.splade_model import SpladeModule

        module = SpladeModule(_cfg(untied_encoders=True)).eval()
        with torch.no_grad():
            q, d, scores = module(self._batch(), self._batch())
        self.assertEqual(q.shape[0], 2)
        self.assertEqual(scores.shape, (2,))

    def test_encode_without_is_query_raises_only_when_untied(self):
        from embedding_train.splade_model import SpladeModule

        untied = SpladeModule(_cfg(untied_encoders=True)).eval()
        with self.assertRaises(ValueError):
            untied.encode(self._batch())  # would silently use the doc encoder

        tied = SpladeModule(_cfg()).eval()
        with torch.no_grad():
            tied.encode(self._batch())  # unaffected

    def test_untied_plus_compile_is_rejected_before_loading_weights(self):
        from embedding_train.splade_model import SpladeModule

        with self.assertRaises(ValueError):
            SpladeModule(_cfg(untied_encoders=True, compile=True))

    def test_invalid_enums_rejected_at_construction(self):
        from embedding_train.splade_model import SpladeModule

        for override in ({"df_activation": "Paper"}, {"reg_type_q": "df-flops"},
                         {"df_half": 1.5}):
            with self.subTest(**override):
                with self.assertRaises(ValueError):
                    SpladeModule(_cfg(**override))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.path.isdir(HF_CACHE), "needs a cached deepset/gbert-base")
class FrozenDocEncoderTests(unittest.TestCase):
    """Freezing the doc encoder keeps the existing 113M-doc index valid.

    The failure mode is silent: if the doc encoder still receives gradient, or
    Lightning's per-epoch model.train() re-enables its dropout, the document
    vectors drift and the deployed index quietly stops matching the model.
    """

    def _batch(self, rows=2, length=4):
        return {
            "input_ids": torch.randint(1, 100, (rows, length)),
            "attention_mask": torch.ones(rows, length, dtype=torch.long),
        }

    def test_requires_untied(self):
        from embedding_train.splade_model import SpladeModule

        with self.assertRaises(ValueError):
            SpladeModule(_cfg(freeze_doc_encoder=True))  # untied not set

    def test_doc_encoder_frozen_query_encoder_trainable(self):
        from embedding_train.splade_model import SpladeModule

        module = SpladeModule(_cfg(untied_encoders=True, freeze_doc_encoder=True))
        self.assertFalse(any(p.requires_grad for p in module.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in module.query_encoder.parameters()))

    def test_doc_encoder_stays_eval_after_train_call(self):
        from embedding_train.splade_model import SpladeModule

        module = SpladeModule(_cfg(untied_encoders=True, freeze_doc_encoder=True))
        module.train()  # Lightning does this at the start of every epoch
        self.assertFalse(module.encoder.training)
        self.assertTrue(module.query_encoder.training)

    def test_doc_vectors_are_deterministic_under_train_mode(self):
        from embedding_train.splade_model import SpladeModule

        module = SpladeModule(_cfg(untied_encoders=True, freeze_doc_encoder=True))
        module.train()
        batch = self._batch()
        with torch.no_grad():
            first = module.encode(batch, is_query=False)
            second = module.encode(batch, is_query=False)
        # identical only if dropout is genuinely off in the frozen encoder
        self.assertTrue(torch.equal(first, second))


@unittest.skipUnless(os.path.isdir(HF_CACHE), "needs a cached deepset/gbert-base")
class FrozenDocEncoderGraphTests(unittest.TestCase):
    """The frozen doc side must build NO autograd graph.

    It first OOM'd an 80GB H100: with no grad-requiring parameter,
    torch.utils.checkpoint degenerates to a passthrough that retains the
    [batch, seq, vocab] activations instead of recomputing them, so freezing
    cost MORE memory than not freezing.
    """

    def _batch(self, rows=2, length=4):
        return {
            "input_ids": torch.randint(1, 100, (rows, length)),
            "attention_mask": torch.ones(rows, length, dtype=torch.long),
        }

    def test_frozen_doc_output_has_no_grad_fn_but_query_does(self):
        from embedding_train.splade_model import SpladeModule

        module = SpladeModule(_cfg(untied_encoders=True, freeze_doc_encoder=True))
        module.train()
        doc = module.encode(self._batch(), is_query=False)
        query = module.encode(self._batch(), is_query=True)
        self.assertIsNone(doc.grad_fn, "frozen doc side must not build a graph")
        self.assertIsNotNone(query.grad_fn, "query side must remain trainable")

    def test_gradients_still_reach_the_query_encoder(self):
        from embedding_train.splade_model import SpladeModule

        module = SpladeModule(_cfg(untied_encoders=True, freeze_doc_encoder=True))
        module.train()
        q, d, scores = module(self._batch(), self._batch())
        scores.sum().backward()
        query_grads = [p.grad for p in module.query_encoder.parameters() if p.grad is not None]
        self.assertTrue(query_grads, "query encoder received no gradient")
        self.assertTrue(all(p.grad is None for p in module.encoder.parameters()),
                        "frozen doc encoder must receive no gradient")
