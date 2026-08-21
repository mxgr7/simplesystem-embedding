"""The serving vocab mask must equal the training vocab mask, exactly.

Why this test exists: `soup2b50` was trained with `model.fold_vocab_mask=True`,
which zeroes every cased/diacritic output dimension. Dimensions zeroed in training
never receive a gradient, so the L1/FLOPS regulariser never prices them and their
activations are arbitrary. Serving that checkpoint through a backend that masked
only the special ids took its query vector from **5.1 to ~4,530 non-zeros** — 4,530
posting lists per query instead of 5, i.e. the exact opposite of the property the
model was selected for. Nothing failed; the service answered every request.

So the mask is part of the trained model's contract. These tests pin the two
things that make that safe:

1. `splade-service/backend.py:build_vocab_mask` produces bit-identical output to
   `embedding_train.splade_model.SpladeModule`'s `special_token_vocab_mask`, over
   every combination of the train-time mask options.
2. `is_cased_token` agrees between the two implementations token for token.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from embedding_train.splade_model import SpladeModule, is_cased_token as train_is_cased

from conftest import load_flat_service

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
try:
    from tests.test_splade import (  # reuse the stub encoder/tokenizer wiring
        SPECIAL_TOKEN_IDS,
        VOCAB_SIZE,
        _MlmEncoderStub,
        build_splade_cfg,
    )
finally:
    sys.path.remove(str(REPO))
splade = load_flat_service(
    "splade_service", REPO / "splade-service", "backend"
)


VOCAB = {
    "[PAD]": 0, "[CLS]": 1, "[SEP]": 2,
    "haus": 3, "Haus": 4, "häuser": 5, "haeuser": 6,
    "##ung": 7, "##Ung": 8, "strasse": 9, "straße": 10,
    "und": 11, "der": 12, "10": 13, "x": 14, "École": 15,
}


class VocabMaskParityTest(unittest.TestCase):
    def setUp(self):
        self.backend = splade.backend
        self.tokenizer = SimpleNamespace(
            all_special_ids=list(SPECIAL_TOKEN_IDS),
            get_vocab=lambda: dict(VOCAB),
        )

    def _training_mask(self, **overrides):
        """Build the real SpladeModule against the same stub tokenizer the serving
        side is given, and read the buffer it registers."""
        with patch(
            "embedding_train.splade_model.AutoModelForMaskedLM.from_pretrained",
            side_effect=lambda *args, **kwargs: _MlmEncoderStub(),
        ), patch(
            "embedding_train.splade_model.load_fast_tokenizer",
            side_effect=lambda *args, **kwargs: SimpleNamespace(
                all_special_ids=list(SPECIAL_TOKEN_IDS),
                get_vocab=lambda: dict(VOCAB),
            ),
        ):
            module = SpladeModule(build_splade_cfg(**overrides))
        return module.special_token_vocab_mask.detach().clone()

    def _serving_mask(self, **model_hp):
        mask, detail = self.backend.build_vocab_mask(
            self.tokenizer, VOCAB_SIZE, model_hp
        )
        return mask, detail

    def test_cased_predicate_matches_training(self):
        for token in VOCAB:
            self.assertEqual(
                self.backend.is_cased_token(token), train_is_cased(token),
                f"cased predicate disagrees on {token!r}",
            )

    def test_fold_mask_off_matches_training(self):
        train = self._training_mask(fold_vocab_mask=False)
        serve, detail = self._serving_mask(fold_vocab_mask=False)
        self.assertTrue(torch.equal(serve.float(), train.float()))
        self.assertEqual(detail["cased_masked"], 0)
        self.assertFalse(detail["fold_vocab_mask"])

    def test_fold_mask_on_matches_training(self):
        train = self._training_mask(fold_vocab_mask=True)
        serve, detail = self._serving_mask(fold_vocab_mask=True)
        self.assertTrue(torch.equal(serve.float(), train.float()))
        # Haus, häuser, ##Ung, École -- plus [PAD]/[CLS]/[SEP], which the predicate
        # also calls cased because they contain capitals. Harmless (they are masked
        # as special ids anyway), but it means cased_masked is not a count of
        # *content* tokens.
        self.assertEqual(detail["cased_masked"], 7)
        self.assertTrue(detail["fold_vocab_mask"])

    def test_eszett_is_not_treated_as_cased(self):
        """`ß` survives the fold mask even though fold_de rewrites it to `ss`.

        Not a bug to fix here: `is_cased_token` is uppercase-or-combining-diacritic,
        and NFD leaves `ß` a single non-combining codepoint. Pinned because the
        serving mask must mirror training exactly — "improving" the predicate on one
        side alone would serve a vector the model was never trained to produce.
        """
        serve, _ = self._serving_mask(fold_vocab_mask=True)
        self.assertTrue(bool(serve[VOCAB["straße"]]), "straße should stay unmasked")
        self.assertFalse(bool(serve[VOCAB["häuser"]]), "häuser should be masked")

    def test_stopword_mask_matches_training(self):
        stopwords = [VOCAB["und"], VOCAB["der"]]
        train = self._training_mask(stopword_mask_ids=stopwords)
        serve, detail = self._serving_mask(stopword_mask_ids=stopwords)
        self.assertTrue(torch.equal(serve.float(), train.float()))
        self.assertEqual(detail["stopword_masked"], 2)

    def test_both_masks_compose_like_training(self):
        stopwords = [VOCAB["und"]]
        train = self._training_mask(fold_vocab_mask=True,
                                    stopword_mask_ids=stopwords)
        serve, _ = self._serving_mask(fold_vocab_mask=True,
                                      stopword_mask_ids=stopwords)
        self.assertTrue(torch.equal(serve.float(), train.float()))

    def test_mask_digest_distinguishes_configurations(self):
        """The digest is what the reindex client pins, so different masks must not
        collide -- two booleans in a contract could not tell these apart."""
        digests = {
            self._serving_mask(fold_vocab_mask=False)[1]["vocab_mask_sha256"],
            self._serving_mask(fold_vocab_mask=True)[1]["vocab_mask_sha256"],
            self._serving_mask(stopword_mask_ids=[VOCAB["und"]])[1]["vocab_mask_sha256"],
        }
        self.assertEqual(len(digests), 3)

    def test_out_of_range_ids_are_ignored(self):
        """A stoplist referring to a bigger vocabulary must not index past the end
        -- the training code guards this the same way."""
        _, detail = self._serving_mask(stopword_mask_ids=[VOCAB["und"], VOCAB_SIZE + 5])
        self.assertEqual(detail["stopword_masked"], 1)


if __name__ == "__main__":
    unittest.main()
