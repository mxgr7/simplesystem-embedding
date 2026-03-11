import unittest

import torch

from embedding_train.precision import (
    pack_binary_embeddings,
    quantize_embeddings,
    resolve_embedding_precision,
    score_embedding_pairs,
    serialize_embeddings,
)


class EmbeddingPrecisionTests(unittest.TestCase):
    def test_resolve_embedding_precision_rejects_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "Unsupported embedding precision"):
            resolve_embedding_precision("fp8")

    def test_quantizes_to_int8(self):
        embeddings = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])

        quantized = quantize_embeddings(embeddings, "int8")

        self.assertEqual(quantized.tolist(), [[-127, -64, 0, 64, 127]])

    def test_quantizes_to_sign_values(self):
        embeddings = torch.tensor([[-0.1, 0.0, 0.2]])

        quantized = quantize_embeddings(embeddings, "sign")

        self.assertEqual(quantized.tolist(), [[-1, 1, 1]])

    def test_packs_binary_embeddings_into_bytes(self):
        embeddings = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0]])

        packed = pack_binary_embeddings(embeddings)

        self.assertEqual(packed.tolist(), [[168]])
        self.assertEqual(serialize_embeddings(packed, "binary"), [b"\xa8"])

    def test_scores_binary_embeddings_with_packed_bits(self):
        query_embeddings = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        same_offer_embeddings = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        opposite_offer_embeddings = torch.tensor([[-1.0, 1.0, -1.0, 1.0]])

        same_scores = score_embedding_pairs(
            query_embeddings,
            same_offer_embeddings,
            "binary",
        )
        opposite_scores = score_embedding_pairs(
            query_embeddings,
            opposite_offer_embeddings,
            "binary",
        )

        self.assertEqual(same_scores.tolist(), [1.0])
        self.assertEqual(opposite_scores.tolist(), [-1.0])

    def test_scores_sign_embeddings(self):
        query_embeddings = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
        offer_embeddings = torch.tensor([[4.0, -3.0, 2.0, -1.0]])

        scores = score_embedding_pairs(query_embeddings, offer_embeddings, "sign")

        self.assertEqual(scores.tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
