import unittest

import torch

from embedding_train.losses import in_batch_contrastive_loss


class InBatchContrastiveLossTests(unittest.TestCase):
    def test_lower_loss_when_positive_offers_match_queries(self):
        query_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        matching_offer_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        swapped_offer_embeddings = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        labels = torch.tensor([1.0, 1.0])
        query_ids = ["q1", "q2"]

        matching_loss = in_batch_contrastive_loss(
            query_embeddings,
            matching_offer_embeddings,
            query_ids,
            labels,
            scale=20.0,
        )
        swapped_loss = in_batch_contrastive_loss(
            query_embeddings,
            swapped_offer_embeddings,
            query_ids,
            labels,
            scale=20.0,
        )

        self.assertLess(matching_loss.item(), swapped_loss.item())

    def test_multiple_positive_offers_reduce_loss(self):
        query_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        offer_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        query_ids = ["q1", "q1", "q2"]
        single_positive_labels = torch.tensor([1.0, 0.0, 1.0])
        multiple_positive_labels = torch.tensor([1.0, 1.0, 1.0])

        single_positive_loss = in_batch_contrastive_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            single_positive_labels,
            scale=20.0,
        )
        multiple_positive_loss = in_batch_contrastive_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            multiple_positive_labels,
            scale=20.0,
        )

        self.assertLess(multiple_positive_loss.item(), single_positive_loss.item())

    def test_returns_zero_when_batch_has_no_positive_offers(self):
        query_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        offer_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0.0, 0.0])
        query_ids = ["q1", "q2"]

        loss = in_batch_contrastive_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            scale=20.0,
        )

        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
