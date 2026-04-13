import unittest

import torch

from embedding_train.losses import in_batch_contrastive_loss, in_batch_triplet_loss


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

        self.assertTrue(torch.isclose(loss, torch.tensor(0.0), atol=1e-6))


class InBatchTripletLossTests(unittest.TestCase):
    def test_prefers_same_query_negative_over_cross_query_fallback(self):
        query_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        offer_embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.95, 0.05]])
        labels = torch.tensor([1.0, 0.0, 0.0])
        query_ids = ["q1", "q1", "q2"]

        loss = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
        )

        self.assertTrue(torch.isclose(loss, torch.tensor(0.0), atol=1e-6))

    def test_falls_back_to_hardest_cross_query_negative(self):
        query_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
        offer_embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        labels = torch.tensor([1.0, 0.0, 0.0])
        query_ids = ["q1", "q2", "q3"]

        loss = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
        )

        self.assertTrue(torch.isclose(loss, torch.tensor(0.1), atol=1e-6))

    def test_excludes_same_query_exact_offers_from_negatives(self):
        query_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        offer_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([1.0, 1.0, 0.0])
        query_ids = ["q1", "q1", "q2"]

        loss = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
        )

        self.assertEqual(loss.item(), 0.0)

    def test_returns_zero_when_batch_has_no_valid_triplets(self):
        query_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        offer_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0.0, 0.0])
        query_ids = ["q1", "q2"]

        loss = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
        )

        self.assertEqual(loss.item(), 0.0)

    def test_semi_hard_prefers_hardest_negative_below_positive_similarity(self):
        query_embeddings = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        offer_embeddings = torch.tensor(
            [[0.8, 0.0], [1.5, 0.0], [0.5, 0.0], [0.3, 0.0]]
        )
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
        query_ids = ["q1", "q1", "q1", "q2"]

        loss = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
            negative_selection="semi_hard",
        )

        # hardest below-positive same-query negative has sim 0.5:
        # relu(0.5 - 0.8 + 0.2) = 0.0
        self.assertTrue(torch.isclose(loss, torch.tensor(0.0), atol=1e-6))

    def test_hardest_mode_selects_above_positive_negative(self):
        query_embeddings = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        offer_embeddings = torch.tensor(
            [[0.8, 0.0], [1.5, 0.0], [0.5, 0.0], [0.3, 0.0]]
        )
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
        query_ids = ["q1", "q1", "q1", "q2"]

        loss = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
            negative_selection="hardest",
        )

        # hardest same-query negative has sim 1.5:
        # relu(1.5 - 0.8 + 0.2) = 0.9
        self.assertTrue(torch.isclose(loss, torch.tensor(0.9), atol=1e-6))

    def test_semi_hard_falls_back_to_hardest_when_all_negatives_above_positive(self):
        query_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        offer_embeddings = torch.tensor([[0.8, 0.0], [1.5, 0.0]])
        labels = torch.tensor([1.0, 0.0])
        query_ids = ["q1", "q1"]

        loss = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
            negative_selection="semi_hard",
        )

        # no same-query negative sits below positive (0.8), so semi-hard
        # falls back to the hardest in the pool (1.5):
        # relu(1.5 - 0.8 + 0.2) = 0.9
        self.assertTrue(torch.isclose(loss, torch.tensor(0.9), atol=1e-6))

    def test_default_negative_selection_is_semi_hard(self):
        query_embeddings = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        offer_embeddings = torch.tensor(
            [[0.8, 0.0], [1.5, 0.0], [0.5, 0.0], [0.3, 0.0]]
        )
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
        query_ids = ["q1", "q1", "q1", "q2"]

        default_loss = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
        )
        semi_hard_loss = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
            negative_selection="semi_hard",
        )

        self.assertTrue(torch.isclose(default_loss, semi_hard_loss, atol=1e-6))

    def test_return_stats_reports_zero_fallback_when_semi_hard_negative_exists(self):
        query_embeddings = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        offer_embeddings = torch.tensor(
            [[0.8, 0.0], [1.5, 0.0], [0.5, 0.0], [0.3, 0.0]]
        )
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
        query_ids = ["q1", "q1", "q1", "q2"]

        loss, stats = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
            negative_selection="semi_hard",
            return_stats=True,
        )

        self.assertEqual(stats["valid_anchor_count"], 1)
        self.assertEqual(stats["semi_hard_fallback_count"], 0)
        self.assertEqual(stats["semi_hard_fallback_share"], 0.0)
        self.assertTrue(torch.isclose(loss, torch.tensor(0.0), atol=1e-6))

    def test_return_stats_counts_fallback_when_no_semi_hard_negative_exists(self):
        query_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        offer_embeddings = torch.tensor([[0.8, 0.0], [1.5, 0.0]])
        labels = torch.tensor([1.0, 0.0])
        query_ids = ["q1", "q1"]

        loss, stats = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
            negative_selection="semi_hard",
            return_stats=True,
        )

        self.assertEqual(stats["valid_anchor_count"], 1)
        self.assertEqual(stats["semi_hard_fallback_count"], 1)
        self.assertEqual(stats["semi_hard_fallback_share"], 1.0)
        self.assertTrue(torch.isclose(loss, torch.tensor(0.9), atol=1e-6))

    def test_return_stats_reports_zero_fallback_in_hardest_mode(self):
        query_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        offer_embeddings = torch.tensor([[0.8, 0.0], [1.5, 0.0]])
        labels = torch.tensor([1.0, 0.0])
        query_ids = ["q1", "q1"]

        _, stats = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
            negative_selection="hardest",
            return_stats=True,
        )

        # In hardest mode the "fallback" concept is meaningless, so the metric
        # is fixed to 0 to avoid polluting cross-mode dashboards.
        self.assertEqual(stats["valid_anchor_count"], 1)
        self.assertEqual(stats["semi_hard_fallback_count"], 0)
        self.assertEqual(stats["semi_hard_fallback_share"], 0.0)

    def test_return_stats_returns_zeros_when_no_valid_triplets(self):
        query_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        offer_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0.0, 0.0])
        query_ids = ["q1", "q2"]

        loss, stats = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
            return_stats=True,
        )

        self.assertEqual(stats["valid_anchor_count"], 0)
        self.assertEqual(stats["semi_hard_fallback_count"], 0)
        self.assertEqual(stats["semi_hard_fallback_share"], 0.0)
        self.assertEqual(loss.item(), 0.0)

    def test_return_stats_fallback_share_is_partial_when_some_anchors_lack_semi_hard(
        self,
    ):
        # Two positive anchors (rows 0 and 3), each with their own same-query pool:
        # - row 0 / q1: same-query negatives at sim 1.5 (above pos 0.8) and 0.5
        #   (below). Has a semi-hard negative.
        # - row 3 / q2: same-query negative at sim 1.4 (above pos 0.7).
        #   No same-query negative is below the positive. With no cross-query
        #   negative below either, this row falls back.
        query_embeddings = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        offer_embeddings = torch.tensor(
            [[0.8, 0.0], [1.5, 0.0], [0.5, 0.0], [0.7, 0.0], [1.4, 0.0]]
        )
        labels = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
        query_ids = ["q1", "q1", "q1", "q2", "q2"]

        _, stats = in_batch_triplet_loss(
            query_embeddings,
            offer_embeddings,
            query_ids,
            labels,
            margin=0.2,
            negative_selection="semi_hard",
            return_stats=True,
        )

        self.assertEqual(stats["valid_anchor_count"], 2)
        self.assertEqual(stats["semi_hard_fallback_count"], 1)
        self.assertAlmostEqual(stats["semi_hard_fallback_share"], 0.5)

    def test_rejects_unknown_negative_selection(self):
        query_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        offer_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([1.0, 0.0])
        query_ids = ["q1", "q2"]

        with self.assertRaisesRegex(ValueError, "Unsupported negative_selection"):
            in_batch_triplet_loss(
                query_embeddings,
                offer_embeddings,
                query_ids,
                labels,
                margin=0.2,
                negative_selection="super_hard",
            )


if __name__ == "__main__":
    unittest.main()
