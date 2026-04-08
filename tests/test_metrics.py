import math
import unittest

from embedding_train.metrics import (
    compute_binary_retrieval_metrics,
    compute_exact_retrieval_metrics,
    compute_precision_metrics,
    compute_ranking_metrics,
)


class ComputeRankingMetricsTests(unittest.TestCase):
    def test_computes_ndcg_per_query_and_excludes_irrelevant_only_queries(self):
        rows = [
            {"query_id": "q1", "score": 0.9, "raw_label": "Substitute"},
            {"query_id": "q1", "score": 0.8, "raw_label": "Exact"},
            {"query_id": "q1", "score": 0.7, "raw_label": "Irrelevant"},
            {"query_id": "q2", "score": 0.6, "raw_label": "Exact"},
            {"query_id": "q2", "score": 0.5, "raw_label": "Irrelevant"},
            {"query_id": "q3", "score": 0.4, "raw_label": "Irrelevant"},
        ]

        metrics = compute_ranking_metrics(rows)

        self.assertEqual(metrics["evaluated_queries"], 3.0)
        self.assertEqual(metrics["eligible_queries"], 2.0)
        self.assertTrue(math.isclose(metrics["ndcg@1"], 0.55))
        self.assertTrue(math.isclose(metrics["ndcg@5"], 0.8437750838894885))
        self.assertTrue(math.isclose(metrics["ndcg@10"], 0.8437750838894885))

    def test_raises_for_unknown_labels(self):
        with self.assertRaisesRegex(ValueError, "Unknown relevance label"):
            compute_ranking_metrics(
                [{"query_id": "q1", "score": 1.0, "raw_label": "Unknown"}]
            )

    def test_rejects_synthetic_negative_labels_if_they_reach_metrics(self):
        with self.assertRaisesRegex(ValueError, "Unknown relevance label"):
            compute_ranking_metrics(
                [
                    {
                        "query_id": "q1",
                        "score": 1.0,
                        "raw_label": "SyntheticNegative",
                    }
                ]
            )


class ComputeExactRetrievalMetricsTests(unittest.TestCase):
    def test_computes_exact_retrieval_metrics_from_ranked_rows(self):
        rows = [
            {"query_id": "q1", "rank": 1, "raw_label": "Irrelevant"},
            {"query_id": "q1", "rank": 2, "raw_label": "Exact"},
            {"query_id": "q1", "rank": 3, "raw_label": "Substitute"},
            {"query_id": "q2", "rank": 1, "raw_label": "Exact"},
            {"query_id": "q2", "rank": 2, "raw_label": "Irrelevant"},
            {"query_id": "q3", "rank": 1, "raw_label": "Substitute"},
        ]

        metrics = compute_exact_retrieval_metrics(
            rows,
            evaluated_query_ids=["q1", "q2", "q3"],
            eligible_query_ids=["q1", "q2"],
        )

        self.assertEqual(metrics["evaluated_queries"], 3.0)
        self.assertEqual(metrics["eligible_queries"], 2.0)
        self.assertTrue(math.isclose(metrics["exact_success@1"], 0.5))
        self.assertTrue(math.isclose(metrics["exact_recall@5"], 1.0))
        self.assertTrue(math.isclose(metrics["exact_recall@10"], 1.0))
        self.assertTrue(math.isclose(metrics["exact_mrr"], 0.75))

    def test_uses_score_order_when_rank_is_missing(self):
        rows = [
            {"query_id": "q1", "score": 0.9, "raw_label": "Substitute"},
            {"query_id": "q1", "score": 0.8, "raw_label": "Exact"},
            {"query_id": "q1", "score": 0.1, "raw_label": "Irrelevant"},
        ]

        metrics = compute_exact_retrieval_metrics(rows)

        self.assertEqual(metrics["evaluated_queries"], 1.0)
        self.assertEqual(metrics["eligible_queries"], 1.0)
        self.assertTrue(math.isclose(metrics["exact_success@1"], 0.0))
        self.assertTrue(math.isclose(metrics["exact_recall@5"], 1.0))
        self.assertTrue(math.isclose(metrics["exact_mrr"], 0.5))

    def test_reports_zero_when_queries_have_no_exact_offer(self):
        rows = [
            {"query_id": "q1", "rank": 1, "raw_label": "Substitute"},
            {"query_id": "q1", "rank": 2, "raw_label": "Irrelevant"},
        ]

        metrics = compute_exact_retrieval_metrics(
            rows,
            evaluated_query_ids=["q1"],
            eligible_query_ids=[],
        )

        self.assertEqual(metrics["evaluated_queries"], 1.0)
        self.assertEqual(metrics["eligible_queries"], 0.0)
        self.assertEqual(metrics["exact_success@1"], 0.0)
        self.assertEqual(metrics["exact_recall@5"], 0.0)
        self.assertEqual(metrics["exact_recall@10"], 0.0)
        self.assertEqual(metrics["exact_mrr"], 0.0)


class ComputeBinaryMetricHelpersTests(unittest.TestCase):
    def test_computes_precision_at_k_for_exact_relevance(self):
        rows = [
            {"query_id": "q1", "rank": 1, "raw_label": "Substitute"},
            {"query_id": "q1", "rank": 2, "raw_label": "Exact"},
            {"query_id": "q2", "rank": 1, "raw_label": "Exact"},
            {"query_id": "q2", "rank": 2, "raw_label": "Irrelevant"},
            {"query_id": "q3", "rank": 1, "raw_label": "Substitute"},
        ]

        metrics = compute_precision_metrics(
            rows,
            ks=(1, 2),
            evaluated_query_ids=["q1", "q2", "q3"],
            eligible_query_ids=["q1", "q2"],
        )

        self.assertEqual(metrics["evaluated_queries"], 3.0)
        self.assertEqual(metrics["eligible_queries"], 2.0)
        self.assertTrue(math.isclose(metrics["precision@1"], 0.5))
        self.assertTrue(math.isclose(metrics["precision@2"], 0.5))

    def test_supports_custom_relevant_labels_for_binary_retrieval(self):
        rows = [
            {"query_id": "q1", "rank": 1, "raw_label": "Substitute"},
            {"query_id": "q1", "rank": 2, "raw_label": "Exact"},
            {"query_id": "q2", "rank": 1, "raw_label": "Complement"},
            {"query_id": "q2", "rank": 2, "raw_label": "Irrelevant"},
        ]

        metrics = compute_binary_retrieval_metrics(
            rows,
            ks=(1, 2),
            relevant_labels=("Exact", "Substitute"),
        )

        self.assertEqual(metrics["evaluated_queries"], 2.0)
        self.assertEqual(metrics["eligible_queries"], 1.0)
        self.assertTrue(math.isclose(metrics["recall@1"], 1.0))
        self.assertTrue(math.isclose(metrics["recall@2"], 1.0))
        self.assertTrue(math.isclose(metrics["mrr"], 1.0))


if __name__ == "__main__":
    unittest.main()
