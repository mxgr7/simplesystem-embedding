import math
import unittest

from embedding_train.metrics import compute_ranking_metrics


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


if __name__ == "__main__":
    unittest.main()
