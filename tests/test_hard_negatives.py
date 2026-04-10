import unittest

from embedding_train.batching import (
    AnchorQueryBatchBuilder,
    HARD_NEGATIVE_LABEL,
    SYNTHETIC_NEGATIVE_LABEL,
    RandomQueryPoolBuilder,
    build_batch_stats,
    build_hard_negative_record,
)
from embedding_train.hard_negatives import mine_hard_negatives_from_results


def make_record(query_id, offer_id, label, query_text, offer_text):
    return {
        "query_id": query_id,
        "offer_id": offer_id,
        "query_text": query_text,
        "offer_text": offer_text,
        "label": label,
        "raw_label": "Exact" if label > 0.5 else "Irrelevant",
    }


def build_inputs_with_hard_negatives():
    positive_records_by_query = {
        "q1": [
            make_record("q1", "q1-p1", 1.0, "anchor query", "anchor positive 1"),
            make_record("q1", "q1-p2", 1.0, "anchor query", "anchor positive 2"),
            make_record("q1", "q1-p3", 1.0, "anchor query", "anchor positive 3"),
        ],
        "q2": [
            make_record("q2", "q2-p1", 1.0, "foreign query 2", "foreign positive 1"),
            make_record("q2", "q2-p2", 1.0, "foreign query 2", "foreign positive 2"),
        ],
        "q3": [
            make_record("q3", "q3-p1", 1.0, "foreign query 3", "foreign positive 3"),
            make_record("q3", "q3-p2", 1.0, "foreign query 3", "foreign positive 4"),
        ],
    }
    negative_records_by_query = {
        "q1": [
            make_record("q1", "q1-n1", 0.0, "anchor query", "anchor negative 1"),
        ],
        "q2": [
            make_record("q2", "q2-n1", 0.0, "foreign query 2", "foreign negative 1"),
            make_record("q2", "q2-n2", 0.0, "foreign query 2", "foreign negative 2"),
        ],
        "q3": [
            make_record("q3", "q3-n1", 0.0, "foreign query 3", "foreign negative 3"),
        ],
    }
    hard_negative_records_by_query = {
        "q1": [
            {"offer_id": "hn-1", "offer_text": "hard negative 1"},
            {"offer_id": "hn-2", "offer_text": "hard negative 2"},
            {"offer_id": "hn-3", "offer_text": "hard negative 3"},
        ],
    }
    synthetic_negative_offer_pool = [
        {
            "offer_source_query_id": record["query_id"],
            "offer_id": record["offer_id"],
            "offer_text": record["offer_text"],
        }
        for records_by_query in (positive_records_by_query, negative_records_by_query)
        for records in records_by_query.values()
        for record in records
    ]

    return {
        "positive_records_by_query": positive_records_by_query,
        "negative_records_by_query": negative_records_by_query,
        "hard_negative_records_by_query": hard_negative_records_by_query,
        "eligible_query_ids": ["q1"],
        "synthetic_negative_offer_pool": synthetic_negative_offer_pool,
        "seed": 7,
    }


class BuildHardNegativeRecordTests(unittest.TestCase):
    def test_creates_record_with_hard_negative_label(self):
        record = build_hard_negative_record(
            "q1",
            "anchor query",
            {"offer_id": "hn-1", "offer_text": "hard negative 1"},
        )

        self.assertEqual(record["query_id"], "q1")
        self.assertEqual(record["query_text"], "anchor query")
        self.assertEqual(record["offer_id"], "hn-1")
        self.assertEqual(record["offer_text"], "hard negative 1")
        self.assertEqual(record["label"], 0.0)
        self.assertEqual(record["raw_label"], HARD_NEGATIVE_LABEL)


class BuildBatchStatsTests(unittest.TestCase):
    def test_counts_hard_negatives_separately(self):
        records = [
            make_record("q1", "q1-p1", 1.0, "q", "p"),
            {"query_id": "q1", "offer_id": "hn-1", "label": 0.0, "raw_label": HARD_NEGATIVE_LABEL},
            {"query_id": "q1", "offer_id": "sn-1", "label": 0.0, "raw_label": SYNTHETIC_NEGATIVE_LABEL},
            make_record("q1", "q1-n1", 0.0, "q", "n"),
        ]

        stats = build_batch_stats(records)

        self.assertEqual(stats["positive_count"], 1)
        self.assertEqual(stats["hard_negative_count"], 1)
        self.assertEqual(stats["cross_query_negative_count"], 1)
        self.assertEqual(stats["same_query_negative_count"], 1)


class AnchorQueryBatchBuilderHardNegativeTests(unittest.TestCase):
    def test_uses_hard_negatives_when_same_query_negatives_exhausted(self):
        inputs = build_inputs_with_hard_negatives()
        builder = AnchorQueryBatchBuilder(
            positive_records_by_query=inputs["positive_records_by_query"],
            negative_records_by_query=inputs["negative_records_by_query"],
            eligible_query_ids=inputs["eligible_query_ids"],
            synthetic_negative_offer_pool=inputs["synthetic_negative_offer_pool"],
            batch_size=4,
            n_pos_samples_per_query=2,
            n_neg_samples_per_query=2,
            seed=inputs["seed"],
            hard_negative_records_by_query=inputs["hard_negative_records_by_query"],
        )

        batch = builder.build_batch("q1")
        records = batch["records"]

        self.assertEqual(len(records), 4)
        self.assertEqual(batch["batch_stats"]["positive_count"], 2)
        self.assertEqual(batch["batch_stats"]["same_query_negative_count"], 1)
        self.assertEqual(batch["batch_stats"]["hard_negative_count"], 1)

    def test_hard_negatives_fill_batch_before_synthetic(self):
        inputs = build_inputs_with_hard_negatives()
        builder = AnchorQueryBatchBuilder(
            positive_records_by_query=inputs["positive_records_by_query"],
            negative_records_by_query=inputs["negative_records_by_query"],
            eligible_query_ids=inputs["eligible_query_ids"],
            synthetic_negative_offer_pool=inputs["synthetic_negative_offer_pool"],
            batch_size=8,
            n_pos_samples_per_query=2,
            n_neg_samples_per_query=2,
            seed=inputs["seed"],
            hard_negative_records_by_query=inputs["hard_negative_records_by_query"],
        )

        batch = builder.build_batch("q1")
        stats = batch["batch_stats"]

        self.assertEqual(len(batch["records"]), 8)
        self.assertGreater(stats["hard_negative_count"], 0)

    def test_falls_back_to_synthetic_when_hard_negatives_exhausted(self):
        inputs = build_inputs_with_hard_negatives()
        inputs["hard_negative_records_by_query"]["q1"] = [
            {"offer_id": "hn-1", "offer_text": "hard negative 1"},
        ]
        builder = AnchorQueryBatchBuilder(
            positive_records_by_query=inputs["positive_records_by_query"],
            negative_records_by_query=inputs["negative_records_by_query"],
            eligible_query_ids=inputs["eligible_query_ids"],
            synthetic_negative_offer_pool=inputs["synthetic_negative_offer_pool"],
            batch_size=8,
            n_pos_samples_per_query=2,
            n_neg_samples_per_query=2,
            seed=inputs["seed"],
            hard_negative_records_by_query=inputs["hard_negative_records_by_query"],
        )

        batch = builder.build_batch("q1")
        stats = batch["batch_stats"]

        self.assertEqual(len(batch["records"]), 8)
        self.assertGreater(stats["cross_query_negative_count"], 0)

    def test_works_without_hard_negatives(self):
        inputs = build_inputs_with_hard_negatives()
        builder = AnchorQueryBatchBuilder(
            positive_records_by_query=inputs["positive_records_by_query"],
            negative_records_by_query=inputs["negative_records_by_query"],
            eligible_query_ids=inputs["eligible_query_ids"],
            synthetic_negative_offer_pool=inputs["synthetic_negative_offer_pool"],
            batch_size=5,
            n_pos_samples_per_query=2,
            n_neg_samples_per_query=2,
            seed=inputs["seed"],
        )

        batch = builder.build_batch("q1")
        self.assertEqual(len(batch["records"]), 5)
        self.assertEqual(batch["batch_stats"]["hard_negative_count"], 0)


class RandomQueryPoolBuilderHardNegativeTests(unittest.TestCase):
    def test_includes_hard_negatives_in_pool(self):
        inputs = build_inputs_with_hard_negatives()
        builder = RandomQueryPoolBuilder(
            positive_records_by_query=inputs["positive_records_by_query"],
            negative_records_by_query=inputs["negative_records_by_query"],
            eligible_query_ids=inputs["eligible_query_ids"],
            synthetic_negative_offer_pool=inputs["synthetic_negative_offer_pool"],
            n_pos_samples_per_query=2,
            n_neg_samples_per_query=2,
            seed=inputs["seed"],
            hard_negative_records_by_query=inputs["hard_negative_records_by_query"],
        )

        pool = builder.build_pool()
        hard_negative_records = [
            r for r in pool if r.get("raw_label") == HARD_NEGATIVE_LABEL
        ]

        self.assertGreater(len(hard_negative_records), 0)
        for record in hard_negative_records:
            self.assertEqual(record["label"], 0.0)
            self.assertEqual(record["query_id"], "q1")


class MineHardNegativesFromResultsTests(unittest.TestCase):
    def test_excludes_known_positives(self):
        query_rows = [{"query_id": "q1", "query_text": "test query"}]
        scores = [[0.9, 0.85, 0.8, 0.75]]
        indices = [[0, 1, 2, 3]]
        metadata_by_id = {
            0: {"offer_id_b64": "pos-1", "offer_text": "positive offer"},
            1: {"offer_id_b64": "neg-1", "offer_text": "hard neg 1"},
            2: {"offer_id_b64": "neg-2", "offer_text": "hard neg 2"},
            3: {"offer_id_b64": "neg-3", "offer_text": "hard neg 3"},
        }
        positive_offer_ids_by_query = {"q1": {"pos-1"}}

        results = mine_hard_negatives_from_results(
            query_rows,
            scores,
            indices,
            metadata_by_id,
            positive_offer_ids_by_query,
            offer_id_column="offer_id_b64",
            max_negatives_per_query=10,
        )

        offer_ids = {r["offer_id"] for r in results}
        self.assertNotIn("pos-1", offer_ids)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["offer_id"], "neg-1")
        self.assertEqual(results[0]["rank"], 1)

    def test_respects_max_negatives_per_query(self):
        query_rows = [{"query_id": "q1", "query_text": "test query"}]
        scores = [[0.9, 0.85, 0.8]]
        indices = [[0, 1, 2]]
        metadata_by_id = {
            0: {"offer_id_b64": "neg-1", "offer_text": "hard neg 1"},
            1: {"offer_id_b64": "neg-2", "offer_text": "hard neg 2"},
            2: {"offer_id_b64": "neg-3", "offer_text": "hard neg 3"},
        }

        results = mine_hard_negatives_from_results(
            query_rows,
            scores,
            indices,
            metadata_by_id,
            positive_offer_ids_by_query={},
            offer_id_column="offer_id_b64",
            max_negatives_per_query=2,
        )

        self.assertEqual(len(results), 2)

    def test_skips_invalid_faiss_ids(self):
        query_rows = [{"query_id": "q1", "query_text": "test query"}]
        scores = [[0.9, 0.85]]
        indices = [[-1, 0]]
        metadata_by_id = {
            0: {"offer_id_b64": "neg-1", "offer_text": "hard neg 1"},
        }

        results = mine_hard_negatives_from_results(
            query_rows,
            scores,
            indices,
            metadata_by_id,
            positive_offer_ids_by_query={},
            offer_id_column="offer_id_b64",
            max_negatives_per_query=10,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["offer_id"], "neg-1")

    def test_multiple_queries(self):
        query_rows = [
            {"query_id": "q1", "query_text": "query 1"},
            {"query_id": "q2", "query_text": "query 2"},
        ]
        scores = [[0.9, 0.8], [0.7, 0.6]]
        indices = [[0, 1], [2, 3]]
        metadata_by_id = {
            0: {"offer_id_b64": "o1", "offer_text": "offer 1"},
            1: {"offer_id_b64": "o2", "offer_text": "offer 2"},
            2: {"offer_id_b64": "o3", "offer_text": "offer 3"},
            3: {"offer_id_b64": "o4", "offer_text": "offer 4"},
        }
        positive_offer_ids_by_query = {"q1": {"o1"}, "q2": {"o3"}}

        results = mine_hard_negatives_from_results(
            query_rows,
            scores,
            indices,
            metadata_by_id,
            positive_offer_ids_by_query,
            offer_id_column="offer_id_b64",
            max_negatives_per_query=10,
        )

        q1_results = [r for r in results if r["query_id"] == "q1"]
        q2_results = [r for r in results if r["query_id"] == "q2"]

        self.assertEqual(len(q1_results), 1)
        self.assertEqual(q1_results[0]["offer_id"], "o2")
        self.assertEqual(len(q2_results), 1)
        self.assertEqual(q2_results[0]["offer_id"], "o4")


if __name__ == "__main__":
    unittest.main()
