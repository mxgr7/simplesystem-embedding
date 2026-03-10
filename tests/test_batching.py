import unittest

from embedding_train.batching import (
    AnchorQueryBatchBuilder,
    SYNTHETIC_NEGATIVE_LABEL,
    build_synthetic_negative_record,
)


def make_record(query_id, offer_id, label, query_text, offer_text):
    return {
        "query_id": query_id,
        "offer_id": offer_id,
        "query_text": query_text,
        "offer_text": offer_text,
        "label": label,
        "raw_label": "Exact" if label > 0.5 else "Irrelevant",
    }


def build_builder(batch_size=6, n_pos_samples_per_query=2, n_neg_samples_per_query=2):
    positive_records_by_query = {
        "q1": [
            make_record("q1", "q1-p1", 1.0, "anchor query", "anchor positive 1"),
            make_record("q1", "q1-p2", 1.0, "anchor query", "anchor positive 2"),
            make_record("q1", "q1-p3", 1.0, "anchor query", "anchor positive 3"),
            make_record("q1", "q1-p4", 1.0, "anchor query", "anchor positive 4"),
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
            make_record("q3", "q3-n2", 0.0, "foreign query 3", "foreign negative 4"),
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

    return AnchorQueryBatchBuilder(
        positive_records_by_query=positive_records_by_query,
        negative_records_by_query=negative_records_by_query,
        eligible_query_ids=["q1"],
        synthetic_negative_offer_pool=synthetic_negative_offer_pool,
        batch_size=batch_size,
        n_pos_samples_per_query=n_pos_samples_per_query,
        n_neg_samples_per_query=n_neg_samples_per_query,
        seed=7,
    )


class AnchorQueryBatchBuilderTests(unittest.TestCase):
    def test_build_synthetic_negative_record_uses_anchor_query_and_foreign_offer(self):
        record = build_synthetic_negative_record(
            "q1",
            "anchor query",
            {
                "offer_source_query_id": "q2",
                "offer_id": "q2-p1",
                "offer_text": "foreign positive 1",
            },
        )

        self.assertEqual(
            record,
            {
                "query_id": "q1",
                "query_text": "anchor query",
                "offer_id": "q2-p1",
                "offer_text": "foreign positive 1",
                "label": 0.0,
                "raw_label": SYNTHETIC_NEGATIVE_LABEL,
            },
        )

    def test_build_synthetic_negative_record_rejects_same_query_offer(self):
        with self.assertRaisesRegex(
            ValueError, "Synthetic negative offer must come from a foreign query"
        ):
            build_synthetic_negative_record(
                "q1",
                "anchor query",
                {
                    "offer_source_query_id": "q1",
                    "offer_id": "q1-p1",
                    "offer_text": "anchor positive 1",
                },
            )

    def test_builds_batch_centered_on_single_anchor_query(self):
        batch = build_builder(batch_size=5).build_batch("q1")

        self.assertEqual({record["query_id"] for record in batch["records"]}, {"q1"})
        self.assertEqual(batch["batch_stats"]["anchor_query_id"], "q1")

    def test_meets_positive_and_negative_minimums(self):
        batch = build_builder(batch_size=5).build_batch("q1")
        labels = [record["label"] for record in batch["records"]]
        negative_count = len(labels) - int(sum(labels))

        self.assertGreaterEqual(batch["batch_stats"]["positive_count"], 2)
        self.assertGreaterEqual(negative_count, 2)
        self.assertEqual(
            batch["batch_stats"]["same_query_negative_count"]
            + batch["batch_stats"]["cross_query_negative_count"],
            negative_count,
        )

    def test_consumes_same_query_negatives_before_cross_query_backfill(self):
        batch = build_builder(batch_size=5).build_batch("q1")
        records = batch["records"]

        self.assertEqual(records[2]["offer_id"], "q1-n1")
        self.assertEqual(records[3]["raw_label"], SYNTHETIC_NEGATIVE_LABEL)
        self.assertEqual(batch["batch_stats"]["same_query_negative_count"], 1)
        self.assertEqual(batch["batch_stats"]["cross_query_negative_count"], 1)

    def test_backfills_negative_shortfall_before_extra_anchor_examples(self):
        batch = build_builder(batch_size=6).build_batch("q1")
        records = batch["records"]

        self.assertEqual(records[2]["offer_id"], "q1-n1")
        self.assertEqual(records[3]["raw_label"], SYNTHETIC_NEGATIVE_LABEL)
        self.assertEqual(records[4]["raw_label"], "Exact")
        self.assertEqual(records[5]["raw_label"], "Exact")

    def test_uses_fallback_negatives_after_anchor_examples_are_exhausted(self):
        builder = build_builder(batch_size=7)
        batch = builder.build_batch("q1")
        records = batch["records"]

        self.assertTrue(
            all(record["offer_id"].startswith("q1-") for record in records[:3])
        )
        self.assertEqual(records[3]["raw_label"], SYNTHETIC_NEGATIVE_LABEL)
        self.assertEqual(records[4]["raw_label"], "Exact")
        self.assertEqual(records[5]["raw_label"], "Exact")
        self.assertEqual(records[6]["raw_label"], SYNTHETIC_NEGATIVE_LABEL)

    def test_synthetic_negatives_keep_anchor_query_and_foreign_offer(self):
        batch = build_builder(batch_size=5).build_batch("q1")
        synthetic_records = [
            record
            for record in batch["records"]
            if record["raw_label"] == SYNTHETIC_NEGATIVE_LABEL
        ]

        self.assertTrue(synthetic_records)
        self.assertEqual(
            {record["query_text"] for record in synthetic_records}, {"anchor query"}
        )
        self.assertTrue(
            all(
                not record["offer_id"].startswith("q1-") for record in synthetic_records
            )
        )
        self.assertTrue(
            {record["offer_text"] for record in synthetic_records}.issubset(
                {
                    "foreign positive 1",
                    "foreign positive 2",
                    "foreign positive 3",
                    "foreign positive 4",
                    "foreign negative 1",
                    "foreign negative 2",
                    "foreign negative 3",
                    "foreign negative 4",
                }
            )
        )
        self.assertEqual({record["label"] for record in synthetic_records}, {0.0})


if __name__ == "__main__":
    unittest.main()
