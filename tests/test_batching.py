import unittest
from types import SimpleNamespace
from unittest import mock

from embedding_train.batching import (
    AnchorQueryBatchBuilder,
    AnchorQueryBatchDataset,
    RandomQueryPoolBuilder,
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


def build_query_sampling_inputs():
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

    return {
        "positive_records_by_query": positive_records_by_query,
        "negative_records_by_query": negative_records_by_query,
        "eligible_query_ids": ["q1"],
        "synthetic_negative_offer_pool": synthetic_negative_offer_pool,
        "seed": 7,
    }


def build_builder(batch_size=6, n_pos_samples_per_query=2, n_neg_samples_per_query=2):
    inputs = build_query_sampling_inputs()

    return AnchorQueryBatchBuilder(
        positive_records_by_query=inputs["positive_records_by_query"],
        negative_records_by_query=inputs["negative_records_by_query"],
        eligible_query_ids=inputs["eligible_query_ids"],
        synthetic_negative_offer_pool=inputs["synthetic_negative_offer_pool"],
        batch_size=batch_size,
        n_pos_samples_per_query=n_pos_samples_per_query,
        n_neg_samples_per_query=n_neg_samples_per_query,
        seed=inputs["seed"],
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


class RandomQueryPoolBuilderTests(unittest.TestCase):
    def test_build_pool_uses_all_real_records_for_eligible_queries(self):
        inputs = build_query_sampling_inputs()
        inputs["eligible_query_ids"] = ["q1", "q2"]
        builder = RandomQueryPoolBuilder(
            positive_records_by_query=inputs["positive_records_by_query"],
            negative_records_by_query=inputs["negative_records_by_query"],
            eligible_query_ids=inputs["eligible_query_ids"],
            synthetic_negative_offer_pool=inputs["synthetic_negative_offer_pool"],
            n_pos_samples_per_query=2,
            n_neg_samples_per_query=2,
            seed=inputs["seed"],
        )

        pool = builder.build_pool()
        offer_ids = {record["offer_id"] for record in pool}

        self.assertTrue(
            {"q1-p1", "q1-p2", "q1-p3", "q1-p4", "q1-n1"}.issubset(offer_ids)
        )
        self.assertTrue({"q2-p1", "q2-p2", "q2-n1", "q2-n2"}.issubset(offer_ids))
        self.assertGreater(len(pool), 9)

    def test_build_pool_backfills_only_negative_shortfall(self):
        inputs = build_query_sampling_inputs()
        builder = RandomQueryPoolBuilder(
            positive_records_by_query=inputs["positive_records_by_query"],
            negative_records_by_query=inputs["negative_records_by_query"],
            eligible_query_ids=inputs["eligible_query_ids"],
            synthetic_negative_offer_pool=inputs["synthetic_negative_offer_pool"],
            n_pos_samples_per_query=2,
            n_neg_samples_per_query=2,
            seed=inputs["seed"],
        )

        pool = builder.build_pool()
        q1_records = [record for record in pool if record["query_id"] == "q1"]
        synthetic_records = [
            record
            for record in q1_records
            if record["raw_label"] == SYNTHETIC_NEGATIVE_LABEL
        ]
        real_negative_records = [
            record for record in q1_records if record["offer_id"] == "q1-n1"
        ]

        self.assertEqual(len(q1_records), 6)
        self.assertEqual(len(synthetic_records), 1)
        self.assertEqual(len(real_negative_records), 1)
        self.assertEqual(sum(record["label"] > 0.5 for record in q1_records), 4)

    def test_build_pool_keeps_synthetic_negative_foreign_to_query(self):
        inputs = build_query_sampling_inputs()
        builder = RandomQueryPoolBuilder(
            positive_records_by_query=inputs["positive_records_by_query"],
            negative_records_by_query=inputs["negative_records_by_query"],
            eligible_query_ids=inputs["eligible_query_ids"],
            synthetic_negative_offer_pool=inputs["synthetic_negative_offer_pool"],
            n_pos_samples_per_query=2,
            n_neg_samples_per_query=2,
            seed=inputs["seed"],
        )

        pool = builder.build_pool()
        synthetic_records = [
            record for record in pool if record["raw_label"] == SYNTHETIC_NEGATIVE_LABEL
        ]

        self.assertEqual(len(synthetic_records), 1)
        self.assertEqual(synthetic_records[0]["query_id"], "q1")
        self.assertFalse(synthetic_records[0]["offer_id"].startswith("q1-"))


class AnchorQueryBatchDatasetTests(unittest.TestCase):
    def _make_multi_query_builder(self):
        inputs = build_query_sampling_inputs()
        inputs["eligible_query_ids"] = ["q1", "q2", "q3"]
        # q2/q3 only have 2 positives; use n_pos_samples_per_query=2 to stay eligible.
        return AnchorQueryBatchBuilder(
            positive_records_by_query=inputs["positive_records_by_query"],
            negative_records_by_query=inputs["negative_records_by_query"],
            eligible_query_ids=inputs["eligible_query_ids"],
            synthetic_negative_offer_pool=inputs["synthetic_negative_offer_pool"],
            batch_size=4,
            n_pos_samples_per_query=2,
            n_neg_samples_per_query=1,
            seed=inputs["seed"],
        )

    def _run_shard(self, batches_per_epoch, worker_id, num_workers, initial_seed):
        dataset = AnchorQueryBatchDataset(
            self._make_multi_query_builder(), batches_per_epoch
        )
        info = SimpleNamespace(id=worker_id, num_workers=num_workers)
        with mock.patch(
            "embedding_train.batching.torch.utils.data.get_worker_info",
            return_value=info,
        ), mock.patch(
            "embedding_train.batching.torch.initial_seed",
            return_value=initial_seed,
        ):
            return list(iter(dataset))

    def test_no_worker_info_yields_full_epoch(self):
        dataset = AnchorQueryBatchDataset(self._make_multi_query_builder(), 9)
        with mock.patch(
            "embedding_train.batching.torch.utils.data.get_worker_info",
            return_value=None,
        ):
            batches = list(iter(dataset))
        self.assertEqual(len(batches), 9)

    def test_sharding_divides_batches_across_workers(self):
        # 9 batches over 4 workers: ceil split 3,2,2,2 → total 9, none duplicated.
        counts = [
            len(self._run_shard(9, worker_id=i, num_workers=4, initial_seed=1000 + i))
            for i in range(4)
        ]
        self.assertEqual(sorted(counts, reverse=True), [3, 2, 2, 2])
        self.assertEqual(sum(counts), 9)

    def test_workers_produce_disjoint_anchor_sequences(self):
        # Same batches_per_epoch per worker, different initial_seed → different
        # anchor choices. Without the per-worker re-seed, both workers would
        # replay the pickled-at-construction RNG and yield identical sequences.
        worker_0 = self._run_shard(6, worker_id=0, num_workers=2, initial_seed=1001)
        worker_1 = self._run_shard(6, worker_id=1, num_workers=2, initial_seed=2002)
        anchors_0 = [b["batch_stats"]["anchor_query_id"] for b in worker_0]
        anchors_1 = [b["batch_stats"]["anchor_query_id"] for b in worker_1]
        self.assertEqual(len(anchors_0), 3)
        self.assertEqual(len(anchors_1), 3)
        self.assertNotEqual(anchors_0, anchors_1)


if __name__ == "__main__":
    unittest.main()
