import unittest
from unittest.mock import patch

import pandas as pd
import torch
from omegaconf import OmegaConf

from embedding_train.data import EmbeddingDataModule


class _TokenizerStub:
    pad_token = "[PAD]"
    eos_token = "[EOS]"

    def __call__(
        self,
        texts,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        del padding, truncation, max_length, return_tensors
        batch_size = len(texts)
        return {
            "input_ids": torch.ones((batch_size, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 2), dtype=torch.long),
        }


def build_cfg(**data_overrides):
    return OmegaConf.create(
        {
            "seed": 42,
            "model": {"model_name": "stub-model"},
            "data": {
                "path": "/tmp/dataset.parquet",
                "positive_label": "Exact",
                "batch_size": 16,
                "train_batching_mode": "random_pairs",
                "n_pos_samples_per_query": 2,
                "n_neg_samples_per_query": 2,
                "log_batch_stats": True,
                "num_workers": 0,
                "pin_memory": False,
                "max_query_length": 32,
                "max_offer_length": 256,
                "val_fraction": 0.05,
                "clean_html": True,
                "limit_rows": None,
                "query_template": "{{ query_term }}",
                "offer_template": "{{ name }}",
                **data_overrides,
            },
        }
    )


class EmbeddingDataModuleConfigTests(unittest.TestCase):
    @patch("embedding_train.data.AutoTokenizer.from_pretrained")
    def test_rejects_unknown_train_batching_mode(self, from_pretrained):
        from_pretrained.return_value = _TokenizerStub()

        with self.assertRaisesRegex(ValueError, "Unsupported train batching mode"):
            EmbeddingDataModule(build_cfg(train_batching_mode="unknown_mode"))

    @patch("embedding_train.data.AutoTokenizer.from_pretrained")
    def test_rejects_anchor_query_batch_size_smaller_than_minimum(
        self, from_pretrained
    ):
        from_pretrained.return_value = _TokenizerStub()

        with self.assertRaisesRegex(
            ValueError,
            r"batch_size must be at least n_pos_samples_per_query \+ n_neg_samples_per_query",
        ):
            EmbeddingDataModule(
                build_cfg(
                    train_batching_mode="anchor_query",
                    batch_size=3,
                    n_pos_samples_per_query=2,
                    n_neg_samples_per_query=2,
                )
            )

    @patch("embedding_train.data.AutoTokenizer.from_pretrained")
    def test_accepts_random_pairs_with_default_phase_one_settings(
        self, from_pretrained
    ):
        from_pretrained.return_value = _TokenizerStub()

        datamodule = EmbeddingDataModule(build_cfg())

        self.assertEqual(datamodule.train_batching_mode, "random_pairs")


class EmbeddingDataModuleMetadataTests(unittest.TestCase):
    def build_frame(self):
        return pd.DataFrame(
            [
                {
                    "query_id": "q1",
                    "offer_id_b64": "o1",
                    "query_term": "query one",
                    "name": "offer one",
                    "label": "Exact",
                },
                {
                    "query_id": "q1",
                    "offer_id_b64": "o2",
                    "query_term": "query one",
                    "name": "offer two",
                    "label": "Exact",
                },
                {
                    "query_id": "q2",
                    "offer_id_b64": "o3",
                    "query_term": "query two",
                    "name": "offer three",
                    "label": "Exact",
                },
                {
                    "query_id": "q2",
                    "offer_id_b64": "o4",
                    "query_term": "query two",
                    "name": "offer four",
                    "label": "Exact",
                },
                {
                    "query_id": "q2",
                    "offer_id_b64": "o5",
                    "query_term": "query two",
                    "name": "offer five",
                    "label": "Irrelevant",
                },
                {
                    "query_id": "q3",
                    "offer_id_b64": "o6",
                    "query_term": "query three",
                    "name": "offer six",
                    "label": "Exact",
                },
                {
                    "query_id": "q3",
                    "offer_id_b64": "o7",
                    "query_term": "query three",
                    "name": "offer seven",
                    "label": "Irrelevant",
                },
                {
                    "query_id": "q3",
                    "offer_id_b64": "o8",
                    "query_term": "query three",
                    "name": "offer eight",
                    "label": "Irrelevant",
                },
                {
                    "query_id": "q3",
                    "offer_id_b64": "o9",
                    "query_term": "query three",
                    "name": "offer nine",
                    "label": "Irrelevant",
                },
                {
                    "query_id": "q3",
                    "offer_id_b64": "o10",
                    "query_term": "query three",
                    "name": "offer ten",
                    "label": "Irrelevant",
                },
            ]
        )

    @patch("embedding_train.data.pd.read_parquet")
    @patch("embedding_train.data.AutoTokenizer.from_pretrained")
    def test_builds_train_metadata_and_eligibility_from_positive_counts_only(
        self, from_pretrained, read_parquet
    ):
        from_pretrained.return_value = _TokenizerStub()
        read_parquet.return_value = self.build_frame()
        datamodule = EmbeddingDataModule(
            build_cfg(
                val_fraction=0.0,
                n_pos_samples_per_query=2,
                n_neg_samples_per_query=3,
            )
        )

        datamodule.setup()

        self.assertEqual(
            sorted(datamodule.positive_records_by_query), ["q1", "q2", "q3"]
        )
        self.assertEqual(sorted(datamodule.negative_records_by_query), ["q2", "q3"])
        self.assertEqual(datamodule.eligible_query_ids, ["q1", "q2"])
        self.assertEqual(len(datamodule.positive_records_by_query["q1"]), 2)
        self.assertEqual(len(datamodule.negative_records_by_query["q2"]), 1)
        self.assertEqual(len(datamodule.synthetic_negative_offer_pool), 10)
        self.assertEqual(
            datamodule.synthetic_negative_offer_pool[0],
            {
                "offer_source_query_id": "q1",
                "offer_id": "o1",
                "offer_text": "offer one",
            },
        )
        self.assertEqual(datamodule.dataset_stats["eligible_train_queries"], 2)

    @patch("embedding_train.data.pd.read_parquet")
    @patch("embedding_train.data.AutoTokenizer.from_pretrained")
    def test_anchor_query_mode_raises_when_no_query_has_enough_positives(
        self, from_pretrained, read_parquet
    ):
        from_pretrained.return_value = _TokenizerStub()
        read_parquet.return_value = pd.DataFrame(
            [
                {
                    "query_id": "q1",
                    "offer_id_b64": "o1",
                    "query_term": "query one",
                    "name": "offer one",
                    "label": "Exact",
                },
                {
                    "query_id": "q2",
                    "offer_id_b64": "o2",
                    "query_term": "query two",
                    "name": "offer two",
                    "label": "Exact",
                },
            ]
        )
        datamodule = EmbeddingDataModule(
            build_cfg(
                train_batching_mode="anchor_query",
                val_fraction=0.0,
                n_pos_samples_per_query=2,
                n_neg_samples_per_query=0,
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "No train query has at least n_pos_samples_per_query=2 positive rows",
        ):
            datamodule.setup()

    @patch("embedding_train.data.pd.read_parquet")
    @patch("embedding_train.data.AutoTokenizer.from_pretrained")
    def test_random_pairs_allows_setup_even_without_eligible_anchor_queries(
        self, from_pretrained, read_parquet
    ):
        from_pretrained.return_value = _TokenizerStub()
        read_parquet.return_value = pd.DataFrame(
            [
                {
                    "query_id": "q1",
                    "offer_id_b64": "o1",
                    "query_term": "query one",
                    "name": "offer one",
                    "label": "Exact",
                }
            ]
        )
        datamodule = EmbeddingDataModule(
            build_cfg(
                train_batching_mode="random_pairs",
                val_fraction=0.0,
                n_pos_samples_per_query=2,
            )
        )

        datamodule.setup()

        self.assertEqual(datamodule.eligible_query_ids, [])

    @patch("embedding_train.data.pd.read_parquet")
    @patch("embedding_train.data.AutoTokenizer.from_pretrained")
    def test_anchor_query_train_dataloader_uses_single_anchor_query_batches(
        self, from_pretrained, read_parquet
    ):
        from_pretrained.return_value = _TokenizerStub()
        read_parquet.return_value = pd.DataFrame(
            [
                {
                    "query_id": "q1",
                    "offer_id_b64": "q1-p1",
                    "query_term": "query one",
                    "name": "offer one",
                    "label": "Exact",
                },
                {
                    "query_id": "q1",
                    "offer_id_b64": "q1-p2",
                    "query_term": "query one",
                    "name": "offer two",
                    "label": "Exact",
                },
                {
                    "query_id": "q1",
                    "offer_id_b64": "q1-n1",
                    "query_term": "query one",
                    "name": "offer three",
                    "label": "Irrelevant",
                },
                {
                    "query_id": "q2",
                    "offer_id_b64": "q2-p1",
                    "query_term": "query two",
                    "name": "offer four",
                    "label": "Exact",
                },
                {
                    "query_id": "q2",
                    "offer_id_b64": "q2-n1",
                    "query_term": "query two",
                    "name": "offer five",
                    "label": "Irrelevant",
                },
            ]
        )
        datamodule = EmbeddingDataModule(
            build_cfg(
                train_batching_mode="anchor_query",
                batch_size=4,
                val_fraction=0.0,
                n_pos_samples_per_query=2,
                n_neg_samples_per_query=1,
            )
        )

        datamodule.setup()
        batch = next(iter(datamodule.train_dataloader()))

        self.assertEqual(set(batch["query_ids"]), {"q1"})
        self.assertEqual(batch["labels"].size(0), 4)
        self.assertEqual(batch["query_inputs"]["input_ids"].shape[0], 4)
        self.assertIn("SyntheticNegative", batch["raw_labels"])
        self.assertEqual(batch["batch_stats"]["anchor_query_id"], "q1")
        self.assertGreaterEqual(batch["batch_stats"]["positive_count"], 2)
        self.assertGreaterEqual(
            batch["batch_stats"]["same_query_negative_count"]
            + batch["batch_stats"]["cross_query_negative_count"],
            1,
        )

    @patch("embedding_train.data.pd.read_parquet")
    @patch("embedding_train.data.AutoTokenizer.from_pretrained")
    def test_random_pairs_collate_shape_remains_unchanged(
        self, from_pretrained, read_parquet
    ):
        from_pretrained.return_value = _TokenizerStub()
        read_parquet.return_value = self.build_frame()
        datamodule = EmbeddingDataModule(
            build_cfg(
                train_batching_mode="random_pairs",
                batch_size=3,
                val_fraction=0.0,
            )
        )

        datamodule.setup()
        batch = next(iter(datamodule.train_dataloader()))

        self.assertEqual(batch["labels"].size(0), 3)
        self.assertNotIn("batch_stats", batch)
        self.assertEqual(len(batch["query_ids"]), 3)
        self.assertEqual(batch["query_inputs"]["input_ids"].shape[0], 3)

    @patch("embedding_train.data.pd.read_parquet")
    @patch("embedding_train.data.AutoTokenizer.from_pretrained")
    def test_validation_dataloader_remains_exhaustive_and_real_only(
        self, from_pretrained, read_parquet
    ):
        from_pretrained.return_value = _TokenizerStub()
        read_parquet.return_value = self.build_frame()
        datamodule = EmbeddingDataModule(
            build_cfg(
                train_batching_mode="anchor_query",
                batch_size=2,
                val_fraction=0.34,
                n_pos_samples_per_query=2,
                n_neg_samples_per_query=0,
            )
        )

        datamodule.setup()
        val_batches = list(iter(datamodule.val_dataloader()))
        val_raw_labels = [
            raw_label for batch in val_batches for raw_label in batch["raw_labels"]
        ]
        val_row_count = sum(batch["labels"].size(0) for batch in val_batches)

        self.assertEqual(val_row_count, len(datamodule.val_dataset))
        self.assertNotIn("SyntheticNegative", val_raw_labels)
        self.assertTrue(all("batch_stats" not in batch for batch in val_batches))


if __name__ == "__main__":
    unittest.main()
