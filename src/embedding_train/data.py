import random
from typing import cast

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from embedding_train.batching import AnchorQueryBatchBuilder, AnchorQueryBatchDataset
from embedding_train.rendering import RowTextRenderer

VALID_TRAIN_BATCHING_MODES = {"random_pairs", "anchor_query"}


def resolve_train_batching_mode(train_batching_mode):
    normalized = str(train_batching_mode).strip().lower()

    if normalized in VALID_TRAIN_BATCHING_MODES:
        return normalized

    choices = "|".join(sorted(VALID_TRAIN_BATCHING_MODES))
    raise ValueError(
        "Unsupported train batching mode: "
        f"{train_batching_mode}. Expected one of {choices}"
    )


class PairDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class EmbeddingDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_batching_mode = resolve_train_batching_mode(
            cfg.data.train_batching_mode
        )
        self.row_renderer = RowTextRenderer(cfg.data)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.model_name, use_fast=True
        )
        self.train_dataset = None
        self.val_dataset = None
        self.dataset_stats = {}
        self.positive_records_by_query = {}
        self.negative_records_by_query = {}
        self.eligible_query_ids = []
        self.synthetic_negative_offer_pool = []
        self._validate_batching_config()

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _validate_batching_config(self):
        n_pos_samples_per_query = int(self.cfg.data.n_pos_samples_per_query)
        n_neg_samples_per_query = int(self.cfg.data.n_neg_samples_per_query)

        if n_pos_samples_per_query < 1:
            raise ValueError("n_pos_samples_per_query must be at least 1")

        if n_neg_samples_per_query < 0:
            raise ValueError("n_neg_samples_per_query must be at least 0")

        if self.train_batching_mode != "anchor_query":
            return

        minimum_batch_size = n_pos_samples_per_query + n_neg_samples_per_query
        batch_size = int(self.cfg.data.batch_size)

        if batch_size < minimum_batch_size:
            raise ValueError(
                "batch_size must be at least "
                f"n_pos_samples_per_query + n_neg_samples_per_query "
                f"({minimum_batch_size}) when train_batching_mode=anchor_query"
            )

    def setup(self, stage=None):
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        frame = pd.read_parquet(self.cfg.data.path)

        if "_rn" in frame.columns:
            frame = frame.drop(columns=["_rn"])

        if self.cfg.data.limit_rows:
            frame = frame.head(int(self.cfg.data.limit_rows))

        query_ids = [
            value
            for value in frame["query_id"].dropna().astype(str).unique().tolist()
            if value
        ]
        randomizer = random.Random(self.cfg.seed)
        randomizer.shuffle(query_ids)

        val_size = int(len(query_ids) * float(self.cfg.data.val_fraction))
        if float(self.cfg.data.val_fraction) > 0 and val_size == 0:
            val_size = 1

        val_query_ids = set(query_ids[:val_size])

        columns = list(frame.columns)
        train_records = []
        val_records = []
        skipped_rows = 0

        for values in frame.itertuples(index=False, name=None):
            row = dict(zip(columns, values))
            record = self._build_record(row)
            if record is None:
                skipped_rows += 1
                continue

            if record["query_id"] in val_query_ids:
                val_records.append(record)
            else:
                train_records.append(record)

        self.train_dataset = PairDataset(train_records)
        self.val_dataset = PairDataset(val_records)
        self._build_train_metadata(train_records)
        self.dataset_stats = {
            "train_rows": len(train_records),
            "val_rows": len(val_records),
            "train_queries": len({record["query_id"] for record in train_records}),
            "val_queries": len({record["query_id"] for record in val_records}),
            "skipped_rows": skipped_rows,
            "eligible_train_queries": len(self.eligible_query_ids),
        }

        print("Loaded dataset:", self.dataset_stats)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized. Call setup() first.")

        if self.train_batching_mode == "anchor_query":
            train_batch_dataset = cast(
                Dataset, self._build_anchor_query_train_dataset()
            )
            return DataLoader(
                train_batch_dataset,
                batch_size=None,
                num_workers=int(self.cfg.data.num_workers),
                pin_memory=bool(self.cfg.data.pin_memory),
                collate_fn=self.collate_fn,
            )

        train_dataset = cast(Dataset, self.train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=int(self.cfg.data.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized. Call setup() first.")

        val_dataset = cast(Dataset, self.val_dataset)

        return DataLoader(
            val_dataset,
            batch_size=int(self.cfg.data.batch_size),
            shuffle=False,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        batch_records = batch
        batch_stats = None

        if isinstance(batch, dict) and "records" in batch:
            batch_records = batch["records"]
            batch_stats = batch.get("batch_stats")

        query_texts = [item["query_text"] for item in batch_records]
        offer_texts = [item["offer_text"] for item in batch_records]

        query_inputs = self.tokenizer(
            query_texts,
            padding=True,
            truncation=True,
            max_length=int(self.cfg.data.max_query_length),
            return_tensors="pt",
        )

        offer_inputs = self.tokenizer(
            offer_texts,
            padding=True,
            truncation=True,
            max_length=int(self.cfg.data.max_offer_length),
            return_tensors="pt",
        )

        collated_batch = {
            "query_inputs": dict(query_inputs),
            "offer_inputs": dict(offer_inputs),
            "labels": torch.tensor(
                [item["label"] for item in batch_records], dtype=torch.float32
            ),
            "query_ids": [item["query_id"] for item in batch_records],
            "offer_ids": [item["offer_id"] for item in batch_records],
            "raw_labels": [item["raw_label"] for item in batch_records],
        }

        if batch_stats is not None:
            collated_batch["batch_stats"] = dict(batch_stats)

        return collated_batch

    def _build_anchor_query_train_dataset(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized. Call setup() first.")

        batch_builder = AnchorQueryBatchBuilder(
            positive_records_by_query=self.positive_records_by_query,
            negative_records_by_query=self.negative_records_by_query,
            eligible_query_ids=self.eligible_query_ids,
            synthetic_negative_offer_pool=self.synthetic_negative_offer_pool,
            batch_size=int(self.cfg.data.batch_size),
            n_pos_samples_per_query=int(self.cfg.data.n_pos_samples_per_query),
            n_neg_samples_per_query=int(self.cfg.data.n_neg_samples_per_query),
            seed=int(self.cfg.seed),
        )
        train_rows = len(self.train_dataset)
        batch_size = int(self.cfg.data.batch_size)
        batches_per_epoch = max(1, (train_rows + batch_size - 1) // batch_size)
        return AnchorQueryBatchDataset(batch_builder, batches_per_epoch)

    def _build_record(self, row):
        return self.row_renderer.build_training_record(row)

    def _build_train_metadata(self, train_records):
        positive_records_by_query = {}
        negative_records_by_query = {}
        synthetic_negative_offer_pool = []

        for record in train_records:
            query_id = record["query_id"]

            if record["label"] > 0.5:
                positive_records_by_query.setdefault(query_id, []).append(record)
            else:
                negative_records_by_query.setdefault(query_id, []).append(record)

            synthetic_negative_offer_pool.append(
                {
                    "offer_source_query_id": query_id,
                    "offer_id": record["offer_id"],
                    "offer_text": record["offer_text"],
                }
            )

        n_pos_samples_per_query = int(self.cfg.data.n_pos_samples_per_query)
        eligible_query_ids = [
            query_id
            for query_id, records in positive_records_by_query.items()
            if len(records) >= n_pos_samples_per_query
        ]
        eligible_query_ids.sort()

        self.positive_records_by_query = positive_records_by_query
        self.negative_records_by_query = negative_records_by_query
        self.eligible_query_ids = eligible_query_ids
        self.synthetic_negative_offer_pool = synthetic_negative_offer_pool

        if self.train_batching_mode == "anchor_query" and not self.eligible_query_ids:
            raise ValueError(
                "No train query has at least "
                f"n_pos_samples_per_query={n_pos_samples_per_query} positive rows"
            )
