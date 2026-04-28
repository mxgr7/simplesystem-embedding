import random
import sys
from collections import Counter
from typing import cast

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from embedding_train.rendering import RowTextRenderer

from cross_encoder_train.labels import LABEL_TO_ID, NUM_CLASSES, encode_label


class PairDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class CrossEncoderDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.row_renderer = RowTextRenderer(cfg.data)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.model_name, use_fast=True
        )
        self.train_dataset = None
        self.val_dataset = None
        self.dataset_stats = {}
        self.class_counts = [0] * NUM_CLASSES
        self.class_weights = [1.0] * NUM_CLASSES

    def setup(self, stage=None):
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        all_records, skipped_rows, skipped_unknown_label = self._prepare_records()
        train_records, val_records, split_stats = self._split_records_by_query_id(
            all_records
        )

        if not train_records:
            raise ValueError(
                "Train split is empty. "
                f"target_val_queries={split_stats['target_val_queries']} "
                f"selected_val_queries={split_stats['selected_val_queries']}. "
                "Set data.val_fraction below 1.0."
            )

        self.train_dataset = PairDataset(train_records)
        self.val_dataset = PairDataset(val_records)
        self._compute_class_weights(train_records)
        self.dataset_stats = self._build_dataset_stats(
            train_records,
            val_records,
            skipped_rows,
            skipped_unknown_label,
            split_stats,
        )
        print("Loaded dataset:", self.dataset_stats, file=sys.stderr)

    def train_dataloader(self):
        return DataLoader(
            cast(Dataset, self.train_dataset),
            batch_size=int(self.cfg.data.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            cast(Dataset, self.val_dataset),
            batch_size=int(self.cfg.data.batch_size),
            shuffle=False,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        query_texts = [item["query_text"] for item in batch]
        offer_texts = [item["offer_text"] for item in batch]

        encoded = self.tokenizer(
            query_texts,
            offer_texts,
            padding=True,
            truncation="only_second",
            max_length=int(self.cfg.data.max_pair_length),
            return_tensors="pt",
            return_token_type_ids=True,
        )

        return {
            "inputs": dict(encoded),
            "labels": torch.tensor(
                [item["label_id"] for item in batch], dtype=torch.long
            ),
            "query_ids": [item["query_id"] for item in batch],
            "offer_ids": [item["offer_id"] for item in batch],
            "raw_labels": [item["raw_label"] for item in batch],
            "query_texts": query_texts,
            "offer_texts": offer_texts,
        }

    def _prepare_records(self):
        frame = pd.read_parquet(self.cfg.data.path)

        if "_rn" in frame.columns:
            frame = frame.drop(columns=["_rn"])

        if self.cfg.data.limit_rows:
            frame = frame.head(int(self.cfg.data.limit_rows))

        columns = list(frame.columns)
        records = []
        skipped_rows = 0
        skipped_unknown_label = 0

        for values in frame.itertuples(index=False, name=None):
            row = dict(zip(columns, values))
            base_record = self.row_renderer.build_training_record(row)
            if base_record is None:
                skipped_rows += 1
                continue

            raw_label = base_record["raw_label"]
            if raw_label not in LABEL_TO_ID:
                skipped_unknown_label += 1
                continue

            records.append(
                {
                    "query_id": base_record["query_id"],
                    "offer_id": base_record["offer_id"],
                    "query_text": base_record["query_text"],
                    "offer_text": base_record["offer_text"],
                    "raw_label": raw_label,
                    "label_id": encode_label(raw_label),
                }
            )

        return records, skipped_rows, skipped_unknown_label

    def _split_records_by_query_id(self, records):
        if not records:
            return [], [], self._empty_split_stats()

        records_by_query = {}
        query_ids = []
        seen_query_ids = set()

        for index, record in enumerate(records):
            query_id = record["query_id"]
            group_id = query_id or f"query_row:{index}"
            records_by_query.setdefault(group_id, []).append(record)

            if query_id and query_id not in seen_query_ids:
                seen_query_ids.add(query_id)
                query_ids.append(query_id)

        randomizer = random.Random(self.cfg.seed)
        randomizer.shuffle(query_ids)

        val_size = self._compute_target_val_size(query_ids)
        if val_size == 0:
            return (
                records,
                [],
                {
                    "target_val_queries": 0,
                    "selected_val_queries": 0,
                    "connected_components": len(records_by_query),
                },
            )

        val_query_ids = set(query_ids[:val_size])
        train_records = []
        val_records = []

        for group_id, group_records in records_by_query.items():
            if group_id in val_query_ids:
                val_records.extend(group_records)
            else:
                train_records.extend(group_records)

        return (
            train_records,
            val_records,
            {
                "target_val_queries": val_size,
                "selected_val_queries": len(val_query_ids),
                "connected_components": len(records_by_query),
            },
        )

    def _compute_target_val_size(self, query_ids):
        val_size = int(len(query_ids) * float(self.cfg.data.val_fraction))
        if float(self.cfg.data.val_fraction) > 0 and val_size == 0 and query_ids:
            val_size = 1
        return val_size

    def _empty_split_stats(self):
        return {
            "target_val_queries": 0,
            "selected_val_queries": 0,
            "connected_components": 0,
        }

    def _compute_class_weights(self, train_records):
        counts = Counter(record["label_id"] for record in train_records)
        class_counts = [int(counts.get(i, 0)) for i in range(NUM_CLASSES)]
        total = sum(class_counts)
        weights = []
        for count in class_counts:
            if count == 0:
                weights.append(0.0)
            else:
                weights.append(total / (NUM_CLASSES * count))
        self.class_counts = class_counts
        self.class_weights = weights

    def _build_dataset_stats(
        self,
        train_records,
        val_records,
        skipped_rows,
        skipped_unknown_label,
        split_stats,
    ):
        train_class_counts = Counter(record["raw_label"] for record in train_records)
        val_class_counts = Counter(record["raw_label"] for record in val_records)
        stats = {
            "train_rows": len(train_records),
            "val_rows": len(val_records),
            "train_queries": len({r["query_id"] for r in train_records}),
            "val_queries": len({r["query_id"] for r in val_records}),
            "skipped_rows": skipped_rows,
            "skipped_unknown_label": skipped_unknown_label,
            **split_stats,
        }
        for label in LABEL_TO_ID:
            stats[f"train_{label.lower()}_rows"] = int(train_class_counts.get(label, 0))
            stats[f"val_{label.lower()}_rows"] = int(val_class_counts.get(label, 0))
        for index, weight in enumerate(self.class_weights):
            stats[f"class_weight_{index}"] = float(weight)
        return stats
