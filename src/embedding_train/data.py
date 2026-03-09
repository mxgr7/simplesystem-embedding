import random

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from embedding_train.text import (
    build_template,
    clean_html_text,
    flatten_category_paths,
    normalize_text,
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
        self.query_template = build_template(cfg.data.query_template)
        self.offer_template = build_template(cfg.data.offer_template)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.model_name, use_fast=True
        )
        self.train_dataset = None
        self.val_dataset = None
        self.dataset_stats = {}

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        self.dataset_stats = {
            "train_rows": len(train_records),
            "val_rows": len(val_records),
            "train_queries": len({record["query_id"] for record in train_records}),
            "val_queries": len({record["query_id"] for record in val_records}),
            "skipped_rows": skipped_rows,
        }

        print("Loaded dataset:", self.dataset_stats)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg.data.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.cfg.data.batch_size),
            shuffle=False,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        query_texts = [item["query_text"] for item in batch]
        offer_texts = [item["offer_text"] for item in batch]

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

        return {
            "query_inputs": dict(query_inputs),
            "offer_inputs": dict(offer_inputs),
            "labels": torch.tensor(
                [item["label"] for item in batch], dtype=torch.float32
            ),
            "query_ids": [item["query_id"] for item in batch],
            "offer_ids": [item["offer_id"] for item in batch],
            "raw_labels": [item["raw_label"] for item in batch],
        }

    def _build_record(self, row):
        context = {}

        for key, value in row.items():
            context[key] = self._safe_value(value)

        context["query_term"] = normalize_text(context.get("query_term"))
        context["name"] = normalize_text(context.get("name"))
        context["manufacturer_name"] = normalize_text(context.get("manufacturer_name"))
        context["article_number"] = normalize_text(context.get("article_number"))
        context["category_text"] = flatten_category_paths(context.get("category_paths"))
        description = context.get("description")

        if self.cfg.data.clean_html:
            context["clean_description"] = clean_html_text(description)
        else:
            context["clean_description"] = normalize_text(description)

        query_text = normalize_text(self.query_template.render(**context))
        offer_text = normalize_text(self.offer_template.render(**context))

        if not query_text or not offer_text:
            return None

        return {
            "query_id": normalize_text(context.get("query_id")),
            "offer_id": normalize_text(context.get("offer_id_b64")),
            "query_text": query_text,
            "offer_text": offer_text,
            "label": 1.0
            if context.get("label") == self.cfg.data.positive_label
            else 0.0,
            "raw_label": normalize_text(context.get("label")),
        }

    def _safe_value(self, value):
        if value is None:
            return ""

        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass

        return value
