import hashlib
import json
import pickle
import random
import sys
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from lightning import LightningDataModule
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from embedding_train.batching import (
    HARD_NEGATIVE_LABEL,
    AnchorQueryBatchBuilder,
    AnchorQueryBatchDataset,
    RandomQueryPoolBuilder,
    build_batch_stats,
)
from embedding_train.rendering import RowTextRenderer, resolve_column_mapping

HARD_NEGATIVE_PROVENANCE = "hard_negative"
SEMI_HARD_NEGATIVE_PROVENANCE = "semi_hard_negative"
VALID_NEGATIVE_PROVENANCES = (
    HARD_NEGATIVE_PROVENANCE,
    SEMI_HARD_NEGATIVE_PROVENANCE,
)

VALID_TRAIN_BATCHING_MODES = {"random_pairs", "anchor_query", "random_query_pool"}
VALID_VAL_SPLIT_MODES = {"offer_connected_component", "query_id"}
PREPARED_RECORDS_CACHE_SCHEMA_VERSION = 1
RANDOM_QUERY_POOL_CACHE_SCHEMA_VERSION = 1
DEFAULT_PREPARED_RECORDS_CACHE_DIR = ".cache/prepared_dataset"


def build_setup_progress(total_rows):
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=24),
        TaskProgressColumn(),
        TextColumn("[dim]({task.completed:,.0f}/{task.total:,.0f} rows)"),
        TextColumn(
            "[dim]prepared={task.fields[prepared_rows]:,.0f} skipped={task.fields[skipped_rows]:,.0f}"
        ),
        TimeElapsedColumn(),
        console=Console(file=sys.stderr),
        transient=False,
    )


def update_setup_progress(
    progress, task_id, processed_rows, total_rows, prepared_rows, skipped_rows
):
    progress.update(
        task_id,
        completed=min(processed_rows, total_rows),
        prepared_rows=prepared_rows,
        skipped_rows=skipped_rows,
    )


def resolve_train_batching_mode(train_batching_mode):
    normalized = str(train_batching_mode).strip().lower()

    if normalized in VALID_TRAIN_BATCHING_MODES:
        return normalized

    choices = "|".join(sorted(VALID_TRAIN_BATCHING_MODES))
    raise ValueError(
        "Unsupported train batching mode: "
        f"{train_batching_mode}. Expected one of {choices}"
    )


def resolve_val_split_mode(val_split_mode):
    normalized = str(val_split_mode).strip().lower()

    if normalized in VALID_VAL_SPLIT_MODES:
        return normalized

    choices = "|".join(sorted(VALID_VAL_SPLIT_MODES))
    raise ValueError(
        f"Unsupported val split mode: {val_split_mode}. Expected one of {choices}"
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
        self.val_split_mode = resolve_val_split_mode(
            getattr(cfg.data, "val_split_mode", "query_id")
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
        self.hard_negative_records_by_query = {}
        self.semi_hard_negative_records_by_query = {}
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

        all_records, skipped_rows = self._load_or_prepare_records()
        train_records = []
        val_records = []

        if self.val_split_mode == "query_id":
            train_records, val_records, split_stats = self._split_records_by_query_id(
                all_records
            )
        else:
            (
                train_records,
                val_records,
                split_stats,
            ) = self._split_records_by_offer_connected_components(all_records)

        if not train_records and val_records:
            if self.val_split_mode == "query_id":
                raise ValueError(
                    "Train split is empty after query-id splitting. "
                    f"target_val_queries={split_stats['target_val_queries']} "
                    f"selected_val_queries={split_stats['selected_val_queries']}. "
                    "Set data.val_fraction=0.0 or reduce it below 1.0."
                )

            raise ValueError(
                "Train split is empty after offer-connected-component splitting. "
                f"target_val_queries={split_stats['target_val_queries']} "
                f"selected_val_queries={split_stats['selected_val_queries']} "
                f"connected_components={split_stats['connected_components']}. "
                "A non-zero data.val_fraction moved every connected component into "
                "validation. Set data.val_fraction=0.0 or provide data with more "
                "than one disconnected query-offer component."
            )

        self.train_dataset = PairDataset(train_records)
        self.val_dataset = PairDataset(val_records)
        self._build_train_metadata(train_records)
        train_offer_ids = {record["offer_id"] for record in train_records}
        val_offer_ids = {record["offer_id"] for record in val_records}
        total_query_count = len(
            {record["query_id"] for record in train_records + val_records}
        )
        total_offer_count = len(train_offer_ids | val_offer_ids)
        total_record_count = len(train_records) + len(val_records)
        train_positive_rows = sum(
            1 for record in train_records if record["label"] > 0.5
        )
        val_positive_rows = sum(1 for record in val_records if record["label"] > 0.5)
        train_negative_rows = len(train_records) - train_positive_rows
        val_negative_rows = len(val_records) - val_positive_rows
        self.dataset_stats = {
            "train_rows": len(train_records),
            "val_rows": len(val_records),
            "train_queries": len({record["query_id"] for record in train_records}),
            "val_queries": len({record["query_id"] for record in val_records}),
            "train_offers": len(train_offer_ids),
            "val_offers": len(val_offer_ids),
            "train_positive_rows": train_positive_rows,
            "val_positive_rows": val_positive_rows,
            "train_negative_rows": train_negative_rows,
            "val_negative_rows": val_negative_rows,
            "val_query_share": (
                len({record["query_id"] for record in val_records}) / total_query_count
                if total_query_count
                else 0.0
            ),
            "val_offer_share": (
                len(val_offer_ids) / total_offer_count if total_offer_count else 0.0
            ),
            "val_row_share": (
                len(val_records) / total_record_count if total_record_count else 0.0
            ),
            "train_positive_rate": (
                train_positive_rows / len(train_records) if train_records else 0.0
            ),
            "val_positive_rate": (
                val_positive_rows / len(val_records) if val_records else 0.0
            ),
            "shared_offers_between_train_and_val": len(train_offer_ids & val_offer_ids),
            "skipped_rows": skipped_rows,
            "eligible_train_queries": len(self.eligible_query_ids),
            "hard_negative_queries": len(self.hard_negative_records_by_query),
            "hard_negative_records": sum(
                len(records)
                for records in self.hard_negative_records_by_query.values()
            ),
            "semi_hard_negative_queries": len(
                self.semi_hard_negative_records_by_query
            ),
            "semi_hard_negative_records": sum(
                len(records)
                for records in self.semi_hard_negative_records_by_query.values()
            ),
            "val_split_mode": self.val_split_mode,
            **split_stats,
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

        if self.train_batching_mode == "random_query_pool":
            train_dataset = cast(Dataset, self._build_random_query_pool_train_dataset())
            return DataLoader(
                train_dataset,
                batch_size=int(self.cfg.data.batch_size),
                shuffle=True,
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
            "query_texts": query_texts,
            "offer_texts": offer_texts,
        }

        if (
            batch_stats is None
            and self.train_batching_mode == "random_query_pool"
            and bool(self.cfg.data.log_batch_stats)
        ):
            batch_stats = build_batch_stats(batch_records)

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
            hard_negative_records_by_query=self.hard_negative_records_by_query,
            semi_hard_negative_records_by_query=self.semi_hard_negative_records_by_query,
        )
        train_rows = len(self.train_dataset)
        batch_size = int(self.cfg.data.batch_size)
        batches_per_epoch = max(1, (train_rows + batch_size - 1) // batch_size)
        return AnchorQueryBatchDataset(batch_builder, batches_per_epoch)

    def _build_random_query_pool_train_dataset(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized. Call setup() first.")

        cache_path = self._random_query_pool_cache_path()

        if cache_path is not None and cache_path.exists():
            try:
                with cache_path.open("rb") as handle:
                    pool_records = pickle.load(handle)
                print(
                    f"Loaded random query pool from cache: {cache_path} "
                    f"({len(pool_records)} records)",
                    file=sys.stderr,
                )
                return PairDataset(pool_records)
            except Exception as exc:
                print(
                    f"Failed to load random query pool cache at {cache_path}: {exc}. "
                    "Rebuilding.",
                    file=sys.stderr,
                )

        pool_builder = RandomQueryPoolBuilder(
            positive_records_by_query=self.positive_records_by_query,
            negative_records_by_query=self.negative_records_by_query,
            eligible_query_ids=self.eligible_query_ids,
            synthetic_negative_offer_pool=self.synthetic_negative_offer_pool,
            n_pos_samples_per_query=int(self.cfg.data.n_pos_samples_per_query),
            n_neg_samples_per_query=int(self.cfg.data.n_neg_samples_per_query),
            seed=int(self.cfg.seed),
            hard_negative_records_by_query=self.hard_negative_records_by_query,
            semi_hard_negative_records_by_query=self.semi_hard_negative_records_by_query,
        )
        pool_records = pool_builder.build_pool()

        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                with tmp_path.open("wb") as handle:
                    pickle.dump(
                        pool_records, handle, protocol=pickle.HIGHEST_PROTOCOL
                    )
                tmp_path.replace(cache_path)
                print(
                    f"Cached random query pool to: {cache_path}",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(
                    f"Failed to write random query pool cache to {cache_path}: {exc}",
                    file=sys.stderr,
                )

        return PairDataset(pool_records)

    def _random_query_pool_cache_path(self):
        data_cfg = self.cfg.data

        if not bool(data_cfg.get("cache_prepared_dataset", True)):
            return None

        source_path = Path(str(data_cfg.path))
        try:
            source_stat = source_path.stat()
        except OSError:
            return None

        def sidecar_stat(path_value):
            if not path_value:
                return None, None
            sidecar_path = Path(str(path_value))
            try:
                sidecar_stat_result = sidecar_path.stat()
            except OSError:
                return None, None
            return sidecar_path, sidecar_stat_result

        hard_path, hard_stat = sidecar_stat(
            getattr(data_cfg, "hard_negatives_path", None)
        )
        semi_hard_path, semi_hard_stat = sidecar_stat(
            getattr(data_cfg, "semi_hard_negatives_path", None)
        )

        cache_dir_cfg = data_cfg.get(
            "prepare_cache_dir", DEFAULT_PREPARED_RECORDS_CACHE_DIR
        )
        cache_dir = Path(str(cache_dir_cfg))

        key_payload = {
            "schema_version": RANDOM_QUERY_POOL_CACHE_SCHEMA_VERSION,
            "source_path": str(source_path.resolve()),
            "source_mtime_ns": source_stat.st_mtime_ns,
            "source_size": source_stat.st_size,
            "hard_negatives_path": (
                str(hard_path.resolve()) if hard_path is not None else None
            ),
            "hard_negatives_mtime_ns": (
                hard_stat.st_mtime_ns if hard_stat is not None else None
            ),
            "hard_negatives_size": (
                hard_stat.st_size if hard_stat is not None else None
            ),
            "semi_hard_negatives_path": (
                str(semi_hard_path.resolve())
                if semi_hard_path is not None
                else None
            ),
            "semi_hard_negatives_mtime_ns": (
                semi_hard_stat.st_mtime_ns if semi_hard_stat is not None else None
            ),
            "semi_hard_negatives_size": (
                semi_hard_stat.st_size if semi_hard_stat is not None else None
            ),
            "limit_rows": data_cfg.limit_rows,
            "column_mapping": resolve_column_mapping(data_cfg),
            "query_template": str(data_cfg.query_template),
            "offer_template": str(data_cfg.offer_template),
            "positive_label": str(data_cfg.positive_label),
            "clean_html": bool(data_cfg.clean_html),
            "val_fraction": float(data_cfg.val_fraction),
            "val_split_mode": str(data_cfg.val_split_mode),
            "seed": int(self.cfg.seed),
            "n_pos_samples_per_query": int(data_cfg.n_pos_samples_per_query),
            "n_neg_samples_per_query": int(data_cfg.n_neg_samples_per_query),
        }
        serialized = json.dumps(key_payload, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        return cache_dir / f"random_query_pool-{digest}.pkl"

    def _load_or_prepare_records(self):
        cache_path = self._prepared_records_cache_path()

        if cache_path is not None and cache_path.exists():
            try:
                with cache_path.open("rb") as handle:
                    cached = pickle.load(handle)
                print(
                    f"Loaded prepared dataset from cache: {cache_path} "
                    f"({len(cached['records'])} records, "
                    f"{cached['skipped_rows']} skipped)",
                    file=sys.stderr,
                )
                return cached["records"], cached["skipped_rows"]
            except Exception as exc:
                print(
                    f"Failed to load prepared dataset cache at {cache_path}: {exc}. "
                    "Rebuilding.",
                    file=sys.stderr,
                )

        all_records, skipped_rows = self._prepare_records()

        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                with tmp_path.open("wb") as handle:
                    pickle.dump(
                        {"records": all_records, "skipped_rows": skipped_rows},
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                tmp_path.replace(cache_path)
                print(
                    f"Cached prepared dataset to: {cache_path}",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(
                    f"Failed to write prepared dataset cache to {cache_path}: {exc}",
                    file=sys.stderr,
                )

        return all_records, skipped_rows

    def _prepare_records(self):
        frame = pd.read_parquet(self.cfg.data.path)

        if "_rn" in frame.columns:
            frame = frame.drop(columns=["_rn"])

        if self.cfg.data.limit_rows:
            frame = frame.head(int(self.cfg.data.limit_rows))

        columns = list(frame.columns)
        all_records = []
        skipped_rows = 0
        total_rows = len(frame)

        with build_setup_progress(total_rows) as progress:
            task_id = progress.add_task(
                "Preparing dataset",
                total=max(1, total_rows),
                prepared_rows=0,
                skipped_rows=0,
            )

            for processed_rows, values in enumerate(
                frame.itertuples(index=False, name=None), start=1
            ):
                row = dict(zip(columns, values))
                record = self._build_record(row)
                if record is None:
                    skipped_rows += 1
                else:
                    all_records.append(record)

                update_setup_progress(
                    progress,
                    task_id,
                    processed_rows,
                    max(1, total_rows),
                    len(all_records),
                    skipped_rows,
                )

        return all_records, skipped_rows

    def _prepared_records_cache_path(self):
        data_cfg = self.cfg.data

        if not bool(data_cfg.get("cache_prepared_dataset", True)):
            return None

        source_path = Path(str(data_cfg.path))
        try:
            stat_result = source_path.stat()
        except OSError:
            return None

        cache_dir_cfg = data_cfg.get(
            "prepare_cache_dir", DEFAULT_PREPARED_RECORDS_CACHE_DIR
        )
        cache_dir = Path(str(cache_dir_cfg))

        key_payload = {
            "schema_version": PREPARED_RECORDS_CACHE_SCHEMA_VERSION,
            "source_path": str(source_path.resolve()),
            "source_mtime_ns": stat_result.st_mtime_ns,
            "source_size": stat_result.st_size,
            "limit_rows": data_cfg.limit_rows,
            "column_mapping": resolve_column_mapping(data_cfg),
            "query_template": str(data_cfg.query_template),
            "offer_template": str(data_cfg.offer_template),
            "positive_label": str(data_cfg.positive_label),
            "clean_html": bool(data_cfg.clean_html),
        }
        serialized = json.dumps(key_payload, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        return cache_dir / f"prepared-{digest}.pkl"

    def _build_record(self, row):
        return self.row_renderer.build_training_record(row)

    def _empty_split_stats(self):
        return {
            "target_val_queries": 0,
            "selected_val_queries": 0,
            "connected_components": 0,
            "train_connected_components": 0,
            "val_connected_components": 0,
        }

    def _compute_target_val_size(self, query_ids):
        val_size = int(len(query_ids) * float(self.cfg.data.val_fraction))
        if float(self.cfg.data.val_fraction) > 0 and val_size == 0 and query_ids:
            val_size = 1

        return val_size

    def _split_records_by_query_id(self, records):
        if not records:
            return [], [], self._empty_split_stats()

        records_by_query = {}
        query_ids = []
        seen_query_ids = set()
        train_records = []
        val_records = []

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
                    "train_connected_components": len(records_by_query),
                    "val_connected_components": 0,
                },
            )

        val_query_ids = set(query_ids[:val_size])

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
                "train_connected_components": (
                    len(records_by_query) - len(val_query_ids)
                ),
                "val_connected_components": len(val_query_ids),
            },
        )

    def _split_records_by_offer_connected_components(self, records):
        if not records:
            return [], [], self._empty_split_stats()

        query_nodes = {}
        parent = {}
        component_records = {}
        component_query_ids = {}
        query_ids = []
        record_query_nodes = []
        seen_query_ids = set()
        train_records = []
        val_records = []

        def ensure_node(node):
            if node not in parent:
                parent[node] = node

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left, right):
            left_root = find(left)
            right_root = find(right)

            if left_root == right_root:
                return left_root

            parent[right_root] = left_root
            return left_root

        for index, record in enumerate(records):
            query_id = record["query_id"]
            offer_id = record["offer_id"]

            if query_id:
                query_node = query_nodes.get(query_id)
                if query_node is None:
                    query_node = f"query:{query_id}"
                    query_nodes[query_id] = query_node
            else:
                query_node = f"query_row:{index}"

            ensure_node(query_node)
            record_query_nodes.append(query_node)

            if query_id and query_id not in seen_query_ids:
                seen_query_ids.add(query_id)
                query_ids.append(query_id)

            if offer_id:
                offer_node = f"offer:{offer_id}"
                ensure_node(offer_node)
                union(query_node, offer_node)

        for record, query_node in zip(records, record_query_nodes):
            component_id = find(query_node)
            component_records.setdefault(component_id, []).append(record)

            if record["query_id"]:
                component_query_ids.setdefault(component_id, set()).add(
                    record["query_id"]
                )

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
                    "connected_components": len(component_records),
                    "train_connected_components": len(component_records),
                    "val_connected_components": 0,
                },
            )

        val_component_ids = set()
        val_query_count = 0

        for query_id in query_ids:
            component_id = find(query_nodes[query_id])

            if component_id in val_component_ids:
                continue

            val_component_ids.add(component_id)
            val_query_count += len(component_query_ids.get(component_id, set()))

            if val_query_count >= val_size:
                break

        for component_id in component_records:
            if component_id in val_component_ids:
                val_records.extend(component_records[component_id])
            else:
                train_records.extend(component_records[component_id])

        return (
            train_records,
            val_records,
            {
                "target_val_queries": val_size,
                "selected_val_queries": val_query_count,
                "connected_components": len(component_records),
                "train_connected_components": (
                    len(component_records) - len(val_component_ids)
                ),
                "val_connected_components": len(val_component_ids),
            },
        )

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
        (
            self.hard_negative_records_by_query,
            self.semi_hard_negative_records_by_query,
        ) = self._load_mined_negatives(positive_records_by_query)

        if (
            self.train_batching_mode in {"anchor_query", "random_query_pool"}
            and not self.eligible_query_ids
        ):
            raise ValueError(
                "No train query has at least "
                f"n_pos_samples_per_query={n_pos_samples_per_query} positive rows"
            )

    def _load_mined_negatives(self, positive_records_by_query):
        hard_negatives_path = getattr(self.cfg.data, "hard_negatives_path", None)
        semi_hard_negatives_path = getattr(
            self.cfg.data, "semi_hard_negatives_path", None
        )

        positive_offer_ids_by_query = {
            query_id: {r["offer_id"] for r in records}
            for query_id, records in positive_records_by_query.items()
        }

        hard_negative_records_by_query = {}
        semi_hard_negative_records_by_query = {}

        self._load_negative_sidecar(
            hard_negatives_path,
            positive_offer_ids_by_query,
            hard_negative_records_by_query,
            semi_hard_negative_records_by_query,
            default_provenance=HARD_NEGATIVE_PROVENANCE,
            label="hard negatives",
        )
        self._load_negative_sidecar(
            semi_hard_negatives_path,
            positive_offer_ids_by_query,
            hard_negative_records_by_query,
            semi_hard_negative_records_by_query,
            default_provenance=SEMI_HARD_NEGATIVE_PROVENANCE,
            label="semi-hard negatives",
        )

        return hard_negative_records_by_query, semi_hard_negative_records_by_query

    def _load_negative_sidecar(
        self,
        path,
        positive_offer_ids_by_query,
        hard_negative_records_by_query,
        semi_hard_negative_records_by_query,
        default_provenance,
        label,
    ):
        if not path:
            return

        frame = pd.read_parquet(path)
        required_columns = {"query_id", "offer_id", "offer_text"}
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(
                f"{label.capitalize()} file is missing columns: "
                f"{', '.join(sorted(missing))}"
            )

        has_provenance_column = "provenance" in frame.columns
        loaded_hard = 0
        loaded_semi_hard = 0
        excluded_count = 0
        skipped_unknown_provenance = 0

        for row in frame.itertuples(index=False):
            query_id = str(getattr(row, "query_id", "") or "").strip()
            offer_id = str(getattr(row, "offer_id", "") or "").strip()
            offer_text = str(getattr(row, "offer_text", "") or "").strip()

            if not query_id or not offer_id or not offer_text:
                continue

            if offer_id in positive_offer_ids_by_query.get(query_id, set()):
                excluded_count += 1
                continue

            if has_provenance_column:
                row_provenance = str(
                    getattr(row, "provenance", "") or default_provenance
                ).strip() or default_provenance
            else:
                row_provenance = default_provenance

            if row_provenance == HARD_NEGATIVE_PROVENANCE:
                target = hard_negative_records_by_query
            elif row_provenance == SEMI_HARD_NEGATIVE_PROVENANCE:
                target = semi_hard_negative_records_by_query
            else:
                skipped_unknown_provenance += 1
                continue

            target.setdefault(query_id, []).append(
                {"offer_id": offer_id, "offer_text": offer_text}
            )
            if target is hard_negative_records_by_query:
                loaded_hard += 1
            else:
                loaded_semi_hard += 1

        print(
            f"Loaded {label} from {path}: "
            f"hard={loaded_hard} semi_hard={loaded_semi_hard} "
            f"(excluded {excluded_count} positives, "
            f"skipped {skipped_unknown_provenance} unknown provenance)"
        )
