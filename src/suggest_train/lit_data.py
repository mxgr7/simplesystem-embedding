"""Datasets and DataModule for the suggest LM.

Two training variants:

* ``variant='a'`` — targets-only. Each row is a single target query;
  sampling weights are per-row event counts. Sequence: ``<s> target </s>``.
  Loss on every non-pad position.
* ``variant='b'`` — prefix-conditioned. Each row is a ``(prefix, target)``
  pair; sampling weights are per-pair event counts. Sequence:
  ``<s> prefix <sep> target </s>``. Loss only on the completion tokens
  (positions strictly after ``<sep>``).
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import PAIRS_DIR, TARGETS_DIR, TOKENIZER_DIR
from .tokenizer import Tokenizer

LABEL_IGNORE = -100


def _load_targets_table(
    targets_dir: Path, split: str
) -> tuple[list[str], np.ndarray]:
    glob = f"{targets_dir}/split={split}/**/*.parquet"
    con = duckdb.connect(":memory:")
    df = con.execute(
        f"""
        SELECT target, sum(count)::BIGINT AS count
        FROM read_parquet('{glob}', hive_partitioning = TRUE)
        GROUP BY target
        ORDER BY count DESC
        """
    ).fetchdf()
    targets = df["target"].tolist()
    counts = df["count"].to_numpy(dtype=np.int64)
    return targets, counts


def _tokenize_corpus(
    tokenizer: Tokenizer, targets: list[str], max_seq_len: int
) -> list[list[int]]:
    """Pre-tokenize each target and prepend BOS / append EOS."""
    out: list[list[int]] = []
    bos, eos = tokenizer.bos_id, tokenizer.eos_id
    body_cap = max_seq_len - 2  # leave room for BOS + EOS
    for target in targets:
        body = tokenizer.sp.encode(target, out_type=int)
        if len(body) > body_cap:
            body = body[:body_cap]
        out.append([bos, *body, eos])
    return out


class TargetsDataset(Dataset):
    """In-memory dataset of pre-tokenized target sequences (option a)."""

    def __init__(self, encoded: list[list[int]]) -> None:
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> list[int]:
        return self.encoded[idx]


def collate_lm(batch: list[list[int]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(seq) for seq in batch)
    b = len(batch)
    input_ids = torch.full((b, max_len - 1), pad_id, dtype=torch.long)
    targets = torch.full((b, max_len - 1), LABEL_IGNORE, dtype=torch.long)
    attention_mask = torch.zeros((b, max_len - 1), dtype=torch.long)
    for i, seq in enumerate(batch):
        n = len(seq)
        # input = seq[:-1], target = seq[1:]
        in_len = n - 1
        input_ids[i, :in_len] = torch.tensor(seq[:-1], dtype=torch.long)
        targets[i, :in_len] = torch.tensor(seq[1:], dtype=torch.long)
        attention_mask[i, :in_len] = 1
    return {
        "input_ids": input_ids,
        "labels": targets,
        "attention_mask": attention_mask,
    }


def _load_pairs_table(
    pairs_dir: Path, split: str
) -> tuple[list[str], list[str], np.ndarray]:
    glob = f"{pairs_dir}/split={split}/**/*.parquet"
    con = duckdb.connect(":memory:")
    df = con.execute(
        f"""
        SELECT prefix, target, sum(count)::BIGINT AS count
        FROM read_parquet('{glob}', hive_partitioning = TRUE)
        GROUP BY prefix, target
        ORDER BY count DESC
        """
    ).fetchdf()
    return (
        df["prefix"].tolist(),
        df["target"].tolist(),
        df["count"].to_numpy(dtype=np.int64),
    )


def _tokenize_pairs(
    tokenizer: Tokenizer,
    prefixes: list[str],
    targets: list[str],
    max_seq_len: int,
) -> list[tuple[list[int], int]]:
    """Tokenize each (prefix, target) into the option-b sequence.

    Returns a list of ``(seq, prefix_token_count)`` pairs where ``seq`` is
    ``[BOS, *prefix_ids, SEP, *target_ids, EOS]`` truncated to fit
    ``max_seq_len``. ``prefix_token_count`` is ``len(prefix_ids)``.
    """
    bos, eos, sep = tokenizer.bos_id, tokenizer.eos_id, tokenizer.sep_id
    out: list[tuple[list[int], int]] = []
    for prefix, target in zip(prefixes, targets):
        p_ids = tokenizer.sp.encode(prefix, out_type=int)
        t_ids = tokenizer.sp.encode(target, out_type=int)
        # Reserve BOS + SEP + EOS = 3 specials; budget rest.
        budget = max_seq_len - 3
        if len(p_ids) + len(t_ids) > budget:
            # Prefer keeping the target whole; truncate prefix from the
            # left (rare, but possible for very long search strings).
            t_keep = min(len(t_ids), max(1, budget - 1))
            p_keep = max(0, budget - t_keep)
            p_ids = p_ids[:p_keep]
            t_ids = t_ids[:t_keep]
        seq = [bos, *p_ids, sep, *t_ids, eos]
        out.append((seq, len(p_ids)))
    return out


class PairsDataset(Dataset):
    """In-memory dataset of pre-tokenized (prefix, target) sequences."""

    def __init__(self, encoded: list[tuple[list[int], int]]) -> None:
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.encoded[idx]


def collate_pairs(
    batch: list[tuple[list[int], int]], pad_id: int
) -> dict[str, torch.Tensor]:
    """Collate option-b batches with completion-only loss masking."""
    max_len = max(len(seq) for seq, _ in batch)
    b = len(batch)
    input_ids = torch.full((b, max_len - 1), pad_id, dtype=torch.long)
    labels = torch.full((b, max_len - 1), LABEL_IGNORE, dtype=torch.long)
    attention_mask = torch.zeros((b, max_len - 1), dtype=torch.long)
    for i, (seq, p_len) in enumerate(batch):
        n = len(seq)
        in_len = n - 1
        input_ids[i, :in_len] = torch.tensor(seq[:-1], dtype=torch.long)
        attention_mask[i, :in_len] = 1
        # labels = seq[1:]. Positions 0..p_len index into [p1..p_pn, SEP];
        # we mask those out and keep labels[p_len + 1:] = [t1, ..., EOS].
        full_labels = torch.tensor(seq[1:], dtype=torch.long)
        keep_from = p_len + 1
        labels[i, keep_from:in_len] = full_labels[keep_from:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


class SuggestLMDataModule(LightningDataModule):
    def __init__(
        self,
        targets_dir: Path = TARGETS_DIR,
        tokenizer_dir: Path = TOKENIZER_DIR,
        max_seq_len: int = 64,
        batch_size: int = 256,
        val_batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_samples_per_epoch: int | None = None,
        seed: int = 0,
        variant: str = "a",
        pairs_dir: Path = PAIRS_DIR,
    ) -> None:
        super().__init__()
        self.targets_dir = Path(targets_dir)
        self.pairs_dir = Path(pairs_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.max_seq_len = int(max_seq_len)
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.train_samples_per_epoch = train_samples_per_epoch
        self.seed = int(seed)
        self.variant = str(variant).lower()
        if self.variant not in ("a", "b"):
            raise ValueError(f"Unknown variant {self.variant!r}, expected a/b.")

        self.tokenizer: Tokenizer | None = None
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.train_weights: np.ndarray | None = None
        self.dataset_stats: dict[str, float] = {}

    def _setup_a(self) -> None:
        t0 = time.time()
        train_targets, train_counts = _load_targets_table(
            self.targets_dir, "train"
        )
        train_encoded = _tokenize_corpus(
            self.tokenizer, train_targets, self.max_seq_len
        )
        self.train_dataset = TargetsDataset(train_encoded)
        self.train_weights = train_counts.astype(np.float64)
        print(
            f"[lit_data] train: {len(train_encoded):,} unique targets, "
            f"{int(train_counts.sum()):,} weighted events "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )
        t0 = time.time()
        val_targets, val_counts = _load_targets_table(
            self.targets_dir, "eval"
        )
        val_encoded = _tokenize_corpus(
            self.tokenizer, val_targets, self.max_seq_len
        )
        self.val_dataset = TargetsDataset(val_encoded)
        print(
            f"[lit_data] val:   {len(val_encoded):,} unique targets, "
            f"{int(val_counts.sum()):,} events "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )

    def _setup_b(self) -> None:
        t0 = time.time()
        train_pref, train_tgt, train_counts = _load_pairs_table(
            self.pairs_dir, "train"
        )
        train_encoded = _tokenize_pairs(
            self.tokenizer, train_pref, train_tgt, self.max_seq_len
        )
        self.train_dataset = PairsDataset(train_encoded)
        self.train_weights = train_counts.astype(np.float64)
        print(
            f"[lit_data] train: {len(train_encoded):,} unique pairs, "
            f"{int(train_counts.sum()):,} weighted events "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )
        t0 = time.time()
        val_pref, val_tgt, val_counts = _load_pairs_table(
            self.pairs_dir, "eval"
        )
        val_encoded = _tokenize_pairs(
            self.tokenizer, val_pref, val_tgt, self.max_seq_len
        )
        self.val_dataset = PairsDataset(val_encoded)
        print(
            f"[lit_data] val:   {len(val_encoded):,} unique pairs, "
            f"{int(val_counts.sum()):,} events "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )

    def setup(self, stage: str | None = None) -> None:
        if self.tokenizer is None:
            self.tokenizer = Tokenizer.load(self.tokenizer_dir / "spm.model")

        if self.train_dataset is None:
            if self.variant == "a":
                self._setup_a()
            else:
                self._setup_b()

        if self.variant == "a":
            token_lens = np.array(
                [len(seq) for seq in self.train_dataset.encoded],
                dtype=np.int64,
            )
        else:
            token_lens = np.array(
                [len(seq) for seq, _ in self.train_dataset.encoded],
                dtype=np.int64,
            )
        self.dataset_stats = {
            f"train_unique_{'targets' if self.variant=='a' else 'pairs'}":
                float(len(self.train_dataset)),
            f"val_unique_{'targets' if self.variant=='a' else 'pairs'}":
                float(len(self.val_dataset)),
            "train_total_events": float(self.train_weights.sum()),
            "train_seq_len_mean": float(token_lens.mean()),
            "train_seq_len_p95": float(np.percentile(token_lens, 95)),
            "train_seq_len_max": float(token_lens.max()),
        }

    def _collate(self):
        pad_id = self.tokenizer.pad_id
        if self.variant == "a":
            return lambda b: collate_lm(b, pad_id)
        return lambda b: collate_pairs(b, pad_id)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None or self.train_weights is None:
            raise RuntimeError("setup() has not been called")
        n_samples = (
            self.train_samples_per_epoch
            if self.train_samples_per_epoch
            else len(self.train_dataset)
        )
        generator = torch.Generator().manual_seed(self.seed)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(self.train_weights, dtype=torch.double),
            num_samples=int(n_samples),
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate(),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("setup() has not been called")
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate(),
        )
