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

import random
import string
import time
from pathlib import Path

import duckdb
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

from .data import PAIRS_DIR, TARGETS_DIR, TOKENIZER_DIR
from .tokenizer import Tokenizer

LABEL_IGNORE = -100

# Operations applied by the synthetic-noise dataset. Calibrated to produce a
# single edit per call — we want the model to see realistic single-typo
# perturbations, not garbled strings.
_NOISE_OPS = ("delete", "insert", "substitute", "swap")
_NOISE_INSERT_ALPHABET = string.ascii_lowercase + "äöüß "
_NOISE_SUB_ALPHABET = string.ascii_lowercase + "äöüß"


def _perturb_one(s: str, rng: random.Random) -> str:
    """Apply a single random char-level edit to ``s``. Used for synthetic
    typo augmentation when the real session-mined corpus is exhausted."""
    n = len(s)
    if n == 0:
        return s
    candidates = ["insert"]
    if n >= 1:
        candidates += ["delete", "substitute"]
    if n >= 2:
        candidates.append("swap")
    op = rng.choice(candidates)
    if op == "delete":
        i = rng.randrange(n)
        return s[:i] + s[i + 1 :]
    if op == "insert":
        i = rng.randrange(n + 1)
        return s[:i] + rng.choice(_NOISE_INSERT_ALPHABET) + s[i:]
    if op == "substitute":
        i = rng.randrange(n)
        return s[:i] + rng.choice(_NOISE_SUB_ALPHABET) + s[i + 1 :]
    # swap
    i = rng.randrange(n - 1)
    return s[:i] + s[i + 1] + s[i] + s[i + 2 :]


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


def _tokenize_one_pair(
    tokenizer: Tokenizer, prefix: str, target: str, max_seq_len: int
) -> tuple[list[int], int]:
    """Encode one (prefix, target) into the variant-b sequence
    ``[BOS, *prefix_ids, SEP, *target_ids, EOS]``, truncating from the left
    of the prefix if the budget is tight."""
    bos, eos, sep = tokenizer.bos_id, tokenizer.eos_id, tokenizer.sep_id
    p_ids = tokenizer.sp.encode(prefix, out_type=int)
    t_ids = tokenizer.sp.encode(target, out_type=int)
    budget = max_seq_len - 3  # BOS + SEP + EOS
    if len(p_ids) + len(t_ids) > budget:
        t_keep = min(len(t_ids), max(1, budget - 1))
        p_keep = max(0, budget - t_keep)
        p_ids = p_ids[:p_keep]
        t_ids = t_ids[:t_keep]
    seq = [bos, *p_ids, sep, *t_ids, eos]
    return seq, len(p_ids)


def _tokenize_pairs(
    tokenizer: Tokenizer,
    prefixes: list[str],
    targets: list[str],
    max_seq_len: int,
) -> list[tuple[list[int], int]]:
    """Tokenize each (prefix, target) into the option-b sequence."""
    return [
        _tokenize_one_pair(tokenizer, p, t, max_seq_len)
        for p, t in zip(prefixes, targets)
    ]


class PairsDataset(Dataset):
    """In-memory dataset of pre-tokenized (prefix, target) sequences."""

    def __init__(self, encoded: list[tuple[list[int], int]]) -> None:
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.encoded[idx]


class NoisyPrefixDataset(Dataset):
    """Variant-b dataset that perturbs the prefix at sample time.

    We hold raw (prefix, target) strings and re-tokenize on every call so
    each access can apply a different random edit. Re-tokenization adds a
    few microseconds per sample but only fires for the synthetic-noise
    fraction of training (typically 10%), so the cost is negligible.

    The rng is initialized lazily per worker so multi-worker DataLoaders
    don't all see the same noise sequence."""

    def __init__(
        self,
        prefixes: list[str],
        targets: list[str],
        tokenizer: Tokenizer,
        max_seq_len: int,
        seed: int = 0,
    ) -> None:
        if len(prefixes) != len(targets):
            raise ValueError("prefixes/targets length mismatch")
        self.prefixes = prefixes
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self._seed_base = int(seed)
        self._rng: random.Random | None = None

    def _rng_local(self) -> random.Random:
        if self._rng is None:
            wid = torch.utils.data.get_worker_info()
            wid_n = wid.id if wid else 0
            # Mix worker id with the configured seed so each worker sees a
            # different but reproducible noise sequence.
            self._rng = random.Random(self._seed_base + wid_n * 7919 + 1)
        return self._rng

    def __len__(self) -> int:
        return len(self.prefixes)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        rng = self._rng_local()
        prefix = _perturb_one(self.prefixes[idx], rng)
        return _tokenize_one_pair(
            self.tokenizer, prefix, self.targets[idx], self.max_seq_len
        )


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
        typo_pairs_dir: Path | None = None,
        p_typo: float = 0.0,
        p_synthetic: float = 0.0,
    ) -> None:
        super().__init__()
        self.targets_dir = Path(targets_dir)
        self.pairs_dir = Path(pairs_dir)
        self.typo_pairs_dir = Path(typo_pairs_dir) if typo_pairs_dir else None
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

        # Source-mix probabilities for variant 'b'. Together with the
        # implicit clean fraction (1 - p_typo - p_synthetic) they describe
        # the probability that a sampled training row comes from each
        # source. Validated below.
        self.p_typo = float(p_typo)
        self.p_synthetic = float(p_synthetic)
        if self.p_typo < 0 or self.p_synthetic < 0:
            raise ValueError("p_typo and p_synthetic must be >= 0")
        if self.p_typo + self.p_synthetic > 1.0 + 1e-6:
            raise ValueError(
                f"p_typo + p_synthetic must be <= 1.0; got "
                f"{self.p_typo + self.p_synthetic}"
            )
        if self.variant == "a" and (self.p_typo > 0 or self.p_synthetic > 0):
            raise ValueError(
                "p_typo / p_synthetic only apply to variant='b'"
            )
        if self.p_typo > 0 and self.typo_pairs_dir is None:
            raise ValueError("p_typo > 0 requires typo_pairs_dir")

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
        # 1) Clean (prefix, target) pairs — always loaded.
        t0 = time.time()
        clean_pref, clean_tgt, clean_counts = _load_pairs_table(
            self.pairs_dir, "train"
        )
        clean_encoded = _tokenize_pairs(
            self.tokenizer, clean_pref, clean_tgt, self.max_seq_len
        )
        clean_ds = PairsDataset(clean_encoded)
        print(
            f"[lit_data] clean train: {len(clean_encoded):,} pairs, "
            f"{int(clean_counts.sum()):,} weighted events "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

        # 2) Real typo→correction pairs — optional.
        typo_ds: Dataset | None = None
        typo_counts: np.ndarray | None = None
        if self.typo_pairs_dir is not None and self.p_typo > 0:
            t0 = time.time()
            typo_pref, typo_tgt, typo_counts = _load_pairs_table(
                self.typo_pairs_dir, "train"
            )
            typo_encoded = _tokenize_pairs(
                self.tokenizer, typo_pref, typo_tgt, self.max_seq_len
            )
            typo_ds = PairsDataset(typo_encoded)
            print(
                f"[lit_data] typo  train: {len(typo_encoded):,} pairs, "
                f"{int(typo_counts.sum()):,} weighted events "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

        # 3) Synthetic-noise wrapper over the clean strings — optional.
        noise_ds: NoisyPrefixDataset | None = None
        if self.p_synthetic > 0:
            noise_ds = NoisyPrefixDataset(
                prefixes=clean_pref,
                targets=clean_tgt,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                seed=self.seed,
            )

        # Combine into one dataset + per-row weights normalized so each
        # source contributes its target probability mass.
        sources: list[Dataset] = [clean_ds]
        weight_blocks: list[np.ndarray] = []

        p_clean = max(0.0, 1.0 - self.p_typo - self.p_synthetic)
        clean_w = clean_counts.astype(np.float64)
        clean_total = float(clean_w.sum()) or 1.0
        weight_blocks.append(clean_w * (p_clean / clean_total))

        if typo_ds is not None and typo_counts is not None:
            sources.append(typo_ds)
            typo_w = typo_counts.astype(np.float64)
            typo_total = float(typo_w.sum()) or 1.0
            weight_blocks.append(typo_w * (self.p_typo / typo_total))

        if noise_ds is not None:
            sources.append(noise_ds)
            # Reuse clean count distribution so noise lands on popular rows
            # in proportion to the clean signal — same shape, same seeding.
            weight_blocks.append(clean_w * (self.p_synthetic / clean_total))

        if len(sources) == 1:
            self.train_dataset = clean_ds
        else:
            self.train_dataset = ConcatDataset(sources)
        self.train_weights = np.concatenate(weight_blocks).astype(np.float64)
        self._train_source_lens = tuple(len(s) for s in sources)
        self._train_source_p = (p_clean, self.p_typo, self.p_synthetic)

        # Eval: clean pairs only. Typo eval gets its own slice if/when we
        # want a dedicated metric — wire it in once the eval harness exposes
        # a hook for it.
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
            f"({time.time() - t0:.1f}s)",
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
            seq_lens = [len(seq) for seq in self.train_dataset.encoded]
        else:
            # The clean PairsDataset is always the first source. Use its
            # encoded list as the basis for length stats — typo / noise
            # rows sit alongside it under the same max_seq_len budget.
            base_ds = self.train_dataset
            if isinstance(base_ds, ConcatDataset):
                base_ds = base_ds.datasets[0]
            seq_lens = [len(seq) for seq, _ in base_ds.encoded]
        token_lens = np.array(seq_lens, dtype=np.int64)
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
        if self.variant == "b":
            (p_clean, p_typo, p_synth) = self._train_source_p
            self.dataset_stats.update({
                "p_clean": p_clean,
                "p_typo": p_typo,
                "p_synthetic": p_synth,
                "train_n_sources": float(len(self._train_source_lens)),
            })

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
