"""Most-Popular-Completion (MPC) baseline.

Builds a sorted prefix index over the training targets corpus and answers
``topk(prefix, k)`` by:

  1. Locating the contiguous slice of targets that start with ``prefix`` via
     two ``bisect_left`` calls on the sorted array.
  2. Sorting that slice by descending event-weighted count.
  3. Returning the top-k targets that are *strictly longer* than ``prefix``.

Two variants ship out of the box:

  * ``pooled`` — all train target rows, aggregated by exact target string.
  * ``filtered_non_oci_standard`` — same, but restricted to
    ``oci_user='false'`` and ``search_articles_by='STANDARD'``. The plan
    expects this to be a cleaner head distribution; we'll measure on eval.
"""

from __future__ import annotations

import argparse
import time
from bisect import bisect_left
from pathlib import Path

import duckdb
import numpy as np

from .data import TARGETS_DIR

VARIANTS = ("pooled", "filtered_non_oci_standard")


def _build_where(variant: str) -> str:
    if variant == "pooled":
        return ""
    if variant == "filtered_non_oci_standard":
        return "WHERE oci_user='false' AND search_articles_by='STANDARD'"
    raise ValueError(f"Unknown MPC variant: {variant}. Pick one of {VARIANTS}.")


class MPC:
    """Lightweight prefix lookup over a sorted (target, count) array."""

    def __init__(self, targets: np.ndarray, counts: np.ndarray) -> None:
        if targets.shape != counts.shape:
            raise ValueError("targets and counts must have the same shape")
        self.targets = targets
        self.counts = counts

    @classmethod
    def build(
        cls,
        targets_dir: Path = TARGETS_DIR,
        variant: str = "pooled",
    ) -> "MPC":
        glob = f"{targets_dir}/split=train/**/*.parquet"
        where = _build_where(variant)
        con = duckdb.connect(":memory:")
        df = con.execute(
            f"""
            SELECT target, sum(count)::BIGINT AS count
            FROM read_parquet('{glob}', hive_partitioning = TRUE)
            {where}
            GROUP BY target
            ORDER BY target
            """
        ).fetchdf()
        targets = df["target"].to_numpy(dtype=object)
        counts = df["count"].to_numpy(dtype=np.int64)
        return cls(targets, counts)

    def topk(self, prefix: str, k: int = 10) -> list[str]:
        return [t for t, _ in self.topk_with_counts(prefix, k)]

    def topk_with_counts(
        self, prefix: str, k: int = 10
    ) -> list[tuple[str, int]]:
        if k <= 0 or len(self.targets) == 0:
            return []
        if not prefix:
            # Empty prefix: rank everything by count, return top-k.
            n = self.counts.size
            cap = min(k, n)
            idx = np.argpartition(-self.counts, cap - 1)[:cap]
            idx = idx[np.argsort(-self.counts[idx])]
            return [(self.targets[i], int(self.counts[i])) for i in idx[:k]]

        # Find the contiguous slice of targets that start with `prefix`.
        # bisect on a numpy object array is fine — element access returns a
        # plain Python str so comparisons go through the str fast path.
        lo = bisect_left(self.targets, prefix)
        hi = bisect_left(self.targets, prefix + "\uffff")
        if hi <= lo:
            return []

        plen = len(prefix)
        slice_targets = self.targets[lo:hi]
        slice_counts = self.counts[lo:hi]

        # Drop exact-prefix matches (target == prefix). The plan defines a
        # completion as strictly longer than the prefix.
        keep_mask = np.fromiter(
            (len(t) > plen for t in slice_targets),
            count=slice_targets.size,
            dtype=bool,
        )
        if not keep_mask.all():
            slice_targets = slice_targets[keep_mask]
            slice_counts = slice_counts[keep_mask]

        if slice_targets.size == 0:
            return []

        cap = min(k, slice_counts.size)
        if slice_counts.size <= cap:
            order = np.argsort(-slice_counts)
        else:
            partition = np.argpartition(-slice_counts, cap - 1)[:cap]
            order = partition[np.argsort(-slice_counts[partition])]
        return [
            (slice_targets[i], int(slice_counts[i])) for i in order[:k]
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, targets=self.targets, counts=self.counts)

    @classmethod
    def load(cls, path: Path) -> "MPC":
        data = np.load(path, allow_pickle=True)
        return cls(data["targets"], data["counts"])

    def __len__(self) -> int:
        return int(self.targets.size)

    def total_events(self) -> int:
        return int(self.counts.sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build the MPC artifact.")
    pb.add_argument("--targets-dir", type=Path, default=TARGETS_DIR)
    pb.add_argument("--variant", choices=VARIANTS, default="pooled")
    pb.add_argument("--out", type=Path, required=True)

    pt = sub.add_parser("topk", help="Print top-k completions for a prefix.")
    pt.add_argument("--mpc-path", type=Path, required=True)
    pt.add_argument("--k", type=int, default=10)
    pt.add_argument("prefix")

    args = p.parse_args()
    if args.cmd == "build":
        t0 = time.time()
        print(f"Building MPC ({args.variant}) from {args.targets_dir}...",
              flush=True)
        mpc = MPC.build(args.targets_dir, variant=args.variant)
        print(
            f"  size: {len(mpc):,} unique targets, "
            f"{mpc.total_events():,} events  "
            f"(built in {time.time()-t0:.1f}s)",
            flush=True,
        )
        mpc.save(args.out)
        print(f"  wrote {args.out}", flush=True)
    elif args.cmd == "topk":
        mpc = MPC.load(args.mpc_path)
        results = mpc.topk(args.prefix, args.k)
        for i, t in enumerate(results, 1):
            print(f"  {i:2d}. {t}")


if __name__ == "__main__":
    main()
