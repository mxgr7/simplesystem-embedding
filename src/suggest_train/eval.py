"""Evaluation harness for the suggest models.

Computes MRR@10 plus recall@{1, 5, 10} on the held-out ``split=eval`` pair
events, stratified by ``prefix_len`` bucket and ``oci_user``. Pair counts
are used as the row weight, so the metric is event-weighted (a popular
``(prefix, target)`` pair contributes more than a tail one).

Models are passed in as a callable
``model_fn(prefix: str, k: int) -> list[str]`` returning the top-k candidate
completions for ``prefix`` (in rank order, no duplicates). The harness
caches one ``topk`` call per distinct prefix.

CLI
---
    suggest-eval --model mpc  --mpc-path /path/to/mpc.npz
    suggest-eval --model lm   --lm-ckpt  /path/to/best.ckpt
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypedDict

import duckdb
import numpy as np

from .data import PAIRS_DIR

ModelFn = Callable[[str, int], list[str]]

DEFAULT_K = 10
RECALL_KS = (1, 5, 10)
PREFIX_LEN_BUCKETS = ("1", "2", "3", "4-7", "8+")


class EvalRow(TypedDict):
    prefix: str
    target: str
    prefix_len: int
    oci_user: str
    search_articles_by: str
    count: float


def prefix_len_bucket(prefix_len: int) -> str:
    if prefix_len <= 1:
        return "1"
    if prefix_len == 2:
        return "2"
    if prefix_len == 3:
        return "3"
    if prefix_len <= 7:
        return "4-7"
    return "8+"


def load_eval_rows(
    pairs_dir: Path = PAIRS_DIR,
    sample_prefixes: int | None = None,
    seed: int = 0,
) -> list[EvalRow]:
    """Load eval rows. If ``sample_prefixes`` is set, draw N distinct
    prefixes without replacement weighted by event count, then keep only
    rows whose prefix was sampled. This gives an unbiased estimator of
    the event-weighted metric while keeping pre-compute time bounded."""
    glob = f"{pairs_dir}/split=eval/**/*.parquet"
    con = duckdb.connect(":memory:")
    df = con.execute(
        f"""
        SELECT prefix, target, prefix_len, oci_user, search_articles_by,
               sum(count)::BIGINT AS count
        FROM read_parquet('{glob}', hive_partitioning = TRUE)
        GROUP BY ALL
        """
    ).fetchdf()
    rows: list[EvalRow] = [
        {
            "prefix": r.prefix,
            "target": r.target,
            "prefix_len": int(r.prefix_len),
            "oci_user": r.oci_user,
            "search_articles_by": r.search_articles_by,
            "count": float(r.count),
        }
        for r in df.itertuples(index=False)
    ]
    if sample_prefixes:
        prefix_weights: dict[str, float] = {}
        for r in rows:
            prefix_weights[r["prefix"]] = (
                prefix_weights.get(r["prefix"], 0.0) + r["count"]
            )
        prefixes = list(prefix_weights.keys())
        weights = np.array(
            [prefix_weights[p] for p in prefixes], dtype=np.float64
        )
        probs = weights / weights.sum()
        n = min(int(sample_prefixes), len(prefixes))
        rng = np.random.default_rng(seed)
        sampled_idx = rng.choice(
            len(prefixes), size=n, replace=False, p=probs
        )
        keep = {prefixes[i] for i in sampled_idx}
        # Re-normalize per-prefix counts so each kept prefix contributes
        # weight 1.0 in total. With PPS sampling without replacement and
        # fixed-N draws, simple mean across sampled prefixes is the
        # unbiased estimator of the event-weighted population mean.
        rows = [r for r in rows if r["prefix"] in keep]
        for r in rows:
            r["count"] = r["count"] / prefix_weights[r["prefix"]]
    return rows


class _StratumAcc:
    __slots__ = ("weight", "rr_sum", "recall_sum")

    def __init__(self) -> None:
        self.weight = 0.0
        self.rr_sum = 0.0
        self.recall_sum = {k: 0.0 for k in RECALL_KS}

    def add(self, count: float, rank: int | None) -> None:
        self.weight += count
        if rank is not None and rank >= 1 and rank <= DEFAULT_K:
            self.rr_sum += count / rank
            for k in RECALL_KS:
                if rank <= k:
                    self.recall_sum[k] += count

    def finalize(self) -> dict[str, float]:
        if self.weight == 0:
            out = {f"mrr@{DEFAULT_K}": 0.0, "weight": 0.0}
            for k in RECALL_KS:
                out[f"recall@{k}"] = 0.0
            return out
        out: dict[str, float] = {
            f"mrr@{DEFAULT_K}": self.rr_sum / self.weight,
            "weight": self.weight,
        }
        for k in RECALL_KS:
            out[f"recall@{k}"] = self.recall_sum[k] / self.weight
        return out


def evaluate(
    model_fn: ModelFn,
    rows: Iterable[EvalRow],
    k: int = DEFAULT_K,
    progress_every: int | None = None,
) -> dict[str, dict[str, float]]:
    """Run a model against a set of eval rows.

    Returns a dict of stratum-name to metric dict. The ``"overall"`` key
    holds the unstratified result; other keys follow the form
    ``"prefix_len:<bucket>"``, ``"oci_user:<value>"``, etc.
    """
    if k != DEFAULT_K:
        raise ValueError(
            "evaluate() is wired for k=10 today. Loosen DEFAULT_K to add more."
        )

    overall = _StratumAcc()
    by_prefix_len: dict[str, _StratumAcc] = defaultdict(_StratumAcc)
    by_oci: dict[str, _StratumAcc] = defaultdict(_StratumAcc)
    by_search_mode: dict[str, _StratumAcc] = defaultdict(_StratumAcc)

    cache: dict[str, list[str]] = {}
    n_rows = 0
    t0 = time.time()

    for row in rows:
        n_rows += 1
        prefix = row["prefix"]
        if prefix not in cache:
            cache[prefix] = model_fn(prefix, k)
        candidates = cache[prefix]

        rank: int | None = None
        target = row["target"]
        for i, cand in enumerate(candidates):
            if cand == target:
                rank = i + 1
                break

        count = row["count"]
        bucket = prefix_len_bucket(row["prefix_len"])
        overall.add(count, rank)
        by_prefix_len[bucket].add(count, rank)
        by_oci[row["oci_user"]].add(count, rank)
        by_search_mode[row["search_articles_by"]].add(count, rank)

        if progress_every and n_rows % progress_every == 0:
            elapsed = time.time() - t0
            cache_size = len(cache)
            print(
                f"  ... {n_rows:>9,} rows  ({elapsed:5.1f}s)  "
                f"cache={cache_size:>7,}  "
                f"running mrr@10={overall.rr_sum / max(overall.weight, 1):.4f}",
                flush=True,
            )

    out: dict[str, dict[str, float]] = {"overall": overall.finalize()}
    for bucket in PREFIX_LEN_BUCKETS:
        out[f"prefix_len:{bucket}"] = by_prefix_len[bucket].finalize()
    for oci_value, acc in sorted(by_oci.items()):
        out[f"oci_user:{oci_value}"] = acc.finalize()
    for sm_value, acc in sorted(by_search_mode.items()):
        out[f"search_articles_by:{sm_value}"] = acc.finalize()
    out["_meta"] = {
        "n_rows": float(n_rows),
        "n_distinct_prefixes": float(len(cache)),
        "elapsed_sec": float(time.time() - t0),
    }
    return out


def format_report(report: dict[str, dict[str, float]]) -> str:
    cols = (f"mrr@{DEFAULT_K}", *(f"recall@{k}" for k in RECALL_KS), "weight")
    lines = [f"{'stratum':32}  " + "  ".join(f"{c:>10}" for c in cols)]
    lines.append("-" * len(lines[0]))
    for name in ("overall",):
        m = report[name]
        lines.append(_fmt_row(name, m, cols))
    lines.append("")
    for bucket in PREFIX_LEN_BUCKETS:
        m = report[f"prefix_len:{bucket}"]
        lines.append(_fmt_row(f"prefix_len:{bucket}", m, cols))
    lines.append("")
    for key in sorted(k for k in report if k.startswith("oci_user:")):
        lines.append(_fmt_row(key, report[key], cols))
    lines.append("")
    for key in sorted(k for k in report if k.startswith("search_articles_by:")):
        lines.append(_fmt_row(key, report[key], cols))
    return "\n".join(lines)


def _fmt_row(name: str, m: dict[str, float], cols: tuple[str, ...]) -> str:
    parts = [f"{name:32}"]
    for c in cols:
        v = m.get(c, 0.0)
        if c == "weight":
            parts.append(f"{v:>10.2f}" if abs(v - round(v)) > 1e-6
                         else f"{int(round(v)):>10,}")
        else:
            parts.append(f"{v:>10.4f}")
    return "  ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", required=True, choices=("mpc", "lm"),
        help="Which model to evaluate.",
    )
    p.add_argument("--pairs-dir", type=Path, default=PAIRS_DIR)
    p.add_argument("--mpc-path", type=Path, default=None,
                   help="Path to a built MPC artifact (.npz).")
    p.add_argument("--lm-ckpt", type=Path, default=None,
                   help="Path to a Lightning .ckpt for the LM.")
    p.add_argument("--lm-tokenizer", type=Path, default=None,
                   help="Tokenizer dir; defaults to TOKENIZER_DIR.")
    p.add_argument("--lm-beam-width", type=int, default=20)
    p.add_argument("--lm-batch-size", type=int, default=128)
    p.add_argument("--lm-prefix-batch", type=int, default=64,
                   help="How many prefixes to beam-search jointly per "
                        "forward pass (only used by the LM model).")
    p.add_argument("--lm-device", default="cuda")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to write the JSON report.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only score the first N eval rows (debug).")
    p.add_argument("--sample-prefixes", type=int, default=0,
                   help="If >0, fast-eval on N prefixes drawn without "
                        "replacement weighted by event count "
                        "(unbiased estimator of event-weighted MRR).")
    p.add_argument("--sample-seed", type=int, default=0)
    args = p.parse_args()

    print(f"[1/3] Loading eval rows from {args.pairs_dir}/split=eval/...",
          flush=True)
    t0 = time.time()
    rows = load_eval_rows(
        args.pairs_dir,
        sample_prefixes=args.sample_prefixes if args.sample_prefixes > 0 else None,
        seed=args.sample_seed,
    )
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"      loaded {len(rows):,} rows in {time.time()-t0:.1f}s",
          flush=True)

    print("[2/3] Loading model...", flush=True)
    t0 = time.time()
    if args.model == "mpc":
        from .mpc import MPC
        if args.mpc_path is None:
            raise SystemExit("--mpc-path is required for --model mpc")
        model = MPC.load(args.mpc_path)
        model_fn: ModelFn = model.topk
        precomputed: dict[str, list[str]] | None = None
    else:
        from .infer import load_lm_searcher
        if args.lm_ckpt is None:
            raise SystemExit("--lm-ckpt is required for --model lm")
        searcher = load_lm_searcher(
            ckpt_path=args.lm_ckpt,
            tokenizer_dir=args.lm_tokenizer,
            beam_width=args.lm_beam_width,
            prefix_batch_size=args.lm_prefix_batch,
            device=args.lm_device,
        )
        # Pre-compute top-k for every unique prefix in one batched pass —
        # this is what makes LM eval fast enough to run end-to-end.
        unique_prefixes = sorted({row["prefix"] for row in rows})
        print(
            f"      pre-computing LM top-{DEFAULT_K} for "
            f"{len(unique_prefixes):,} unique prefixes "
            f"(prefix_batch={args.lm_prefix_batch})...",
            flush=True,
        )
        t1 = time.time()
        precomputed = {}
        report_every = max(1000, len(unique_prefixes) // 20)
        for chunk_start in range(0, len(unique_prefixes), args.lm_prefix_batch):
            chunk = unique_prefixes[chunk_start : chunk_start + args.lm_prefix_batch]
            chunk_results = searcher.topk_batch(chunk, DEFAULT_K)
            for prefix, result in zip(chunk, chunk_results):
                precomputed[prefix] = result
            done = chunk_start + len(chunk)
            if done % report_every < args.lm_prefix_batch:
                rate = done / max(time.time() - t1, 1e-6)
                eta = (len(unique_prefixes) - done) / max(rate, 1e-6)
                print(
                    f"      ... {done:>7,}/{len(unique_prefixes):,} prefixes  "
                    f"({rate:.0f}/s, eta {eta:5.0f}s)",
                    flush=True,
                )
        print(
            f"      pre-compute done in {time.time()-t1:.1f}s",
            flush=True,
        )

        def model_fn(prefix: str, k: int) -> list[str]:
            return precomputed[prefix][:k]
    print(f"      loaded in {time.time()-t0:.1f}s", flush=True)

    print("[3/3] Scoring...", flush=True)
    report = evaluate(
        model_fn,
        rows,
        progress_every=max(20_000, len(rows) // 10),
    )

    print()
    print(format_report(report))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
