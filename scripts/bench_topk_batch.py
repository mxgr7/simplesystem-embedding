"""One-off micro-benchmark: sweep prefix_batch_size for topk_batch.

Picks a fixed slice of unique eval prefixes and runs the LM beam search
at several batch sizes, reporting prefixes/sec and peak GPU memory so we
can size the full eval run.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import torch

from suggest_train.data import PAIRS_DIR, TOKENIZER_DIR
from suggest_train.infer import load_lm_searcher


def collect_prefixes(n: int) -> list[str]:
    glob = f"{PAIRS_DIR}/split=eval/**/*.parquet"
    con = duckdb.connect(":memory:")
    df = con.execute(
        f"""
        SELECT prefix
        FROM read_parquet('{glob}', hive_partitioning = TRUE)
        GROUP BY prefix
        ORDER BY prefix
        LIMIT {n}
        """
    ).fetchdf()
    return df["prefix"].tolist()


def run_one(searcher, prefixes: list[str], k: int, batch: int) -> dict:
    searcher.prefix_batch_size = batch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = searcher.topk_batch(prefixes, k)
    elapsed = time.time() - t0
    n_with_results = sum(1 for r in out if r)
    return {
        "batch": batch,
        "prefixes": len(prefixes),
        "elapsed_sec": elapsed,
        "throughput_per_s": len(prefixes) / max(elapsed, 1e-6),
        "peak_mem_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "with_results": n_with_results,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--n-prefixes", type=int, default=2000)
    p.add_argument("--batch-sizes", type=int, nargs="+",
                   default=[32, 64, 128, 256, 512, 1024, 2048])
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    prefixes = collect_prefixes(args.n_prefixes)
    print(f"Loaded {len(prefixes):,} prefixes")

    searcher = load_lm_searcher(
        ckpt_path=args.ckpt,
        beam_width=20,
        prefix_batch_size=args.batch_sizes[0],
        device="cuda",
    )

    print(f"{'batch':>6}  {'thru(/s)':>10}  {'elapsed':>8}  {'peak(MB)':>9}  ok")
    for batch in args.batch_sizes:
        try:
            r = run_one(searcher, prefixes, args.k, batch)
            print(
                f"{r['batch']:>6}  {r['throughput_per_s']:>10.1f}  "
                f"{r['elapsed_sec']:>7.1f}s  {r['peak_mem_mb']:>9.0f}  "
                f"{r['with_results']:>4}/{r['prefixes']}",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"{batch:>6}  OOM ({e!s:.60})", flush=True)
            torch.cuda.empty_cache()
            break


if __name__ == "__main__":
    main()
