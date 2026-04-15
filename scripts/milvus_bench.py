"""Reusable Milvus search benchmark.

Measures the `client.search` round-trip only — does not include model
inference or post-processing. Use this to compare index types, nprobe
settings, or hardware. Output is a set of latency tables plus an optional
recall@k sweep against a high-nprobe reference.

Typical usage:
    .venv/bin/python scripts/milvus_bench.py                  # default
    .venv/bin/python scripts/milvus_bench.py --nprobes 1,64,256
    .venv/bin/python scripts/milvus_bench.py --json /tmp/b.json --recall
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from pymilvus import Collection, MilvusClient, connections, utility


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", default="19530")
    p.add_argument("--collection", default="offers")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-queries", type=int, default=16,
                   help="Number of distinct random query vectors to generate. "
                        "Each trial draws uniformly from this pool.")
    p.add_argument("--trials", type=int, default=8,
                   help="Measurements per latency cell (after warmup).")
    p.add_argument("--warmup", type=int, default=2,
                   help="Unmeasured warmup queries per cell.")
    p.add_argument("--limit", type=int, default=10, help="top-k")
    p.add_argument("--nprobes", default="4,16,64",
                   help="Comma-separated nprobe values for IVF indexes. "
                        "Ignored for FLAT.")
    p.add_argument("--batches", default="1,5",
                   help="Comma-separated batch sizes for the batch-latency sweep.")
    p.add_argument("--batch-nprobe", type=int, default=64,
                   help="nprobe to use during the batch sweep.")
    p.add_argument("--recall", action="store_true",
                   help="Run a recall@k sweep vs the highest nprobe as reference.")
    p.add_argument("--recall-ref-nprobe", type=int, default=0,
                   help="nprobe used as recall ground-truth reference. "
                        "Defaults to max of --nprobes.")
    p.add_argument("--json", type=str, default="",
                   help="Write the full results table to this JSON path.")
    p.add_argument("--prime", action="store_true",
                   help="Run random queries until per-query latency stabilizes "
                        "(forces Milvus to fully load segments) before benching.")
    p.add_argument("--prime-nprobe", type=int, default=64)
    p.add_argument("--prime-max-rounds", type=int, default=15,
                   help="Max prime rounds (each round = 5 queries).")
    return p.parse_args()


def get_index_info(col: Collection) -> dict[str, Any]:
    indexes = col.indexes
    if not indexes:
        return {"index_type": "NONE"}
    idx = indexes[0]
    params = dict(idx.params) if idx.params else {}
    return {
        "field": idx.field_name,
        "index_type": params.get("index_type", "?"),
        "metric_type": params.get("metric_type", "?"),
        "params": params.get("params", {}),
    }


def gen_queries(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.astype(np.float16)


def time_once(client: MilvusClient, collection: str, data: list[np.ndarray],
              limit: int, params: dict) -> float:
    t0 = time.perf_counter()
    client.search(collection_name=collection, data=data, limit=limit,
                  search_params=params, output_fields=["id"])
    return (time.perf_counter() - t0) * 1000.0


def bench_cell(
    client: MilvusClient, collection: str, queries: np.ndarray,
    batch_size: int, limit: int, params: dict,
    trials: int, warmup: int, rng: np.random.Generator,
) -> dict[str, float]:
    n = queries.shape[0]

    def pick_batch() -> list[np.ndarray]:
        idx = rng.choice(n, size=batch_size, replace=False)
        return [queries[i] for i in idx]

    for _ in range(warmup):
        time_once(client, collection, pick_batch(), limit, params)

    samples = [time_once(client, collection, pick_batch(), limit, params)
               for _ in range(trials)]
    samples.sort()

    def pct(p: float) -> float:
        k = max(0, min(len(samples) - 1, int(round((len(samples) - 1) * p))))
        return samples[k]

    return {
        "min_ms": min(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "max_ms": max(samples),
        "mean_ms": statistics.fmean(samples),
        "trials": trials,
    }


def fmt_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) for c, w in zip(cells, widths))


def print_latency_table(
    title: str, rows: list[tuple[str, dict[str, float]]],
) -> None:
    print(f"\n{title}")
    header = ["label", "min", "p50", "p95", "p99", "max"]
    widths = [18, 8, 8, 8, 8, 8]
    print("  " + fmt_row(header, widths))
    print("  " + fmt_row(["-" * w for w in widths], widths))
    for label, m in rows:
        cells = [
            label,
            f"{m['min_ms']:.1f}",
            f"{m['p50_ms']:.1f}",
            f"{m['p95_ms']:.1f}",
            f"{m['p99_ms']:.1f}",
            f"{m['max_ms']:.1f}",
        ]
        print("  " + fmt_row(cells, widths))


def bench_recall(
    client: MilvusClient, collection: str, queries: np.ndarray,
    nprobes: list[int], ref_nprobe: int, limit: int,
) -> list[tuple[int, float]]:
    ref_params = {"metric_type": "COSINE", "params": {"nprobe": ref_nprobe}}
    ref_ids_per_query: list[set] = []
    for q in queries:
        res = client.search(
            collection_name=collection, data=[q], limit=limit,
            search_params=ref_params, output_fields=["id"],
        )
        ref_ids_per_query.append({h["entity"]["id"] for h in res[0]})

    out: list[tuple[int, float]] = []
    for nprobe in nprobes:
        params = {"metric_type": "COSINE", "params": {"nprobe": nprobe}}
        overlap = 0
        for q, ref in zip(queries, ref_ids_per_query):
            res = client.search(
                collection_name=collection, data=[q], limit=limit,
                search_params=params, output_fields=["id"],
            )
            got = {h["entity"]["id"] for h in res[0]}
            overlap += len(ref & got)
        recall = overlap / (len(queries) * limit)
        out.append((nprobe, recall))
    return out


def main() -> None:
    args = build_args()
    nprobes = parse_int_list(args.nprobes)
    batches = parse_int_list(args.batches)
    ref_nprobe = args.recall_ref_nprobe or max(nprobes)

    print(f"Connecting to {args.host}:{args.port}")
    connections.connect(host=args.host, port=args.port)
    if not utility.has_collection(args.collection):
        raise SystemExit(f"collection {args.collection!r} missing")

    col = Collection(args.collection)
    num_entities = col.num_entities
    info = get_index_info(col)
    print(f"collection={args.collection} num_entities={num_entities:,}")
    print(f"index={info}")

    print("load()...")
    t0 = time.time()
    col.load()
    load_s = time.time() - t0
    print(f"  load() returned in {load_s:.1f}s")

    print(f"generating {args.n_queries} random unit vectors (dim={args.dim}, "
          f"seed={args.seed})")
    queries = gen_queries(args.n_queries, args.dim, args.seed)

    client = MilvusClient(f"http://{args.host}:{args.port}")
    rng = np.random.default_rng(args.seed + 1)

    is_flat = info.get("index_type") == "FLAT"
    results: dict[str, Any] = {
        "collection": args.collection,
        "num_entities": num_entities,
        "index": info,
        "load_s": load_s,
        "config": {
            "trials": args.trials,
            "warmup": args.warmup,
            "limit": args.limit,
            "n_queries": args.n_queries,
            "nprobes": nprobes if not is_flat else [],
            "batches": batches,
            "batch_nprobe": args.batch_nprobe,
        },
        "prime": None,
        "single_query": [],
        "batched": [],
        "recall": [],
    }

    if args.prime:
        print("\nPriming segments (running random queries until latency "
              "stabilizes)...")
        prime_params = (
            {"metric_type": "COSINE", "params": {}}
            if is_flat else
            {"metric_type": "COSINE", "params": {"nprobe": args.prime_nprobe}}
        )
        prev_avg = float("inf")
        stable = 0
        prime_start = time.time()
        total_queries = 0
        for round_idx in range(args.prime_max_rounds):
            round_samples = []
            for _ in range(5):
                q = queries[rng.integers(queries.shape[0])]
                ms = time_once(client, args.collection, [q], args.limit, prime_params)
                round_samples.append(ms)
            total_queries += 5
            avg = statistics.fmean(round_samples)
            print(f"  round {round_idx+1:2d}: avg={avg:7.1f} ms  "
                  f"({total_queries} queries, {int(time.time()-prime_start):3d}s)")
            # Stable iff the per-round average has stopped dropping meaningfully
            # (within 15% of the previous round) AND is below 200 ms.
            if prev_avg < float("inf") and avg > 0.85 * prev_avg and avg < 200:
                stable += 1
            else:
                stable = 0
            if stable >= 3:
                print(f"  stable for 3 rounds — prime complete.")
                break
            prev_avg = avg
        results["prime"] = {
            "rounds": round_idx + 1,
            "total_queries": total_queries,
            "final_avg_ms": avg,
            "stable": stable >= 3,
            "elapsed_s": time.time() - prime_start,
        }
    # --- Block 1: single-query latency across nprobes
    if is_flat:
        params = {"metric_type": "COSINE", "params": {}}
        m = bench_cell(
            client, args.collection, queries, batch_size=1,
            limit=args.limit, params=params,
            trials=args.trials, warmup=args.warmup, rng=rng,
        )
        results["single_query"].append({"nprobe": None, **m})
        print_latency_table(
            f"Single query, FLAT, limit={args.limit} "
            f"({args.trials} trials, {args.warmup} warmup):",
            [("flat", m)],
        )
    else:
        rows = []
        for nprobe in nprobes:
            params = {"metric_type": "COSINE", "params": {"nprobe": nprobe}}
            m = bench_cell(
                client, args.collection, queries, batch_size=1,
                limit=args.limit, params=params,
                trials=args.trials, warmup=args.warmup, rng=rng,
            )
            results["single_query"].append({"nprobe": nprobe, **m})
            rows.append((f"nprobe={nprobe}", m))
        print_latency_table(
            f"Single query, {info['index_type']}, limit={args.limit} "
            f"({args.trials} trials, {args.warmup} warmup):",
            rows,
        )

    # --- Block 2: batched latency
    params = (
        {"metric_type": "COSINE", "params": {}}
        if is_flat else
        {"metric_type": "COSINE", "params": {"nprobe": args.batch_nprobe}}
    )
    batch_rows = []
    for bs in batches:
        if bs > queries.shape[0]:
            continue
        m = bench_cell(
            client, args.collection, queries, batch_size=bs,
            limit=args.limit, params=params,
            trials=args.trials, warmup=args.warmup, rng=rng,
        )
        per_q = {k: (v / bs if k.endswith("_ms") else v) for k, v in m.items()}
        results["batched"].append({"batch_size": bs, "total": m, "per_query": per_q})
        batch_rows.append((f"batch={bs}", m))
    probe_label = "" if is_flat else f" (nprobe={args.batch_nprobe})"
    print_latency_table(
        f"Batched search total latency{probe_label}, limit={args.limit}:",
        batch_rows,
    )

    # also show per-query amortized for batches
    if batch_rows:
        print(f"\nBatched search amortized per-query latency{probe_label}:")
        header = ["label", "min/q", "p50/q", "p95/q", "max/q"]
        widths = [18, 8, 8, 8, 8]
        print("  " + fmt_row(header, widths))
        print("  " + fmt_row(["-" * w for w in widths], widths))
        for item, (label, _) in zip(results["batched"], batch_rows):
            per_q = item["per_query"]
            cells = [
                label,
                f"{per_q['min_ms']:.1f}",
                f"{per_q['p50_ms']:.1f}",
                f"{per_q['p95_ms']:.1f}",
                f"{per_q['max_ms']:.1f}",
            ]
            print("  " + fmt_row(cells, widths))

    # --- Block 3: recall
    if args.recall and not is_flat:
        print(f"\nRecall@{args.limit} vs reference nprobe={ref_nprobe} "
              f"(on {args.n_queries} random queries):")
        recall_rows = bench_recall(
            client, args.collection, queries,
            nprobes=nprobes, ref_nprobe=ref_nprobe, limit=args.limit,
        )
        for nprobe, r in recall_rows:
            marker = "  (ref)" if nprobe == ref_nprobe else ""
            print(f"  nprobe={nprobe:4d}  recall={r:.3f}{marker}")
        results["recall"] = [
            {"nprobe": nprobe, "recall": r, "is_ref": nprobe == ref_nprobe}
            for nprobe, r in recall_rows
        ]
    elif args.recall and is_flat:
        print("\n(skipping recall sweep — FLAT has recall=1.0 by definition)")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
