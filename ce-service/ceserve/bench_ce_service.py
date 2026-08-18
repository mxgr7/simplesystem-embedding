#!/usr/bin/env python3
"""Client-side k-sweep against a running ce-service.

What this measures that `pipeline/bench_ce_t4.py` cannot: the SERVICE overhead —
HTTP, JSON parse, base64 decode, the asyncio hop — on top of the in-process
forward. `bench_ce_t4.py` put a 120-candidate window at 78.1 ms p50 on a T4 with
4 pinned cores; the acceptance bar here is p50 <= 150 ms at k=120 and <= 15 ms of
overhead over that 78.1.

Windows come from `pipeline/build_ce_bench_windows.py`, which reads the article
tokens the indexer ACTUALLY STORED for real coarse-ranker windows. That matters:
latency is driven by the batch's longest member after padding, a per-window order
statistic, and a fixture rebuilt from a training-time text column would have a
different one.

Run it TWICE on a co-tenanted box — idle, and with splade under load. They are
co-tenants on one card, Turing has no MIG, and the interaction has never been
measured with a latency-sensitive consumer present.

    python -m ceserve.bench_ce_service --url http://127.0.0.1:8139 \\
        --windows ce_bench_windows.jsonl.gz --out bench_ce_service.json

Dependencies: stdlib only. Deliberately — this runs on the serving box, and
anything it needed installed would be another thing that can differ there.

MXG-144.
"""
import argparse
import gzip
import json
import random
import statistics
import time
import urllib.error
import urllib.request

# `bench_ce_t4.py`'s k-sweep points, plus the two the query service can actually
# produce (`default-window-size` 120, `max-window-size` 200) and the service cap.
DEFAULT_KS = (1, 10, 30, 60, 120, 160, 200, 256)


def load_windows(path):
    rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def build_body(windows, k, rng):
    """One request of exactly k candidates.

    For k > one window's size, stitch DISTINCT windows rather than repeating
    one: repeating would sample the same padded-length order statistic several
    times and understate the width, which is the quantity that sets the cost.
    """
    window = rng.choice(windows)
    candidates = list(window["candidates"])
    if k > len(candidates):
        pool = [w for w in windows if w is not window]
        rng.shuffle(pool)
        for other in pool:
            candidates.extend(other["candidates"])
            if len(candidates) >= k:
                break
    seen, unique = set(), []
    for entry in candidates:
        if entry["id"] in seen:
            continue
        seen.add(entry["id"])
        unique.append(entry)
        if len(unique) == k:
            break
    return {
        "query": window["query_term"],
        "segment": window["segment"],
        "candidates": unique,
    }, len(unique)


def post(url, body, api_key, timeout):
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{url}/rerank", data=data, headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return (time.perf_counter() - started) * 1000, payload, len(data)


def percentiles(values):
    ordered = sorted(values)

    def pick(p):
        return ordered[min(len(ordered) - 1, int(len(ordered) * p))]

    return {
        "n": len(ordered),
        "p50": round(pick(0.50), 2),
        "p90": round(pick(0.90), 2),
        "p95": round(pick(0.95), 2),
        "p99": round(pick(0.99), 2),
        "mean": round(statistics.fmean(ordered), 2),
        "max": round(ordered[-1], 2),
    }


def cell(url, windows, k, api_key, warmup, iters, budget_s, timeout, seed):
    rng = random.Random(seed + k)
    plan = [build_body(windows, k, rng) for _ in range(warmup + iters)]
    for body, _ in plan[:warmup]:
        post(url, body, api_key, timeout)

    latencies, widths, scored, skipped, bytes_out = [], [], 0, 0, []
    started = time.perf_counter()
    for body, actual_k in plan[warmup:]:
        ms, payload, size = post(url, body, api_key, timeout)
        latencies.append(ms)
        widths.append(payload["padded_width"])
        scored += payload["n_scored"]
        skipped += payload["n_skipped"]
        bytes_out.append(size)
        if len(latencies) >= 20 and time.perf_counter() - started > budget_s:
            break

    stats = percentiles(latencies)
    stats.update({
        "k": k,
        "requested_k": k,
        "actual_k": actual_k,
        "padded_width_p50": percentiles(widths)["p50"],
        "padded_width_max": percentiles(widths)["max"],
        "request_kb_p50": round(percentiles(bytes_out)["p50"] / 1024, 1),
        "n_scored": scored,
        "n_skipped": skipped,
        "pairs_per_s": round(scored / (sum(latencies) / 1000), 1) if latencies else 0,
    })
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8139")
    parser.add_argument("--windows", default="ce_bench_windows.jsonl.gz")
    parser.add_argument("--out", default="bench_ce_service.json")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--budget-s", type=float, default=20.0,
                        help="wall-clock cap per cell once >= 20 samples are in")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="idle",
                        help="free text recorded in the output, e.g. "
                             "'idle' or 'splade-under-load'")
    args = parser.parse_args()

    windows = load_windows(args.windows)
    with urllib.request.urlopen(f"{args.url}/metadata", timeout=10) as response:
        metadata = json.load(response)
    print(f"serving_contract = {metadata['serving_contract']}")
    print(f"device = {metadata.get('gpu_name', metadata['device'])} "
          f"{metadata.get('gpu_capability', '')} dtype={metadata['dtype']}")
    print(f"{len(windows)} windows x {len(windows[0]['candidates'])} candidates")

    cells = []
    for k in [int(v) for v in args.ks.split(",") if v.strip()]:
        if k > metadata["max_inputs_per_request"]:
            print(f"k={k} skipped: above MAX_INPUTS_PER_REQUEST="
                  f"{metadata['max_inputs_per_request']} (it would 413)")
            continue
        try:
            stats = cell(args.url, windows, k, args.api_key, args.warmup,
                         args.iters, args.budget_s, args.timeout, args.seed)
        except urllib.error.HTTPError as exc:
            print(f"k={k}: HTTP {exc.code} {exc.read()[:200]!r}")
            continue
        cells.append(stats)
        print(f"k={stats['actual_k']:4d}  p50={stats['p50']:8.2f} "
              f"p95={stats['p95']:8.2f} p99={stats['p99']:8.2f} ms   "
              f"width p50={stats['padded_width_p50']:5.0f}  "
              f"req {stats['request_kb_p50']:5.1f} KB  "
              f"{stats['pairs_per_s']:7.1f} pairs/s")

    payload = {
        "label": args.label,
        "url": args.url,
        "metadata": metadata,
        "windows": len(windows),
        "cells": cells,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    print(f"wrote {args.out}")

    # The two gates, stated rather than left to the reader.
    at_120 = next((c for c in cells if c["actual_k"] == 120), None)
    if at_120:
        overhead = at_120["p50"] - 78.1
        print(f"\nk=120 p50 {at_120['p50']:.1f} ms against the 150 ms budget: "
              f"{'PASS' if at_120['p50'] <= 150 else 'FAIL'}")
        print(f"service overhead over bench_ce_t4.py's in-process 78.1 ms: "
              f"{overhead:+.1f} ms — {'PASS' if overhead <= 15 else 'FAIL'} "
              f"(bar is <= 15 ms)")


if __name__ == "__main__":
    main()
