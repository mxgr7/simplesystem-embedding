"""Closed-loop concurrency load test for the search API.

Drives ``POST /{collection}/_search`` with N concurrent workers for a fixed
duration, replaying real PostHog queries weighted by their 30-day frequency
(falling back to uniform sampling). Use ``--sweep`` to walk a concurrency
ladder in one run and find the saturation point.

Latency reported is end-to-end wall-clock from the client (so it includes
embedding-service + Milvus + fusion). Sweeps are *back-to-back*; they share
warm caches with the previous step, which is what you want when probing
"how much concurrent traffic can this instance handle right now".

Quick examples:
    .venv/bin/python scripts/loadtest_search_api.py \\
        --url http://localhost:8001 --collection offers \\
        --sweep 1,2,4,8,16,32,64 --duration 20

    .venv/bin/python scripts/loadtest_search_api.py \\
        --url https://search.example.com --collection offers \\
        --concurrency 32 --duration 60 --api-key "$API_KEY" \\
        --raw-out /tmp/load_raw.tsv

Auth: pass ``--api-key`` or set ``API_KEY`` in the environment / .env. Omit
both if the server runs without auth.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import signal
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = REPO_ROOT / "reports" / "validation" / "posthog_queries.tsv"


@dataclass
class StepResult:
    concurrency: int
    duration_s: float
    latencies_ms: list[float] = field(default_factory=list)
    statuses: dict[int, int] = field(default_factory=dict)
    errors: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0

    @property
    def total(self) -> int:
        return sum(self.statuses.values()) + self.errors

    @property
    def ok(self) -> int:
        return sum(c for s, c in self.statuses.items() if 200 <= s < 300)

    @property
    def rps(self) -> float:
        elapsed = max(self.ended_at - self.started_at, 1e-9)
        return self.total / elapsed


def load_queries(path: Path, weighted: bool, max_queries: int | None) -> tuple[list[str], list[float] | None]:
    """Return (queries, weights). ``weights`` is None for uniform sampling.

    Drops the literal ``None`` placeholder and empty queries. Keeps short
    queries (e.g. ``"s"``) since they reflect real traffic.
    """
    queries: list[str] = []
    counts: list[float] = []
    with path.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header != ["query", "count"]:
            raise SystemExit(
                f"unexpected header in {path}: {header!r}; expected ['query','count']"
            )
        for row in reader:
            if len(row) != 2:
                continue
            q, c = row[0], row[1]
            if not q or q == "None":
                continue
            try:
                cnt = float(c)
            except ValueError:
                continue
            if cnt <= 0:
                continue
            queries.append(q)
            counts.append(cnt)
    if not queries:
        raise SystemExit(f"no usable queries found in {path}")
    if max_queries is not None and len(queries) > max_queries:
        # Keep the top-N by count — preserves the head of the distribution
        # which is what dominates real traffic anyway.
        order = sorted(range(len(queries)), key=lambda i: counts[i], reverse=True)[:max_queries]
        queries = [queries[i] for i in order]
        counts = [counts[i] for i in order]
    return queries, (counts if weighted else None)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, q))


async def worker(
    client: httpx.AsyncClient,
    url: str,
    body_template: dict,
    headers: dict,
    queries: list[str],
    weights: list[float] | None,
    rng: random.Random,
    deadline: float,
    result: StepResult,
    record_after: float,
) -> None:
    while True:
        now = time.perf_counter()
        if now >= deadline:
            return
        if weights is None:
            q = rng.choice(queries)
        else:
            q = rng.choices(queries, weights=weights, k=1)[0]
        body = dict(body_template, query=q)
        t0 = time.perf_counter()
        try:
            resp = await client.post(url, json=body, headers=headers)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if t0 >= record_after:
                result.statuses[resp.status_code] = result.statuses.get(resp.status_code, 0) + 1
                result.latencies_ms.append(elapsed_ms)
        except (httpx.HTTPError, asyncio.TimeoutError):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if t0 >= record_after:
                result.errors += 1
                result.latencies_ms.append(elapsed_ms)


async def run_step(
    url: str,
    body_template: dict,
    headers: dict,
    queries: list[str],
    weights: list[float] | None,
    concurrency: int,
    duration_s: float,
    warmup_s: float,
    timeout_s: float,
    seed: int,
) -> StepResult:
    result = StepResult(concurrency=concurrency, duration_s=duration_s)
    limits = httpx.Limits(
        max_connections=concurrency * 2,
        max_keepalive_connections=concurrency * 2,
    )
    timeout = httpx.Timeout(timeout_s, connect=min(5.0, timeout_s))
    async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=False) as client:
        start = time.perf_counter()
        deadline = start + warmup_s + duration_s
        record_after = start + warmup_s
        result.started_at = record_after
        rngs = [random.Random(seed + i) for i in range(concurrency)]
        tasks = [
            asyncio.create_task(
                worker(
                    client, url, body_template, headers,
                    queries, weights, rngs[i], deadline, result, record_after,
                )
            )
            for i in range(concurrency)
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            result.ended_at = time.perf_counter()
    return result


def format_step_row(r: StepResult) -> str:
    lat = r.latencies_ms
    if lat:
        mean = statistics.fmean(lat)
        p50 = percentile(lat, 50)
        p90 = percentile(lat, 90)
        p95 = percentile(lat, 95)
        p99 = percentile(lat, 99)
        p999 = percentile(lat, 99.9)
        mx = max(lat)
    else:
        mean = p50 = p90 = p95 = p99 = p999 = mx = float("nan")
    err_pct = (100.0 * (r.total - r.ok) / r.total) if r.total else 0.0
    return (
        f"{r.concurrency:>4}  {r.total:>7}  {r.rps:>8.1f}  "
        f"{err_pct:>5.1f}%  "
        f"{mean:>7.1f}  {p50:>7.1f}  {p90:>7.1f}  "
        f"{p95:>7.1f}  {p99:>7.1f}  {p999:>7.1f}  {mx:>7.1f}"
    )


def print_table(results: list[StepResult]) -> None:
    print()
    print(
        "conc  reqs        rps  err%      mean      p50      p90     "
        " p95      p99     p999      max  (ms)"
    )
    print("-" * 96)
    for r in results:
        print(format_step_row(r))


def print_status_breakdown(results: list[StepResult]) -> None:
    all_codes: set[int] = set()
    for r in results:
        all_codes.update(r.statuses.keys())
    if all_codes == {200} and not any(r.errors for r in results):
        return
    print()
    print("Status code breakdown:")
    for r in results:
        parts = [f"{code}: {r.statuses[code]}" for code in sorted(r.statuses)]
        if r.errors:
            parts.append(f"transport_err: {r.errors}")
        print(f"  conc={r.concurrency:>4}  " + ", ".join(parts))


def write_raw_log(path: Path, results: list[StepResult]) -> None:
    # Per-request rows are not retained (we only keep latencies and status
    # tallies to bound memory). Instead we dump per-step latency arrays so
    # post-hoc analysis can recompute any percentile.
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["concurrency", "latency_ms"])
        for r in results:
            for lat in r.latencies_ms:
                w.writerow([r.concurrency, f"{lat:.3f}"])


def parse_int_list(s: str) -> list[int]:
    out = [int(x) for x in s.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected non-empty comma-separated list of ints")
    if any(n < 1 for n in out):
        raise argparse.ArgumentTypeError("concurrency values must be >= 1")
    return out


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="Base URL of the search API (e.g. http://localhost:8001).")
    p.add_argument("--collection", default="offers", help="Dense collection name in the path. Default: offers.")
    p.add_argument("--index", default="offers", help="Value sent as the body's 'index' field. Default: offers.")
    p.add_argument("--api-key", default=None, help="API key. Falls back to API_KEY env. Omit for unauthed servers.")
    p.add_argument("--queries-file", type=Path, default=DEFAULT_QUERIES,
                   help=f"TSV with query<TAB>count header. Default: {DEFAULT_QUERIES.relative_to(REPO_ROOT)}")
    p.add_argument("--max-queries", type=int, default=10000,
                   help="Cap the query pool to the top-N by count. 0 disables the cap. Default: 10000.")
    p.add_argument("--uniform", action="store_true",
                   help="Sample queries uniformly. Default: weighted by PostHog 30-day frequency.")
    p.add_argument("--concurrency", type=int, default=None,
                   help="Single concurrency level (mutually exclusive with --sweep).")
    p.add_argument("--sweep", type=parse_int_list, default=None,
                   help="Comma-separated concurrency ladder, e.g. '1,2,4,8,16,32,64'.")
    p.add_argument("--duration", type=float, default=20.0,
                   help="Seconds to record at each step (after warmup). Default: 20.")
    p.add_argument("--warmup", type=float, default=3.0,
                   help="Seconds of unrecorded warmup at each step. Default: 3.")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Per-request timeout in seconds. Default: 30.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", default=None,
                   help="Search mode override (vector|bm25|hybrid|hybrid_classified). "
                        "Omit to use the server default.")
    p.add_argument("--k", type=int, default=None, help="Top-N override.")
    p.add_argument("--raw-out", type=Path, default=None,
                   help="Optional TSV path for per-request latencies (concurrency<TAB>latency_ms).")
    p.add_argument("--abort-on-error-rate", type=float, default=None,
                   help="If set (0-1), stop the sweep when a step's error rate exceeds this fraction.")
    return p.parse_args()


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = build_args()

    if (args.concurrency is None) == (args.sweep is None):
        print("error: pass exactly one of --concurrency or --sweep", file=sys.stderr)
        return 2
    levels = [args.concurrency] if args.concurrency is not None else args.sweep

    api_key = args.api_key or os.environ.get("API_KEY") or ""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"

    base = args.url.rstrip("/")
    # The canonical /{collection}/_search route is now the F2 stub (returns
    # empty until F3..F5 land); _search_v0 still serves the legacy hybrid
    # behaviour the loadtest measures.
    path = f"/{args.collection}/_search_v0"
    qs_parts: list[str] = []
    if args.mode:
        qs_parts.append(f"mode={args.mode}")
    if args.k is not None:
        qs_parts.append(f"k={args.k}")
    qs = ("?" + "&".join(qs_parts)) if qs_parts else ""
    url = f"{base}{path}{qs}"

    body_template = {"index": args.index, "query": ""}

    max_queries = None if args.max_queries == 0 else args.max_queries
    queries, weights = load_queries(args.queries_file, weighted=not args.uniform, max_queries=max_queries)
    print(f"Loaded {len(queries)} queries from {args.queries_file} "
          f"(weighted={'no' if args.uniform else 'yes'})")
    print(f"Target: {url}")
    print(f"Plan: concurrency={levels}, warmup={args.warmup}s, duration={args.duration}s/step, "
          f"timeout={args.timeout}s")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    aborted = False
    def _stop(*_):
        nonlocal aborted
        aborted = True
    loop.add_signal_handler(signal.SIGINT, _stop)

    results: list[StepResult] = []
    try:
        for c in levels:
            if aborted:
                print("Aborted before completing all steps.", file=sys.stderr)
                break
            print(f"\n>> step concurrency={c}", flush=True)
            r = loop.run_until_complete(
                run_step(
                    url=url,
                    body_template=body_template,
                    headers=headers,
                    queries=queries,
                    weights=weights,
                    concurrency=c,
                    duration_s=args.duration,
                    warmup_s=args.warmup,
                    timeout_s=args.timeout,
                    seed=args.seed,
                )
            )
            results.append(r)
            print(format_step_row(r))
            if args.abort_on_error_rate is not None and r.total > 0:
                err_rate = (r.total - r.ok) / r.total
                if err_rate > args.abort_on_error_rate:
                    print(f"Error rate {err_rate:.1%} exceeded "
                          f"--abort-on-error-rate {args.abort_on_error_rate:.1%}; stopping sweep.",
                          file=sys.stderr)
                    break
    finally:
        loop.close()

    print_table(results)
    print_status_breakdown(results)

    if args.raw_out is not None:
        write_raw_log(args.raw_out, results)
        print(f"\nPer-request latencies written to {args.raw_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
