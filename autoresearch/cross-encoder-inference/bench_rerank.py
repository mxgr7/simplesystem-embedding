"""End-to-end /rerank latency benchmark — single-client, p95 of warm requests.

Usage:
  uv run python autoresearch/cross-encoder-inference/bench_rerank.py \\
    --ckpt /abs/path/to/soup.ckpt \\
    [--serve-dtype bf16] \\
    [--port 8080] \\
    [--n-warmup 5] [--n-measure 30] \\
    [--fixture autoresearch/cross-encoder-inference/fixture_2000x512.json] \\
    [--rebuild-fixture] \\
    [--data-path /abs/path/to/queries_offers_merged_labeled.parquet] \\
    [--keep-server]   # leave server running on exit (for repeated --no-start runs)
    [--no-start]      # don't spawn a server; assume one is already on --port

Spins up `cross_encoder_serve.server:app` on localhost, warms it, then sends
N sequential requests of (1 query × 2000 offers, every pair padded so the
tokenized input hits max_pair_length=512). Prints p50 / p95 / p99 wall time
and peak VRAM.

This is THE latency measurement for the cross-encoder inference autoresearch
program. Its p95 output gates keep/discard against the < 1000 ms target.

Output (machine-greppable):
    n_warmup=5 n_measure=30 n_offers=2000 max_pair_length=512
    p50_ms=1690.5
    p95_ms=1742.0
    p99_ms=1755.3
    peak_vram_gb=7.8
    health.autocast_dtype=bf16
    health.attn_implementation=sdpa

The fixture is FROZEN: built once from the labeled parquet on first run,
saved to JSON, reused thereafter. Changing the fixture invalidates historical
results.tsv numbers; use --rebuild-fixture deliberately and re-anchor.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = Path(__file__).resolve().parent / "fixture_2000x512.json"
DEFAULT_DATA_PATH = (
    REPO_ROOT.parent / "data" / "queries_offers_esci" / "queries_offers_merged_labeled.parquet"
)
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
LD_PATH = "/usr/lib/x86_64-linux-gnu"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

# Long German filler — appended to every offer's `description` field so that
# rendered (query, offer) tokenization always overshoots max_pair_length=512
# and HF truncation `only_second` truncates the offer side to exactly 512.
# Worst-case latency: every pair fills the entire 512-token budget.
_FILLER = (
    " weitere Informationen Beschreibung Produktdetails Spezifikation Maße "
    "Technische Daten Lieferumfang Hinweise Anwendungsbereich Material "
    "Verarbeitung Qualität Garantie Zertifizierung Sicherheitshinweise "
    "Montageanleitung Bedienungsanleitung Wartung Pflege Reinigung "
) * 30

_OFFER_KEYS = (
    "offer_id", "name", "manufacturer_name", "manufacturer_article_number",
    "manufacturer_article_type", "article_number", "ean", "category_paths",
    "description",
)


def _safe_str(v) -> str:
    if v is None:
        return ""
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return ""
    except Exception:
        pass
    return str(v)


def build_fixture(data_path: str, n_offers: int = 2000, seed: int = 0) -> dict:
    """Pick `n_offers` real rows from the labeled parquet, pad descriptions,
    return a {query, offers[]} dict ready to POST to /rerank.

    Deterministic for fixed (data_path, n_offers, seed). The query is the most
    common query_term in the sampled rows (so it's a plausible real query).
    """
    import pandas as pd

    print(f"[fixture] loading {data_path}", file=sys.stderr)
    df = pd.read_parquet(data_path)
    if len(df) < n_offers:
        raise RuntimeError(f"parquet has only {len(df)} rows; need {n_offers}")
    sample = df.sample(n=n_offers, random_state=seed).reset_index(drop=True)

    # Pick a representative query: the most-common query_term in the sample.
    query = str(sample["query_term"].mode().iloc[0])
    print(f"[fixture] query={query!r}  rows={len(sample)}", file=sys.stderr)

    offers = []
    for i, row in sample.iterrows():
        offer = {k: _safe_str(row.get(k, "")) for k in _OFFER_KEYS}
        # Make offer_id unique even if the parquet sample has dups
        if not offer["offer_id"]:
            offer["offer_id"] = f"bench-{i}"
        offer["description"] = (offer["description"] + _FILLER).strip()
        offers.append(offer)

    return {"query": query, "offers": offers}


def verify_fixture_lengths(fixture: dict, model_name: str, max_pair_length: int,
                           config_dir: str, n_check: int = 8) -> dict:
    """Spot-check that `n_check` random offers tokenize to exactly max_pair_length.

    Returns {n_checked, n_at_cap, min_len, max_len}. If n_at_cap < n_checked,
    the fixture isn't worst-case and the latency number is misleading.
    """
    import random
    from hydra import compose, initialize_config_dir
    from transformers import AutoTokenizer

    from embedding_train.rendering import RowTextRenderer

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="cross_encoder")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    renderer = RowTextRenderer(cfg.data)

    rng = random.Random(0)
    idx = rng.sample(range(len(fixture["offers"])), k=min(n_check, len(fixture["offers"])))
    lens = []
    for i in idx:
        offer = fixture["offers"][i]
        row = {**offer, "query_term": fixture["query"]}
        ctx = renderer.build_context(row)
        q_text = renderer.render_query_text(row, context=ctx)
        o_text = renderer.render_offer_text(row, context=ctx)
        enc = tok(q_text, o_text, truncation="only_second",
                  max_length=max_pair_length, return_tensors=None)
        lens.append(len(enc["input_ids"]))
    return {
        "n_checked": len(lens),
        "n_at_cap": sum(1 for L in lens if L == max_pair_length),
        "min_len": min(lens),
        "max_len": max(lens),
    }


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def start_server(ckpt: str, port: int, serve_dtype: str,
                 extra_env: Optional[dict] = None) -> subprocess.Popen:
    env = os.environ.copy()
    ld = env.get("LD_LIBRARY_PATH", "")
    if LD_PATH not in ld.split(":"):
        env["LD_LIBRARY_PATH"] = f"{LD_PATH}:{ld}" if ld else LD_PATH
    env["CKPT"] = ckpt
    env["SERVE_DTYPE"] = serve_dtype
    # We're benchmarking CE-alone; never load LGBM here even if it's set in the shell.
    env.pop("LGBM", None)
    if extra_env:
        env.update(extra_env)

    cmd = ["uv", "run", "uvicorn", "cross_encoder_serve.server:app",
           "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    print(f"[server] launching: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_health(url: str, timeout_s: float = 240.0) -> dict:
    deadline = time.monotonic() + timeout_s
    last_err = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                body = json.loads(r.read())
            if body.get("status") == "ok":
                return body
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            last_err = e
        time.sleep(2)
    raise RuntimeError(f"server did not become ready in {timeout_s}s: last_err={last_err!r}")


def stop_server(proc: subprocess.Popen, log_path: Optional[Path] = None,
                print_tail_on_fail: bool = True) -> None:
    try:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    finally:
        captured = ""
        if proc.stdout is not None:
            try:
                captured = proc.stdout.read() or ""
            except Exception:
                pass
        if log_path is not None:
            try:
                log_path.write_text(captured)
            except Exception:
                pass
        if print_tail_on_fail and captured:
            print("--- server stdout/stderr (last 80 lines) ---", file=sys.stderr)
            for line in captured.splitlines()[-80:]:
                print(line, file=sys.stderr)
            print("--- end server log ---", file=sys.stderr)


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------

def post_rerank(url: str, payload_bytes: bytes, timeout_s: float = 300.0) -> tuple[float, int]:
    """Send /rerank request, return (wall_ms, n_returned)."""
    req = urllib.request.Request(
        f"{url}/rerank",
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        body = r.read()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    parsed = json.loads(body)
    return wall_ms, parsed.get("n_returned", 0)


def vram_poller(stop_event: threading.Event, samples: list[int]) -> None:
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, timeout=2,
            ).strip().splitlines()
            if out:
                samples.append(int(out[0]))
        except Exception:
            pass
        time.sleep(0.1)


def percentile(values: list[float], q: float) -> float:
    """Type-7 quantile on a sorted copy. q in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    h = (len(s) - 1) * (q / 100.0)
    lo = int(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (h - lo)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=False,
                   help="Lightning .ckpt path. Required unless --no-start.")
    p.add_argument("--serve-dtype", default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--n-warmup", type=int, default=2,
                   help="Default 2: numbers stabilize after the first request once the "
                        "model is loaded. Bump to 5+ for tighter measurement once latency "
                        "drops (cost amortizes as p95 falls).")
    p.add_argument("--n-measure", type=int, default=10,
                   help="Default 10: p95 ≈ max(samples) so noise is ~max-mean of the run "
                        "(~0.4 s at the 9 s baseline; <100 ms once latency drops below 2 s). "
                        "Bump to 30+ once latency drops near the 1000 ms target — at that "
                        "point each measurement costs ~1 s, not ~9 s.")
    p.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    p.add_argument("--rebuild-fixture", action="store_true",
                   help="Rebuild the fixture from the parquet (deterministic, seed=0).")
    p.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    p.add_argument("--server-log", default=None,
                   help="Optional: write server stdout/stderr here on shutdown.")
    p.add_argument("--no-start", action="store_true",
                   help="Don't spawn a server; assume one is already on --port.")
    p.add_argument("--keep-server", action="store_true",
                   help="Don't terminate the server on exit.")
    return p.parse_args()


def main():
    args = parse_args()
    fixture_path = Path(args.fixture)
    url = f"http://127.0.0.1:{args.port}"

    # 1. Fixture: load or build
    if args.rebuild_fixture or not fixture_path.exists():
        if not Path(args.data_path).exists():
            raise SystemExit(f"--data-path does not exist: {args.data_path}")
        fixture = build_fixture(args.data_path, n_offers=2000, seed=0)
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(fixture))
        print(f"[fixture] wrote {fixture_path} ({fixture_path.stat().st_size/1e6:.1f} MB)",
              file=sys.stderr)
    else:
        fixture = json.loads(fixture_path.read_text())
        print(f"[fixture] loaded {fixture_path} (n_offers={len(fixture['offers'])})",
              file=sys.stderr)

    payload_bytes = json.dumps(fixture).encode("utf-8")

    # 2. Server: start (or assume running)
    proc: Optional[subprocess.Popen] = None
    if not args.no_start:
        if not args.ckpt:
            raise SystemExit("--ckpt required unless --no-start")
        proc = start_server(args.ckpt, args.port, args.serve_dtype)

    try:
        health = wait_for_health(url, timeout_s=240.0)
        print(f"[server] ready: {health}", file=sys.stderr)

        # 3. Quick sanity check on fixture (only if we have the parquet on hand)
        try:
            from hydra import compose, initialize_config_dir
            with initialize_config_dir(config_dir=args.config_dir, version_base="1.3"):
                cfg = compose(config_name="cross_encoder")
            check = verify_fixture_lengths(
                fixture, str(cfg.model.model_name),
                int(cfg.data.max_pair_length), args.config_dir, n_check=8,
            )
            print(f"[fixture] length_check: {check}", file=sys.stderr)
            if check["n_at_cap"] < check["n_checked"]:
                print("[fixture] WARNING: not all sampled pairs hit the token cap; "
                      "fixture may not represent worst case.", file=sys.stderr)
        except Exception as e:
            print(f"[fixture] length_check skipped: {e}", file=sys.stderr)

        # 4. Warmup
        print(f"[bench] warmup: {args.n_warmup} requests", file=sys.stderr)
        for i in range(args.n_warmup):
            wall_ms, n_ret = post_rerank(url, payload_bytes)
            print(f"[bench]   warm {i+1}: {wall_ms:.1f} ms (n_ret={n_ret})", file=sys.stderr)

        # 5. Measure (with VRAM polling)
        vram_samples: list[int] = []
        stop_event = threading.Event()
        poller = threading.Thread(target=vram_poller, args=(stop_event, vram_samples),
                                  daemon=True)
        poller.start()

        timings: list[float] = []
        n_returned_set = set()
        print(f"[bench] measure: {args.n_measure} requests", file=sys.stderr)
        for i in range(args.n_measure):
            wall_ms, n_ret = post_rerank(url, payload_bytes)
            timings.append(wall_ms)
            n_returned_set.add(n_ret)
            if (i + 1) % 5 == 0:
                print(f"[bench]   {i+1}/{args.n_measure}: last={wall_ms:.1f} ms",
                      file=sys.stderr)

        stop_event.set()
        poller.join(timeout=2)

        peak_vram_mb = max(vram_samples) if vram_samples else 0
        peak_vram_gb = peak_vram_mb / 1024.0

        # 6. Report
        print()
        print(f"n_warmup={args.n_warmup} n_measure={args.n_measure} "
              f"n_offers={len(fixture['offers'])} max_pair_length={int(cfg.data.max_pair_length)}")
        print(f"p50_ms={percentile(timings, 50):.1f}")
        print(f"p95_ms={percentile(timings, 95):.1f}")
        print(f"p99_ms={percentile(timings, 99):.1f}")
        print(f"mean_ms={statistics.mean(timings):.1f}")
        print(f"min_ms={min(timings):.1f}")
        print(f"max_ms={max(timings):.1f}")
        print(f"peak_vram_gb={peak_vram_gb:.1f}")
        print(f"n_returned_distinct={sorted(n_returned_set)}")
        print(f"health.autocast_dtype={health.get('autocast_dtype', '?')}")
        print(f"health.attn_implementation={health.get('attn_implementation', '?')}")
        print(f"health.device={health.get('device', '?')}")

    finally:
        if proc is not None and not args.keep_server:
            log_path = Path(args.server_log) if args.server_log else None
            stop_server(proc, log_path)


if __name__ == "__main__":
    main()
