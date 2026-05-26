"""TEI throughput load generator for the AWS embedding-instance benchmark.

Pattern adapted from scripts/loadtest_search_api.py (closed-loop concurrent
workers, warmup + measurement windows, latency-percentile reporting). Hits
``POST /embed`` directly on the TEI nginx port. Sweeps a concurrency ladder
to find the saturation point, then sustains at that level for the
measurement window and reports embeddings/sec.

Inputs are synthetic offer-shaped strings that target ~256 tokens after
the multilingual-e5 tokenizer — matches the offer_template in
configs/data/default.yaml. Production parquet data isn't required here:
TEI's throughput on a given GPU depends on the *token-count distribution*,
not on the literal text, so synthetic strings with the right length give
the same answer.

Usage:
    python loadgen.py --url http://10.0.1.42:3000 \\
        --instance-type g6e.xlarge \\
        --sweep 4,8,16,32,64 --batch 32 \\
        --warmup 30 --duration 120 \\
        --results-csv results.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03dZ | %(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logging.Formatter.converter = time.gmtime
log = logging.getLogger("loadgen")

# 32 short German product-catalog phrases. Concatenated with separators
# until we hit the target token count, this approximates the
# multilingual-e5 token distribution of real offers. We don't tokenize
# client-side (would require the model's tokenizer); a char-length proxy
# of ~4 chars/token for German is close enough to land near the target.
_OFFER_FRAGMENTS = [
    "Schraube Edelstahl A2 M6x40 DIN 933 Sechskantkopf",
    "Innensechskantschraube galvanisch verzinkt Festigkeitsklasse 8.8",
    "Stahlplatte gewalzt 200x100x5 mm rostfrei poliert",
    "Werkzeugkoffer aus Aluminium mit Schaumstoffeinlage 24-teilig",
    "Akkuschrauber 18V Lithium-Ionen 2.0 Ah inklusive Ladegeraet",
    "Schleifpapier Korn 120 wasserfest 230x280 mm 10 Stueck",
    "Holzleim wasserfest D3 nach DIN EN 204 500 ml Flasche",
    "Spannungspruefer 12-1000V CAT IV beleuchtet zweipolig",
    "Schutzbrille kratzfest UV-Bestaendig EN166 verstellbar",
    "Arbeitshandschuhe Nitril beschichtet Groesse 10 EN 388",
    "Kabelbinder schwarz UV-bestaendig 200x4.8 mm 100 Stueck",
    "Schrauben-Sortiment Edelstahl 600-teilig Box mit Trennwaenden",
    "Wasserwaage 60cm magnetisch Aluminium drei Libellen Genauigkeit 0.5mm/m",
    "Hammer 500g Stiel aus Hickory deutsche Norm DIN 1041",
    "Fliesenkleber flexibel C2TE S1 25kg fuer Innen und Aussen",
    "Silikon sanitaer transparent Schimmelresistent 310 ml Kartusche",
    "Rohrzange 12 Zoll Stilson chromvanadium gehaertet Backe",
    "Maulschluessel Satz 8-19 mm chromvanadium poliert 8-teilig",
    "Bohrer HSS Cobalt 5% DIN 338 Durchmesser 6 mm geschliffen",
    "Saegeblatt Stichsaege Holz schnell 75mm T-Schaft 5 Stueck",
    "Klebepistole 230V 40W mit Klebesticks 11mm Schmelzpistole",
    "Schlagbohrmaschine 850W 1/2 Zoll Bohrfutter Schnellspann",
    "Multitool oszillierend 300W variable Drehzahl mit Zubehoer",
    "Werkstattlampe LED 30W IP65 kalt-weiss tragbar mit Haken",
    "Verlaengerungskabel 25m 3x1.5mm2 H07RN-F gummi schwarz",
    "Steckdosenleiste 6-fach mit Schalter und Ueberspannungsschutz",
    "Lautsprecherkabel OFC 2x2.5mm2 transparent Meterware Hifi",
    "Putzlappen Baumwolle weiss 10kg Sack saugfaehig fusselarm",
    "Industriestaubsauger 1400W 30L Edelstahlbehaelter nass-trocken",
    "Schraubenzieher Set 6-teilig Schlitz Kreuz isoliert VDE",
    "Kabelschuh Ringoese gelb verzinnt 4-6mm2 M8 100 Stueck",
    "Sicherheitsschuhe S3 SRC Stahlkappe Groesse 43 wasserdicht",
]


@dataclass
class StepResult:
    concurrency: int
    batch: int
    duration_s: float
    embeddings_done: int = 0
    requests_done: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0

    @property
    def emb_per_sec(self) -> float:
        elapsed = max(self.ended_at - self.started_at, 1e-9)
        return self.embeddings_done / elapsed

    @property
    def req_per_sec(self) -> float:
        elapsed = max(self.ended_at - self.started_at, 1e-9)
        return self.requests_done / elapsed


def _make_offer(target_tokens: int, rng: random.Random) -> str:
    """Build a fake offer-template string that should tokenize to ~target_tokens.
    Uses ~4 chars/token as the German XLM-R rough conversion factor."""
    target_chars = target_tokens * 4
    fragments = []
    total = 0
    fragments.append("passage: Article Name:")
    total += len(fragments[0])
    while total < target_chars:
        f = rng.choice(_OFFER_FRAGMENTS)
        fragments.append(f)
        total += len(f) + 1  # +1 for the space joiner
    return " ".join(fragments)


def _build_corpus(target_tokens: int, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [_make_offer(target_tokens, rng) for _ in range(n)]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, q))


async def worker(
    client: httpx.AsyncClient,
    url: str,
    corpus: list[str],
    batch: int,
    rng: random.Random,
    deadline: float,
    result: StepResult,
    record_after: float,
    worker_id: int,
) -> None:
    first_err_logged = False
    while True:
        now = time.perf_counter()
        if now >= deadline:
            return
        inputs = [rng.choice(corpus) for _ in range(batch)]
        body = {"inputs": inputs, "truncate": True}
        t0 = time.perf_counter()
        try:
            resp = await client.post(url, json=body)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if t0 >= record_after:
                if 200 <= resp.status_code < 300:
                    result.embeddings_done += batch
                    result.requests_done += 1
                    result.latencies_ms.append(elapsed_ms)
                else:
                    result.errors += 1
                    if not first_err_logged:
                        body_preview = ""
                        try:
                            body_preview = resp.text[:200]
                        except Exception:
                            pass
                        log.warning("worker %d: HTTP %d (%s)", worker_id,
                                    resp.status_code, body_preview)
                        first_err_logged = True
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            if t0 >= record_after:
                result.errors += 1
                if not first_err_logged:
                    log.warning("worker %d: transport error: %s: %s",
                                worker_id, type(e).__name__, e)
                    first_err_logged = True


async def _progress_ticker(
    result: StepResult,
    step_start: float,
    record_after: float,
    deadline: float,
    interval: float = 5.0,
) -> None:
    """Emit one progress line every `interval` seconds. Before `record_after`
    we're in warmup (no counters accumulate), after that we report recent +
    average throughput against the measurement window."""
    last_emb = 0
    last_t = record_after
    while True:
        now = time.perf_counter()
        remaining = deadline - now
        if remaining <= 0:
            return
        await asyncio.sleep(min(interval, max(0.5, remaining)))
        now = time.perf_counter()
        if now < record_after:
            log.info("  [warmup] elapsed=%5.1fs / %ds remaining=%5.1fs",
                     now - step_start, int(record_after - step_start),
                     record_after - now)
        else:
            emb = result.embeddings_done
            delta_emb = emb - last_emb
            delta_t = max(now - last_t, 1e-9)
            recent_rps = delta_emb / delta_t
            last_emb, last_t = emb, now
            avg_rps = emb / max(now - record_after, 1e-9)
            log.info("  [measure] elapsed=%5.1fs  emb=%7d  recent=%7.1f/s  avg=%7.1f/s  err=%d",
                     now - record_after, emb, recent_rps, avg_rps, result.errors)


async def run_step(
    url: str,
    corpus: list[str],
    concurrency: int,
    batch: int,
    duration_s: float,
    warmup_s: float,
    timeout_s: float,
    seed: int,
) -> StepResult:
    result = StepResult(concurrency=concurrency, batch=batch, duration_s=duration_s)
    limits = httpx.Limits(
        max_connections=concurrency * 2,
        max_keepalive_connections=concurrency * 2,
    )
    timeout = httpx.Timeout(timeout_s, connect=min(5.0, timeout_s))
    log.info("step start: conc=%d batch=%d warmup=%.0fs duration=%.0fs",
             concurrency, batch, warmup_s, duration_s)
    async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=False) as client:
        start = time.perf_counter()
        deadline = start + warmup_s + duration_s
        record_after = start + warmup_s
        result.started_at = record_after
        rngs = [random.Random(seed + i) for i in range(concurrency)]
        worker_tasks = [
            asyncio.create_task(
                worker(client, url, corpus, batch, rngs[i], deadline, result,
                       record_after, worker_id=i)
            )
            for i in range(concurrency)
        ]
        ticker = asyncio.create_task(
            _progress_ticker(result, start, record_after, deadline)
        )
        try:
            await asyncio.gather(*worker_tasks)
        finally:
            ticker.cancel()
            try:
                await ticker
            except asyncio.CancelledError:
                pass
            result.ended_at = time.perf_counter()
    log.info("step done:  conc=%d  emb=%d  err=%d  wall=%.1fs",
             concurrency, result.embeddings_done, result.errors,
             result.ended_at - result.started_at)
    return result


def format_row(r: StepResult) -> str:
    lat = r.latencies_ms
    if lat:
        mean = statistics.fmean(lat)
        p50 = percentile(lat, 50)
        p90 = percentile(lat, 90)
        p99 = percentile(lat, 99)
    else:
        mean = p50 = p90 = p99 = float("nan")
    return (
        f"conc={r.concurrency:>4}  batch={r.batch:>4}  "
        f"emb/s={r.emb_per_sec:>8.1f}  req/s={r.req_per_sec:>7.1f}  "
        f"mean={mean:>6.1f}ms  p50={p50:>6.1f}  p90={p90:>6.1f}  p99={p99:>6.1f}  "
        f"err={r.errors}"
    )


def parse_int_list(s: str) -> list[int]:
    out = [int(x) for x in s.split(",") if x.strip()]
    if not out or any(n < 1 for n in out):
        raise argparse.ArgumentTypeError("expected comma-separated positive ints")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="TEI base URL, e.g. http://10.0.1.42:3000")
    p.add_argument("--instance-type", required=True, help="Recorded in the CSV row.")
    p.add_argument("--max-batch-tokens", type=int, default=0,
                   help="Server's discovered max-batch-tokens (recorded in CSV; not enforced client-side).")
    p.add_argument("--batch-sweep", type=parse_int_list, default=[8, 32, 128],
                   help="Client batch sizes to sweep. Default: 8,32,128.")
    p.add_argument("--conc-sweep", type=parse_int_list, default=[4, 16, 64],
                   help="Concurrency levels to sweep. Default: 4,16,64.")
    p.add_argument("--target-tokens", type=int, default=256,
                   help="Approx target tokens per input. Default: 256.")
    p.add_argument("--corpus-size", type=int, default=1024,
                   help="How many unique synthetic offers to round-robin through.")
    p.add_argument("--warmup", type=float, default=15.0,
                   help="Warmup seconds per cell (shorter since we sweep a matrix). Default: 15.")
    p.add_argument("--duration", type=float, default=60.0,
                   help="Measurement seconds per cell. Default: 60.")
    p.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-csv", type=Path, default=Path("results.csv"),
                   help="CSV file to append the best-cell summary row to.")
    p.add_argument("--matrix-csv", type=Path, default=Path("results_matrix.csv"),
                   help="CSV file to append every-cell measurement row to.")
    args = p.parse_args()

    base = args.url.rstrip("/")
    embed_url = f"{base}/embed"
    health_url = f"{base}/health"

    log.info("instance_type=%s target=%s", args.instance_type, embed_url)
    log.info("building synthetic corpus: %d offers x ~%d tokens",
             args.corpus_size, args.target_tokens)
    corpus = _build_corpus(args.target_tokens, args.corpus_size, args.seed)
    char_lens = [len(c) for c in corpus]
    log.info("corpus stats: chars min=%d p50=%d mean=%.0f max=%d (≈ tokens/4)",
             min(char_lens), int(np.percentile(char_lens, 50)),
             statistics.fmean(char_lens), max(char_lens))
    log.info("sample offer (%d chars): %s...", len(corpus[0]), corpus[0][:160])

    cells = [(b, c) for b in args.batch_sweep for c in args.conc_sweep]
    log.info("plan: %d cells = batch_sweep=%s × conc_sweep=%s  warmup=%.0fs duration=%.0fs/cell  total≈%dmin",
             len(cells), args.batch_sweep, args.conc_sweep, args.warmup, args.duration,
             int((args.warmup + args.duration) * len(cells) / 60))

    # One-shot health probe + smoke-test embed before the real run, so a
    # bad URL or model-load failure surfaces in seconds rather than after
    # 30s of warmup.
    log.info("smoke test: GET %s", health_url)
    try:
        r = httpx.get(health_url, timeout=10.0)
        log.info("  /health -> %d %s", r.status_code, r.text.strip()[:80])
    except Exception as e:
        log.warning("  /health failed: %s: %s", type(e).__name__, e)
    log.info("smoke test: POST %s with 1 input", embed_url)
    try:
        r = httpx.post(embed_url, json={"inputs": [corpus[0]], "truncate": True},
                       timeout=60.0)
        if r.status_code == 200:
            v = r.json()
            log.info("  /embed ok -> %d vector(s), dim=%d, sample[:3]=%s",
                     len(v), len(v[0]) if v else 0,
                     [round(x, 4) for x in (v[0][:3] if v else [])])
        else:
            log.warning("  /embed -> HTTP %d: %s", r.status_code, r.text[:200])
    except Exception as e:
        log.warning("  /embed failed: %s: %s", type(e).__name__, e)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results: list[StepResult] = []
    # Per-batch early-stop: once a concurrency level regresses ≥5%, skip
    # higher concurrencies at the same batch. Different batches don't
    # short-circuit each other.
    try:
        for batch in args.batch_sweep:
            log.info("==== batch=%d ====", batch)
            prev_emb = 0.0
            for conc in args.conc_sweep:
                log.info("  --- cell: batch=%d conc=%d ---", batch, conc)
                r = loop.run_until_complete(
                    run_step(
                        url=embed_url, corpus=corpus,
                        concurrency=conc, batch=batch,
                        duration_s=args.duration, warmup_s=args.warmup,
                        timeout_s=args.timeout, seed=args.seed,
                    )
                )
                results.append(r)
                log.info("  %s", format_row(r))
                if prev_emb > 0 and r.emb_per_sec < prev_emb * 0.95:
                    log.info("  throughput regressed at batch=%d (%.1f -> %.1f) — "
                             "skipping higher conc for this batch",
                             batch, prev_emb, r.emb_per_sec)
                    break
                prev_emb = r.emb_per_sec
    finally:
        loop.close()

    log.info("=== full matrix summary ===")
    for r in results:
        log.info("  %s", format_row(r))

    # Write every cell to the matrix CSV.
    matrix_header = not args.matrix_csv.exists()
    with args.matrix_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if matrix_header:
            w.writerow([
                "instance_type", "max_batch_tokens", "batch", "concurrency",
                "target_tokens", "emb_per_sec", "req_per_sec",
                "p50_ms", "p90_ms", "p99_ms", "errors",
                "duration_s", "warmup_s", "timestamp",
            ])
        ts = int(time.time())
        for r in results:
            lat = r.latencies_ms
            p50 = percentile(lat, 50) if lat else float("nan")
            p90 = percentile(lat, 90) if lat else float("nan")
            p99 = percentile(lat, 99) if lat else float("nan")
            w.writerow([
                args.instance_type, args.max_batch_tokens, r.batch, r.concurrency,
                args.target_tokens,
                f"{r.emb_per_sec:.1f}", f"{r.req_per_sec:.1f}",
                f"{p50:.1f}", f"{p90:.1f}", f"{p99:.1f}", r.errors,
                args.duration, args.warmup, ts,
            ])

    # The "best" cell is the one with the highest sustained emb/s.
    best = max(results, key=lambda r: r.emb_per_sec)
    lat = best.latencies_ms
    p50 = percentile(lat, 50) if lat else float("nan")
    p99 = percentile(lat, 99) if lat else float("nan")

    summary_header = not args.results_csv.exists()
    with args.results_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if summary_header:
            w.writerow([
                "instance_type", "max_batch_tokens", "batch", "concurrency",
                "target_tokens", "emb_per_sec", "req_per_sec",
                "p50_ms", "p99_ms", "errors",
                "duration_s", "warmup_s", "timestamp",
            ])
        w.writerow([
            args.instance_type, args.max_batch_tokens, best.batch, best.concurrency,
            args.target_tokens,
            f"{best.emb_per_sec:.1f}", f"{best.req_per_sec:.1f}",
            f"{p50:.1f}", f"{p99:.1f}", best.errors,
            args.duration, args.warmup, int(time.time()),
        ])
    log.info("best cell: batch=%d conc=%d emb/s=%.1f p50=%.1fms p99=%.1fms err=%d",
             best.batch, best.concurrency, best.emb_per_sec, p50, p99, best.errors)
    log.info("wrote summary row to %s and %d matrix rows to %s",
             args.results_csv, len(results), args.matrix_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
