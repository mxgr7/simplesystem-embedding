"""Run the validation checklist from `hybrid_v0.md §"Validation before
shipping"` against a live search-api.

Inputs (under ``reports/validation/``):
  top200_queries.tsv       classifier precision spot-check
  eans_seen.tsv            codes recall@5 spot-check
  freetext_seen.tsv        free-text regression check

Manual probes (hard-coded in this file): the 0-result fallback queries from
the doc — `din912`, an off-catalog EAN, a typo'd identifier.

Output:
  reports/hybrid_v0_validation.md  — per-step pass/fail + sample tables.

Notes on what's automated and what isn't:
  * Steps 1, 3, 4, 6 in the doc are fully scripted here.
  * Step 2 (codes recall@5) requires ground-truth product mapping. For now
    we report whether the EAN-shaped query produces ANY results at all and
    whether the strict path was selected — the absence of a held-out
    label set means full recall@5 is left to the operator's eyeball.
  * Step 5 (top-of-fused-page noise) is also a manual check; this script
    prints the top 10 for 50 ambiguous-shape queries to make eyeballing
    easy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "search-api"))
from hybrid import is_strict_identifier  # noqa: E402

EAN_RE = re.compile(r"^(\d{8}|\d{12,14})$")
DIGIT_CONTAINING_RE = re.compile(r"\d")

DEFAULT_VALIDATION_DIR = REPO_ROOT / "reports" / "validation"
DEFAULT_REPORT = REPO_ROOT / "reports" / "hybrid_v0_validation.md"

FALLBACK_PROBES = ["din912", "9999999999998", "rj45zzz"]


def load_tsv(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open(encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            q, _, n = line.partition("\t")
            try:
                rows.append((q, int(n) if n else 0))
            except ValueError:
                rows.append((q, 0))
    return rows


def search(
    client: httpx.Client, base: str, headers: dict, collection: str,
    q: str, **params,
) -> dict:
    url = f"{base}/{collection}/_search"
    body = {"query": q, "category": None, "index": "validate"}
    pq = {"debug": "1", **{k: str(v) for k, v in params.items()}}
    r = client.post(url, params=pq, json=body, headers=headers)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────
# Validations
# ─────────────────────────────────────────────────────────────────────

def step1_classifier_precision(top_queries: list[tuple[str, int]]) -> dict:
    flagged: list[tuple[str, int]] = []
    for q, n in top_queries:
        if is_strict_identifier(q):
            flagged.append((q, n))
    return {
        "n_top": len(top_queries),
        "n_flagged": len(flagged),
        "share_flagged_volume": (
            sum(n for _, n in flagged) / max(sum(n for _, n in top_queries), 1)
        ),
        "flagged_queries": flagged,
    }


def step2_codes_recall(
    client: httpx.Client, base: str, headers: dict, collection: str,
    eans: list[tuple[str, int]],
) -> dict:
    rows = []
    for q, n in eans:
        res = search(client, base, headers, collection, q,
                     mode="bm25", k=5, codes_limit=20)
        hits = res.get("hits", [])
        rows.append({
            "q": q, "n": n, "hit_count": len(hits),
            "top_score": hits[0]["_score"] if hits else None,
        })
    matched = sum(1 for r in rows if r["hit_count"] > 0)
    return {"n": len(rows), "with_hits": matched, "rows": rows}


def step3_freetext_regression(
    client: httpx.Client, base: str, headers: dict, collection: str,
    queries: list[tuple[str, int]],
) -> dict:
    rejected = 0
    overlap_ratios: list[float] = []
    rows = []
    for q, n in queries:
        if not is_strict_identifier(q):
            rejected += 1
        # dense-only baseline
        dense = search(client, base, headers, collection, q,
                       mode="vector", k=24)
        # hybrid
        hybrid = search(client, base, headers, collection, q,
                        mode="hybrid", k=24)
        d_ids = {h["_id"] for h in dense.get("hits", [])}
        h_ids = {h["_id"] for h in hybrid.get("hits", [])}
        if d_ids:
            overlap = len(d_ids & h_ids) / len(d_ids)
        else:
            overlap = 1.0 if not h_ids else 0.0
        overlap_ratios.append(overlap)
        # how much did codes contribute?
        codes_hits = [h for h in hybrid.get("hits", []) if h.get("_source_leg") == "rrf"]
        rows.append({
            "q": q, "overlap": overlap,
            "dense_n": len(d_ids), "hybrid_n": len(h_ids),
            "codes_contribution": len(codes_hits),
        })
    return {
        "n": len(queries),
        "n_rejected_by_classifier": rejected,
        "median_overlap": statistics.median(overlap_ratios) if overlap_ratios else 0.0,
        "mean_overlap": statistics.mean(overlap_ratios) if overlap_ratios else 0.0,
        "rows": rows,
    }


def step4_fallback(
    client: httpx.Client, base: str, headers: dict, collection: str,
) -> dict:
    rows = []
    for q in FALLBACK_PROBES:
        res = search(client, base, headers, collection, q,
                     mode="hybrid_classified", k=24, enable_fallback=1)
        hits = res.get("hits", [])
        debug = res.get("_debug", {})
        rows.append({
            "q": q, "hit_count": len(hits),
            "path": debug.get("path"),
            "fallback_fired": debug.get("fallback_fired"),
            "classifier_strict": debug.get("classifier_strict"),
        })
    return {"rows": rows}


def step5_noise_eyeball(
    client: httpx.Client, base: str, headers: dict, collection: str,
    freetext: list[tuple[str, int]], n: int = 50,
) -> dict:
    """Top 10 results in ``hybrid_classified`` for ambiguous-shape queries
    (digit-containing free text). Output is meant to be read manually."""
    candidates = [(q, c) for q, c in freetext if DIGIT_CONTAINING_RE.search(q)]
    sample = candidates[:n]
    rows = []
    for q, _ in sample:
        res = search(client, base, headers, collection, q,
                     mode="hybrid_classified", k=10)
        rows.append({
            "q": q,
            "hits": [
                {"id": h["_id"], "score": h["_score"], "leg": h.get("_source_leg", "")}
                for h in res.get("hits", [])
            ],
            "path": res.get("_debug", {}).get("path"),
        })
    return {"n": len(rows), "rows": rows}


def step6_latency(
    client: httpx.Client, base: str, headers: dict, collection: str,
    queries: list[tuple[str, int]], n: int = 200,
) -> dict:
    """Microbench. Drives each mode and records p50/p95 round-trip."""
    sample = [q for q, _ in queries[:n]]
    out = {}
    for mode in ("vector", "bm25", "hybrid", "hybrid_classified"):
        times: list[float] = []
        for q in sample:
            t0 = time.perf_counter()
            search(client, base, headers, collection, q, mode=mode, k=24)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        out[mode] = {
            "n": len(times),
            "p50": times[len(times) // 2],
            "p95": times[int(len(times) * 0.95)],
            "mean": statistics.mean(times),
        }
    return out


# ─────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────

def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def write_report(out_path: Path, results: dict, base_url: str, collection: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s1 = results["step1"]
    s2 = results["step2"]
    s3 = results["step3"]
    s4 = results["step4"]
    s5 = results["step5"]
    s6 = results["step6"]

    parts = [
        "# hybrid_v0 — validation",
        "",
        f"- search-api: `{base_url}`",
        f"- dense collection: `{collection}`",
        f"- timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## 1. Classifier precision (top-200 PostHog queries)",
        "",
        f"- total queries inspected: {s1['n_top']}",
        f"- queries flagged as strict identifier: {s1['n_flagged']}",
        f"- share of flagged volume: {100*s1['share_flagged_volume']:.1f}%",
        "",
        "Flagged queries (all should be real identifiers — eyeball):",
        "",
        md_table(["query", "events"], s1["flagged_queries"]) if s1["flagged_queries"] else "_(none)_",
        "",
        "## 2. Codes hits on EAN-shaped queries (BM25 mode, k=5)",
        "",
        f"- queries probed: {s2['n']}",
        f"- with at least one hit: {s2['with_hits']} / {s2['n']} "
        f"({100*s2['with_hits']/max(s2['n'],1):.1f}%)",
        "",
        md_table(
            ["query", "events", "hits", "top score"],
            [[r["q"], r["n"], r["hit_count"],
              f"{r['top_score']:.3f}" if r["top_score"] is not None else "—"]
             for r in s2["rows"][:30]],
        ),
        "",
        "## 3. Free-text regression (hybrid vs vector top-24)",
        "",
        f"- queries: {s3['n']}",
        f"- rejected by classifier: {s3['n_rejected_by_classifier']} / {s3['n']}",
        f"- median dense∩hybrid overlap: {100*s3['median_overlap']:.1f}%",
        f"- mean   dense∩hybrid overlap: {100*s3['mean_overlap']:.1f}%",
        "",
        md_table(
            ["query", "overlap", "dense_n", "hybrid_n", "codes_added"],
            [[r["q"], f"{100*r['overlap']:.0f}%", r["dense_n"], r["hybrid_n"],
              r["codes_contribution"]]
             for r in s3["rows"][:20]],
        ),
        "",
        "## 4. 0-result fallback probes",
        "",
        md_table(
            ["query", "hits", "path", "fallback?", "classifier strict?"],
            [[r["q"], r["hit_count"], r["path"],
              "yes" if r["fallback_fired"] else "no",
              "yes" if r["classifier_strict"] else "no"]
             for r in s4["rows"]],
        ),
        "",
        "## 5. Top-of-fused-page eyeball (hybrid_classified)",
        "",
        f"- queries shown: {s5['n']}",
        "",
    ]
    for row in s5["rows"]:
        parts.append(f"### {row['q']}")
        parts.append(f"path: `{row['path']}`")
        parts.append("")
        parts.append(md_table(
            ["#", "id", "score", "leg"],
            [[i+1, h["id"], f"{h['score']:.4f}", h["leg"]]
             for i, h in enumerate(row["hits"])],
        ))
        parts.append("")

    parts += [
        "## 6. Latency (round-trip from this script through search-api)",
        "",
        md_table(
            ["mode", "n", "p50 ms", "p95 ms", "mean ms"],
            [[m, v["n"], f"{v['p50']:.1f}", f"{v['p95']:.1f}", f"{v['mean']:.1f}"]
             for m, v in s6.items()],
        ),
        "",
    ]
    out_path.write_text("\n".join(parts), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://localhost:8001",
                   help="search-api base URL")
    p.add_argument("--collection", default="offers",
                   help="dense collection name passed in the URL path")
    p.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--latency-n", type=int, default=200)
    p.add_argument("--noise-n", type=int, default=50)
    p.add_argument("--no-step6", action="store_true",
                   help="Skip the latency micro-bench (slowest step).")
    args = p.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / "playground-app" / ".env")
    api_key = os.environ.get("SEARCH_API_KEY", "")
    headers = {"X-API-Key": api_key} if api_key else {}

    top200 = load_tsv(args.validation_dir / "top200_queries.tsv")
    eans = load_tsv(args.validation_dir / "eans_seen.tsv")
    freetext = load_tsv(args.validation_dir / "freetext_seen.tsv")

    results: dict = {}
    with httpx.Client(timeout=60.0) as client:
        # Sanity ping.
        url = f"{args.base}/{args.collection}/_search"
        try:
            client.post(url, params={"mode": "vector", "k": "1"},
                        json={"query": "test", "category": None, "index": "ping"},
                        headers=headers).raise_for_status()
        except Exception as exc:
            sys.exit(f"search-api unreachable at {url}: {exc}")

        print("Step 1: classifier precision …")
        results["step1"] = step1_classifier_precision(top200)

        print("Step 2: codes hits on EAN-shaped queries …")
        results["step2"] = step2_codes_recall(client, args.base, headers,
                                              args.collection, eans)

        print("Step 3: free-text regression …")
        results["step3"] = step3_freetext_regression(client, args.base, headers,
                                                     args.collection, freetext)

        print("Step 4: 0-result fallback probes …")
        results["step4"] = step4_fallback(client, args.base, headers, args.collection)

        print(f"Step 5: top-of-fused-page eyeball (n={args.noise_n}) …")
        results["step5"] = step5_noise_eyeball(client, args.base, headers,
                                               args.collection, freetext, n=args.noise_n)

        if args.no_step6:
            results["step6"] = {}
        else:
            print(f"Step 6: latency micro-bench (n={args.latency_n}) …")
            results["step6"] = step6_latency(client, args.base, headers,
                                             args.collection, top200, n=args.latency_n)

    write_report(args.report, results, args.base, args.collection)
    print(f"\nReport written to {args.report}")
    print(f"Step 1 flagged {results['step1']['n_flagged']} of {results['step1']['n_top']} "
          f"top queries as strict.")
    print(f"Step 2 hit rate: {results['step2']['with_hits']}/{results['step2']['n']}.")
    print(f"Step 3 median dense/hybrid overlap: "
          f"{100*results['step3']['median_overlap']:.1f}%.")


if __name__ == "__main__":
    main()
