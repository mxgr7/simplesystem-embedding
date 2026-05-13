"""Render per-regime latency comparison tables from bench_profiles_latency.json.

Per regime: one row per profile variant; columns = p50_ms / p95_ms / p99_ms / mean_ms.

Usage:
  uv run python scripts/render_profiles_latency_table.py
  uv run python scripts/render_profiles_latency_table.py --in <path>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO_ROOT / "reports" / "hnsw_eval_full" / "bench_profiles_latency.json"


def variant_label(profile: str, numc: int | None) -> str:
    if numc is None:
        return profile
    return f"{profile}@{numc}"


VARIANT_ORDER = [
    ("standard", None),
    ("tp18-lex", None),
    ("tp18-vec", 1000),
    ("tp18-vec", 5000),
    ("hybrid", 1000),
    ("hybrid", 5000),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    args = p.parse_args()
    data = json.loads(args.inp.read_text())

    by_regime: dict[str, dict[tuple[str, int | None], dict]] = defaultdict(dict)
    for row in data["results"]:
        by_regime[row["regime"]][(row["profile"], row["numc"])] = row

    print(
        f"\n  n_queries={data['n_queries']}  concurrency={data['concurrency']}  "
        f"k={data['k']}  index={data['es_index']}  "
        f"wall={data['wall_seconds']:.0f}s"
    )

    regime_order = data.get("regimes") or list(by_regime.keys())
    for regime in regime_order:
        if regime not in by_regime:
            continue
        print(f"\n--- {regime} ---")
        hdr = (
            f"  {'profile':>14}  {'p50_ms':>9}  "
            f"{'p95_ms':>9}  {'p99_ms':>9}  {'mean_ms':>9}"
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for prof, numc in VARIANT_ORDER:
            row = by_regime[regime].get((prof, numc))
            if row is None:
                continue
            label = variant_label(prof, numc)
            print(
                f"  {label:>14}  "
                f"{row['p50_ms']:>9.2f}  {row['p95_ms']:>9.2f}  "
                f"{row['p99_ms']:>9.2f}  {row['mean_ms']:>9.2f}"
            )


if __name__ == "__main__":
    main()
