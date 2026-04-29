"""Mine in-session typo → correction pairs from the raw search-event log.

Each emitted row has the same shape as ``training_pairs.parquet`` so the LM
dataloader can read both with the same loader: ``(prefix, target, count)``
plus a ``split`` partition column.

The mining rule is the one we validated empirically:

  Within a single session, for each consecutive pair (prev_event → curr_event):
    * gap_ms <= GAP_SEC * 1000
    * prev had 0 hits, curr had > 0 hits     → user was unsatisfied, then succeeded
    * neither query is a prefix of the other → drop continued-typing & backspaces
    * both are at least MIN_LEN chars        → drop accidental enters / bots
    * Levenshtein(prev, curr) <= MAX_LEV     → restrict to genuine corrections

Aggregate at ``(split, prefix, target)`` so the same correction made in many
sessions accumulates count.

Split assignment uses the **correction**'s day (the curr_event), mirroring
``preprocess.py``: a typo→correction row lands in the same split window as
the natural ``(prefix, target)`` rows it sits next to.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import duckdb

from .data import RAW_DIR

DEFAULT_OUT_ROOT = Path("/data/datasets/suggest")
DEFAULT_GAP_SEC = 30
DEFAULT_EVAL_DAYS = 18
DEFAULT_MAX_LEV = 3
DEFAULT_MIN_LEN = 4


def build(
    con: duckdb.DuckDBPyConnection,
    raw_path: Path,
    gap_sec: int,
    max_lev: int,
    min_len: int,
) -> None:
    raw = str(raw_path)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW filtered AS
        SELECT session_id, timestamp, day,
               normalized_query_term AS q,
               hit_count
        FROM read_parquet('{raw}/**/*.parquet', hive_partitioning = TRUE)
        WHERE normalized_query_term IS NOT NULL
          AND length(normalized_query_term) >= {min_len}
          AND user_type = 'SHOPPER';
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW lagged AS
        SELECT session_id, day,
               q          AS curr_q,
               hit_count  AS curr_hits,
               timestamp  AS curr_ts,
               lag(q)         OVER w AS prev_q,
               lag(hit_count) OVER w AS prev_hits,
               lag(timestamp) OVER w AS prev_ts
        FROM filtered
        WINDOW w AS (PARTITION BY session_id ORDER BY timestamp);
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE typo_pair_events AS
        SELECT day, prev_q AS prefix, curr_q AS target
        FROM lagged
        WHERE prev_q IS NOT NULL
          AND prev_q != curr_q
          AND prev_hits = 0 AND curr_hits > 0
          AND length(prev_q) >= {min_len}
          AND length(curr_q) >= {min_len}
          AND date_diff('millisecond', prev_ts, curr_ts) <= {gap_sec * 1000}
          AND NOT starts_with(curr_q, prev_q)
          AND NOT starts_with(prev_q, curr_q)
          AND levenshtein(prev_q, curr_q) <= {max_lev};
    """)


def write_artifact(
    con: duckdb.DuckDBPyConnection,
    out_path: Path,
    eval_start_iso: str,
) -> None:
    con.execute(f"""
        COPY (
          SELECT
            CASE WHEN day >= DATE '{eval_start_iso}'
                 THEN 'eval' ELSE 'train' END AS split,
            prefix,
            target,
            count(*) AS count
          FROM typo_pair_events
          GROUP BY ALL
        ) TO '{out_path}'
        (FORMAT PARQUET, PARTITION_BY (split), OVERWRITE_OR_IGNORE);
    """)


def print_stats(con: duckdb.DuckDBPyConnection, out_path: Path) -> None:
    glob = f"{out_path}/**/*.parquet"
    rows = con.execute(f"""
        SELECT split,
               count(*) AS rows,
               sum(count) AS events,
               count(DISTINCT prefix) AS d_prefixes,
               count(DISTINCT target) AS d_targets
        FROM read_parquet('{glob}', hive_partitioning = TRUE)
        GROUP BY split ORDER BY split;
    """).fetchall()
    print(
        f"  {'split':5}  {'rows':>10}  {'events':>10}  "
        f"{'d_prefixes':>11}  {'d_targets':>10}"
    )
    for r in rows:
        print(
            f"  {r[0]:5}  {r[1]:>10,}  {r[2]:>10,}  "
            f"{r[3]:>11,}  {r[4]:>10,}"
        )

    print("\n  Top 15 typo → correction pairs (train, by count):", flush=True)
    rows = con.execute(f"""
        SELECT prefix, target, count
        FROM read_parquet('{glob}', hive_partitioning = TRUE)
        WHERE split = 'train'
        ORDER BY count DESC, prefix
        LIMIT 15;
    """).fetchall()
    for r in rows:
        s = f"{r[0]} → {r[1]}"
        print(f"    {s[:80]:80}  {r[2]:>5,}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--gap-sec", type=int, default=DEFAULT_GAP_SEC,
                   help="Max gap between consecutive in-session events.")
    p.add_argument("--max-lev", type=int, default=DEFAULT_MAX_LEV,
                   help="Max Levenshtein distance between typo and correction.")
    p.add_argument("--min-len", type=int, default=DEFAULT_MIN_LEN,
                   help="Drop events whose query is shorter than this.")
    p.add_argument("--eval-days", type=int, default=DEFAULT_EVAL_DAYS,
                   help="Trailing days to assign to the eval split.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing output dir.")
    args = p.parse_args()

    out = args.out_root / "typo_pairs.parquet"
    if out.exists():
        if not args.force:
            sys.exit(f"output exists: {out}. Use --force to overwrite.")
        shutil.rmtree(out)
    args.out_root.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads = 8;")

    print(
        f"[1/3] Building (gap={args.gap_sec}s, max_lev={args.max_lev}, "
        f"min_len={args.min_len})...",
        flush=True,
    )
    t0 = time.time()
    build(con, args.raw_dir, args.gap_sec, args.max_lev, args.min_len)
    print(f"      pipeline done in {time.time() - t0:.1f}s", flush=True)

    n_pair_events = con.execute(
        "SELECT count(*) FROM typo_pair_events"
    ).fetchone()[0]
    if n_pair_events == 0:
        sys.exit("no typo pair events found — relax the filters and retry")

    max_day = con.execute(
        "SELECT max(day) FROM typo_pair_events"
    ).fetchone()[0]
    eval_start = max_day - timedelta(days=args.eval_days - 1)
    print(f"      pair events: {n_pair_events:,}", flush=True)
    print(
        f"      eval window: [{eval_start} .. {max_day}] "
        f"({args.eval_days} days)",
        flush=True,
    )

    print(f"[2/3] Writing → {out}", flush=True)
    t0 = time.time()
    write_artifact(con, out, eval_start.isoformat())
    print(f"      written in {time.time() - t0:.1f}s", flush=True)

    print("[3/3] Stats", flush=True)
    print_stats(con, out)


if __name__ == "__main__":
    main()
