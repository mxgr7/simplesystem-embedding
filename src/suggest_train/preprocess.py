"""Preprocess raw search events into autocomplete training pairs and a
deduplicated target corpus.

Inputs
------
  /data/datasets/suggest/raw_search_events.parquet/   (180-day, day-partitioned)

Outputs
-------
  /data/datasets/suggest/training_pairs.parquet/split={train,eval}/
      prefix, target, prefix_len, target_len,
      oci_user, search_articles_by, count,
      first_seen_day, last_seen_day

  /data/datasets/suggest/targets.parquet/split={train,eval}/
      target, target_len, oci_user, search_articles_by, count,
      first_seen_day, last_seen_day

Both are partitioned by `split`. Train = all but the last EVAL_DAYS,
eval = last EVAL_DAYS. Split is determined by the **target's** day so a
chunk's prefixes and target always land in the same split.

Pipeline
--------
  1. Row filter:  query_term non-empty and ≤ MAX_QUERY_LEN, user_type = SHOPPER.
  2. Per session, ordered by timestamp, cut into chunks at:
       - inter-event gap > GAP_SEC, OR
       - "unrelated" transition: neither query is a prefix of the other AND
         the two queries don't share their first character.
  3. Per chunk, target = longest query_term (tie-break: latest timestamp).
  4. Emit pair (event.query_term, chunk.target) for every event where
       length(prefix) < length(target) AND target.startswith(prefix).
  5. Group + count by (split, prefix, target, oci_user, search_articles_by).
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
DEFAULT_MAX_QUERY_LEN = 100
DEFAULT_EVAL_DAYS = 18


def build_pipeline(con: duckdb.DuckDBPyConnection,
                   raw_path: Path, gap_sec: int, max_qt_len: int) -> None:
    raw = str(raw_path)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW filtered AS
        SELECT session_id, timestamp, day, query_term,
               oci_user, search_articles_by
        FROM read_parquet('{raw}/**/*.parquet', hive_partitioning = TRUE)
        WHERE query_term IS NOT NULL
          AND length(query_term) BETWEEN 1 AND {max_qt_len}
          AND user_type = 'SHOPPER';
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW ordered_v AS
        SELECT *,
               lag(query_term) OVER w AS prev_qt,
               lag(timestamp)  OVER w AS prev_ts
        FROM filtered
        WINDOW w AS (PARTITION BY session_id ORDER BY timestamp);
    """)
    # The chunk break flag: 1 starts a new chunk, 0 stays in the current one.
    # Cumulative sum partitioned by session gives a stable chunk_id.
    con.execute(f"""
        CREATE OR REPLACE TABLE chunked_t AS
        SELECT *,
               sum(
                 CASE
                   WHEN prev_qt IS NULL THEN 1
                   WHEN date_diff('millisecond', prev_ts, timestamp)
                        > {gap_sec * 1000} THEN 1
                   WHEN starts_with(query_term, prev_qt) THEN 0
                   WHEN starts_with(prev_qt, query_term) THEN 0
                   WHEN substr(query_term, 1, 1) = substr(prev_qt, 1, 1) THEN 0
                   ELSE 1
                 END
               ) OVER (
                 PARTITION BY session_id ORDER BY timestamp
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               ) AS chunk_id
        FROM ordered_v;
    """)
    con.execute("""
        CREATE OR REPLACE TABLE chunk_targets AS
        SELECT DISTINCT ON (session_id, chunk_id)
               session_id, chunk_id,
               query_term AS target,
               day AS target_day,
               oci_user AS target_oci_user,
               search_articles_by AS target_search_articles_by
        FROM chunked_t
        ORDER BY session_id, chunk_id,
                 length(query_term) DESC, timestamp DESC;
    """)
    con.execute("""
        CREATE OR REPLACE TABLE pairs_raw AS
        SELECT c.query_term AS prefix,
               t.target,
               t.target_day AS day,
               c.oci_user,
               c.search_articles_by
        FROM chunked_t c
        JOIN chunk_targets t USING (session_id, chunk_id)
        WHERE length(c.query_term) < length(t.target)
          AND starts_with(t.target, c.query_term);
    """)


def write_split_artifacts(con: duckdb.DuckDBPyConnection,
                          out_pairs: Path, out_targets: Path,
                          eval_start_iso: str) -> None:
    con.execute(f"""
        COPY (
          SELECT
            CASE WHEN day >= DATE '{eval_start_iso}'
                 THEN 'eval' ELSE 'train' END AS split,
            prefix,
            target,
            length(prefix) AS prefix_len,
            length(target) AS target_len,
            oci_user,
            search_articles_by,
            count(*) AS count,
            min(day) AS first_seen_day,
            max(day) AS last_seen_day
          FROM pairs_raw
          GROUP BY ALL
        ) TO '{out_pairs}'
        (FORMAT PARQUET, PARTITION_BY (split), OVERWRITE_OR_IGNORE);
    """)
    con.execute(f"""
        COPY (
          SELECT
            CASE WHEN target_day >= DATE '{eval_start_iso}'
                 THEN 'eval' ELSE 'train' END AS split,
            target,
            length(target) AS target_len,
            target_oci_user AS oci_user,
            target_search_articles_by AS search_articles_by,
            count(*) AS count,
            min(target_day) AS first_seen_day,
            max(target_day) AS last_seen_day
          FROM chunk_targets
          GROUP BY ALL
        ) TO '{out_targets}'
        (FORMAT PARQUET, PARTITION_BY (split), OVERWRITE_OR_IGNORE);
    """)


def print_stats(con: duckdb.DuckDBPyConnection,
                out_pairs: Path, out_targets: Path) -> None:
    pairs_glob = f"{out_pairs}/**/*.parquet"
    targets_glob = f"{out_targets}/**/*.parquet"

    rows = con.execute(f"""
        SELECT split,
               count(*) AS pairs,
               sum(count) AS events,
               count(DISTINCT prefix) AS d_prefixes,
               count(DISTINCT target) AS d_targets
        FROM read_parquet('{pairs_glob}', hive_partitioning = TRUE)
        GROUP BY split ORDER BY split;
    """).fetchall()
    print("  Pairs:", flush=True)
    print(f"    {'split':5}  {'rows':>10}  {'events':>12}  "
          f"{'d_prefixes':>11}  {'d_targets':>10}")
    for r in rows:
        print(f"    {r[0]:5}  {r[1]:>10,}  {r[2]:>12,}  "
              f"{r[3]:>11,}  {r[4]:>10,}")

    rows = con.execute(f"""
        SELECT split,
               count(*) AS rows,
               sum(count) AS events,
               count(DISTINCT target) AS d_targets
        FROM read_parquet('{targets_glob}', hive_partitioning = TRUE)
        GROUP BY split ORDER BY split;
    """).fetchall()
    print("  Targets:", flush=True)
    print(f"    {'split':5}  {'rows':>10}  {'events':>12}  {'d_targets':>10}")
    for r in rows:
        print(f"    {r[0]:5}  {r[1]:>10,}  {r[2]:>12,}  {r[3]:>10,}")

    rows = con.execute(f"""
        SELECT prefix_len, sum(count) AS events
        FROM read_parquet('{pairs_glob}', hive_partitioning = TRUE)
        WHERE split = 'train'
        GROUP BY prefix_len ORDER BY prefix_len;
    """).fetchall()
    print("  Prefix length (train, events):", flush=True)
    total = sum(r[1] for r in rows)
    cum = 0
    for r in rows[:20]:
        cum += r[1]
        print(f"    len={r[0]:>3}: {r[1]:>11,}   cum {cum/total:5.1%}")
    if len(rows) > 20:
        tail = sum(r[1] for r in rows[20:])
        print(f"    > 20:    {tail:>11,}")

    rows = con.execute(f"""
        WITH bins AS (
          SELECT CASE
            WHEN target_len BETWEEN 1 AND 5 THEN '1: 1-5'
            WHEN target_len BETWEEN 6 AND 10 THEN '2: 6-10'
            WHEN target_len BETWEEN 11 AND 20 THEN '3: 11-20'
            WHEN target_len BETWEEN 21 AND 40 THEN '4: 21-40'
            ELSE '5: 40+'
          END AS bin, count
          FROM read_parquet('{pairs_glob}', hive_partitioning = TRUE)
          WHERE split = 'train'
        )
        SELECT bin, sum(count) FROM bins GROUP BY bin ORDER BY bin;
    """).fetchall()
    print("  Target length (train, pair events):", flush=True)
    total = sum(r[1] for r in rows)
    for r in rows:
        print(f"    {r[0]:8}  {r[1]:>11,}  {r[1]/total:5.1%}")

    rows = con.execute(f"""
        SELECT oci_user, sum(count)
        FROM read_parquet('{targets_glob}', hive_partitioning = TRUE)
        WHERE split = 'train'
        GROUP BY oci_user ORDER BY 2 DESC;
    """).fetchall()
    total = sum(r[1] for r in rows)
    print("  Train targets by OCI:", flush=True)
    for r in rows:
        print(f"    {str(r[0]):8}  {r[1]:>11,}  {r[1]/total:5.1%}")

    rows = con.execute(f"""
        SELECT search_articles_by, sum(count)
        FROM read_parquet('{targets_glob}', hive_partitioning = TRUE)
        WHERE split = 'train'
        GROUP BY search_articles_by ORDER BY 2 DESC;
    """).fetchall()
    total = sum(r[1] for r in rows)
    print("  Train targets by search mode:", flush=True)
    for r in rows:
        print(f"    {str(r[0]):26}  {r[1]:>11,}  {r[1]/total:5.1%}")

    rows = con.execute(f"""
        SELECT target, sum(count) AS n
        FROM read_parquet('{targets_glob}', hive_partitioning = TRUE)
        WHERE split = 'train' AND oci_user = 'false'
        GROUP BY target ORDER BY n DESC LIMIT 15;
    """).fetchall()
    print("  Top 15 train targets (non-OCI):", flush=True)
    for r in rows:
        print(f"    {r[0][:60]:60}  {r[1]:,}")

    rows = con.execute(f"""
        SELECT prefix, target, sum(count) AS n
        FROM read_parquet('{pairs_glob}', hive_partitioning = TRUE)
        WHERE split = 'train' AND oci_user = 'false'
        GROUP BY prefix, target ORDER BY n DESC LIMIT 15;
    """).fetchall()
    print("  Top 15 train pairs (non-OCI):", flush=True)
    for r in rows:
        s = f"{r[0]} → {r[1]}"
        print(f"    {s[:80]:80}  {r[2]:,}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--gap-sec", type=int, default=DEFAULT_GAP_SEC,
                   help="Inter-event gap (s) above which a new chunk starts.")
    p.add_argument("--max-query-len", type=int,
                   default=DEFAULT_MAX_QUERY_LEN,
                   help="Drop events whose query_term exceeds this length.")
    p.add_argument("--eval-days", type=int, default=DEFAULT_EVAL_DAYS,
                   help="Number of trailing days to use as the eval split.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output dirs.")
    args = p.parse_args()

    out_pairs = args.out_root / "training_pairs.parquet"
    out_targets = args.out_root / "targets.parquet"
    for path in (out_pairs, out_targets):
        if path.exists():
            if not args.force:
                sys.exit(
                    f"output exists: {path}. Use --force to overwrite."
                )
            shutil.rmtree(path)
    args.out_root.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads = 8;")

    print(f"[1/4] Building pipeline (gap={args.gap_sec}s, "
          f"max_qt_len={args.max_query_len})...", flush=True)
    t0 = time.time()
    build_pipeline(con, args.raw_dir, args.gap_sec, args.max_query_len)
    print(f"      pipeline done in {time.time() - t0:.1f}s", flush=True)

    n_raw = con.execute(
        f"SELECT count(*) FROM read_parquet("
        f"'{args.raw_dir}/**/*.parquet', hive_partitioning = TRUE)"
    ).fetchone()[0]
    n_filt = con.execute("SELECT count(*) FROM chunked_t").fetchone()[0]
    n_chunks = con.execute(
        "SELECT count(*) FROM chunk_targets"
    ).fetchone()[0]
    n_pairs_raw = con.execute(
        "SELECT count(*) FROM pairs_raw"
    ).fetchone()[0]
    print(f"      raw events:        {n_raw:>12,}", flush=True)
    print(f"      after row filters: {n_filt:>12,}  "
          f"({n_filt/max(n_raw,1):.1%})", flush=True)
    print(f"      chunks:            {n_chunks:>12,}", flush=True)
    print(f"      raw pairs:         {n_pairs_raw:>12,}", flush=True)

    max_day = con.execute(
        "SELECT max(target_day) FROM chunk_targets"
    ).fetchone()[0]
    eval_start = max_day - timedelta(days=args.eval_days - 1)
    print(f"      max target day:    {max_day}", flush=True)
    print(f"      eval window:       [{eval_start} .. {max_day}] "
          f"({args.eval_days} days)", flush=True)
    eval_start_iso = eval_start.isoformat()

    t0 = time.time()
    print(f"[2/4] Writing pairs   → {out_pairs}", flush=True)
    print(f"[3/4] Writing targets → {out_targets}", flush=True)
    write_split_artifacts(con, out_pairs, out_targets, eval_start_iso)
    print(f"      written in {time.time() - t0:.1f}s", flush=True)

    print("[4/4] Stats", flush=True)
    print_stats(con, out_pairs, out_targets)


if __name__ == "__main__":
    main()
