"""EDA report for the raw search-event corpus.

Run with:
    uv run suggest-eda

Prints aggregations to stdout. Pure SQL via DuckDB, plus a small amount of
in-Python work to reconstruct intra-session typing trajectories.

Sections:
    1. Volume by day / weekday
    2. Null and empty rates per column
    3. User-type and OCI breakdown
    4. Query-length distribution (chars)
    5. Hit-count distribution
    6. Per-session event-count distribution
    7. Typing-trajectory analysis: how often is event[i+1] a prefix-extension
       of event[i] in the same session, vs a fresh query
    8. Within-session inter-event timing
    9. A/B variant breakdown
    10. Top-frequency raw queries (sanity check)
"""

from __future__ import annotations

import argparse
from typing import Any

import duckdb

from .data import RAW_DIR, duckdb_connect


def hr(title: str) -> None:
    print()
    print(f"=== {title} " + "=" * max(0, 70 - len(title)))


def fmt_table(rows: list[tuple], headers: list[str], align: str | None = None) -> str:
    if not rows:
        return "(empty)"
    cols = list(zip(*([tuple(headers)] + rows)))
    widths = [max(len(str(v)) for v in c) for c in cols]
    align = align or "l" * len(headers)
    lines = []
    def fmt_row(r):
        parts = []
        for i, v in enumerate(r):
            s = str(v)
            if align[i] == "r":
                parts.append(s.rjust(widths[i]))
            else:
                parts.append(s.ljust(widths[i]))
        return "  ".join(parts)
    lines.append(fmt_row(headers))
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def q(con: duckdb.DuckDBPyConnection, sql: str) -> list[tuple]:
    return con.execute(sql).fetchall()


def section_volume(con) -> None:
    hr("1. Volume")
    total, days, sessions, distinct_ids = q(con, """
        SELECT
          count(*),
          count(DISTINCT day),
          count(DISTINCT session_id),
          count(DISTINCT distinct_id)
        FROM events
    """)[0]
    print(f"total events:    {total:>12,}")
    print(f"days covered:    {days:>12,}")
    print(f"sessions:        {sessions:>12,}")
    print(f"distinct_ids:    {distinct_ids:>12,}")
    print(f"avg events/day:  {total/max(days,1):>12,.0f}")
    print(f"avg events/sess: {total/max(sessions,1):>12,.2f}")

    rows = q(con, """
        SELECT
          dayofweek(day) AS dow,
          count(*) AS n,
          count(DISTINCT day) AS days
        FROM events
        GROUP BY dow
        ORDER BY dow
    """)
    dow_names = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
    rows = [(dow_names[r[0]], f"{r[1]:,}", r[2], f"{r[1]/max(r[2],1):,.0f}") for r in rows]
    print()
    print(fmt_table(rows, ["dow","events","days","avg/day"], align="lrrr"))


def section_nulls(con) -> None:
    hr("2. Nulls / empties per column")
    string_cols = ["query_term","normalized_query_term","session_id","distinct_id",
                   "user_type","oci_user","feature_search_experiment",
                   "selected_company_id","search_articles_by"]
    other_cols = ["timestamp","hit_count"]
    total = q(con, "SELECT count(*) FROM events")[0][0]
    rows = []
    for c in string_cols:
        sql = f"""
          SELECT
            sum(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END),
            sum(CASE WHEN {c} IS NULL OR trim({c}) = '' THEN 1 ELSE 0 END)
          FROM events
        """
        nulls, ne = q(con, sql)[0]
        rows.append((c, f"{nulls:,}", f"{nulls/total:.1%}", f"{ne:,}", f"{ne/total:.1%}"))
    for c in other_cols:
        nulls = q(con, f"SELECT sum(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) FROM events")[0][0]
        rows.append((c, f"{nulls:,}", f"{nulls/total:.1%}", "-", "-"))
    print(fmt_table(rows, ["column","nulls","pct","null/empty","pct"], align="lrrrr"))


def section_user_type(con) -> None:
    hr("3. User type / OCI breakdown")
    rows = q(con, """
        SELECT user_type, oci_user, count(*) AS n
        FROM events GROUP BY user_type, oci_user ORDER BY n DESC
    """)
    total = sum(r[2] for r in rows)
    rows = [(r[0], r[1], f"{r[2]:,}", f"{r[2]/total:.1%}") for r in rows]
    print(fmt_table(rows, ["user_type","oci_user","events","pct"], align="llrr"))

    rows = q(con, """
        SELECT search_articles_by, count(*) AS n
        FROM events GROUP BY search_articles_by ORDER BY n DESC
    """)
    total = sum(r[1] for r in rows)
    rows = [(r[0], f"{r[1]:,}", f"{r[1]/total:.1%}") for r in rows]
    print()
    print(fmt_table(rows, ["search_articles_by","events","pct"], align="lrr"))


def section_length(con) -> None:
    hr("4. Query length (chars, raw query_term, non-empty rows only)")
    rows = q(con, """
        SELECT
          min(length(query_term)),
          quantile_cont(length(query_term), 0.10),
          quantile_cont(length(query_term), 0.50),
          quantile_cont(length(query_term), 0.90),
          quantile_cont(length(query_term), 0.99),
          max(length(query_term)),
          avg(length(query_term))
        FROM events
        WHERE query_term IS NOT NULL AND length(query_term) > 0
    """)[0]
    labels = ["min","p10","p50","p90","p99","max","mean"]
    for k,v in zip(labels, rows):
        print(f"  {k:>4}: {v}")

    rows = q(con, """
        WITH bins AS (
          SELECT CASE
            WHEN length(query_term) = 1 THEN '01: 1'
            WHEN length(query_term) = 2 THEN '02: 2'
            WHEN length(query_term) = 3 THEN '03: 3'
            WHEN length(query_term) BETWEEN 4 AND 5 THEN '04: 4-5'
            WHEN length(query_term) BETWEEN 6 AND 10 THEN '05: 6-10'
            WHEN length(query_term) BETWEEN 11 AND 20 THEN '06: 11-20'
            WHEN length(query_term) BETWEEN 21 AND 40 THEN '07: 21-40'
            WHEN length(query_term) BETWEEN 41 AND 80 THEN '08: 41-80'
            ELSE '09: 80+'
          END AS bin
          FROM events
          WHERE query_term IS NOT NULL AND length(query_term) > 0
        )
        SELECT bin, count(*) AS n FROM bins GROUP BY bin ORDER BY bin
    """)
    total = sum(r[1] for r in rows)
    rows = [(r[0], f"{r[1]:,}", f"{r[1]/total:.1%}") for r in rows]
    print()
    print(fmt_table(rows, ["len","events","pct"], align="lrr"))


def section_hitcount(con) -> None:
    hr("5. Hit-count distribution (non-empty queries)")
    rows = q(con, """
        SELECT
          sum(CASE WHEN hit_count = 0 THEN 1 ELSE 0 END) AS zero,
          sum(CASE WHEN hit_count BETWEEN 1 AND 9 THEN 1 ELSE 0 END) AS lo,
          sum(CASE WHEN hit_count BETWEEN 10 AND 99 THEN 1 ELSE 0 END) AS mid,
          sum(CASE WHEN hit_count BETWEEN 100 AND 999 THEN 1 ELSE 0 END) AS hi,
          sum(CASE WHEN hit_count BETWEEN 1000 AND 9999 THEN 1 ELSE 0 END) AS xhi,
          sum(CASE WHEN hit_count >= 10000 THEN 1 ELSE 0 END) AS huge,
          count(*) AS total
        FROM events WHERE query_term IS NOT NULL AND length(query_term) > 0
    """)[0]
    labels = ["0", "1-9", "10-99", "100-999", "1k-10k", "10k+"]
    counts = rows[:-1]
    total = rows[-1]
    rows = [(l, f"{c:,}", f"{c/total:.1%}") for l, c in zip(labels, counts)]
    print(fmt_table(rows, ["hit_count","events","pct"], align="lrr"))


def section_session_size(con) -> None:
    hr("6. Per-session event count")
    rows = q(con, """
        WITH per_session AS (
          SELECT session_id, count(*) AS n
          FROM events WHERE session_id IS NOT NULL
          GROUP BY session_id
        )
        SELECT
          count(*) AS n_sessions,
          quantile_cont(n, 0.50) AS p50,
          quantile_cont(n, 0.90) AS p90,
          quantile_cont(n, 0.99) AS p99,
          max(n) AS max,
          avg(n) AS mean
        FROM per_session
    """)[0]
    n_sessions, p50, p90, p99, mx, mean = rows
    print(f"  sessions: {n_sessions:>10,}")
    print(f"  p50:      {p50:>10}")
    print(f"  p90:      {p90:>10}")
    print(f"  p99:      {p99:>10}")
    print(f"  max:      {mx:>10}")
    print(f"  mean:     {mean:>10.2f}")

    rows = q(con, """
        WITH per_session AS (
          SELECT session_id, count(*) AS n
          FROM events WHERE session_id IS NOT NULL
          GROUP BY session_id
        ),
        bins AS (
          SELECT CASE
            WHEN n = 1 THEN '01: 1'
            WHEN n BETWEEN 2 AND 5 THEN '02: 2-5'
            WHEN n BETWEEN 6 AND 10 THEN '03: 6-10'
            WHEN n BETWEEN 11 AND 25 THEN '04: 11-25'
            WHEN n BETWEEN 26 AND 50 THEN '05: 26-50'
            WHEN n BETWEEN 51 AND 100 THEN '06: 51-100'
            WHEN n BETWEEN 101 AND 500 THEN '07: 101-500'
            ELSE '08: 500+'
          END AS bin, n
          FROM per_session
        )
        SELECT bin, count(*) AS sess, sum(n) AS events
        FROM bins GROUP BY bin ORDER BY bin
    """)
    tot_s = sum(r[1] for r in rows); tot_e = sum(r[2] for r in rows)
    rows = [(r[0], f"{r[1]:,}", f"{r[1]/tot_s:.1%}",
             f"{r[2]:,}", f"{r[2]/tot_e:.1%}") for r in rows]
    print()
    print(fmt_table(rows, ["events/session","sessions","pct","events","pct"],
                    align="lrrrr"))


def section_trajectory(con) -> None:
    hr("7. Typing-trajectory transitions (consecutive in-session events)")
    # Compute pairwise transitions: each row = (prev_qt, next_qt) within the
    # same session, ordered by timestamp. Classify each transition.
    print("  computing transitions (session-ordered pairs)...")
    rows = q(con, """
        WITH ordered AS (
          SELECT
            session_id,
            timestamp,
            coalesce(query_term, '') AS qt,
            lag(coalesce(query_term, '')) OVER (PARTITION BY session_id ORDER BY timestamp) AS prev_qt,
            lag(timestamp) OVER (PARTITION BY session_id ORDER BY timestamp) AS prev_ts
          FROM events WHERE session_id IS NOT NULL
        ),
        classified AS (
          SELECT
            CASE
              WHEN prev_qt IS NULL THEN 'first_in_session'
              WHEN prev_qt = '' AND qt = '' THEN 'empty_to_empty'
              WHEN prev_qt = '' AND qt <> '' THEN 'empty_to_typed'
              WHEN prev_qt <> '' AND qt = '' THEN 'typed_to_empty'
              WHEN qt = prev_qt THEN 'same'
              WHEN starts_with(qt, prev_qt) AND length(qt) > length(prev_qt) THEN 'prefix_extend'
              WHEN starts_with(prev_qt, qt) AND length(prev_qt) > length(qt) THEN 'prefix_shrink'
              WHEN length(qt) > 0 AND length(prev_qt) > 0
                   AND substr(qt, 1, least(length(qt), length(prev_qt)) - 1)
                       = substr(prev_qt, 1, least(length(qt), length(prev_qt)) - 1)
                THEN 'shared_prefix'
              ELSE 'unrelated'
            END AS cls,
            (epoch(timestamp) - epoch(prev_ts)) AS dt_s
          FROM ordered
        )
        SELECT
          cls,
          count(*) AS n,
          quantile_cont(dt_s, 0.50) AS p50_s,
          quantile_cont(dt_s, 0.90) AS p90_s
        FROM classified GROUP BY cls ORDER BY n DESC
    """)
    tot = sum(r[1] for r in rows)
    print()
    out = [(r[0], f"{r[1]:,}", f"{r[1]/tot:.1%}",
            f"{r[2]}" if r[2] is not None else "-",
            f"{r[3]}" if r[3] is not None else "-")
           for r in rows]
    print(fmt_table(out, ["transition","n","pct","p50_dt_s","p90_dt_s"],
                    align="lrrrr"))


def section_inter_event(con) -> None:
    hr("8. Inter-event delta within session (seconds)")
    rows = q(con, """
        WITH ordered AS (
          SELECT epoch(timestamp) - epoch(lag(timestamp)
            OVER (PARTITION BY session_id ORDER BY timestamp)) AS dt
          FROM events WHERE session_id IS NOT NULL
        )
        SELECT
          quantile_cont(dt, 0.10),
          quantile_cont(dt, 0.50),
          quantile_cont(dt, 0.90),
          quantile_cont(dt, 0.99),
          avg(dt)
        FROM ordered WHERE dt IS NOT NULL AND dt >= 0
    """)[0]
    p10,p50,p90,p99,mean = rows
    print(f"  p10:  {p10}")
    print(f"  p50:  {p50}")
    print(f"  p90:  {p90}")
    print(f"  p99:  {p99}")
    print(f"  mean: {mean:.2f}")


def section_variant(con) -> None:
    hr("9. A/B variant breakdown ($feature/search-experiment)")
    rows = q(con, """
        SELECT coalesce(feature_search_experiment, '<null>') AS v,
               count(*) AS n
        FROM events GROUP BY v ORDER BY n DESC
    """)
    tot = sum(r[1] for r in rows)
    rows = [(r[0], f"{r[1]:,}", f"{r[1]/tot:.1%}") for r in rows]
    print(fmt_table(rows, ["variant","events","pct"], align="lrr"))


def section_top_queries(con) -> None:
    hr("10. Top 20 raw query_term values (any length)")
    rows = q(con, """
        SELECT query_term, count(*) AS n
        FROM events
        WHERE query_term IS NOT NULL AND length(query_term) > 0
        GROUP BY query_term ORDER BY n DESC LIMIT 20
    """)
    rows = [(r[0][:60], f"{r[1]:,}") for r in rows]
    print(fmt_table(rows, ["query_term","count"], align="lr"))

    print()
    print("  Top 20 normalized_query_term:")
    rows = q(con, """
        SELECT normalized_query_term, count(*) AS n
        FROM events
        WHERE normalized_query_term IS NOT NULL AND length(normalized_query_term) > 0
        GROUP BY normalized_query_term ORDER BY n DESC LIMIT 20
    """)
    rows = [(r[0][:60], f"{r[1]:,}") for r in rows]
    print(fmt_table(rows, ["norm","count"], align="lr"))


SECTIONS = [
    section_volume,
    section_nulls,
    section_user_type,
    section_length,
    section_hitcount,
    section_session_size,
    section_trajectory,
    section_inter_event,
    section_variant,
    section_top_queries,
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(RAW_DIR))
    args = p.parse_args()

    print(f"[suggest-eda] dataset: {args.data}")
    con = duckdb_connect(args.data)
    for fn in SECTIONS:
        fn(con)
    print()


if __name__ == "__main__":
    main()
