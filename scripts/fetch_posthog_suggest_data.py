"""Pull `search_performed` events from PostHog into a day-partitioned parquet
dataset for training the query-autocomplete (`suggest`) model.

Output layout (Hive-style, readable by pyarrow / duckdb / spark):

  /data/datasets/suggest/raw_search_events.parquet/
      day=2025-10-30/data.parquet
      day=2025-10-31/data.parquet
      ...

One row per `search_performed` submit. Unfiltered.

Resumable: days whose `data.parquet` already exists are skipped. Use --force
to re-fetch.

Auth: reads POSTHOG_HOST / POSTHOG_PROJECT_ID / POSTHOG_PERSONAL_API_KEY from
the project `.env` (same as scripts/fetch_posthog_search_queries.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = Path.home() / "simplesystem-embedding" / ".env"
DEFAULT_OUT_DIR = Path("/data/datasets/suggest/raw_search_events.parquet")
DEFAULT_DAYS = 180
EVENT_NAME = "search_performed"
ROW_LIMIT_PER_DAY = 1_000_000

SCHEMA = pa.schema([
    pa.field("query_term", pa.string()),
    pa.field("normalized_query_term", pa.string()),
    pa.field("session_id", pa.string()),
    pa.field("distinct_id", pa.string()),
    pa.field("timestamp", pa.timestamp("us", tz="UTC")),
    pa.field("hit_count", pa.int64()),
    pa.field("user_type", pa.string()),
    pa.field("oci_user", pa.string()),
    pa.field("feature_search_experiment", pa.string()),
    pa.field("selected_company_id", pa.string()),
    pa.field("search_articles_by", pa.string()),
])


def hogql(host: str, project_id: str, api_key: str, query: str,
          timeout: float = 300.0) -> dict:
    url = f"{host.rstrip('/')}/api/projects/{project_id}/query/"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"query": {"kind": "HogQLQuery", "query": query}}
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"HogQL {r.status_code}: {r.text[:1000]}")
        return r.json()


def build_query(day: date) -> str:
    next_day = day + timedelta(days=1)
    return f"""
        SELECT
          toString(properties.queryTerm) AS query_term,
          toString(properties.normalizedQueryTerm) AS normalized_query_term,
          toString(properties['$session_id']) AS session_id,
          toString(distinct_id) AS distinct_id,
          timestamp,
          toInt(properties.searchResults.hitCount) AS hit_count,
          toString(properties.userType) AS user_type,
          toString(properties.ociUser) AS oci_user,
          toString(properties['$feature/search-experiment']) AS feature_search_experiment,
          toString(properties.selectedCompanyId) AS selected_company_id,
          toString(properties.searchParams.searchArticlesBy) AS search_articles_by
        FROM events
        WHERE event = '{EVENT_NAME}'
          AND timestamp >= toDateTime('{day.isoformat()} 00:00:00', 'UTC')
          AND timestamp <  toDateTime('{next_day.isoformat()} 00:00:00', 'UTC')
        ORDER BY timestamp
        LIMIT {ROW_LIMIT_PER_DAY}
    """


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except ValueError:
        return None


def to_table(columns: list[str], rows: list[list]) -> pa.Table:
    """Convert HogQL response rows into an Arrow table matching SCHEMA."""
    if not rows:
        return SCHEMA.empty_table()

    idx = {c: i for i, c in enumerate(columns)}
    n = len(rows)

    def col_str(name: str) -> list[str | None]:
        i = idx[name]
        out: list[str | None] = [None] * n
        for k, r in enumerate(rows):
            v = r[i]
            if v is None or v == "":
                continue
            out[k] = str(v)
        return out

    def col_int(name: str) -> list[int | None]:
        i = idx[name]
        out: list[int | None] = [None] * n
        for k, r in enumerate(rows):
            v = r[i]
            if v is None:
                continue
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                continue
        return out

    def col_ts(name: str) -> list[datetime | None]:
        i = idx[name]
        return [parse_ts(r[i]) for r in rows]

    data = {
        "query_term": col_str("query_term"),
        "normalized_query_term": col_str("normalized_query_term"),
        "session_id": col_str("session_id"),
        "distinct_id": col_str("distinct_id"),
        "timestamp": col_ts("timestamp"),
        "hit_count": col_int("hit_count"),
        "user_type": col_str("user_type"),
        "oci_user": col_str("oci_user"),
        "feature_search_experiment": col_str("feature_search_experiment"),
        "selected_company_id": col_str("selected_company_id"),
        "search_articles_by": col_str("search_articles_by"),
    }
    arrays = [pa.array(data[f.name], type=f.type) for f in SCHEMA]
    return pa.Table.from_arrays(arrays, schema=SCHEMA)


def write_day(out_root: Path, day: date, table: pa.Table) -> Path:
    day_dir = out_root / f"day={day.isoformat()}"
    day_dir.mkdir(parents=True, exist_ok=True)
    tmp = day_dir / "data.parquet.tmp"
    final = day_dir / "data.parquet"
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(final)
    return final


def fetch_with_retry(host: str, project_id: str, api_key: str,
                     day: date, max_attempts: int = 5) -> tuple[list[str], list[list]]:
    q = build_query(day)
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            res = hogql(host, project_id, api_key, q)
            return res.get("columns") or [], res.get("results") or []
        except Exception as exc:
            last_exc = exc
            wait = min(60, 2 ** attempt)
            print(
                f"  [retry {day}] attempt {attempt}/{max_attempts} failed: "
                f"{type(exc).__name__}: {str(exc)[:200]}  (sleep {wait}s)",
                flush=True,
            )
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--days", type=int, default=DEFAULT_DAYS,
                   help="How many days back from today (UTC) to pull. "
                        "Ignored if --start is given.")
    p.add_argument("--start", type=str, default=None,
                   help="Inclusive start date (UTC, YYYY-MM-DD). "
                        "Defaults to today - days.")
    p.add_argument("--end", type=str, default=None,
                   help="Exclusive end date (UTC, YYYY-MM-DD). "
                        "Defaults to today (UTC), so the in-progress day is "
                        "skipped to avoid writing a partial partition.")
    p.add_argument("--force", action="store_true",
                   help="Re-fetch days whose data.parquet already exists.")
    args = p.parse_args()

    load_dotenv(ENV_PATH)
    host = os.environ.get("POSTHOG_HOST", "")
    project_id = os.environ.get("POSTHOG_PROJECT_ID", "")
    api_key = os.environ.get("POSTHOG_PERSONAL_API_KEY", "")
    if not (host and project_id and api_key):
        sys.exit("Missing POSTHOG_HOST / POSTHOG_PROJECT_ID / "
                 "POSTHOG_PERSONAL_API_KEY in environment")

    today_utc = datetime.now(timezone.utc).date()
    end_day = date.fromisoformat(args.end) if args.end else today_utc
    start_day = (date.fromisoformat(args.start) if args.start
                 else end_day - timedelta(days=args.days))
    if start_day >= end_day:
        sys.exit(f"start ({start_day}) must be before end ({end_day})")

    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    n_days = (end_day - start_day).days
    print(
        f"[suggest] {EVENT_NAME}: pulling {n_days} days "
        f"[{start_day} .. {end_day - timedelta(days=1)}]  "
        f"-> {out_root}",
        flush=True,
    )

    total_rows = 0
    fetched_days = 0
    skipped_days = 0
    overall_t0 = time.time()
    day = start_day
    while day < end_day:
        target = out_root / f"day={day.isoformat()}" / "data.parquet"
        if target.exists() and not args.force:
            try:
                n = pq.read_metadata(target).num_rows
            except Exception:
                n = -1
            print(f"  [skip] {day}  rows={n:>7,}  (already exists)", flush=True)
            total_rows += max(n, 0)
            skipped_days += 1
            day += timedelta(days=1)
            continue

        t0 = time.time()
        columns, rows = fetch_with_retry(host, project_id, api_key, day)
        if len(rows) >= ROW_LIMIT_PER_DAY:
            print(
                f"  [WARN] {day} hit ROW_LIMIT_PER_DAY={ROW_LIMIT_PER_DAY:,} — "
                f"results truncated; consider chunking this day by hour",
                flush=True,
            )

        table = to_table(columns, rows)
        path = write_day(out_root, day, table)
        n = table.num_rows
        total_rows += n
        fetched_days += 1
        elapsed = time.time() - t0
        cum_elapsed = time.time() - overall_t0
        days_done = fetched_days + skipped_days
        eta_s = ((n_days - days_done) * (cum_elapsed / max(days_done, 1)))
        print(
            f"  [done] {day}  rows={n:>7,}  cum={total_rows:>10,}  "
            f"{elapsed:>5.1f}s  eta~{eta_s/60:5.1f}m  -> {path.name}",
            flush=True,
        )
        day += timedelta(days=1)

    print(
        f"[suggest] complete. days_fetched={fetched_days} "
        f"days_skipped={skipped_days} total_rows={total_rows:,} "
        f"elapsed={(time.time()-overall_t0)/60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
