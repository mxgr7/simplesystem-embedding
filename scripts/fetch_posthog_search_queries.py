"""Pull `search_performed` queries from PostHog and emit the validation TSVs.

Output (under ``reports/validation/``):
  posthog_queries.tsv      ALL distinct queries with their 30-day count
                           (one row per query). Source of truth for the
                           three derived files below.
  top200_queries.tsv       Top 200 queries by count — used for classifier
                           precision spot-check.
  eans_seen.tsv            Queries that match a strict EAN regex (8/12-14
                           digits) — sample for codes-recall@5 spot-check.
  freetext_seen.tsv        Queries the classifier rejects entirely — sample
                           for the free-text regression check.

Auth:
  Reads POSTHOG_PERSONAL_API_KEY, POSTHOG_PROJECT_ID, POSTHOG_HOST from
  the project ``.env``.

Discovery:
  The PostHog property carrying the raw query string is project-specific.
  By default we pull ``properties.query``; override with ``--query-prop``.
  Run with ``--inspect-properties`` to list the most-common property keys
  on ``search_performed`` events and pick the right one.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "validation"
DAYS_DEFAULT = 30
EVENT_NAME_DEFAULT = "search_performed"

EAN_RE = re.compile(r"^(\d{8}|\d{12,14})$")
HYPHEN_DIGIT_RE = re.compile(
    r"^(?=.{7,}$)(?=(?:[^\d]*\d){3,})[a-z0-9]+(?:-[a-z0-9]+)+$", re.IGNORECASE
)
ALPHA_DIGIT_RE = re.compile(r"^(?=.{7,}$)[a-z]+\d{4,}[a-z0-9]*$", re.IGNORECASE)
GENERIC_TOKENS = frozenset({
    "cr2032", "cr2025", "cr2016", "cr1632", "cr1620",
    "lr44", "lr41", "lr1130", "sr44", "sr41",
    "rj45", "rj11", "rj12",
    "usb-c", "usb-a", "usb-b", "hdmi", "displayport", "vga", "dvi-d",
    "cat5", "cat5e", "cat6", "cat6a", "cat7", "cat8",
    "ffp1", "ffp2", "ffp3", "n95", "n99", "kn95",
    "wd-40", "wd40",
    "m3", "m4", "m5", "m6", "m8", "m10", "m12", "m16", "m20",
    "ip54", "ip65", "ip66", "ip67", "ip68",
})


def hogql(host: str, project_id: str, api_key: str, query: str) -> dict:
    url = f"{host.rstrip('/')}/api/projects/{project_id}/query/"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"query": {"kind": "HogQLQuery", "query": query}}
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise SystemExit(
                f"PostHog HogQL error {r.status_code}: {r.text[:1000]}"
            )
        return r.json()


def inspect_properties(host: str, project_id: str, api_key: str,
                       event: str, days: int) -> None:
    """Print one full property bag from a recent event so the operator can
    identify the right ``--query-prop``."""
    print(f"Sample {event!r} properties (5 most-recent events):")
    q = f"""
        SELECT properties
        FROM events
        WHERE event = {event!r}
          AND timestamp > now() - INTERVAL {days} DAY
        ORDER BY timestamp DESC
        LIMIT 5
    """
    res = hogql(host, project_id, api_key, q)
    for i, row in enumerate(res.get("results", []), start=1):
        print(f"--- event {i} ---")
        print(row[0] if row else "(empty)")


def fetch_queries(host: str, project_id: str, api_key: str,
                  event: str, query_prop: str, days: int,
                  limit: int) -> list[tuple[str, int]]:
    # Trim leading/trailing whitespace and skip empty queries inside HogQL
    # so we get a clean distribution out the door.
    q = f"""
        SELECT
          trim(toString(properties.{query_prop})) AS q,
          count() AS n
        FROM events
        WHERE event = {event!r}
          AND timestamp > now() - INTERVAL {days} DAY
          AND trim(toString(properties.{query_prop})) != ''
        GROUP BY q
        ORDER BY n DESC
        LIMIT {limit}
    """
    res = hogql(host, project_id, api_key, q)
    return [(str(r[0]), int(r[1])) for r in res.get("results", [])]


def is_strict(q: str) -> bool:
    """Mirrors hybrid.is_strict_identifier without importing it (this script
    sits next to the import infrastructure but doesn't depend on it)."""
    q = q.strip().lower()
    if q in GENERIC_TOKENS:
        return False
    if not (4 <= len(q) <= 40):
        return False
    return bool(
        EAN_RE.fullmatch(q)
        or HYPHEN_DIGIT_RE.fullmatch(q)
        or ALPHA_DIGIT_RE.fullmatch(q)
    )


def write_tsv(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("query\tcount\n")
        for q, n in rows:
            f.write(f"{q}\t{n}\n")
    print(f"  wrote {len(rows):>6,} rows -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--days", type=int, default=DAYS_DEFAULT)
    p.add_argument("--event", default=EVENT_NAME_DEFAULT)
    p.add_argument("--query-prop", default="query",
                   help="PostHog property carrying the search query string "
                        "(default: 'query'). Use --inspect-properties to "
                        "discover the correct name for your project.")
    p.add_argument("--max-distinct", type=int, default=200_000,
                   help="Max distinct queries to pull from PostHog "
                        "(default: 200,000). Hits PostHog's row cap "
                        "before this only on extreme volume.")
    p.add_argument("--ean-sample", type=int, default=100,
                   help="How many EAN-shaped queries to write to "
                        "eans_seen.tsv (top by count, default: 100).")
    p.add_argument("--freetext-sample", type=int, default=100)
    p.add_argument("--inspect-properties", action="store_true",
                   help="List the most-common property keys on the event "
                        "and exit. Use this to find the right --query-prop.")
    args = p.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    host = os.environ.get("POSTHOG_HOST", "")
    project_id = os.environ.get("POSTHOG_PROJECT_ID", "")
    api_key = os.environ.get("POSTHOG_PERSONAL_API_KEY", "")
    if not (host and project_id and api_key):
        sys.exit(
            "Missing POSTHOG_HOST / POSTHOG_PROJECT_ID / "
            "POSTHOG_PERSONAL_API_KEY in environment"
        )

    if args.inspect_properties:
        inspect_properties(host, project_id, api_key, args.event, args.days)
        return

    print(f"Fetching distinct queries from {host} project {project_id}, "
          f"event={args.event!r}, last {args.days}d, prop={args.query_prop!r}…")
    rows = fetch_queries(
        host, project_id, api_key,
        event=args.event, query_prop=args.query_prop,
        days=args.days, limit=args.max_distinct,
    )
    print(f"  pulled {len(rows):,} distinct queries "
          f"(total events: {sum(n for _, n in rows):,})")

    write_tsv(args.out_dir / "posthog_queries.tsv", rows)
    write_tsv(args.out_dir / "top200_queries.tsv", rows[:200])

    # EAN-shaped: regex match. Keep top N by frequency.
    ean_rows = [(q, n) for q, n in rows if EAN_RE.fullmatch(q.strip())]
    write_tsv(args.out_dir / "eans_seen.tsv", ean_rows[: args.ean_sample])

    # Free-text: classifier rejects.
    freetext_rows = [(q, n) for q, n in rows if not is_strict(q)]
    write_tsv(
        args.out_dir / "freetext_seen.tsv",
        freetext_rows[: args.freetext_sample],
    )

    # Quick partition summary so the operator can sanity-check classifier
    # rates against hybrid_v0.md §Volume.
    strict_total = sum(n for q, n in rows if is_strict(q))
    grand_total = sum(n for _, n in rows)
    print(
        f"\nClassifier preview against the pulled distribution:"
        f"\n  strict-classified queries: "
        f"{strict_total:,} of {grand_total:,} events "
        f"({100 * strict_total / max(grand_total, 1):.1f}%)"
        f"\n  EAN-only:                  {sum(n for q, n in ean_rows):,} events"
    )


if __name__ == "__main__":
    main()
