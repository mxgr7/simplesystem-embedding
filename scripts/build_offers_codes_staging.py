"""Build the staging parquet for the `offers_codes` Milvus collection.

Source: /data/datasets/offers_embedded_full.parquet/bucket=NN.parquet (16 buckets,
~159.3M rows). For each row, build `text_codes` from the surviving values of
`ean`, `manufacturerArticleNumber`, `article_number` after the structural
filters and global frequency cap from `hybrid_v0.md §text_codes ingest filter`.

Filters (matched lowercase, applied per field):
  Universal — drop if length not in [4, 40], or matches ^0+$ / ^9+$ /
              ^[-_.\\s]+$, or in {k.a., n/a, n.a., null, aucune donnée, #ref!},
              or contains any whitespace.
  EAN extra — length must be in {8, 12, 13, 14}.
  MPN/article extras — drop if pure-letter ([a-zäöüß]+), or Excel-mangled
              scientific notation (contains 'e+' AND wholly digits/punct/Ee+-).

Frequency cap (from §"Frequency cap"):
  Per-row dedupe: count occurrences across DISTINCT (id, lower(value)) pairs.
  Drop any value with count > 500.

Output:
  /data/datasets/offers_codes_staging.parquet/bucket=NN.parquet
    columns: id (string), text_codes (string)
  reports/codes_audit.md  — survivor counts, denylist, sanity checks.

Skipping rule: if all three identifier values drop for a row, the row is
omitted entirely (no PK, no text). The doc's safety rationale: a blank
text_codes would either index nothing or accidentally match the empty query.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

DEFAULT_SRC_DIR = Path("/data/datasets/offers_embedded_full.parquet")
DEFAULT_OUT_DIR = Path("/data/datasets/offers_codes_staging.parquet")
DEFAULT_REPORT = Path(__file__).resolve().parent.parent / "reports" / "codes_audit.md"

FREQ_CAP = 500
NUM_BUCKETS = 16

# Sanity probes: each (value, expected_field_keep) pair lets us catch a
# regressed filter before kicking off the full materialisation. Frequency-cap
# values are checked separately after the denylist is computed.
STRUCTURAL_PROBES: list[tuple[str, str, bool]] = [
    ("00000000",       "ean",  False),  # all-zero placeholder
    ("9999999999999",  "ean",  False),  # all-nines placeholder
    ("n/a",            "mpn",  False),  # literal placeholder
    ("magnet",         "mpn",  False),  # pure letter
    ("8,45601e+11",    "mpn",  False),  # Excel sci-notation
    ("bosch gmbh",     "mpn",  False),  # has whitespace
    ("---",            "mpn",  False),  # < length floor anyway, but covered
    ("4031100000000",  "ean",  True),   # legit EAN shape (caught by freq cap)
    ("rj45",           "mpn",  True),
    ("tze-231",        "mpn",  True),
    ("h07v-k",         "mpn",  True),
    ("4012345678901",  "ean",  True),
]

POST_DENYLIST_PROBES = ["00000000", "4031100000000", "n/a", "magnet"]


def install_macros(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(r"""
CREATE OR REPLACE MACRO is_universal_ok(v) AS (
  v IS NOT NULL
  AND length(v) BETWEEN 4 AND 40
  AND NOT regexp_matches(v, '^0+$')
  AND NOT regexp_matches(v, '^9+$')
  AND NOT regexp_matches(v, '^[-_.[:space:]]+$')
  AND v NOT IN ('k.a.', 'n/a', 'n.a.', 'null', 'aucune donnée', '#ref!')
  AND NOT regexp_matches(v, '\s')
);
CREATE OR REPLACE MACRO is_ean_ok(v) AS (
  is_universal_ok(v) AND length(v) IN (8, 12, 13, 14)
);
CREATE OR REPLACE MACRO is_mpn_ok(v) AS (
  is_universal_ok(v)
  AND NOT regexp_matches(v, '^[a-zäöüß]+$')
  AND NOT (v LIKE '%e+%' AND regexp_matches(v, '^[\d,.eE+\-]+$'))
);
""")


def run_structural_probes(con: duckdb.DuckDBPyConnection) -> list[str]:
    failures: list[str] = []
    for value, field, expected in STRUCTURAL_PROBES:
        macro = {"ean": "is_ean_ok", "mpn": "is_mpn_ok"}[field]
        got = con.execute(f"SELECT {macro}(?)", [value]).fetchone()[0]
        if got != expected:
            failures.append(
                f"  {field}({value!r}): expected={expected} got={got}"
            )
    return failures


def compute_denylist(con: duckdb.DuckDBPyConnection, src_glob: str) -> int:
    con.execute(f"""
CREATE OR REPLACE TEMP VIEW _filtered AS
SELECT id,
  CASE WHEN is_ean_ok(lower(ean)) THEN lower(ean) END AS ean_v,
  CASE WHEN is_mpn_ok(lower(manufacturerArticleNumber))
       THEN lower(manufacturerArticleNumber) END AS mpn_v,
  CASE WHEN is_mpn_ok(lower(article_number)) THEN lower(article_number) END AS art_v
FROM read_parquet('{src_glob}', hive_partitioning=true);

CREATE OR REPLACE TEMP TABLE _per_row_distinct AS
SELECT DISTINCT id, v FROM (
  SELECT id, ean_v AS v FROM _filtered WHERE ean_v IS NOT NULL
  UNION ALL
  SELECT id, mpn_v FROM _filtered WHERE mpn_v IS NOT NULL
  UNION ALL
  SELECT id, art_v FROM _filtered WHERE art_v IS NOT NULL
);

CREATE OR REPLACE TABLE _denylist AS
SELECT v, COUNT(*) AS n
FROM _per_row_distinct
GROUP BY v
HAVING COUNT(*) > ?;
""", [FREQ_CAP])
    return con.execute("SELECT COUNT(*) FROM _denylist").fetchone()[0]


def materialise_bucket(
    con: duckdb.DuckDBPyConnection,
    src_path: Path,
    out_path: Path,
) -> tuple[int, int, int, int, int]:
    """Returns (rows_in, rows_out, rows_with_ean, rows_with_mpn, rows_with_art).
    Counts after both structural + frequency-cap filters."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con.execute(f"""
CREATE OR REPLACE TEMP TABLE _bucket AS
SELECT id,
  CASE WHEN is_ean_ok(lower(ean))
       AND lower(ean) NOT IN (SELECT v FROM _denylist)
       THEN lower(ean) END AS ean_v,
  CASE WHEN is_mpn_ok(lower(manufacturerArticleNumber))
       AND lower(manufacturerArticleNumber) NOT IN (SELECT v FROM _denylist)
       THEN lower(manufacturerArticleNumber) END AS mpn_v,
  CASE WHEN is_mpn_ok(lower(article_number))
       AND lower(article_number) NOT IN (SELECT v FROM _denylist)
       THEN lower(article_number) END AS art_v
FROM read_parquet('{src_path}');
""")

    rows_in = con.execute("SELECT COUNT(*) FROM _bucket").fetchone()[0]
    rows_with_ean, rows_with_mpn, rows_with_art = con.execute(
        "SELECT count(ean_v), count(mpn_v), count(art_v) FROM _bucket"
    ).fetchone()

    # array_to_string(list_distinct(list_filter([…], x->x IS NOT NULL)), ' ')
    # collapses per-row duplicates (eg ean == article_number) into a single
    # token before BM25 sees them.
    con.execute(f"""
COPY (
  SELECT id,
    array_to_string(
      list_distinct(list_filter([ean_v, mpn_v, art_v], x -> x IS NOT NULL)),
      ' '
    ) AS text_codes
  FROM _bucket
  WHERE ean_v IS NOT NULL OR mpn_v IS NOT NULL OR art_v IS NOT NULL
) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
""")
    rows_out = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_path}')"
    ).fetchone()[0]

    return rows_in, rows_out, rows_with_ean, rows_with_mpn, rows_with_art


def write_report(
    report_path: Path,
    *,
    src_dir: Path,
    out_dir: Path,
    totals: dict,
    denylist_size: int,
    denylist_sample: list[tuple[str, int]],
    distinct_indexed: dict,
    sanity_hits: dict,
    timing_s: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# offers_codes — staging audit",
        "",
        f"- Source: `{src_dir}`",
        f"- Output: `{out_dir}/bucket=NN.parquet` (16 files)",
        f"- Frequency cap: > {FREQ_CAP} occurrences",
        f"- Total wall time: {timing_s:.1f}s",
        "",
        "## Row-level survival",
        "",
        f"- rows in source: {totals['rows_in']:,}",
        f"- rows kept (≥1 surviving identifier): {totals['rows_out']:,}",
        f"- rows skipped (all three dropped): {totals['rows_in'] - totals['rows_out']:,}",
        "",
        "## Per-field survival (counted on kept rows; one row may contribute to multiple fields)",
        "",
        f"- rows with surviving EAN: {totals['rows_with_ean']:,}",
        f"- rows with surviving MPN: {totals['rows_with_mpn']:,}",
        f"- rows with surviving article_number: {totals['rows_with_art']:,}",
        "",
        "## Distinct values indexed",
        "",
        f"- distinct EAN values: {distinct_indexed['ean']:,}",
        f"- distinct MPN values: {distinct_indexed['mpn']:,}",
        f"- distinct article_number values: {distinct_indexed['art']:,}",
        "",
        "## Frequency-cap denylist",
        "",
        f"- size: {denylist_size}",
        "- top 30 by occurrence count:",
        "",
        "  | value | count |",
        "  |---|---:|",
    ]
    for v, n in denylist_sample:
        lines.append(f"  | `{v}` | {n:,} |")
    lines += [
        "",
        "## Sanity probes (BM25 query → match in materialised text_codes)",
        "",
        "  | query | rows containing as token |",
        "  |---|---:|",
    ]
    for q, n in sanity_hits.items():
        lines.append(f"  | `{q}` | {n:,} |")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--memory-limit", default="64GB")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--temp-dir", type=Path, default=Path("/tmp/duckdb_offers_codes"))
    args = p.parse_args()

    src_files = sorted(args.src.glob("bucket=*.parquet"))
    if len(src_files) != NUM_BUCKETS:
        raise SystemExit(
            f"Expected {NUM_BUCKETS} buckets in {args.src}, found {len(src_files)}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{args.temp_dir}'")
    install_macros(con)

    print("Running structural-filter probes…", flush=True)
    fails = run_structural_probes(con)
    if fails:
        raise SystemExit("Structural probe failures:\n" + "\n".join(fails))
    print(f"  {len(STRUCTURAL_PROBES)} probes OK", flush=True)

    wall0 = time.time()

    print("\nPass 1: computing frequency-cap denylist over all 16 buckets…", flush=True)
    src_glob = str(args.src / "bucket=*.parquet")
    t0 = time.time()
    denylist_size = compute_denylist(con, src_glob)
    print(f"  denylist size: {denylist_size}  ({time.time()-t0:.1f}s)", flush=True)
    denylist_sample = con.execute(
        "SELECT v, n FROM _denylist ORDER BY n DESC LIMIT 30"
    ).fetchall()

    print("\nPass 2: materialising per-bucket staging parquets…", flush=True)
    totals = {"rows_in": 0, "rows_out": 0,
              "rows_with_ean": 0, "rows_with_mpn": 0, "rows_with_art": 0}
    for src_path in src_files:
        out_path = args.out / src_path.name
        t0 = time.time()
        rin, rout, re_, rm, ra = materialise_bucket(con, src_path, out_path)
        totals["rows_in"] += rin
        totals["rows_out"] += rout
        totals["rows_with_ean"] += re_
        totals["rows_with_mpn"] += rm
        totals["rows_with_art"] += ra
        print(
            f"  {src_path.name}: in={rin:>10,}  out={rout:>10,}  "
            f"ean={re_:>10,} mpn={rm:>10,} art={ra:>10,}  "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )

    out_glob = str(args.out / "bucket=*.parquet")
    print("\nComputing distinct-value counts on the materialised data…", flush=True)
    # Recover per-field distinct counts by re-tokenising the source under the
    # final filters (denylist included). Cheaper than re-reading text_codes.
    distinct_indexed = dict(zip(
        ["ean", "mpn", "art"],
        con.execute(f"""
          WITH s AS (
            SELECT
              CASE WHEN is_ean_ok(lower(ean))
                   AND lower(ean) NOT IN (SELECT v FROM _denylist)
                   THEN lower(ean) END AS ean_v,
              CASE WHEN is_mpn_ok(lower(manufacturerArticleNumber))
                   AND lower(manufacturerArticleNumber) NOT IN (SELECT v FROM _denylist)
                   THEN lower(manufacturerArticleNumber) END AS mpn_v,
              CASE WHEN is_mpn_ok(lower(article_number))
                   AND lower(article_number) NOT IN (SELECT v FROM _denylist)
                   THEN lower(article_number) END AS art_v
            FROM read_parquet('{src_glob}')
          )
          SELECT
            (SELECT count(DISTINCT ean_v) FROM s WHERE ean_v IS NOT NULL),
            (SELECT count(DISTINCT mpn_v) FROM s WHERE mpn_v IS NOT NULL),
            (SELECT count(DISTINCT art_v) FROM s WHERE art_v IS NOT NULL)
        """).fetchone(),
    ))

    print("\nSanity probes against the materialised text_codes corpus…", flush=True)
    sanity_hits: dict[str, int] = {}
    for q in POST_DENYLIST_PROBES:
        # Whitespace tokenisation; match q as a stand-alone token.
        n = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{out_glob}')
            WHERE regexp_matches(text_codes, '(^|\\s)' || ? || '($|\\s)')
            """,
            [q],
        ).fetchone()[0]
        sanity_hits[q] = n
        print(f"  {q!r}: {n:,} rows", flush=True)

    timing_s = time.time() - wall0
    print(f"\nWriting audit report to {args.report}", flush=True)
    write_report(
        args.report,
        src_dir=args.src,
        out_dir=args.out,
        totals=totals,
        denylist_size=denylist_size,
        denylist_sample=denylist_sample,
        distinct_indexed=distinct_indexed,
        sanity_hits=sanity_hits,
        timing_s=timing_s,
    )

    print(f"\nDone in {timing_s:.1f}s. Output: {args.out}")
    print(f"  rows kept: {totals['rows_out']:,} / {totals['rows_in']:,} "
          f"({100*totals['rows_out']/max(totals['rows_in'],1):.2f}%)")


if __name__ == "__main__":
    main()
