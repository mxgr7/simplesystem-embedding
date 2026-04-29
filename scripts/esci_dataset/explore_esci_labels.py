#!/usr/bin/env python3
"""Quality exploration of the labeled ESCI dataset.

Run after extract_esci_labels.py has produced
/data/datasets/queries_offers_esci/queries_offers_merged_labeled.parquet.

Things we check:
  1. Label distribution overall and per (frequency_band, hit_band, source_legs).
  2. Sanity: does the label correlate the way we'd expect with retrieval rank?
     (Top-ranked hybrid_classified candidates should skew Exact.)
  3. Per-query distribution: how many queries have only-Irrelevant or
     only-Exact result sets? Either extreme is interesting.
  4. The 100 cached rows from iter3: their labels should match exactly.
  5. Stratified random samples per label for visual verification.
  6. Top queries by label volume — do head queries have clean labels?
  7. Parse-failure / unlabeled rows: how many, and any pattern?
"""
import json
import random
from pathlib import Path

import duckdb

ROOT = Path("/data/datasets/queries_offers_esci")
LABELED = ROOT / "queries_offers_merged_labeled.parquet" / "part-0.parquet"
ITER3 = "/home/mgerer/annotatorv3/outputs/2026-04-28/21-06-46/annotations-batch.jsonl"

con = duckdb.connect()
con.execute("PRAGMA threads=8")


def section(s):
    print(f"\n=== {s} ===")


def sql(q, fetch=True):
    r = con.execute(q)
    return r.fetchall() if fetch else r


# ── 1. Overall distribution ───────────────────────────────────────
section("Overall label distribution")
total, labeled, null = sql(f"""
    SELECT COUNT(*), COUNT(label),
           SUM(CASE WHEN label IS NULL THEN 1 ELSE 0 END)
    FROM read_parquet('{LABELED}')
""")[0]
print(f"  total rows:  {total:,}")
print(f"  labeled:     {labeled:,} ({labeled/total*100:.2f}%)")
print(f"  unlabeled:   {null:,} ({null/total*100:.2f}%)")

print()
for label, n in sql(f"""
    SELECT label, COUNT(*) FROM read_parquet('{LABELED}')
    GROUP BY 1 ORDER BY 2 DESC
"""):
    print(f"  {str(label):11} {n:>8,}  ({n/total*100:5.2f}%)")


# ── 2. Distribution by hit_band ───────────────────────────────────
section("Label × hit_band")
rows = sql(f"""
    SELECT hit_band,
           COUNT(*) AS n,
           SUM(CASE WHEN label='Exact'      THEN 1 ELSE 0 END) AS e,
           SUM(CASE WHEN label='Substitute' THEN 1 ELSE 0 END) AS s,
           SUM(CASE WHEN label='Complement' THEN 1 ELSE 0 END) AS c,
           SUM(CASE WHEN label='Irrelevant' THEN 1 ELSE 0 END) AS i,
           SUM(CASE WHEN label IS NULL      THEN 1 ELSE 0 END) AS u
    FROM read_parquet('{LABELED}')
    GROUP BY 1 ORDER BY 1
""")
print(f"  {'hit_band':10} {'n':>8} {'E%':>6} {'S%':>6} {'C%':>6} {'I%':>6} {'∅':>5}")
for hb, n, e, s, c, i, u in rows:
    p = lambda x: f"{x/n*100:5.1f}" if n else "-"
    print(f"  {str(hb):10} {n:>8,} {p(e):>6} {p(s):>6} {p(c):>6} {p(i):>6} {u:>5,}")


# ── 3. Distribution by frequency_band ─────────────────────────────
section("Label × frequency_band")
for fb, n, e, s, c, i, u in sql(f"""
    SELECT frequency_band,
           COUNT(*),
           SUM(CASE WHEN label='Exact'      THEN 1 ELSE 0 END),
           SUM(CASE WHEN label='Substitute' THEN 1 ELSE 0 END),
           SUM(CASE WHEN label='Complement' THEN 1 ELSE 0 END),
           SUM(CASE WHEN label='Irrelevant' THEN 1 ELSE 0 END),
           SUM(CASE WHEN label IS NULL      THEN 1 ELSE 0 END)
    FROM read_parquet('{LABELED}')
    GROUP BY 1 ORDER BY 1
"""):
    p = lambda x: f"{x/n*100:5.1f}" if n else "-"
    print(f"  {str(fb):8} {n:>8,}  E={p(e):>5}  S={p(s):>5}  C={p(c):>5}  I={p(i):>5}  ∅={u}")


# ── 4. Distribution by source_legs ────────────────────────────────
section("Label × source_legs (top 8 combinations)")
for legs, n, e, s, c, i in sql(f"""
    SELECT array_to_string(source_legs, ','),
           COUNT(*),
           SUM(CASE WHEN label='Exact'      THEN 1 ELSE 0 END),
           SUM(CASE WHEN label='Substitute' THEN 1 ELSE 0 END),
           SUM(CASE WHEN label='Complement' THEN 1 ELSE 0 END),
           SUM(CASE WHEN label='Irrelevant' THEN 1 ELSE 0 END)
    FROM read_parquet('{LABELED}')
    GROUP BY 1 ORDER BY 2 DESC
    LIMIT 8
"""):
    p = lambda x: f"{x/n*100:5.1f}"
    print(f"  {str(legs):20} {n:>7,}  E={p(e)} S={p(s)} C={p(c)} I={p(i)}")


# ── 5. Label vs hybrid_classified rank ────────────────────────────
section("Label by rank_hybrid_classified bucket (lower rank = higher in result list)")
rows = sql(f"""
    WITH t AS (
        SELECT
            CASE
                WHEN rank_hybrid_classified IS NULL THEN '∅'
                WHEN rank_hybrid_classified <= 3   THEN '01-03'
                WHEN rank_hybrid_classified <= 10  THEN '04-10'
                WHEN rank_hybrid_classified <= 20  THEN '11-20'
                ELSE '21+'
            END AS rank_bucket,
            label
        FROM read_parquet('{LABELED}')
    )
    SELECT rank_bucket,
           COUNT(*) AS n,
           SUM(CASE WHEN label='Exact'      THEN 1 ELSE 0 END) AS e,
           SUM(CASE WHEN label='Substitute' THEN 1 ELSE 0 END) AS s,
           SUM(CASE WHEN label='Complement' THEN 1 ELSE 0 END) AS c,
           SUM(CASE WHEN label='Irrelevant' THEN 1 ELSE 0 END) AS i
    FROM t
    GROUP BY 1 ORDER BY 1
""")
print(f"  {'rank':6} {'n':>9} {'E%':>5} {'S%':>5} {'C%':>5} {'I%':>5}")
for rb, n, e, s, c, i in rows:
    p = lambda x: f"{x/n*100:4.1f}"
    print(f"  {rb:6} {n:>9,} {p(e):>5} {p(s):>5} {p(c):>5} {p(i):>5}")


# ── 6. Per-query label distribution ───────────────────────────────
section("Per-query label spread")
n_queries, n_only_irr, n_only_exact, n_no_exact, n_at_least_1_exact = sql(f"""
    WITH per_q AS (
        SELECT query_id,
               COUNT(*) AS n,
               SUM(CASE WHEN label='Exact'      THEN 1 ELSE 0 END) AS e,
               SUM(CASE WHEN label='Substitute' THEN 1 ELSE 0 END) AS s,
               SUM(CASE WHEN label='Complement' THEN 1 ELSE 0 END) AS c,
               SUM(CASE WHEN label='Irrelevant' THEN 1 ELSE 0 END) AS i
        FROM read_parquet('{LABELED}')
        GROUP BY 1
    )
    SELECT COUNT(*),
           SUM(CASE WHEN i=n THEN 1 ELSE 0 END),
           SUM(CASE WHEN e=n THEN 1 ELSE 0 END),
           SUM(CASE WHEN e=0 THEN 1 ELSE 0 END),
           SUM(CASE WHEN e>=1 THEN 1 ELSE 0 END)
    FROM per_q
""")[0]
print(f"  total queries:                {n_queries:,}")
print(f"  all candidates Irrelevant:    {n_only_irr:,} ({n_only_irr/n_queries*100:5.1f}%)")
print(f"  all candidates Exact:         {n_only_exact:,} ({n_only_exact/n_queries*100:5.1f}%)")
print(f"  no Exact in any candidate:    {n_no_exact:,} ({n_no_exact/n_queries*100:5.1f}%)")
print(f"  >=1 Exact:                    {n_at_least_1_exact:,} ({n_at_least_1_exact/n_queries*100:5.1f}%)")


# ── 7. Random per-label spot-check ─────────────────────────────────
section("Random spot-checks (8 per label)")
random.seed(42)
for label in ("Exact", "Substitute", "Complement", "Irrelevant"):
    print(f"\n  --- {label} ---")
    rows = sql(f"""
        SELECT query_term, name, hit_band, source_legs
        FROM (
            SELECT query_term, name, hit_band, source_legs
            FROM read_parquet('{LABELED}')
            WHERE label = '{label}'
        )
        ORDER BY random()
        LIMIT 8
    """)
    for qt, name, hb, legs in rows:
        n = (name or "")[:65]
        print(f"    [{hb:8}|{','.join(legs or []):11}]  Q={qt!r:35}  P={n!r}")


# ── 8. Cross-check the iter3 sample100 cached rows ────────────────
section("Cross-check: 100 rows from iter3 sample (should be cache-identical)")
iter3_labels = {}
for line in open(ITER3):
    r = json.loads(line)
    ex = r["row"]["example_id"]
    msg = (r["response"]["choices"][0] or {}).get("message") or {}
    try:
        iter3_labels[ex] = json.loads(msg["content"])["label"]
    except Exception:
        iter3_labels[ex] = None

# Build duckdb relation of iter3 expected and join to labeled
import pyarrow as pa
exp_table = pa.table({
    "example_id": list(iter3_labels.keys()),
    "expected_label": list(iter3_labels.values()),
})
con.register("iter3_expected", con.from_arrow(exp_table))
mismatches = sql(f"""
    SELECT lab.example_id, lab.label AS got, exp.expected_label AS expected,
           lab.query_term, lab.name
    FROM read_parquet('{LABELED}') lab
    JOIN iter3_expected exp ON lab.example_id = exp.example_id
    WHERE lab.label IS DISTINCT FROM exp.expected_label
""")
print(f"  100 expected; mismatches: {len(mismatches)}")
for ex, got, exp_, qt, name in mismatches[:10]:
    n = (name or "")[:50]
    print(f"    ex={ex} got={got!r} expected={exp_!r}  Q={qt!r}  P={n!r}")


# ── 9. Top head queries — what did the model do? ─────────────────
section("Top 10 head queries (by event volume): label distribution")
rows = sql(f"""
    SELECT query_term,
           COUNT(*) AS n,
           SUM(CASE WHEN label='Exact'      THEN 1 ELSE 0 END) AS e,
           SUM(CASE WHEN label='Substitute' THEN 1 ELSE 0 END) AS s,
           SUM(CASE WHEN label='Complement' THEN 1 ELSE 0 END) AS c,
           SUM(CASE WHEN label='Irrelevant' THEN 1 ELSE 0 END) AS i
    FROM read_parquet('{LABELED}')
    WHERE frequency_band = 'head'
    GROUP BY 1
    ORDER BY MAX(hit_count_at_search_time) DESC NULLS LAST
    LIMIT 10
""")
for qt, n, e, s, c, i in rows:
    print(f"  {qt!r:40}  n={n:3}  E={e:2}  S={s:2}  C={c:2}  I={i:2}")
