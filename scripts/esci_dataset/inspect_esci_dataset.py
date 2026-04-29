#!/usr/bin/env python3
"""Sanity-check the curated queries + retrieved candidates."""
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = Path("/data/datasets/queries_offers_esci")
Q = ROOT / "queries.parquet"
C = ROOT / "candidates.parquet"


def section(s):
    print(f"\n=== {s} ===")


# ── queries.parquet ─────────────────────────────────────────────
section("queries.parquet")
qt = pq.read_table(Q)
print(f"rows: {qt.num_rows:,}")
print(f"size: {Q.stat().st_size/1e6:.2f} MB")
print(f"schema:")
for f in qt.schema:
    print(f"  {f.name:30} {f.type}")

print("\nfreq_band:")
for k, v in Counter(qt["frequency_band"].to_pylist()).most_common():
    print(f"  {k:8} {v:>6,}")

print("\nhit_band:")
for k, v in Counter(qt["hit_band"].to_pylist()).most_common():
    print(f"  {k:8} {v:>6,}")

print(f"\nMPN-shape: {sum(qt['mpn_shape'].to_pylist()):,}")

print(f"\nlanguages (top 10):")
for k, v in Counter(qt["platform_language"].to_pylist()).most_common(10):
    print(f"  {k!r:8} {v:>6,}")

# uniqueness check
keys = qt["normalized_qt"].to_pylist()
print(f"\nunique normalized_qt: {len(set(keys)):,} (should equal rows={qt.num_rows:,})")

# null counts
print("\nnulls per column (only non-zero shown):")
for f in qt.schema:
    n = pc.sum(pc.is_null(qt[f.name])).as_py() or 0
    if n:
        print(f"  {f.name}: {n}")

# qt_raw length distribution
lens = sorted(len(s) for s in qt["qt_raw"].to_pylist() if s)
def pct(p): return lens[int(p * (len(lens) - 1))]
print(f"\nqt_raw length: p10={pct(.1)}  p50={pct(.5)}  p90={pct(.9)}  p99={pct(.99)}  max={lens[-1]}")

# samples
print("\n10 random qt_raw samples:")
import random; random.seed(1)
for i in random.sample(range(qt.num_rows), 10):
    row = {f.name: qt[f.name][i].as_py() for f in qt.schema}
    print(f"  qid={row['query_id']:>5}  qt={row['qt_raw']!r:50}  fb={row['frequency_band']:5}  hb={row['hit_band']:8}  mpn={row['mpn_shape']}  hits={row['hit_count_at_search_time']}")


# ── candidates.parquet ──────────────────────────────────────────
section("candidates.parquet")
ct = pq.read_table(C)
print(f"rows: {ct.num_rows:,}")
print(f"size: {C.stat().st_size/1e6:.2f} MB")
print(f"schema:")
for f in ct.schema:
    print(f"  {f.name:30} {f.type}")

# distinct query coverage
ct_qids = set(ct["query_id"].to_pylist())
print(f"\nqueries with >=1 candidate: {len(ct_qids):,}/{qt.num_rows:,}")
missing = set(qt["query_id"].to_pylist()) - ct_qids
print(f"queries with 0 candidates: {len(missing):,}")
if missing:
    samples = sorted(missing)[:10]
    print(f"  first 10 missing query_ids: {samples}")

# candidates per query
per_q = Counter(ct["query_id"].to_pylist())
counts = sorted(per_q.values())
def cpct(p): return counts[int(p * (len(counts) - 1))]
print(f"\ncandidates per query: min={counts[0]}  p10={cpct(.1)}  p50={cpct(.5)}  p90={cpct(.9)}  p99={cpct(.99)}  max={counts[-1]}  mean={sum(counts)/len(counts):.1f}")

# source_legs distribution
leg_counts = Counter()
mode_only = Counter()  # how many candidates exclusively from one mode
hc_count = vec_count = bm25_count = 0
for legs in ct["source_legs"].to_pylist():
    legs = tuple(legs)
    leg_counts[legs] += 1
print(f"\nsource_legs combinations:")
for k, v in leg_counts.most_common(10):
    print(f"  {k}: {v:,}")

# How many candidates appeared in each mode
print(f"\ncandidates appearing in each mode:")
for mode in ("hybrid_classified", "vector", "bm25"):
    n = pc.sum(pc.is_valid(ct[f"rank_{mode}"])).as_py()
    print(f"  {mode}: {n:,}  ({n/ct.num_rows*100:.1f}%)")

# overlap stats
both_dense_bm25 = 0
all_three_modes = 0
for hr, vr, br in zip(ct["rank_hybrid_classified"].to_pylist(),
                      ct["rank_vector"].to_pylist(),
                      ct["rank_bm25"].to_pylist()):
    if hr is not None and vr is not None and br is not None:
        all_three_modes += 1
    if vr is not None and br is not None:
        both_dense_bm25 += 1
print(f"\ncandidates from vector AND bm25: {both_dense_bm25:,}")
print(f"candidates from all 3 modes:      {all_three_modes:,}")

# Per (query) hit-band breakdown
print(f"\ncandidates per query, by hit_band:")
qb = dict(zip(qt["query_id"].to_pylist(), qt["hit_band"].to_pylist()))
hb_counts = Counter()
hb_qcount = Counter()
for qid, n in per_q.items():
    hb_counts[qb[qid]] += n
    hb_qcount[qb[qid]] += 1
for hb in ("zero", "1-9", "10-99", "100-999", "1000+"):
    if hb_qcount[hb]:
        print(f"  {hb:8}  queries={hb_qcount[hb]:,}  total_cands={hb_counts[hb]:,}  mean={hb_counts[hb]/hb_qcount[hb]:.1f}")

# Score sanity
print(f"\nscore ranges (non-null):")
for col in ("score_hybrid_classified", "score_vector", "score_bm25"):
    s = [x for x in ct[col].to_pylist() if x is not None]
    if s:
        s.sort()
        print(f"  {col}: min={s[0]:.4f}  p50={s[len(s)//2]:.4f}  max={s[-1]:.4f}")

print()
