#!/usr/bin/env python3
"""Compare two annotator runs over the same input rows.

Usage:
    python compare_annotation_iterations.py <iter_a.jsonl> <iter_b.jsonl>

Both files must be runs over the same dataset (same example_id space).
"""
import json
import sys
from collections import Counter


def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        ex = r["row"]["example_id"]
        msg = (r["response"]["choices"][0] or {}).get("message") or {}
        try:
            label = json.loads(msg["content"]).get("label")
        except Exception:
            label = None
        rows[ex] = (label, r["row"])
    return rows


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = load(a_path), load(b_path)
    common = set(a) & set(b)
    print(f"A: {a_path}  ({len(a)} rows)")
    print(f"B: {b_path}  ({len(b)} rows)")
    print(f"common: {len(common)}")

    a_dist = Counter(a[ex][0] for ex in common)
    b_dist = Counter(b[ex][0] for ex in common)
    print(f"\nA distribution: {dict(a_dist)}")
    print(f"B distribution: {dict(b_dist)}")

    # Transition matrix
    transitions = Counter()
    for ex in common:
        transitions[(a[ex][0], b[ex][0])] += 1

    labels = ["Exact", "Substitute", "Complement", "Irrelevant"]
    print(f"\ntransition matrix (rows=A, cols=B):")
    print(f"  {'':14}" + "".join(f"{c:>12}" for c in labels))
    for r in labels:
        row = [transitions.get((r, c), 0) for c in labels]
        print(f"  {r:14}" + "".join(f"{v:>12}" for v in row))

    n_changed = sum(v for (a_, b_), v in transitions.items() if a_ != b_)
    print(f"\ntotal changed: {n_changed} / {len(common)}")

    # Show changed rows that look like fixes (Exact → Irrelevant) — limit to 8
    print(f"\nrows changed Exact → Irrelevant (likely fixes):")
    for ex in common:
        if a[ex][0] == "Exact" and b[ex][0] == "Irrelevant":
            row = a[ex][1]
            name = (row["name"] or "")[:80]
            print(f"  qid={row['query_id']:>5}  Q={row['query_term']!r:35} P={name!r}")

    print(f"\nrows changed Irrelevant → Exact (potential regressions):")
    for ex in common:
        if a[ex][0] == "Irrelevant" and b[ex][0] == "Exact":
            row = a[ex][1]
            name = (row["name"] or "")[:80]
            print(f"  qid={row['query_id']:>5}  Q={row['query_term']!r:35} P={name!r}")

    print(f"\nrows changed Exact → Substitute (likely fixes for generic queries):")
    cnt = 0
    for ex in common:
        if a[ex][0] == "Exact" and b[ex][0] == "Substitute":
            row = a[ex][1]
            name = (row["name"] or "")[:80]
            print(f"  qid={row['query_id']:>5}  Q={row['query_term']!r:35} P={name!r}")
            cnt += 1
            if cnt >= 10: break


if __name__ == "__main__":
    main()
