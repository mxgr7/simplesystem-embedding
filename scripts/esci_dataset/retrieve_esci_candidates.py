#!/usr/bin/env python3
"""Retrieve search-api candidates for the curated ESCI query set.

For each query, hit POST /offers/_search three times:
  - mode=hybrid_classified, k=30  (production candidate set)
  - mode=vector,            k=20  (dense-only — semantic substitutes)
  - mode=bm25,              k=20  (lexical-only — exact + harder negatives)

Results are unioned by candidate `_id`. We persist only the id (no
materialized product fields — those are hydrated in a later step) plus
per-mode rank, per-mode score, and the set of legs that surfaced it.

Output: /data/datasets/queries_offers_esci/candidates.parquet
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

ENV_PATH = "/home/mgerer/shared/.env"
for line in Path(ENV_PATH).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k, v.strip().strip('"').strip("'"))

API_URL = "http://localhost:8001/offers/_search"
API_KEY = os.environ.get("SEARCH_API_KEY") or "r2zFFDMQRod15xIvo+7Pvf5ei2y6OX7XqrdOPCKo"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

QUERIES_PATH = Path("/data/datasets/queries_offers_esci/queries.parquet")
OUT_PATH = Path("/data/datasets/queries_offers_esci/candidates.parquet")

MODES = [
    ("hybrid_classified", 30),
    ("vector", 20),
    ("bm25", 20),
]
WORKERS = 8
MAX_RETRIES = 4
TIMEOUT_S = 30


def call_one(session: requests.Session, query: str, mode: str, k: int) -> list[dict]:
    """One search call. Returns hits list (possibly empty)."""
    body = {"query": query, "index": "offers-prod"}
    params = {"mode": mode, "k": k}
    for attempt in range(MAX_RETRIES):
        try:
            r = session.post(API_URL, params=params, headers=HEADERS,
                             json=body, timeout=TIMEOUT_S)
            if r.status_code == 503:
                # Concurrency limit — back off
                time.sleep(0.5 * (2 ** attempt))
                continue
            r.raise_for_status()
            return r.json().get("hits", []) or []
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(0.3 * (2 ** attempt))
    return []


def fetch_query(query_id: int, query_text: str) -> list[dict]:
    """Run all three modes for one query and union by _id."""
    sess = requests.Session()
    union: dict[str, dict] = {}
    for mode, k in MODES:
        hits = call_one(sess, query_text, mode, k)
        for rank, h in enumerate(hits, start=1):
            cid = h["_id"]
            rec = union.setdefault(cid, {
                "query_id": query_id,
                "candidate_id": cid,
                "rank_hybrid_classified": None,
                "rank_vector": None,
                "rank_bm25": None,
                "score_hybrid_classified": None,
                "score_vector": None,
                "score_bm25": None,
                "source_legs": set(),
            })
            rec[f"rank_{mode}"] = rank
            rec[f"score_{mode}"] = h["_score"]
            rec["source_legs"].add(h["_source_leg"])
    return list(union.values())


def main():
    print(f"[retrieve] reading queries from {QUERIES_PATH}")
    qt_table = pq.read_table(QUERIES_PATH, columns=["query_id", "qt_raw"])
    query_ids = qt_table["query_id"].to_pylist()
    queries = qt_table["qt_raw"].to_pylist()
    n = len(query_ids)
    print(f"[retrieve] {n} queries; modes={[m for m,_ in MODES]}; workers={WORKERS}")

    rows: list[dict] = []
    failed: list[tuple[int, str, str]] = []
    t0 = time.time()
    next_log = 500

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch_query, qid, qt): qid
                   for qid, qt in zip(query_ids, queries)}
        done = 0
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                rs = fut.result()
                rows.extend(rs)
            except Exception as e:
                failed.append((qid, "", str(e)))
            done += 1
            if done >= next_log:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (n - done) / rate
                print(f"  {done:>6}/{n}  rate={rate:5.1f}/s  "
                      f"rows={len(rows):>7,}  elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
                      f"failed={len(failed)}")
                next_log += 500

    elapsed = time.time() - t0
    print(f"[retrieve] done in {elapsed:.0f}s. {len(rows):,} (query, candidate) rows; "
          f"{len(failed)} query failures")

    # Stable order: by query_id, then by best (lowest) rank across modes
    def best_rank(r):
        return min(x for x in (r["rank_hybrid_classified"], r["rank_vector"],
                               r["rank_bm25"]) if x is not None)
    rows.sort(key=lambda r: (r["query_id"], best_rank(r)))

    retrieved_at = datetime.now(timezone.utc)
    table = pa.table({
        "query_id": pa.array([r["query_id"] for r in rows], type=pa.int32()),
        "candidate_id": [r["candidate_id"] for r in rows],
        "rank_hybrid_classified": pa.array(
            [r["rank_hybrid_classified"] for r in rows], type=pa.int16()),
        "rank_vector": pa.array(
            [r["rank_vector"] for r in rows], type=pa.int16()),
        "rank_bm25": pa.array(
            [r["rank_bm25"] for r in rows], type=pa.int16()),
        "score_hybrid_classified": pa.array(
            [r["score_hybrid_classified"] for r in rows], type=pa.float64()),
        "score_vector": pa.array(
            [r["score_vector"] for r in rows], type=pa.float64()),
        "score_bm25": pa.array(
            [r["score_bm25"] for r in rows], type=pa.float64()),
        "source_legs": pa.array(
            [sorted(r["source_legs"]) for r in rows],
            type=pa.list_(pa.string())),
        "retrieved_at": pa.array(
            [retrieved_at] * len(rows), type=pa.timestamp("us", tz="UTC")),
    })
    pq.write_table(table, OUT_PATH, compression="zstd", compression_level=9)
    print(f"[write] {len(rows):,} rows -> {OUT_PATH} "
          f"({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    if failed:
        # Persist failure log so we can inspect/retry
        with open(OUT_PATH.parent / "retrieval_failures.json", "w") as f:
            json.dump([{"query_id": q, "query": qt, "error": e}
                       for q, qt, e in failed], f, indent=2)
        print(f"[warn] {len(failed)} failures logged")


if __name__ == "__main__":
    main()
