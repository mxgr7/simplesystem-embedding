#!/usr/bin/env python3
"""Mine deeper-retrieval candidates for mono-label queries.

A mono-label query (every labeled candidate carries the same label) gives the
cross-encoder no within-query contrast. We re-hit the search-api at deep
dense depth (mode=vector, k=10000 with dense_limit/num_candidates=10000) and
keep only candidates with cosine < SCORE_MAX (0.70) — empirically the
threshold above which the LLM judge essentially never flips off the parent
mono-label. After filtering out candidates the labeled dataset already has
for that query, we randomly sample up to 30 of the remaining candidates per
query (seeded per query_id for reproducibility). Queries with zero
sub-threshold candidates raise NoBelowThreshold and are persisted to a
skip-log instead of producing empty output.

Output: /data/datasets/queries_offers_esci/candidates_mono_label_negatives.parquet
Skip-log: candidates_mono_label_negatives_no_below.json

Schema matches candidates.parquet so the materialize step can mirror the
existing pipeline. rank_vector / rank_bm25 / score_vector / score_bm25 are
NULL (we only re-retrieved dense). source_legs is always ['dense'].
score_hybrid_classified holds the raw cosine and is always < SCORE_MAX.
"""
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import duckdb
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

ROOT = Path("/data/datasets/queries_offers_esci")
LABELED = ROOT / "queries_offers_merged_labeled.parquet" / "part-0.parquet"
QUERIES = ROOT / "queries.parquet"
OUT_PATH = ROOT / "candidates_mono_label_negatives.parquet"

K_DEEP = 10000
MODE = "vector"  # dense-only
SCORE_MAX = 0.70  # only candidates with cosine < SCORE_MAX qualify as hard negatives
SAMPLE_PER_QUERY = 30
SEED = 42
WORKERS = 8
MAX_RETRIES = 4
TIMEOUT_S = 180


class NoBelowThreshold(Exception):
    """Raised when a query has zero candidates below SCORE_MAX after dedup."""


def load_targets() -> tuple[list[tuple[int, str]], dict[int, set[str]]]:
    """Return (queries_to_retrieve, existing_candidates_per_query).

    queries_to_retrieve: list of (query_id, qt_raw) for mono-{E,S,C} queries.
    existing_candidates_per_query: query_id -> set of candidate_ids already
        in the labeled dataset (so we can dedupe deeper retrieval against
        already-labeled rows).
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    rows = con.execute(f"""
        WITH per_q AS (
            SELECT query_id,
                   COUNT(*) FILTER (WHERE label IS NOT NULL) AS n_labeled,
                   COUNT(DISTINCT label) FILTER (WHERE label IS NOT NULL) AS n_distinct,
                   ANY_VALUE(label) FILTER (WHERE label IS NOT NULL) AS only_label
            FROM read_parquet('{LABELED}')
            GROUP BY 1
        )
        SELECT query_id, only_label
        FROM per_q
        WHERE n_labeled > 0 AND n_distinct = 1
          AND only_label IN ('Exact', 'Substitute', 'Complement')
    """).fetchall()
    target_qids = {qid for qid, _ in rows}
    print(f"[targets] mono-label queries: {len(target_qids):,}")
    by_label = {}
    for _, lab in rows:
        by_label[lab] = by_label.get(lab, 0) + 1
    for lab in ("Exact", "Substitute", "Complement"):
        print(f"  {lab:11} {by_label.get(lab, 0):>5,}")

    # Pull qt_raw for those queries
    qt_rows = con.execute(f"""
        SELECT query_id, qt_raw
        FROM read_parquet('{QUERIES}')
        WHERE query_id IN ({','.join(str(q) for q in target_qids)})
        ORDER BY query_id
    """).fetchall()

    # Existing candidate_ids per query (so we can dedupe)
    cand_rows = con.execute(f"""
        SELECT query_id, offer_id
        FROM read_parquet('{LABELED}')
        WHERE query_id IN ({','.join(str(q) for q in target_qids)})
    """).fetchall()
    existing: dict[int, set[str]] = {}
    for qid, oid in cand_rows:
        existing.setdefault(qid, set()).add(oid)

    return qt_rows, existing


def call_deep(session: requests.Session, query: str, k: int) -> list[dict]:
    """One deep retrieval call.

    For mode=vector, we must also widen the dense candidate pool
    (`dense_limit`) and HNSW search depth (`num_candidates`) to k —
    otherwise the API caps the underlying retrieval at the default 200.
    """
    body = {"query": query, "index": "offers-prod"}
    params = {"mode": MODE, "k": k}
    if MODE == "vector":
        params["dense_limit"] = k
        params["num_candidates"] = k
    for attempt in range(MAX_RETRIES):
        try:
            r = session.post(API_URL, params=params, headers=HEADERS,
                             json=body, timeout=TIMEOUT_S)
            if r.status_code == 503:
                time.sleep(0.5 * (2 ** attempt))
                continue
            r.raise_for_status()
            return r.json().get("hits", []) or []
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(0.3 * (2 ** attempt))
    return []


def fetch_query(query_id: int, query_text: str,
                existing: set[str]) -> list[dict]:
    """Run deep retrieval for one query; dedupe + sample."""
    sess = requests.Session()
    hits = call_deep(sess, query_text, K_DEEP)

    # Keep only candidates with cosine < SCORE_MAX (true outside-cluster
    # candidates) and not already in the labeled dataset. Empirically (see
    # k=2000 sample), candidates with score >= 0.70 are essentially never
    # labeled as anything other than the parent's mono label.
    new_hits = []
    for rank, h in enumerate(hits, start=1):
        if h["_score"] >= SCORE_MAX:
            continue
        cid = h["_id"]
        if cid in existing:
            continue
        new_hits.append({
            "query_id": query_id,
            "candidate_id": cid,
            "rank_hybrid_classified": rank,
            "rank_vector": None,
            "rank_bm25": None,
            "score_hybrid_classified": h["_score"],
            "score_vector": None,
            "score_bm25": None,
            "source_legs": {h["_source_leg"]},
        })

    if not new_hits:
        raise NoBelowThreshold(
            f"query {query_id}: 0 candidates below score {SCORE_MAX} "
            f"in top {K_DEEP} dense hits"
        )

    # Random sample (deterministic per-query seed)
    if len(new_hits) > SAMPLE_PER_QUERY:
        rng = random.Random(f"{SEED}-{query_id}")
        new_hits = rng.sample(new_hits, SAMPLE_PER_QUERY)

    return new_hits


def main():
    print(f"[load] reading targets from {LABELED}")
    qt_rows, existing = load_targets()
    print(f"[load] {len(qt_rows)} queries to retrieve "
          f"(existing candidates avg: "
          f"{sum(len(v) for v in existing.values()) / max(len(existing), 1):.1f})")

    rows: list[dict] = []
    failed: list[tuple[int, str, str]] = []
    no_below: list[int] = []  # queries with 0 candidates below SCORE_MAX
    skipped_few: list[tuple[int, int]] = []  # < SAMPLE_PER_QUERY available
    t0 = time.time()
    next_log = 200
    n = len(qt_rows)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch_query, qid, qt, existing.get(qid, set())): qid
                   for qid, qt in qt_rows}
        done = 0
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                rs = fut.result()
                if len(rs) < SAMPLE_PER_QUERY:
                    skipped_few.append((qid, len(rs)))
                rows.extend(rs)
            except NoBelowThreshold:
                no_below.append(qid)
            except Exception as e:
                failed.append((qid, "", str(e)))
            done += 1
            if done >= next_log:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (n - done) / rate if rate else 0
                print(f"  {done:>5}/{n}  rate={rate:5.1f}/s  "
                      f"rows={len(rows):>6,}  elapsed={elapsed:.0f}s  "
                      f"eta={eta:.0f}s  failed={len(failed)}  "
                      f"no_below={len(no_below)}  "
                      f"under_floor={len(skipped_few)}")
                next_log += 200

    elapsed = time.time() - t0
    print(f"[retrieve] done in {elapsed:.0f}s. "
          f"{len(rows):,} new (query, candidate) rows over "
          f"{len(qt_rows) - len(failed) - len(no_below)} queries; "
          f"{len(failed)} hard failures; "
          f"{len(no_below)} queries with 0 hits below score {SCORE_MAX} (skipped); "
          f"{len(skipped_few)} queries returned < {SAMPLE_PER_QUERY}")
    if no_below:
        # Persist the skipped query ids so we can identify which queries
        # cannot be mined this way (likely 1000+ hit_band head queries
        # with deep semantic clusters).
        skip_path = OUT_PATH.parent / "candidates_mono_label_negatives_no_below.json"
        import json as _json
        skip_path.write_text(_json.dumps(no_below))
        print(f"[skip-log] {len(no_below)} skipped query_ids -> {skip_path}")

    # Stable order: by query_id, then by deep rank
    rows.sort(key=lambda r: (r["query_id"], r["rank_hybrid_classified"]))

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


if __name__ == "__main__":
    main()
