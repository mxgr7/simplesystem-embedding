"""Build a full-corpus brute-force oracle for kNN recall measurements.

One-time expensive operation (~3-5 min wall-clock with parallelism). Pulls
every (articleId, inputHash) from `local-article-index-v2`, resolves vectors
via the Redis TEI cache, then computes article-level exact top-K nearest
articles per query using FAISS `IndexFlatIP` on L2-normalized vectors
(cosine ≡ inner product when normalized).

Reuses the 1k canonical queries from `reports/hnsw_eval/` so recall numbers
are directly comparable across all benches in this project.

Article-level GT (not vector-level): ES kNN over nested `embeddings` returns
top-k articles after collapsing multiple matching embeddings per article to
their best-scoring one. Our oracle mirrors that: per query, take the top-N
vectors, group by articleId, keep the best score per article, return the
top-K unique articles. K=100 is plenty of headroom for typical recall@10
measurements and any reasonable pagination depth.

Speedups (vs the per-sample build_hnsw_eval_dataset.py):
  - PIT + slice across 8 parallel iterators (ES fetch ~38s/slice in parallel
    vs 215s sequential for 1M articles).
  - Parallel Redis MGET across 8 worker threads.
  - Single FAISS IndexFlatIP search for all queries (auto-chunks internally).

Output artifacts to `--out-dir` (default `/data/datasets/hnsw_eval_full/`):
  manifest.json                  — config + corpus stats
  corpus_attrs.parquet           — (article_id, input_hash) per corpus row, 12M rows
  corpus_vectors.npy             — float32[N_corpus, 128], ~6 GB (kept for re-runs)
  ground_truth_vectors.npy       — int32[N_queries, K], vector indices (debugging)
  ground_truth_articleids.parquet — top-K article_ids per query (the primary GT)

Usage:
  uv run python scripts/build_full_corpus_oracle.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import redis

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.projection import HASH_VERSION  # noqa: E402

DEFAULT_QUERIES_DIR = REPO_ROOT / "reports" / "hnsw_eval"
# Big regeneratable caches go to /data so / doesn't fill up.
DEFAULT_OUT_DIR = Path("/data/datasets/hnsw_eval_full")

ES_URL = "http://localhost:9200"
ES_INDEX = "local-article-index-v2"
REDIS_URL = "redis://localhost:6379/0"
DIM = 128
VECTOR_BYTES = DIM * 2  # fp16
GT_TOPK = 100
VECTOR_OVERFETCH = 1000  # fetch top-N vectors per query, then collapse to K unique articles

PIT_KEEP_ALIVE = "30m"
PAGE_SIZE = 2000
ES_SLICES = 8
REDIS_WORKERS = 8
REDIS_MGET_CHUNK = 10000


# -------------------- ES: sliced PIT fetch --------------------


def open_pit(es_url: str) -> str:
    with httpx.Client(base_url=es_url, timeout=60.0) as c:
        r = c.post(f"/{ES_INDEX}/_pit", params={"keep_alive": PIT_KEEP_ALIVE})
        r.raise_for_status()
        return r.json()["id"]


def close_pit(es_url: str, pit_id: str) -> None:
    try:
        with httpx.Client(base_url=es_url, timeout=30.0) as c:
            c.request("DELETE", "/_pit", json={"id": pit_id})
    except Exception:
        pass


def pull_slice(
    es_url: str, pit_id: str, slice_id: int, max_slices: int
) -> tuple[list[str], list[str]]:
    """Iterate one PIT slice; yield (article_ids, input_hashes) for this slice's docs."""
    ids: list[str] = []
    hashes: list[str] = []
    with httpx.Client(base_url=es_url, timeout=300.0) as client:
        search_after: list | None = None
        while True:
            body: dict = {
                "size": PAGE_SIZE,
                "track_total_hits": False,
                "pit": {"id": pit_id, "keep_alive": PIT_KEEP_ALIVE},
                "slice": {"id": slice_id, "max": max_slices},
                "_source": ["articleId", "embeddings.inputHash"],
                "query": {"match_all": {}},
                "sort": [{"_shard_doc": "asc"}],
            }
            if search_after is not None:
                body["search_after"] = search_after
            r = client.post("/_search", json=body)
            r.raise_for_status()
            hits = r.json()["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                src = h["_source"]
                aid = src.get("articleId")
                if not aid:
                    continue
                for e in src.get("embeddings") or []:
                    ih = e.get("inputHash")
                    if not ih:
                        continue
                    ids.append(aid)
                    hashes.append(ih)
            search_after = hits[-1]["sort"]
    return ids, hashes


def pull_all_corpus_hashes(es_url: str) -> tuple[list[str], list[str]]:
    print(f"  opening PIT on {ES_INDEX} ...")
    pit_id = open_pit(es_url)
    try:
        print(f"  pulling across {ES_SLICES} parallel slices ...")
        with ThreadPoolExecutor(max_workers=ES_SLICES) as ex:
            futures = [
                ex.submit(pull_slice, es_url, pit_id, i, ES_SLICES)
                for i in range(ES_SLICES)
            ]
            results = []
            for i, fut in enumerate(as_completed(futures)):
                ids, hashes = fut.result()
                results.append((ids, hashes))
                print(f"    slice {i + 1}/{ES_SLICES} done, {len(hashes):,} pairs")
    finally:
        close_pit(es_url, pit_id)

    all_ids: list[str] = []
    all_hashes: list[str] = []
    for ids, hashes in results:
        all_ids.extend(ids)
        all_hashes.extend(hashes)
    return all_ids, all_hashes


# -------------------- Redis: parallel MGET --------------------


def fetch_vectors_parallel(
    hashes: list[str], redis_url: str
) -> tuple[list[int], np.ndarray]:
    """Bulk MGET fp16 vectors across REDIS_WORKERS threads. Returns
    (kept_indices, fp32 vectors)."""
    keys = [f"tei:{HASH_VERSION}:{h}" for h in hashes]
    chunk_starts = list(range(0, len(keys), REDIS_MGET_CHUNK))

    def fetch_chunk(start: int) -> tuple[int, list[bytes | None]]:
        r = redis.Redis.from_url(redis_url, decode_responses=False)
        chunk = keys[start : start + REDIS_MGET_CHUNK]
        return start, r.mget(chunk)

    raw: list[bytes | None] = [None] * len(keys)
    with ThreadPoolExecutor(max_workers=REDIS_WORKERS) as ex:
        for fut in as_completed(ex.submit(fetch_chunk, s) for s in chunk_starts):
            start, values = fut.result()
            for j, v in enumerate(values):
                raw[start + j] = v

    kept_indices: list[int] = []
    arr = np.empty((len(raw), DIM), dtype=np.float32)
    bad = missing = 0
    write_idx = 0
    for i, v in enumerate(raw):
        if v is None:
            missing += 1
            continue
        if len(v) != VECTOR_BYTES:
            bad += 1
            continue
        kept_indices.append(i)
        arr[write_idx] = np.frombuffer(v, dtype=np.float16).astype(np.float32)
        write_idx += 1
    if missing or bad:
        print(
            f"  warning: {missing:,} missing, {bad:,} malformed (skipped); "
            f"{write_idx:,} of {len(hashes):,} resolved"
        )
    return kept_indices, arr[:write_idx]


# -------------------- oracle: brute-force + article collapse --------------------


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def compute_oracle(
    qvecs: np.ndarray,
    cvecs: np.ndarray,
    article_ids: np.ndarray,
    k: int = GT_TOPK,
    overfetch: int = VECTOR_OVERFETCH,
) -> tuple[np.ndarray, list[list[str]]]:
    """Brute-force vector top-K and article-collapsed top-K per query.

    Returns:
      gt_vectors: int32[N_q, k] vector indices (debugging)
      gt_articles: list of k article_ids per query (primary GT)
    """
    qv = normalize_rows(qvecs)
    cv = normalize_rows(cvecs)
    print(
        f"  building FAISS IndexFlatIP over {cv.shape[0]:,} vectors "
        f"(memory ≈ {cv.nbytes / 1e9:.1f} GB) ..."
    )
    index = faiss.IndexFlatIP(cv.shape[1])
    index.add(cv)

    # Overfetch enough vectors per query that we can collapse to k unique articles.
    fetch_n = min(overfetch, cv.shape[0])
    print(f"  searching top-{fetch_n} for {qv.shape[0]} queries ...")
    _, top_vec = index.search(qv, fetch_n)  # int64

    gt_vectors = np.empty((qv.shape[0], k), dtype=np.int32)
    gt_articles: list[list[str]] = []

    for q_idx in range(qv.shape[0]):
        gt_vectors[q_idx] = top_vec[q_idx, :k].astype(np.int32)
        seen: list[str] = []
        seen_set: set[str] = set()
        for vi in top_vec[q_idx]:
            aid = article_ids[vi]
            if aid in seen_set:
                continue
            seen_set.add(aid)
            seen.append(aid)
            if len(seen) >= k:
                break
        if len(seen) < k:
            # Pad if fewer than k unique articles found in the overfetch window.
            seen.extend([""] * (k - len(seen)))
        gt_articles.append(seen)
    return gt_vectors, gt_articles


# -------------------- main --------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries-dir", type=Path, default=DEFAULT_QUERIES_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--es-url", default=ES_URL)
    p.add_argument("--redis-url", default=REDIS_URL)
    p.add_argument("--gt-topk", type=int, default=GT_TOPK)
    p.add_argument(
        "--overfetch",
        type=int,
        default=VECTOR_OVERFETCH,
        help="Top-N vectors per query before article-collapse (default 1000)",
    )
    p.add_argument(
        "--skip-corpus-save",
        action="store_true",
        help="Don't write corpus_vectors.npy (saves 6 GB of disk).",
    )
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # Load queries from existing eval dir.
    print(f"[1/5] Loading 1k canonical queries from {args.queries_dir} ...")
    queries_tbl = pq.read_table(args.queries_dir / "queries.parquet")
    qvecs = np.load(args.queries_dir / "query_vectors.npy")
    queries = queries_tbl["query"].to_pylist()
    print(f"      {qvecs.shape[0]} queries loaded")

    # Pull every (articleId, inputHash) from local-v2 via sliced PIT.
    t0 = time.time()
    print(f"\n[2/5] Pulling all (articleId, inputHash) from {args.es_url}/{ES_INDEX} ...")
    all_aids, all_hashes = pull_all_corpus_hashes(args.es_url)
    print(
        f"      {len(all_hashes):,} (articleId, inputHash) pairs across "
        f"{len(set(all_aids)):,} unique articles in {time.time() - t0:.1f}s"
    )

    # Resolve vectors from Redis (parallel MGET).
    t0 = time.time()
    print(
        f"\n[3/5] Fetching vectors from Redis at {args.redis_url} "
        f"(key prefix tei:{HASH_VERSION}:) ..."
    )
    kept_indices, cvecs = fetch_vectors_parallel(all_hashes, args.redis_url)
    aids = np.array([all_aids[i] for i in kept_indices])
    hashes = [all_hashes[i] for i in kept_indices]
    print(
        f"      resolved {cvecs.shape[0]:,} vectors "
        f"({cvecs.nbytes / 1e9:.2f} GB fp32) in {time.time() - t0:.1f}s"
    )

    # Brute-force oracle.
    t0 = time.time()
    print(f"\n[4/5] Computing brute-force oracle (top-{args.gt_topk} articles per query) ...")
    gt_vectors, gt_articles = compute_oracle(
        qvecs, cvecs, aids, k=args.gt_topk, overfetch=args.overfetch
    )
    print(f"      done in {time.time() - t0:.1f}s")

    # Write artifacts.
    print(f"\n[5/5] Writing artifacts to {out}/ ...")
    pq.write_table(
        pa.table({"article_id": list(aids), "input_hash": hashes}),
        out / "corpus_attrs.parquet",
    )
    np.save(out / "ground_truth_vectors.npy", gt_vectors)
    pq.write_table(
        pa.table({"top_article_ids": gt_articles}),
        out / "ground_truth_articleids.parquet",
    )
    if not args.skip_corpus_save:
        np.save(out / "corpus_vectors.npy", cvecs)
        print(f"      corpus_vectors.npy = {cvecs.nbytes / 1e9:.2f} GB")

    manifest = {
        "queries": int(qvecs.shape[0]),
        "n_corpus_vectors": int(cvecs.shape[0]),
        "n_unique_articles": int(len(set(aids.tolist()))),
        "dim": DIM,
        "gt_topk": args.gt_topk,
        "vector_overfetch": args.overfetch,
        "es_url": args.es_url,
        "es_index": ES_INDEX,
        "redis_url": args.redis_url,
        "hash_version": HASH_VERSION,
        "queries_source": str(args.queries_dir),
        "skip_corpus_save": args.skip_corpus_save,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"      manifest.json + corpus_attrs.parquet + ground_truth_*.* written")
    print(f"\nDone. Oracle ready at {out}/")


if __name__ == "__main__":
    main()
