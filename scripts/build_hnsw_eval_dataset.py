"""Build the dataset for benchmarking HNSW params under FT_ELASTIC.

Three artifacts (v1 default: 1k queries × 200k article sample, unfiltered):

  queries.parquet         (query, count)                — top-N queries by frequency
  query_vectors.npy       float32[N, 128]               — TEI-embedded query vectors
  vector_article_ids.parquet (article_id)               — article_id per corpus vector row
  corpus_vectors.npy      float32[M, 128]               — all embeddings from the article sample
  ground_truth.npy        int32[N, GT_TOPK]             — exact top-K corpus indices per query
  manifest.json                                         — config snapshot

Sources:
  Queries: /data/datasets/posthog_queries.parquet/*.parquet
           event=search_performed, qt non-null. Identifier-shaped queries (EAN,
           alphanumeric SKUs) are filtered out by default — they short-circuit
           through identifier profiles and don't exercise the vector path.

  Vectors: local-article-index-v1 on localhost:9200. Random-scored sample of
           N articles, all `embeddings.vector` entries extracted (one row per
           vector, tagged with its article_id).

Ground truth is computed at the *vector* level via FAISS IndexFlatIP on
L2-normalized vectors (cosine ≡ inner product when normalized). HNSW returns
top-k vectors before article-collapse, so vector-level recall is the right
target for graph tuning.

Usage:
  uv run python scripts/build_hnsw_eval_dataset.py \
    --queries 1000 --articles 200000 --out-dir reports/hnsw_eval
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import duckdb
import faiss
import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "hnsw_eval"

POSTHOG_DIR = "/data/datasets/posthog_queries.parquet"
ES_URL = "http://localhost:9200"
ES_INDEX = "local-article-index-v1"
TEI_URL = "http://localhost:8080"
DIM = 128
GT_TOPK = 100

TEI_BATCH = 32          # matches TEI --max-client-batch-size
ES_PAGE_SIZE = 1000
PIT_KEEP_ALIVE = "5m"
RANDOM_SEED = 42

# Heuristic: identifier-shaped queries (EAN, alphanumeric SKU, hyphen-digit).
# These short-circuit through lexical profiles in production and don't exercise
# the vector retriever, so they distort HNSW recall measurements if included.
EAN_RE = re.compile(r"^(\d{8}|\d{12,14})$")
ALPHA_DIGIT_RE = re.compile(r"^(?=.{7,}$)[a-z]+\d{4,}[a-z0-9]*$", re.IGNORECASE)
HYPHEN_DIGIT_RE = re.compile(
    r"^(?=.{7,}$)(?=(?:[^\d]*\d){3,})[a-z0-9]+(?:-[a-z0-9]+)+$", re.IGNORECASE
)


def is_identifier(q: str) -> bool:
    q = q.strip().lower()
    return bool(
        EAN_RE.fullmatch(q)
        or ALPHA_DIGIT_RE.fullmatch(q)
        or HYPHEN_DIGIT_RE.fullmatch(q)
    )


def mine_queries(n: int, drop_identifiers: bool) -> list[tuple[str, int]]:
    con = duckdb.connect()
    con.execute(f"SET threads={os.cpu_count() or 8};")
    con.execute("PRAGMA disable_progress_bar;")
    # Over-fetch so we still have N rows after the identifier filter.
    over = n * 3 if drop_identifiers else n
    rows = con.execute(
        f"""
        SELECT trim(qt) AS q, count(*) AS n
        FROM read_parquet('{POSTHOG_DIR}/*.parquet')
        WHERE event = 'search_performed'
          AND qt IS NOT NULL
          AND length(trim(qt)) > 0
        GROUP BY q
        ORDER BY n DESC
        LIMIT {over}
        """
    ).fetchall()
    if drop_identifiers:
        rows = [(q, c) for (q, c) in rows if not is_identifier(q)]
    return rows[:n]


def embed_queries(queries: list[str]) -> np.ndarray:
    out = np.empty((len(queries), DIM), dtype=np.float32)
    with httpx.Client(base_url=TEI_URL, timeout=120.0) as client:
        for i in range(0, len(queries), TEI_BATCH):
            batch = queries[i : i + TEI_BATCH]
            r = client.post("/embed", json={"inputs": batch, "truncate": True})
            r.raise_for_status()
            arr = np.asarray(r.json(), dtype=np.float32)
            if arr.shape != (len(batch), DIM):
                raise RuntimeError(
                    f"TEI returned unexpected shape {arr.shape} for batch of {len(batch)}"
                )
            out[i : i + len(batch)] = arr
    return out


def pull_article_vectors(n_articles: int) -> tuple[list[str], np.ndarray]:
    """Random-score-paginate `n_articles` from ES, expanding each into its
    embeddings entries. Returns (article_ids per vector, vectors)."""
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    with httpx.Client(base_url=ES_URL, timeout=180.0) as client:
        # Open a point-in-time view for stable pagination.
        r = client.post(f"/{ES_INDEX}/_pit", params={"keep_alive": PIT_KEEP_ALIVE})
        r.raise_for_status()
        pit_id = r.json()["id"]

        try:
            search_after: list | None = None
            articles_seen = 0
            while articles_seen < n_articles:
                page = min(ES_PAGE_SIZE, n_articles - articles_seen)
                body: dict = {
                    "size": page,
                    "track_total_hits": False,
                    "pit": {"id": pit_id, "keep_alive": PIT_KEEP_ALIVE},
                    "_source": ["articleId", "embeddings.vector"],
                    "query": {
                        "function_score": {
                            "query": {"match_all": {}},
                            "random_score": {"seed": RANDOM_SEED, "field": "_seq_no"},
                        }
                    },
                    "sort": [{"_score": "desc"}, {"_shard_doc": "asc"}],
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
                    if aid is None:
                        continue
                    for e in src.get("embeddings") or []:
                        v = e.get("vector")
                        if v is None:
                            continue
                        ids.append(aid)
                        vecs.append(np.asarray(v, dtype=np.float32))

                articles_seen += len(hits)
                search_after = hits[-1]["sort"]
        finally:
            try:
                client.request("DELETE", "/_pit", json={"id": pit_id})
            except Exception:
                pass

    if not vecs:
        raise SystemExit("No embeddings found in the sampled articles.")
    return ids, np.vstack(vecs)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def compute_ground_truth(qvecs: np.ndarray, cvecs: np.ndarray, topk: int) -> np.ndarray:
    """Exact top-k vector indices per query under cosine (IP on normalized vectors)."""
    q = normalize_rows(qvecs)
    c = normalize_rows(cvecs)
    index = faiss.IndexFlatIP(c.shape[1])
    index.add(c)
    _, I = index.search(q, topk)
    return I.astype(np.int32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--articles", type=int, default=200_000)
    p.add_argument("--gt-topk", type=int, default=GT_TOPK)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--keep-identifiers",
        action="store_true",
        help="Don't filter out EAN/SKU-shaped queries (default: drop them).",
    )
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[1/4] Mining top-{args.queries} queries from {POSTHOG_DIR} ...")
    rows = mine_queries(args.queries, drop_identifiers=not args.keep_identifiers)
    queries = [q for (q, _) in rows]
    counts = [c for (_, c) in rows]
    print(
        f"      got {len(queries):,} queries "
        f"(top: {queries[0]!r} ×{counts[0]:,}, tail: {queries[-1]!r} ×{counts[-1]:,})"
        f" in {time.time() - t0:.1f}s"
    )

    t0 = time.time()
    print(f"[2/4] Embedding queries via TEI at {TEI_URL} ...")
    qvecs = embed_queries(queries)
    print(f"      shape {qvecs.shape} in {time.time() - t0:.1f}s")

    t0 = time.time()
    print(
        f"[3/4] Pulling {args.articles:,} random articles' embeddings "
        f"from {ES_URL}/{ES_INDEX} ..."
    )
    aids, cvecs = pull_article_vectors(args.articles)
    print(
        f"      {cvecs.shape[0]:,} vectors across {len(set(aids)):,} articles "
        f"(avg {cvecs.shape[0] / max(len(set(aids)), 1):.2f}/article) "
        f"in {time.time() - t0:.1f}s"
    )

    t0 = time.time()
    print(
        f"[4/4] Computing exact top-{args.gt_topk} ground truth "
        f"({qvecs.shape[0]} × {cvecs.shape[0]:,} brute force) ..."
    )
    gt = compute_ground_truth(qvecs, cvecs, args.gt_topk)
    print(f"      shape {gt.shape} in {time.time() - t0:.1f}s")

    # Write artifacts.
    pq.write_table(pa.table({"query": queries, "count": counts}), out / "queries.parquet")
    np.save(out / "query_vectors.npy", qvecs)
    pq.write_table(pa.table({"article_id": aids}), out / "vector_article_ids.parquet")
    np.save(out / "corpus_vectors.npy", cvecs)
    np.save(out / "ground_truth.npy", gt)

    manifest = {
        "queries": len(queries),
        "articles_requested": args.articles,
        "vectors_total": int(cvecs.shape[0]),
        "unique_articles": len(set(aids)),
        "dim": DIM,
        "gt_topk": args.gt_topk,
        "tei_url": TEI_URL,
        "es_url": ES_URL,
        "es_index": ES_INDEX,
        "posthog_source": POSTHOG_DIR,
        "random_seed": RANDOM_SEED,
        "drop_identifiers": not args.keep_identifiers,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote artifacts to {out}/")


if __name__ == "__main__":
    main()
