"""Sweep HNSW configurations against actual Elasticsearch and report recall@k
vs. brute-force ground truth.

For each (m, ef_construction):
  1. Create a fresh ES test index with int8_hnsw + the given params
  2. Bulk-load the corpus vectors as flat top-level docs (one doc per vector)
  3. Refresh + force-merge to a single segment (consistent across configs)
  4. Sweep num_candidates (= ef_search), measuring recall@k + p50/p95 latency
  5. Delete the index (unless --keep-indices)

Each vector becomes one top-level doc { idx: int, vector: dense_vector[128] }.
Flat top-level vectors are the right unit for HNSW *graph* tuning — nested-kNN
article-collapse happens at query time and is independent of HNSW params, so
adding nesting would only slow this down without changing the recall numbers.

Runtime note: building an ES HNSW index per config is much slower than a
FAISS sweep (minutes per config, not seconds). The cost buys recall numbers
against the exact implementation we ship — including int8 quantization, the
Lucene HNSW heuristics, and post-merge graph state.

Usage:
  uv run python scripts/bench_hnsw.py --in-dir reports/hnsw_eval --k 10
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "reports" / "hnsw_eval"

ES_URL = "http://localhost:9200"
INDEX_PREFIX = "hnsw-bench"
BULK_BATCH = 5000
BULK_CONCURRENCY = 4

# (m, ef_construction) — index-time params
INDEX_CONFIGS: list[tuple[int, int]] = [
    (16, 100),
    (16, 200),
    (32, 100),
    (32, 200),
    (32, 400),
    (64, 200),
    (64, 400),
]

# num_candidates (= ef_search) — query-time params
NUM_CANDIDATES = [50, 100, 200, 500, 1000, 2000]


def create_index(client: httpx.Client, name: str, m: int, ef_construction: int, dim: int) -> None:
    client.delete(f"/{name}", params={"ignore_unavailable": "true"})
    body = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1",
                "translog": {"durability": "async"},
            }
        },
        "mappings": {
            "properties": {
                "idx": {"type": "integer"},
                "vector": {
                    "type": "dense_vector",
                    "dims": dim,
                    "similarity": "cosine",
                    "index": True,
                    "index_options": {
                        "type": "int8_hnsw",
                        "m": m,
                        "ef_construction": ef_construction,
                    },
                },
            }
        },
    }
    r = client.put(f"/{name}", json=body)
    r.raise_for_status()


def _bulk_batch(
    client: httpx.Client, name: str, vectors: np.ndarray, start: int, end: int
) -> int:
    lines: list[str] = []
    for i in range(start, end):
        lines.append('{"index":{"_id":"' + str(i) + '"}}')
        lines.append(json.dumps({"idx": i, "vector": vectors[i].tolist()}))
    body = "\n".join(lines) + "\n"
    r = client.post(
        f"/{name}/_bulk",
        content=body,
        headers={"Content-Type": "application/x-ndjson"},
    )
    r.raise_for_status()
    j = r.json()
    if j.get("errors"):
        for item in j["items"]:
            op = next(iter(item.values()))
            if "error" in op:
                raise RuntimeError(f"bulk error at start={start}: {op['error']}")
    return end - start


def bulk_load(client: httpx.Client, name: str, vectors: np.ndarray) -> int:
    n = vectors.shape[0]
    starts = list(range(0, n, BULK_BATCH))
    loaded = 0
    with ThreadPoolExecutor(max_workers=BULK_CONCURRENCY) as ex:
        futures = [
            ex.submit(_bulk_batch, client, name, vectors, s, min(s + BULK_BATCH, n))
            for s in starts
        ]
        for f in as_completed(futures):
            loaded += f.result()
    return loaded


def finalize(client: httpx.Client, name: str) -> None:
    client.post(f"/{name}/_refresh").raise_for_status()
    client.post(
        f"/{name}/_forcemerge",
        params={"max_num_segments": "1"},
        timeout=900.0,
    ).raise_for_status()


def index_store_bytes(client: httpx.Client, name: str) -> int:
    r = client.get(f"/{name}/_stats/store")
    r.raise_for_status()
    return int(r.json()["indices"][name]["primaries"]["store"]["size_in_bytes"])


def search_one(
    client: httpx.Client,
    name: str,
    query_vec: np.ndarray,
    k: int,
    num_candidates: int,
) -> tuple[list[int], float]:
    body = {
        "knn": {
            "field": "vector",
            "query_vector": query_vec.tolist(),
            "k": k,
            "num_candidates": num_candidates,
        },
        "_source": ["idx"],
        "size": k,
    }
    t0 = time.perf_counter()
    r = client.post(f"/{name}/_search", json=body)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    hits = r.json()["hits"]["hits"]
    return [h["_source"]["idx"] for h in hits], elapsed_ms


def recall_at_k(hits_per_query: list[list[int]], gt: np.ndarray, k: int) -> float:
    matches = 0
    for h, gr in zip(hits_per_query, gt):
        matches += len(set(h[:k]) & set(gr[:k].tolist()))
    return matches / (len(hits_per_query) * k)


def delete_index(client: httpx.Client, name: str) -> None:
    try:
        client.delete(f"/{name}", params={"ignore_unavailable": "true"})
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--es-url", default=ES_URL)
    p.add_argument("--index-prefix", default=INDEX_PREFIX)
    p.add_argument("--k", type=int, default=10, help="recall@k target (default 10)")
    p.add_argument(
        "--keep-indices",
        action="store_true",
        help="Skip cleanup so indices can be inspected after the run.",
    )
    args = p.parse_args()

    qvecs = np.load(args.in_dir / "query_vectors.npy")
    cvecs = np.load(args.in_dir / "corpus_vectors.npy")
    gt = np.load(args.in_dir / "ground_truth.npy")
    manifest = json.loads((args.in_dir / "manifest.json").read_text())

    if gt.shape[1] < args.k:
        raise SystemExit(
            f"ground truth has top-{gt.shape[1]} but --k={args.k}; "
            "rebuild dataset with a larger --gt-topk."
        )

    print(
        f"loaded: {qvecs.shape[0]} queries, {cvecs.shape[0]:,} corpus vectors, "
        f"dim={qvecs.shape[1]}"
    )
    print(f"manifest: {manifest}")
    print(f"sweeping recall@{args.k} against ES at {args.es_url}\n")

    hdr = (
        f"{'M':>3} {'efC':>4} {'numC':>5}  "
        f"{'load_s':>7}  {'merge_s':>8}  {'store_MB':>9}  "
        f"{'p50_ms':>7}  {'p95_ms':>7}  {'rec@' + str(args.k):>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    results: list[dict] = []
    limits = httpx.Limits(
        max_connections=BULK_CONCURRENCY * 2,
        max_keepalive_connections=BULK_CONCURRENCY * 2,
    )
    with httpx.Client(base_url=args.es_url, timeout=300.0, limits=limits) as client:
        for (m, efc) in INDEX_CONFIGS:
            name = f"{args.index_prefix}-m{m}-ef{efc}"
            try:
                create_index(client, name, m, efc, cvecs.shape[1])

                t0 = time.time()
                bulk_load(client, name, cvecs)
                load_s = time.time() - t0

                t0 = time.time()
                finalize(client, name)
                merge_s = time.time() - t0

                store_mb = index_store_bytes(client, name) / 1024 / 1024

                for numc in NUM_CANDIDATES:
                    eff_numc = max(numc, args.k)
                    # warm-up — first kNN call pages graph state from disk
                    search_one(client, name, qvecs[0], args.k, eff_numc)

                    hits_per_query: list[list[int]] = []
                    latencies: list[float] = []
                    for q in qvecs:
                        h, ms = search_one(client, name, q, args.k, eff_numc)
                        hits_per_query.append(h)
                        latencies.append(ms)

                    r = recall_at_k(hits_per_query, gt, args.k)
                    p50 = float(np.percentile(latencies, 50))
                    p95 = float(np.percentile(latencies, 95))

                    print(
                        f"{m:>3} {efc:>4} {eff_numc:>5}  "
                        f"{load_s:>7.1f}  {merge_s:>8.1f}  {store_mb:>9.1f}  "
                        f"{p50:>7.2f}  {p95:>7.2f}  {r:>8.4f}"
                    )
                    results.append(
                        {
                            "m": m,
                            "ef_construction": efc,
                            "num_candidates": eff_numc,
                            "load_s": round(load_s, 2),
                            "merge_s": round(merge_s, 2),
                            "store_mb": round(store_mb, 2),
                            "query_ms_p50": round(p50, 3),
                            "query_ms_p95": round(p95, 3),
                            f"recall_at_{args.k}": round(r, 5),
                        }
                    )
            finally:
                if not args.keep_indices:
                    delete_index(client, name)

    out_path = args.in_dir / f"bench_recall_at_{args.k}_es.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
