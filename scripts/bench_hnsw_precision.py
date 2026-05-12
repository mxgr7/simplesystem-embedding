"""Sweep dense_vector precision/quantization at fixed (m, ef_construction).

Counterpart to bench_hnsw.py. Where bench_hnsw.py sweeps the (m, ef_construction)
× num_candidates grid at a fixed precision (int8_hnsw), this script fixes
(m, ef_construction) and sweeps the precision axis instead:

  - hnsw         — full fp32, no quantization (reference recall ceiling)
  - int8_hnsw    — int8 scalar quantization (4× smaller than fp32)
  - int4_hnsw    — int4 scalar quantization (8× smaller)
  - bbq_hnsw     — 1-bit binary quantization + rotation + on-disk rerank (32× smaller)

Same harness as bench_hnsw.py — bulk-loads the corpus, force-merges to one
segment, runs the same 1k queries from build_hnsw_eval_dataset.py against
the same brute-force ground truth, and reports recall@k, latency, store size.

Usage:
  uv run python scripts/bench_hnsw_precision.py --k 10
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
INDEX_PREFIX = "hnsw-prec-bench"
BULK_BATCH = 5000
BULK_CONCURRENCY = 4

DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 100

PRECISIONS: list[str] = ["hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw"]
NUM_CANDIDATES = [50, 100, 200, 500, 1000, 2000]


def create_index(
    client: httpx.Client,
    name: str,
    precision: str,
    m: int,
    ef_construction: int,
    dim: int,
) -> None:
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
                        "type": precision,
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
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--m", type=int, default=DEFAULT_M)
    p.add_argument(
        "--ef-construction", type=int, default=DEFAULT_EF_CONSTRUCTION
    )
    p.add_argument("--keep-indices", action="store_true")
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
    print(
        f"fixing m={args.m}, ef_construction={args.ef_construction}; "
        f"sweeping precision × num_candidates against ES at {args.es_url}\n"
    )

    hdr = (
        f"{'precision':>11} {'numC':>5}  "
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
        for precision in PRECISIONS:
            name = f"{args.index_prefix}-{precision.replace('_', '-')}"
            try:
                create_index(
                    client, name, precision, args.m, args.ef_construction, cvecs.shape[1]
                )

                t0 = time.time()
                bulk_load(client, name, cvecs)
                load_s = time.time() - t0

                t0 = time.time()
                finalize(client, name)
                merge_s = time.time() - t0

                store_mb = index_store_bytes(client, name) / 1024 / 1024

                for numc in NUM_CANDIDATES:
                    eff_numc = max(numc, args.k)
                    search_one(client, name, qvecs[0], args.k, eff_numc)  # warm-up

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
                        f"{precision:>11} {eff_numc:>5}  "
                        f"{load_s:>7.1f}  {merge_s:>8.1f}  {store_mb:>9.1f}  "
                        f"{p50:>7.2f}  {p95:>7.2f}  {r:>8.4f}"
                    )
                    results.append(
                        {
                            "precision": precision,
                            "m": args.m,
                            "ef_construction": args.ef_construction,
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

    out_path = (
        args.in_dir
        / f"bench_recall_at_{args.k}_precision_m{args.m}_efc{args.ef_construction}.json"
    )
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
