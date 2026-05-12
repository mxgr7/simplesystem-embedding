"""Bench HNSW under realistic filter regimes.

Counterpart to bench_hnsw.py and bench_hnsw_precision.py. Where those measure
recall against an *unfiltered* kNN, this script attaches a production-shape
`nested(offers)` filter clause to every query and measures how recall@k degrades
as filter selectivity tightens. Answers the operational question: what
`num_candidates` do we actually need in production to hold recall under the
filters TEST_PROFILE_18 will always carry (ACL + facets)?

Filter regimes (selectivities estimated from offers-doc cardinalities on
local-article-index-v2; effective selectivity on our 200k-article sample is
within a factor of 2 of these):

  unfiltered            ~100%    sanity baseline (matches bench_hnsw_precision)
  acl-top                ~20%    single top catalogVersionId
  acl-mid                ~12%    mid-ranked catalogVersionId
  acl-top+mfr-mid       ~0.5%    + Weidmüller manufacturer
  acl-top+mfr-small    ~0.05%    + ATORN manufacturer

All five regimes run against the same ES test index — one shard, fp32 `hnsw`,
m=16, ef_construction=100, force-merged to one segment. Each doc carries its
source article's full offers list under `nested(offers)`, so the filter
expression is byte-equivalent to what production will issue.

Ground truth is recomputed per regime — exact top-k among the *passing*
corpus vectors, computed by brute-force IP on L2-normalized vectors.

Output:
  reports/hnsw_eval/bench_recall_at_<k>_filtered.json
  reports/hnsw_eval/article_filter_attrs.parquet (cached lookup, fetched once)

Usage:
  uv run python scripts/bench_hnsw_filtered.py --k 10
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "reports" / "hnsw_eval"

ES_URL = "http://localhost:9200"
ES_SOURCE_INDEX = "local-article-index-v2"
TEST_INDEX = "hnsw-bench-filtered"

BULK_BATCH = 5000
BULK_CONCURRENCY = 4
TERMS_BATCH = 5000

# Filter values chosen from the global cardinality survey of
# local-article-index-v2 (May 2026). See module docstring.
TOP_CATALOG = "844224b4-e40b-4180-bf04-4f74d702627d"  # global #1
MID_CATALOG = "fcf0f564-e4ea-4f93-b21b-9f690c687732"  # global #20
MFR_MID = "Weidmüller"
MFR_SMALL = "ATORN"

# Index params — fp32 hnsw at production defaults (per FT_ELASTIC_IMPORT §2.1.1)
INDEX_M = 16
INDEX_EF_CONSTRUCTION = 100
PRECISION = "hnsw"

# num_candidates sweep — same grid as bench_hnsw.py for comparability
NUM_CANDIDATES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

# -------------------- regime definitions --------------------


def _matches_acl(offers: list[dict], acl: str) -> bool:
    return any(acl in (o.get("catalogVersionIds") or []) for o in offers)


def _matches_acl_and_mfr(offers: list[dict], acl: str, mfr: str) -> bool:
    """Same-offer semantics: at least one offer must satisfy *both* constraints."""
    return any(
        acl in (o.get("catalogVersionIds") or []) and o.get("manufacturerName") == mfr
        for o in offers
    )


REGIMES: dict[str, dict] = {
    "unfiltered": {
        "mask": lambda offers: True,
        "filter_spec": None,
    },
    "acl-top": {
        "mask": lambda offers: _matches_acl(offers, TOP_CATALOG),
        "filter_spec": {"acl": TOP_CATALOG},
    },
    "acl-mid": {
        "mask": lambda offers: _matches_acl(offers, MID_CATALOG),
        "filter_spec": {"acl": MID_CATALOG},
    },
    "acl-top+mfr-mid": {
        "mask": lambda offers: _matches_acl_and_mfr(offers, TOP_CATALOG, MFR_MID),
        "filter_spec": {"acl": TOP_CATALOG, "mfr": MFR_MID},
    },
    "acl-top+mfr-small": {
        "mask": lambda offers: _matches_acl_and_mfr(offers, TOP_CATALOG, MFR_SMALL),
        "filter_spec": {"acl": TOP_CATALOG, "mfr": MFR_SMALL},
    },
}


def build_es_filter(filter_spec: dict | None) -> list:
    """Translate a regime's filter spec into a knn.filter clause list."""
    if filter_spec is None:
        return []
    inner: list[dict] = []
    if "acl" in filter_spec:
        inner.append({"terms": {"offers.catalogVersionIds": [filter_spec["acl"]]}})
    if "mfr" in filter_spec:
        inner.append({"term": {"offers.manufacturerName": filter_spec["mfr"]}})
    return [
        {
            "nested": {
                "path": "offers",
                "score_mode": "none",
                "query": {"bool": {"filter": inner}},
            }
        }
    ]


# -------------------- article attrs fetch --------------------


def fetch_article_attrs(
    client: httpx.Client, article_ids: list[str]
) -> dict[str, list[dict]]:
    """Pull per-article offers list (catalogVersionIds + manufacturerName only).

    Returns {articleId: [{catalogVersionIds: [...], manufacturerName: ...}, ...]}.
    Articles whose offers carry no manufacturerName get None there.
    """
    attrs: dict[str, list[dict]] = {}
    unique = sorted(set(article_ids))
    for i in range(0, len(unique), TERMS_BATCH):
        batch = unique[i : i + TERMS_BATCH]
        r = client.post(
            f"/{ES_SOURCE_INDEX}/_search",
            json={
                "size": len(batch),
                "track_total_hits": False,
                "_source": [
                    "articleId",
                    "offers.catalogVersionIds",
                    "offers.manufacturerName",
                ],
                "query": {"terms": {"articleId": batch}},
            },
        )
        r.raise_for_status()
        for hit in r.json()["hits"]["hits"]:
            src = hit["_source"]
            aid = src.get("articleId")
            if not aid:
                continue
            offers = []
            for o in src.get("offers") or []:
                offers.append(
                    {
                        "catalogVersionIds": o.get("catalogVersionIds") or [],
                        "manufacturerName": o.get("manufacturerName"),
                    }
                )
            attrs[aid] = offers
    return attrs


def cache_attrs(attrs: dict[str, list[dict]], path: Path) -> None:
    aids = list(attrs.keys())
    offers_json = [json.dumps(attrs[a]) for a in aids]
    pq.write_table(pa.table({"article_id": aids, "offers": offers_json}), path)


def load_attrs_cache(path: Path) -> dict[str, list[dict]]:
    t = pq.read_table(path)
    aids = t["article_id"].to_pylist()
    offers_json = t["offers"].to_pylist()
    return {a: json.loads(j) for a, j in zip(aids, offers_json)}


# -------------------- masks + filtered ground truth --------------------


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def compute_masks(
    row_article_ids: list[str], attrs: dict[str, list[dict]]
) -> dict[str, np.ndarray]:
    """For each regime, build a boolean mask over corpus rows."""
    masks: dict[str, np.ndarray] = {}
    # Resolve once per article to avoid repeated dict lookups.
    art_offers = [attrs.get(aid, []) for aid in row_article_ids]
    for name, regime in REGIMES.items():
        mask_fn = regime["mask"]
        masks[name] = np.fromiter(
            (mask_fn(o) for o in art_offers), dtype=bool, count=len(art_offers)
        )
    return masks


def compute_filtered_gt(
    qv: np.ndarray, cv: np.ndarray, mask: np.ndarray, k: int, chunk: int = 200
) -> np.ndarray:
    """Brute-force exact top-k corpus indices per query, restricted to mask=True rows."""
    if mask.sum() < k:
        # Pad with -1 for queries where the filter passes fewer than k rows.
        out = np.full((qv.shape[0], k), -1, dtype=np.int32)
        passing = np.where(mask)[0]
        if len(passing) == 0:
            return out
        cv_pass = cv[passing]
        for start in range(0, qv.shape[0], chunk):
            stop = min(start + chunk, qv.shape[0])
            sims = qv[start:stop] @ cv_pass.T
            top_local = np.argsort(-sims, axis=1)[:, : min(k, len(passing))]
            out[start:stop, : top_local.shape[1]] = passing[top_local]
        return out
    out = np.empty((qv.shape[0], k), dtype=np.int32)
    not_mask = ~mask
    for start in range(0, qv.shape[0], chunk):
        stop = min(start + chunk, qv.shape[0])
        sims = qv[start:stop] @ cv.T
        sims[:, not_mask] = -np.inf
        top_idx = np.argpartition(-sims, k, axis=1)[:, :k]
        # sort within
        for i in range(stop - start):
            order = np.argsort(-sims[i, top_idx[i]])
            out[start + i] = top_idx[i][order]
    return out


# -------------------- ES test index --------------------


def create_index(client: httpx.Client, name: str, dim: int) -> None:
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
                        "type": PRECISION,
                        "m": INDEX_M,
                        "ef_construction": INDEX_EF_CONSTRUCTION,
                    },
                },
                "offers": {
                    "type": "nested",
                    "properties": {
                        "catalogVersionIds": {"type": "keyword"},
                        "manufacturerName": {"type": "keyword"},
                    },
                },
            }
        },
    }
    r = client.put(f"/{name}", json=body)
    r.raise_for_status()


def _bulk_batch(
    client: httpx.Client,
    name: str,
    vectors: np.ndarray,
    row_article_ids: list[str],
    attrs: dict[str, list[dict]],
    start: int,
    end: int,
) -> int:
    lines: list[str] = []
    for i in range(start, end):
        lines.append('{"index":{"_id":"' + str(i) + '"}}')
        offers = attrs.get(row_article_ids[i], [])
        lines.append(
            json.dumps(
                {"idx": i, "vector": vectors[i].tolist(), "offers": offers}
            )
        )
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


def bulk_load(
    client: httpx.Client,
    name: str,
    vectors: np.ndarray,
    row_article_ids: list[str],
    attrs: dict[str, list[dict]],
) -> int:
    n = vectors.shape[0]
    starts = list(range(0, n, BULK_BATCH))
    loaded = 0
    with ThreadPoolExecutor(max_workers=BULK_CONCURRENCY) as ex:
        futures = [
            ex.submit(
                _bulk_batch,
                client,
                name,
                vectors,
                row_article_ids,
                attrs,
                s,
                min(s + BULK_BATCH, n),
            )
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


def delete_index(client: httpx.Client, name: str) -> None:
    try:
        client.delete(f"/{name}", params={"ignore_unavailable": "true"})
    except Exception:
        pass


# -------------------- search + recall --------------------


def search_one(
    client: httpx.Client,
    name: str,
    query_vec: np.ndarray,
    k: int,
    num_candidates: int,
    filter_clauses: list,
) -> tuple[list[int], float]:
    body: dict = {
        "knn": {
            "field": "vector",
            "query_vector": query_vec.tolist(),
            "k": k,
            "num_candidates": num_candidates,
        },
        "_source": ["idx"],
        "size": k,
    }
    if filter_clauses:
        body["knn"]["filter"] = filter_clauses
    t0 = time.perf_counter()
    r = client.post(f"/{name}/_search", json=body)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    hits = r.json()["hits"]["hits"]
    return [h["_source"]["idx"] for h in hits], elapsed_ms


def recall_at_k(
    hits_per_query: list[list[int]], gt: np.ndarray, k: int
) -> float:
    """Mean |hits ∩ gt| / k across queries. GT rows may be padded with -1 when
    the filter passed fewer than k items; those padding entries don't count."""
    total_score = 0.0
    n = len(hits_per_query)
    for h, gr in zip(hits_per_query, gt):
        gr_set = set(int(x) for x in gr[:k] if x != -1)
        if not gr_set:
            # Filter passed zero items; only correct answer is empty hits.
            total_score += 1.0 if len(h) == 0 else 0.0
            continue
        hit_set = set(h[:k])
        # Normalize by min(k, |gr_set|) so queries with <k passing items aren't
        # penalized for not returning k.
        denom = min(k, len(gr_set))
        total_score += len(hit_set & gr_set) / denom
    return total_score / n


# -------------------- main --------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--es-url", default=ES_URL)
    p.add_argument("--index-name", default=TEST_INDEX)
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--refresh-attrs",
        action="store_true",
        help="Re-fetch article attrs from ES, ignoring the cached parquet.",
    )
    p.add_argument("--keep-index", action="store_true")
    args = p.parse_args()

    qvecs = np.load(args.in_dir / "query_vectors.npy")
    cvecs = np.load(args.in_dir / "corpus_vectors.npy")
    aids_tbl = pq.read_table(args.in_dir / "vector_article_ids.parquet")
    row_article_ids: list[str] = aids_tbl["article_id"].to_pylist()
    manifest = json.loads((args.in_dir / "manifest.json").read_text())

    print(
        f"loaded: {qvecs.shape[0]} queries, {cvecs.shape[0]:,} corpus vectors, "
        f"dim={qvecs.shape[1]}"
    )
    print(f"manifest: {manifest}\n")

    attrs_cache = args.in_dir / "article_filter_attrs.parquet"
    limits = httpx.Limits(
        max_connections=BULK_CONCURRENCY * 2,
        max_keepalive_connections=BULK_CONCURRENCY * 2,
    )

    with httpx.Client(base_url=args.es_url, timeout=300.0, limits=limits) as client:
        if attrs_cache.exists() and not args.refresh_attrs:
            print(f"[1/4] Loading cached article attrs from {attrs_cache.name} ...")
            attrs = load_attrs_cache(attrs_cache)
            print(f"      {len(attrs):,} articles in cache")
        else:
            t0 = time.time()
            unique = sorted(set(row_article_ids))
            print(
                f"[1/4] Fetching offers per article for {len(unique):,} articles "
                f"from {args.es_url}/{ES_SOURCE_INDEX} ..."
            )
            attrs = fetch_article_attrs(client, unique)
            print(f"      got {len(attrs):,} in {time.time() - t0:.1f}s")
            cache_attrs(attrs, attrs_cache)
            print(f"      cached to {attrs_cache.name}")

        # --- masks + per-regime ground truth (numpy, fast) ---
        t0 = time.time()
        print(f"\n[2/4] Computing per-regime masks + filtered ground truth ...")
        masks = compute_masks(row_article_ids, attrs)
        qv_n = normalize_rows(qvecs)
        cv_n = normalize_rows(cvecs)
        regime_gt: dict[str, np.ndarray] = {}
        selectivities: dict[str, float] = {}
        for name, mask in masks.items():
            sel = float(mask.mean())
            selectivities[name] = sel
            n_pass = int(mask.sum())
            print(f"      {name:>20}  selectivity={sel*100:6.3f}%  passing={n_pass:>7,}")
            regime_gt[name] = compute_filtered_gt(qv_n, cv_n, mask, args.k)
        print(f"      done in {time.time() - t0:.1f}s")

        # --- build + populate one test index, reused across all regimes ---
        try:
            print(
                f"\n[3/4] Building ES test index {args.index_name} "
                f"({PRECISION}, m={INDEX_M}, ef_construction={INDEX_EF_CONSTRUCTION}) ..."
            )
            create_index(client, args.index_name, cvecs.shape[1])

            t0 = time.time()
            bulk_load(client, args.index_name, cvecs, row_article_ids, attrs)
            load_s = time.time() - t0
            print(f"      bulk-loaded {cvecs.shape[0]:,} docs in {load_s:.1f}s")

            t0 = time.time()
            finalize(client, args.index_name)
            merge_s = time.time() - t0
            store_mb = index_store_bytes(client, args.index_name) / 1024 / 1024
            print(
                f"      refresh + force-merge in {merge_s:.1f}s, store={store_mb:.1f} MB"
            )

            # --- sweep regimes × num_candidates ---
            print(f"\n[4/4] Sweeping {len(REGIMES)} regimes × {len(NUM_CANDIDATES)} num_candidates")
            hdr = (
                f"{'regime':>20} {'sel%':>6} {'numC':>5}  "
                f"{'p50_ms':>7}  {'p95_ms':>7}  {'rec@' + str(args.k):>8}"
            )
            print(hdr)
            print("-" * len(hdr))

            results: list[dict] = []
            for regime_name, regime in REGIMES.items():
                filter_clauses = build_es_filter(regime["filter_spec"])
                gt = regime_gt[regime_name]
                sel = selectivities[regime_name]
                for numc in NUM_CANDIDATES:
                    eff_numc = max(numc, args.k)
                    # warm-up
                    search_one(
                        client, args.index_name, qvecs[0], args.k, eff_numc, filter_clauses
                    )

                    hits_per_query: list[list[int]] = []
                    latencies: list[float] = []
                    for q in qvecs:
                        h, ms = search_one(
                            client,
                            args.index_name,
                            q,
                            args.k,
                            eff_numc,
                            filter_clauses,
                        )
                        hits_per_query.append(h)
                        latencies.append(ms)

                    r = recall_at_k(hits_per_query, gt, args.k)
                    p50 = float(np.percentile(latencies, 50))
                    p95 = float(np.percentile(latencies, 95))

                    print(
                        f"{regime_name:>20} {sel*100:>5.2f}% {eff_numc:>5}  "
                        f"{p50:>7.2f}  {p95:>7.2f}  {r:>8.4f}"
                    )
                    results.append(
                        {
                            "regime": regime_name,
                            "filter_spec": regime["filter_spec"],
                            "selectivity": round(sel, 6),
                            "num_candidates": eff_numc,
                            "precision": PRECISION,
                            "m": INDEX_M,
                            "ef_construction": INDEX_EF_CONSTRUCTION,
                            "query_ms_p50": round(p50, 3),
                            "query_ms_p95": round(p95, 3),
                            f"recall_at_{args.k}": round(r, 5),
                        }
                    )

        finally:
            if not args.keep_index:
                delete_index(client, args.index_name)

    out_path = args.in_dir / f"bench_recall_at_{args.k}_filtered.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
