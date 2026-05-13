"""Production-shape HNSW bench against local-article-index-v2.

Differs from scripts/bench_hnsw*.py (which bulk-load a small flat test index):
  - Queries the REAL prod-shape ES index (32 shards × ~375k articles per shard,
    nested embeddings.vector, int8_hnsw m=16 efc=100 on v2).
  - Recall is measured against the **full-corpus oracle** (12M vectors,
    brute-force article-level top-100) at reports/hnsw_eval_full/.
  - 8 filter regimes mirror the sweep 3 production-realistic envelope.
  - Configurable query --concurrency.

Phases (each prints elapsed/ETA):
  [1/4] Sliced-PIT pull of per-article filter attrs across all ~6M articles
        (cached to article_filter_attrs.parquet — skip on re-runs).
  [2/4] Build per-regime corpus masks + compute filtered article-level GT
        from the 12M-vector oracle (cached to filtered_gt.parquet).
  [3/4] Warmup one query per (regime, numC).
  [4/4] Sweep 8 regimes × 8 num_candidates × 1000 queries, --concurrency parallel.

Usage:
  uv run python scripts/bench_hnsw_prod_shape.py --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_QUERIES_DIR = REPO_ROOT / "reports" / "hnsw_eval"
DEFAULT_ORACLE_DIR = REPO_ROOT / "reports" / "hnsw_eval_full"

ES_URL = "http://localhost:9200"
ES_INDEX = "local-article-index-v2"

PIT_KEEP_ALIVE = "30m"
PAGE_SIZE = 1000
ES_SLICES = 8

# Filter values — match scripts/bench_hnsw_filtered.py exactly.
TOP_CATALOG = "844224b4-e40b-4180-bf04-4f74d702627d"
MID_CATALOG = "fcf0f564-e4ea-4f93-b21b-9f690c687732"
MFR_MID = "Weidmüller"
MFR_SMALL = "ATORN"
CAT_TOP = "07 - Halbzeuge / Montagematerial¦Halbzeuge¦Schrauben / Muttern"
PRICE_GTE = 50.0
PRICE_LTE = 200.0

NUM_CANDIDATES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
BENCH_PROGRESS_EVERY = 100  # print update every N queries within a cell


# -------------------- progress helper --------------------


def fmt_eta(done: int, total: int, t0: float) -> str:
    """Format `done/total  elapsed=Xs  rate=Y/s  ETA=Zs` (or `--` while warming up)."""
    elapsed = time.time() - t0
    if done == 0:
        return f"{done}/{total}  elapsed={elapsed:5.1f}s  ETA=    --"
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - done)
    eta = remaining / rate if rate > 0 else 0.0
    return f"{done}/{total}  elapsed={elapsed:5.1f}s  rate={rate:6.1f}/s  ETA={eta:6.1f}s"


# -------------------- regime definitions --------------------


def _offers_match(offers: list[dict], spec: dict) -> bool:
    def offer_ok(o: dict) -> bool:
        if "acl" in spec and spec["acl"] not in (o.get("catalogVersionIds") or []):
            return False
        if "mfr" in spec and o.get("manufacturerName") != spec["mfr"]:
            return False
        if "cat" in spec and spec["cat"] not in (o.get("categoryPath3") or []):
            return False
        return True

    return any(offer_ok(o) for o in offers)


def _prices_match(prices: list[float], spec: dict) -> bool:
    if "price_gte" not in spec and "price_lte" not in spec:
        return True
    lo = spec.get("price_gte", float("-inf"))
    hi = spec.get("price_lte", float("inf"))
    return any(lo <= p <= hi for p in prices)


def _matches(article: dict, spec: dict | None) -> bool:
    if spec is None:
        return True
    offers_side = {k: v for k, v in spec.items() if k in ("acl", "mfr", "cat")}
    prices_side = {k: v for k, v in spec.items() if k in ("price_gte", "price_lte")}
    if offers_side and not _offers_match(article["offers"], offers_side):
        return False
    if prices_side and not _prices_match(article["prices"], prices_side):
        return False
    return True


REGIMES: dict[str, dict] = {
    "unfiltered": {"filter_spec": None},
    "acl-top": {"filter_spec": {"acl": TOP_CATALOG}},
    "acl-mid": {"filter_spec": {"acl": MID_CATALOG}},
    "acl-top+cat-top": {"filter_spec": {"acl": TOP_CATALOG, "cat": CAT_TOP}},
    "acl-top+price-50-200": {
        "filter_spec": {"acl": TOP_CATALOG, "price_gte": PRICE_GTE, "price_lte": PRICE_LTE}
    },
    "acl-top+mfr-mid": {"filter_spec": {"acl": TOP_CATALOG, "mfr": MFR_MID}},
    "acl-top+mfr-mid+price-50-200": {
        "filter_spec": {
            "acl": TOP_CATALOG,
            "mfr": MFR_MID,
            "price_gte": PRICE_GTE,
            "price_lte": PRICE_LTE,
        }
    },
    "acl-top+mfr-small": {"filter_spec": {"acl": TOP_CATALOG, "mfr": MFR_SMALL}},
}


def build_es_filter(filter_spec: dict | None) -> list:
    """Translate a regime spec into a knn.filter list (offers-side in one nested,
    prices-side in another) — byte-equivalent to TEST_PROFILE_18."""
    if filter_spec is None:
        return []
    clauses: list[dict] = []
    offers_inner: list[dict] = []
    if "acl" in filter_spec:
        offers_inner.append({"terms": {"offers.catalogVersionIds": [filter_spec["acl"]]}})
    if "mfr" in filter_spec:
        offers_inner.append({"term": {"offers.manufacturerName": filter_spec["mfr"]}})
    if "cat" in filter_spec:
        offers_inner.append({"term": {"offers.categoryPaths.upToLevel3": filter_spec["cat"]}})
    if offers_inner:
        clauses.append(
            {
                "nested": {
                    "path": "offers",
                    "score_mode": "none",
                    "query": {"bool": {"filter": offers_inner}},
                }
            }
        )
    if "price_gte" in filter_spec or "price_lte" in filter_spec:
        rng: dict = {}
        if "price_gte" in filter_spec:
            rng["gte"] = filter_spec["price_gte"]
        if "price_lte" in filter_spec:
            rng["lte"] = filter_spec["price_lte"]
        clauses.append(
            {
                "nested": {
                    "path": "prices",
                    "score_mode": "none",
                    "query": {"range": {"prices.price": rng}},
                }
            }
        )
    return clauses


# -------------------- phase 1: sliced PIT attrs fetch --------------------


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


def pull_attrs_slice(
    es_url: str,
    pit_id: str,
    slice_id: int,
    max_slices: int,
    progress_q: list,
) -> list[dict]:
    """Iterate one PIT slice; collect filter attrs per article."""
    out: list[dict] = []
    with httpx.Client(base_url=es_url, timeout=300.0) as client:
        search_after: list | None = None
        last_logged = 0
        while True:
            body: dict = {
                "size": PAGE_SIZE,
                "track_total_hits": False,
                "pit": {"id": pit_id, "keep_alive": PIT_KEEP_ALIVE},
                "slice": {"id": slice_id, "max": max_slices},
                "_source": [
                    "articleId",
                    "offers.catalogVersionIds",
                    "offers.manufacturerName",
                    "offers.categoryPaths.upToLevel3",
                    "prices.price",
                ],
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
                offers = []
                for o in src.get("offers") or []:
                    cat_raw = (o.get("categoryPaths") or {}).get("upToLevel3")
                    if cat_raw is None:
                        cat_list = []
                    elif isinstance(cat_raw, list):
                        cat_list = cat_raw
                    else:
                        cat_list = [cat_raw]
                    offers.append(
                        {
                            "catalogVersionIds": o.get("catalogVersionIds") or [],
                            "manufacturerName": o.get("manufacturerName"),
                            "categoryPath3": cat_list,
                        }
                    )
                prices = [
                    p["price"]
                    for p in src.get("prices") or []
                    if p.get("price") is not None
                ]
                out.append({"articleId": aid, "offers": offers, "prices": prices})
            search_after = hits[-1]["sort"]
            # publish per-slice progress every ~10k docs
            if len(out) - last_logged >= 10_000:
                progress_q.append((slice_id, len(out)))
                last_logged = len(out)
    progress_q.append((slice_id, len(out)))
    return out


def pull_all_attrs(es_url: str, n_total_estimate: int) -> dict[str, dict]:
    print(f"  opening PIT on {ES_INDEX} ...")
    pit_id = open_pit(es_url)
    progress_q: list = []  # appended as (slice_id, current_count) tuples
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=ES_SLICES) as ex:
            futures = {
                ex.submit(
                    pull_attrs_slice, es_url, pit_id, i, ES_SLICES, progress_q
                ): i
                for i in range(ES_SLICES)
            }
            slice_counts = {i: 0 for i in range(ES_SLICES)}
            completed = 0
            pending = set(futures.values())
            # poll progress while futures run
            while completed < ES_SLICES:
                done_now = [f for f in futures if f.done() and futures[f] in pending]
                # drain progress queue
                while progress_q:
                    sid, n = progress_q.pop(0)
                    slice_counts[sid] = n
                total_so_far = sum(slice_counts.values())
                pct = (
                    f"{100 * total_so_far / n_total_estimate:4.1f}%"
                    if n_total_estimate
                    else "    "
                )
                msg = (
                    f"  [{pct}] {total_so_far:>9,}/{n_total_estimate:>9,} attrs  "
                    f"{fmt_eta(total_so_far, n_total_estimate or 1, t0)}"
                )
                sys.stdout.write("\r" + msg.ljust(110))
                sys.stdout.flush()
                for f in done_now:
                    pending.discard(futures[f])
                    completed += 1
                if completed < ES_SLICES:
                    time.sleep(1.0)
            sys.stdout.write("\n")
            sys.stdout.flush()
            all_results: list[list[dict]] = [list() for _ in range(ES_SLICES)]
            for f, sid in futures.items():
                all_results[sid] = f.result()
    finally:
        close_pit(es_url, pit_id)
    attrs: dict[str, dict] = {}
    for chunk in all_results:
        for rec in chunk:
            attrs[rec["articleId"]] = {"offers": rec["offers"], "prices": rec["prices"]}
    return attrs


def cache_attrs(attrs: dict[str, dict], path: Path) -> None:
    aids = list(attrs.keys())
    blobs = [
        json.dumps({"offers": attrs[a]["offers"], "prices": attrs[a]["prices"]})
        for a in aids
    ]
    pq.write_table(pa.table({"article_id": aids, "attrs": blobs}), path)


def load_attrs_cache(path: Path) -> dict[str, dict]:
    t = pq.read_table(path)
    aids = t["article_id"].to_pylist()
    blobs = t["attrs"].to_pylist()
    out: dict[str, dict] = {}
    for a, b in zip(aids, blobs):
        out[a] = json.loads(b)
    return out


# -------------------- phase 2: masks + filtered GT --------------------


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def compute_masks(
    row_article_ids: np.ndarray, attrs: dict[str, dict]
) -> dict[str, np.ndarray]:
    """Per-corpus-row bool mask for each regime.

    Two-stage: (1) compute per-unique-article pass for each regime (~6M Python
    ops × 8 regimes); (2) numpy gather to expand back to per-corpus-row mask.
    Avoids 12M × 8 = 96M redundant per-row checks.
    """
    n = len(row_article_ids)
    unique_aids = sorted(attrs.keys())
    n_unique = len(unique_aids)
    aid_to_idx = {a: i for i, a in enumerate(unique_aids)}
    # row_idx[i] = position in unique_aids, or n_unique sentinel for unknown
    t_map = time.time()
    row_idx = np.fromiter(
        (aid_to_idx.get(a, n_unique) for a in row_article_ids),
        dtype=np.int64,
        count=n,
    )
    print(
        f"  built row→unique mapping ({n:,} rows, {n_unique:,} unique) "
        f"in {time.time() - t_map:.1f}s"
    )

    masks: dict[str, np.ndarray] = {}
    t0 = time.time()
    for ridx, (name, regime) in enumerate(REGIMES.items()):
        spec = regime["filter_spec"]
        # +1 slot for the unknown-sentinel (n_unique), always False
        pass_arr = np.zeros(n_unique + 1, dtype=bool)
        for i, aid in enumerate(unique_aids):
            pass_arr[i] = _matches(attrs[aid], spec)
        m = pass_arr[row_idx]
        masks[name] = m
        sel = float(m.mean())
        print(
            f"  [{ridx + 1}/{len(REGIMES)}] {name:>30}  sel={sel * 100:6.3f}%  "
            f"passing={int(m.sum()):>10,}  elapsed={time.time() - t0:.1f}s"
        )
    return masks


def filtered_gt(
    qv: np.ndarray,
    cv: np.ndarray,
    mask: np.ndarray,
    row_article_ids: np.ndarray,
    k: int,
    overfetch: int,
    chunk: int,
    label: str,
) -> list[list[str]]:
    """Filtered article-level top-K GT for each query.

    Strategy: chunk queries, BLAS for IP, mask non-passing cols to -inf,
    top-{overfetch} vectors, group by article_id, keep first-seen top-K aids.
    """
    n_q = qv.shape[0]
    passing = int(mask.sum())
    out: list[list[str]] = [[] for _ in range(n_q)]
    if passing == 0:
        for q_idx in range(n_q):
            out[q_idx] = [""] * k
        print(f"    {label}  passing=0 → all-empty GT (filter passes nothing)")
        return out

    not_mask = ~mask
    eff_overfetch = min(overfetch, cv.shape[0])
    t0 = time.time()
    for start in range(0, n_q, chunk):
        stop = min(start + chunk, n_q)
        sims = qv[start:stop] @ cv.T
        sims[:, not_mask] = -np.inf
        if eff_overfetch < cv.shape[0]:
            top_idx = np.argpartition(-sims, eff_overfetch - 1, axis=1)[:, :eff_overfetch]
        else:
            top_idx = np.tile(np.arange(cv.shape[0]), (stop - start, 1))
        for i in range(stop - start):
            order = np.argsort(-sims[i, top_idx[i]])
            ordered = top_idx[i][order]
            seen: list[str] = []
            seen_set: set[str] = set()
            for vi in ordered:
                aid = row_article_ids[vi]
                if not aid or aid in seen_set:
                    continue
                seen_set.add(aid)
                seen.append(aid)
                if len(seen) >= k:
                    break
            if len(seen) < k:
                seen.extend([""] * (k - len(seen)))
            out[start + i] = seen
        msg = f"    {label}  {fmt_eta(stop, n_q, t0)}"
        sys.stdout.write("\r" + msg.ljust(110))
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return out


def cache_filtered_gt(
    gts: dict[str, list[list[str]]], path: Path
) -> None:
    rows = []
    for name, gt in gts.items():
        for q_idx, aids in enumerate(gt):
            rows.append((name, q_idx, aids))
    pq.write_table(
        pa.table(
            {
                "regime": [r[0] for r in rows],
                "query_idx": [r[1] for r in rows],
                "top_article_ids": [r[2] for r in rows],
            }
        ),
        path,
    )


def load_filtered_gt(path: Path) -> dict[str, list[list[str]]]:
    t = pq.read_table(path)
    regimes = t["regime"].to_pylist()
    qidxs = t["query_idx"].to_pylist()
    aids_list = t["top_article_ids"].to_pylist()
    grouped: dict[str, dict[int, list[str]]] = {}
    for r, qi, aids in zip(regimes, qidxs, aids_list):
        grouped.setdefault(r, {})[qi] = list(aids)
    out: dict[str, list[list[str]]] = {}
    for r, qmap in grouped.items():
        n_q = max(qmap.keys()) + 1
        out[r] = [qmap[i] for i in range(n_q)]
    return out


# -------------------- phase 3+4: ES bench --------------------


def search_one(
    client: httpx.Client,
    index: str,
    query_vec: np.ndarray,
    k: int,
    num_candidates: int,
    filter_clauses: list,
) -> tuple[list[str], float]:
    body: dict = {
        "knn": {
            "field": "embeddings.vector",
            "query_vector": query_vec.tolist(),
            "k": k,
            "num_candidates": num_candidates,
        },
        "_source": ["articleId"],
        "size": k,
    }
    if filter_clauses:
        body["knn"]["filter"] = filter_clauses
    t0 = time.perf_counter()
    r = client.post(f"/{index}/_search", json=body)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    hits = r.json()["hits"]["hits"]
    return [h["_source"]["articleId"] for h in hits], elapsed_ms


def recall_at_k_articles(
    hits_per_query: list[list[str]],
    gt: list[list[str]],
    k: int,
) -> float:
    """Article-level recall@k. GT may be padded with '' when filter passes fewer."""
    total = 0.0
    n = len(hits_per_query)
    for h, gr in zip(hits_per_query, gt):
        gr_set = {a for a in gr[:k] if a}
        if not gr_set:
            total += 1.0 if len(h) == 0 else 0.0
            continue
        h_set = {a for a in h[:k] if a}
        denom = min(k, len(gr_set))
        total += len(h_set & gr_set) / denom
    return total / n


def run_cell(
    es_url: str,
    index: str,
    qvecs: np.ndarray,
    k: int,
    num_candidates: int,
    filter_clauses: list,
    concurrency: int,
    cell_label: str,
) -> tuple[list[list[str]], list[float]]:
    """Run all 1000 queries for one (regime × numC) cell, parallel."""
    n_q = qvecs.shape[0]
    hits: list[list[str]] = [[] for _ in range(n_q)]
    lat: list[float] = [0.0] * n_q
    limits = httpx.Limits(
        max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2
    )
    t0 = time.time()
    done = 0
    next_print = 0
    with httpx.Client(base_url=es_url, timeout=120.0, limits=limits) as client:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(
                    search_one,
                    client,
                    index,
                    qvecs[i],
                    k,
                    num_candidates,
                    filter_clauses,
                ): i
                for i in range(n_q)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                hits[i], lat[i] = fut.result()
                done += 1
                if done >= next_print or done == n_q:
                    msg = f"    {cell_label}  {fmt_eta(done, n_q, t0)}"
                    sys.stdout.write("\r" + msg.ljust(110))
                    sys.stdout.flush()
                    next_print = done + BENCH_PROGRESS_EVERY
    sys.stdout.write("\n")
    sys.stdout.flush()
    return hits, lat


# -------------------- main --------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries-dir", type=Path, default=DEFAULT_QUERIES_DIR)
    p.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    p.add_argument("--es-url", default=ES_URL)
    p.add_argument("--es-index", default=ES_INDEX)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--gt-overfetch",
        type=int,
        default=2000,
        help="Top-N vectors per query before article-collapse for filtered GT.",
    )
    p.add_argument(
        "--gt-chunk",
        type=int,
        default=16,
        help="Query chunk size for filtered GT computation (memory-bound).",
    )
    p.add_argument(
        "--refresh-attrs",
        action="store_true",
        help="Re-fetch article attrs from ES, ignoring cached parquet.",
    )
    p.add_argument(
        "--refresh-gt",
        action="store_true",
        help="Re-compute filtered GT, ignoring cached parquet.",
    )
    p.add_argument(
        "--num-candidates",
        type=int,
        nargs="+",
        default=NUM_CANDIDATES,
        help="num_candidates values to sweep.",
    )
    p.add_argument(
        "--regimes",
        type=str,
        nargs="+",
        default=None,
        help="Subset of regimes to bench (default: all 8).",
    )
    p.add_argument(
        "--warmup-passes",
        type=int,
        default=3,
        help="Warmup passes; each fires 1 query per (regime, numC, diverse-q).",
    )
    p.add_argument(
        "--warmup-queries",
        type=int,
        default=4,
        help="Number of distinct query vectors used during warmup.",
    )
    args = p.parse_args()

    overall_t0 = time.time()

    # ---------- load queries + oracle ----------
    print(f"[0/4] Loading queries + oracle ...")
    qvecs = np.load(args.queries_dir / "query_vectors.npy").astype(np.float32)
    n_q = qvecs.shape[0]
    print(f"      {n_q} query vectors loaded (dim={qvecs.shape[1]})")
    corpus_attrs = pq.read_table(args.oracle_dir / "corpus_attrs.parquet")
    row_article_ids = np.array(corpus_attrs["article_id"].to_pylist())
    n_corpus = len(row_article_ids)
    print(f"      {n_corpus:,} corpus rows from oracle (corpus_attrs.parquet)")
    oracle_manifest = json.loads((args.oracle_dir / "manifest.json").read_text())
    print(f"      oracle manifest: {oracle_manifest}")

    # ---------- phase 1: per-article filter attrs ----------
    attrs_cache = args.oracle_dir / "article_filter_attrs.parquet"
    print(f"\n[1/4] Per-article filter attrs (cache: {attrs_cache.name}) ...")
    if attrs_cache.exists() and not args.refresh_attrs:
        t0 = time.time()
        print(f"      loading cached attrs ...")
        attrs = load_attrs_cache(attrs_cache)
        print(f"      {len(attrs):,} articles loaded in {time.time() - t0:.1f}s")
    else:
        t0 = time.time()
        print(f"      sliced-PIT pull across {ES_SLICES} slices ...")
        attrs = pull_all_attrs(args.es_url, oracle_manifest["n_unique_articles"])
        print(f"      pulled {len(attrs):,} articles in {time.time() - t0:.1f}s")
        t1 = time.time()
        cache_attrs(attrs, attrs_cache)
        print(f"      cached to {attrs_cache.name} in {time.time() - t1:.1f}s")

    # ---------- phase 2: masks + filtered GT ----------
    selected_regimes = (
        args.regimes if args.regimes else list(REGIMES.keys())
    )
    for r in selected_regimes:
        if r not in REGIMES:
            raise SystemExit(f"unknown regime: {r}")

    print(f"\n[2/4] Per-regime masks + filtered article-level GT (k={args.k}) ...")
    t0 = time.time()
    masks = compute_masks(row_article_ids, attrs)
    print(f"      masks done in {time.time() - t0:.1f}s")

    gt_cache = args.oracle_dir / f"filtered_gt_top{args.k}.parquet"
    if gt_cache.exists() and not args.refresh_gt:
        print(f"      loading cached filtered GT from {gt_cache.name} ...")
        all_gt = load_filtered_gt(gt_cache)
        missing = [r for r in selected_regimes if r not in all_gt]
        if missing:
            print(f"      cache missing regimes {missing} → recomputing")
        else:
            print(f"      all {len(selected_regimes)} regimes hit cache")
    else:
        all_gt = {}
        missing = list(selected_regimes)

    if missing:
        print(f"      computing filtered GT for {len(missing)} regimes "
              f"(overfetch={args.gt_overfetch}, chunk={args.gt_chunk}) ...")
        t0 = time.time()
        print(f"      loading corpus_vectors.npy ...")
        cvecs = np.load(args.oracle_dir / "corpus_vectors.npy")
        print(f"      loaded {cvecs.shape} fp32 ({cvecs.nbytes / 1e9:.2f} GB) "
              f"in {time.time() - t0:.1f}s")
        t0 = time.time()
        print(f"      L2-normalizing queries + corpus ...")
        qv_n = normalize_rows(qvecs)
        cv_n = normalize_rows(cvecs)
        del cvecs
        print(f"      normalized in {time.time() - t0:.1f}s")
        for ridx, name in enumerate(missing):
            t_r = time.time()
            label = f"[{ridx + 1}/{len(missing)}] {name}"
            all_gt[name] = filtered_gt(
                qv_n,
                cv_n,
                masks[name],
                row_article_ids,
                args.k,
                args.gt_overfetch,
                args.gt_chunk,
                label,
            )
            print(f"      {name} GT done in {time.time() - t_r:.1f}s")
        cache_filtered_gt(all_gt, gt_cache)
        print(f"      filtered GT cached to {gt_cache.name}")
        # free large corpus
        del cv_n, qv_n

    # ---------- phase 3: warmup ----------
    # Cold first-query on 32 shards can take 10+ seconds. Do a few passes
    # with diverse query vectors to fault-in HNSW graph pages across shards.
    warmup_passes = max(0, args.warmup_passes)
    warmup_n_q = max(1, args.warmup_queries)
    warmup_qs = [int(i * n_q / warmup_n_q) for i in range(warmup_n_q)]
    n_warm = (
        len(selected_regimes) * len(args.num_candidates) * len(warmup_qs) * warmup_passes
    )
    print(
        f"\n[3/4] Warming up shards ({len(selected_regimes)} regimes × "
        f"{len(args.num_candidates)} numC × {len(warmup_qs)} diverse queries × "
        f"{warmup_passes} passes = {n_warm} ops) ..."
    )
    t0 = time.time()
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=8)
    with httpx.Client(base_url=args.es_url, timeout=300.0, limits=limits) as client:
        done = 0
        for _ in range(warmup_passes):
            for regime_name in selected_regimes:
                filter_clauses = build_es_filter(REGIMES[regime_name]["filter_spec"])
                for numc in args.num_candidates:
                    eff = max(numc, args.k)
                    for qi in warmup_qs:
                        search_one(client, args.es_index, qvecs[qi], args.k, eff, filter_clauses)
                        done += 1
                        msg = f"      warmup  {fmt_eta(done, n_warm, t0)}"
                        sys.stdout.write("\r" + msg.ljust(110))
                        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

    # ---------- phase 4: bench loop ----------
    n_cells = len(selected_regimes) * len(args.num_candidates)
    print(f"\n[4/4] Sweeping {len(selected_regimes)} regimes × "
          f"{len(args.num_candidates)} num_candidates × {n_q} queries "
          f"(concurrency={args.concurrency}) — {n_cells} cells total")
    hdr = (
        f"  {'regime':>30}  {'sel%':>6}  {'numC':>5}  "
        f"{'p50_ms':>7}  {'p95_ms':>7}  {'p99_ms':>7}  {'rec@' + str(args.k):>8}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    results: list[dict] = []
    cell_idx = 0
    phase_t0 = time.time()
    for regime_name in selected_regimes:
        filter_clauses = build_es_filter(REGIMES[regime_name]["filter_spec"])
        sel = float(masks[regime_name].mean())
        gt = all_gt[regime_name]
        for numc in args.num_candidates:
            cell_idx += 1
            eff = max(numc, args.k)
            cell_label = (
                f"cell {cell_idx}/{n_cells}: {regime_name} numC={eff}"
            )
            hits, lat = run_cell(
                args.es_url,
                args.es_index,
                qvecs,
                args.k,
                eff,
                filter_clauses,
                args.concurrency,
                cell_label,
            )
            rec = recall_at_k_articles(hits, gt, args.k)
            p50 = float(np.percentile(lat, 50))
            p95 = float(np.percentile(lat, 95))
            p99 = float(np.percentile(lat, 99))
            print(
                f"  {regime_name:>30}  {sel * 100:>5.2f}%  {eff:>5}  "
                f"{p50:>7.2f}  {p95:>7.2f}  {p99:>7.2f}  {rec:>8.4f}"
            )
            results.append(
                {
                    "regime": regime_name,
                    "filter_spec": REGIMES[regime_name]["filter_spec"],
                    "selectivity": round(sel, 6),
                    "num_candidates": eff,
                    "query_ms_p50": round(p50, 3),
                    "query_ms_p95": round(p95, 3),
                    "query_ms_p99": round(p99, 3),
                    f"recall_at_{args.k}": round(rec, 5),
                    "n_queries": n_q,
                    "concurrency": args.concurrency,
                }
            )
            elapsed = time.time() - phase_t0
            rate = cell_idx / elapsed if elapsed > 0 else 0
            cells_left = n_cells - cell_idx
            eta = cells_left / rate if rate > 0 else 0
            print(
                f"  ── phase4 progress: {cell_idx}/{n_cells} cells  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
            )

    out_path = args.oracle_dir / f"bench_prod_shape_top{args.k}.json"
    out_path.write_text(
        json.dumps(
            {
                "es_url": args.es_url,
                "es_index": args.es_index,
                "n_queries": n_q,
                "n_corpus_vectors": n_corpus,
                "concurrency": args.concurrency,
                "k": args.k,
                "wall_seconds": round(time.time() - overall_t0, 1),
                "results": results,
            },
            indent=2,
        )
    )
    print(
        f"\nDone in {time.time() - overall_t0:.1f}s. Wrote {out_path}"
    )


if __name__ == "__main__":
    main()
