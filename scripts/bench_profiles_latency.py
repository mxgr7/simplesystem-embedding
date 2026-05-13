"""Latency bench — STANDARD vs TEST_PROFILE_18 (lexical / vector / hybrid).

All four profile families hit the same v3 index (local-article-index-v3,
fp32 hnsw, additive mapping per FT_ELASTIC_IMPORT.md §2.1) so the comparison
isolates query-shape from infrastructure.

Profiles benched:
  - STANDARD             : §1.2 — three sub-queries, pre/post filter split,
                                  search_type=dfs_query_then_fetch.
  - TP18-lexical         : §2.2.1 pruned offer bool (clauses b–g only),
                                  single filter list, no DFS.
  - TP18-vector @ numC   : §2.2.2 top-level knn block.
  - TP18-hybrid @ numC   : §2.2.5 app-side RRF — parallel legs + phase-2 page fetch.

Customer-article-number sub-query is dropped (no customer context in synthetic
bench); recall/relevance are out of scope (latency-only by user direction).

Usage:
  uv run python scripts/bench_profiles_latency.py --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse helpers from the prod-shape bench (same regime defs, same harness shape).
from scripts.bench_hnsw_prod_shape import (  # noqa: E402
    REGIMES,
    TOP_CATALOG,
    fmt_eta,
)

DEFAULT_QUERIES_DIR = REPO_ROOT / "reports" / "hnsw_eval"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "hnsw_eval_full"

ES_URL = "http://localhost:9200"
ES_INDEX = "local-article-index-v3"

# 5 production-shape regimes (same as Sweep 4).
BENCH_REGIMES = [
    "unfiltered",
    "acl-top",
    "acl-top+cat-top",
    "acl-top+mfr-mid",
    "acl-top+price-50-200",
]

NUMC_VARIANTS = [1000, 5000]  # for TP18-vector and TP18-hybrid kNN leg

DEFAULT_K = 50
BENCH_PROGRESS_EVERY = 100

# STANDARD synthetic-keywords config — analyzer matches the field's index-time
# analyzer (german_strict); boost is a sensible middle value. Exact values are
# configurable in prod (SyntheticKeywordsProperties).
SYN_KW_ANALYZER = "german_strict"
SYN_KW_BOOST = 50


# ============================================================
# Filter construction
# ============================================================


def _offers_inner_for_spec(spec: dict, keys: tuple[str, ...]) -> list[dict]:
    """Inner clauses for one nested(offers, ...) block — same-offer semantics
    require ALL constraints in one nested clause (§2.2.2 construction rule)."""
    inner: list[dict] = []
    if "acl" in keys and "acl" in spec:
        inner.append({"terms": {"offers.catalogVersionIds": [spec["acl"]]}})
    if "mfr" in keys and "mfr" in spec:
        inner.append({"term": {"offers.manufacturerName": spec["mfr"]}})
    if "cat" in keys and "cat" in spec:
        inner.append({"term": {"offers.categoryPaths.upToLevel3": spec["cat"]}})
    return inner


def _prices_nested_for_spec(spec: dict) -> dict | None:
    if "price_gte" not in spec and "price_lte" not in spec:
        return None
    rng: dict = {}
    if "price_gte" in spec:
        rng["gte"] = spec["price_gte"]
    if "price_lte" in spec:
        rng["lte"] = spec["price_lte"]
    return {
        "nested": {
            "path": "prices",
            "score_mode": "none",
            "query": {"range": {"prices.price": rng}},
        }
    }


def _wrap_offers_nested(inner: list[dict]) -> dict | None:
    if not inner:
        return None
    return {
        "nested": {
            "path": "offers",
            "score_mode": "none",
            "query": {"bool": {"filter": inner}},
        }
    }


def build_tp18_filter(spec: dict | None) -> list[dict]:
    """Full flat filter list for TP18 (knn.filter / bool.filter)."""
    if spec is None:
        return []
    clauses: list[dict] = []
    offers_inner = _offers_inner_for_spec(spec, ("acl", "mfr", "cat"))
    nested = _wrap_offers_nested(offers_inner)
    if nested:
        clauses.append(nested)
    prices = _prices_nested_for_spec(spec)
    if prices:
        clauses.append(prices)
    return clauses


def build_standard_filters(spec: dict | None) -> tuple[list[dict], list[dict]]:
    """STANDARD pre/post split (§1.3.1).

    Pre-filter (bool.filter) carries structural pre-filters — ACL only in our
    regimes. Post-filter carries user-selected facets — manufacturer, category,
    price range. Same-offer semantics: ACL gets its own nested clause; facets
    that also touch offers share a single nested clause."""
    if spec is None:
        return [], []
    pre: list[dict] = []
    post: list[dict] = []
    if "acl" in spec:
        nested = _wrap_offers_nested(
            _offers_inner_for_spec({"acl": spec["acl"]}, ("acl",))
        )
        if nested:
            pre.append(nested)
    facet_offers_inner = _offers_inner_for_spec(spec, ("mfr", "cat"))
    facet_nested = _wrap_offers_nested(facet_offers_inner)
    if facet_nested:
        post.append(facet_nested)
    facet_prices = _prices_nested_for_spec(spec)
    if facet_prices:
        post.append(facet_prices)
    return pre, post


# ============================================================
# Query body builders
# ============================================================


def _offer_clauses_std(q: str) -> list[dict]:
    """STANDARD offer sub-query — clauses (a)+(b)+(c)+(d)+(e)+(f)+(g) per §1.2.1."""
    return [
        # (a) cross-fields multi_match — boost 99
        {
            "multi_match": {
                "query": q,
                "fields": [
                    "offers.articleNumber.seg^5",
                    "offers.name.de_joined^5",
                    "offers.name.de_strict^3",
                    "offers.manufacturerName.de2^2",
                    "offers.vendorName.de2^2",
                    "offers.keywords.de2^1",
                ],
                "type": "cross_fields",
                "analyzer": "german_strict",
                "operator": "and",
                "tie_breaker": 1.0,
                "minimum_should_match": "100%",
                "fuzzy_transpositions": False,
                "auto_generate_synonyms_phrase_query": False,
                "boost": 99,
            }
        },
        # (b) articleNumber.raw — boost 50
        _id_match("offers.articleNumber.raw", q, "or", "1", 50),
        # (c) articleNumber.normalized — boost 50
        _id_match("offers.articleNumber.normalized", q, "or", "1", 50),
        # (d) articleNumber.seg AND — boost 30
        _id_match("offers.articleNumber.seg", q, "and", "100%", 30),
        # (e) manufacturerArticleNumber.raw — boost 30
        _id_match("offers.manufacturerArticleNumber.raw", q, "or", "1", 30),
        # (f) manufacturerArticleNumber.normalized — boost 30
        _id_match("offers.manufacturerArticleNumber.normalized", q, "or", "1", 30),
        # (g) ean.raw — boost 30
        _id_match("offers.ean.raw", q, "or", "1", 30),
    ]


def _offer_clauses_tp18(q: str) -> list[dict]:
    """TP18-lexical offer sub-query — STANDARD's (b)–(g) only (no clause a, no
    cross-fields multi_match); §2.2.1."""
    return [
        _id_match("offers.articleNumber.raw", q, "or", "1", 50),
        _id_match("offers.articleNumber.normalized", q, "or", "1", 50),
        _id_match("offers.articleNumber.seg", q, "and", "100%", 30),
        _id_match("offers.manufacturerArticleNumber.raw", q, "or", "1", 30),
        _id_match("offers.manufacturerArticleNumber.normalized", q, "or", "1", 30),
        _id_match("offers.ean.raw", q, "or", "1", 30),
    ]


def _id_match(field: str, q: str, operator: str, msm: str, boost: int) -> dict:
    return {
        "match": {
            field: {
                "query": q,
                "operator": operator,
                "minimum_should_match": msm,
                "fuzziness": "0",
                "fuzzy_transpositions": False,
                "auto_generate_synonyms_phrase_query": False,
                "boost": boost,
            }
        }
    }


def _syn_kw_clause(q: str) -> dict:
    return {
        "match": {
            "syntheticKeywords.de2": {
                "query": q,
                "analyzer": SYN_KW_ANALYZER,
                "operator": "and",
                "boost": SYN_KW_BOOST,
            }
        }
    }


def body_standard(q: str, pre: list[dict], post: list[dict], size: int) -> dict:
    """STANDARD body (§1.2). Customer-artno sub-query dropped."""
    offer_nested = {
        "nested": {
            "path": "offers",
            "score_mode": "max",
            "query": {"bool": {"should": _offer_clauses_std(q)}},
        }
    }
    body: dict = {
        "query": {
            "bool": {
                "should": [offer_nested, _syn_kw_clause(q)],
                "minimum_should_match": 1,
                **({"filter": pre} if pre else {}),
            }
        },
        "_source": ["articleId"],
        "size": size,
        "track_total_hits": False,
    }
    if post:
        body["post_filter"] = {"bool": {"must": post}}
    return body


def body_tp18_lex(q: str, filters: list[dict], size: int, _source: bool) -> dict:
    """TP18-lexical body (§2.2.1)."""
    offer_nested = {
        "nested": {
            "path": "offers",
            "score_mode": "max",
            "query": {"bool": {"should": _offer_clauses_tp18(q)}},
        }
    }
    body: dict = {
        "query": {
            "bool": {
                "must": [offer_nested],
                **({"filter": filters} if filters else {}),
            }
        },
        "size": size,
        "track_total_hits": False,
    }
    body["_source"] = ["articleId"] if _source else False
    return body


def body_tp18_vec(
    vec: np.ndarray,
    filters: list[dict],
    k: int,
    num_candidates: int,
    size: int,
    _source: bool,
) -> dict:
    """TP18-vector body (§2.2.2) — legacy top-level knn block."""
    knn: dict = {
        "field": "embeddings.vector",
        "query_vector": vec.tolist(),
        "k": k,
        "num_candidates": num_candidates,
    }
    if filters:
        knn["filter"] = filters
    body: dict = {
        "knn": knn,
        "size": size,
        "track_total_hits": False,
    }
    body["_source"] = ["articleId"] if _source else False
    return body


def body_phase2(page_ids: list[str], filters: list[dict], size: int) -> dict:
    """Hybrid phase-2 page fetch (§2.2.5) — bool filter pinned to fused ids."""
    f: list[dict] = [{"ids": {"values": page_ids}}]
    f.extend(filters)
    return {
        "query": {"bool": {"filter": f}},
        "_source": ["articleId"],
        "size": size,
        "track_total_hits": False,
    }


# ============================================================
# Per-profile timers
# ============================================================


def _ids_from_hits(resp_json: dict) -> list[str]:
    hits = resp_json.get("hits", {}).get("hits", [])
    return [h.get("_id") for h in hits if h.get("_id")]


def time_standard(
    client: httpx.Client,
    index: str,
    query_text: str,
    pre: list[dict],
    post: list[dict],
    size: int,
) -> tuple[int, float]:
    body = body_standard(query_text, pre, post, size)
    t0 = time.perf_counter()
    r = client.post(
        f"/{index}/_search",
        json=body,
        params={"search_type": "dfs_query_then_fetch"},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return len(r.json().get("hits", {}).get("hits", [])), elapsed_ms


def time_tp18_lex(
    client: httpx.Client,
    index: str,
    query_text: str,
    filters: list[dict],
    size: int,
) -> tuple[int, float]:
    body = body_tp18_lex(query_text, filters, size, _source=True)
    t0 = time.perf_counter()
    r = client.post(f"/{index}/_search", json=body)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return len(r.json().get("hits", {}).get("hits", [])), elapsed_ms


def time_tp18_vec(
    client: httpx.Client,
    index: str,
    query_vec: np.ndarray,
    filters: list[dict],
    k: int,
    num_candidates: int,
    size: int,
) -> tuple[int, float]:
    body = body_tp18_vec(
        query_vec, filters, k=k, num_candidates=num_candidates, size=size, _source=True
    )
    t0 = time.perf_counter()
    r = client.post(f"/{index}/_search", json=body)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return len(r.json().get("hits", {}).get("hits", [])), elapsed_ms


def time_tp18_hybrid(
    client: httpx.Client,
    leg_pool: ThreadPoolExecutor,
    index: str,
    query_text: str,
    query_vec: np.ndarray,
    filters: list[dict],
    leg_depth: int,
    page_size: int,
) -> tuple[int, float, float, float, float]:
    """Hybrid timing — parallel legs + phase-2 page fetch (§2.2.5).

    `leg_depth` controls both legs' candidate pool size: kNN leg uses
    k=num_candidates=size=leg_depth (ES requires num_candidates >= k); lex leg
    uses size=leg_depth. Doc-spec'd shipping config is leg_depth=5000.

    Returns (n_hits_phase2, total_ms, t_lex_ms, t_knn_ms, t_phase2_ms).
    """
    lex_body = body_tp18_lex(query_text, filters, size=leg_depth, _source=False)
    knn_body = body_tp18_vec(
        query_vec,
        filters,
        k=leg_depth,
        num_candidates=leg_depth,
        size=leg_depth,
        _source=False,
    )

    def _post_leg(body: dict) -> tuple[list[str], float]:
        t = time.perf_counter()
        r = client.post(f"/{index}/_search", json=body)
        ms = (time.perf_counter() - t) * 1000
        r.raise_for_status()
        return _ids_from_hits(r.json()), ms

    t_total0 = time.perf_counter()
    f_lex = leg_pool.submit(_post_leg, lex_body)
    f_knn = leg_pool.submit(_post_leg, knn_body)
    lex_ids, t_lex_ms = f_lex.result()
    knn_ids, t_knn_ms = f_knn.result()
    fused_ids = rrf_fuse(lex_ids, knn_ids)
    page_ids = fused_ids[:page_size]
    if not page_ids:
        return 0, (time.perf_counter() - t_total0) * 1000, t_lex_ms, t_knn_ms, 0.0
    t_p2 = time.perf_counter()
    r = client.post(
        f"/{index}/_search", json=body_phase2(page_ids, filters, page_size)
    )
    t_phase2_ms = (time.perf_counter() - t_p2) * 1000
    r.raise_for_status()
    n_hits = len(r.json().get("hits", {}).get("hits", []))
    total_ms = (time.perf_counter() - t_total0) * 1000
    return n_hits, total_ms, t_lex_ms, t_knn_ms, t_phase2_ms


def rrf_fuse(lex_ids: list[str], knn_ids: list[str], k: int = 60) -> list[str]:
    """App-side RRF fusion — score(id) = Σ 1/(k + rank_1based)."""
    scores: dict[str, float] = {}
    for rank, _id in enumerate(lex_ids, start=1):
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank)
    for rank, _id in enumerate(knn_ids, start=1):
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda _id: (-scores[_id], _id))


# ============================================================
# Cell runner
# ============================================================


def run_cell(
    es_url: str,
    index: str,
    profile: str,
    regime: str,
    qtexts: list[str],
    qvecs: np.ndarray,
    spec: dict | None,
    concurrency: int,
    k: int,
    num_candidates: int | None,
    cell_label: str,
) -> list[float]:
    """Run all queries for one cell; return per-query latencies (ms)."""
    n_q = len(qtexts)
    lat: list[float] = [0.0] * n_q
    limits = httpx.Limits(
        max_connections=concurrency * 4, max_keepalive_connections=concurrency * 4
    )
    t0 = time.time()
    done = 0
    next_print = 0

    pre, post = build_standard_filters(spec)
    flat = build_tp18_filter(spec)

    with httpx.Client(base_url=es_url, timeout=180.0, limits=limits) as client:
        # Hybrid needs a leg-pool to fire two POSTs concurrently per query.
        leg_pool = ThreadPoolExecutor(max_workers=concurrency * 2) if profile == "hybrid" else None

        def _run(idx: int) -> float:
            qt = qtexts[idx]
            qv = qvecs[idx]
            if profile == "standard":
                _, ms = time_standard(client, index, qt, pre, post, k)
            elif profile == "tp18-lex":
                _, ms = time_tp18_lex(client, index, qt, flat, k)
            elif profile == "tp18-vec":
                assert num_candidates is not None
                _, ms = time_tp18_vec(client, index, qv, flat, k, num_candidates, k)
            elif profile == "hybrid":
                assert num_candidates is not None
                _, ms, *_ = time_tp18_hybrid(
                    client, leg_pool, index, qt, qv, flat, num_candidates, k
                )
            else:
                raise ValueError(f"unknown profile: {profile}")
            return ms

        try:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = {ex.submit(_run, i): i for i in range(n_q)}
                for fut in as_completed(futures):
                    i = futures[fut]
                    lat[i] = fut.result()
                    done += 1
                    if done >= next_print or done == n_q:
                        msg = f"    {cell_label}  {fmt_eta(done, n_q, t0)}"
                        sys.stdout.write("\r" + msg.ljust(120))
                        sys.stdout.flush()
                        next_print = done + BENCH_PROGRESS_EVERY
        finally:
            if leg_pool is not None:
                leg_pool.shutdown(wait=True)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return lat


# ============================================================
# Main
# ============================================================


def percentile(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-dir", type=Path, default=DEFAULT_QUERIES_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--es-url", default=ES_URL)
    parser.add_argument("--es-index", default=ES_INDEX)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Page size (top-K).")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--queries-limit",
        type=int,
        default=None,
        help="Cap queries per cell for smoke tests.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["standard", "tp18-lex", "tp18-vec", "hybrid"],
        choices=["standard", "tp18-lex", "tp18-vec", "hybrid"],
    )
    parser.add_argument(
        "--regimes", nargs="+", default=BENCH_REGIMES,
        choices=BENCH_REGIMES,
    )
    parser.add_argument(
        "--numc", type=int, nargs="+", default=NUMC_VARIANTS,
        help="num_candidates values for TP18-vector and TP18-hybrid kNN leg.",
    )
    parser.add_argument("--warmup-passes", type=int, default=2)
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (defaults to <report-dir>/bench_profiles_latency.json).",
    )
    parser.add_argument(
        "--log-first-bodies",
        action="store_true",
        help="Print the first request body per profile for eyeball validation.",
    )
    args = parser.parse_args()

    overall_t0 = time.time()
    out_path = args.out or (args.report_dir / "bench_profiles_latency.json")

    # ---------- load queries ----------
    print("[0/3] Loading queries ...")
    qtbl = pq.read_table(args.queries_dir / "queries.parquet")
    qtexts = qtbl["query"].to_pylist()
    qvecs = np.load(args.queries_dir / "query_vectors.npy").astype(np.float32)
    assert len(qtexts) == qvecs.shape[0], (
        f"row mismatch: queries.parquet={len(qtexts)}, query_vectors.npy={qvecs.shape[0]}"
    )
    if args.queries_limit:
        qtexts = qtexts[: args.queries_limit]
        qvecs = qvecs[: args.queries_limit]
    n_q = len(qtexts)
    print(f"      {n_q} queries loaded (text + {qvecs.shape[1]}-dim vectors)")

    # ---------- regime → filter spec ----------
    for r in args.regimes:
        if r not in REGIMES:
            raise SystemExit(f"unknown regime: {r}")
    regime_specs = {r: REGIMES[r]["filter_spec"] for r in args.regimes}

    # ---------- enumerate cells ----------
    cells: list[dict] = []
    for regime in args.regimes:
        spec = regime_specs[regime]
        for profile in args.profiles:
            if profile in ("tp18-vec", "hybrid"):
                for numc in args.numc:
                    cells.append(
                        {"profile": profile, "regime": regime, "numc": numc, "spec": spec}
                    )
            else:
                cells.append(
                    {"profile": profile, "regime": regime, "numc": None, "spec": spec}
                )
    print(f"\n[1/3] {len(cells)} cells × {n_q} queries (concurrency={args.concurrency})")

    # ---------- optional body logging ----------
    if args.log_first_bodies:
        print("\n--- first request body per profile (query=qtexts[0], spec=acl-top) ---")
        spec = regime_specs.get("acl-top") or {"acl": TOP_CATALOG}
        pre, post = build_standard_filters(spec)
        flat = build_tp18_filter(spec)
        print("[STANDARD]")
        print(json.dumps(body_standard(qtexts[0], pre, post, args.k), indent=2)[:1500])
        print("\n[TP18-LEX]")
        print(json.dumps(body_tp18_lex(qtexts[0], flat, args.k, True), indent=2)[:1500])
        print("\n[TP18-VEC numC=1000]")
        body = body_tp18_vec(qvecs[0], flat, args.k, 1000, args.k, True)
        # truncate the giant query_vector for readability
        body["knn"]["query_vector"] = (
            body["knn"]["query_vector"][:3] + ["...(128 dims)"]
        )
        print(json.dumps(body, indent=2)[:1500])
        print("\n[PHASE-2 (hybrid page fetch)]")
        print(json.dumps(body_phase2(["dummy-id"], flat, args.k), indent=2)[:1500])
        print("\n--- end body dump ---\n")

    # ---------- warmup ----------
    warm_qs = (
        [int(i * n_q / args.warmup_queries) for i in range(args.warmup_queries)]
        if args.warmup_queries > 0
        else []
    )
    warm_passes = args.warmup_passes
    n_warm = len(cells) * len(warm_qs) * warm_passes
    print(
        f"\n[2/3] Warmup ({len(cells)} cells × {len(warm_qs)} queries × "
        f"{warm_passes} passes = {n_warm} ops) ..."
    )
    t0 = time.time()
    if n_warm > 0:
        limits = httpx.Limits(max_connections=4, max_keepalive_connections=4)
        with httpx.Client(base_url=args.es_url, timeout=300.0, limits=limits) as wclient:
            leg_pool = ThreadPoolExecutor(max_workers=4)
            try:
                done = 0
                for _ in range(warm_passes):
                    for cell in cells:
                        for qi in warm_qs:
                            qt, qv = qtexts[qi], qvecs[qi]
                            try:
                                _exec_one(
                                    wclient,
                                    leg_pool,
                                    args.es_index,
                                    cell,
                                    qt,
                                    qv,
                                    args.k,
                                )
                            except httpx.HTTPStatusError as e:
                                print(f"\nWARMUP FAILED: {cell['profile']}/{cell['regime']}: {e}")
                                print(f"body: {e.response.text[:500]}")
                                raise
                            done += 1
                            msg = f"      warmup  {fmt_eta(done, n_warm, t0)}"
                            sys.stdout.write("\r" + msg.ljust(110))
                            sys.stdout.flush()
            finally:
                leg_pool.shutdown(wait=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    # ---------- bench ----------
    print(f"\n[3/3] Benching {len(cells)} cells ...")
    hdr = (
        f"  {'profile':>12}  {'regime':>22}  {'numC':>5}  "
        f"{'p50_ms':>8}  {'p95_ms':>8}  {'p99_ms':>8}  {'mean_ms':>8}  {'n':>5}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    results: list[dict] = []
    phase_t0 = time.time()
    for ci, cell in enumerate(cells, start=1):
        cell_label = (
            f"cell {ci}/{len(cells)}: {cell['profile']} {cell['regime']}"
            + (f" numC={cell['numc']}" if cell["numc"] is not None else "")
        )
        lats = run_cell(
            es_url=args.es_url,
            index=args.es_index,
            profile=cell["profile"],
            regime=cell["regime"],
            qtexts=qtexts,
            qvecs=qvecs,
            spec=cell["spec"],
            concurrency=args.concurrency,
            k=args.k,
            num_candidates=cell["numc"],
            cell_label=cell_label,
        )
        p50 = percentile(lats, 50)
        p95 = percentile(lats, 95)
        p99 = percentile(lats, 99)
        mean = statistics.fmean(lats) if lats else 0.0
        print(
            f"  {cell['profile']:>12}  {cell['regime']:>22}  "
            f"{str(cell['numc'] or '-'):>5}  "
            f"{p50:>8.2f}  {p95:>8.2f}  {p99:>8.2f}  {mean:>8.2f}  {len(lats):>5}"
        )
        results.append(
            {
                "profile": cell["profile"],
                "regime": cell["regime"],
                "numc": cell["numc"],
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "p99_ms": round(p99, 3),
                "mean_ms": round(mean, 3),
                "n_queries": len(lats),
            }
        )
        elapsed = time.time() - phase_t0
        rate = ci / elapsed if elapsed > 0 else 0
        eta = (len(cells) - ci) / rate if rate > 0 else 0
        print(
            f"  ── progress: {ci}/{len(cells)} cells  "
            f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
        )

    out_payload = {
        "es_url": args.es_url,
        "es_index": args.es_index,
        "n_queries": n_q,
        "concurrency": args.concurrency,
        "k": args.k,
        "regimes": args.regimes,
        "numc_variants": args.numc,
        "profiles": args.profiles,
        "wall_seconds": round(time.time() - overall_t0, 1),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))
    print(f"\nDone in {time.time() - overall_t0:.1f}s. Wrote {out_path}")


def _exec_one(
    client: httpx.Client,
    leg_pool: ThreadPoolExecutor,
    index: str,
    cell: dict,
    qt: str,
    qv: np.ndarray,
    k: int,
) -> None:
    """Single-shot warmup execution dispatching by profile."""
    spec = cell["spec"]
    pre, post = build_standard_filters(spec)
    flat = build_tp18_filter(spec)
    profile = cell["profile"]
    if profile == "standard":
        time_standard(client, index, qt, pre, post, k)
    elif profile == "tp18-lex":
        time_tp18_lex(client, index, qt, flat, k)
    elif profile == "tp18-vec":
        time_tp18_vec(client, index, qv, flat, k, cell["numc"], k)
    elif profile == "hybrid":
        time_tp18_hybrid(client, leg_pool, index, qt, qv, flat, cell["numc"], k)


if __name__ == "__main__":
    main()
