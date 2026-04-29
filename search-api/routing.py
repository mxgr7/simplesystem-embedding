"""F9 routing + F4 sort/paging — dispatch a SearchRequest across the
two-collection topology with sort, hitCount and relevance-pool bounds.

Two paths, picked deterministically by F9 §"Routing rule":

  Path A (vector-first) — no per-offer filter applies:
    1. ANN + BM25 on `articles_v{N}` constrained to `article_expr`.
    2. RRF fuse over `article_hash`.
    3. Resolve offers: query `offers_v{N}` filtered by `article_hash IN [...]`.
    4. Pick representative offer per hash (sort-aware); price post-pass;
       sort + paginate.

  Path B (filter-first) — at least one per-offer filter applies:
    1. Bounded probe: query `offers_v{N}` with `offer_expr`, limit
       `PATH_B_HASH_LIMIT + 1`.
    2. If distinct hashes > limit → fall back to Path A with
       `recall_clipped=True` (under-recalls selective-but-not-tight
       filters; documented in spec §2.4).
    3. Otherwise: ANN + BM25 on `articles_v{N}` with
       `article_hash IN [probe-hashes] AND article_expr`.
    4. Re-attach offers from the probe; sort-aware representative
       selection; price post-pass; sort + paginate.

F4 sort + hitCount layers on top:

  * Sort: relevance (default), articleId, name, price (asc|desc).
    Multi-key requests apply only the first key.
  * Relevance-pool bounding: when sort is non-relevance AND a query
    string is present, the ranked candidate pool is capped at
    `RELEVANCE_POOL_MAX` and floor-pruned at `RELEVANCE_SCORE_FLOOR ×
    top_score`. Browse (no query) skips both bounds.
  * hitCount: total filtered article count via a separate `query()`
    pass capped at `HITCOUNT_CAP`. Clipped flag set if the cap fires
    or Path B's probe overflowed.

Path A and Path B are *not* recall-equivalent under selective offer
filters — choosing the wrong path silently under-recalls. The rule is
therefore deterministic, not a cost heuristic.

`articleIdsFilter` (legacy offer PK) stays on the offer side — preserves
the spec semantic that the request asks for *specific offers*. No
hash-resolution round-trip.

`blocked_eclass_vendors_filters` per-vendor entries (correlate offer
vendor with article eclass) are split-incompatible at the expr level;
this module applies them as a Python post-pass over the candidate set.
Global entries (no vendor restriction) push down via `build_article_expr`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

import numpy as np
from pymilvus import MilvusClient

import aggregations
from filters import (
    build_article_expr,
    build_offer_expr,
    has_per_vendor_blocked_eclass,
)
from hybrid import Hit, is_strict_identifier, rrf_merge
from models import (
    EClassVersion,
    SearchMode,
    SearchRequest,
    SortDirection,
    Summaries,
)
from prices import resolve_price
from sorting import (
    SortField,
    SortPlan,
    _Materialised,
    bound_relevance_pool,
    pick_representative,
    sort_items,
)


_DEFAULT_DIRECTION = SortDirection.DESC

EmbedFn = Callable[[str], Awaitable[list[float]]]

# Milvus 2.6's `proxy.maxResultWindow` quota caps `(offset + limit)` on
# `query()` at 16384 by default. Path B's bounded probe needs `limit =
# PATH_B_HASH_LIMIT + 1` (the +1 detects overflow), so the practical
# ceiling on `PATH_B_HASH_LIMIT` is 16383. F9 doc's 25k figure was an
# analytical p95-latency target — the Milvus quota dominates. Operators
# can raise the quota via `proxy.maxResultWindow` + Milvus restart;
# until then, Path A fallback handles requests that overflow.
DEFAULT_PATH_B_HASH_LIMIT = 16_383
_MILVUS_MAX_QUERY_WINDOW = 16_384

DEFAULT_DENSE_POOL = 200
DEFAULT_BM25_POOL = 200
DEFAULT_RRF_K = 60

# F4 — relevance-pool bounds and hitCount cap.
DEFAULT_RELEVANCE_POOL_MAX = 200
DEFAULT_RELEVANCE_SCORE_FLOOR = 0.20
DEFAULT_HITCOUNT_CAP = 10_000


# ──────────────────────────────────────────────────────────────────────
# Result + dispatch entry
# ──────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class DispatchResult:
    hits: list[Hit]
    debug: dict
    recall_clipped: bool = False
    hit_count: int = 0
    hit_count_clipped: bool = False
    summaries: Summaries | None = None


@dataclass(slots=True)
class _Timings:
    path: str = ""
    article_expr: str | None = None
    offer_expr: str | None = None
    probe_hits: int | None = None
    probe_overflowed: bool = False
    distinct_hashes: int | None = None
    article_rank_ms: float | None = None
    offer_resolve_ms: float | None = None
    embed_ms: float | None = None
    dense_ms: float | None = None
    bm25_ms: float | None = None
    sort_field: str | None = None
    relevance_bound_dropped: int | None = None
    hit_count_query_ms: float | None = None


async def dispatch_dedup(
    req: SearchRequest,
    *,
    page: int = 1,
    page_size: int,
    overfetch_n: int,
    sort_plan: SortPlan | None = None,
    client: MilvusClient,
    embed: EmbedFn,
    articles_collection: str,
    offers_collection: str,
    path_b_hash_limit: int = DEFAULT_PATH_B_HASH_LIMIT,
    dense_pool: int = DEFAULT_DENSE_POOL,
    bm25_pool: int = DEFAULT_BM25_POOL,
    rrf_k: int = DEFAULT_RRF_K,
    num_candidates: int | None = None,
    relevance_pool_max: int = DEFAULT_RELEVANCE_POOL_MAX,
    relevance_score_floor: float = DEFAULT_RELEVANCE_SCORE_FLOOR,
    hitcount_cap: int = DEFAULT_HITCOUNT_CAP,
) -> DispatchResult:
    """F9 + F4 dispatcher. Returns a `DispatchResult` carrying the page
    of hits (one per article, representative offer per sort rule), the
    full filtered article hit count (capped + clipped flag), and a
    debug envelope."""
    if sort_plan is None:
        sort_plan = SortPlan(SortField.RELEVANCE, _DEFAULT_DIRECTION)
    article_expr = build_article_expr(req)
    offer_expr = build_offer_expr(req)
    timings = _Timings(
        article_expr=article_expr, offer_expr=offer_expr,
        sort_field=sort_plan.field.value,
    )
    query_text = (req.query or "").strip()

    # F5: SUMMARIES_ONLY skips the rank/materialise/sort/page work entirely
    # and just fetches summary data + count. Returns empty hits.
    if req.search_mode is SearchMode.SUMMARIES_ONLY:
        return await _dispatch_summaries_only(
            req,
            article_expr=article_expr, offer_expr=offer_expr,
            client=client,
            articles_collection=articles_collection,
            offers_collection=offers_collection,
            path_b_hash_limit=path_b_hash_limit,
            hitcount_cap=hitcount_cap,
            timings=timings,
        )

    price_active = _price_active(req)
    rank_limit = _rank_limit(
        sort_plan, query_text, page=page, page_size=page_size,
        overfetch_n=overfetch_n, price_active=price_active,
        relevance_pool_max=relevance_pool_max,
        hitcount_cap=hitcount_cap,
        dense_pool=dense_pool,
    )

    # Captured here so the F5 summary fetch can constrain by the
    # probe-distinct hashes when in Path B.
    path_b_distinct_hashes: list[str] | None = None

    # Path selection (F9 deterministic rule).
    if offer_expr is None:
        timings.path = "A"
        materialised, hit_count, hit_count_clipped = await _path_a(
            req, query_text, article_expr,
            sort_plan=sort_plan,
            rank_limit=rank_limit,
            relevance_pool_max=relevance_pool_max,
            relevance_score_floor=relevance_score_floor,
            hitcount_cap=hitcount_cap,
            price_active=price_active,
            client=client, embed=embed,
            articles_collection=articles_collection,
            offers_collection=offers_collection,
            dense_pool=dense_pool, bm25_pool=bm25_pool,
            rrf_k=rrf_k, num_candidates=num_candidates,
            timings=timings,
        )
        recall_clipped = False
    else:
        # Path B — bounded probe on offers.
        t0 = time.perf_counter()
        probe_rows = await asyncio.to_thread(
            _offer_probe, client, offers_collection,
            offer_expr=offer_expr, limit=path_b_hash_limit + 1,
        )
        timings.offer_resolve_ms = (time.perf_counter() - t0) * 1000
        timings.probe_hits = len(probe_rows)

        distinct_hashes = sorted({r["article_hash"] for r in probe_rows})
        timings.distinct_hashes = len(distinct_hashes)

        if len(distinct_hashes) > path_b_hash_limit:
            # Probe overflow → Path A fallback with clipped hit count
            # (the true count is unknown beyond the probe limit).
            timings.probe_overflowed = True
            timings.path = "A_fallback"
            materialised, _hit_count_unused, _clipped_unused = await _path_a(
                req, query_text, article_expr,
                sort_plan=sort_plan,
                rank_limit=rank_limit,
                relevance_pool_max=relevance_pool_max,
                relevance_score_floor=relevance_score_floor,
                hitcount_cap=hitcount_cap,
                price_active=price_active,
                client=client, embed=embed,
                articles_collection=articles_collection,
                offers_collection=offers_collection,
                dense_pool=dense_pool, bm25_pool=bm25_pool,
                rrf_k=rrf_k, num_candidates=num_candidates,
                timings=timings,
                skip_count=True,
            )
            # hit_count for the fallback request is not the article-side
            # count — the user filtered by per-offer expr that we
            # couldn't fully enforce. Report the probe-limit lower bound
            # with `hit_count_clipped=True`.
            hit_count = path_b_hash_limit
            hit_count_clipped = True
            recall_clipped = True
        else:
            timings.path = "B"
            path_b_distinct_hashes = distinct_hashes
            materialised, hit_count, hit_count_clipped = await _path_b(
                req, query_text, article_expr, distinct_hashes, probe_rows,
                sort_plan=sort_plan,
                rank_limit=rank_limit,
                relevance_pool_max=relevance_pool_max,
                relevance_score_floor=relevance_score_floor,
                hitcount_cap=hitcount_cap,
                price_active=price_active,
                client=client, embed=embed,
                articles_collection=articles_collection,
                dense_pool=dense_pool, bm25_pool=bm25_pool,
                rrf_k=rrf_k, num_candidates=num_candidates,
                timings=timings,
            )
            recall_clipped = False

    # Final sort, then page slice. _Materialised already carries the
    # representative offer + sort-key data; sort_items applies the
    # primary sort with the universal articleId-asc tiebreak.
    sorted_items = sort_items(materialised, sort_plan)
    page_offset = max(0, (page - 1) * page_size)
    page_slice = sorted_items[page_offset:page_offset + page_size]
    hits = _to_hits(page_slice, sort_plan)

    # F5: BOTH mode runs aggregations over the full filtered hit set
    # (independent of the page slice). HITS_ONLY skips. SUMMARIES_ONLY
    # returned at the top.
    summaries: Summaries | None = None
    if req.search_mode is SearchMode.BOTH and req.summaries:
        summaries = await _compute_summaries(
            req,
            article_expr=article_expr, offer_expr=offer_expr,
            path_b_distinct_hashes=path_b_distinct_hashes,
            client=client,
            articles_collection=articles_collection,
            offers_collection=offers_collection,
            hitcount_cap=hitcount_cap,
        )

    return DispatchResult(
        hits=hits,
        debug=_debug(timings),
        recall_clipped=recall_clipped,
        hit_count=hit_count,
        hit_count_clipped=hit_count_clipped,
        summaries=summaries,
    )


def _rank_limit(
    sort_plan: SortPlan,
    query_text: str,
    *,
    page: int,
    page_size: int,
    overfetch_n: int,
    price_active: bool,
    relevance_pool_max: int,
    hitcount_cap: int,
    dense_pool: int,
) -> int:
    """How many candidate articles the rank step should fetch.

      relevance + query: max(page * page_size * overfetch, dense_pool)
        — over-fetch covers both paging and price post-pass drops.
      relevance + browse: hitcount_cap — ordering is undefined, so
        return as many as the hitcount cap allows so the page slice
        can land anywhere in the filtered set.
      non-relevance + query: relevance_pool_max — F4 bounds the pool.
      non-relevance + browse: hitcount_cap — full filtered set sortable.
    """
    page_window = max(page * page_size, 1) * (overfetch_n if price_active else 1)
    if sort_plan.is_relevance:
        if query_text:
            return max(page_window, dense_pool)
        return hitcount_cap
    if query_text:
        return relevance_pool_max
    return hitcount_cap


# ──────────────────────────────────────────────────────────────────────
# Path A
# ──────────────────────────────────────────────────────────────────────

async def _path_a(
    req: SearchRequest,
    query_text: str,
    article_expr: str | None,
    *,
    sort_plan: SortPlan,
    rank_limit: int,
    relevance_pool_max: int,
    relevance_score_floor: float,
    hitcount_cap: int,
    price_active: bool,
    client: MilvusClient,
    embed: EmbedFn,
    articles_collection: str,
    offers_collection: str,
    dense_pool: int,
    bm25_pool: int,
    rrf_k: int,
    num_candidates: int | None,
    timings: _Timings,
    skip_count: bool = False,
) -> tuple[list[_Materialised], int, bool]:
    """Vector-first path. Article expression pushed down to ANN/BM25;
    offer resolve attaches per-hash offers (no offer-side filter)."""
    if query_text:
        ranked_hashes = await _rank_articles(
            query_text, article_expr, client=client, embed=embed,
            articles_collection=articles_collection,
            limit=rank_limit, dense_pool=dense_pool, bm25_pool=bm25_pool,
            rrf_k=rrf_k, num_candidates=num_candidates, timings=timings,
        )
    elif article_expr is not None:
        # Browse-only mode: any article matching article_expr.
        t0 = time.perf_counter()
        rows = await asyncio.to_thread(
            _article_browse, client, articles_collection,
            article_expr=article_expr, limit=rank_limit,
        )
        timings.article_rank_ms = (time.perf_counter() - t0) * 1000
        ranked_hashes = [(r["article_hash"], 0.0) for r in rows]
    else:
        # No query, no article filter, no offer filter — no defensible
        # default browse. Mirrors the legacy `_filter_only_browse`
        # behaviour of returning nothing.
        return [], 0, False

    # Relevance-pool bound: only when sort is non-relevance AND query
    # is present. Browse traffic and relevance-default sort skip both
    # bounds (per F4 §"Relevance-pool bounding").
    if not sort_plan.is_relevance and query_text:
        before = len(ranked_hashes)
        ranked_hashes = bound_relevance_pool(
            ranked_hashes,
            pool_max=relevance_pool_max, score_floor=relevance_score_floor,
        )
        timings.relevance_bound_dropped = before - len(ranked_hashes)

    if not ranked_hashes:
        if skip_count:
            return [], 0, False
        hit_count, clipped = await _count_articles(
            client, articles_collection,
            article_expr=article_expr, hashes=None, cap=hitcount_cap,
            timings=timings,
        )
        return [], hit_count, clipped

    hashes = [h for h, _ in ranked_hashes]

    # Concurrent: offer resolve, article meta (name/eclass), hit count.
    offers_task = asyncio.to_thread(
        _resolve_offers, client, offers_collection,
        hashes=hashes, offer_expr=None,
        need_eclass_post_pass=has_per_vendor_blocked_eclass(req),
    )
    meta_task = _fetch_article_meta(
        client, articles_collection, hashes, req, sort_plan=sort_plan,
    )

    t0 = time.perf_counter()
    if skip_count:
        offers_by_hash, article_meta = await asyncio.gather(offers_task, meta_task)
        hit_count, clipped = 0, False
    else:
        count_task = _count_articles(
            client, articles_collection,
            article_expr=article_expr, hashes=None, cap=hitcount_cap,
            timings=timings,
        )
        offers_by_hash, article_meta, (hit_count, clipped) = await asyncio.gather(
            offers_task, meta_task, count_task,
        )
    timings.offer_resolve_ms = (timings.offer_resolve_ms or 0.0) + (time.perf_counter() - t0) * 1000

    materialised = _materialise(
        ranked_hashes, offers_by_hash, article_meta, req,
        sort_plan=sort_plan, price_active=price_active,
    )
    return materialised, hit_count, clipped


# ──────────────────────────────────────────────────────────────────────
# Path B
# ──────────────────────────────────────────────────────────────────────

async def _path_b(
    req: SearchRequest,
    query_text: str,
    article_expr: str | None,
    distinct_hashes: list[str],
    probe_rows: list[dict],
    *,
    sort_plan: SortPlan,
    rank_limit: int,
    relevance_pool_max: int,
    relevance_score_floor: float,
    hitcount_cap: int,
    price_active: bool,
    client: MilvusClient,
    embed: EmbedFn,
    articles_collection: str,
    dense_pool: int,
    bm25_pool: int,
    rrf_k: int,
    num_candidates: int | None,
    timings: _Timings,
) -> tuple[list[_Materialised], int, bool]:
    """Filter-first path. Probe already enforced offer_expr; rank the
    matching hashes (or browse if no query); re-attach probe offers."""
    offers_by_hash: dict[str, list[dict]] = {}
    for r in probe_rows:
        offers_by_hash.setdefault(r["article_hash"], []).append(r)

    if not distinct_hashes:
        return [], 0, False

    constrained_hash_expr = _hash_in_expr(distinct_hashes)

    if query_text:
        constrained_expr = _and_exprs(constrained_hash_expr, article_expr)
        ranked_hashes = await _rank_articles(
            query_text, constrained_expr, client=client, embed=embed,
            articles_collection=articles_collection,
            limit=rank_limit, dense_pool=dense_pool, bm25_pool=bm25_pool,
            rrf_k=rrf_k, num_candidates=num_candidates, timings=timings,
        )
    elif article_expr is not None:
        constrained_expr = _and_exprs(constrained_hash_expr, article_expr)
        t0 = time.perf_counter()
        rows = await asyncio.to_thread(
            _article_browse, client, articles_collection,
            article_expr=constrained_expr, limit=rank_limit,
        )
        timings.article_rank_ms = (time.perf_counter() - t0) * 1000
        ranked_hashes = [(r["article_hash"], 0.0) for r in rows]
    else:
        # No query, no article_expr — probe order is the only ordering.
        ranked_hashes = [(h, 0.0) for h in distinct_hashes]

    if not sort_plan.is_relevance and query_text:
        before = len(ranked_hashes)
        ranked_hashes = bound_relevance_pool(
            ranked_hashes,
            pool_max=relevance_pool_max, score_floor=relevance_score_floor,
        )
        timings.relevance_bound_dropped = before - len(ranked_hashes)

    # hitCount: count of articles in `hash IN distinct_hashes AND
    # article_expr`. When article_expr is None, every distinct probe
    # hash is a hit by definition.
    if article_expr is None:
        hit_count = len(distinct_hashes)
        clipped = False
    else:
        hit_count, clipped = await _count_articles(
            client, articles_collection,
            article_expr=article_expr, hashes=distinct_hashes,
            cap=hitcount_cap, timings=timings,
        )

    if not ranked_hashes:
        return [], hit_count, clipped

    hashes = [h for h, _ in ranked_hashes]
    article_meta = await _fetch_article_meta(
        client, articles_collection, hashes, req, sort_plan=sort_plan,
    )

    materialised = _materialise(
        ranked_hashes, offers_by_hash, article_meta, req,
        sort_plan=sort_plan, price_active=price_active,
    )
    return materialised, hit_count, clipped


# ──────────────────────────────────────────────────────────────────────
# Article ranking (dense + BM25 + RRF)
# ──────────────────────────────────────────────────────────────────────

async def _rank_articles(
    query_text: str,
    article_expr: str | None,
    *,
    client: MilvusClient,
    embed: EmbedFn,
    articles_collection: str,
    limit: int,
    dense_pool: int,
    bm25_pool: int,
    rrf_k: int,
    num_candidates: int | None,
    timings: _Timings,
) -> list[tuple[str, float]]:
    """Hybrid ANN + BM25 over `articles_v{N}`, RRF-fused on
    `article_hash`. Mirrors the classifier in `hybrid.py` for routing
    multi-word vs single-token strict-identifier queries."""

    # Multi-word → vector-only (BM25 over identifier corpus contributes
    # noise on phrase queries — same logic as legacy hybrid_classified).
    multi_word = len(query_text.split()) > 1
    strict_id = not multi_word and is_strict_identifier(query_text)

    async def do_dense() -> list[tuple[str, float]]:
        t0 = time.perf_counter()
        vec = await embed(query_text)
        timings.embed_ms = (time.perf_counter() - t0) * 1000
        if not vec:
            return []
        t1 = time.perf_counter()
        out = await asyncio.to_thread(
            _dense_search_articles, client, articles_collection,
            vec=vec, limit=max(limit, dense_pool),
            num_candidates=num_candidates, filter_expr=article_expr,
        )
        timings.dense_ms = (time.perf_counter() - t1) * 1000
        return out

    async def do_bm25() -> list[tuple[str, float]]:
        t0 = time.perf_counter()
        out = await asyncio.to_thread(
            _bm25_search_articles, client, articles_collection,
            text=query_text.lower(), limit=max(limit, bm25_pool),
            filter_expr=article_expr,
        )
        timings.bm25_ms = (time.perf_counter() - t0) * 1000
        return out

    if multi_word:
        dense = await do_dense()
        return dense[:limit]

    if strict_id:
        bm = await do_bm25()
        if bm:
            return bm[:limit]
        # Empty strict → fall through to hybrid (legacy fallback behavior).

    dense, bm = await asyncio.gather(do_dense(), do_bm25())
    return rrf_merge([dense, bm], k=rrf_k, top_n=limit)


# ──────────────────────────────────────────────────────────────────────
# Milvus helpers
# ──────────────────────────────────────────────────────────────────────

_OFFER_PROBE_FIELDS = ["id", "article_hash", "vendor_id", "prices"]


def _offer_probe(
    client: MilvusClient,
    collection: str,
    *,
    offer_expr: str,
    limit: int,
) -> list[dict]:
    """Bounded probe on `offers_v{N}` for Path B's first step."""
    return client.query(
        collection_name=collection,
        filter=offer_expr,
        output_fields=_OFFER_PROBE_FIELDS,
        limit=limit,
    )


def _resolve_offers(
    client: MilvusClient,
    collection: str,
    *,
    hashes: list[str],
    offer_expr: str | None,
    need_eclass_post_pass: bool,
) -> dict[str, list[dict]]:
    """Path A's resolve step: `hash IN [...]` (and optional offer_expr).
    Returns offers grouped by article_hash."""
    if not hashes:
        return {}
    expr = _hash_in_expr(hashes)
    if offer_expr:
        expr = f"({expr}) and ({offer_expr})"
    # Cap the resolve fetch at Milvus's max query window. At production
    # scale with effective_k > 256 hashes × 64 worst-case offers/article
    # this would exceed 16384; pages will be short of pageSize when this
    # bites. Operators can raise the quota via `proxy.maxResultWindow`.
    fetch_limit = min(len(hashes) * _MAX_OFFERS_PER_ARTICLE, _MILVUS_MAX_QUERY_WINDOW)
    rows = client.query(
        collection_name=collection,
        filter=expr,
        output_fields=_OFFER_PROBE_FIELDS,
        limit=fetch_limit,
    )
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["article_hash"], []).append(r)
    return grouped


# Worst-case offers per article in production. Bounds the resolve
# query's `limit`; very high vs typical (~3.2 offers/article) but
# protects against pathological articles with hundreds of vendors. If
# this is hit at runtime the page will be short — cheap to raise.
_MAX_OFFERS_PER_ARTICLE = 64


def _article_browse(
    client: MilvusClient,
    collection: str,
    *,
    article_expr: str,
    limit: int,
) -> list[dict]:
    """No-query browse: `query()` against article_expr. Returns hash list
    with no defensible ordering — Milvus returns segment-internal order."""
    return client.query(
        collection_name=collection,
        filter=article_expr,
        output_fields=["article_hash"],
        limit=limit,
    )


def _dense_search_articles(
    client: MilvusClient,
    collection: str,
    *,
    vec: Sequence[float],
    limit: int,
    num_candidates: int | None,
    filter_expr: str | None,
) -> list[tuple[str, float]]:
    """Dense ANN on `articles_v{N}`. Mirrors hybrid._dense_search but
    targets `article_hash` as the id field."""
    query = np.asarray(vec, dtype=np.float16)
    params: dict = {}
    if num_candidates is not None and num_candidates > 0:
        params["ef"] = num_candidates
    kwargs: dict = {
        "collection_name": collection,
        "data": [query],
        "anns_field": "offer_embedding",
        "limit": limit,
        "search_params": {"metric_type": "COSINE", "params": params},
        "output_fields": ["article_hash"],
    }
    if filter_expr:
        kwargs["filter"] = filter_expr
    res = client.search(**kwargs)
    raw = res[0] if res else []
    out: list[tuple[str, float]] = []
    for h in raw:
        ent = h.get("entity", {}) if isinstance(h, dict) else {}
        out.append((str(ent.get("article_hash", "")), float(h["distance"])))
    out.sort(key=lambda r: (-r[1], r[0]))
    return out


def _bm25_search_articles(
    client: MilvusClient,
    collection: str,
    *,
    text: str,
    limit: int,
    filter_expr: str | None,
) -> list[tuple[str, float]]:
    """BM25 on `articles_v{N}.sparse_codes` (the F6-absorbed identifier
    corpus). Pushes the article-side filter expr down — no separate
    intersect-with-filter shim needed (the legacy hybrid had a
    standalone codes collection with no scalars; F9's BM25 is
    co-located with article scalars)."""
    kwargs: dict = {
        "collection_name": collection,
        "data": [text],
        "anns_field": "sparse_codes",
        "limit": limit,
        "search_params": {"metric_type": "BM25"},
        "output_fields": ["article_hash"],
    }
    if filter_expr:
        kwargs["filter"] = filter_expr
    res = client.search(**kwargs)
    raw = res[0] if res else []
    out: list[tuple[str, float]] = []
    for h in raw:
        ent = h.get("entity", {}) if isinstance(h, dict) else {}
        out.append((str(ent.get("article_hash", "")), float(h["distance"])))
    out.sort(key=lambda r: (-r[1], r[0]))
    return out


# ──────────────────────────────────────────────────────────────────────
# Materialisation: representative-offer pick + price post-pass
# ──────────────────────────────────────────────────────────────────────

def _materialise(
    ranked_hashes: list[tuple[str, float]],
    offers_by_hash: dict[str, list[dict]],
    article_meta: dict[str, dict] | None,
    req: SearchRequest,
    *,
    sort_plan: SortPlan,
    price_active: bool,
) -> list[_Materialised]:
    """For each ranked hash, run the per-vendor blocked-eclass post-pass
    (when active), pick a sort-aware representative offer, and capture
    the article-level data needed for the final sort.

    Articles whose offers all fail the post-pass drop. The returned list
    is in rank order (caller applies the final sort)."""
    out: list[_Materialised] = []
    sas = req.selected_article_sources
    pf = req.price_filter
    bound_currency = pf.currency_code if pf else "EUR"
    per_vendor_blocked = has_per_vendor_blocked_eclass(req)

    def _resolver(o: dict) -> "Decimal | None":
        # Returns the resolved price under the request scope, or None if
        # no in-scope price exists OR the price falls outside the
        # request's price filter bounds. The materialiser's caller treats
        # None as "this offer doesn't pass the price post-pass" — so a
        # filter-active request automatically drops out-of-bounds offers.
        from prices import decode_minor_units
        resolved = resolve_price(
            o.get("prices"),
            currency=req.currency,
            source_price_list_ids=sas.source_price_list_ids,
        )
        if resolved is None:
            return None
        if pf and pf.min is not None and resolved < decode_minor_units(pf.min, bound_currency):
            return None
        if pf and pf.max is not None and resolved > decode_minor_units(pf.max, bound_currency):
            return None
        return resolved

    for hash_, score in ranked_hashes:
        offers = offers_by_hash.get(hash_, [])
        if not offers:
            continue

        if per_vendor_blocked and article_meta is not None:
            article = article_meta.get(hash_, {})
            offers = [
                o for o in offers
                if not _per_vendor_blocked(o.get("vendor_id"), article, req)
            ]
            if not offers:
                continue

        chosen = pick_representative(
            offers, plan=sort_plan,
            price_filter_active=price_active,
            price_resolver=_resolver,
        )
        if chosen is None:
            continue
        offer, resolved_price = chosen

        article_name = None
        if sort_plan.field is SortField.NAME and article_meta is not None:
            article_name = (article_meta.get(hash_) or {}).get("name")

        out.append(_Materialised(
            article_hash=hash_,
            relevance_score=score,
            representative_offer=offer,
            resolved_price=resolved_price,
            article_name=article_name,
        ))
    return out


def _to_hits(items: list[_Materialised], sort_plan: SortPlan) -> list[Hit]:
    """Convert sorted _Materialised items into Hits.

    For relevance sort, score is the RRF score and source is `rrf`.
    For non-relevance sort, score is None per spec §3 (the wire schema
    accepts null) and source is `sort` for telemetry."""
    if sort_plan.is_relevance:
        return [
            Hit(id=str(m.representative_offer["id"]),
                score=m.relevance_score, source="rrf")
            for m in items
        ]
    return [
        Hit(id=str(m.representative_offer["id"]), score=0.0, source="sort")
        for m in items
    ]


def _per_vendor_blocked(
    vendor_id: str | None,
    article: dict,
    req: SearchRequest,
) -> bool:
    """True if (offer's vendor IN entry.vendor_ids) AND (article eclass
    IN block_true minus block_false) for any per-vendor entry."""
    if not vendor_id:
        return False
    for entry in req.blocked_eclass_vendors_filters:
        if not entry.vendor_ids or vendor_id not in entry.vendor_ids:
            continue
        block_true = {g.e_class_group_code for g in entry.blocked_e_class_groups if g.value}
        block_false = {g.e_class_group_code for g in entry.blocked_e_class_groups if not g.value}
        if not block_true:
            continue
        field = _ECLASS_FIELD[entry.e_class_version]
        codes = set(article.get(field) or [])
        if codes & block_true and not (codes & block_false):
            return True
    return False


_ECLASS_FIELD = {
    EClassVersion.ECLASS_5_1: "eclass5_code",
    EClassVersion.ECLASS_7_1: "eclass7_code",
    EClassVersion.S2CLASS: "s2class_code",
}


async def _fetch_article_meta(
    client: MilvusClient,
    articles_collection: str,
    hashes: list[str],
    req: SearchRequest,
    *,
    sort_plan: SortPlan,
) -> dict[str, dict] | None:
    """Fetch article-level fields needed downstream. Always batched into
    a single query; returns None when no fields are needed.

    Fields fetched on demand:
      - `name` if sort=name (required for the final sort key).
      - `eclassN_code` for each version referenced by per-vendor entries
        in `blocked_eclass_vendors_filters` (required for the Python
        post-pass on offer × article eclass correlation).
    """
    if not hashes:
        return None
    needed: set[str] = set()
    if sort_plan.field is SortField.NAME:
        needed.add("name")
    if has_per_vendor_blocked_eclass(req):
        needed |= {_ECLASS_FIELD[e.e_class_version]
                   for e in req.blocked_eclass_vendors_filters
                   if e.vendor_ids}
    if not needed:
        return None
    fields = sorted(needed)
    rows = await asyncio.to_thread(
        client.query,
        collection_name=articles_collection,
        filter=_hash_in_expr(hashes),
        output_fields=["article_hash", *fields],
        limit=len(hashes),
    )
    return {str(r["article_hash"]): r for r in rows}


# ──────────────────────────────────────────────────────────────────────
# F5 — summary fetch + compute
# ──────────────────────────────────────────────────────────────────────

async def _dispatch_summaries_only(
    req: SearchRequest,
    *,
    article_expr: str | None,
    offer_expr: str | None,
    client: MilvusClient,
    articles_collection: str,
    offers_collection: str,
    path_b_hash_limit: int,
    hitcount_cap: int,
    timings: _Timings,
) -> DispatchResult:
    """SUMMARIES_ONLY fast path: skip ANN/BM25, ranking, materialise,
    sort, page. Just identify the filtered hit set, fetch the summary
    columns, compute aggregations, return."""
    distinct_hashes: list[str] | None = None
    hit_count = 0
    hit_count_clipped = False
    recall_clipped = False

    if offer_expr is None:
        # Path A — count articles matching article_expr.
        timings.path = "A_summaries_only"
        hit_count, hit_count_clipped = await _count_articles(
            client, articles_collection,
            article_expr=article_expr, hashes=None, cap=hitcount_cap,
            timings=timings,
        )
    else:
        # Path B — bounded probe for distinct hashes.
        timings.path = "B_summaries_only"
        t0 = time.perf_counter()
        probe_rows = await asyncio.to_thread(
            _offer_probe, client, offers_collection,
            offer_expr=offer_expr, limit=path_b_hash_limit + 1,
        )
        timings.offer_resolve_ms = (time.perf_counter() - t0) * 1000
        timings.probe_hits = len(probe_rows)
        distinct_hashes = sorted({r["article_hash"] for r in probe_rows})
        timings.distinct_hashes = len(distinct_hashes)

        if len(distinct_hashes) > path_b_hash_limit:
            # Probe overflow: summaries computed on the truncated set
            # are necessarily incomplete. Return clipped counts.
            timings.probe_overflowed = True
            recall_clipped = True
            hit_count = path_b_hash_limit
            hit_count_clipped = True
        elif article_expr is None:
            hit_count = len(distinct_hashes)
        else:
            hit_count, hit_count_clipped = await _count_articles(
                client, articles_collection,
                article_expr=article_expr, hashes=distinct_hashes,
                cap=hitcount_cap, timings=timings,
            )

    summaries: Summaries | None = None
    if req.summaries and not recall_clipped:
        summaries = await _compute_summaries(
            req,
            article_expr=article_expr, offer_expr=offer_expr,
            path_b_distinct_hashes=distinct_hashes,
            client=client,
            articles_collection=articles_collection,
            offers_collection=offers_collection,
            hitcount_cap=hitcount_cap,
        )

    return DispatchResult(
        hits=[],
        debug=_debug(timings),
        recall_clipped=recall_clipped,
        hit_count=hit_count,
        hit_count_clipped=hit_count_clipped,
        summaries=summaries,
    )


async def _compute_summaries(
    req: SearchRequest,
    *,
    article_expr: str | None,
    offer_expr: str | None,
    path_b_distinct_hashes: list[str] | None,
    client: MilvusClient,
    articles_collection: str,
    offers_collection: str,
    hitcount_cap: int,
) -> Summaries:
    """Fetch summary fields from articles + offers (only what the
    requested kinds need) and call into the aggregations module.

    Field-set planning is in `aggregations.{article,offer}_fields_needed`.
    Fetches are capped at `min(hitcount_cap, _MILVUS_MAX_QUERY_WINDOW)`;
    when summaries clip, the overall response already carries
    `hitCountClipped: true`."""
    article_fields = aggregations.article_fields_needed(req)
    offer_fields = aggregations.offer_fields_needed(req)
    needs_articles = aggregations.needs_article_fetch(req)
    needs_offers = aggregations.needs_offer_fetch(req)
    fetch_limit = min(hitcount_cap, _MILVUS_MAX_QUERY_WINDOW)

    article_rows: list[dict] = []
    offer_rows: list[dict] = []

    if needs_articles:
        if path_b_distinct_hashes is not None:
            expr = _and_exprs(_hash_in_expr(path_b_distinct_hashes), article_expr)
        else:
            expr = article_expr or 'article_hash != ""'
        article_rows = await asyncio.to_thread(
            client.query,
            collection_name=articles_collection,
            filter=expr,
            output_fields=sorted(article_fields),
            limit=fetch_limit,
        )

    if needs_offers:
        if offer_expr is not None:
            # Path B: the same offer_expr that ran in the probe (the
            # probe might have truncated fields, so we refetch with the
            # summary-required field set).
            offer_rows = await asyncio.to_thread(
                client.query,
                collection_name=offers_collection,
                filter=offer_expr,
                output_fields=sorted(offer_fields),
                limit=fetch_limit,
            )
        else:
            # Path A: filter by hash IN matched articles. If we already
            # fetched articles above, reuse those hashes; otherwise
            # query articles for the hash set first.
            if not article_rows:
                seed = await asyncio.to_thread(
                    client.query,
                    collection_name=articles_collection,
                    filter=article_expr or 'article_hash != ""',
                    output_fields=["article_hash"],
                    limit=fetch_limit,
                )
                hashes = [str(r["article_hash"]) for r in seed]
            else:
                hashes = [str(r["article_hash"]) for r in article_rows]
            if hashes:
                offer_rows = await asyncio.to_thread(
                    client.query,
                    collection_name=offers_collection,
                    filter=_hash_in_expr(hashes),
                    output_fields=sorted(offer_fields),
                    limit=fetch_limit,
                )

    return aggregations.compute_summaries(
        req, article_rows=article_rows, offer_rows=offer_rows,
    )


async def _count_articles(
    client: MilvusClient,
    articles_collection: str,
    *,
    article_expr: str | None,
    hashes: list[str] | None,
    cap: int,
    timings: _Timings,
) -> tuple[int, bool]:
    """Run a count(*)-style pass. Returns `(count, clipped)` where
    `clipped` is true when the result hit the cap (the true count is
    `>= count`).

    `expr` composition: `article_expr ∧ (hash IN [hashes])`. Both are
    optional; if both None we count every row in the collection (with
    `article_hash != ""` as the always-true sentinel).
    """
    expr = _and_exprs(
        article_expr,
        _hash_in_expr(hashes) if hashes else None,
    ) or 'article_hash != ""'
    fetch_limit = min(cap + 1, _MILVUS_MAX_QUERY_WINDOW)
    t0 = time.perf_counter()
    rows = await asyncio.to_thread(
        client.query,
        collection_name=articles_collection,
        filter=expr,
        output_fields=["article_hash"],
        limit=fetch_limit,
    )
    timings.hit_count_query_ms = (timings.hit_count_query_ms or 0.0) + (time.perf_counter() - t0) * 1000
    if len(rows) > cap:
        return (cap, True)
    return (len(rows), False)


# ──────────────────────────────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────────────────────────────

def _hash_in_expr(hashes: list[str]) -> str:
    quoted = ", ".join(f'"{h}"' for h in hashes)
    return f"article_hash in [{quoted}]"


def _and_exprs(*parts: str | None) -> str | None:
    keep = [p for p in parts if p]
    if not keep:
        return None
    if len(keep) == 1:
        return keep[0]
    return " and ".join(f"({p})" for p in keep)


def _price_active(req: SearchRequest) -> bool:
    pf = req.price_filter
    return pf is not None and (pf.min is not None or pf.max is not None)


def _debug(t: _Timings) -> dict:
    return {
        "path": t.path,
        "article_expr": t.article_expr,
        "offer_expr": t.offer_expr,
        "sort_field": t.sort_field,
        "probe_hits": t.probe_hits,
        "probe_overflowed": t.probe_overflowed,
        "distinct_hashes": t.distinct_hashes,
        "relevance_bound_dropped": t.relevance_bound_dropped,
        "embed_ms": _r(t.embed_ms),
        "dense_ms": _r(t.dense_ms),
        "bm25_ms": _r(t.bm25_ms),
        "article_rank_ms": _r(t.article_rank_ms),
        "offer_resolve_ms": _r(t.offer_resolve_ms),
        "hit_count_query_ms": _r(t.hit_count_query_ms),
    }


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 1)
