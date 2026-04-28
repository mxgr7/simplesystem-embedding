"""F9 routing — dispatch a SearchRequest across the two-collection topology.

Two paths, picked deterministically by F9 §"Routing rule":

  Path A (vector-first) — no per-offer filter applies:
    1. ANN + BM25 on `articles_v{N}` constrained to `article_expr`.
    2. RRF fuse over `article_hash`.
    3. Resolve offers: query `offers_v{N}` filtered by `article_hash IN [...]`.
    4. Pick representative offer per hash; price post-pass; paginate.

  Path B (filter-first) — at least one per-offer filter applies:
    1. Bounded probe: query `offers_v{N}` with `offer_expr`, limit
       `PATH_B_HASH_LIMIT + 1`.
    2. If distinct hashes > limit → fall back to Path A with
       `recall_clipped=True` (under-recalls selective-but-not-tight
       filters; documented in spec §2.4).
    3. Otherwise: ANN + BM25 on `articles_v{N}` with
       `article_hash IN [probe-hashes] AND article_expr`.
    4. Re-attach offers from the probe (no extra round-trip);
       representative selection; price post-pass; paginate.

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
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Sequence

import numpy as np
from pymilvus import MilvusClient

from filters import (
    build_article_expr,
    build_offer_expr,
    has_per_vendor_blocked_eclass,
)
from hybrid import Hit, is_strict_identifier, rrf_merge
from models import EClassVersion, SearchRequest
from prices import passes_price_filter

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


# ──────────────────────────────────────────────────────────────────────
# Result + dispatch entry
# ──────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class DispatchResult:
    hits: list[Hit]
    debug: dict
    recall_clipped: bool = False


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


async def dispatch_dedup(
    req: SearchRequest,
    *,
    page_size: int,
    overfetch_n: int,
    client: MilvusClient,
    embed: EmbedFn,
    articles_collection: str,
    offers_collection: str,
    path_b_hash_limit: int = DEFAULT_PATH_B_HASH_LIMIT,
    dense_pool: int = DEFAULT_DENSE_POOL,
    bm25_pool: int = DEFAULT_BM25_POOL,
    rrf_k: int = DEFAULT_RRF_K,
    num_candidates: int | None = None,
) -> DispatchResult:
    """F9 dedup-topology dispatcher. Returns a `DispatchResult` whose
    `hits` are offer-level (one Hit per article, representative offer
    selected per F9 representative rule)."""
    article_expr = build_article_expr(req)
    offer_expr = build_offer_expr(req)
    timings = _Timings(article_expr=article_expr, offer_expr=offer_expr)
    query_text = (req.query or "").strip()

    price_active = _price_active(req)
    effective_k = (
        page_size * overfetch_n if price_active and page_size > 0
        else max(page_size, 1)
    )

    if offer_expr is None:
        # Path A — no per-offer constraint to enforce.
        hits, debug = await _path_a(
            req, query_text, article_expr, effective_k,
            client=client, embed=embed,
            articles_collection=articles_collection,
            offers_collection=offers_collection,
            dense_pool=dense_pool, bm25_pool=bm25_pool,
            rrf_k=rrf_k, num_candidates=num_candidates,
            timings=timings,
        )
        timings.path = "A"
        return DispatchResult(hits=hits, debug={**_debug(timings), **debug})

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
        # Probe overflow → Path A fallback. Per F9: accepts under-recall
        # for selective-but-not-tight filters.
        timings.probe_overflowed = True
        hits, debug = await _path_a(
            req, query_text, article_expr, effective_k,
            client=client, embed=embed,
            articles_collection=articles_collection,
            offers_collection=offers_collection,
            dense_pool=dense_pool, bm25_pool=bm25_pool,
            rrf_k=rrf_k, num_candidates=num_candidates,
            timings=timings,
        )
        timings.path = "A_fallback"
        return DispatchResult(hits=hits, debug={**_debug(timings), **debug}, recall_clipped=True)

    timings.path = "B"
    hits, debug = await _path_b(
        req, query_text, article_expr, distinct_hashes, probe_rows,
        page_size=page_size, effective_k=effective_k,
        client=client, embed=embed,
        articles_collection=articles_collection,
        dense_pool=dense_pool, bm25_pool=bm25_pool,
        rrf_k=rrf_k, num_candidates=num_candidates,
        price_active=price_active,
        timings=timings,
    )
    return DispatchResult(hits=hits, debug={**_debug(timings), **debug})


# ──────────────────────────────────────────────────────────────────────
# Path A
# ──────────────────────────────────────────────────────────────────────

async def _path_a(
    req: SearchRequest,
    query_text: str,
    article_expr: str | None,
    effective_k: int,
    *,
    client: MilvusClient,
    embed: EmbedFn,
    articles_collection: str,
    offers_collection: str,
    dense_pool: int,
    bm25_pool: int,
    rrf_k: int,
    num_candidates: int | None,
    timings: _Timings,
) -> tuple[list[Hit], dict]:
    """Vector-first path. Article expression pushed down to ANN/BM25;
    offer resolve attaches per-hash offers (no offer-side filter)."""
    if query_text:
        ranked_hashes = await _rank_articles(
            query_text, article_expr, client=client, embed=embed,
            articles_collection=articles_collection,
            limit=effective_k, dense_pool=dense_pool, bm25_pool=bm25_pool,
            rrf_k=rrf_k, num_candidates=num_candidates, timings=timings,
        )
    else:
        # No query string → no defensible ANN ordering. Browse-only:
        # query articles for any hash matching article_expr (no order).
        # If no article_expr either, return nothing (mirrors legacy
        # _filter_only_browse on no-filter no-query).
        if article_expr is None:
            return [], {}
        t0 = time.perf_counter()
        rows = await asyncio.to_thread(
            _article_browse, client, articles_collection,
            article_expr=article_expr, limit=effective_k,
        )
        timings.article_rank_ms = (time.perf_counter() - t0) * 1000
        ranked_hashes = [(r["article_hash"], 0.0) for r in rows]

    if not ranked_hashes:
        return [], {}

    # Resolve offers: hash IN [ranked]. No offer-side expr filter.
    hashes = [h for h, _ in ranked_hashes]
    t0 = time.perf_counter()
    offers_by_hash = await asyncio.to_thread(
        _resolve_offers, client, offers_collection,
        hashes=hashes, offer_expr=None,
        need_eclass_post_pass=has_per_vendor_blocked_eclass(req),
    )
    timings.offer_resolve_ms = (timings.offer_resolve_ms or 0.0) + (time.perf_counter() - t0) * 1000

    # Per-vendor blocked_eclass_vendors needs article eclass codes for
    # the filtered post-pass. Fetch from articles when active.
    article_meta = await _maybe_fetch_article_meta(
        client, articles_collection, hashes, req,
    )

    return _materialise_hits(
        ranked_hashes, offers_by_hash, article_meta, req,
    ), {}


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
    page_size: int,
    effective_k: int,
    client: MilvusClient,
    embed: EmbedFn,
    articles_collection: str,
    dense_pool: int,
    bm25_pool: int,
    rrf_k: int,
    num_candidates: int | None,
    price_active: bool,
    timings: _Timings,
) -> tuple[list[Hit], dict]:
    """Filter-first path. Bounded probe already returned `probe_rows`
    constrained to `offer_expr`; this function ranks the matching hashes
    via the article ANN/BM25 (constrained to `hash IN distinct_hashes`)
    and re-attaches offers from the probe."""
    # Group probe rows by hash so we can re-attach without another query.
    offers_by_hash: dict[str, list[dict]] = {}
    for r in probe_rows:
        offers_by_hash.setdefault(r["article_hash"], []).append(r)

    if not distinct_hashes:
        return [], {}

    if query_text:
        # Constrain article ANN/BM25 to the probe hashes + article_expr.
        constrained_expr = _and_exprs(
            _hash_in_expr(distinct_hashes), article_expr,
        )
        ranked_hashes = await _rank_articles(
            query_text, constrained_expr, client=client, embed=embed,
            articles_collection=articles_collection,
            limit=effective_k, dense_pool=dense_pool, bm25_pool=bm25_pool,
            rrf_k=rrf_k, num_candidates=num_candidates, timings=timings,
        )
    elif article_expr is not None:
        # No query but article_expr applies: query articles to filter the
        # probe hashes by article-side constraints (categories, eclass,
        # manufacturer). Order is unspecified — Milvus returns whatever.
        constrained_expr = _and_exprs(
            _hash_in_expr(distinct_hashes), article_expr,
        )
        t0 = time.perf_counter()
        rows = await asyncio.to_thread(
            _article_browse, client, articles_collection,
            article_expr=constrained_expr, limit=effective_k,
        )
        timings.article_rank_ms = (time.perf_counter() - t0) * 1000
        ranked_hashes = [(r["article_hash"], 0.0) for r in rows]
    else:
        # No query AND no article_expr: present probe results in
        # deterministic id-ascending order. Milvus's row order is stable
        # within a query but not meaningful — pick a deterministic
        # representative-offer order so callers see consistent results.
        ranked_hashes = [(h, 0.0) for h in distinct_hashes]

    if not ranked_hashes:
        return [], {}

    # Per-vendor blocked_eclass_vendors needs article eclass codes.
    hashes = [h for h, _ in ranked_hashes]
    article_meta = await _maybe_fetch_article_meta(
        client, articles_collection, hashes, req,
    )

    return _materialise_hits(
        ranked_hashes, offers_by_hash, article_meta, req,
    ), {}


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

def _materialise_hits(
    ranked_hashes: list[tuple[str, float]],
    offers_by_hash: dict[str, list[dict]],
    article_meta: dict[str, dict] | None,
    req: SearchRequest,
) -> list[Hit]:
    """For each ranked hash, pick a representative offer and apply the
    price post-pass + per-vendor blocked_eclass_vendors filter (Python).
    Articles whose offers all fail post-pass drop from the response."""
    out: list[Hit] = []
    sas = req.selected_article_sources
    pf = req.price_filter
    price_active = _price_active(req)

    for hash_, score in ranked_hashes:
        offers = offers_by_hash.get(hash_, [])
        if not offers:
            continue

        # Per-vendor blocked_eclass_vendors filter (Python post-pass).
        # Drops offers whose vendor is in a restricted list AND whose
        # article eclass falls into the blocked set.
        if article_meta is not None and has_per_vendor_blocked_eclass(req):
            article = article_meta.get(hash_, {})
            offers = [
                o for o in offers
                if not _per_vendor_blocked(o.get("vendor_id"), article, req)
            ]
            if not offers:
                continue

        # Price post-pass: pick the alphabetically-lowest id offer that
        # passes; if none pass, drop the article.
        candidates = sorted(offers, key=lambda o: str(o.get("id", "")))
        chosen: dict | None = None
        for o in candidates:
            if not price_active or passes_price_filter(
                o.get("prices"),
                request_currency=req.currency,
                source_price_list_ids=sas.source_price_list_ids,
                bound_currency_code=pf.currency_code if pf else "EUR",
                min_minor=pf.min if pf else None,
                max_minor=pf.max if pf else None,
            ):
                chosen = o
                break
        if chosen is None:
            continue

        out.append(Hit(id=str(chosen["id"]), score=score, source="rrf"))
    return out


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


async def _maybe_fetch_article_meta(
    client: MilvusClient,
    articles_collection: str,
    hashes: list[str],
    req: SearchRequest,
) -> dict[str, dict] | None:
    """Fetch eclass codes per article hash when the per-vendor
    blocked-eclass-vendors post-pass is active. None otherwise."""
    if not has_per_vendor_blocked_eclass(req) or not hashes:
        return None
    fields = sorted({_ECLASS_FIELD[e.e_class_version]
                     for e in req.blocked_eclass_vendors_filters
                     if e.vendor_ids})
    rows = await asyncio.to_thread(
        client.query,
        collection_name=articles_collection,
        filter=_hash_in_expr(hashes),
        output_fields=["article_hash", *fields],
        limit=len(hashes),
    )
    return {str(r["article_hash"]): r for r in rows}


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
        "probe_hits": t.probe_hits,
        "probe_overflowed": t.probe_overflowed,
        "distinct_hashes": t.distinct_hashes,
        "embed_ms": _r(t.embed_ms),
        "dense_ms": _r(t.dense_ms),
        "bm25_ms": _r(t.bm25_ms),
        "article_rank_ms": _r(t.article_rank_ms),
        "offer_resolve_ms": _r(t.offer_resolve_ms),
    }


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 1)
