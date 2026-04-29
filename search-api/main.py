"""Thin JSON wrapper around a TEI embedder + Milvus dense + BM25 codes search.

Endpoint: ``POST /{collection}/_search`` — `collection` names the dense
collection (e.g. ``offers``). The codes companion collection (e.g.
``offers_codes``) is server-configured. Returns Elasticsearch-shaped hits.

Search behaviour is fully parametrised by query string:

  mode                vector | bm25 | hybrid | hybrid_classified  (default
                      hybrid_classified). See ``search-api/hybrid.py``.
  k                   final top-N returned (default SEARCH_TOP_K).
  dense_limit         dense candidate pool in hybrid path (default 200).
  codes_limit         codes candidate pool in hybrid path (default 20).
  strict_codes_limit  codes pool in strict path (default 500).
  rrf_k               RRF k constant (default 60).
  num_candidates      HNSW efSearch (must be >= k; ignored by non-HNSW).
  enable_fallback     1|0 — strict-path 0-result fallback to hybrid (default 1).
  debug               1|0 — include `_debug` in response (default 0).

Environment variables:
  EMBED_URL                 Base URL of a TEI-compatible embedding service.
  QUERY_PREFIX              String prepended to every query before embedding.
  MILVUS_URI                Milvus gRPC URI.
  MILVUS_CODES_COLLECTION   Codes collection name (default "offers_codes").
  SEARCH_TOP_K              Default ``k`` (default 100).
  ID_FIELDS                 Per-collection override of the Milvus field
                            returned as ``_id``: ``col=field,col2=field2``.
  API_KEY                   Shared secret. Required via
                            ``Authorization: ApiKey <key>`` or ``X-API-Key``.
  MAX_CONCURRENT_SEARCHES   Hard cap on in-flight search requests. Excess
                            requests return 503 + ``Retry-After: 1``. Sized
                            to keep p99 well under SLO under burst load;
                            saturation throughput sits well below the cap
                            so this only fires during true overload.
                            Default 64. Set to 0 to disable.
  USE_DEDUP_TOPOLOGY        F9 feature flag. ``1`` routes /{collection}/_search
                            via the dedup-topology dispatcher (`routing.py`);
                            unset/``0`` keeps the legacy single-collection
                            path. Default off.
  MILVUS_ARTICLES_COLLECTION
                            F9: name (or alias) of the article-side collection
                            for the dedup path. Default "articles".
  PATH_B_HASH_LIMIT         F9: bounded-probe ceiling for Path B. Above this,
                            Path B falls back to Path A with `recallClipped`.
                            Default 16383 — Milvus 2.6's `proxy.maxResultWindow`
                            quota caps `(offset+limit)` at 16384 on `query()`.
                            Raise the Milvus quota + restart to push higher.
  RELEVANCE_POOL_MAX        F4: candidate-pool ceiling when sort is non-relevance
                            AND a query string is present. Default 200.
  RELEVANCE_SCORE_FLOOR     F4: relative score floor (×top_score). Candidates
                            below this floor drop from the relevance pool when
                            non-relevance sort is active. Default 0.20.
  HITCOUNT_CAP              F4: maximum count returned in `metadata.hitCount`.
                            When the cap fires, `hitCountClipped: true`.
                            Default 10000.
  PRICE_FILTER_OVERFETCH_N  F3: page over-fetch factor when a price filter is
                            active. Default 10.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from embed_client import EmbedClient
from filters import build_milvus_expr
from hybrid import Hit, Mode, SearchParams, run_search
from models import (
    Article,
    Metadata,
    SearchMode,
    SearchRequest,
    SearchResponse,
    Summaries,
    parse_sort_params,
)
from prices import passes_price_filter
from routing import (
    DEFAULT_HITCOUNT_CAP,
    DEFAULT_PATH_B_HASH_LIMIT,
    DEFAULT_RELEVANCE_POOL_MAX,
    DEFAULT_RELEVANCE_SCORE_FLOOR,
    dispatch_dedup,
)
from sorting import parse_plan as parse_sort_plan

BASE_DIR = Path(__file__).resolve().parent
OPENAPI_YAML_PATH = BASE_DIR / "openapi.yaml"
_OPENAPI_YAML_TEXT = OPENAPI_YAML_PATH.read_text()
_OPENAPI_SPEC = yaml.safe_load(_OPENAPI_YAML_TEXT)

# Paths served without auth or the concurrency gate. The hand-written
# OpenAPI doc and its viewers are public so that integrators can fetch
# the contract without provisioning an API key.
_PUBLIC_PATHS = frozenset({
    "/metrics",
    "/openapi.json",
    "/openapi.yaml",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
})

_DEFAULT_ID_FIELD = "id"
_DEFAULT_CODES_COLLECTION = "offers_codes"
_DEFAULT_MAX_CONCURRENCY = 64
_DEFAULT_ARTICLES_COLLECTION = "articles"


class _ConcurrencyGate:
    """Non-blocking in-process concurrency cap.

    asyncio is single-threaded, so the int read+write between awaits is
    atomic — no lock needed. ``limit <= 0`` disables the gate.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.inflight = 0

    def try_acquire(self) -> bool:
        if self.limit <= 0:
            return True
        if self.inflight >= self.limit:
            return False
        self.inflight += 1
        return True

    def release(self) -> None:
        if self.limit > 0 and self.inflight > 0:
            self.inflight -= 1


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _parse_id_fields(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            raise RuntimeError(
                f"ID_FIELDS entry {item!r} is not of the form 'collection=field'"
            )
        out[key] = value
    return out


class LegacySearchRequest(BaseModel):
    """Original search-api body served at /{collection}/_search_v0.

    Kept alive as a deprecated alias so the playground app, the loadtest
    runner, and `scripts/validate_hybrid.py` keep producing real hits
    while F3..F5 fill in the new `/{collection}/_search` semantics. To
    be removed once the new contract is fully behaviour-complete.
    """
    query: str
    category: str | None = Field(default=None)
    index: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")

    app.state.embed = EmbedClient(_required_env("EMBED_URL"))
    milvus_uri = _required_env("MILVUS_URI")
    # Two clients per hybrid_v0.md "decoupled clients" — same URI, separate
    # connection state so future per-collection settings can diverge.
    app.state.milvus = MilvusClient(milvus_uri)
    app.state.codes_milvus = MilvusClient(milvus_uri)
    app.state.codes_collection = os.environ.get(
        "MILVUS_CODES_COLLECTION", _DEFAULT_CODES_COLLECTION
    )
    app.state.query_prefix = os.environ.get("QUERY_PREFIX", "")
    app.state.top_k = int(os.environ.get("SEARCH_TOP_K", "100"))
    app.state.id_fields = _parse_id_fields(os.environ.get("ID_FIELDS", ""))
    app.state.api_key = os.environ.get("API_KEY", "")
    app.state.gate = _ConcurrencyGate(
        int(os.environ.get("MAX_CONCURRENT_SEARCHES", _DEFAULT_MAX_CONCURRENCY))
    )
    # F9 dedup-topology routing: when on, /{collection}/_search dispatches
    # via routing.dispatch_dedup against the article + offer collections
    # below. When off, the legacy single-collection path is unchanged.
    app.state.use_dedup_topology = _flag(
        os.environ.get("USE_DEDUP_TOPOLOGY"), default=False,
    )
    app.state.articles_collection = os.environ.get(
        "MILVUS_ARTICLES_COLLECTION", _DEFAULT_ARTICLES_COLLECTION,
    )
    app.state.path_b_hash_limit = int(
        os.environ.get("PATH_B_HASH_LIMIT", DEFAULT_PATH_B_HASH_LIMIT)
    )
    app.state.relevance_pool_max = int(
        os.environ.get("RELEVANCE_POOL_MAX", DEFAULT_RELEVANCE_POOL_MAX)
    )
    app.state.relevance_score_floor = float(
        os.environ.get("RELEVANCE_SCORE_FLOOR", DEFAULT_RELEVANCE_SCORE_FLOOR)
    )
    app.state.hitcount_cap = int(
        os.environ.get("HITCOUNT_CAP", DEFAULT_HITCOUNT_CAP)
    )
    try:
        yield
    finally:
        await app.state.embed.aclose()


app = FastAPI(lifespan=lifespan)


def _custom_openapi() -> dict:
    return _OPENAPI_SPEC


app.openapi = _custom_openapi  # type: ignore[method-assign]


@app.get("/openapi.yaml", include_in_schema=False)
async def openapi_yaml() -> Response:
    return Response(content=_OPENAPI_YAML_TEXT, media_type="application/yaml")


Instrumentator().instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    expected: str = request.app.state.api_key
    if not expected:
        return await call_next(request)

    ok = False
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "apikey ":
        ok = secrets.compare_digest(auth[7:].strip(), expected)
    if not ok:
        xkey = request.headers.get("x-api-key", "")
        if xkey:
            ok = secrets.compare_digest(xkey, expected)
    if not ok:
        return JSONResponse(
            status_code=401,
            content={"detail": "invalid or missing api key"},
            headers={"WWW-Authenticate": 'ApiKey realm="search-api"'},
        )
    return await call_next(request)


@app.middleware("http")
async def limit_concurrency(request: Request, call_next):
    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)
    gate: _ConcurrencyGate = request.app.state.gate
    if not gate.try_acquire():
        return JSONResponse(
            status_code=503,
            content={"detail": "search-api at concurrency limit"},
            headers={"Retry-After": "1"},
        )
    try:
        return await call_next(request)
    finally:
        gate.release()


def _parse_mode(raw: str | None) -> Mode:
    if not raw:
        return Mode.HYBRID_CLASSIFIED
    try:
        return Mode(raw)
    except ValueError:
        valid = ", ".join(m.value for m in Mode)
        raise HTTPException(
            status_code=400,
            detail=f"invalid mode {raw!r}; expected one of: {valid}",
        )


def _flag(raw: str | None, default: bool) -> bool:
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@app.post("/{collection}/_search_v0", deprecated=True)
async def search_v0(
    body: LegacySearchRequest,
    request: Request,
    collection: str = PathParam(..., min_length=1, max_length=255),
    mode: str | None = Query(default=None),
    k: int | None = Query(default=None, ge=1),
    dense_limit: int | None = Query(default=None, ge=1),
    codes_limit: int | None = Query(default=None, ge=1),
    strict_codes_limit: int | None = Query(default=None, ge=1),
    rrf_k: int | None = Query(default=None, ge=1),
    num_candidates: int | None = Query(default=None, ge=1),
    enable_fallback: str | None = Query(default=None),
    debug: str | None = Query(default=None),
) -> dict:
    dense_client: MilvusClient = request.app.state.milvus
    codes_client: MilvusClient = request.app.state.codes_milvus
    codes_collection: str = request.app.state.codes_collection

    parsed_mode = _parse_mode(mode)
    needs_dense = parsed_mode in (Mode.VECTOR, Mode.HYBRID, Mode.HYBRID_CLASSIFIED)
    needs_codes = parsed_mode != Mode.VECTOR

    if needs_dense and not dense_client.has_collection(collection):
        raise HTTPException(
            status_code=404,
            detail=f"Milvus dense collection {collection!r} does not exist",
        )
    if needs_codes and not codes_client.has_collection(codes_collection):
        raise HTTPException(
            status_code=404,
            detail=f"Milvus codes collection {codes_collection!r} does not exist",
        )

    final_k = k if k is not None else request.app.state.top_k
    params = SearchParams(
        mode=parsed_mode,
        k=final_k,
        dense_limit=dense_limit if dense_limit is not None else 200,
        codes_limit=codes_limit if codes_limit is not None else 20,
        strict_codes_limit=strict_codes_limit if strict_codes_limit is not None else 500,
        rrf_k=rrf_k if rrf_k is not None else 60,
        num_candidates=num_candidates,
        enable_fallback=_flag(enable_fallback, default=True),
    )

    # HNSW efSearch ≥ the dense leg's limit. For vector mode that's `k`;
    # for hybrid/hybrid_classified it's `dense_limit` (which the dense leg
    # uses as its candidate pool — k is the post-fusion top-N).
    if num_candidates is not None:
        dense_leg_limit = (
            params.k if params.mode == Mode.VECTOR else params.dense_limit
        )
        if num_candidates < dense_leg_limit:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"num_candidates ({num_candidates}) must be >= the dense "
                    f"leg's limit ({dense_leg_limit})"
                ),
            )

    query = body.query.strip()
    if not query:
        return {"hits": []}

    rendered = f"{request.app.state.query_prefix}{query}"
    id_field: str = request.app.state.id_fields.get(collection, _DEFAULT_ID_FIELD)

    async def embed_fn(text: str) -> list[float]:
        # The dense path embeds with QUERY_PREFIX prepended; BM25 path uses
        # the raw lowercased query (handled inside hybrid.run_search). We
        # always pass the rendered (prefixed) query into the embedder.
        vectors = await request.app.state.embed.embed([rendered])
        return vectors[0] if vectors else []

    hits, debug_info = await run_search(
        query,
        params,
        dense_client=dense_client,
        codes_client=codes_client,
        embed=embed_fn,
        dense_collection=collection,
        codes_collection=codes_collection,
        dense_id_field=id_field,
    )

    es_hits = [
        {
            "_index": body.index,
            "_id": h.id,
            "_score": h.score,
            "_source_leg": h.source,
        }
        for h in hits
    ]
    response: dict = {"hits": es_hits}
    if _flag(debug, default=False):
        response["_debug"] = debug_info
    return response


_DEFAULT_PRICE_FILTER_OVERFETCH_N = 10


def _price_filter_active(body: SearchRequest) -> bool:
    pf = body.price_filter
    return pf is not None and (pf.min is not None or pf.max is not None)


def _filter_only_browse(
    client: MilvusClient, collection: str, *, expr: str, limit: int, id_field: str,
) -> list[Hit]:
    """No query string → run a Milvus `query()` against the scalar filter.

    Mirrors the legacy "browse a category without a search term" path.
    Without an `expr` (no filters either) we return nothing — there is
    no defensible default ordering.
    """
    rows = client.query(
        collection_name=collection,
        filter=expr,
        output_fields=[id_field],
        limit=limit,
    )
    return [Hit(id=str(r.get(id_field, "")), score=0.0, source="filter") for r in rows]


def _hydrate_prices(
    client: MilvusClient, collection: str, ids: list[str], id_field: str,
) -> dict[str, list]:
    if not ids:
        return {}
    quoted = ", ".join(f'"{i}"' for i in ids)
    rows = client.query(
        collection_name=collection,
        filter=f"{id_field} in [{quoted}]",
        output_fields=[id_field, "prices"],
        limit=len(ids),
    )
    return {str(r.get(id_field, "")): r.get("prices") or [] for r in rows}


async def _search_dedup(
    body: SearchRequest,
    request: Request,
    collection: str,
    *,
    page: int,
    page_size: int,
    overfetch_n: int,
    sort_clauses: list,
) -> SearchResponse:
    """F9 + F4 dispatch: route through `routing.dispatch_dedup`, convert
    the result to the `SearchResponse` envelope. SUMMARIES_ONLY skips
    article hydration but still populates hitCount."""
    client: MilvusClient = request.app.state.milvus
    articles_collection: str = request.app.state.articles_collection
    offers_collection: str = collection
    path_b_hash_limit: int = request.app.state.path_b_hash_limit

    if not client.has_collection(articles_collection):
        raise HTTPException(
            status_code=404,
            detail=f"Milvus articles collection {articles_collection!r} does not exist",
        )
    if not client.has_collection(offers_collection):
        raise HTTPException(
            status_code=404,
            detail=f"Milvus offers collection {offers_collection!r} does not exist",
        )

    query_text = (body.query or "").strip()

    async def embed_fn(_text: str) -> list[float]:
        rendered = f"{request.app.state.query_prefix}{query_text}"
        vectors = await request.app.state.embed.embed([rendered])
        return vectors[0] if vectors else []

    sort_plan = parse_sort_plan(sort_clauses)

    result = await dispatch_dedup(
        body,
        page=page, page_size=page_size, overfetch_n=overfetch_n,
        sort_plan=sort_plan,
        client=client, embed=embed_fn,
        articles_collection=articles_collection,
        offers_collection=offers_collection,
        path_b_hash_limit=path_b_hash_limit,
        relevance_pool_max=request.app.state.relevance_pool_max,
        relevance_score_floor=request.app.state.relevance_score_floor,
        hitcount_cap=request.app.state.hitcount_cap,
    )

    # SUMMARIES_ONLY: empty articles[] but real hitCount (F5 fills in
    # summaries; this packet wires the mode flag).
    skip_articles = body.search_mode is SearchMode.SUMMARIES_ONLY
    articles = (
        [] if skip_articles
        else [Article(articleId=h.id, score=h.score) for h in result.hits]
    )

    page_count = (
        (result.hit_count + page_size - 1) // page_size
        if page_size > 0 and result.hit_count > 0 else 0
    )

    return SearchResponse(
        articles=articles,
        summaries=Summaries(),
        metadata=Metadata(
            page=page,
            pageSize=page_size,
            pageCount=page_count,
            term=body.query or "",
            hitCount=result.hit_count,
            recallClipped=result.recall_clipped,
            hitCountClipped=result.hit_count_clipped,
        ),
    )


@app.post("/{collection}/_search", response_model=SearchResponse, response_model_by_alias=True)
async def search(
    body: SearchRequest,
    request: Request,
    collection: str = PathParam(..., min_length=1, max_length=255),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=0, le=500, alias="pageSize"),
    sort: list[str] = Query(default_factory=list),
) -> SearchResponse:
    """F3 — scalar filtering + price-resolution post-pass.

    Filters listed in spec §4.3 are AND-composed into a Milvus expr that
    the dense leg pushes down and the BM25 leg intersects against. The
    `priceFilter` is applied in Python after over-fetching by a factor
    of `PRICE_FILTER_OVERFETCH_N` (default 10).

    Pagination, sort, accurate `hitCount`, and `summaries` are F4/F5.
    Until then the response carries `hitCount = len(articles)` (page
    slice only) and empty summaries.
    """
    try:
        sort_clauses = parse_sort_params(sort)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    overfetch_n = int(os.environ.get("PRICE_FILTER_OVERFETCH_N", _DEFAULT_PRICE_FILTER_OVERFETCH_N))

    # F9 dedup-topology dispatch (flag-gated). The URL's `collection`
    # is the offers-side alias under dedup mode; the article-side comes
    # from `MILVUS_ARTICLES_COLLECTION`. Legacy path below is unchanged.
    if request.app.state.use_dedup_topology:
        return await _search_dedup(
            body, request, collection,
            page=page, page_size=page_size, overfetch_n=overfetch_n,
            sort_clauses=sort_clauses,
        )

    expr = build_milvus_expr(body)

    dense_client: MilvusClient = request.app.state.milvus
    codes_client: MilvusClient = request.app.state.codes_milvus
    codes_collection: str = request.app.state.codes_collection
    id_field: str = request.app.state.id_fields.get(collection, _DEFAULT_ID_FIELD)
    price_active = _price_filter_active(body)

    # Effective k for the retrieval phase. Over-fetch when the price
    # filter is active so that drop-outs in the post-pass don't starve
    # the page.
    effective_k = page_size * overfetch_n if price_active and page_size > 0 else max(page_size, 1)

    query_text = (body.query or "").strip()
    if not query_text:
        if not expr:
            hits: list[Hit] = []
        else:
            hits = await asyncio.to_thread(
                _filter_only_browse, dense_client, collection,
                expr=expr, limit=effective_k, id_field=id_field,
            )
    else:
        params = SearchParams(
            mode=Mode.HYBRID_CLASSIFIED,
            k=effective_k,
            dense_limit=max(effective_k, 200),
        )

        async def embed_fn(text: str) -> list[float]:
            rendered = f"{request.app.state.query_prefix}{query_text}"
            vectors = await request.app.state.embed.embed([rendered])
            return vectors[0] if vectors else []

        hits, _debug = await run_search(
            query_text,
            params,
            dense_client=dense_client,
            codes_client=codes_client,
            embed=embed_fn,
            dense_collection=collection,
            codes_collection=codes_collection,
            dense_id_field=id_field,
            filter_expr=expr,
        )

    if price_active and hits:
        prices_by_id = await asyncio.to_thread(
            _hydrate_prices, dense_client, collection, [h.id for h in hits], id_field,
        )
        sas = body.selected_article_sources
        pf = body.price_filter
        kept: list[Hit] = []
        for h in hits:
            if passes_price_filter(
                prices_by_id.get(h.id),
                request_currency=body.currency,
                source_price_list_ids=sas.source_price_list_ids,
                bound_currency_code=pf.currency_code,
                min_minor=pf.min,
                max_minor=pf.max,
            ):
                kept.append(h)
        hits = kept

    hits = hits[:page_size]
    articles = [Article(articleId=h.id, score=h.score) for h in hits]
    return SearchResponse(
        articles=articles,
        summaries=Summaries(),
        metadata=Metadata(
            page=page,
            pageSize=page_size,
            pageCount=0,
            term=body.query,
            hitCount=len(articles),
        ),
    )
