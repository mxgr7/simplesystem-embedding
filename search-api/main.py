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
"""

from __future__ import annotations

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
from hybrid import Hit, Mode, SearchParams, run_search

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


class SearchRequest(BaseModel):
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


@app.post("/{collection}/_search")
async def search(
    body: SearchRequest,
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
