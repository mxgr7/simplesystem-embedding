"""Embedding Service — FastAPI wrapper over TEI with a KVRocks cache.

Request flow per `POST /embed`:
  1. auth (optional bearer)
  2. normalise + validate inputs (8 NUL-separated fields each, ≤ MAX_INPUTS_PER_REQUEST)
  3. hash each input (sha256[:16].hex)
  4. MGET cache (50 ms timeout, treat failure as full-miss)
  5. for misses: split → render template → batch-call TEI (semaphore-bounded)
  6. stitch hits+misses, return JSON list[list[float]] at fp16 precision
  7. BackgroundTasks: MSET miss results back to cache (fire-and-forget)

Backpressure:
  - Layer 1: uvicorn --limit-concurrency (set in compose / launch cmd)
  - Layer 2: app-level inflight counter, 429 with Retry-After if over MAX_INFLIGHT
  - Layer 3: per-request 5s budget (asyncio.wait_for) → 504 on timeout
  - Layer 4: 256-input-per-request cap → 413

See `/home/mgerer/.claude/plans/create-a-new-service-joyful-rocket.md`.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import yaml
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from cache import EmbeddingCache, fp16_from_bytes, vec_bytes_from_fp16
from config import load_config
from embed_client import TEIClient
from hashing import article_hash
from models import EmbedRequest
from rendering import N_FIELDS, render_from_nul

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
OPENAPI_YAML_PATH = BASE_DIR / "openapi.yaml"

_PUBLIC_PATHS = frozenset({
    "/metrics",
    "/openapi.json",
    "/openapi.yaml",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/healthz",
})


# --- Prometheus metrics ----------------------------------------------------
# Created at import time so they're registered in the default registry that
# `Instrumentator` exposes at /metrics.

_INFLIGHT = Gauge(
    "embedding_service_inflight",
    "Current number of in-flight /embed requests (post-cache-read).",
)
_REQUESTS = Counter(
    "embedding_service_requests_total",
    "Completed /embed requests by status.",
    ("status",),
)
_TEI_SEM_WAIT = Histogram(
    "embedding_service_tei_semaphore_wait_seconds",
    "Time spent waiting for a TEI concurrency slot (per chunk).",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
_CACHE_HIT_RATIO = Gauge(
    "embedding_service_cache_hit_ratio",
    "EWMA of cache hit ratio across recent requests.",
)


class _HitRatioEWMA:
    """Exponentially-weighted moving average of (hits / total) per request."""
    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha
        self._value = 0.0
        self._initialised = False

    def update(self, hits: int, total: int) -> None:
        if total == 0:
            return
        sample = hits / total
        if not self._initialised:
            self._value = sample
            self._initialised = True
        else:
            self._value = self._alpha * sample + (1 - self._alpha) * self._value
        _CACHE_HIT_RATIO.set(self._value)


# --- Lifespan --------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.cfg = cfg
    app.state.cache = EmbeddingCache(
        cfg.kvrocks_url,
        read_timeout_s=cfg.kvrocks_read_timeout_ms / 1000.0,
        max_connections=cfg.kvrocks_max_connections,
    )
    app.state.tei = TEIClient(
        cfg.tei_url,
        max_client_batch=cfg.tei_max_client_batch,
        max_concurrency=cfg.tei_max_concurrency,
    )
    app.state.inflight = 0
    app.state.hit_ratio = _HitRatioEWMA()
    log.info(
        "embedding-service up: TEI=%s KVROCKS=%s MAX_INFLIGHT=%d",
        cfg.tei_url, cfg.kvrocks_url, cfg.max_inflight,
    )
    try:
        yield
    finally:
        await app.state.tei.aclose()
        await app.state.cache.aclose()


app = FastAPI(lifespan=lifespan)

# OpenAPI YAML served verbatim — same convention as search-api / acl.
_OPENAPI_YAML_TEXT: str | None = None
_OPENAPI_SPEC: dict | None = None
if OPENAPI_YAML_PATH.exists():
    _OPENAPI_YAML_TEXT = OPENAPI_YAML_PATH.read_text()
    _OPENAPI_SPEC = yaml.safe_load(_OPENAPI_YAML_TEXT)

    def _custom_openapi() -> dict:
        return _OPENAPI_SPEC  # type: ignore[return-value]

    app.openapi = _custom_openapi  # type: ignore[method-assign]

    @app.get("/openapi.yaml", include_in_schema=False)
    async def openapi_yaml() -> Response:
        return Response(content=_OPENAPI_YAML_TEXT, media_type="application/yaml")


Instrumentator().instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)


# --- Middleware: optional bearer auth --------------------------------------

@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    expected: str = request.app.state.cfg.api_key
    if not expected:
        return await call_next(request)

    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        if secrets.compare_digest(auth[7:].strip(), expected):
            return await call_next(request)

    return JSONResponse(
        status_code=401,
        content={"detail": "invalid or missing api key"},
        headers={"WWW-Authenticate": 'Bearer realm="embedding-service"'},
    )


# --- Health ---------------------------------------------------------------

@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict:
    return {"ok": True}


# --- /embed handler -------------------------------------------------------

@app.post("/embed")
async def embed(
    body: EmbedRequest,
    request: Request,
    background: BackgroundTasks,
) -> list[list[float]]:
    cfg = request.app.state.cfg
    try:
        return await asyncio.wait_for(
            _embed_handler(body, request, background),
            timeout=cfg.request_budget_s,
        )
    except asyncio.TimeoutError:
        _REQUESTS.labels(status="504").inc()
        raise HTTPException(status_code=504, detail="request budget exhausted")


async def _embed_handler(
    body: EmbedRequest,
    request: Request,
    background: BackgroundTasks,
) -> list[list[float]]:
    cfg = request.app.state.cfg
    cache: EmbeddingCache = request.app.state.cache
    tei: TEIClient = request.app.state.tei

    # 1. Normalise inputs to list[str].
    inputs = body.inputs if isinstance(body.inputs, list) else [body.inputs]
    n = len(inputs)
    if n == 0:
        _REQUESTS.labels(status="400").inc()
        raise HTTPException(status_code=400, detail="`inputs` must not be empty")
    if n > cfg.max_inputs_per_request:
        _REQUESTS.labels(status="413").inc()
        raise HTTPException(
            status_code=413,
            detail=(
                f"too many inputs in one request ({n} > {cfg.max_inputs_per_request})"
            ),
        )

    # 2. Cheap NUL-field-count validation up front. Rendering only on miss.
    for i, text in enumerate(inputs):
        field_count = text.count("\x00") + 1
        if field_count != N_FIELDS:
            _REQUESTS.labels(status="400").inc()
            raise HTTPException(
                status_code=400,
                detail=(
                    f"input[{i}]: expected exactly {N_FIELDS} NUL-separated "
                    f"fields, got {field_count}"
                ),
            )

    # 3. Hash all inputs.
    hashes = [article_hash(t) for t in inputs]

    # 4. Cache read (50 ms, falls back to all-miss on failure).
    cached = await cache.mget(hashes)

    # 5. Inflight gate (post-MGET so 100%-hit requests don't get 429'd).
    if cached.count(None) > 0:
        if request.app.state.inflight >= cfg.max_inflight:
            _REQUESTS.labels(status="429").inc()
            raise HTTPException(
                status_code=429,
                detail="embedding-service at concurrency limit",
                headers={"Retry-After": f"{cfg.retry_after_s:.2f}"},
            )
        request.app.state.inflight += 1
    else:
        request.app.state.inflight += 1  # cache-hit-only path still tracked

    _INFLIGHT.set(request.app.state.inflight)

    try:
        return await _serve(
            inputs, hashes, cached, body.truncate,
            cache=cache, tei=tei, background=background,
            hit_ratio=request.app.state.hit_ratio,
        )
    finally:
        request.app.state.inflight -= 1
        _INFLIGHT.set(request.app.state.inflight)


async def _serve(
    inputs: list[str],
    hashes: list[str],
    cached: list[bytes | None],
    truncate: bool,
    *,
    cache: EmbeddingCache,
    tei: TEIClient,
    background: BackgroundTasks,
    hit_ratio: _HitRatioEWMA,
) -> list[list[float]]:
    """Inner handler — runs under the inflight counter + request budget."""
    n = len(inputs)
    miss_indices: list[int] = []
    miss_texts: list[str] = []
    miss_hashes: list[str] = []
    miss_hash_seen: dict[str, int] = {}  # first miss-index for each unique hash

    for i, (text, h, v) in enumerate(zip(inputs, hashes, cached)):
        if v is None:
            if h not in miss_hash_seen:
                miss_hash_seen[h] = len(miss_texts)
                miss_texts.append(render_from_nul(text))
                miss_hashes.append(h)
            miss_indices.append(i)

    hits = n - len(miss_indices)
    cache.stats.hits += hits
    cache.stats.misses += len(miss_indices)
    hit_ratio.update(hits, n)

    new_vectors: np.ndarray | None = None
    if miss_texts:
        sem_wait_before = tei.semaphore_wait_total_s
        new_vectors = await tei.embed(miss_texts, truncate=truncate)
        sem_wait_delta = tei.semaphore_wait_total_s - sem_wait_before
        # Observation per request — total wait across all chunks. Good
        # enough for tuning the queue depth; per-chunk granularity would
        # need a counter on the client itself.
        _TEI_SEM_WAIT.observe(sem_wait_delta)
        # Schedule cache write-back (one entry per unique miss hash).
        new_bytes = {
            h: vec_bytes_from_fp16(new_vectors[j])
            for j, h in enumerate(miss_hashes)
        }
        background.add_task(cache.mset, new_bytes)

    # Stitch in input order: hits from cache, misses from new_vectors.
    out: list[list[float]] = []
    for i, (h, v) in enumerate(zip(hashes, cached)):
        if v is not None:
            out.append(fp16_from_bytes(v).tolist())
        else:
            # new_vectors is indexed by the unique-miss order
            j = miss_hash_seen[h]
            out.append(new_vectors[j].tolist())  # type: ignore[index]

    _REQUESTS.labels(status="200").inc()
    return out
