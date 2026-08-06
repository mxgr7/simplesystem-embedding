import asyncio
import logging
import math
import secrets
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from backend_pool import BackendPool
from cache import SparseCache
from codec import pack_sparse, unpack_sparse
from config import Config
from constants import VOCAB_SIZE
from fold_de import fold_de
from hashing import input_hash
from rendering import canonical_input, render_from_nul
from schemas import AddBackendRequest, EmbedRequest, PatchBackendRequest
from text import normalize_text


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

PUBLIC_PATHS = {"/healthz", "/readyz", "/metrics", "/docs", "/openapi.json"}
REQUESTS = Counter("splade_service_requests_total", "Requests", ("status",))
CACHE_HITS = Counter("splade_service_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("splade_service_cache_misses_total", "Cache misses")
INFLIGHT = Gauge("splade_service_inflight", "Miss requests being processed")
MAX_QUERY_CHARS = 4096


@asynccontextmanager
async def lifespan(app):
    config = Config()
    cache = SparseCache(
        config.kvrocks_url,
        config.cache_read_timeout_s,
        config.cache_connections,
    )
    pool = BackendPool(config.probe_interval_s)
    for url in config.backend_urls:
        await pool.add(url, api_key=config.backend_api_key)
    pool.start()
    app.state.config = config
    app.state.cache = cache
    app.state.pool = pool
    app.state.inflight = 0
    try:
        yield
    finally:
        await pool.aclose()
        await cache.aclose()


app = FastAPI(title="SPLADE Service", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def authenticate(request, call_next):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    config = request.app.state.config
    expected = (
        config.admin_api_key or config.api_key
        if request.url.path.startswith("/admin")
        else config.api_key
    )
    if not expected:
        return await call_next(request)
    value = request.headers.get("authorization", "")
    if value[:7].lower() == "bearer " and secrets.compare_digest(
        value[7:].strip(), expected
    ):
        return await call_next(request)
    return JSONResponse(status_code=401, content={"detail": "invalid api key"})


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/readyz")
async def readyz(request: Request):
    cache_ok = False
    try:
        cache_ok = bool(await asyncio.wait_for(request.app.state.cache.ping(), 2))
    except Exception:
        pass
    backend_ok = request.app.state.pool.ready()
    ready = cache_ok and backend_ok
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "ready": ready,
            "checks": {"cache": cache_ok, "backend": backend_ok},
        },
    )


@app.get("/admin/backends")
async def list_backends(request: Request):
    return request.app.state.pool.snapshots()


@app.post("/admin/backends")
async def add_backend(body: AddBackendRequest, request: Request):
    try:
        return await request.app.state.pool.add(
            body.url,
            body.weight,
            body.max_concurrency,
            body.max_client_batch,
            body.timeout_s,
            body.api_key,
        )
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.patch("/admin/backends/{backend_id}")
async def patch_backend(backend_id, body: PatchBackendRequest, request: Request):
    try:
        return await request.app.state.pool.set_weight(backend_id, body.weight)
    except KeyError:
        raise HTTPException(status_code=404, detail="backend not found")


@app.delete("/admin/backends/{backend_id}", status_code=202)
async def delete_backend(backend_id, request: Request):
    try:
        return await request.app.state.pool.remove(backend_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="backend not found")


@app.post("/embed")
async def embed(body: EmbedRequest, request: Request, background: BackgroundTasks):
    config = request.app.state.config
    try:
        return await asyncio.wait_for(
            _embed(body, request, background),
            timeout=config.request_budget_s,
        )
    except asyncio.TimeoutError:
        REQUESTS.labels("504").inc()
        raise HTTPException(status_code=504, detail="request budget exhausted")


@app.post("/embed-query")
async def embed_query(body: EmbedRequest, request: Request):
    config = request.app.state.config
    try:
        return await asyncio.wait_for(
            _embed_query(body, request),
            timeout=config.request_budget_s,
        )
    except asyncio.TimeoutError:
        REQUESTS.labels("504").inc()
        raise HTTPException(status_code=504, detail="request budget exhausted")


async def _embed_query(body, request):
    config = request.app.state.config
    inputs = body.inputs if isinstance(body.inputs, list) else [body.inputs]
    if not inputs:
        raise HTTPException(status_code=400, detail="inputs must not be empty")
    if len(inputs) > config.max_inputs:
        raise HTTPException(status_code=413, detail="too many inputs")

    texts = [fold_de(normalize_text(value)) for value in inputs]
    if any(not value for value in texts):
        REQUESTS.labels("400").inc()
        raise HTTPException(
            status_code=400,
            detail="inputs must not contain empty queries",
        )
    if any(len(value) > MAX_QUERY_CHARS for value in texts):
        REQUESTS.labels("413").inc()
        raise HTTPException(status_code=413, detail="query input is too long")
    if request.app.state.inflight >= config.max_inflight:
        REQUESTS.labels("429").inc()
        raise HTTPException(
            status_code=429,
            detail="SPLADE service at concurrency limit",
            headers={"Retry-After": "1"},
        )

    request.app.state.inflight += 1
    INFLIGHT.set(request.app.state.inflight)
    try:
        vectors = await request.app.state.pool.encode(texts, document=False)
        if len(vectors) != len(texts):
            raise ValueError("backend response cardinality does not match inputs")
        for vector in vectors:
            if not isinstance(vector, dict):
                raise ValueError("backend query vector must be an object")
            for token, weight in vector.items():
                try:
                    token_id = int(token)
                except (TypeError, ValueError):
                    raise ValueError(f"invalid query token {token!r}") from None
                if (
                    str(token_id) != str(token)
                    or token_id < 0
                    or token_id >= VOCAB_SIZE
                    or isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(weight)
                    or weight <= 0
                ):
                    raise ValueError(f"invalid query sparse entry {token!r}: {weight!r}")
        REQUESTS.labels("200").inc()
        return vectors
    finally:
        request.app.state.inflight -= 1
        INFLIGHT.set(request.app.state.inflight)


async def _embed(body, request, background):
    config = request.app.state.config
    inputs = body.inputs if isinstance(body.inputs, list) else [body.inputs]
    if not inputs:
        raise HTTPException(status_code=400, detail="inputs must not be empty")
    if len(inputs) > config.max_inputs:
        raise HTTPException(status_code=413, detail="too many inputs")
    try:
        canonical = [canonical_input(value) for value in inputs]
    except ValueError as exc:
        REQUESTS.labels("400").inc()
        raise HTTPException(status_code=400, detail=str(exc))

    hashes = [input_hash(value) for value in canonical]
    cached = await request.app.state.cache.mget(hashes)
    missing = sum(value is None for value in cached)
    CACHE_HITS.inc(len(inputs) - missing)
    CACHE_MISSES.inc(missing)

    if missing and request.app.state.inflight >= config.max_inflight:
        REQUESTS.labels("429").inc()
        raise HTTPException(
            status_code=429,
            detail="SPLADE service at concurrency limit",
            headers={"Retry-After": "1"},
        )

    request.app.state.inflight += int(bool(missing))
    INFLIGHT.set(request.app.state.inflight)
    try:
        unique_hashes = []
        texts = []
        seen = set()
        for value, value_hash, packed in zip(canonical, hashes, cached):
            if packed is None and value_hash not in seen:
                seen.add(value_hash)
                unique_hashes.append(value_hash)
                texts.append(render_from_nul(value))

        vectors = (
            await request.app.state.pool.encode(texts, document=True)
            if texts
            else []
        )
        new_values = {
            value_hash: pack_sparse(vector)
            for value_hash, vector in zip(unique_hashes, vectors)
        }
        if new_values:
            background.add_task(request.app.state.cache.mset, new_values)

        output = []
        for value_hash, packed in zip(hashes, cached):
            output.append(unpack_sparse(packed or new_values[value_hash]))
        REQUESTS.labels("200").inc()
        return output
    finally:
        request.app.state.inflight -= int(bool(missing))
        INFLIGHT.set(request.app.state.inflight)
