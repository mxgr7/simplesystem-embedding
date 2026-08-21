import asyncio
import logging
import math
import secrets
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import REGISTRY, Counter, Gauge
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_fastapi_instrumentator import Instrumentator

from backend_pool import BackendPool, HealthPolicy, NoHealthyBackendError
from cache import SparseCache
from codec import pack_sparse, unpack_sparse
from config import Config
from constants import VOCAB_SIZE, model_metadata
from fold_de import fold_de
from hashing import input_hash
from rendering import canonical_input, render_from_nul
from schemas import AddBackendRequest, EmbedRequest, PatchBackendRequest
from text import normalize_text


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

PUBLIC_PATHS = {"/healthz", "/readyz", "/metadata", "/metrics", "/docs", "/openapi.json"}
REQUESTS = Counter("splade_service_requests_total", "Requests", ("status",))
CACHE_HITS = Counter("splade_service_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("splade_service_cache_misses_total", "Cache misses")
INFLIGHT = Gauge("splade_service_inflight", "Miss requests being processed")
MAX_QUERY_CHARS = 4096
LOG_THROTTLE_S = 60.0
_throttled_last = {}
_throttled_suppressed = {}


def log_throttled(key, message, *args):
    """Log at most once per minute per key, carrying the suppressed count.

    A condition that affects every request has to be logged per *state*, not per
    request: on the dense side the same condition wrote 268k tracebacks and
    2.3 GB of container log over one outage, which buried the single line that
    said what had happened.
    """
    now = time.monotonic()
    last = _throttled_last.get(key, float("-inf"))
    if now - last < LOG_THROTTLE_S:
        _throttled_suppressed[key] = _throttled_suppressed.get(key, 0) + 1
        return
    suppressed = _throttled_suppressed.pop(key, 0)
    _throttled_last[key] = now
    if suppressed:
        log.warning(
            message + " (%d more in the last %.0fs)",
            *args, suppressed, LOG_THROTTLE_S,
        )
    else:
        log.warning(message, *args)


class BackendHealthCollector:
    """Exports `splade_service_backend_healthy`, one sample per registered backend.

    A scrape-time collector rather than a Gauge written from the probe loop, for two reasons.
    Backends come and go through /admin/backends, so the label set is dynamic and a Gauge would
    keep exporting a stale child for a backend that has been removed. And a Gauge is only as
    fresh as whatever writes it, whereas this reads the pool on every scrape.

    That second property is narrower than it looks, and this docstring used to overclaim it: the
    boolean being read is still written only by `_probe_loop`, so a loop that has died freezes it
    and the collector re-exports the stale value just as faithfully as a Gauge would.
    `splade_service_probe_loop_last_iteration_timestamp` below is what actually catches that, and
    it is the metric `SpladeProbeLoopStalled` alerts on.

    The counters already here cannot cover this. `splade_service_requests_total{status="504"}`
    only rises when a client asks for something, so a backend that dies while the indexer is
    idle is invisible until the next batch -- and with SPLADE_REQUIRED=true that batch does not
    degrade, it retries and dead-letters. Shape matches the dense side's
    `embedding_service_tei_backend_healthy` so both stacks alert the same way.
    """

    def __init__(self, app):
        self.app = app

    def collect(self):
        healthy = GaugeMetricFamily(
            "splade_service_backend_healthy",
            "1 if the backend passed its most recent verify probe, 0 otherwise",
            labels=["backend", "url"],
        )
        draining = GaugeMetricFamily(
            "splade_service_backend_draining",
            "1 if the backend is draining (weight 0, finishing in-flight work)",
            labels=["backend", "url"],
        )
        generation = GaugeMetricFamily(
            "splade_service_backend_client_generation",
            "How many times this backend's HTTP client has been recycled",
            labels=["backend", "url"],
        )
        pool = getattr(self.app.state, "pool", None)
        # `.get` on everything but the id: a KeyError in a collector makes /metrics return 500,
        # which Prometheus reads as the whole job being down -- a false alarm that would mask
        # the real ones. Degrading to a blank label is strictly better than that.
        for snapshot in pool.snapshots() if pool is not None else []:
            labels = [snapshot["id"], snapshot.get("url", "")]
            healthy.add_metric(labels, float(snapshot.get("healthy", False)))
            draining.add_metric(labels, float(snapshot.get("draining", False)))
            generation.add_metric(labels, float(snapshot.get("client_generation", 0)))
        yield healthy
        yield draining
        yield generation

        # The freshness of everything above. `healthy` is written only by the
        # probe loop, so a loop that dies leaves it frozen at its last value and
        # the outage reads as steady state -- a scrape-time collector re-exports
        # that stale value just as faithfully as a push gauge would. This is the
        # one signal a frozen loop cannot fake.
        stats = pool.stats() if pool is not None else {}
        last_iteration = GaugeMetricFamily(
            "splade_service_probe_loop_last_iteration_timestamp",
            "Unix time of the last completed probe round; stale = recovery is "
            "not running",
        )
        last_iteration.add_metric(
            [], float(stats.get("probe_loop_last_iteration_at", 0.0))
        )
        yield last_iteration
        errors = CounterMetricFamily(
            "splade_service_probe_loop_errors",
            "Probe rounds that timed out or raised (the loop keeps going)",
        )
        errors.add_metric([], float(stats.get("probe_loop_errors", 0)))
        yield errors


@asynccontextmanager
async def lifespan(app):
    config = Config()
    cache = SparseCache(
        config.kvrocks_url,
        config.cache_read_timeout_s,
        config.cache_connections,
    )
    pool = BackendPool(
        HealthPolicy(
            unhealthy_after=config.unhealthy_after,
            probe_interval_s=config.probe_interval_s,
            probe_timeout_s=config.probe_timeout_s,
            probe_round_timeout_s=config.probe_round_timeout_s,
            half_open_interval_s=config.half_open_interval_s,
            client_recycle_after_s=config.client_recycle_after_s,
            pool_timeout_recycle_after=config.pool_timeout_recycle_after,
        )
    )
    for url in config.backend_urls:
        # `require_verify=False`: a backend that is down or still loading its
        # checkpoint at boot must not take the frontend down with it. It is
        # registered unhealthy, /readyz stays 503, and the probe loop brings it
        # in. POST /admin/backends keeps the strict form.
        await pool.add(
            url,
            max_concurrency=config.backend_pool_concurrency,
            max_client_batch=config.backend_max_client_batch,
            timeout_s=config.backend_timeout_s,
            api_key=config.backend_api_key,
            require_verify=False,
        )
    pool.start()
    app.state.config = config
    app.state.cache = cache
    app.state.pool = pool
    app.state.inflight = 0
    # Registered here, not at import, so repeated app construction in tests does not collide on
    # the default registry -- and unregistered on the way out for the same reason.
    collector = BackendHealthCollector(app)
    REGISTRY.register(collector)
    try:
        yield
    finally:
        REGISTRY.unregister(collector)
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


def no_backend_error(config, path, exc):
    """503 instead of the unhandled 500 this used to be.

    Unhandled, `NoHealthyBackendError` surfaced as a bare 500: uncounted by
    `splade_service_requests_total` and invisible to every alert built on it --
    `SpladeGatewayTimeouts` watches 504, `SpladeNoBackendAtAll` only fires when
    the metric disappears entirely. 503 is also the honest status, and it tells
    the indexer to retry rather than dead-letter.
    """
    REQUESTS.labels("503").inc()
    log_throttled("no_backend", "%s rejected: %s", path, exc)
    return HTTPException(
        status_code=503,
        detail="no healthy SPLADE backend available",
        headers={"Retry-After": f"{config.retry_after_s:.0f}"},
    )


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


@app.get("/metadata")
async def metadata(request: Request):
    """Which checkpoint this service is actually serving.

    A client that only calls /embed cannot otherwise tell. That matters more than it sounds: this
    service is deliberately able to serve any checkpoint through one contract (`SPLADE_MODEL_ID` and
    `SPLADE_MODEL_SHA256` are env-overridable on both sides so a second checkpoint gets its own cache
    namespace), and `Backend.verify` only ensures the frontend and its backends *agree* — a pool
    consistently labelled as the wrong checkpoint is perfectly healthy and answers /readyz with 200.
    An indexer writing `spladeModelVersion` from its own config has no way to notice.

    So: `expected` is what this frontend believes it serves, `backends` is what each verified
    backend reports. Callers should assert on both. Public, like /healthz and /readyz — it discloses
    a model name and two digests, and requiring a key would put the check behind the very
    misconfiguration it exists to catch.
    """
    return {
        "expected": model_metadata(),
        "backends": request.app.state.pool.contracts(),
    }


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
    except NoHealthyBackendError as exc:
        raise no_backend_error(config, "/embed", exc)


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
    except NoHealthyBackendError as exc:
        raise no_backend_error(config, "/embed-query", exc)


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
