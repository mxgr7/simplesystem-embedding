"""The HTTP surface. `uvicorn ceserve.app:app`.

MXG-144. Deliberately one process: no frontend/backend split, no backend pool,
no KVRocks cache. Reasoning is in README.md; the short version is that this
serves one GPU to one caller inside a 150 ms budget, and every one of those
mechanisms buys latency or a failure mode in exchange for flexibility nothing
here needs. `/metadata`'s field names are kept in the pooled shape so a later
split is a refactor rather than a wire change.
"""
import asyncio
import logging
import secrets
import time
from contextlib import asynccontextmanager

import numpy as np
import orjson
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import REGISTRY
from prometheus_fastapi_instrumentator import Instrumentator

from ceserve.config import Config
from ceserve.constants import (
    HEAD_EXTRA,
    MODEL_ID,
    MODEL_SHA256,
    QUERY_CONTRACT,
    TOKENIZER_VERSION,
)
from ceserve.fold_de import fold_de
from ceserve.metrics import (
    ASSEMBLE_MS,
    CANDIDATES,
    CANDIDATES_PER_REQUEST,
    FORWARD_MS,
    INFLIGHT,
    PADDED_WIDTH,
    REQUESTS,
    TOTAL_MS,
    CeRuntimeCollector,
)
from ceserve.schemas import RerankRequest, RerankResponse
from ceserve.scorer import CrossEncoderScorer, ce_score
from ceserve.splice import (
    TokenDecodeError,
    clamp_max_len,
    decode_token_ids,
    encode_query,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PUBLIC_PATHS = {"/healthz", "/readyz", "/metadata", "/metrics", "/docs",
                "/openapi.json"}

SKIP_NO_TOKENS = "no_tokens"
SKIP_DECODE = "decode_failed"
SKIP_VERSION = "tokenizer_version_mismatch"
_SKIP_METRIC = {
    SKIP_NO_TOKENS: "skipped_no_tokens",
    SKIP_DECODE: "skipped_decode",
    SKIP_VERSION: "skipped_version",
}


class RequestError(Exception):
    """A malformed REQUEST — as opposed to a bad candidate, which is skipped."""

    def __init__(self, detail, status=400, code="invalid_request"):
        super().__init__(detail)
        self.detail = detail
        self.status = status
        self.code = code


def _json(payload, status=200):
    """Serialize with orjson and hand FastAPI finished bytes.

    Declaring a `response_model` instead would make FastAPI validate every
    result row through pydantic on the way out — the same per-item cost the
    request side avoids (see schemas.py), on a response that carries up to 256
    rows.
    """
    return Response(content=orjson.dumps(payload), status_code=status,
                    media_type="application/json")


def _error(status, code, detail):
    REQUESTS.labels(str(status)).inc()
    return _json({"error": code, "detail": detail}, status=status)


@asynccontextmanager
async def lifespan(app):
    config = Config()
    scorer = CrossEncoderScorer(config)
    log.info("loaded %s on %s (%s), max_len=%d, contract=%s",
             MODEL_ID, scorer.device, scorer.dtype_name, scorer.max_len,
             scorer.serving_contract)
    # The warmup forward runs BEFORE /readyz can go green. MXG-111: splade kept
    # /health, /readyz and /embed green while /encode-packed 500'd, because
    # nothing exercised the path that was broken.
    shape = scorer.warmup()
    log.info("warmup forward ok %s", shape)

    app.state.config = config
    app.state.scorer = scorer
    app.state.inflight = 0
    # Registered here, not at import, so repeated app construction in tests does
    # not collide on the default registry — and unregistered on the way out for
    # the same reason.
    collector = CeRuntimeCollector(app)
    REGISTRY.register(collector)
    try:
        yield
    finally:
        REGISTRY.unregister(collector)


app = FastAPI(title="CE Service", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def authenticate(request, call_next):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    expected = request.app.state.config.api_key
    if not expected:
        return await call_next(request)
    value = request.headers.get("authorization", "")
    if value[:7].lower() == "bearer " and secrets.compare_digest(
        value[7:].strip(), expected
    ):
        return await call_next(request)
    return JSONResponse(status_code=401,
                        content={"error": "unauthorized", "detail": "invalid api key"})


@app.get("/healthz")
async def healthz():
    """Liveness only. Says nothing about the model — that is /readyz."""
    return {"ok": True}


@app.get("/readyz")
async def readyz(request: Request):
    scorer = getattr(request.app.state, "scorer", None)
    checks = {
        "model": scorer is not None,
        "warmup": bool(scorer is not None and scorer.warmed_up),
        "degraded": bool(scorer is not None and scorer.degraded),
    }
    ready = checks["model"] and checks["warmup"] and not checks["degraded"]
    return _json({"ready": ready, "checks": checks}, status=200 if ready else 503)


@app.get("/metadata")
async def metadata(request: Request):
    """Which checkpoint, at which width, in which dtype, through which splice.

    Public, for the same reason `splade-service`'s is: a client that only calls
    /rerank cannot otherwise tell what it is talking to, and requiring a key
    would put the check behind the very misconfiguration it exists to catch. It
    discloses a model name, two digests and a set of integers.

    `serving_contract` is the field callers should assert on. `model_sha256`
    alone cannot express HOW a score was produced — the same checkpoint at
    fp16/L192 and fp32/L256 returns different numbers, and MXG-111 found exactly
    that hole in the SPLADE keyspace.
    """
    scorer = request.app.state.scorer
    config = request.app.state.config
    payload = scorer.metadata()
    payload.update({
        "max_inputs_per_request": config.max_inputs,
        "max_inflight": config.max_inflight,
        "request_budget_s": config.request_budget_s,
    })
    return payload


# ------------------------------------------------------------------ rerank --

def _parse_request(payload, config, scorer):
    """Validate the REQUEST. Per-candidate problems are handled in `_partition`."""
    if not isinstance(payload, dict):
        raise RequestError("body must be a JSON object")
    if "segment" in payload:
        # The retired pre-MXG-177 contract. Rejected on PRESENCE of the key —
        # a null value included — because a caller that still sends it was built
        # against the prefixed contract and its scores would be silently wrong.
        # Logged here (not in `_error`, which deliberately does not log) with no
        # query text and no candidate contents.
        log.error(
            "contract violation: request carries the retired 'segment' key; "
            "this service implements query contract %s (MXG-177)",
            QUERY_CONTRACT,
        )
        raise RequestError(
            "the 'segment' key is not part of query contract "
            f"{QUERY_CONTRACT}; send the raw query only"
        )
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise RequestError("query must be a non-empty string")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RequestError("candidates must be a non-empty array")
    if len(candidates) > config.max_inputs:
        raise RequestError(
            f"{len(candidates)} candidates exceeds MAX_INPUTS_PER_REQUEST="
            f"{config.max_inputs}",
            status=413, code="too_many_candidates")
    try:
        max_len = clamp_max_len(payload.get("max_len"), scorer.max_len)
    except ValueError as exc:
        raise RequestError(str(exc)) from None

    seen = set()
    for entry in candidates:
        if not isinstance(entry, dict):
            raise RequestError("each candidate must be a JSON object")
        cid = entry.get("id")
        if not isinstance(cid, str) or not cid:
            raise RequestError("each candidate needs a non-empty string id")
        if cid in seen:
            # The caller's window is unique by ArticleId; a duplicate means the
            # window was built wrong, and scoring it twice would hide that.
            raise RequestError(f"duplicate candidate id {cid!r}")
        seen.add(cid)
    return query, max_len, candidates


def _partition(candidates, expected_version, config):
    """(scored-side arrays, skipped entries).

    A single corrupt blob is a `skipped` entry, never a 400: one bad `_source`
    must not fail a 120-candidate window. Wholesale decode failure IS a contract
    break, so it escalates — mirroring the query service's own
    `max-missing-content-ratio`.
    """
    ids, arts, counts, skipped, decode_failures = [], [], [], [], 0
    for entry in candidates:
        cid = entry["id"]
        blob = entry.get("tokens_b64")
        version = entry.get("tokenizer_version")
        if blob is None or blob == "":
            skipped.append({"id": cid, "reason": SKIP_NO_TOKENS,
                            "detail": "candidate carries no ceTokenIds"})
            continue
        # Version BEFORE decode: ids from another vocabulary decode perfectly
        # and are exactly what must never reach the model.
        if version != expected_version:
            skipped.append({
                "id": cid, "reason": SKIP_VERSION,
                "detail": f"got {version!r}, expected {expected_version!r}"})
            continue
        try:
            decoded = decode_token_ids(blob, entry.get("token_count"))
        except TokenDecodeError as exc:
            decode_failures += 1
            skipped.append({"id": cid, "reason": SKIP_DECODE,
                            "detail": str(exc)})
            continue
        ids.append(cid)
        arts.append(decoded)
        counts.append(int(decoded.shape[0]))

    if candidates and decode_failures / len(candidates) > config.max_decode_failure_ratio:
        raise RequestError(
            f"{decode_failures}/{len(candidates)} candidates failed to decode "
            f"(> CE_MAX_DECODE_FAILURE_RATIO={config.max_decode_failure_ratio}); "
            "that is a wire-format break, not data noise")
    return ids, arts, counts, skipped


def _build_response(scorer, query_ids, ids, counts, probs, skipped, max_len,
                    n_input, padded_width):
    nq = int(min(query_ids.shape[0], max(1, max_len - HEAD_EXTRA - 1)))
    budget = max(1, max_len - nq - HEAD_EXTRA)
    scores = ce_score(probs) if len(ids) else np.zeros(0)
    results = [
        {
            "id": cid,
            "ce_score": float(scores[i]),
            "ce_p_e": float(probs[i][0]),
            "ce_p_s": float(probs[i][1]),
            "ce_p_c": float(probs[i][2]),
            "ce_p_i": float(probs[i][3]),
            "article_token_count": counts[i],
            "truncated": counts[i] > budget,
        }
        for i, cid in enumerate(ids)
    ]
    return {
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "tokenizer_version": TOKENIZER_VERSION,
        "serving_contract": scorer.serving_contract,
        "query_contract": QUERY_CONTRACT,
        "max_len": max_len,
        "n_input": n_input,
        "n_scored": len(results),
        "n_skipped": len(skipped),
        "query_token_count": nq,
        "padded_width": padded_width,
        "results": results,
        "skipped": skipped,
    }


@app.post(
    "/rerank",
    # The models describe the contract for /docs and are NOT bound to the
    # handler: FastAPI would then validate the request through pydantic (~6 ms
    # of a 150 ms budget at k=120, scaling linearly to the 256 cap) and would
    # answer 422 for anything malformed, taking the error contract away from
    # `_parse_request`. Same arrangement `cross_encoder_serve` uses and states.
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": RerankRequest.model_json_schema()}
            },
        }
    },
    responses={200: {"model": RerankResponse}},
)
async def rerank(request: Request):
    """Score (query, article) pairs from the article ids the indexer stored.

    ⚠️ `asyncio.wait_for` around `asyncio.to_thread` does NOT cancel the running
    thread. REQUEST_BUDGET_S is a RESPONSE deadline, not a work deadline: on a
    timeout the caller gets a 504 and the GPU keeps going. Work is bounded by
    MAX_INPUTS_PER_REQUEST (which caps one forward) and MAX_INFLIGHT (which caps
    the queue). Do NOT raise MAX_INFLIGHT believing the budget protects you.
    """
    config = request.app.state.config
    started = time.perf_counter()
    try:
        return await asyncio.wait_for(_rerank(request, config),
                                      timeout=config.request_budget_s)
    except asyncio.TimeoutError:
        return _error(504, "request_budget_exhausted",
                      f"exceeded REQUEST_BUDGET_S={config.request_budget_s}")
    except RequestError as exc:
        return _error(exc.status, exc.code, exc.detail)
    finally:
        TOTAL_MS.observe(time.perf_counter() - started)


async def _rerank(request, config):
    scorer = request.app.state.scorer
    if scorer is None or not scorer.warmed_up or scorer.degraded:
        raise RequestError("model is not ready", status=503,
                           code="model_not_ready")

    raw = await request.body()
    try:
        payload = orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        raise RequestError(f"body is not valid JSON: {exc}") from None

    query, max_len, candidates = _parse_request(payload, config, scorer)

    folded = fold_de(query)
    if not folded.strip():
        # A nonblank raw query whose folded form carries nothing to encode
        # (combining marks alone, or characters that decompose to whitespace).
        # A query-level decline, not an error: HTTP 200, no inference, no log.
        # Both arrays are empty BY DESIGN — the one carve-out to the "every
        # input id comes back exactly once" contract; callers branch on
        # `declined_reason`. The ticket names the exact reason string.
        REQUESTS.labels("200").inc()
        return _json({
            "model_id": MODEL_ID,
            "model_sha256": MODEL_SHA256,
            "tokenizer_version": TOKENIZER_VERSION,
            "serving_contract": scorer.serving_contract,
            "query_contract": QUERY_CONTRACT,
            "declined_reason": "empty_folded_query",
            "max_len": max_len,
            "n_input": len(candidates),
            "n_scored": 0,
            "n_skipped": 0,
            "results": [],
            "skipped": [],
        })

    if request.app.state.inflight >= config.max_inflight:
        # A fast refusal beats a queue: the caller degrades to upstream order on
        # a refusal but WAITS on a queue, and waiting is what blows the budget.
        raise RequestError("CE service at concurrency limit", status=429,
                           code="at_capacity")

    t0 = time.perf_counter()
    ids, arts, counts, skipped = _partition(
        candidates, config.tokenizer_version, config)
    ASSEMBLE_MS.observe(time.perf_counter() - t0)

    query_ids = encode_query(scorer.tokenizer, folded)

    padded_width = 0
    probs = np.zeros((0, 4), dtype=np.float32)
    if ids:
        request.app.state.inflight += 1
        INFLIGHT.set(request.app.state.inflight)
        t1 = time.perf_counter()
        try:
            probs, stats = await asyncio.to_thread(
                scorer.score, query_ids, arts, max_len)
            padded_width = stats["padded_width"]
        except Exception as exc:
            # A CUDA OOM or an illegal memory access poisons the context, and a
            # process that keeps answering afterwards serves garbage. Mark
            # degraded so /readyz 503s and the orchestrator restarts us.
            scorer.degraded = True
            log.exception("inference failed; marking the service degraded")
            raise RequestError(f"inference failed: {exc}", status=500,
                               code="inference_failed") from None
        finally:
            FORWARD_MS.observe(time.perf_counter() - t1)
            request.app.state.inflight -= 1
            INFLIGHT.set(request.app.state.inflight)
        PADDED_WIDTH.observe(padded_width)

    response = _build_response(scorer, query_ids, ids, counts, probs, skipped,
                               max_len, len(candidates), padded_width)
    CANDIDATES_PER_REQUEST.observe(len(candidates))
    CANDIDATES.labels("scored").inc(len(ids))
    for entry in skipped:
        CANDIDATES.labels(_SKIP_METRIC[entry["reason"]]).inc()
    REQUESTS.labels("200").inc()
    if payload.get("debug"):
        response["timings"] = {
            "decode_ms": round((time.perf_counter() - t0) * 1000, 3),
            "chunks": 0 if not ids else int(np.ceil(len(ids) / scorer.chunk)),
        }
    return _json(response)
