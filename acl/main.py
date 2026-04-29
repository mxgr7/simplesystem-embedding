"""Article-search ACL service — narrowed legacy contract → ftsearch.

A2 wires the request mapper + a real call to ftsearch. A3 will wrap
the response in the legacy envelope; today the raw ftsearch response
flows through (which already mostly matches per F2's contract
alignment). A4 categorises errors; A5 layers in tracing, retries,
and RED metrics — the same operational pieces the search-api side
got from F7.

Default ports per spec §3 / packet A1: app on **8081**, metrics on
**9090**. Both expose `security: []` — internal service, no per-request
auth (§9 #7).

The OpenAPI is the contract source of truth: changes land in
`acl/openapi.yaml` first, then both the request mapper (A2) and the
response mapper (A3) implement against it.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acl.clients.ftsearch import FtsearchClient
from acl.mapping.request import map_request
from acl.mapping.response import map_response
from acl.models import LegacySearchRequest

BASE_DIR = Path(__file__).resolve().parent
OPENAPI_YAML_PATH = BASE_DIR / "openapi.yaml"
_OPENAPI_YAML_TEXT = OPENAPI_YAML_PATH.read_text()
_OPENAPI_SPEC = yaml.safe_load(_OPENAPI_YAML_TEXT)

# --- error envelope (§3.1) -----------------------------------------------

class _ErrorBody(BaseModel):
    """Legacy error envelope — `{message, details, timestamp}`. Must
    match exactly so existing next-gen clients don't break."""
    model_config = ConfigDict(extra="forbid")
    message: str
    details: list[str] = Field(default_factory=list)
    timestamp: str


def _error(status: int, message: str, *, details: list[str] | None = None) -> JSONResponse:
    """Build the legacy error envelope. `timestamp` is ISO-8601 UTC."""
    return JSONResponse(
        status_code=status,
        content=_ErrorBody(
            message=message,
            details=details or [],
            timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
        ).model_dump(),
    )


# --- FastAPI lifecycle ---------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # A2: open the ftsearch HTTP client at startup so it's shared
    # across all request handlers (httpx connection pooling).
    # `FTSEARCH_URL` defaults to the docker-compose service name +
    # the search-api port. `MILVUS_ARTICLES_COLLECTION` is the alias
    # ftsearch routes against — defaults to `articles` per F9 alias
    # workflow.
    app.state.ftsearch_url = os.environ.get(
        "FTSEARCH_URL", "http://search-api:8001",
    )
    app.state.ftsearch = FtsearchClient(
        app.state.ftsearch_url,
        default_collection=os.environ.get(
            "MILVUS_ARTICLES_COLLECTION", "articles",
        ),
    )
    try:
        yield
    finally:
        await app.state.ftsearch.aclose()


app = FastAPI(lifespan=lifespan, title="article-search-acl")


def _custom_openapi() -> dict:
    return _OPENAPI_SPEC


app.openapi = _custom_openapi  # type: ignore[method-assign]


@app.get("/openapi.yaml", include_in_schema=False)
async def openapi_yaml() -> Response:
    return Response(content=_OPENAPI_YAML_TEXT, media_type="application/yaml")


# Metrics endpoint via prometheus-fastapi-instrumentator. Note: per
# packet A1 spec, /metrics should run on a SEPARATE uvicorn instance
# on port 9090. The MVP exposes it on the app port too — operators
# can pin it to 9090-only by shipping a uvicorn config that mounts
# only this route. See the README for the dual-port deployment recipe.
Instrumentator().instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False,
)


# --- routes --------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> dict:
    """Liveness/readiness probe. Returns OK as long as the process is
    running and the OpenAPI is loaded."""
    return {"status": "ok"}


@app.post("/article-features/search")
async def search(
    body: LegacySearchRequest,
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=0, le=500, alias="pageSize"),
    sort: list[str] = Query(default_factory=list),
) -> JSONResponse:
    """A2 — translate the legacy DTO into an ftsearch request, POST
    it, and forward the response body. A3 will wrap the response in
    the legacy envelope shape (the wire shapes already mostly align
    per F2's contract design, so the forward-as-is is close to
    correct on the happy path).

    Errors propagate via the FastAPI exception handler in
    `http_error_handler` below — A4 will categorise them (4xx from
    bad input, 5xx from upstream).
    """
    ftsearch_request = map_request(
        body, page=page, page_size=page_size, sort=sort,
    )
    client: FtsearchClient = request.app.state.ftsearch
    try:
        ftsearch_body = await client.search(
            ftsearch_request.body, params=ftsearch_request.params,
        )
    except httpx.HTTPStatusError as exc:
        # ftsearch returned 4xx or 5xx — wrap in the legacy envelope.
        # A4 will categorise each upstream code into a corresponding
        # legacy status; for now we forward the status code verbatim.
        return _error(
            status=exc.response.status_code,
            message="ftsearch returned a non-2xx response",
            details=[exc.response.text[:200]],
        )
    except httpx.RequestError as exc:
        # Network-level failure (connect refused, timeout, etc.).
        return _error(
            status=503,
            message="ftsearch unreachable",
            details=[type(exc).__name__],
        )
    legacy_body = map_response(ftsearch_body, explain=body.explain)
    return JSONResponse(content=legacy_body, status_code=200)


# --- error handlers ------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_error_handler(_request, exc: HTTPException) -> JSONResponse:
    """Wrap every HTTPException in the legacy error envelope. A4 will
    layer richer error categorisation (validation vs upstream vs
    timeout)."""
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return _error(status=exc.status_code, message=detail)
