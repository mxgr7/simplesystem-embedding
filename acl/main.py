"""Article-search ACL service — narrowed legacy contract → ftsearch.

A1 (skeleton + OpenAPI). The endpoint stub returns 501 until A2 wires
request mapping + A3 wires response mapping + A4 wires the error
contract. A5 layers in tracing, retries, RED metrics — the same
operational pieces the search-api side got from F7.

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

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

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
    # Future: open ftsearch HTTP client here, share across requests.
    # A2/A5 land that pattern. For now the stub doesn't talk to
    # ftsearch at all so there's nothing to initialize.
    app.state.ftsearch_url = os.environ.get(
        "FTSEARCH_URL", "http://search-api:8001",
    )
    yield


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
async def search() -> JSONResponse:
    """Stub — A2/A3 fill this in. Returns 501 with the legacy error
    envelope so callers see a well-shaped response even pre-implementation."""
    return _error(
        status=501,
        message="Not implemented",
        details=[
            "ACL skeleton stub — A2 (request mapper) and A3 (response "
            "mapper) land the actual translation to ftsearch."
        ],
    )


# --- error handlers ------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_error_handler(_request, exc: HTTPException) -> JSONResponse:
    """Wrap every HTTPException in the legacy error envelope. A4 will
    layer richer error categorisation (validation vs upstream vs
    timeout)."""
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return _error(status=exc.status_code, message=detail)
