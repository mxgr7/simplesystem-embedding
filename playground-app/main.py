"""FastAPI + htmx search playground over a Milvus offers collection.

Environment variables:
  EMBED_URL            Base URL of a TEI-compatible embedding service.
  QUERY_PREFIX         String prepended to every query before embedding.
                       Matches the training-time query template (e.g. "query: "
                       for E5-family models). Default: empty.
  MILVUS_URI           Milvus gRPC URI (e.g. http://localhost:19530).
  MILVUS_COLLECTION    Milvus collection name (default: ``offers``).
  PAGE_SIZE            Results per page / per "load more" click (default: 50).
  SEARCH_TOP_K         Max retrievable hits per query (default: 100).
"""

from __future__ import annotations

import base64
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Path as PathParam, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from embed_client import EmbedClient
from milvus_search import OUTPUT_FIELDS, Hit, MilvusSearch
from offer_lookup import OfferLookup

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR.parent / "logs" / "playground-app.log"


def _build_request_logger() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("playground.requests")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(
        isinstance(h, logging.FileHandler)
        and Path(getattr(h, "baseFilename", "")) == LOG_FILE
        for h in logger.handlers
    ):
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    return logger


request_logger = _build_request_logger()


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _parse_positive_int(raw: str, default):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")

    app.state.embed_url = _required_env("EMBED_URL")
    app.state.embed = EmbedClient(app.state.embed_url)
    app.state.milvus = MilvusSearch(
        uri=_required_env("MILVUS_URI"),
        collection=os.environ.get("MILVUS_COLLECTION", "offers"),
    )
    app.state.page_size = int(os.environ.get("PAGE_SIZE", "50"))
    app.state.top_k = int(os.environ.get("SEARCH_TOP_K", "100"))
    app.state.query_prefix = os.environ.get("QUERY_PREFIX", "")
    app.state.auth_user = os.environ.get("PLAYGROUND_USER", "admin")
    app.state.auth_password = os.environ.get("PLAYGROUND_PASSWORD", "")
    app.state.milvus_info = app.state.milvus.describe()
    app.state.offers = OfferLookup(_required_env("OFFERS_PARQUET_DIR"))
    try:
        yield
    finally:
        await app.state.embed.aclose()
        app.state.offers.close()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    query = f"?{request.url.query}" if request.url.query else ""
    client = request.client.host if request.client else "-"
    request_logger.info(
        "%s %s %s%s %d %.1fms",
        client,
        request.method,
        request.url.path,
        query,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.middleware("http")
async def require_basic_auth(request: Request, call_next):
    password = request.app.state.auth_password
    if not password:
        return await call_next(request)
    user = request.app.state.auth_user
    header = request.headers.get("authorization", "")
    ok = False
    if header[:6].lower() == "basic ":
        try:
            decoded = base64.b64decode(header[6:]).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            decoded = ""
        u, sep, p = decoded.partition(":")
        if sep:
            ok = secrets.compare_digest(u, user) and secrets.compare_digest(p, password)
    if not ok:
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="playground"'},
        )
    return await call_next(request)


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def _css_version() -> str:
    return str(int((BASE_DIR / "static" / "app.css").stat().st_mtime))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"query": "", "initial_html": "", "css_version": _css_version()},
    )


@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str = Query("", min_length=0),
    offset: int = Query(0, ge=0),
    k: str = Query(""),
    num_candidates: str = Query(""),
) -> HTMLResponse:
    q = q.strip()
    page_size: int = request.app.state.page_size
    top_k: int = _parse_positive_int(k, default=request.app.state.top_k)
    ef: int | None = _parse_positive_int(num_candidates, default=None)

    is_htmx = request.headers.get("hx-request") == "true"
    is_load_more = is_htmx and offset > 0

    if not q:
        ctx = {"query": "", "cards": [], "next_offset": None, "total_hits": 0,
               "took_ms": 0, "embed_ms": 0}
        if is_htmx:
            return templates.TemplateResponse(request, "results.html", ctx)
        return templates.TemplateResponse(
            request, "index.html", {**ctx, "initial_html": ""}
        )

    t_total = time.perf_counter()

    rendered_query = f"{request.app.state.query_prefix}{q}"
    t0 = time.perf_counter()
    try:
        vectors = await request.app.state.embed.embed([rendered_query])
    except Exception as exc:
        logging.getLogger("playground").exception("embed request failed")
        return _render_error(
            request,
            query=q,
            error={
                "title": "Embedding service unavailable",
                "detail": f"{type(exc).__name__}: {exc}".strip()
                          or "The embedding backend did not respond.",
            },
            is_htmx=is_htmx,
            is_load_more=is_load_more,
        )
    embed_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    try:
        if vectors:
            hits, search_params = request.app.state.milvus.search(
                vectors[0], limit=top_k, ef=ef
            )
        else:
            hits, search_params = [], {"metric_type": "COSINE", "params": {}}
    except Exception as exc:
        logging.getLogger("playground").exception("milvus search failed")
        return _render_error(
            request,
            query=q,
            error={
                "title": "Milvus search failed",
                "detail": f"{type(exc).__name__}: {exc}".strip()
                          or "Milvus did not return a result.",
            },
            is_htmx=is_htmx,
            is_load_more=is_load_more,
        )
    milvus_ms = (time.perf_counter() - t0) * 1000

    visible = hits[: offset + page_size]
    page_slice = visible[offset:]
    cards = [_build_card(h) for h in page_slice]

    next_offset = offset + page_size if len(hits) > offset + page_size else None
    total_hits = len(hits)
    took_ms = int((time.perf_counter() - t_total) * 1000)

    top_score = hits[0].score if hits else None
    last_score = hits[-1].score if hits else None

    info = request.app.state.milvus_info

    ctx = {
        "query": q,
        "cards": cards,
        "next_offset": next_offset,
        "total_hits": total_hits,
        "took_ms": took_ms,
        "embed_ms": int(embed_ms),
        "debug": {
            # Query
            "query": q,
            "rendered_query": rendered_query,
            "query_prefix": request.app.state.query_prefix,
            "embed_url": request.app.state.embed_url,
            "embed_dim": len(vectors[0]) if vectors else 0,
            "offset": offset,
            "page_size": page_size,
            "top_k": top_k,
            "num_candidates": ef,
            # Timing
            "took_ms": took_ms,
            "embed_ms": round(embed_ms, 1),
            "milvus_ms": round(milvus_ms, 1),
            "other_ms": round(max(0.0, took_ms - embed_ms - milvus_ms), 1),
            # Results
            "hits": total_hits,
            "shown": len(cards),
            "top_score": round(top_score, 4) if top_score is not None else None,
            "last_score": round(last_score, 4) if last_score is not None else None,
            # Milvus static info
            "milvus_uri": info.uri,
            "collection": info.collection,
            "row_count": info.row_count,
            "load_state": info.load_state,
            "partitions": info.num_partitions,
            "shards": info.num_shards,
            "vector_field": info.vector_field,
            "vector_dim": info.vector_dim,
            "vector_dtype": info.vector_dtype,
            "index_type": info.index_type,
            "metric_type": info.metric_type,
            "index_params": info.index_params,
            "indexed_rows": info.indexed_rows,
            "index_state": info.index_state,
            "scalar_indexes": info.scalar_indexes,
            # Search-time params actually sent to Milvus
            "search_params": search_params,
            "output_fields": OUTPUT_FIELDS,
        },
    }

    if is_load_more:
        return templates.TemplateResponse(request, "_card_items.html", ctx)

    if is_htmx:
        return templates.TemplateResponse(request, "results.html", ctx)

    fragment = templates.get_template("results.html").render(request=request, **ctx)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"query": q, "initial_html": fragment, "css_version": _css_version()},
    )


def _render_error(
    request: Request,
    *,
    query: str,
    error: dict,
    is_htmx: bool,
    is_load_more: bool,
) -> HTMLResponse:
    ctx = {
        "query": query,
        "cards": [],
        "next_offset": None,
        "total_hits": 0,
        "took_ms": 0,
        "embed_ms": 0,
        "error": error,
    }
    if is_load_more:
        return templates.TemplateResponse(request, "_error_li.html", ctx)
    if is_htmx:
        return templates.TemplateResponse(request, "results.html", ctx)
    fragment = templates.get_template("results.html").render(request=request, **ctx)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"query": query, "initial_html": fragment, "css_version": _css_version()},
    )


@app.get("/offer/{offer_id}", response_class=HTMLResponse)
async def offer_details(
    request: Request,
    offer_id: str = PathParam(..., min_length=1, max_length=128),
) -> HTMLResponse:
    t0 = time.perf_counter()
    record = request.app.state.offers.get(offer_id)
    took_ms = int((time.perf_counter() - t0) * 1000)
    if record is None:
        return templates.TemplateResponse(
            request,
            "_offer_modal.html",
            {"offer_id": offer_id, "record": None, "took_ms": took_ms},
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "_offer_modal.html",
        {"offer_id": offer_id, "record": record, "took_ms": took_ms},
    )


def _build_card(hit: Hit) -> dict:
    return {
        "id": hit.id,
        "score": hit.score,
        "name": hit.name or "(unnamed offer)",
        "manufacturer": hit.manufacturer,
        "ean": hit.ean,
        "article_number": hit.article_number,
        "catalog_version_ids": hit.catalog_version_ids,
        "category_paths": hit.category_paths,
    }
