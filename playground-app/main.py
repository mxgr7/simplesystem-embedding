"""FastAPI + htmx playground for the offers index.

Search itself is delegated to the sibling search-api service (a single
source of truth for hybrid behaviour); the playground only renders the UI
and joins display fields keyed on the IDs that come back. Run them both
through ``compose.yaml``.

Environment variables:
  EMBED_URL                Reported in the debug panel; actual embedding
                           happens inside search-api.
  QUERY_PREFIX             Reported in debug panel; actual prefixing also
                           happens inside search-api.
  MILVUS_URI               Milvus gRPC URI. Used for the display-field
                           lookup and for debug-panel collection metadata.
  MILVUS_COLLECTION        Dense collection name (default ``offers``).
  MILVUS_CODES_COLLECTION  Codes collection (default ``offers_codes``).
  SEARCH_API_URL           Base URL of the search-api service
                           (default ``http://localhost:8001``).
  SEARCH_API_KEY           API key for search-api (sent as ``X-API-Key``).
  PAGE_SIZE                Results per page / per "load more" click
                           (default 50).
  SEARCH_TOP_K             Default ``k`` when not overridden in the form.
"""

from __future__ import annotations

import base64
import logging
import os
import secrets
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Path as PathParam, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_fastapi_instrumentator import Instrumentator
from pymilvus import MilvusClient

from milvus_search import (
    OUTPUT_FIELDS,
    CollectionInfo,
    Display,
    OffersLookup,
    describe_collection,
)
from offer_lookup import OfferLookup
from random_query import RandomQueryPicker

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR.parent / "logs" / "playground-app.log"

VALID_MODES = ("vector", "bm25", "hybrid", "hybrid_classified")


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


def _parse_optional_positive_int(raw: str) -> int | None:
    return _parse_positive_int(raw, default=None)


def _flag(raw: str, default: bool) -> bool:
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_mode(raw: str) -> str:
    raw = (raw or "").strip()
    if raw in VALID_MODES:
        return raw
    return "hybrid_classified"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")

    milvus_uri = _required_env("MILVUS_URI")
    dense_col = os.environ.get("MILVUS_COLLECTION", "offers")
    codes_col = os.environ.get("MILVUS_CODES_COLLECTION", "offers_codes")

    app.state.milvus_uri = milvus_uri
    app.state.dense_collection = dense_col
    app.state.codes_collection = codes_col
    app.state.milvus = MilvusClient(milvus_uri)
    app.state.offers_lookup = OffersLookup(app.state.milvus, dense_col)

    app.state.search_api_url = os.environ.get(
        "SEARCH_API_URL", "http://localhost:8001"
    ).rstrip("/")
    app.state.search_api_key = os.environ.get("SEARCH_API_KEY", "")
    app.state.http = httpx.AsyncClient(timeout=30.0)

    app.state.embed_url = os.environ.get("EMBED_URL", "")
    app.state.query_prefix = os.environ.get("QUERY_PREFIX", "")
    app.state.page_size = int(os.environ.get("PAGE_SIZE", "50"))
    app.state.top_k = int(os.environ.get("SEARCH_TOP_K", "100"))

    app.state.auth_user = os.environ.get("PLAYGROUND_USER", "admin")
    app.state.auth_password = os.environ.get("PLAYGROUND_PASSWORD", "")

    app.state.dense_info = describe_collection(app.state.milvus, milvus_uri, dense_col)
    try:
        app.state.codes_info = describe_collection(app.state.milvus, milvus_uri, codes_col)
    except Exception:
        # The codes collection may not exist yet (e.g. before the bulk
        # import has run). Surface as missing in the debug panel rather
        # than crashing the whole app.
        logging.getLogger("playground").warning(
            "codes collection %r not available at startup", codes_col,
        )
        app.state.codes_info = None

    app.state.offers = OfferLookup(_required_env("OFFERS_PARQUET_DIR"))
    app.state.random_query = RandomQueryPicker(_required_env("QUERIES_PARQUET_DIR"))
    try:
        yield
    finally:
        await app.state.http.aclose()
        app.state.offers.close()


app = FastAPI(lifespan=lifespan)


Instrumentator().instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
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
    if request.url.path == "/metrics":
        return await call_next(request)
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
        {"query": "", "initial_html": "", "css_version": _css_version(),
         "form": _empty_form_state()},
    )


@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str = Query("", min_length=0),
    offset: int = Query(0, ge=0),
    mode: str = Query(""),
    k: str = Query(""),
    num_candidates: str = Query(""),
    dense_limit: str = Query(""),
    codes_limit: str = Query(""),
    rrf_k: str = Query(""),
    enable_fallback: str = Query(""),
) -> HTMLResponse:
    q = q.strip()
    page_size: int = request.app.state.page_size

    parsed_mode = _parse_mode(mode)
    top_k: int = _parse_positive_int(k, default=request.app.state.top_k)
    nc: int | None = _parse_optional_positive_int(num_candidates)
    dl: int | None = _parse_optional_positive_int(dense_limit)
    cl: int | None = _parse_optional_positive_int(codes_limit)
    rk: int | None = _parse_optional_positive_int(rrf_k)
    fb: bool = _flag(enable_fallback, default=True)
    form_state = _form_state(parsed_mode, k, num_candidates, dense_limit,
                             codes_limit, rrf_k, fb)

    is_htmx = request.headers.get("hx-request") == "true"
    is_load_more = is_htmx and offset > 0

    if not q:
        ctx = {"query": "", "cards": [], "next_offset": None, "total_hits": 0,
               "took_ms": 0, "embed_ms": 0, "form": form_state}
        if is_htmx:
            return templates.TemplateResponse(request, "results.html", ctx)
        return templates.TemplateResponse(
            request, "index.html",
            {**ctx, "initial_html": "", "css_version": _css_version()}
        )

    t_total = time.perf_counter()

    api_params: dict[str, str] = {"mode": parsed_mode, "k": str(top_k), "debug": "1"}
    if nc is not None: api_params["num_candidates"] = str(nc)
    if dl is not None: api_params["dense_limit"] = str(dl)
    if cl is not None: api_params["codes_limit"] = str(cl)
    if rk is not None: api_params["rrf_k"] = str(rk)
    api_params["enable_fallback"] = "1" if fb else "0"

    headers = {}
    if request.app.state.search_api_key:
        headers["X-API-Key"] = request.app.state.search_api_key

    url = (
        f"{request.app.state.search_api_url}/"
        f"{request.app.state.dense_collection}/_search"
    )
    payload = {"query": q, "category": None, "index": "playground"}

    t0 = time.perf_counter()
    try:
        resp = await request.app.state.http.post(
            url, params=api_params, json=payload, headers=headers
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = (exc.response.text or "").strip()
        return _render_error(
            request, query=q,
            error={
                "title": f"search-api returned {exc.response.status_code}",
                "detail": body[:500] or exc.response.reason_phrase,
            },
            is_htmx=is_htmx, is_load_more=is_load_more, form=form_state,
        )
    except Exception as exc:
        logging.getLogger("playground").exception("search-api call failed")
        return _render_error(
            request, query=q,
            error={
                "title": "Search backend unavailable",
                "detail": f"{type(exc).__name__}: {exc}".strip()
                          or "Could not reach the search-api service.",
            },
            is_htmx=is_htmx, is_load_more=is_load_more, form=form_state,
        )
    api_ms = (time.perf_counter() - t0) * 1000

    body = resp.json()
    api_hits = body.get("hits", [])
    api_debug = body.get("_debug", {}) or {}

    ids = [h["_id"] for h in api_hits if h.get("_id")]
    t0 = time.perf_counter()
    by_id = request.app.state.offers_lookup.fetch(ids) if ids else {}
    lookup_ms = (time.perf_counter() - t0) * 1000

    ordered: list[tuple[dict, Display]] = []
    for h in api_hits:
        hid = h.get("_id", "")
        disp = by_id.get(hid)
        if disp is None:
            # IDs not found in the dense collection — would indicate
            # offers_codes drift from offers. Skip rather than render a
            # broken card.
            continue
        ordered.append((h, disp))

    visible = ordered[: offset + page_size]
    page_slice = visible[offset:]
    cards = [_build_card(h, d) for h, d in page_slice]

    next_offset = offset + page_size if len(ordered) > offset + page_size else None
    total_hits = len(ordered)
    took_ms = int((time.perf_counter() - t_total) * 1000)

    top_score = ordered[0][0].get("_score") if ordered else None
    last_score = ordered[-1][0].get("_score") if ordered else None
    legs_seen = sorted({h.get("_source_leg", "") for h, _ in ordered})

    category_facet = _aggregate_category_l1([d for _, d in ordered])

    ctx = {
        "query": q,
        "cards": cards,
        "next_offset": next_offset,
        "total_hits": total_hits,
        "took_ms": took_ms,
        "embed_ms": int(api_debug.get("embed_ms") or 0),
        "category_facet": category_facet,
        "form": form_state,
        "debug": _debug_payload(
            request, q, parsed_mode, top_k, api_params, api_debug,
            api_ms, lookup_ms, took_ms, total_hits, len(cards),
            top_score, last_score, legs_seen,
        ),
    }

    if is_load_more:
        return templates.TemplateResponse(request, "_card_items.html", ctx)
    if is_htmx:
        return templates.TemplateResponse(request, "results.html", ctx)
    fragment = templates.get_template("results.html").render(request=request, **ctx)
    return templates.TemplateResponse(
        request, "index.html",
        {"query": q, "initial_html": fragment, "css_version": _css_version()},
    )


def _empty_form_state() -> dict:
    return {
        "mode": "hybrid_classified", "k": "", "num_candidates": "",
        "dense_limit": "", "codes_limit": "", "rrf_k": "",
        "enable_fallback": True,
    }


def _form_state(
    mode: str, k: str, nc: str, dl: str, cl: str, rk: str, fb: bool,
) -> dict:
    return {
        "mode": mode, "k": k, "num_candidates": nc,
        "dense_limit": dl, "codes_limit": cl, "rrf_k": rk,
        "enable_fallback": fb,
    }


def _debug_payload(
    request: Request,
    q: str, mode: str, top_k: int,
    api_params: dict, api_debug: dict,
    api_ms: float, lookup_ms: float, took_ms: int,
    total_hits: int, shown: int,
    top_score: float | None, last_score: float | None,
    legs_seen: list[str],
) -> dict:
    rendered = f"{request.app.state.query_prefix}{q}"
    dense: CollectionInfo = request.app.state.dense_info
    codes: CollectionInfo | None = request.app.state.codes_info

    return {
        "query": q,
        "rendered_query": rendered,
        "query_prefix": request.app.state.query_prefix,
        "embed_url": request.app.state.embed_url,
        "embed_dim": dense.vector_dim if dense else 0,
        "page_size": request.app.state.page_size,
        "top_k": top_k,
        "mode": mode,
        "search_api_url": request.app.state.search_api_url,
        "api_params": api_params,
        "path": api_debug.get("path"),
        "classifier_strict": api_debug.get("classifier_strict"),
        "fallback_fired": api_debug.get("fallback_fired"),
        "embed_ms": api_debug.get("embed_ms"),
        "dense_ms": api_debug.get("dense_ms"),
        "codes_ms": api_debug.get("codes_ms"),
        "dense_hits": api_debug.get("dense_hits"),
        "codes_hits": api_debug.get("codes_hits"),
        "params_echo": api_debug.get("params"),
        "took_ms": took_ms,
        "api_ms": round(api_ms, 1),
        "lookup_ms": round(lookup_ms, 1),
        "other_ms": round(max(0.0, took_ms - api_ms - lookup_ms), 1),
        "hits": total_hits,
        "shown": shown,
        "top_score": round(top_score, 4) if top_score is not None else None,
        "last_score": round(last_score, 4) if last_score is not None else None,
        "legs_seen": legs_seen,
        "milvus_uri": request.app.state.milvus_uri,
        "dense": _info_dict(dense),
        "codes": _info_dict(codes) if codes else None,
        "output_fields": OUTPUT_FIELDS,
    }


def _info_dict(info: CollectionInfo | None) -> dict | None:
    if info is None:
        return None
    return {
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
    }


def _render_error(
    request: Request, *, query: str, error: dict,
    is_htmx: bool, is_load_more: bool, form: dict,
) -> HTMLResponse:
    ctx = {
        "query": query, "cards": [], "next_offset": None, "total_hits": 0,
        "took_ms": 0, "embed_ms": 0, "error": error, "form": form,
    }
    if is_load_more:
        return templates.TemplateResponse(request, "_error_li.html", ctx)
    if is_htmx:
        return templates.TemplateResponse(request, "results.html", ctx)
    fragment = templates.get_template("results.html").render(request=request, **ctx)
    return templates.TemplateResponse(
        request, "index.html",
        {"query": query, "initial_html": fragment, "css_version": _css_version()},
    )


@app.get("/random-query")
async def random_query(request: Request) -> dict:
    return {"query": request.app.state.random_query.pick()}


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


def _aggregate_category_l1(displays: list[Display], limit: int = 15) -> list[dict]:
    counter: Counter[str] = Counter()
    for d in displays:
        for path in d.category_paths:
            if path:
                counter[path[0]] += 1
    return [{"name": name, "count": count} for name, count in counter.most_common(limit)]


def _build_card(api_hit: dict, d: Display) -> dict:
    return {
        "id": d.id,
        "score": float(api_hit.get("_score", 0.0)),
        "score_source": api_hit.get("_source_leg", ""),
        "name": d.name or "(unnamed offer)",
        "manufacturer": d.manufacturer,
        "ean": d.ean,
        "article_number": d.article_number,
        "catalog_version_ids": d.catalog_version_ids,
        "category_paths": d.category_paths,
    }
