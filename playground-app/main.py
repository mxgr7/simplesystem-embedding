"""FastAPI + htmx search playground over a Milvus offers collection.

Environment variables:
  EMBED_URL            Base URL of a TEI-compatible embedding service.
  QUERY_PREFIX         String prepended to every query before embedding.
                       Matches the training-time query template (e.g. "query: "
                       for E5-family models). Default: empty.
  MILVUS_URI           Milvus gRPC URI (e.g. http://localhost:19530).
  MILVUS_COLLECTION    Milvus collection name (default: ``offers``).
  OFFERS_PARQUET_GLOB  DuckDB-readable glob to the offers parquet files.
  PAGE_SIZE            Results per page / per "load more" click (default: 10).
  SEARCH_TOP_K         Max retrievable hits per query (default: 200).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from catalog import Catalog, Offer
from embed_client import EmbedClient
from milvus_search import MilvusSearch

BASE_DIR = Path(__file__).resolve().parent


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")

    app.state.embed = EmbedClient(_required_env("EMBED_URL"))
    app.state.milvus = MilvusSearch(
        uri=_required_env("MILVUS_URI"),
        collection=os.environ.get("MILVUS_COLLECTION", "offers"),
    )
    app.state.catalog = Catalog(_required_env("OFFERS_PARQUET_GLOB"))
    app.state.page_size = int(os.environ.get("PAGE_SIZE", "10"))
    app.state.top_k = int(os.environ.get("SEARCH_TOP_K", "200"))
    app.state.query_prefix = os.environ.get("QUERY_PREFIX", "")
    try:
        yield
    finally:
        await app.state.embed.aclose()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "index.html", {"query": "", "initial_html": ""}
    )


@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str = Query("", min_length=0),
    offset: int = Query(0, ge=0),
) -> HTMLResponse:
    q = q.strip()
    page_size: int = request.app.state.page_size
    top_k: int = request.app.state.top_k

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
    vectors = await request.app.state.embed.embed([rendered_query])
    embed_ms = (time.perf_counter() - t0) * 1000
    hits = request.app.state.milvus.search(vectors[0], limit=top_k) if vectors else []

    visible = hits[: offset + page_size]
    page_slice = visible[offset:]

    offers: dict[str, Offer] = request.app.state.catalog.lookup(
        [h.id for h in page_slice]
    )
    cards = [
        _build_card(h.id, h.score, offers.get(h.id))
        for h in page_slice
    ]

    next_offset = offset + page_size if len(hits) > offset + page_size else None
    total_hits = len(hits)
    took_ms = int((time.perf_counter() - t_total) * 1000)

    ctx = {
        "query": q,
        "cards": cards,
        "next_offset": next_offset,
        "total_hits": total_hits,
        "took_ms": took_ms,
        "embed_ms": int(embed_ms),
    }

    if is_load_more:
        return templates.TemplateResponse(request, "_card_items.html", ctx)

    if is_htmx:
        return templates.TemplateResponse(request, "results.html", ctx)

    fragment = templates.get_template("results.html").render(request=request, **ctx)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"query": q, "initial_html": fragment},
    )


def _build_card(hex_id: str, score: float, offer: Offer | None) -> dict:
    if offer is None:
        return {
            "id": hex_id,
            "score": score,
            "name": "(not found in catalog)",
            "manufacturer": "",
            "ean": "",
            "article_number": "",
        }
    return {
        "id": hex_id,
        "score": score,
        "name": offer.name or "(unnamed offer)",
        "manufacturer": offer.manufacturer,
        "ean": offer.ean,
        "article_number": offer.article_number,
    }
