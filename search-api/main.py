"""Thin JSON wrapper around a TEI embedder + Milvus search.

Sibling of ``playground-app/`` but without htmx/UI: accepts a query, embeds
it, runs a cosine search against a Milvus collection selected by URL path,
and returns Elasticsearch-shaped hits (``_index``, ``_id``, ``_score``).

Query params ``k`` and ``num_candidates`` override the top-k and the HNSW
``efSearch`` pool size on a per-request basis; ``num_candidates`` must be
>= ``k`` and is ignored by non-HNSW indexes.

Environment variables:
  EMBED_URL         Base URL of a TEI-compatible embedding service.
  QUERY_PREFIX      String prepended to every query before embedding.
  MILVUS_URI        Milvus gRPC URI (e.g. http://localhost:19530).
  SEARCH_TOP_K      Default max hits per query when ``k`` is not set
                    (default: 100).
  ID_FIELDS         Per-collection override of the Milvus field returned as
                    ``_id``. Format: ``col1=field1,col2=field2``. Collections
                    not listed fall back to ``id``.
  API_KEY           Shared secret. Clients must present it via either
                    ``Authorization: ApiKey <key>`` or ``X-API-Key: <key>``.
                    If unset/empty, auth is disabled.
"""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from embed_client import EmbedClient

BASE_DIR = Path(__file__).resolve().parent


_DEFAULT_ID_FIELD = "id"


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
    app.state.milvus = MilvusClient(_required_env("MILVUS_URI"))
    app.state.query_prefix = os.environ.get("QUERY_PREFIX", "")
    app.state.top_k = int(os.environ.get("SEARCH_TOP_K", "100"))
    app.state.id_fields = _parse_id_fields(os.environ.get("ID_FIELDS", ""))
    app.state.api_key = os.environ.get("API_KEY", "")
    try:
        yield
    finally:
        await app.state.embed.aclose()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
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


@app.post("/{collection}/_search")
async def search(
    body: SearchRequest,
    collection: str = PathParam(..., min_length=1, max_length=255),
    k: int | None = Query(default=None, ge=1),
    num_candidates: int | None = Query(default=None, ge=1),
) -> dict:
    client: MilvusClient = app.state.milvus
    if not client.has_collection(collection):
        raise HTTPException(
            status_code=404,
            detail=f"Milvus collection {collection!r} does not exist",
        )

    limit = k if k is not None else app.state.top_k
    if num_candidates is not None and num_candidates < limit:
        raise HTTPException(
            status_code=400,
            detail=f"num_candidates ({num_candidates}) must be >= k ({limit})",
        )

    query = body.query.strip()
    if not query:
        return {"hits": []}

    rendered = f"{app.state.query_prefix}{query}"
    vectors = await app.state.embed.embed([rendered])
    if not vectors:
        return {"hits": []}

    # Collection stores fp16 vectors; matching the query precision flushes
    # subnormals to 0 instead of tripping Milvus's underflow validator.
    id_field: str = app.state.id_fields.get(collection, _DEFAULT_ID_FIELD)
    vec = np.asarray(vectors[0], dtype=np.float16)
    # num_candidates maps to HNSW efSearch; ignored by non-HNSW indexes.
    params: dict = {}
    if num_candidates is not None:
        params["ef"] = num_candidates
    results = client.search(
        collection_name=collection,
        data=[vec],
        limit=limit,
        search_params={"metric_type": "COSINE", "params": params},
        output_fields=[id_field],
    )

    raw = results[0] if results else []
    hits = [
        {
            "_index": body.index,
            "_id": str(h["entity"].get(id_field, "")),
            "_score": float(h["distance"]),
        }
        for h in raw
    ]
    hits.sort(key=lambda h: h["_score"], reverse=True)
    return {"hits": hits}
