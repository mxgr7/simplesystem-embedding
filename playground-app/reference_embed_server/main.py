"""Reference TEI-compatible embedding server for the fine-tuned checkpoint.

Exposes ``POST /embed`` with payload ``{"inputs": [str, ...]}`` returning a
list of float vectors — the same shape HuggingFace Text Embeddings Inference
(TEI) uses. The playground UI consumes this endpoint via ``EMBED_URL``.

This server exists because the fine-tuned checkpoint uses a custom query
template (``RowTextRenderer``) and optional projection head that TEI does not
know about. Swapping to TEI is fine once the checkpoint is exported to a
sentence-transformers directory.

Environment variables:
  CHECKPOINT   Path to the Lightning checkpoint (.ckpt).
  DEVICE       auto | cpu | cuda | cuda:0 | mps (default: auto).
  HOST, PORT   Uvicorn bind (default: 0.0.0.0:8080).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from embedding_train.infer import build_tokenizer, encode_texts, resolve_device
from embedding_train.model import load_embedding_module_from_checkpoint
from embedding_train.rendering import RowTextRenderer


class EmbedRequest(BaseModel):
    inputs: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    checkpoint = os.environ.get("CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("CHECKPOINT env var must point to a .ckpt file")
    if not Path(checkpoint).exists():
        raise RuntimeError(f"CHECKPOINT not found: {checkpoint}")

    device = resolve_device(os.environ.get("DEVICE", "auto"))
    model, cfg = load_embedding_module_from_checkpoint(checkpoint, map_location="cpu")
    model = model.to(device).eval()

    app.state.model = model
    app.state.tokenizer = build_tokenizer(cfg.model.model_name)
    app.state.renderer = RowTextRenderer(cfg.data)
    app.state.max_length = int(cfg.data.max_query_length)
    app.state.device = device
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/embed")
async def embed(req: EmbedRequest) -> list[list[float]]:
    rendered = [app.state.renderer.render_query_text({"query_term": q}) for q in req.inputs]
    embs = encode_texts(
        model=app.state.model,
        tokenizer=app.state.tokenizer,
        texts=rendered,
        max_length=app.state.max_length,
        encode_batch_size=max(len(rendered), 1),
        device=app.state.device,
    )
    return embs.cpu().float().tolist()
