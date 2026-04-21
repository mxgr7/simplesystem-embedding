"""Tiny mock TEI-compatible embedder for local UI smoke tests.

Returns a deterministic unit vector per query string (hash-seeded) with the
dimensionality the real index expects. Not useful for retrieval quality —
exists only to let the playground UI be exercised without a checkpoint.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


DIM = int(os.environ.get("MOCK_EMBED_DIM", "128"))


class EmbedRequest(BaseModel):
    inputs: list[str]


app = FastAPI()


def _vector_for(text: str) -> list[float]:
    seed = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "dim": DIM}


@app.post("/embed")
async def embed(req: EmbedRequest) -> list[list[float]]:
    return [_vector_for(t) for t in req.inputs]
