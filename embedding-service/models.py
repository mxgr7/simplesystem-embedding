"""Pydantic schemas for the /embed endpoint.

Mirrors TEI's native shape — `{"inputs": str | list[str], "truncate":
bool}` in, `[[float, ...], ...]` out — so existing TEI clients work
unchanged after pointing `EMBED_URL` at this service.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    inputs: str | list[str]
    truncate: bool = Field(default=True)


# TEI's response is a bare 2D array. FastAPI handles that fine when the
# handler returns `list[list[float]]` directly; no wrapper model needed.


class AddBackendRequest(BaseModel):
    """Body for `POST /admin/backends` — register another TEI endpoint at
    runtime (e.g. a larger remote GPU for a one-off indexing run)."""
    url: str = Field(description="TEI base URL, e.g. https://big-gpu.example.com")
    weight: float = Field(default=1.0, gt=0, description="Relative capacity; misses route proportionally to weight.")
    max_concurrency: int = Field(default=16, gt=0, description="Match the backend's TEI --max-concurrent-requests.")
    max_client_batch: int = Field(default=8, gt=0, description="Match the backend's TEI --max-client-batch-size.")
    timeout_s: float = Field(default=30.0, gt=0, description="Per-request HTTP timeout for this backend.")


class PatchBackendRequest(BaseModel):
    """Body for `PATCH /admin/backends/{id}` — adjust routing weight.
    Weight 0 drains the backend to fallback-only (used when no weighted
    backend is selectable)."""
    weight: float = Field(ge=0)
