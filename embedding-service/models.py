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
