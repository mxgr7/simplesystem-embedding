"""HTTP client for a TEI-compatible embedding service.

TEI (HuggingFace Text Embeddings Inference) exposes ``POST /embed`` with
payload ``{"inputs": [str, ...]}`` and returns ``[[float, ...], ...]``. Any
server that speaks this shape is a drop-in replacement.
"""

from __future__ import annotations

import httpx


class EmbedClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.post(
            f"{self._base_url}/embed",
            json={"inputs": texts},
        )
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()
