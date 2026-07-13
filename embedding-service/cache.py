"""Async KVRocks-backed embedding cache.

KVRocks speaks the Redis protocol, so we use `redis.asyncio` and treat it
as a plain Redis client. The cache holds fp16 vectors as raw bytes
(256 B per 128-d vector), keyed by `tei:v2:{hash}`. We share the
keyspace with `indexer/tei_cache.py` so prewarmed entries are hit by
serving paths too.

Failure posture: cache outages must not take the API down. MGET errors
or timeouts log a warning and the handler proceeds as if every key
missed — TEI will fill the gap. MSET (background) errors are logged
silently; the next miss will rewrite them.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import redis.asyncio as aioredis

from hashing import cache_key

log = logging.getLogger(__name__)

# Model is 128-d fp16. Hardcoded so corruption (wrong-length values) is
# caught on read rather than served. Bump together with the model.
VECTOR_DIM = 128
VECTOR_BYTES = VECTOR_DIM * 2


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    bytes_written: int = 0
    read_errors: int = 0
    write_errors: int = 0


class EmbeddingCache:
    """Async wrapper over an aioredis client. One instance per process,
    created in the FastAPI lifespan startup."""

    def __init__(
        self,
        url: str,
        *,
        read_timeout_s: float,
        max_connections: int,
    ) -> None:
        # `decode_responses=False` keeps GET results as bytes — we store
        # raw fp16 buffers, not utf-8.
        self._client = aioredis.from_url(
            url,
            decode_responses=False,
            max_connections=max_connections,
        )
        self._read_timeout_s = read_timeout_s
        self.stats = CacheStats()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def mget(self, hashes: Sequence[str]) -> list[bytes | None]:
        """Fetch fp16 bytes for `hashes`, preserving order.

        Returns None for misses, malformed values, *and* for the entire
        batch if MGET errors / times out (cache-failure = treat-as-miss
        per the plan). Hit/miss stats are updated; read-error increments
        `read_errors` and treats all entries as misses without raising.
        """
        if not hashes:
            return []
        keys = [cache_key(h) for h in hashes]
        try:
            raw = await asyncio.wait_for(
                self._client.mget(keys),
                timeout=self._read_timeout_s,
            )
        except (asyncio.TimeoutError, aioredis.RedisError) as e:
            self.stats.read_errors += 1
            log.warning(
                "kvrocks MGET failed (%s) — treating %d keys as misses",
                e, len(hashes),
            )
            return [None] * len(hashes)

        out: list[bytes | None] = []
        for h, v in zip(hashes, raw):
            if v is None:
                out.append(None)
            elif len(v) != VECTOR_BYTES:
                log.warning(
                    "kvrocks: unexpected byte length for %s (got %d, expected %d) — miss",
                    h, len(v), VECTOR_BYTES,
                )
                out.append(None)
            else:
                out.append(v)
        return out

    async def mset(self, hash_to_bytes: dict[str, bytes]) -> None:
        """Pipeline-set many fp16 vectors. Failures log + bump
        `write_errors`; never raise (this is called from BackgroundTasks
        and the client has already been served)."""
        if not hash_to_bytes:
            return
        try:
            pipe = self._client.pipeline(transaction=False)
            for h, b in hash_to_bytes.items():
                pipe.set(cache_key(h), b)
            await pipe.execute()
            self.stats.bytes_written += sum(len(b) for b in hash_to_bytes.values())
        except aioredis.RedisError as e:
            self.stats.write_errors += 1
            log.warning(
                "kvrocks MSET failed (%s) for %d keys — entries will refill on next miss",
                e, len(hash_to_bytes),
            )

    async def scan_any(self, match: str = "tei:*", *, timeout_s: float) -> bool:
        """True iff at least one key matching `match` exists (/readyz).

        Deliberately NOT DBSIZE: KVRocks serves a cached counter that
        resets to 0 after a restart until an explicit `DBSIZE SCAN`
        recount, so DBSIZE>0 false-negatives on exactly the weekly-bootup
        path this check gates. COUNT 100 because MATCH filters *after*
        fetching COUNT entries, so COUNT 1 could return empty batches
        indefinitely. Unlike mget/mset this raises on error or timeout —
        the caller must distinguish unreachable from empty.
        """
        async def _scan() -> bool:
            cursor = 0
            while True:
                cursor, keys = await self._client.scan(
                    cursor=cursor, match=match, count=100,
                )
                if keys:
                    return True
                if cursor == 0:
                    return False

        return await asyncio.wait_for(_scan(), timeout=timeout_s)


def vec_bytes_from_fp16(arr: np.ndarray) -> bytes:
    """fp16 ndarray → raw bytes for cache storage. Shape (128,) → 256 B."""
    assert arr.dtype == np.float16, f"expected fp16, got {arr.dtype}"
    assert arr.shape == (VECTOR_DIM,), f"expected ({VECTOR_DIM},), got {arr.shape}"
    return arr.tobytes()


def fp16_from_bytes(b: bytes) -> np.ndarray:
    """Decode raw cache bytes → fp16 ndarray. Caller is responsible for
    only passing values that already cleared the length check in MGET."""
    return np.frombuffer(b, dtype=np.float16)
