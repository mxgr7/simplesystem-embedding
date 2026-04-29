"""Hash-keyed TEI embedding cache for the F9 bulk indexer.

`embed_articles` is the single entry point: hand it a batch of
`(article_hash, article_row)` pairs, it returns one fp16 vector per
article, hitting Redis first and only calling TEI on cache misses. New
embeddings are written back so subsequent bulk runs (or cross-shard
hash repeats within the same run) skip the GPU.

Cache shape:
  - Key:   `tei:{HASH_VERSION}:{article_hash}`
  - Value: raw fp16 bytes (256 B for a 128-d vector)
  - TTL:   none — bulk-cycle cache, manually flushed on hash-version
           bumps. The HASH_VERSION prefix means a bump leaves the old
           keys orphaned (Redis evicts when memory pressure hits) but
           never serves a stale embedding to the new code path.

At F9 production scale (~130M unique embeddings, fp16×128) the cache
footprint is ~33 GB of values + ~5–10 GB of Redis key/structure
overhead — fits a single Redis box with 64 GB RAM. See F9 spec
"TEI cache topology" for the in-memory-dict-vs-Redis decision.

TEI HTTP contract: `POST /embed` with `{"inputs": [str, ...]}`,
returns `[[float, ...], ...]`. Same shape as `search-api/embed_client.py`
(query path); we reuse the contract here on the indexer side. The
search-api client is async (FastAPI request lifecycle); the indexer
client is sync (one big serial bulk run, async overhead would buy
nothing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import httpx
import numpy as np
import redis

from indexer.embedding_text import article_to_text
from indexer.projection import HASH_VERSION

log = logging.getLogger(__name__)

# Bytes-per-vector at fp16, 128-d. Hardcoded check on cache reads to
# catch corruption / dim mismatches early rather than serving a wrong
# vector to Milvus.
_VECTOR_DIM = 128
_VECTOR_BYTES = _VECTOR_DIM * 2


def _cache_key(article_hash: str) -> str:
    return f"tei:{HASH_VERSION}:{article_hash}"


@dataclass
class TEICacheStats:
    """Per-call counters surfaced to the orchestrator's progress logging.
    Reset by the caller (`indexer.bulk`) between phases."""
    hits: int = 0
    misses: int = 0
    tei_calls: int = 0
    bytes_written: int = 0


class TEICache:
    """Sync TEI client with a hash-keyed Redis cache in front. One
    instance per bulk run; share across worker threads if the
    orchestrator goes parallel later."""

    def __init__(
        self,
        *,
        tei_url: str,
        redis_client: redis.Redis,
        tei_batch_size: int = 64,
        timeout_s: float = 60.0,
    ) -> None:
        self._tei_url = tei_url.rstrip("/")
        self._redis = redis_client
        self._tei_batch = tei_batch_size
        self._http = httpx.Client(timeout=timeout_s)
        self.stats = TEICacheStats()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "TEICache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- internal helpers --------------------------------------------

    def _redis_mget(self, hashes: list[str]) -> list[bytes | None]:
        """Pipelined MGET. Returns one entry per hash in the same order;
        None for cache misses or for stored values whose byte length
        doesn't match `_VECTOR_BYTES` (treated as misses + warned, so a
        corrupted entry doesn't poison downstream)."""
        if not hashes:
            return []
        raw = self._redis.mget([_cache_key(h) for h in hashes])
        out: list[bytes | None] = []
        for h, v in zip(hashes, raw):
            if v is None:
                out.append(None)
            elif len(v) != _VECTOR_BYTES:
                log.warning(
                    "redis cache: unexpected byte length for %s (got %d, expected %d) — treating as miss",
                    h, len(v), _VECTOR_BYTES,
                )
                out.append(None)
            else:
                out.append(v)
        return out

    def _redis_mset(self, hash_to_bytes: dict[str, bytes]) -> int:
        if not hash_to_bytes:
            return 0
        pipe = self._redis.pipeline(transaction=False)
        for h, b in hash_to_bytes.items():
            pipe.set(_cache_key(h), b)
        pipe.execute()
        return sum(len(b) for b in hash_to_bytes.values())

    def _tei_embed(self, texts: list[str]) -> np.ndarray:
        """One TEI call. Returns shape (n, dim) fp32 array — caller
        casts to fp16 for storage. Splits into `_tei_batch` chunks so
        the request payload stays bounded for TEI's per-request memory
        ceiling.

        `truncate: true` is set per-request because the production
        model checkpoint (`useful-cub-58-st`) was trained with a
        ~32-token max_input_length. Article texts (`embedding_text.py`)
        can run longer when an article carries multiple categories or
        eclass codes; truncating right-side keeps the higher-signal
        prefix (`name`, `manufacturerName`) intact. Without this flag
        TEI returns 413 + 'inputs must have less than 32 tokens'."""
        if not texts:
            return np.empty((0, _VECTOR_DIM), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), self._tei_batch):
            chunk = texts[i : i + self._tei_batch]
            resp = self._http.post(
                f"{self._tei_url}/embed",
                json={"inputs": chunk, "truncate": True},
            )
            resp.raise_for_status()
            arr = np.asarray(resp.json(), dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != _VECTOR_DIM:
                raise RuntimeError(
                    f"TEI returned unexpected shape {arr.shape} (expected (*, {_VECTOR_DIM}))"
                )
            chunks.append(arr)
            self.stats.tei_calls += 1
        return np.concatenate(chunks, axis=0)

    # --- public API --------------------------------------------------

    def embed_articles(
        self,
        articles: Iterable[dict],
    ) -> dict[str, np.ndarray]:
        """Hash-keyed embedding lookup + fill.

        Input: iterable of article rows (each must have
        `article_hash`). Output: dict `article_hash → fp16 vector`
        with one entry per unique hash in the batch.

        Cache flow per batch:
          1. MGET all hashes from Redis (1 round-trip).
          2. For misses: render text, batch-call TEI, cast fp16.
          3. MSET all new fp16 bytes back (1 round-trip).
          4. Return one vector per unique hash.

        Duplicate hashes within the same batch (two offers of the same
        article) consume one cache read and one TEI call — the dict
        keying naturally collapses them."""
        # Collapse to unique-hash list while remembering one row per hash
        # for text rendering (any of the duplicate rows works since the
        # embedded fields are invariant within a hash group).
        rep_by_hash: dict[str, dict] = {}
        for a in articles:
            h = a["article_hash"]
            if h not in rep_by_hash:
                rep_by_hash[h] = a

        if not rep_by_hash:
            return {}

        ordered_hashes = list(rep_by_hash.keys())
        cached = self._redis_mget(ordered_hashes)

        # Build (hash, text) list for misses; render text once per miss.
        miss_hashes: list[str] = []
        miss_texts: list[str] = []
        for h, v in zip(ordered_hashes, cached):
            if v is None:
                miss_hashes.append(h)
                miss_texts.append(article_to_text(rep_by_hash[h]))

        if miss_texts:
            new_fp32 = self._tei_embed(miss_texts)
            new_fp16 = new_fp32.astype(np.float16)
            new_bytes = {
                h: new_fp16[i].tobytes()
                for i, h in enumerate(miss_hashes)
            }
            self.stats.bytes_written += self._redis_mset(new_bytes)
        else:
            new_fp16 = np.empty((0, _VECTOR_DIM), dtype=np.float16)

        # Stitch results back. `cached[i]` is fp16 bytes (or None);
        # `new_fp16[j]` indexes into the miss-side array.
        out: dict[str, np.ndarray] = {}
        miss_idx = 0
        for h, v in zip(ordered_hashes, cached):
            if v is None:
                out[h] = new_fp16[miss_idx]
                miss_idx += 1
                self.stats.misses += 1
            else:
                out[h] = np.frombuffer(v, dtype=np.float16)
                self.stats.hits += 1
        return out


__all__ = ["TEICache", "TEICacheStats"]
