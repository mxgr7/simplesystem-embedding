"""Compact binary codec for SPLADE sparse vectors in Redis + the ES bulk body.

A SPLADE doc vector is a {token_id: weight} map over ~800-2000 non-zero vocab
dims. JSON is ~3x larger and slow to parse at 12M-doc scale, so the Redis cache
stores each vector as:  uint32 n  then  n x (uint16 token_id, float16 weight).
gbert vocab is 31,102 < 65,536, so token ids fit uint16.

`merge_max` combines an article's per-hash vectors into one (element-wise max over
token weights = union of activated tokens = SPLADE-max pooling extended across the
article's unique texts), the single `sparse_vector` we index per article.
"""
from __future__ import annotations

import struct

import numpy as np

_HEAD = struct.Struct("<I")  # n


def pack_sparse(vec: dict[int, float]) -> bytes:
    """{token_id: weight} -> bytes (uint32 n, then n*(uint16 id, fp16 weight))."""
    if not vec:
        return _HEAD.pack(0)
    ids = np.fromiter(vec.keys(), dtype=np.int64, count=len(vec))
    ws = np.fromiter(vec.values(), dtype=np.float32, count=len(vec))
    order = np.argsort(ids)
    ids16 = ids[order].astype(np.uint16)
    ws16 = ws[order].astype(np.float16)
    return _HEAD.pack(len(vec)) + ids16.tobytes() + ws16.tobytes()


def unpack_sparse(buf: bytes) -> dict[str, float]:
    """bytes -> {str(token_id): float weight} (str keys = ES sparse_vector form)."""
    (n,) = _HEAD.unpack_from(buf, 0)
    if n == 0:
        return {}
    off = _HEAD.size
    ids = np.frombuffer(buf, dtype=np.uint16, count=n, offset=off)
    ws = np.frombuffer(buf, dtype=np.float16, count=n, offset=off + 2 * n)
    return {str(int(i)): float(w) for i, w in zip(ids, ws)}


def merge_max(dicts: list[dict[str, float]]) -> dict[str, float]:
    """Element-wise max over token weights across an article's per-hash vectors."""
    if len(dicts) == 1:
        return dicts[0]
    out: dict[str, float] = {}
    for d in dicts:
        for k, w in d.items():
            p = out.get(k)
            if p is None or w > p:
                out[k] = w
    return out
