import math
import struct

import numpy as np

from constants import TOP_K, VOCAB_SIZE


_HEAD = struct.Struct("<H")


def pack_sparse(vector):
    items = []
    for token_id, weight in vector.items():
        token_id = int(token_id)
        weight = float(weight)
        if 0 <= token_id < VOCAB_SIZE and weight > 0 and math.isfinite(weight):
            items.append((token_id, weight))
    items.sort(key=lambda item: (-item[1], item[0]))
    items = items[:TOP_K]
    if not items:
        return _HEAD.pack(0)
    items.sort(key=lambda item: item[0])
    ids = np.asarray([item[0] for item in items], dtype=np.uint16)
    weights = np.asarray([item[1] for item in items], dtype=np.float16)
    positive = weights > 0
    ids = ids[positive]
    weights = weights[positive]
    return _HEAD.pack(len(ids)) + ids.tobytes() + weights.tobytes()


def unpack_sparse(value):
    if len(value) < _HEAD.size:
        raise ValueError("sparse value is truncated")
    (count,) = _HEAD.unpack_from(value)
    expected = _HEAD.size + count * 4
    if count > TOP_K or len(value) != expected:
        raise ValueError("invalid sparse value length")
    ids = np.frombuffer(value, dtype=np.uint16, count=count, offset=_HEAD.size)
    weights = np.frombuffer(
        value,
        dtype=np.float16,
        count=count,
        offset=_HEAD.size + count * 2,
    )
    if np.any(ids >= VOCAB_SIZE) or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("invalid sparse value contents")
    return {str(int(token_id)): float(weight) for token_id, weight in zip(ids, weights)}
