import math
import struct

import numpy as np

from constants import TOP_K, VOCAB_SIZE


_HEAD = struct.Struct("<H")
_BATCH_HEAD = struct.Struct("<4sI")
BATCH_MAGIC = b"SPB1"


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
    ids = np.asarray([item[0] for item in items], dtype="<u2")
    weights = np.asarray([item[1] for item in items], dtype="<f2")
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
    ids = np.frombuffer(value, dtype="<u2", count=count, offset=_HEAD.size)
    weights = np.frombuffer(
        value,
        dtype="<f2",
        count=count,
        offset=_HEAD.size + count * 2,
    )
    if np.any(ids >= VOCAB_SIZE) or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("invalid sparse value contents")
    return {str(int(token_id)): float(weight) for token_id, weight in zip(ids, weights)}


def pack_sparse_arrays(token_ids, weights):
    ids = np.asarray(token_ids)
    values = np.asarray(weights, dtype="<f2")
    if ids.ndim != 1 or values.ndim != 1 or len(ids) != len(values):
        raise ValueError("sparse arrays must be same-length vectors")
    if len(ids) > TOP_K:
        raise ValueError("sparse array exceeds top-k")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("sparse token ids must be integers")
    if np.any(ids < 0) or np.any(ids >= VOCAB_SIZE):
        raise ValueError("sparse array contains an invalid token")
    ids = ids.astype("<u2")
    positive = np.isfinite(values) & (values > 0)
    ids = ids[positive]
    values = values[positive]
    order = np.argsort(ids)
    ids = ids[order]
    values = values[order]
    return _HEAD.pack(len(ids)) + ids.tobytes() + values.tobytes()


def pack_sparse_batch(vectors):
    rows = [pack_sparse(vector) for vector in vectors]
    return _BATCH_HEAD.pack(BATCH_MAGIC, len(rows)) + b"".join(rows)


def pack_sparse_rows(rows):
    for row in rows:
        if len(row) < _HEAD.size:
            raise ValueError("sparse row is truncated")
        (count,) = _HEAD.unpack_from(row)
        if count > TOP_K or len(row) != _HEAD.size + count * 4:
            raise ValueError("invalid sparse row length")
    return _BATCH_HEAD.pack(BATCH_MAGIC, len(rows)) + b"".join(rows)


def unpack_sparse_batch(value):
    if len(value) < _BATCH_HEAD.size:
        raise ValueError("sparse batch is truncated")
    magic, count = _BATCH_HEAD.unpack_from(value)
    if magic != BATCH_MAGIC:
        raise ValueError("invalid sparse batch magic")
    offset = _BATCH_HEAD.size
    output = []
    for _ in range(count):
        if offset + _HEAD.size > len(value):
            raise ValueError("sparse batch row is truncated")
        (entries,) = _HEAD.unpack_from(value, offset)
        row_size = _HEAD.size + entries * 4
        if offset + row_size > len(value):
            raise ValueError("sparse batch row is truncated")
        output.append(unpack_sparse(value[offset:offset + row_size]))
        offset += row_size
    if offset != len(value):
        raise ValueError("sparse batch has trailing bytes")
    return output
