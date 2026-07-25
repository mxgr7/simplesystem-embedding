import hashlib

from constants import CACHE_PREFIX


def input_hash(value):
    return hashlib.sha256(value.encode("utf-8")).digest()[:16].hex()


def cache_key(value_hash):
    return f"{CACHE_PREFIX}{value_hash}"
