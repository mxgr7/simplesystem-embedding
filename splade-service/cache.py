import asyncio
import logging

import redis.asyncio as aioredis

from codec import unpack_sparse
from hashing import cache_key


log = logging.getLogger(__name__)


class SparseCache:
    def __init__(self, url, read_timeout_s, max_connections):
        self.client = aioredis.from_url(
            url,
            decode_responses=False,
            max_connections=max_connections,
        )
        self.read_timeout_s = read_timeout_s
        self.hits = 0
        self.misses = 0
        self.read_errors = 0
        self.write_errors = 0

    async def aclose(self):
        await self.client.aclose()

    async def ping(self):
        return await self.client.ping()

    async def mget(self, hashes):
        if not hashes:
            return []
        try:
            values = await asyncio.wait_for(
                self.client.mget([cache_key(value) for value in hashes]),
                timeout=self.read_timeout_s,
            )
        except (asyncio.TimeoutError, aioredis.RedisError) as exc:
            self.read_errors += 1
            log.warning("KVRocks MGET failed: %s", exc)
            return [None] * len(hashes)

        output = []
        for value in values:
            if value is None:
                output.append(None)
                continue
            try:
                unpack_sparse(value)
            except ValueError as exc:
                log.warning("ignoring malformed SPLADE cache value: %s", exc)
                output.append(None)
            else:
                output.append(value)
        return output

    async def mset(self, values):
        if not values:
            return
        try:
            pipe = self.client.pipeline(transaction=False)
            for value_hash, packed in values.items():
                pipe.set(cache_key(value_hash), packed)
            await pipe.execute()
        except aioredis.RedisError as exc:
            self.write_errors += 1
            log.warning("KVRocks MSET failed: %s", exc)
