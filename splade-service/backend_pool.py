import asyncio
import itertools
import logging

import httpx

from constants import model_metadata


log = logging.getLogger(__name__)
EXPECTED_METADATA = model_metadata()
TRANSIENT = {408, 429, 500, 502, 503, 504}


class Backend:
    def __init__(
        self,
        backend_id,
        url,
        weight,
        max_concurrency,
        max_client_batch,
        timeout_s,
        api_key,
    ):
        self.id = backend_id
        self.url = url.rstrip("/")
        self.weight = weight
        self.max_client_batch = max_client_batch
        self.inflight = 0
        self.healthy = False
        self.draining = False
        self.failures = 0
        self.sem = asyncio.Semaphore(max_concurrency)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(
            base_url=self.url,
            headers=headers,
            timeout=timeout_s,
            limits=httpx.Limits(
                max_connections=max_concurrency * 2,
                max_keepalive_connections=max_concurrency,
            ),
        )

    async def verify(self):
        response = await self.client.get("/metadata", timeout=10)
        response.raise_for_status()
        metadata = response.json()
        mismatches = {
            key: (EXPECTED_METADATA[key], metadata.get(key))
            for key in EXPECTED_METADATA
            if metadata.get(key) != EXPECTED_METADATA[key]
        }
        if mismatches:
            raise ValueError(f"backend model contract mismatch: {mismatches}")
        self.healthy = True

    async def probe(self):
        try:
            await self.verify()
            self.failures = 0
        except Exception as exc:
            self.failures += 1
            if self.failures >= 2:
                self.healthy = False
            log.debug("backend %s probe failed: %s", self.id, exc)

    async def encode(self, texts):
        try:
            async with self.sem:
                response = await self.client.post(
                    "/encode", json={"inputs": texts, "document": True}
                )
                response.raise_for_status()
                vectors = response.json()
                if len(vectors) != len(texts) or not all(
                    isinstance(vector, dict) for vector in vectors
                ):
                    raise ValueError("backend returned an invalid vector batch")
                self.failures = 0
                self.healthy = True
                return vectors
        finally:
            self.inflight -= 1

    async def aclose(self):
        await self.client.aclose()

    def snapshot(self):
        return {
            "id": self.id,
            "url": self.url,
            "weight": self.weight,
            "healthy": self.healthy,
            "draining": self.draining,
            "inflight": self.inflight,
            "max_client_batch": self.max_client_batch,
        }


class BackendPool:
    def __init__(self, probe_interval_s=5):
        self.backends = {}
        self.ids = itertools.count(1)
        self.probe_interval_s = probe_interval_s
        self.probe_task = None
        self.lock = asyncio.Lock()

    async def add(
        self,
        url,
        weight=1,
        max_concurrency=1,
        max_client_batch=8,
        timeout_s=120,
        api_key="",
    ):
        backend = Backend(
            f"b{next(self.ids)}",
            url,
            weight,
            max_concurrency,
            max_client_batch,
            timeout_s,
            api_key,
        )
        try:
            await backend.verify()
        except Exception:
            await backend.aclose()
            raise
        async with self.lock:
            self.backends[backend.id] = backend
        return backend.snapshot()

    def start(self):
        self.probe_task = asyncio.create_task(self._probe_loop())

    async def aclose(self):
        if self.probe_task:
            self.probe_task.cancel()
            try:
                await self.probe_task
            except asyncio.CancelledError:
                pass
        await asyncio.gather(
            *(backend.aclose() for backend in self.backends.values()),
            return_exceptions=True,
        )

    def snapshots(self):
        return [backend.snapshot() for backend in self.backends.values()]

    def ready(self):
        return any(
            backend.healthy and not backend.draining
            for backend in self.backends.values()
        )

    async def set_weight(self, backend_id, weight):
        backend = self.backends.get(backend_id)
        if not backend:
            raise KeyError(backend_id)
        backend.weight = weight
        return backend.snapshot()

    async def remove(self, backend_id):
        backend = self.backends.get(backend_id)
        if not backend:
            raise KeyError(backend_id)
        backend.draining = True
        asyncio.create_task(self._drain(backend))
        return backend.snapshot()

    async def _drain(self, backend):
        for _ in range(600):
            if backend.inflight == 0:
                break
            await asyncio.sleep(0.1)
        async with self.lock:
            self.backends.pop(backend.id, None)
        await backend.aclose()

    def _select(self, excluded):
        candidates = [
            backend
            for backend in self.backends.values()
            if backend.healthy
            and not backend.draining
            and backend.id not in excluded
        ]
        if not candidates:
            return None
        weighted = [backend for backend in candidates if backend.weight > 0]
        candidates = weighted or candidates
        backend = min(
            candidates,
            key=lambda item: item.inflight / max(item.weight, 0.000001),
        )
        backend.inflight += 1
        return backend

    async def _encode_chunk(self, chunk):
        excluded = set()
        last_error = None
        for attempt in range(5):
            backend = self._select(excluded)
            if backend is None:
                if last_error:
                    raise last_error
                raise RuntimeError("no healthy SPLADE backend available")
            try:
                return await backend.encode(chunk)
            except Exception as exc:
                last_error = exc
                transient = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code in TRANSIENT
                )
                if not transient:
                    raise
                backend.failures += 1
                if backend.failures >= 2:
                    backend.healthy = False
                excluded.add(backend.id)
                if not any(
                    candidate.healthy
                    and not candidate.draining
                    and candidate.id not in excluded
                    for candidate in self.backends.values()
                ):
                    await asyncio.sleep(min(0.25 * (2 ** attempt), 2))
                    excluded.clear()
        raise last_error or RuntimeError("SPLADE backend retries exhausted")

    async def encode(self, texts):
        if not texts:
            return []
        sizes = [
            backend.max_client_batch
            for backend in self.backends.values()
            if not backend.draining
        ]
        chunk_size = min(sizes) if sizes else 8
        chunks = [
            texts[index:index + chunk_size]
            for index in range(0, len(texts), chunk_size)
        ]
        batches = await asyncio.gather(
            *(self._encode_chunk(chunk) for chunk in chunks)
        )
        return [vector for batch in batches for vector in batch]

    async def _probe_loop(self):
        while True:
            await asyncio.sleep(self.probe_interval_s)
            await asyncio.gather(
                *(backend.probe() for backend in list(self.backends.values())),
                return_exceptions=True,
            )
