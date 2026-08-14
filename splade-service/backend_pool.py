import asyncio
import itertools
import logging

import httpx

from constants import ENCODING_VERSION, model_metadata


log = logging.getLogger(__name__)
EXPECTED_METADATA = model_metadata()
TRANSIENT = {408, 429, 500, 502, 503, 504}

# `model_metadata()` pins WHICH CHECKPOINT a backend serves. It says nothing about
# HOW that checkpoint is executed, so two backends can pass it while producing
# different vectors -- an H100 on bf16 and a T4 on fp16 differ here and agree on
# every key above. The reindex client already pins these across backends
# (`validate_backends`); the serving pool did not, which is how a burst backend
# could land in the pool and write its vectors into the shared cache keyspace.
ENCODER_CONTRACT = ("document_compute_dtype", "document_encoding_version",
                    "fold_vocab_mask", "vocab_mask_sha256")


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
        # Last payload seen by verify(). Kept rather than discarded so the frontend can answer
        # "which checkpoint is actually being served" without a second round trip: a client that
        # only ever calls /embed has no other way to tell, and the encoder-identity fields
        # (document_encoding_version, fold_vocab_mask, vocab_mask_sha256) exist only here.
        self.metadata = {}
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
        self.metadata = metadata
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

    async def encode(self, texts, document=True):
        try:
            async with self.sem:
                response = await self.client.post(
                    "/encode", json={"inputs": texts, "document": document}
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
            self._check_encoder_contract(backend)
        except Exception:
            await backend.aclose()
            raise
        async with self.lock:
            self.backends[backend.id] = backend
        return backend.snapshot()

    def _check_encoder_contract(self, backend):
        """Reject a backend that executes the pinned checkpoint differently.

        Two places can define the expected contract, and both are checked because
        they fail in different ways. `SPLADE_ENCODING_VERSION` is the operator's
        declaration and also what the cache keyspace is namespaced by, so a
        mismatch there means cached vectors would be filed under a name that does
        not describe them. An already-registered backend is the empirical one: the
        first backend in defines the contract for the rest, exactly as
        `validate_backends` does on the reindex side.
        """
        got = {key: backend.metadata.get(key) for key in ENCODER_CONTRACT}
        declared = ENCODING_VERSION
        if declared and got["document_encoding_version"] != declared:
            raise ValueError(
                f"backend {backend.url} encodes as "
                f"{got['document_encoding_version']!r} but SPLADE_ENCODING_VERSION "
                f"declares {declared!r}; the cache keyspace is namespaced by the "
                "declared value, so this would file its vectors under the wrong name"
            )
        for other in self.backends.values():
            expected = {key: other.metadata.get(key) for key in ENCODER_CONTRACT}
            if got != expected:
                differing = {k: (expected[k], got[k]) for k in got if got[k] != expected[k]}
                raise ValueError(
                    f"backend {backend.url} encoder contract differs from "
                    f"{other.url}: {differing}"
                )
            break

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

    def contracts(self):
        """Encoder identity per backend, for the frontend's GET /metadata.

        Deliberately narrow: these are the keys that decide whether a vector produced now is
        comparable with one already in the index. `model_id`/`model_sha256` name the checkpoint;
        `document_encoding_version` names the encoder build (dtype, codec, compiled head);
        `fold_vocab_mask` plus `vocab_mask_sha256` pin the exact kept-dimension set, which two
        booleans could not. Tuning knobs (batch size, overlap, device) are left out on purpose —
        a client asserting on those would break on a harmless redeploy.

        Backends that have never verified report an empty contract rather than being omitted, so a
        half-up pool is visible rather than silently looking like a healthy smaller one.
        """
        keys = (
            "model_id",
            "model_sha256",
            "document_encoding_version",
            "fold_vocab_mask",
            "vocab_mask_sha256",
        )
        return [
            {
                "id": backend.id,
                "healthy": backend.healthy,
                "draining": backend.draining,
                **{key: backend.metadata.get(key) for key in keys},
            }
            for backend in self.backends.values()
        ]

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

    async def _encode_chunk(self, chunk, document=True):
        excluded = set()
        last_error = None
        for attempt in range(5):
            backend = self._select(excluded)
            if backend is None:
                if last_error:
                    raise last_error
                raise RuntimeError("no healthy SPLADE backend available")
            try:
                return await backend.encode(chunk, document=document)
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

    async def encode(self, texts, document=True):
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
            *(
                self._encode_chunk(chunk, document=document)
                for chunk in chunks
            )
        )
        return [vector for batch in batches for vector in batch]

    async def _probe_loop(self):
        while True:
            await asyncio.sleep(self.probe_interval_s)
            await asyncio.gather(
                *(backend.probe() for backend in list(self.backends.values())),
                return_exceptions=True,
            )
