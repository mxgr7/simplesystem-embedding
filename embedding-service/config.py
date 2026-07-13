"""Env-driven config for the embedding-service.

Centralises all environment-variable reads so the rest of the service
treats config as a dataclass rather than scattering `os.environ.get(...)`
calls. Loaded once at FastAPI lifespan startup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    tei_url: str
    kvrocks_url: str
    api_key: str
    admin_api_key: str
    port: int
    tei_max_client_batch: int
    tei_max_concurrency: int
    tei_weight: float
    tei_timeout_s: float
    tei_probe_interval_s: float
    tei_drain_timeout_s: float
    kvrocks_read_timeout_ms: int
    kvrocks_max_connections: int
    max_inflight: int
    max_inputs_per_request: int
    request_budget_s: float
    retry_after_s: float
    readyz_tei_timeout_s: float
    readyz_cache_timeout_s: float


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def load_config() -> Config:
    tei_max_concurrency = _int("TEI_MAX_CONCURRENCY", 16)
    return Config(
        tei_url=os.environ.get("TEI_URL", "http://localhost:8080").rstrip("/"),
        kvrocks_url=os.environ.get("KVROCKS_URL", "redis://localhost:6666/0"),
        api_key=os.environ.get("API_KEY", ""),
        # Admin endpoints (/admin/backends) authenticate against this key;
        # falls back to API_KEY when unset, so a single key still works.
        admin_api_key=os.environ.get("ADMIN_API_KEY", ""),
        port=_int("PORT", 8082),
        tei_max_client_batch=_int("TEI_MAX_CLIENT_BATCH", 8),
        tei_max_concurrency=tei_max_concurrency,
        # Bootstrap backend (seeded from TEI_URL) routing defaults.
        tei_weight=_float("TEI_WEIGHT", 1.0),
        tei_timeout_s=_float("TEI_TIMEOUT_S", 30.0),
        tei_probe_interval_s=_float("TEI_PROBE_INTERVAL_S", 5.0),
        tei_drain_timeout_s=_float("TEI_DRAIN_TIMEOUT_S", 30.0),
        kvrocks_read_timeout_ms=_int("KVROCKS_READ_TIMEOUT_MS", 50),
        kvrocks_max_connections=_int("KVROCKS_MAX_CONNECTIONS", 64),
        # 8× TEI concurrency = ~8 batch-rounds of buffering before we 429.
        max_inflight=_int("MAX_INFLIGHT", 8 * tei_max_concurrency),
        max_inputs_per_request=_int("MAX_INPUTS_PER_REQUEST", 256),
        request_budget_s=_float("REQUEST_BUDGET_S", 5.0),
        retry_after_s=_float("RETRY_AFTER_S", 0.3),
        # /readyz check budgets. The TEI canary fails fast instead of
        # running the pool's full 5s retry budget — the poller retries
        # anyway. The cache scan gets more than the 50ms hot-path read
        # timeout: right after reboot the first SCAN hits cold disk.
        readyz_tei_timeout_s=_float("READYZ_TEI_TIMEOUT_S", 3.0),
        readyz_cache_timeout_s=_float("READYZ_CACHE_TIMEOUT_S", 2.0),
    )
