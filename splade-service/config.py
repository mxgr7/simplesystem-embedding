import os


class Config:
    def __init__(self):
        self.kvrocks_url = os.environ.get(
            "KVROCKS_URL", "redis://localhost:6666/0"
        )
        self.backend_urls = [
            value.strip().rstrip("/")
            for value in os.environ.get(
                "BACKEND_URLS", "http://localhost:8138"
            ).split(",")
            if value.strip()
        ]
        self.backend_api_key = os.environ.get("BACKEND_API_KEY", "")
        self.api_key = os.environ.get("API_KEY", "")
        self.admin_api_key = os.environ.get("ADMIN_API_KEY", "")
        self.max_inputs = int(os.environ.get("MAX_INPUTS_PER_REQUEST", "256"))
        self.max_inflight = int(os.environ.get("MAX_INFLIGHT", "32"))
        self.request_budget_s = float(os.environ.get("REQUEST_BUDGET_S", "120"))
        self.cache_read_timeout_s = float(
            os.environ.get("KVROCKS_READ_TIMEOUT_MS", "100")
        ) / 1000
        self.cache_connections = int(
            os.environ.get("KVROCKS_MAX_CONNECTIONS", "64")
        )
        self.probe_interval_s = float(
            os.environ.get("BACKEND_PROBE_INTERVAL_S", "5")
        )
        # Recovery knobs (MXG-166), ported from the dense wrapper's MXG-159 fix.
        # A probe is bounded outside httpx so a wedged connection pool cannot
        # stop recovery; a backend that stays unhealthy that long gets a fresh
        # HTTP client; while nothing is selectable, one trial per interval is let
        # through to find out whether the backend is back. Defaults are the
        # intended production values, so compose does not set them.
        self.probe_timeout_s = float(
            os.environ.get("BACKEND_PROBE_TIMEOUT_S", "2")
        )
        self.probe_round_timeout_s = float(
            os.environ.get("BACKEND_PROBE_ROUND_TIMEOUT_S", "10")
        )
        self.unhealthy_after = int(
            os.environ.get("BACKEND_UNHEALTHY_AFTER", "2")
        )
        self.half_open_interval_s = float(
            os.environ.get("BACKEND_HALF_OPEN_INTERVAL_S", "5")
        )
        self.client_recycle_after_s = float(
            os.environ.get("BACKEND_CLIENT_RECYCLE_AFTER_S", "60")
        )
        self.pool_timeout_recycle_after = int(
            os.environ.get("BACKEND_POOL_TIMEOUT_RECYCLE_AFTER", "3")
        )
        # Retry-After on the 503 a no-healthy-backend request now gets instead of
        # an unhandled 500.
        self.retry_after_s = float(os.environ.get("RETRY_AFTER_S", "1"))
        # How BACKEND_URLS are registered at startup. These exist because the
        # BackendPool.add defaults (8 / 1) are the wrong shape for a real client and
        # were only ever corrected at runtime through POST /admin/backends, which a
        # restart silently reverts. BackendPool.encode chunks by the min
        # max_client_batch across non-draining backends, so a batch of 8 splits a
        # 128-input indexer request into 16 chunks serialized behind a semaphore of
        # 1, which pegs inflight at MAX_INFLIGHT and sheds the surplus as 429.
        self.backend_max_client_batch = int(
            os.environ.get("BACKEND_MAX_CLIENT_BATCH", "64")
        )
        self.backend_pool_concurrency = int(
            os.environ.get("BACKEND_POOL_CONCURRENCY", "3")
        )
