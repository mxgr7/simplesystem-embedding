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
