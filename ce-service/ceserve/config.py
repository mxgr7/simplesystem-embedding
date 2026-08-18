"""Environment in one place, mirroring `splade-service/config.py`.

Values are read once at lifespan start. Anything that is a *contract* rather
than a tuning knob lives in `constants.py`; anything asserted at boot is checked
in `scorer.py` so the failure message can name both the value and the env var.
"""
import os

from ceserve.constants import TOKENIZER_VERSION


def _flag(name, default):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    def __init__(self):
        self.model_dir = os.environ.get("CE_MODEL_DIR", "/model")
        self.device = os.environ.get("CE_DEVICE", "cuda").strip().lower()
        self.allow_cpu = _flag("CE_ALLOW_CPU", "0")
        self.dtype = os.environ.get("CE_DTYPE", "fp16").strip().lower()
        self.max_len = int(os.environ.get("CE_MAX_LEN", "192"))
        # One forward at k <= 256, which is the shape 0.654 ms/candidate was
        # measured in. Chunking above that is a memory guard, not a batching
        # strategy.
        self.forward_chunk = int(os.environ.get("CE_FORWARD_CHUNK", "256"))
        self.assert_golden = _flag("CE_ASSERT_GOLDEN", "1")
        self.tokenizer_version = TOKENIZER_VERSION

        self.api_key = os.environ.get("API_KEY", "")
        # 413 above this. 0.654 * 256 + 0.3 = ~167 ms, inside the response
        # budget below; assertion 12 in scorer.py enforces that relationship
        # rather than trusting these two defaults to stay consistent.
        self.max_inputs = int(os.environ.get("MAX_INPUTS_PER_REQUEST", "256"))
        # NOT 32. One forward in flight plus one assembling. 32 queues ~2.5 s of
        # latent work behind a 150 ms budget, which is worse than a fast 429:
        # the Kotlin caller degrades to upstream order on a refusal, but waits
        # on a queue.
        self.max_inflight = int(os.environ.get("MAX_INFLIGHT", "2"))
        # A RESPONSE deadline, not a work deadline — see the comment on
        # `app._rerank`. Above the worst legal request (~167 ms) and well below
        # the Kotlin client's timeout, so a slow request is refused by the
        # server with a diagnosable 504 rather than by the client with an
        # opaque socket timeout.
        self.request_budget_s = float(os.environ.get("REQUEST_BUDGET_S", "0.5"))
        # Mirrors the query service's `rerank.fallback.max-missing-content-ratio`.
        # A single corrupt blob is a skipped candidate; wholesale decode failure
        # is a contract break and becomes a 400.
        self.max_decode_failure_ratio = float(
            os.environ.get("CE_MAX_DECODE_FAILURE_RATIO", "0.5")
        )
        self.torch_num_threads = int(os.environ.get("TORCH_NUM_THREADS", "2"))
