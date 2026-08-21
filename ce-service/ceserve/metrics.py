"""Prometheus surface.

Bucket boundaries are chosen around the 150 ms fine-ranker budget rather than
prometheus_client's default `[.005, .01, .025, ...]` scale, which puts every
real CE request in the `+Inf` bucket and makes p95 unquantifiable exactly where
it matters.

MXG-144.
"""
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily

# Seconds, bracketing the budget: 78 ms is the measured p50 for a 120-candidate
# window and 150 ms is the ceiling, so the interesting resolution is 0.05-0.25.
LATENCY_BUCKETS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.15, 0.25, 0.5, 1.0)

REQUESTS = Counter(
    "ce_service_requests_total", "Rerank requests by response status", ("status",)
)
CANDIDATES = Counter(
    "ce_service_candidates_total",
    "Candidates by disposition",
    ("outcome",),  # scored | skipped_version | skipped_decode | skipped_no_tokens
)
INFLIGHT = Gauge("ce_service_inflight", "Rerank requests being processed")

TOTAL_MS = Histogram(
    "ce_service_request_seconds",
    "End-to-end /rerank handler time",
    buckets=LATENCY_BUCKETS,
)
FORWARD_MS = Histogram(
    "ce_service_forward_seconds",
    "Model forward time (splice + GPU), excluding JSON and HTTP",
    buckets=LATENCY_BUCKETS,
)
ASSEMBLE_MS = Histogram(
    "ce_service_decode_seconds",
    "Base64 decode + token validation, per request",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
)

CANDIDATES_PER_REQUEST = Histogram(
    "ce_service_candidates_per_request",
    "Candidates in one /rerank call",
    buckets=(1, 10, 30, 60, 120, 160, 200, 229, 256, 512),
)
PADDED_WIDTH = Histogram(
    "ce_service_padded_width",
    "Padded sequence width of the widest chunk in a request",
    # THE latency driver, and the one a request-duration histogram cannot
    # explain: a window whose p50 padded width drifts from 154 to 192 costs 25%
    # more with no other metric moving.
    buckets=(32, 64, 96, 128, 160, 192, 256, 384, 512),
)


class CeRuntimeCollector:
    """Scrape-time runtime state: readiness, degradation, GPU memory.

    A collector rather than Gauges written from the request path, for the reason
    `splade-service`'s `BackendHealthCollector` docstring gives: a gauge is only
    as fresh as whatever writes it, so a process that has stopped serving would
    keep reporting its last healthy value forever — which is precisely the
    failure this metric exists to catch. `ce_service_requests_total` cannot
    cover it either, because it only moves when a client asks for something, and
    a CE that dies between search bursts is invisible until the next one.
    """

    def __init__(self, app):
        self.app = app

    def collect(self):
        scorer = getattr(self.app.state, "scorer", None)
        ready = GaugeMetricFamily(
            "ce_service_ready",
            "1 if the model is loaded, warmed up and not degraded",
        )
        degraded = GaugeMetricFamily(
            "ce_service_degraded",
            "1 if a forward raised and the CUDA context is presumed poisoned",
        )
        ready.add_metric(
            [], float(bool(scorer is not None
                           and getattr(scorer, "warmed_up", False)
                           and not getattr(scorer, "degraded", True)))
        )
        degraded.add_metric([], float(bool(getattr(scorer, "degraded", False))))
        yield ready
        yield degraded

        allocated = GaugeMetricFamily(
            "ce_service_gpu_memory_allocated_bytes", "torch.cuda.memory_allocated"
        )
        reserved = GaugeMetricFamily(
            "ce_service_gpu_memory_reserved_bytes", "torch.cuda.memory_reserved"
        )
        # `.get`-style defensiveness throughout: a raise inside a collector makes
        # /metrics return 500, which Prometheus reads as the whole job being
        # down — a false alarm that would mask the real ones.
        try:
            import torch

            if scorer is not None and scorer.device.type == "cuda":
                index = scorer.device.index or 0
                allocated.add_metric([], float(torch.cuda.memory_allocated(index)))
                reserved.add_metric([], float(torch.cuda.memory_reserved(index)))
        except Exception:  # pragma: no cover - defensive
            pass
        yield allocated
        yield reserved
