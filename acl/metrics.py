"""ACL RED metrics — supplements `prometheus-fastapi-instrumentator`'s
default request-level R/E/D with ACL-specific counters per F7-style
spec.

Cardinality discipline: every label has a small bounded set so the
cross-product per metric stays under ~50 series.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# `outcome`: how the ftsearch call ended. Bounded to a tiny vocabulary
# so dashboards can split errors from successes without touching
# upstream status codes.
OUTCOME_LABELS: tuple[str, ...] = (
    "success", "upstream_4xx", "upstream_5xx", "network_error", "exhausted",
)

# Histogram buckets — sub-millisecond floor for fast cache-hit-style
# responses, top out past the legacy SLO so an over-budget request
# is still bucketed instead of clamped.
LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
)


# Per-call latency to ftsearch (excludes any retry backoff sleeps —
# observed once per attempt). Useful for SLO dashboards that need to
# isolate the upstream from the ACL's own overhead.
ftsearch_call_duration_seconds = Histogram(
    "acl_ftsearch_call_duration_seconds",
    "Per-attempt latency from ACL → ftsearch (excludes retry backoffs).",
    labelnames=("outcome",),
    buckets=LATENCY_BUCKETS,
)

# One increment per retry attempt (NOT per request) — the difference
# tells operators how chatty the retry chain is in steady state.
ftsearch_retries_fired_total = Counter(
    "acl_ftsearch_retries_fired_total",
    "Number of retry attempts fired against ftsearch (excludes initial attempt).",
)

# Fired once per request when the retry chain exhausts and the ACL
# gives up — the leading indicator for an upstream degradation.
ftsearch_retries_exhausted_total = Counter(
    "acl_ftsearch_retries_exhausted_total",
    "Number of /article-features/search requests that ran out of retries.",
)


def record_call(outcome: str, duration_s: float) -> None:
    """Emit per-attempt latency + outcome bucket."""
    bucket = outcome if outcome in OUTCOME_LABELS else "network_error"
    ftsearch_call_duration_seconds.labels(outcome=bucket).observe(duration_s)


def record_retry_fired() -> None:
    ftsearch_retries_fired_total.inc()


def record_retry_exhausted() -> None:
    ftsearch_retries_exhausted_total.inc()


__all__ = [
    "ftsearch_call_duration_seconds",
    "ftsearch_retries_fired_total",
    "ftsearch_retries_exhausted_total",
    "record_call",
    "record_retry_fired",
    "record_retry_exhausted",
    "OUTCOME_LABELS",
]
