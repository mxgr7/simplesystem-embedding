"""F7 §"RED metrics" — custom Prometheus metrics on top of
`prometheus-fastapi-instrumentator`'s built-in request-level R/E/D.

The instrumentator covers `http_requests_total{method,path,status}` +
`http_request_duration_seconds{method,path,le}` for free. F7 wants
those broken out by search-specific facets:

  - sort kind  (relevance | name | price_asc | price_desc | articleId)
  - dispatch route (path_a | path_b | path_b_overflow_fallback)
  - has_summaries (true | false)

plus a counter for fallback fires (Path B → A overflow) and another
for hitCount-cap clipping. Each label has bounded cardinality (≤5)
so the cross-product stays small (~30 series per metric).

Modules call into `record_search` once per request — the helper
fans out to the duration histogram + request counter. Errors go to
`record_search_error` instead.

Module-level instances so `from metrics import REGISTRY` returns the
same objects across imports — needed for the FastAPI middleware to
register them on startup.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# ---- label sets (bounded cardinality) ----------------------------------

# Mirrors `models.SortDirection` × `models.SortClause` enums.
SORT_LABELS: tuple[str, ...] = (
    "relevance",
    "name_asc", "name_desc",
    "price_asc", "price_desc",
    "articleId_asc", "articleId_desc",
)

# Path B overflow falls back to Path A but is logically a different
# code path — surface it explicitly so operators can spot the recall
# cliff in the dashboards.
ROUTE_LABELS: tuple[str, ...] = (
    "path_a",
    "path_b",
    "path_b_overflow_fallback",
    "filter_only_browse",       # no query string, no offer expr
    "legacy_single_collection", # USE_DEDUP_TOPOLOGY=false
)

HAS_SUMMARIES_LABELS: tuple[str, ...] = ("true", "false")

# Histogram buckets — geometric from 5ms to 30s, covering the legacy
# SLO range (p50 < 1s, p99 < 5s) with enough resolution to spot
# bimodal distributions.
DURATION_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
)


# ---- metrics -----------------------------------------------------------

search_requests_total = Counter(
    "ftsearch_search_requests_total",
    "Successful /search requests, broken out by sort × route × has_summaries.",
    labelnames=("sort", "route", "has_summaries"),
)

search_duration_seconds = Histogram(
    "ftsearch_search_duration_seconds",
    "End-to-end /search request duration.",
    labelnames=("sort", "route", "has_summaries"),
    buckets=DURATION_BUCKETS,
)

search_errors_total = Counter(
    "ftsearch_search_errors_total",
    "Failed /search requests by error kind (status_4xx | status_5xx | exception).",
    labelnames=("sort", "route", "error_kind"),
)

# Fired when Path B's bounded probe returns more than PATH_B_HASH_LIMIT
# distinct hashes and we fall back to Path A — recall is clipped.
# Operators watch this for systemic over-permissive offer filters.
recall_clipped_total = Counter(
    "ftsearch_recall_clipped_total",
    "Path B → Path A overflow events (recall_clipped=true responses).",
)

# Fired when hitCount = HITCOUNT_CAP — operators may want to bump the
# cap or accept that >cap hits is "many" rather than a precise number.
hitcount_clipped_total = Counter(
    "ftsearch_hitcount_clipped_total",
    "/search responses where the hitCount cap fired.",
)


# ---- helpers -----------------------------------------------------------

def _sort_label(sort_clauses: list) -> str:
    """Coerce the parsed sort plan into a single label. Multi-key
    sorts use only the first key per spec §4.5; same here for the
    label."""
    if not sort_clauses:
        return "relevance"
    first = sort_clauses[0]
    field = getattr(first, "field", None)
    direction = getattr(first, "direction", None)
    if field is None:
        return "relevance"
    field_name = getattr(field, "value", str(field))
    if field_name == "relevance":
        return "relevance"
    direction_name = getattr(direction, "value", "asc")
    label = f"{field_name}_{direction_name}"
    return label if label in SORT_LABELS else "relevance"


def record_search(
    *,
    sort_clauses: list,
    route: str,
    has_summaries: bool,
    duration_s: float,
    recall_clipped: bool = False,
    hit_count_clipped: bool = False,
) -> None:
    """Emit the per-request metrics for a successful /search response.
    Call once per response, post-dispatch."""
    sort_label = _sort_label(sort_clauses)
    summaries_label = "true" if has_summaries else "false"
    route_label = route if route in ROUTE_LABELS else "path_a"  # safe default
    search_requests_total.labels(
        sort=sort_label, route=route_label, has_summaries=summaries_label,
    ).inc()
    search_duration_seconds.labels(
        sort=sort_label, route=route_label, has_summaries=summaries_label,
    ).observe(duration_s)
    if recall_clipped:
        recall_clipped_total.inc()
    if hit_count_clipped:
        hitcount_clipped_total.inc()


def record_search_error(
    *,
    sort_clauses: list,
    route: str,
    error_kind: str,
) -> None:
    """Emit error counter on a failed /search response. `error_kind` is
    a small bounded vocabulary: `status_4xx`, `status_5xx`, `exception`."""
    sort_label = _sort_label(sort_clauses)
    route_label = route if route in ROUTE_LABELS else "unknown"
    search_errors_total.labels(
        sort=sort_label, route=route_label, error_kind=error_kind,
    ).inc()


__all__ = [
    "record_search",
    "record_search_error",
    "search_requests_total",
    "search_duration_seconds",
    "search_errors_total",
    "recall_clipped_total",
    "hitcount_clipped_total",
    "SORT_LABELS",
    "ROUTE_LABELS",
    "HAS_SUMMARIES_LABELS",
]
