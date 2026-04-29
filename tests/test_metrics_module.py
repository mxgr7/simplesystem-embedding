"""Unit tests for `search-api/metrics.py`.

The module wraps Prometheus counters/histograms with bounded labels.
We don't assert on the wire format — just that the helpers route to
the right series, label cardinality stays bounded, and the sort-clause
classifier produces a stable label per parsed plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "search-api"))

import pytest  # noqa: E402

from metrics import (  # noqa: E402
    HAS_SUMMARIES_LABELS,
    ROUTE_LABELS,
    SORT_LABELS,
    _sort_label,
    hitcount_clipped_total,
    recall_clipped_total,
    record_search,
    record_search_error,
    search_duration_seconds,
    search_errors_total,
    search_requests_total,
)
from models import SortClause, SortDirection  # noqa: E402


def _counter_value(c) -> float:
    return c._value.get()


def _histogram_observation_count(metric, **labels) -> float:
    """Total observations for a labelled histogram.
    `prometheus_client` stores per-bucket counts non-cumulatively, so
    summing every bucket gives the total observation count."""
    return sum(b.get() for b in metric.labels(**labels)._buckets)


def test_label_sets_have_bounded_cardinality() -> None:
    """Cardinality discipline — each label set stays small enough that
    the cross-product per metric stays under ~100 series."""
    assert len(SORT_LABELS) == 7
    assert len(ROUTE_LABELS) == 5
    assert len(HAS_SUMMARIES_LABELS) == 2


def test_sort_label_relevance_default() -> None:
    assert _sort_label([]) == "relevance"


def test_sort_label_named_sort() -> None:
    sort = SortClause(field="name", direction=SortDirection.ASC)
    assert _sort_label([sort]) == "name_asc"


def test_sort_label_only_first_clause_wins() -> None:
    """Multi-key sorts use only the first key per spec §4.5; metrics
    label follows the same rule."""
    s1 = SortClause(field="price", direction=SortDirection.DESC)
    s2 = SortClause(field="name", direction=SortDirection.ASC)
    assert _sort_label([s1, s2]) == "price_desc"


def test_record_search_increments_counter_and_histogram() -> None:
    """A single record_search call must bump the request counter for
    its label tuple AND observe a sample on the duration histogram
    for the same tuple."""
    labels = dict(sort="relevance", route="path_a", has_summaries="false")
    before_count = search_requests_total.labels(**labels)._value.get()
    before_obs = _histogram_observation_count(search_duration_seconds, **labels)

    # Empty sort clauses == relevance default, matching the labels above.
    record_search(
        sort_clauses=[], route="path_a", has_summaries=False,
        duration_s=0.123,
    )
    assert search_requests_total.labels(**labels)._value.get() == before_count + 1
    assert _histogram_observation_count(search_duration_seconds, **labels) == before_obs + 1


def test_record_search_unknown_route_falls_back_to_path_a() -> None:
    """An unrecognised route label must NOT explode the cardinality —
    the helper coerces to a known label."""
    before = search_requests_total.labels(
        sort="relevance", route="path_a", has_summaries="false",
    )._value.get()
    record_search(
        sort_clauses=[], route="totally_made_up_route", has_summaries=False,
        duration_s=0.0,
    )
    after = search_requests_total.labels(
        sort="relevance", route="path_a", has_summaries="false",
    )._value.get()
    assert after == before + 1


def test_record_search_error_counts_kinds_separately() -> None:
    """Errors keep the sort + route label so dashboards can break out
    error spikes the same way as success requests."""
    sort = SortClause(field="articleId", direction=SortDirection.DESC)
    labels_4xx = dict(sort="articleId_desc", route="unknown", error_kind="status_4xx")
    labels_5xx = dict(sort="articleId_desc", route="unknown", error_kind="status_5xx")
    before_4xx = search_errors_total.labels(**labels_4xx)._value.get()
    before_5xx = search_errors_total.labels(**labels_5xx)._value.get()

    record_search_error(sort_clauses=[sort], route="unknown", error_kind="status_4xx")
    record_search_error(sort_clauses=[sort], route="unknown", error_kind="status_5xx")
    record_search_error(sort_clauses=[sort], route="unknown", error_kind="status_5xx")

    assert search_errors_total.labels(**labels_4xx)._value.get() == before_4xx + 1
    assert search_errors_total.labels(**labels_5xx)._value.get() == before_5xx + 2


def test_recall_clipped_counter_fires_when_flag_true() -> None:
    before = _counter_value(recall_clipped_total)
    record_search(
        sort_clauses=[], route="path_b_overflow_fallback",
        has_summaries=False, duration_s=0.0, recall_clipped=True,
    )
    assert _counter_value(recall_clipped_total) == before + 1


def test_hitcount_clipped_counter_fires_when_flag_true() -> None:
    before = _counter_value(hitcount_clipped_total)
    record_search(
        sort_clauses=[], route="path_a", has_summaries=False,
        duration_s=0.0, hit_count_clipped=True,
    )
    assert _counter_value(hitcount_clipped_total) == before + 1


def test_recall_and_hitcount_dont_fire_when_false() -> None:
    """Default flags shouldn't accidentally bump the clip counters."""
    before_r = _counter_value(recall_clipped_total)
    before_h = _counter_value(hitcount_clipped_total)
    record_search(
        sort_clauses=[], route="path_a", has_summaries=False, duration_s=0.0,
    )
    assert _counter_value(recall_clipped_total) == before_r
    assert _counter_value(hitcount_clipped_total) == before_h
