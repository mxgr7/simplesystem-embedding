"""Unit tests for `search-api/tracing.py`.

Validates W3C trace-context parsing + forwarding:
  - Well-formed `traceparent` extracts trace_id + span_id.
  - Malformed / all-zero / missing → graceful (None values; no error).
  - `baggage` header parsed; property suffixes stripped.
  - `headers_for_forwarding` only echoes the documented baggage subset
    (userId, companyId, customerOciSessionId) — no leakage of arbitrary
    upstream baggage entries.
  - `log_request_context` emits exactly one log line when trace_id
    present, zero otherwise.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "search-api"))

import pytest  # noqa: E402

from tracing import (  # noqa: E402
    PROPAGATED_BAGGAGE_FIELDS,
    extract_trace_context,
    log_request_context,
    parse_baggage,
    parse_traceparent,
)


# ---- traceparent --------------------------------------------------------

def test_traceparent_well_formed() -> None:
    tp = "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01"
    trace_id, span_id, flags = parse_traceparent(tp)
    assert trace_id == "1234567890abcdef1234567890abcdef"
    assert span_id == "1234567890abcdef"
    assert flags == "01"


def test_traceparent_missing_returns_nones() -> None:
    assert parse_traceparent(None) == (None, None, None)
    assert parse_traceparent("") == (None, None, None)


def test_traceparent_malformed_returns_nones() -> None:
    """Wrong number of segments / non-hex chars / wrong segment lengths
    all reject."""
    for bad in (
        "garbage",
        "00-1234-5678-01",                      # too-short fields
        "00-1234567890abcdef1234567890abcdef-1234567890abcdef",  # missing flags
        "ZZ-1234567890abcdef1234567890abcdef-1234567890abcdef-01",  # non-hex
    ):
        assert parse_traceparent(bad) == (None, None, None), f"should reject {bad!r}"


def test_traceparent_all_zero_ids_invalid() -> None:
    """Per W3C spec — all-zero trace_id or span_id is invalid."""
    bad_trace = "00-" + "0" * 32 + "-1234567890abcdef-01"
    bad_span = "00-1234567890abcdef1234567890abcdef-" + "0" * 16 + "-01"
    assert parse_traceparent(bad_trace) == (None, None, None)
    assert parse_traceparent(bad_span) == (None, None, None)


# ---- baggage ------------------------------------------------------------

def test_baggage_parses_simple_pairs() -> None:
    assert parse_baggage("userId=42,companyId=7") == {"userId": "42", "companyId": "7"}


def test_baggage_strips_property_suffixes() -> None:
    """`key=value;ttl=60` → drop the `;ttl=60` part."""
    assert parse_baggage("userId=42;ttl=60,companyId=7") == {"userId": "42", "companyId": "7"}


def test_baggage_skips_malformed_entries() -> None:
    assert parse_baggage("userId=42,malformed,companyId=7") == {
        "userId": "42", "companyId": "7",
    }


def test_baggage_empty_or_none() -> None:
    assert parse_baggage(None) == {}
    assert parse_baggage("") == {}


# ---- extract_trace_context ----------------------------------------------

def test_extract_from_dict_headers() -> None:
    ctx = extract_trace_context({
        "traceparent": "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01",
        "baggage": "userId=99,companyId=7,unknown=junk",
    })
    assert ctx.trace_id == "1234567890abcdef1234567890abcdef"
    assert ctx.baggage == {"userId": "99", "companyId": "7", "unknown": "junk"}


def test_extract_case_insensitive_header_lookup() -> None:
    """HTTP headers are case-insensitive — Starlette mixed-cases them."""
    ctx = extract_trace_context({
        "Traceparent": "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01",
        "BAGGAGE": "userId=99",
    })
    assert ctx.trace_id is not None
    assert ctx.baggage["userId"] == "99"


def test_extract_with_no_trace_headers_returns_empty_ctx() -> None:
    ctx = extract_trace_context({})
    assert ctx.trace_id is None
    assert ctx.baggage == {}


# ---- headers_for_forwarding --------------------------------------------

def test_forwarding_includes_traceparent_and_subset_of_baggage() -> None:
    ctx = extract_trace_context({
        "traceparent": "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01",
        "tracestate": "vendor=blob",
        "baggage": "userId=42,companyId=7,customerOciSessionId=abc,internal=secret",
    })
    out = ctx.headers_for_forwarding()
    assert out["traceparent"] == "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01"
    assert out["tracestate"] == "vendor=blob"
    # Only the propagated subset; arbitrary upstream baggage stays out.
    forwarded = parse_baggage(out["baggage"])
    assert forwarded == {"userId": "42", "companyId": "7", "customerOciSessionId": "abc"}
    assert "internal" not in forwarded


def test_forwarding_omits_baggage_header_when_no_propagated_fields() -> None:
    """If the baggage carries only fields outside the documented set,
    we don't emit a baggage header — saves a few bytes per outbound."""
    ctx = extract_trace_context({
        "traceparent": "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01",
        "baggage": "irrelevant=junk",
    })
    out = ctx.headers_for_forwarding()
    assert "baggage" not in out
    assert "traceparent" in out


def test_forwarding_returns_empty_when_no_trace_headers() -> None:
    ctx = extract_trace_context({})
    assert ctx.headers_for_forwarding() == {}


def test_propagated_baggage_field_set_locked() -> None:
    """Guard against accidentally widening the baggage allowlist —
    leaking arbitrary upstream fields would be an internal-data leak."""
    assert PROPAGATED_BAGGAGE_FIELDS == frozenset({
        "userId", "companyId", "customerOciSessionId",
    })


# ---- log_request_context ------------------------------------------------

def test_log_request_context_emits_one_line_with_trace_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx = extract_trace_context({
        "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        "baggage": "userId=42",
    })
    with caplog.at_level(logging.INFO, logger="tracing"):
        log_request_context(ctx, route="/_search")
    assert len(caplog.records) == 1
    record = caplog.records[0]
    msg = record.getMessage()
    assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in msg
    assert "bbbbbbbbbbbbbbbb" in msg
    assert "userId" in msg
    assert "/_search" in msg


def test_log_request_context_silent_when_trace_id_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No traceparent → don't pollute logs with empty entries."""
    ctx = extract_trace_context({})
    with caplog.at_level(logging.INFO, logger="tracing"):
        log_request_context(ctx)
    assert len(caplog.records) == 0
