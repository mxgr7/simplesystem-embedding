"""W3C trace-context + baggage extraction and propagation, ACL side.

Mirrors `search-api/tracing.py` — same parsing rules, same propagated
baggage subset (`userId`, `companyId`, `customerOciSessionId`), same
log-shipper-friendly correlation log.

Trade-off vs the full opentelemetry SDK is identical to the ftsearch
side: lightweight implementation that satisfies "trace_id appears in
ACL logs and is forwarded to ftsearch", at the cost of NOT integrating
into a full OTLP pipeline. When ftsearch's tracing module evolves to
use the SDK, port these together.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger(__name__)

PROPAGATED_BAGGAGE_FIELDS: frozenset[str] = frozenset({
    "userId",
    "companyId",
    "customerOciSessionId",
})

_TRACEPARENT_RE = re.compile(
    r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$"
)


@dataclass(slots=True)
class TraceContext:
    trace_id: str | None = None
    span_id: str | None = None
    flags: str | None = None
    raw_traceparent: str | None = None
    raw_tracestate: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)

    def headers_for_forwarding(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.raw_traceparent:
            out["traceparent"] = self.raw_traceparent
        if self.raw_tracestate:
            out["tracestate"] = self.raw_tracestate
        forwarded = {
            k: v for k, v in self.baggage.items()
            if k in PROPAGATED_BAGGAGE_FIELDS
        }
        if forwarded:
            out["baggage"] = ",".join(f"{k}={v}" for k, v in forwarded.items())
        return out


def parse_traceparent(value: str | None) -> tuple[str | None, str | None, str | None]:
    if not value:
        return None, None, None
    m = _TRACEPARENT_RE.match(value.strip())
    if not m:
        return None, None, None
    _version, trace_id, span_id, flags = m.groups()
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None, None, None
    return trace_id, span_id, flags


def parse_baggage(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    out: dict[str, str] = {}
    for entry in value.split(","):
        entry = entry.split(";")[0].strip()
        if "=" not in entry:
            continue
        k, _, v = entry.partition("=")
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def extract_trace_context(headers: dict[str, str] | Iterable[tuple[str, str]]) -> TraceContext:
    if not isinstance(headers, dict):
        headers = {k.lower(): v for k, v in headers}
    else:
        headers = {k.lower(): v for k, v in headers.items()}

    raw_traceparent = headers.get("traceparent")
    raw_tracestate = headers.get("tracestate")
    raw_baggage = headers.get("baggage")
    trace_id, span_id, flags = parse_traceparent(raw_traceparent)
    return TraceContext(
        trace_id=trace_id,
        span_id=span_id,
        flags=flags,
        raw_traceparent=raw_traceparent if trace_id else None,
        raw_tracestate=raw_tracestate,
        baggage=parse_baggage(raw_baggage),
    )


def log_request_context(ctx: TraceContext, *, route: str | None = None) -> None:
    if ctx.trace_id is None:
        return
    fields = {
        "trace_id": ctx.trace_id,
        "span_id": ctx.span_id,
    }
    if route:
        fields["route"] = route
    forwarded = {
        k: v for k, v in ctx.baggage.items() if k in PROPAGATED_BAGGAGE_FIELDS
    }
    if forwarded:
        fields["baggage"] = forwarded
    log.info("trace_context %s", fields)


__all__ = [
    "TraceContext",
    "PROPAGATED_BAGGAGE_FIELDS",
    "extract_trace_context",
    "parse_traceparent",
    "parse_baggage",
    "log_request_context",
]
