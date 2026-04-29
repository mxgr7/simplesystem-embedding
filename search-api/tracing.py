"""F7 §"Tracing baggage" — W3C trace context + baggage extraction
and propagation.

Light-touch implementation that satisfies the F7 acceptance criterion
("a request carrying `traceparent` shows the same trace ID in
ftsearch logs") without pulling in the full opentelemetry SDK / OTLP
pipeline. The pieces:

  - **Inbound parse**: a FastAPI middleware reads `traceparent`,
    `tracestate`, and `baggage` headers; stashes a `TraceContext` on
    `request.state`; emits one structured log line per request with
    the trace_id + span_id + selected baggage entries.
  - **Outbound forward**: a helper builds the headers dict for
    forwarding to the embedder. Each outbound call passes them via
    `httpx`'s `headers=` kwarg.
  - Pymilvus pass-through is **not** wired here — Milvus 2.6's gRPC
    backend doesn't surface trace IDs in its server logs (per the
    spec's "where Milvus exposes it" hedge), so propagation buys
    nothing today. Re-evaluate when Milvus's OTLP support lands.

The next-gen Java service uses Spring's W3C baggage with the remote
fields `userId`, `companyId`, `customerOciSessionId` (per
`article/search/query/.../application.yml:56-60`). We honour the same
field names.

If a request arrives with no `traceparent`, we don't synthesize one —
the upstream `traceparent` is the source of truth, and creating a
fake trace_id would muddle log-side correlation. Logs simply omit the
trace_id field in that case.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger(__name__)

# Per spec §"Tracing baggage" — same field set as the legacy Java
# service's `management.tracing.baggage.remote-fields`.
PROPAGATED_BAGGAGE_FIELDS: frozenset[str] = frozenset({
    "userId",
    "companyId",
    "customerOciSessionId",
})

# W3C `traceparent` regex per https://www.w3.org/TR/trace-context/#traceparent-header
# `version-traceid-parentid-flags`. Reject malformed headers (caller
# bug or spoof) rather than treat them as valid.
_TRACEPARENT_RE = re.compile(
    r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$"
)


@dataclass(slots=True)
class TraceContext:
    """Parsed trace context from a single inbound request. None values
    indicate the corresponding header was absent or malformed."""
    trace_id: str | None = None
    span_id: str | None = None
    flags: str | None = None
    raw_traceparent: str | None = None
    raw_tracestate: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)

    def headers_for_forwarding(self) -> dict[str, str]:
        """Build the W3C headers to send to a downstream service. Filter
        baggage to the propagated subset — we don't echo arbitrary
        upstream baggage fields, which could leak internal context."""
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
    """Parse a `traceparent` header. Returns (trace_id, span_id, flags)
    or (None, None, None) on missing/invalid input."""
    if not value:
        return None, None, None
    m = _TRACEPARENT_RE.match(value.strip())
    if not m:
        return None, None, None
    _version, trace_id, span_id, flags = m.groups()
    if trace_id == "0" * 32 or span_id == "0" * 16:
        # Spec: all-zero trace_id or span_id is invalid.
        return None, None, None
    return trace_id, span_id, flags


def parse_baggage(value: str | None) -> dict[str, str]:
    """Parse a W3C `baggage` header. Format: `key1=val1,key2=val2;props`.
    We ignore property-suffixes (`;ttl=60`) since we don't store them.
    Malformed entries are skipped silently — operators see the same
    request flow even if the upstream baggage is dirty."""
    if not value:
        return {}
    out: dict[str, str] = {}
    for entry in value.split(","):
        entry = entry.split(";")[0].strip()  # strip property metadata
        if "=" not in entry:
            continue
        k, _, v = entry.partition("=")
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def extract_trace_context(headers: dict[str, str] | Iterable[tuple[str, str]]) -> TraceContext:
    """Build a `TraceContext` from a request's headers. Accepts either
    a dict or an iterable of (k, v) pairs (Starlette/FastAPI exposes
    both shapes; we accept either to keep callers flexible)."""
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
    """Emit one structured log line per request capturing the trace
    context. A log shipper (e.g. Loki + Grafana) can correlate this
    with downstream logs sharing the same trace_id. No trace context
    → no log line; we don't pollute logs with empty entries."""
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
