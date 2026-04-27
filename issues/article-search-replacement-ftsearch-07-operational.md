# F7 — Operational glue (Bounded consistency, tracing, retries, timeouts)

**Category**: ftsearch (`./search-api/`)
**Depends on**: F2, F3, F4, F5
**Unblocks**: A6 (integration acceptance against a realistic deployment)

References: spec §4.7, §4.8, §9 #7, §9 #8.

**Legacy reference** (next-gen): tracing baggage in `article/search/query/src/main/resources/application.yml:56-60` (`management.tracing.baggage.remote-fields: [userId, companyId, customerOciSessionId]` — Spring W3C baggage, no header prefix). Retry policy in same file lines 91-94 (5 / 500ms / 1.5× / 5s — confirmed). Prometheus naming in `query/src/main/java/com/simplesystem/nextgen/article/search/query/infrastructure/elastic/ElasticsearchMetrics.java` (namespace `article.search.query.infrastructure.elasticsearch.*`, `retries=` label tag for cardinality control).

**Latency budget**: legacy SLO is **p99 < 5s, p50 < 1s**. Pick per-call timeouts so total request latency stays inside p99 even after the maximum-allowed retry chain.

## Scope

The operational surface ftsearch needs once the capability work (F1..F6) is in. None of these change behaviour at the API surface — they make the service production-shaped under the resilience/observability requirements in §4.7.

## In scope

- **Milvus consistency level**: set `Bounded` on every Milvus call ftsearch makes (search, query, count). Wire via a single helper so adding new call sites can't drift.
- **Tracing baggage**:
  - Accept W3C `traceparent` (and `tracestate` if present) on inbound requests; propagate to all outbound Milvus calls and downstream embedder calls.
  - Forward the next-gen W3C baggage entries `userId`, `companyId`, `customerOciSessionId` (carried in the `baggage` header per W3C Baggage spec — Spring side uses `management.tracing.baggage.remote-fields`) through to outbound logs and metrics.
  - Use `opentelemetry-api` Python idioms; do not invent custom header names.
- **Retries**: retry transient Milvus and embedder failures on idempotent calls. Use the same policy shape as legacy (§4.7): max 5 attempts, 500ms base, 1.5× multiplier, max 5s delay. Do not retry 4xx-class errors. Cap total retry budget so the per-request latency stays within SLO.
- **Timeouts**: explicit per-call timeouts on Milvus and embedder; total request budget ≤ existing p99. Document the picked values and how to override via env.
- **RED metrics**: rate / errors / duration for the hot path, broken out by:
  - Mode (single mode now — `STANDARD` — so this is mostly future-proofing).
  - Sort kind (relevance / name / price / articleId).
  - Whether classifier routed to strict vs. hybrid.
  - Whether `summaries[]` was non-empty.
  Use the existing `prometheus-fastapi-instrumentator`; supplement with custom counters for fallback fires and for hitcount-cap clipping.
- **Health**: keep `/healthz`; add a Spring-actuator-compatible `/actuator/health` endpoint if ops asks for it (the spec leaves this conditional).
- **Optional caching** for category / eClass hierarchy lookups (small, hot, almost-static). Bounded in size; flushed on alias swap.

## Out of scope

- Authentication — spec §9 #7 is explicit: no per-request auth on ftsearch (internal service).
- Request-level rate limiting — already covered by the existing concurrency cap in `search-api/main.py`.

## Deliverables

- Tracing middleware accepting and forwarding W3C headers + baggage.
- A single `with_consistency(Bounded)` helper used at all Milvus call sites.
- Retry / timeout wrappers on Milvus + embedder calls.
- New Prometheus metrics with documented label cardinality.
- Optional in-memory cache (sized + TTL'd).
- Configuration knobs documented in `search-api/main.py` module docstring.

## Acceptance

- Synthetic Milvus failures (one-off transient, then success) are masked by retries.
- Synthetic Milvus timeouts surface as 5xx within the documented budget — no silent hang.
- A request carrying `traceparent` shows the same trace ID in ftsearch logs and (where Milvus exposes it) in Milvus logs.
- Prometheus scrape includes the new labels and metrics.
- Bounded consistency is set on every Milvus call (verified by audit, ideally automated via the helper).
