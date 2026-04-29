# A5 — Resilience + observability (retries, timeouts, baggage, RED metrics)

**Category**: ACL (new FastAPI service in this repo)
**Depends on**: A2 (request flow), A3 (response flow)
**Unblocks**: A6

References: spec §4.7 (resilience table), §9 #7 (no auth).

## Status

✅ **Done** — commit `[A5]` (see git log; lands the same patterns F7 added to ftsearch). `acl/clients/ftsearch.py` retries on transient failures (5/500ms/1.5×/5s, 5s total budget); 4xx raises immediately. Per-call timeout `FTSEARCH_TIMEOUT_MS` (default 4500ms) sized to fit inside the legacy p99 SLO with ~500ms ACL headroom. `acl/tracing.py` mirrors `search-api/tracing.py` — middleware extracts traceparent + baggage subset, forwards on every retry attempt. `acl/metrics.py` adds `acl_ftsearch_call_duration_seconds{outcome}` histogram + `_retries_fired_total` + `_retries_exhausted_total` counters (bounded cardinality). 8 unit tests cover retry behaviour + tracing forwarding on every attempt + metrics emission.

**Legacy reference** (next-gen): retry policy values in `article/search/query/src/main/resources/application.yml:91-94` (5 / 500ms / 1.5× / 5s). Tracing baggage W3C entries `userId`, `companyId`, `customerOciSessionId` from same file lines 56-60. Prometheus naming convention from `…/infrastructure/elastic/ElasticsearchMetrics.java` (namespace + `retries=` label tag for cardinality control).

**Latency budget**: legacy SLO **p99 < 5s, p50 < 1s**. Default `FTSEARCH_TIMEOUT_MS=4500` (per call) leaves ~500ms ACL budget; reduce if the retry chain pushes p99 over.

## Scope

Bring the ACL up to the operational shape §4.7 prescribes. The ACL is "thin" but it is on the request path and adds a network hop; resilience and observability glue is where that hop earns its keep instead of becoming a new failure mode.

## In scope

- **Retries on transient ftsearch failures**:
  - Same policy shape as legacy / F7: max 5 attempts, 500ms base, 1.5× multiplier, max 5s delay.
  - Retry on idempotent failures (network errors, 5xx, 503 with `Retry-After`). Do NOT retry on 4xx — those are caller errors and re-running them is pointless.
  - Cap total request time so retries cannot push the request past the legacy SLO; document the budget.
- **Explicit ftsearch call timeout**: per-call, configurable via env `FTSEARCH_TIMEOUT_MS` (default **4500**, sized to keep total p99 < 5s with the retry chain).
- **Tracing baggage** forwarding (§4.7):
  - Accept W3C `traceparent` (and `tracestate` if present) on inbound requests.
  - Forward on every outbound ftsearch call.
  - Forward W3C `baggage`-header entries `userId`, `companyId`, `customerOciSessionId` onto ftsearch and into ACL logs (matches Spring's `management.tracing.baggage.remote-fields`).
  - Use `opentelemetry-api`; do not invent custom header names.
- **RED metrics** (Rate / Errors / Duration) for the ACL hop:
  - Use `prometheus-fastapi-instrumentator`.
  - Add counters for ftsearch retries fired and retry exhaustion.
  - Add a histogram for ACL → ftsearch latency separate from total request latency.
  - Bound label cardinality (e.g. don't put raw `userId` in a label).
- **Caching**: explicitly none in the ACL (it is stateless mapper) — confirm and document. Caching of hierarchies, if any, lives in ftsearch (F7).
- **Health**: `/healthz` returns 200 if the app is alive; no deep dependency probes (don't make a Milvus call from healthz). Add a Spring-actuator-compatible `/actuator/health` only if next-gen ops requires it.

## Out of scope

- Authentication — none on either hop (§9 #7).
- Circuit breakers — not specified by §4.7; out of scope unless explicitly requested.

## Deliverables

- Retry/timeout wrapper around the httpx client from A2.
- Tracing middleware accepting + forwarding W3C headers and baggage.
- Prometheus metrics + documented label cardinality.
- README operational notes on configuration knobs and SLO budgets.

## Acceptance

- Synthetic transient ftsearch failures (one-off 503, then 200) are masked by retries; total latency stays within budget.
- Synthetic ftsearch hangs surface as 5xx with a clean envelope (A4 wires the envelope) within the configured timeout.
- A request carrying `traceparent` shows the same trace ID in ACL logs and forwarded ftsearch calls.
- Prometheus scrape includes the new retry / latency metrics.
- 4xx from ftsearch is propagated immediately, not retried.
