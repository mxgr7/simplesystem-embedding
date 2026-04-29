# article-search-acl

Anti-corruption layer in front of the new ftsearch (`./search-api/`).
Exposes a narrowed derivative of the legacy article-search OpenAPI on
`POST /article-features/search`, translates each request into one or
more ftsearch calls, and assembles the legacy response envelope.

This directory ships the **packet A1 skeleton** — the runnable FastAPI
app + `openapi.yaml` + 501 stub. Subsequent packets fill in the
behavior:

| Packet | Lands |
| --- | --- |
| A1 (this) | Skeleton, OpenAPI, healthz/metrics, 501 stub |
| A2 | Request mapper (legacy DTO → ftsearch DTO) |
| A3 | Response mapper (ftsearch DTO → legacy envelope) |
| A4 | Error contract (validation, upstream, timeout categorisation) |
| A5 | Resilience + observability (retries, tracing, RED metrics) |
| A6 | Acceptance suite |

The OpenAPI is the contract source of truth: every documented
deviation from the legacy contract (`issues/article-search-replacement-spec.md`
§2) is encoded into the schema. Schema changes land here first.

## Run

Local docker-compose stack (alongside ftsearch + TEI + Redis):

```sh
docker compose -f playground-app/compose.yaml up acl
```

The service answers on port 8081:

  - `GET  /healthz`              → `{"status":"ok"}`
  - `GET  /openapi.yaml`         → the contract YAML
  - `GET  /docs`                 → swagger UI
  - `GET  /metrics`              → prometheus exposition
  - `POST /article-features/search` → 501 stub until A2/A3 land

Direct invocation outside docker:

```sh
cd acl
uvicorn main:app --host 0.0.0.0 --port 8081
```

## Ports

Per spec §3 / packet A1:

  - **App**: 8081 (`/healthz`, `/article-features/search`).
  - **Metrics**: 9090 — production-target for a separate uvicorn
    instance scraped by Prometheus on a different port from the app.
    The MVP exposes `/metrics` on the app port too (8081/metrics);
    operators wanting the dual-port deployment ship a second
    `uvicorn` process pinned to the metrics route only.

## OpenAPI

Located at [`acl/openapi.yaml`](./openapi.yaml). The `/openapi.yaml`
endpoint serves it directly so `swagger-ui` instances + client
generators can fetch a stable copy.

Validation (CI smoke):

```sh
uv run python -c "
import yaml
from openapi_spec_validator import validate_spec
validate_spec(yaml.safe_load(open('acl/openapi.yaml')))
print('OK')
"
```

## Deviations encoded

  - **§2.1** `searchArticlesBy` is enum `[STANDARD]` only (single value).
  - **§2.2** `articles[].explanation` is the literal `"N/A"` when
    `explain=true`. Schema unchanged on the wire.
  - **§2.3** non-relevance sorts on a queried request operate on a
    relevance-bounded candidate pool (see ftsearch §"RELEVANCE_POOL_MAX").
    Wire shape unchanged; documented as a behavioral deviation in
    the OpenAPI description.

## Configuration

Environment variables:

  - `FTSEARCH_URL` (default `http://search-api:8001`) — base URL of
    the downstream ftsearch service. A2 will use this for the actual
    HTTP calls.

A2-A5 will add tracing baggage forwarding (mirroring the ftsearch
implementation in `search-api/tracing.py`), retry policy, and timeout
budgets shaped to the same legacy SLO (p99 < 5s, p50 < 1s).
