# A1 — ACL skeleton + narrowed legacy OpenAPI

**Category**: ACL (new FastAPI service in this repo)
**Depends on**: F2 (frozen ftsearch contract)
**Unblocks**: A2, A3, A4, A5

References: spec §1, §2 (the deviations the OpenAPI must encode), §3 (verbatim contract).

## Status

✅ **Done** — commit `3aaac3d`. New `./acl/` directory with FastAPI app on port 8081 (`/healthz`, `/openapi.yaml`, `/metrics`, `POST /article-features/search` initially returned 501 stub, replaced by A2). Hand-authored OpenAPI 3.0 in `acl/openapi.yaml` encodes every §3 field and the §2 deviations (single-value `searchArticlesBy`, explanation stub, sort behavior). Wired into `playground-app/compose.yaml`. 8 unit tests cover healthz, openapi.yaml served + valid 3.0, schema invariants (single-value enum, pageSize cap, currency pattern, empty security).

**Legacy reference** (next-gen): canonical legacy contract at `api-spec/specs/article-search/spec.yaml` (paths, error envelope at lines 246-278) and `query-search-api.yaml` (`SearchParams` lines 113-237 — defines `searchMode`, `searchArticlesBy`, all filters, sort, summaries). Health endpoint convention in legacy is Spring Boot Actuator (`/actuator/health`, `/actuator/prometheus`).

## Scope

Stand up the ACL service skeleton in this repo and check in the narrowed legacy OpenAPI as the single source of truth for what next-gen will call. The OpenAPI is the most consequential artefact of this packet — every deviation from the legacy contract (§2) is encoded into the schema, so subsequent packets implement *this* OpenAPI rather than improvising.

## In scope

- New top-level directory **`./acl/`** (sits next to `./search-api/`) housing a FastAPI app:
  - Single endpoint stub: `POST /article-features/search` returning a 501-equivalent until A2/A3 fill it in.
  - `/healthz` on app port; `/metrics` (via `prometheus-fastapi-instrumentator`) on metrics port.
  - **App port 8081, metrics port 9090** (separate uvicorn instance for the metrics port). `security: []` on all endpoints (§9 #7 — no auth, internal service).
  - Service name **`article-search-acl`**.
  - Packaging that fits this repo's conventions (`pyproject.toml`, lint, formatter).
  - Deployed in the **same docker-compose stack** as ftsearch (extend `playground-app/compose.yaml` or add a sibling compose file referencing the same TEI + Milvus services).
- `acl/openapi.yaml` containing the narrowed legacy contract:
  - Mirror `api-spec/specs/article-search/*.yaml` (canonical legacy spec) for `POST /article-features/search` only — other paths are out of scope.
  - Apply every deviation from §2:
    - **§2.1** `searchArticlesBy` enum reduced to a single-value enum: `[STANDARD]` only. Field stays required.
    - **§2.2** `explain: boolean` request and `articles[].explanation: string?` response stay verbatim — implementation is a stub but the schema is unchanged.
  - All other request/response fields per §3, including the `selectedArticleSources` block, all filters, `summaries[]`, both currency roles (top-level and `priceFilter.currencyCode`), `searchMode`, `sort` query params, `page`, `pageSize` (max 500), and the full response envelope.
  - Error envelope `{message, details, timestamp}` per §3.1.
  - Operation IDs and tags shaped so that next-gen's existing client generator produces a usable Java client.
- A round-trip script that loads `acl/openapi.yaml` through a standard validator (e.g. `openapi-spec-validator`) in CI.
- README explaining: the ACL is a thin DTO mapper; the OpenAPI is authoritative; changes to the contract land here first.

## Out of scope

- Request mapping — A2.
- Response mapping — A3.
- Error handler — A4.
- Resilience / observability glue — A5.
- Tests — A6.

## Deliverables

- New `./acl/` directory with a runnable FastAPI skeleton.
- `acl/openapi.yaml` checked in.
- Health, metrics, and stub endpoint live.
- CI step validating the OpenAPI.

## Acceptance

- The ACL boots and answers `/healthz` on port 8081, `/metrics` on 9090.
- The OpenAPI validates and round-trips through a swagger UI without errors.
- Every §3 field is present in the schema; every §2 deviation is encoded (smoke-test by attempting `searchArticlesBy: ARTICLE_NUMBER` against the schema and getting a clear validation error).

## Open questions for this packet

(none — directory `./acl/`, ports 8081 (app) / 9090 (metrics), same compose as ftsearch.)
