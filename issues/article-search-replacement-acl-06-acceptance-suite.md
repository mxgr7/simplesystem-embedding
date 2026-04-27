# A6 — Acceptance test suite (§10)

**Category**: ACL (new FastAPI service in this repo)
**Depends on**: A1..A5, F1..F7, I1 (populated Milvus collection)
**Unblocks**: ship

References: spec §10 in full.

## Scope

The single closing artefact of the project: an automated test suite that drives the running ACL → ftsearch → Milvus stack against a controlled fixture and asserts every line of §10. Result equality vs. legacy is explicitly NOT a goal (§10 prologue) — the suite is contract- and behaviour-focused on the new stack alone.

## In scope

- A `acl/tests/acceptance/` (or equivalent) suite that runs against a real ACL deployment with a real ftsearch behind it, populated by I1 with a fixture dataset (small, deterministic, checked into the repo or generated).
- Tests grouped by the §10 bullet list:
  - **Schema compliance**: every accepted request shape returns 200 + a body that validates against `acl/openapi.yaml`.
  - **Per-filter narrowing** (§4.3): each filter, on the fixture, narrows the hit set in the documented direction. One test per filter.
  - **Sort ordering** (§4.2): each sort produces the documented order on the fixture. One test per sort × direction.
  - **Aggregation correctness** (§4.4): each summary kind is computed over the *full* filtered hit set; counts are internally consistent (`Σ group counts ≤ metadata.hitCount`). One test per summary kind, each verifying mode interaction.
  - **`hitCount`**: equals the count of the full filtered set, independent of `pageSize`. Plus a test that exercises and documents the safety cap (set the cap low; observe clipping).
  - **`articleId` round-trip**: emit format is `{friendlyId}:{base64UrlEncodedArticleNumber}` on the wire; that exact value round-trips through `articleIdsFilter` and through ID-based fetches.
  - **Error envelopes** (§3.1): malformed input → HTTP 400 with `{message, details, timestamp}`; 5xx → "Internal server error".
  - **Dropped enums** (§2.1): every dropped enum value (`ALL_ATTRIBUTES`, `ARTICLE_NUMBER`, `CUSTOMER_ARTICLE_NUMBER`, `VENDOR_ARTICLE_NUMBER`, `EAN`, `TEST_PROFILE_01..20`) → HTTP 400 with field-level message.
  - **`explain` stub** (§2.2): `explain: true` → every `articles[].explanation == "N/A"`; `explain: false` (or omitted) → null/absent.
  - **Tracing baggage**: a request carrying `traceparent`, `userId`, `companyId`, `customerOciSessionId` results in the same trace and baggage being visible at ftsearch (verified via fixture-injected logging).
  - **Retry / timeout** (§4.7): inject ftsearch failures to exercise both retry-then-success and retry-exhaustion paths.
- **Captured-traffic smoke run** (§10 closing paragraph): play a small slice of captured production traffic against the new stack, assert no 5xx and no schema validation failures. Do NOT compare result sets to legacy — captured traffic is an *input corpus*, not a regression oracle.
- Suite is wired into CI; runs against a docker-compose stack (Milvus + ftsearch + ACL + a fixture loader) so it is reproducible.

## Out of scope

- Result equality against legacy ES.
- Performance / load testing — that has its own packet under `scripts/loadtest_search_api.py`.
- Customer-article-number search — explicitly dropped.

## Deliverables

- Acceptance test directory with one test file per §10 group.
- Fixture loader (re-using I1's projection module) that hydrates Milvus deterministically.
- `docker-compose.acceptance.yml` (or equivalent) bringing up the stack.
- CI step running the suite on every change to `acl/`, `search-api/`, or the indexer.

## Acceptance

- Every §10 bullet is exercised by at least one named test.
- The suite passes against a freshly-stood-up stack from a clean state.
- The captured-traffic smoke run completes without 5xx or schema violations.
- A failing acceptance test produces a clear diagnostic (which §10 line is broken, which fixture row triggered it).

## Open questions for this packet

- Fixture format: synthetic JSON checked in, or a small slice of MongoDB exported through I1's projection? Recommendation: synthetic and deterministic — captured-traffic playback is separate.
- Captured-traffic source: PostHog logs already exist in `scripts/fetch_posthog_search_queries.py`; reuse that.
