# A6 — Acceptance test suite (§10)

**Category**: ACL (new FastAPI service in this repo)
**Depends on**: A1..A5, F1..F7, I1 (populated Milvus collection)
**Unblocks**: ship

References: spec §10 in full.

## Status

🟡 **Partial — happy-path acceptance landed; per-filter / per-sort / per-aggregation expansion + captured-traffic smoke deferred.**

Landed in commit `17ddc62`:
  - End-to-end ACL → ftsearch → real Milvus via `httpx.ASGITransport` wiring the ACL TestClient to the real search-api app in-process.
  - Fixture: sample_200 loaded into ephemeral `articles_v4_a6` + `offers_v5_a6` via the indexer's `load_split` (same code path production indexer runs).
  - 7 §10 cases covered: schema compliance, articleId round-trip, explain stub both directions, dropped-enum rejection, score field dropped, ACL-internal metadata fields dropped.
  - Surfaced + fixed one mapper bug (the customer-article-number fields ftsearch rejects but legacy carries — see A2 status update).

Replay harness landed in commit `c56fe9c`:
  - `scripts/replay_legacy_parity.py` — POSTs each request to legacy + ACL, computes a recursive shape sketch (paths + scalar types, arrays collapsed to a merged element shape), reports `HARD` type mismatches separately from `SOFT` key-presence diffs. Empty arrays on either side are treated as compatible so a small Milvus corpus producing 0 hits doesn't drown out real shape bugs. 24 unit tests in `tests/test_replay_legacy_parity.py`.
  - This is the *tool* for the captured-traffic smoke; it does not by itself complete A6 — a captured request-body corpus (JSONL of legacy request bodies) is still needed before it can run against prod traffic.

Deferred:
  - Per-filter narrowing tests (each §4.3 filter × one assertion). The existing `test_search_dedup_integration.py` already covers each filter against ftsearch directly; ACL pass-through expansion is mechanical.
  - Per-sort × direction tests (§4.2). Same pattern.
  - Per-aggregation correctness tests (§4.4).
  - PostHog captured-traffic smoke run — harness exists (`scripts/replay_legacy_parity.py`); blocked on capturing a request-body corpus (PostHog's `search_performed` carries the query string only, not the full request body, so capture needs a sidecar/proxy in front of legacy or an alternative source).

**Fixture corpus** (locked): synthetic deterministic fixtures for §10 contract tests; **PostHog** (via `scripts/fetch_posthog_search_queries.py`) for the captured-traffic smoke run. Reference legacy aggregation fixtures at `next-gen/article/search/query/src/test/resources/articles_aggregations/` for shape/content inspiration where useful.

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

(none — synthetic deterministic fixtures for contract tests; PostHog corpus for captured-traffic smoke.)
