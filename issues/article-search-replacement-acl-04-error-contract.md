# A4 — Legacy error envelopes + dropped-enum rejection

**Category**: ACL (new FastAPI service in this repo)
**Depends on**: A1
**Unblocks**: A6

References: spec §2.1 (rejection), §3.1 (envelope + table), §10 (acceptance lines).

**Legacy reference** (next-gen): error envelope schema at `api-spec/specs/article-search/spec.yaml:246-278` (`{message, details, timestamp}`, `details` is array of `{field, message}`). Dropped enum constants enumerated in `article/search/query/src/main/java/com/simplesystem/nextgen/article/search/query/api/ArticleSearchOperations.java:91-117` — full list to reject: `ALL_ATTRIBUTES`, `ARTICLE_NUMBER`, `CUSTOMER_ARTICLE_NUMBER`, `VENDOR_ARTICLE_NUMBER`, `EAN`, `TEST_PROFILE_01..20`.

## Scope

Reshape every error path so the ACL produces the legacy `{message, details, timestamp}` envelope, including for the input shapes §2.1 has dropped (stale enum values from old next-gen clients).

The ACL's OpenAPI is the validation source of truth — Pydantic will reject most malformed input automatically. This packet replaces FastAPI's default 422 envelope with the legacy 400 shape and adds the §3.1 table cases.

## In scope

- Custom FastAPI exception handlers for:
  - **Bean-validation-style failures**: produced by Pydantic. Reshape into `{ "message": "Validation failure", "details": [{"field": "<jsonPath>", "message": "<rule>"}], "timestamp": "<ISO-8601>" }`. HTTP 400.
  - **Constraint violations**: cross-field validations the ACL does itself (e.g. `priceFilter.currencyCode` must be non-null when `priceFilter.min` or `max` is set, per §3). Reshape with `{ "message": "Constraint violation", "details": [{"propertyPath": "<dot.path>", "message": "<rule>"}], "timestamp": "<ISO-8601>" }`. HTTP 400.
  - **JSON parse failure**: HTTP 400 with the parser's exception message.
  - **Anything else (5xx)**: HTTP 500 with `{ "message": "Internal server error", "details": [], "timestamp": "<ISO-8601>" }`.
- **Dropped-enum rejection** (§2.1):
  - The OpenAPI declares `searchArticlesBy: STANDARD` only (single-value enum), so Pydantic rejects `ARTICLE_NUMBER`, `CUSTOMER_ARTICLE_NUMBER`, `VENDOR_ARTICLE_NUMBER`, `EAN`, `ALL_ATTRIBUTES`, `TEST_PROFILE_01..20` automatically.
  - Confirm the rejection message says `must be 'STANDARD'` (or wraps the enum mismatch into that exact text), so callers see a useful message.
- **Defence-in-depth on the ftsearch side**: ftsearch (F2) also validates; this packet is about presenting validation failures in the legacy shape, not about the only validation in the system.
- Confirm no information leakage — error messages do not contain stack traces, database errors, or internal hostnames; shape is bounded to the §3.1 envelope.
- Tests:
  - Each case in the §3.1 table produces the correct envelope.
  - Each dropped enum value (the full list above) produces a 400 with `field=searchArticlesBy`.
  - 5xx wraps to "Internal server error".

## Out of scope

- ftsearch's own validation behaviour (handled by F2).
- Retries / fallbacks for transient ftsearch failures — A5.
- Tracing baggage on errors — A5.

## Deliverables

- Exception handlers registered on the FastAPI app.
- Unit tests covering every case in §3.1 and the §2.1 enum list.
- Dropped-enum rejection messages reviewed for clarity.

## Acceptance

- Each row of §3.1 produces the documented status + envelope.
- Each enum value listed in §10 ("Every request shape dropped by §2 returns HTTP 400 …") returns HTTP 400 with a `searchArticlesBy` field-level message.
- 5xx returns the legacy 500 envelope verbatim — no FastAPI default.
- Pydantic's default 422 path is no longer reachable in production.
