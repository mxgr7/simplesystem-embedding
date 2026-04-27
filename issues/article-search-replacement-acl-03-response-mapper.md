# A3 — ftsearch → legacy response mapper

**Category**: ACL (new FastAPI service in this repo)
**Depends on**: A1 (OpenAPI), A2 (request flow), F2..F5
**Unblocks**: A5, A6

References: spec §2.2 (explain stub), §3 (response shape), §4.4 (summary shapes), §9 #4 (articleId).

## Scope

Translate the ftsearch response into the legacy response envelope. Apply the only behavioural deviation that lives in the ACL — the `explain` stub (§2.2). Everything else is pass-through reshape.

## In scope

- Pydantic models for the legacy response (matching `acl/openapi.yaml` from A1).
- A response mapper (`acl/mapping/response.py`) producing the legacy DTO from the ftsearch response:
  - **`articles[]`**: each entry carries `articleId` (verbatim from ftsearch — already in legacy `{friendlyId}:{base64Url(articleNumber)}` format per §9 #4) and `explanation`.
    - When the inbound request had `explain: true`: `explanation = "N/A"` for every article (§2.2).
    - When `explain: false` or omitted: `explanation` is null/absent.
  - **`summaries`** block: reshape ftsearch's summary payloads into the legacy field names:
    - `vendorSummaries`, `manufacturerSummaries`, `featureSummaries` — straightforward field renames if needed.
    - `pricesSummary` — array of `{min, max, currencyCode}` entries.
    - `categoriesSummary` with `currentCategoryPathElements`, `sameLevel`, `children`.
    - `eClass5Categories` / `eClass7Categories` / `s2ClassCategories` with `selectedEClassGroup`, `sameLevel`, `children`.
    - `eClassesAggregations: [{id, count}, …]`.
    - Summaries not requested or not produced by ftsearch are omitted / null per the OpenAPI shape.
  - **`metadata`**: `page`, `pageSize`, `pageCount`, `term`, `hitCount` — passthrough.
- Mode interaction:
  - `HITS_ONLY` — `summaries` empty/omitted (match what ftsearch returns and what the OpenAPI declares).
  - `SUMMARIES_ONLY` — `articles[]` empty.
  - `BOTH` — both populated.
- Tests: shape assertions per mode, per summary kind, plus the explain-stub matrix (`explain: true` → all `"N/A"`; `explain: false` → all null/absent).

## Out of scope

- Generating useful explanations — out of scope per §2.2 (and §9 #3 calls this out as a deliberate parking).
- Errors — A4.
- Retries / fallbacks when ftsearch returns 5xx — A5.

## Deliverables

- `acl/mapping/response.py` + Pydantic models.
- Wiring in `acl/main.py` so the response mapper runs after the ftsearch call.
- Unit tests covering each summary kind, each mode, and the explain matrix.

## Acceptance

- A round-trip on a synthetic ftsearch response produces a legacy-shaped body that validates against `acl/openapi.yaml`.
- `articles[].articleId` on the wire is byte-identical to what ftsearch returned (no truncation, no re-encoding).
- `explain: true` populates `"N/A"` on every article; `explain: false` leaves `explanation` null/absent across the board.
- Mode interaction matches the §3 mode rules.

## Open questions for this packet

- Field naming alignment between ftsearch's response (F2) and legacy: confirm whether F2 chose legacy names verbatim or chose its own; if its own, this mapper does the rename. (Recommendation for F2: pick legacy-shaped names where the mapping is 1:1; saves work here.)
