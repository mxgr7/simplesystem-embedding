# Comprehensive F2 contract test suite — testing log (2026-05-05)

Goal: replace the legacy parity replay machinery with a comprehensive
spec-driven test suite that exercises every aspect of
`search-api/openapi.yaml` (POST `/{collection}/_search`) against the
real catalog loaded into Milvus (`articles_v6` + `offers_v6`).

## Working facts about the loaded catalog

Captured 2026-05-05 from a live `pymilvus` scan of the loaded
collections. These pin known-good values for deterministic test
scenarios.

### Collections

| Collection      | Rows         | Notes |
| --------------- | ------------ | ----- |
| `articles_v6`   | 982 753      | Article-side: name, manufacturerName, category_l1..l5, eclass5/7, s2class, per-currency price min/max, BM25 codes, dense embedding |
| `offers_v6`     | 1 963 490    | Offer-side: vendor_id, catalog_version_id, prices JSON, delivery_time, features array, core marker arrays, relationship arrays, price_list_ids, currencies array, per-currency price min/max |
| `offers_codes`  | 158 269 705  | BM25 sparse codes for legacy `_search_v0` path |

### Known-good values to drive tests

* **Vendor IDs** (offer-side): `6a67b8b5-9b7c-47f0-92d5-1dfd8812a505`,
  `216c5d41-b64a-42f1-b084-d7e3419b2219`, `e22f1ac6-14bc-4287-ab0d-1f34c1780f2e`,
  `01054f55-c50c-452b-8822-ee11be4788c9`
* **Catalog version IDs**: `afe615ec-beee-4283-a530-24519843b399`,
  `830c5984-51a4-44e9-a56a-a2e77fbd6568`, `a73870cc-1844-4c17-a55c-998b1f017949`
* **Price list IDs**: `0330f11c-9d00-4a58-9978-76e4d71fed80` and many more
* **Manufacturers** (article-side): `Würth`, `GARANT`, `SMC`, `TOOLCRAFT`, `Siemens`
* **Category L1**: `Zerspanung`, `Elektromaterial`, `Verbindungselemente`
* **EClass5 codes** (int): 21010101, 23110101, 27260701, 27269134, 32169090
* **Currencies present in `currencies` array**: `eur` (dominant), `chf`
* **Currency-code columns**: `eur`, `chf`, `huf`, `pln`, `gbp`, `czk`, `cny`
* **Delivery time values**: 1..32 days observed
* **EUR price spread**: 0.10 .. ~1929 (a sentinel `FLT_MAX` shows up for
  rows with no eur price — must exclude when sampling bounds)
* **Feature value examples** (offer-side, format `key=value`):
  `Ursprungsland=IT`, `Eigenschaften=chemikalienbeständig`
* **Free-text query candidates**: `schraube`, `bohrer`, `drill`,
  `siemens`, `kabel`, `dichtung`, `airplane` (use ones with known hits
  to make hit-count assertions stable)

### Pre-existing services state at start of session

* Milvus + minio + etcd up at `localhost:19530`
* Redis up at `localhost:6379`
* Observability (grafana/prometheus/caddy) up
* search-api / acl / playground-app: **not** running — tests run the
  FastAPI app in-process via `TestClient`, with `EMBED_URL=http://embed.invalid`
  to force the no-query browse path or to fail loud on the dense leg
  unless we explicitly mock the embedder.

## Run log

### Iteration 1 — `nullable: true` on `$ref` fields

Suite stopped at `TestValidator.test_envelope_with_articles_and_summaries`
with the error:

```
- ['summaries', 'categoriesSummary']: None is not of type 'object'
- ['summaries', 'eClass5Categories']:  None is not of type 'object'
- ['summaries', 'eClass7Categories']:  None is not of type 'object'
- ['summaries', 's2ClassCategories']:  None is not of type 'object'
```

**Root cause:** OpenAPI 3.0 lets you write `{ $ref: '…/X', nullable: true }`
to mean "X or null". My `_openapi_to_jsonschema` converter only handles
`nullable: true` when the schema has a literal `type:` field — it
doesn't expand the `$ref + nullable` shape into `anyOf: [$ref, {type:null}]`.

**Fix:** in `_openapi_to_jsonschema.walk()`, when a node has both
`$ref` and `nullable: true`, emit `anyOf: [{ $ref: <ref> }, { type:
"null" }]` instead of leaving the bare `$ref` alone.

### Iteration 2 — `Metadata.recallClipped` / `hitCountClipped` missing from spec

```
- ['metadata']: Additional properties are not allowed
  ('hitCountClipped', 'recallClipped' were unexpected)
```

**Root cause:** The `Metadata` pydantic model in
`search-api/models.py` declares `recallClipped: bool = false` and
`hitCountClipped: bool = false` — surfaced by F4/F9 and emitted on
every response. The OpenAPI YAML's `Metadata` schema never grew the
two fields. Real spec drift — the OpenAPI is documented (in `main.py`
header) as the contract source of truth.

**Fix:** add both fields to `components.schemas.Metadata` in
`search-api/openapi.yaml`, with the same description text the
pydantic model carries.

### Iteration 3 — `queryString` is the legacy ACL field name; F2 uses `query`

```
{"detail":[{"type":"extra_forbidden","loc":["body","queryString"], …}]}
```

**Root cause:** test bug — I copied a legacy ACL request shape that
uses `queryString`. The F2 ftsearch contract uses `query`.

**Fix:** rename `queryString` → `query` in the test bodies.

### Iteration 4 — `pageSize` is a query param, not a body field

Test sent `pageSize` inside the JSON body. F2 contract puts pagination
exclusively on the query string (parameters block). Fixed by moving
to `?pageSize=…`.

### Iteration 5 — green at 88 tests

After the spec fix and the two test bugs, the suite was 88-green.
Subsequent commits added five more classes (auth alternatives,
negative bodies, behaviour, summary content, deeper summaries,
minimal-body). Stable at 111 passing.

## State of adjacent suites

Sanity-ran the broader test sweep once the F2 file went green. The
following pre-existing failures are **unrelated** to F2 and predate
this work:

* `test_offers_collection_schema` — depends on a Milvus alias
  `offers_v_alias` that no longer exists in the running cluster.
* `test_duckdb_*`, `test_index_*`, `test_projection`, `test_tei_cache`
  — depend on tooling/parquet fixtures unrelated to the F2 path.
* `test_catalog_benchmark` — pre-existing tokenizer-stub bug in
  `infer.py`.

F2/ACL-adjacent suites (search-api contract, search-dedup integration,
all ACL acceptance + skeleton + integration tests + the new F2 file)
all pass: 192 passed, 27 skipped, 0 failed.

### Iteration 6 — `metadata.term` divergence between dedup and legacy paths

```
term '' != expected None for query None
```

**Root cause:** main.py wires `term` differently in the two F2 paths:
- legacy single-collection: `term=body.query` (None propagates).
- F9 dedup-topology (the path articles_v6/offers_v6 use):
  `term=body.query or ""` (None becomes "").

Spec declares `term: { type: string, nullable: true }`, so both wire-
shapes are valid, but the two paths disagree on what they emit for
the same input. Real bug: the dedup path silently coerces null → "".

**Fix:** align dedup path with legacy + spec — drop the `or ""` so
None propagates. Tested implicitly by the F2 suite via the
term-echoes-query-text test.

## Coverage summary at 129-passing checkpoint

`tests/test_f2_contract_against_catalog.py` covers, against the live
`articles_v6` + `offers_v6` corpus:

* OpenAPI document health: 3.x meta-validation; every `$ref`
  resolves; every (path, verb, status, media) entry references a
  declared schema; pydantic models stay in lockstep with the YAML
  for `SearchRequest`, `Metadata`, `Summaries`, `Article`,
  `SearchMode`, `SummaryKind`, `EClassVersion`.
* Validator self-test (3 cases).
* Path/query parameters: collection, page (default + custom +
  rejection of 0), pageSize (0, 500, 501, default), sort (every
  field × direction, multi-key first-key-only acceptance, unknown
  field/direction rejection), missing-collection 404.
* Body validation: every required field rejection, every sub-schema
  `additionalProperties=false` rejection, currency pattern, search
  mode enum, search-articles-by/explain (legacy fields explicitly
  rejected), nested `requiredFeatures`/`blockedEClassGroups`
  /`selectedArticleSources` extras, summaries enum membership.
* searchMode envelope: HITS_ONLY (summaries suppressed even when
  requested), SUMMARIES_ONLY (no articles), BOTH (both populated).
* Filters (each tested): vendorIdsFilter (with offer-ownership
  assertion), articleIdsFilter (round-trips a real id), manufacturer
  filter, eClassesFilter, currentEClass{5,7,S2}Code, max delivery,
  required features, blocked eclass vendors, core sortiment, core
  articles vendors, closed marketplace toggle (CV scope swap),
  s2ClassForProductCategories, eClassesAggregations.
* Relationship filters: accessoriesFor, sparePartsFor, similarTo.
* Price filter: min+max, only-max, only-min, narrowing assertion,
  invalid currency rejection, missing currencyCode rejection.
* Sort ordering: articleId asc/desc actually sorted; relevance sort
  returns descending scores when query is set; multi-key first-key
  only.
* Pagination: pageCount math, no overlap between adjacent pages,
  page=9999 returns empty, idempotence on identical requests.
* Summaries: every SummaryKind yields a valid envelope; vendor and
  manufacturer buckets contain the filtered values; bucket counts
  do not exceed total hitCount; categoriesSummary echoes
  currentCategoryPathElements; eClass5Categories shape; pricesSummary
  currency code pattern + min ≤ max; featureSummaries parent ≥ max
  child count.
* Auth: 401 + Error envelope when API_KEY is set and missing/wrong;
  both `Authorization: ApiKey` and `X-API-Key` accepted; /metrics
  and /openapi.yaml bypass auth; /openapi.json equals /openapi.yaml.
* Concurrency: 503 + `Retry-After: 1` + Error envelope when the gate
  is saturated.
* Behaviour: high-volume vendor → hits>0; unknown vendor / unknown
  manufacturer → 0; HITCOUNT_CAP=1 with article-side filter clips
  hitCount and sets hitCountClipped=true; Article.score is null in
  browse and numeric in query path; Unicode (`größe`) and long (~1.8KB)
  query strings round-trip.
* Term echo: schraube → "schraube"; "" → ""; null/omitted → null.
* Minimal valid body (only the spec's three required fields).

### Fixes shipped while building the suite

1. `Metadata.recallClipped` and `Metadata.hitCountClipped` were
   emitted by the implementation but missing from
   `search-api/openapi.yaml`. Added to the spec.
2. `_search_dedup` emitted `metadata.term = ""` for `query=null`
   while the legacy single-collection path emits `null`. The dedup
   path was wrong — corrected to match the spec (`term: nullable`)
   and the legacy path.

## Final state at suite milestone

* `tests/test_f2_contract_against_catalog.py`: **171 passing,
  stable across multiple consecutive runs** (~11s wall time per run).
* Adjacent F2 + ACL suites (search-api contract, search-dedup
  integration, ACL acceptance/skeleton/integration/error/resilience,
  filter integration): clean, 60 environment-gated skips, no
  regressions from the dedup-path `term` fix.
* Unrelated repo-wide failures (`test_duckdb_*`, `test_index_*`,
  `test_projection`, `test_tei_cache`, `test_offers_collection_schema
  alias tests`, `test_catalog_benchmark`) all pre-existed this work
  and stem from missing parquet fixtures, missing milvus aliases,
  or a tokenizer-stub bug in `infer.py`.

## Test-class index

| Class | What it covers |
| --- | --- |
| `TestValidator` | Validator self-test (envelope/ extra/ required/ count rejection) |
| `TestParameters` | page/pageSize/sort path+query parameters |
| `TestBodyValidation` | Required-field and per-field rejections |
| `TestSearchModes` | HITS_ONLY/SUMMARIES_ONLY/BOTH envelope shapes |
| `TestFilters` | Each scalar filter atom from spec §4.3 |
| `TestRelationships` | accessoriesFor / sparePartsFor / similarTo |
| `TestPriceFilter` | min/max/only-min/only-max/narrowing/invalid currency |
| `TestSortOrderings` | articleId/relevance/multi-key/explicit-sort beats relevance |
| `TestPagination` | pageCount math, page overlap, summary stability across pages |
| `TestSummaries` | Each SummaryKind yields a valid envelope |
| `TestAuthAndConcurrency` | API_KEY 401 + 503 gate + HITCOUNT_CAP |
| `TestBehaviour` | Filter narrowing assertions on the live catalog |
| `TestFreeTextQuery` | query body field, term echo, Unicode, long strings |
| `TestArticleIdsFilter` | Real offer-id round-trips |
| `TestPaginationEdges` | pageSize=500 boundary, page=9999 |
| `TestSummaryContent` | Summary buckets contain filtered values, count caps |
| `TestOpenAPIDocument` | Spec/impl parity on every schema, examples valid, $refs resolve |
| `TestNegativeBodies` | Per-sub-schema additional-properties rejection |
| `TestAuthHeaderAlternatives` | Both auth header forms; metrics/openapi public bypass |
| `TestPageSizeZero` | pageSize=0 keeps real hitCount |
| `TestMinimalBody` | Request with only the spec-required fields |
| `TestDeeperSummaries` | Per-summary content invariants |
