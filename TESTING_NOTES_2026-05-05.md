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
