# CODES_IMPORT — implementation plan for hybrid_v0

Companion to `hybrid_v0.md`. This file is the concrete, codebase-aware plan
the implementation will follow. Naming, paths, and API shapes here override
anything different in `hybrid_v0.md`.

## Decisions locked in (from clarifying round)

| # | Decision |
|---|---|
| 1 | Dense collection stays named `offers`. The new collection is `offers_codes`. |
| 2 | A single `POST /{collection}/_search` endpoint is kept and parametrised. The collection in the URL is the **dense** collection (or any other dense collection); the codes collection is configured server-side and joined in based on a `mode` query parameter. |
| 3 | PostHog credentials live in `.env` (`POSTHOG_PERSONAL_API_KEY`, `POSTHOG_PROJECT_ID`, `POSTHOG_HOST`); a small fetch helper script will land in `scripts/`. |
| 4 | `offers_codes` is built via the same MinIO `do_bulk_insert` path used by the existing scripts. |
| 5 | Strict-path tied-score tiebreaker: deterministic ascending `id` sort. v0-quality, just to keep A/B logging reproducible. The doc still says product-level ordering is out of scope. |
| 6 | Source data path: `/data/datasets/offers_embedded_full.parquet/`. |
| 7 | New `offers_codes` collection: every field other than the PK is `mmap_enabled=True`; the SPARSE_INVERTED_INDEX has `mmap.enabled=true`. PK and the auto-generated sparse vector field stay as the only RAM-resident things. |

## High-level architecture

```
                          ┌────────────────────┐
                  query → │  /offers/_search   │ ──┐ params: mode, k, codes_limit,
                          │   (search-api)     │   │  dense_limit, rrf_k,
                          └────────────────────┘   │  num_candidates,
                                  │                │  enable_fallback
                                  ▼                │
                  ┌─────────────────────────────────┘
                  │
   modes:         ▼
   - vector            → dense (offers) only
   - bm25              → BM25 (offers_codes) only
   - hybrid            → dense ‖ bm25, RRF fused          (no classifier)
   - hybrid_classified → classifier picks strict or hybrid (matches doc)
                          + optional 0-result fallback
```

The `_search` endpoint stays a thin shim. All routing logic lives in
`search-api/hybrid.py` so the playground app can import the same module and
expose the same knobs.

## Phase 1 — Audit + materialise `text_codes`

Output of this phase: `offers_codes_staging.parquet/` with two columns
(`id`, `text_codes`), one row per surviving offer, plus `reports/codes_audit.md`
documenting counts and sample drops.

### Files

- `scripts/build_offers_codes_staging.py` — DuckDB-driven materialisation.

### Algorithm (two passes over the 16 buckets)

1. **Pass 1: per-row dedupe → frequency counts.**
   Build a temp table that, for each row, emits the distinct lowercase values
   from `ean`, `manufacturerArticleNumber`, `article_number` after applying
   the universal + field-specific structural filters. Aggregate counts and
   write the frequency-cap denylist (`count > 500`) to a small parquet for
   reuse.

2. **Pass 2: materialise.** For each row, apply the same structural filters,
   subtract the frequency denylist, lower-case + space-join the surviving
   values into `text_codes`. Skip the row if all three fields drop. Write
   `id` + `text_codes` to `offers_codes_staging.parquet/part-NN.parquet`
   (one staged parquet per source bucket, to fit the existing pipelined
   stage+submit pattern).

### Filter pseudocode

```python
DROP_RE = [
    r"^0+$",          # all zeros
    r"^9+$",          # all nines
    r"^[-_.\s]+$",    # only punctuation/whitespace
]
DROP_LITERAL = {"k.a.", "n/a", "n.a.", "null", "aucune donnée", "#ref!"}

def keep_universal(v: str) -> bool:
    if not v or len(v) < 4 or len(v) > 40: return False
    if any(re.match(p, v) for p in DROP_RE):  return False
    if v.lower() in DROP_LITERAL:              return False
    if re.search(r"\s", v):                    return False  # CRITICAL
    return True

def keep_ean(v): return keep_universal(v) and len(v) in {8, 12, 13, 14}

def keep_mpn(v):
    if not keep_universal(v): return False
    lo = v.lower()
    if re.fullmatch(r"[a-zäöüß]+", lo): return False
    if "e+" in lo and re.fullmatch(r"[\d,.eE+\-]+", v): return False
    return True

keep_article = keep_mpn
```

### Audit verifications (echoes hybrid_v0.md §Result)

After Pass 2, write `reports/codes_audit.md` with:
- total rows kept vs dropped (target ≈ 158.27M / 1.01M)
- per-field survival counts vs the doc's targets
- distinct values indexed per field (≈ 15.0M / 22.0M / 49.6M)
- the frequency denylist size (~80–120) + 30 sample values
- sanity checks: `00000000`, `4031100000000`, `n/a`, `magnet` should all be
  absent from the materialised `text_codes` corpus.

If any number diverges materially from the doc's, halt and surface to user
before Phase 2.

## Phase 2 — Build `offers_codes` collection

### Files

- `scripts/milvus_bulk_import_offers_codes.py` — sibling of
  `milvus_bulk_import_playground.py`.

### Schema

```python
schema = client.create_schema()
schema.add_field("id",          DataType.VARCHAR, max_length=64, is_primary=True)
schema.add_field("text_codes",  DataType.VARCHAR, max_length=512, mmap_enabled=True,
                 enable_analyzer=True,
                 analyzer_params={
                     "tokenizer": "whitespace",
                     "filter": [
                         "lowercase",
                         {"type": "length", "min": 4, "max": 40},
                     ],
                 })
schema.add_field("sparse_codes", DataType.SPARSE_FLOAT_VECTOR)

schema.add_function(Function(
    name="bm25_codes",
    function_type=FunctionType.BM25,
    input_field_names=["text_codes"],
    output_field_names=["sparse_codes"],
))
```

Index (defined **before** the first bulk-insert submit, per APRIL_21.md):

```python
idx.add_index("sparse_codes",
              index_type="SPARSE_INVERTED_INDEX",
              metric_type="BM25",
              params={"mmap.enabled": "true"})
```

Memory invariant: only `id` and `sparse_codes` (the BM25 index head) are
RAM-resident; `text_codes` field data is mmap-backed; the inverted posting
lists are mmap-backed via `mmap.enabled=true` on the index params.

### Pipeline

Mirror `scripts/milvus_bulk_import_playground.py`:
1. `connect()` → `build_collection(drop_existing=True)` → define index.
2. Stage each `offers_codes_staging.parquet/part-NN.parquet` as-is to MinIO
   (no bytes-level conversion needed — no dense vector). Submit
   `do_bulk_insert` immediately.
3. `wait_for_jobs` → `flush` → `wait_index_finished("sparse_codes")` →
   `col.load()`.

### Post-load smoke tests

```python
client.search("offers_codes", data=["00000000"],     anns_field="sparse_codes",
              limit=10, search_params={"metric_type": "BM25"})  # → 0 hits
client.search("offers_codes", data=["4031100000000"], …)        # → 0 hits
client.search("offers_codes", data=["n/a"],          …)         # → 0 hits
client.search("offers_codes", data=["magnet"],       …)         # → 0 hits
client.search("offers_codes", data=["rj45"],         …)         # → many
```

Document the wall-clock + final `num_entities` in `reports/codes_audit.md`.

## Phase 3 — Hybrid query path + parametrised `_search`

### New module: `search-api/hybrid.py`

Exports (signatures stable; both `search-api` and `playground-app` import
this so behaviour stays identical across surfaces):

```python
class Mode(StrEnum):
    VECTOR             = "vector"
    BM25               = "bm25"
    HYBRID             = "hybrid"               # always run both, RRF
    HYBRID_CLASSIFIED  = "hybrid_classified"    # classifier routes

@dataclass(slots=True)
class SearchParams:
    mode: Mode = Mode.HYBRID_CLASSIFIED
    k: int = 24                       # final top_n returned to the caller
    dense_limit: int = 200            # candidate pool from offers
    codes_limit: int = 20             # codes pool in hybrid path
    strict_codes_limit: int = 500     # codes pool in strict path
    rrf_k: int = 60
    num_candidates: int | None = None # HNSW efSearch
    enable_fallback: bool = True      # 0-result fallback on strict path

@dataclass(slots=True)
class Hit:
    id: str
    score: float
    source: Literal["dense", "bm25", "rrf"]    # what produced this score

def is_strict_identifier(q: str) -> bool: ...
def rrf_merge(result_lists, k: int, top_n: int) -> list[tuple[str, float]]: ...

async def run_search(
    q: str,
    params: SearchParams,
    *,
    dense_client: MilvusClient,
    codes_client: MilvusClient,
    embed: Callable[[str], Awaitable[list[float]]],
    dense_collection: str = "offers",
    codes_collection: str = "offers_codes",
) -> tuple[list[Hit], dict]:
    """Returns (hits, debug_info). debug_info includes path taken, per-leg
    latency, per-leg hit counts, and whether fallback fired."""
```

Behaviour by mode:
- `vector`      — dense only. `params.dense_limit` is the candidate pool;
  the response is truncated to `params.k`. `num_candidates` = HNSW `ef`.
- `bm25`        — codes only with `params.codes_limit`, BM25 score returned.
- `hybrid`      — dense + codes always run, RRF-fused with `rrf_k`. The
  classifier is **not** consulted (this is the "unconditional hybrid"
  knob the user asked for).
- `hybrid_classified` — runs `is_strict_identifier`. If true: BM25 only with
  `strict_codes_limit`, scored by raw BM25; if 0 results and
  `enable_fallback`, fall through to the same path `hybrid` would have
  taken. Otherwise: `hybrid`.

The classifier is exactly the regex set from `hybrid_v0.md §Query
classifier`, including the explicit reject test cases (`post-it`, `t-shirt`,
`u-power`, `o-ringe`, `uni-ball`).

Strict-path tiebreaker (decision #5): when BM25 returns scores tied to
machine precision, sort tied groups ascending by `id`. Implemented by a
secondary sort key after the BM25 score.

### `search-api/main.py` changes

- Add `MILVUS_CODES_COLLECTION` env (default `offers_codes`).
- Open a second `MilvusClient` (sharing `MILVUS_URI`) for codes — keeps
  hybrid_v0.md's "decoupled clients" stance and sidesteps any future
  per-collection connection settings.
- Extend the `POST /{collection}/_search` handler: parse all `SearchParams`
  fields from query string (with the existing `k`/`num_candidates` for
  backward compatibility). Drop straight through to `run_search`. Include
  the `debug_info` from `run_search` under a `_debug` key, gated by a
  `?debug=1` flag so production responses stay tight.
- Output `_score` semantics: BM25 raw on `bm25` and strict path; cosine on
  `vector`; RRF score on hybrid + hybrid_classified non-strict; the `source`
  on each hit makes this unambiguous on the wire.

### `playground-app` changes

- Add the same knobs to the search form — a `mode` dropdown, sliders/inputs
  for `dense_limit`, `codes_limit`, `rrf_k`, a `enable_fallback` checkbox —
  defaulted to `hybrid_classified` + fallback ON.
- `playground-app/milvus_search.py` becomes a thin caller of the same
  `hybrid.run_search` (so behaviour is provably identical to search-api).
  The display-field lookup (`name`, `manufacturerName`, `category_*`, etc.)
  stays a `MilvusClient.query` against `offers` keyed by the IDs that came
  back from `run_search`. The codes collection has no display fields.
- Surface in the debug panel: the path taken (`strict` / `hybrid` /
  `fallback`), per-leg latency, per-leg hit counts.

### Tests

`tests/test_hybrid.py` (new):
- `is_strict_identifier`: every doc example for both branches (matches +
  rejects), length-floor, length-cap, classification-rate sanity on a fixed
  list of canned PostHog-shaped queries.
- `rrf_merge`: tie ordering, partial overlap, single-list, zero-list.
- `run_search` with stub clients/embed: every mode hits the right legs;
  strict-path 0-result fallback only fires when `enable_fallback=True`;
  `hybrid_classified` + free-text query equals the `hybrid` result for that
  query.

## Phase 4 — Validation

Output: `reports/hybrid_v0_validation.md`.

### PostHog dataset extraction (one-time, scripted)

`scripts/fetch_posthog_search_queries.py`:
- Reads `POSTHOG_PERSONAL_API_KEY` / `POSTHOG_PROJECT_ID` / `POSTHOG_HOST`
  from `.env`.
- Pulls 30 days of `search_performed` events via the HogQL `query` API.
- Writes three artefacts to `reports/validation/`:
  - `top200_queries.tsv` — most-frequent queries, with their counts. Used
    for classifier precision spot-check.
  - `eans_seen.tsv` — distinct queries that match the EAN regex, with
    counts. Sample 100 by frequency for the recall check.
  - `freetext_seen.tsv` — distinct queries that do **not** match the
    classifier regex, with counts. Sample 100 by frequency for the regression
    check.

(All three live under `reports/validation/` so they're easy to refresh.)

### Validation runs

`scripts/validate_hybrid.py` reads the three TSVs and runs:
1. **Classifier precision** on `top200_queries.tsv`. Print every flagged
   query with its top 3 BM25 hits; manual eyeball check; capture pass/fail
   in the report.
2. **Codes recall@5** on the 100 sampled EAN queries: each must surface a
   match in the top 5.
3. **Free-text regression** on the 100 sampled free-text queries: classifier
   rejects them; in `hybrid` mode codes contributes ≤ X% of top 24 (define
   X — initial target 5%); fused result top 24 IDs overlap with dense-only
   top 24 by ≥ 95%.
4. **0-result fallback**: hard-coded `din912`, an off-catalog EAN, a typo'd
   identifier — all return non-empty results.
5. **Top-of-fused-page noise**: 50 ambiguous-shape queries (sampled from
   `freetext_seen.tsv` filtered for digit-containing patterns); manual eyeball
   over top 10 in `hybrid_classified`.
6. **Latency** — extend `scripts/milvus_bench.py` to drive each `mode`
   against a 200-query sample. Record p50/p95 strict-only / hybrid /
   dense-only.

## Phase 5 — Wire-up & monitoring

### Compose / env

- Append `MILVUS_CODES_COLLECTION=offers_codes` and `HYBRID_DEFAULT_MODE=hybrid_classified`
  to `playground-app/.env.example` and `playground-app/.env`.
- No new processes — the codes collection lives in the same Milvus,
  reusing the existing service.

### Monitoring (doc-only deliverable)

The list of A/B metrics from hybrid_v0.md §"Things to monitor in the A/B"
goes verbatim into the validation report's tail, with the PostHog event
field names that should carry them. Wiring those to PostHog is a separate
ticket — out of scope for this PR.

## Out of scope (deliberately, mirroring hybrid_v0.md)

- Re-ingesting the dense collection. We don't touch `offers`.
- Cross-encoder rerank, query rewriting, synonyms, typo tolerance.
- A `sparse_main` BM25 over titles/descriptions — only `sparse_codes`.
- Per-source weight tuning (`WeightedRanker`).
- The strict-path secondary sort beyond the `id`-asc tiebreaker — see
  hybrid_v0.md "Out of scope".

## File touch list (for review)

- `scripts/build_offers_codes_staging.py`            — new (Phase 1)
- `scripts/milvus_bulk_import_offers_codes.py`       — new (Phase 2)
- `search-api/hybrid.py`                              — new (Phase 3)
- `search-api/main.py`                                — extended (Phase 3)
- `playground-app/main.py`                            — extended (Phase 3)
- `playground-app/milvus_search.py`                   — refactored (Phase 3)
- `playground-app/templates/index.html` (+ partials)  — extended (Phase 3)
- `tests/test_hybrid.py`                              — new (Phase 3)
- `scripts/fetch_posthog_search_queries.py`           — new (Phase 4)
- `scripts/validate_hybrid.py`                        — new (Phase 4)
- `scripts/milvus_bench.py`                           — extended (Phase 4)
- `playground-app/.env.example`, compose.yaml         — bumped (Phase 5)
- `reports/codes_audit.md`                            — generated (Phase 1+2)
- `reports/hybrid_v0_validation.md`                   — generated (Phase 4)
