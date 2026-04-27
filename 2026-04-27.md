# Hybrid Search v0 — `offers_codes` Import + Classifier Tightening — 2026-04-27

End-to-end build of the BM25 codes companion collection beside the existing
159M-row dense `offers`, plus a fully parametrised search-api so the
playground can A/B every knob. Followed by two post-validation fixes:
the strict-identifier classifier was too loose (pre-empting hybrid for
generic tokens like `rj45`/`wd-40`), and the RRF merge wasn't
reproducible across calls.

Plan doc: `CODES_IMPORT.md` (frozen at decision time).
Design doc: `hybrid_v0.md` (the spec).

Commits today (all on `main`, not yet pushed):

```
ffe2646 Make RRF reproducible across calls
49d5120 Tighten strict-identifier classifier
a75e36b Add offers_codes collection and hybrid search modes
058a9af Add prometheus + grafana observability stack
```

## 1. Staging — `scripts/build_offers_codes_staging.py`

Two-pass DuckDB pipeline over the source parquet
(`/data/datasets/offers_embedded_full.parquet`, 159,275,274 rows). Pass A
builds the audit (frequency cap denylist, per-field survival counts).
Pass B writes 16 bucket files at
`/data/datasets/offers_codes_staging.parquet/bucket=NN.parquet`.

Filters per identifier slot (EAN / MPN / article_number):

| filter | applies to | drops |
|---|---|---|
| length 4–40 | all | trivial / runaway values |
| `^0+$`, `^9+$` | all | placeholder fillers |
| whitespace-containing | all | tokenizer hostility |
| placeholder literals (`k.a.`, `n/a`, `n.a.`, `null`, `aucune donnée`, `#ref!`) | all | per audit |
| length ∈ {8, 12, 13, 14} | EAN | non-EAN sneaking in |
| pure-letter | MPN, article | not a code |
| Excel scientific notation (`/^\d+(\.\d+)?e[+-]?\d+$/i`) | MPN, article | spreadsheet damage |
| frequency > 500 | all (post per-row dedupe) | denylist of 81 mass-noise tokens |

Outcome:

```
rows in source:                   159,275,274
rows kept (≥1 surviving id):      158,269,705
rows skipped (all 3 dropped):       1,005,569
distinct EAN values:               14,975,546
distinct MPN values:               22,011,435
distinct article_number values:    49,584,819
total wall time:                       232.3s
```

Audit at `reports/codes_audit.md`.

## 2. Bulk import — `scripts/milvus_bulk_import_offers_codes.py`

Same single-node Milvus 2.6.15 already hosting `offers` (159M) and
`offers_playground` (18.3M). 16 parallel `do_bulk_insert` jobs.

### Schema

```python
fields = [
    FieldSchema("id", VARCHAR, max_length=64, is_primary=True),
    FieldSchema("text_codes", VARCHAR, max_length=512,
                mmap_enabled=True,
                enable_analyzer=True,
                analyzer_params={
                    "tokenizer": "whitespace",
                    "filter": ["lowercase",
                               {"type": "length", "min": 4, "max": 40}],
                }),
    FieldSchema("sparse_codes", SPARSE_FLOAT_VECTOR),  # server-generated
]
schema.add_function(Function(
    name="bm25_codes",
    function_type=FunctionType.BM25,
    input_field_names=["text_codes"],
    output_field_names=["sparse_codes"],
))
```

Index: `SPARSE_INVERTED_INDEX`, `metric_type=BM25`,
`params={"mmap.enabled": "true"}`. Both **field-level** mmap on the
analyzed VARCHAR column **and** **index-level** mmap on the sparse
inverted index — the two switches APRIL_23's gotcha note called out.

### Run

```
rows:              158,269,705
upload+submit:           61.7s   (16 staging files)
ingest wait:          1168.6s   (19.5 min — PreImport→Import→Sort→IndexBuilding)
flush + load:           ~80s
total wall:           1311.2s   (21.9 min)
```

Smoke probes (post-load):

```
exact MPN  "tze-231"             →  5 hits, top score 18.3,  3 ms
exact EAN  "4031100000000"       →  3 hits, top score 16.9,  4 ms
non-existent "rj45zzz"           →  0 hits,                   2 ms
strict path on real ID            →  20-500-cap behaviour OK
```

## 3. search-api — `search-api/hybrid.py` + parametrised `_search`

Single endpoint `POST /{collection}/_search` now drives every leg
combination via query-string knobs. Modes:

```
vector             dense ANN only
bm25               BM25 over offers_codes only
hybrid             dense + bm25, RRF fused (classifier NOT consulted)
hybrid_classified  classifier picks strict (BM25, large limit) or hybrid;
                   strict-path 0-result fallback to hybrid (toggleable)
```

Knobs: `k`, `dense_limit`, `codes_limit`, `strict_codes_limit`, `rrf_k`,
`num_candidates`, `enable_fallback`, `debug`. All also surfaced in the
playground UI form (with localStorage persistence).

Two `MilvusClient` instances per the "decoupled clients" pattern in
`hybrid_v0.md` — same URI, separate connection state for future
per-collection settings.

`playground-app` was refactored to delegate search to search-api over
HTTP and use Milvus only for display-field lookup against `offers`.

## 4. Validation — `scripts/validate_hybrid.py`

PostHog pull at `scripts/fetch_posthog_search_queries.py` — 30 days of
`search_performed` events → 50,000 distinct queries, 340,418 total
events. Property carrying the query string is `queryTerm` (discovered
via `--inspect-properties`).

Six-step run:

| step | result |
|---|---|
| 1. classifier precision @ top-200 | 1/200 flagged, all real (`4003773035466`) |
| 2. EAN-recall@5 | 86/100 EAN-shaped queries return ≥1 hit |
| 3. free-text regression (overlap dense vs hybrid top-24) | median 100%, mean 99.9% |
| 4. fallback behaviour | `din912` → fallback fires; real EAN → 3 strict hits; gibberish → fallback fires |
| 5. (skipped, manual UI eyeballing) | — |
| 6. latency p50/p95/p99 over 100 mixed queries | vector 55.9 / 80.8 / 110.9 ms; bm25 4.9 / 7.6 / 11.2 ms; hybrid 59.8 / 86.4 / 121 ms; hybrid_classified 55.4 / 81.3 / 117 ms |

Full report at `reports/hybrid_v0_validation.md`.

## 5. Fix — strict-identifier classifier was too loose

Step 1 of validation was clean (1/200 flagged) but eyeballing the
hybrid_classified path on real PostHog volume revealed the real
problem: **industry-generic short tokens were getting routed to strict
when they shouldn't be.** Examples: `rj45`, `cr2032`, `wd-40`, `ffp2`,
`din912`, `lr44`, `dtw300`, `h07v-k`. These are class names, not
opaque MPNs — the dense leg handles them correctly with high recall;
BM25 atomic-token retrieval returns ≤ a handful of hits and the strict
path doesn't fall back when those few hits exist, so the user sees a
truncated/wrong result.

Two-part fix shipped in `49d5120`:

**A. Tightened regex.** Hyphenated and alpha-then-digit patterns now
demand both length ≥7 AND a meaningful digit count:

```python
ID_PATTERNS = [
    r"\d{8}",                                                    # EAN-8
    r"\d{12,14}",                                                # UPC/EAN-13/GTIN-14
    r"(?=.{7,}$)(?=(?:[^\d]*\d){3,})[a-z0-9]+(?:-[a-z0-9]+)+",   # hyphenated: 7+ chars, 3+ digits
    r"(?=.{7,}$)[a-z]+\d{4,}[a-z0-9]*",                          # alpha-digit: 7+ chars, 4+ digits
]
```

**B. Static denylist.** ~50 tokens (battery cells, network
connectors, cable categories, respirator classes, lubricants, metric
threads, IP ratings) for defense-in-depth — most are already filtered
by length, but the explicit list documents intent and protects against
future regex relaxation.

Impact on the 30-day distribution:

| | distinct queries | events |
|---|---|---|
| old strict | 4,712 (9.4%) | 14,023 (4.1%) |
| new strict | 3,569 (7.1%) | 10,194 (3.0%) |
| flipped to hybrid | 1,143 (2.3%) | 3,829 (1.1%) |

Top flipped tokens: `cr2032` (60 evt), `wd40` (55), `CR2032` (48),
`h07v-k` (26), `ffp2` (23), `LR44` (15), `rj45` (15), `din912` (13),
`ffp3` (13), `wd-40` (12). Exactly the bucket of false-positive
classifications we wanted to drop.

### What about real catalog codes that also got flipped?

A handful of legitimate short catalog codes flip to hybrid: `LIN090`
(40 evt), `L01820` (24), `92-600` (20), `S0720` (10), `LC422` (11).
These are real article numbers but fall below the 7-char floor.
They're not lost — the hybrid path's BM25 leg with `codes_limit=20`
still ranks them via exact-token match; they just don't get the
500-result strict treatment. **Worth monitoring** but the conservative
move was to ship the flip and revisit if user-facing recall drops.

The mirror classifier in `scripts/fetch_posthog_search_queries.py` was
synced so the validation TSVs reflect the same partition the API
produces.

## 6. Fix — RRF wasn't reproducible across calls

Three back-to-back hybrid calls for the same query could return
different hit lists. Two non-determinism sources:

1. **Final merge sort.** `rrf_merge` did
   `sorted(scores.items(), key=lambda x: -x[1])`, so tied fused
   scores broke by Python's stable sort over dict insertion order —
   which depends on which leg happened to insert that id first.

2. **Per-leg rank assignment.** The hybrid path's BM25 call passed
   `tied_id_asc=False` (only the strict path used `True`). When BM25
   returns several docs with identical scores, Milvus's order across
   the tie boundary is unspecified — so the rank-N vs rank-N+1
   assignment, and thus the RRF contribution `1/(k+rank)`, is
   non-deterministic.

Fix in `ffe2646`:

- Both `_dense_search` and `_bm25_search` now sort
  `(-score, id_asc)` unconditionally before returning. Removed the
  `tied_id_asc` flag — it was always-on now.
- `rrf_merge` final sort key changed from `-x[1]` to `(-x[1], x[0])`.

Verified live: three identical calls per query for `schraube`, `rj45`,
`tze-231`, `kabel` returned byte-identical 20-hit lists.

Two new unit tests pin the contract:
`test_ties_broken_by_id_ascending` and
`test_deterministic_across_leg_ordering`. Total tests: 18, all green.

## 7. Follow-up thoughts

1. **Monitor the 1,143 flipped queries.** The classifier tightening
   is conservative but routes some short article codes
   (`LIN090`, `L01820`, `92-600`, `S0720`, `LC422`) into hybrid mode
   instead of strict. Should compare top-5 result quality
   pre/post-flip on a sample of these once the change is on prod
   traffic. If recall@5 holds, leave the tightening as-is. If it
   drops materially, consider a third pattern: `≥6 chars AND ≥3
   digits` for hyphenated, with a slightly stricter denylist.

2. **Whitespace-tokenizer asymmetry: `DIN 912` vs `DIN912`.** PostHog
   logs show users typing both. The tokenizer splits on whitespace,
   so `DIN 912` indexes (and queries) as two separate tokens — `din`
   (filtered out by length≥4? no, `din` is 3 chars, dropped) and
   `912` (also length<4, dropped). Net effect: `DIN 912` returns
   nothing, `DIN912` returns the right thing. Worth either (a) a
   server-side query-time normalisation that strips whitespace
   between alphanumerics for short tokens, or (b) a client-side
   pre-rewrite. Short term: just live with it; users self-correct.

3. **`text_codes` 512-byte cap.** No real EAN/MPN gets near 512
   bytes, but a row's three fields are concatenated into a single
   `text_codes` value. Worst case: very long article numbers across
   all three fields. Not seen in audit; flag if it shows up.

4. **HNSW efSearch coupling.** When hybrid mode is selected,
   `dense_limit=200` (default) means we ask HNSW for 200 candidates,
   so `num_candidates` (efSearch) must be ≥200. The validator in
   `search-api/main.py` enforces this against the *effective* dense
   leg limit. Worth a cron-style check that no playground-saved
   query has `num_candidates < dense_limit`.

5. **Static denylist will drift.** The 50-token list is empirically
   grounded today but new generics will appear. Plan: rerun
   `fetch_posthog_search_queries.py` quarterly, sort the
   currently-strict-classified tokens by event count, eyeball the
   top 50 for industry-generic patterns, append.

6. **Observability.** `058a9af` ships prometheus
   instrumentation on search-api. Useful follow-up: a Grafana panel
   that breaks `_search` requests by `path` (vector/bm25/hybrid/
   strict/fallback). Strict-rate and fallback-rate are the two
   numbers most likely to surface a regression early. The `_debug`
   payload already contains both — easy to lift into metrics.

7. **The `id-asc` tiebreak is meaningless to the user.** It's a
   reproducibility hack, not a ranking signal. If a real
   product-level secondary sort is ever added (e.g., by vendor
   priority or stock status), it should slot in at the leg-sort
   step and the RRF final-sort step in lockstep. Document that
   constraint when the time comes.

8. **Bulk-importer reuse.** The two-pass DuckDB stager + parallel
   `do_bulk_insert` pattern in
   `scripts/milvus_bulk_import_offers_codes.py` is now the third
   variation on the same shape (after `offers` and
   `offers_playground`). A small shared helper module would cut
   ~150 lines off the next one. Not worth doing speculatively —
   wait for the fourth.

## Reproduce

```bash
# 1. Stage
nohup uv run --no-project --with duckdb --with pyarrow \
  python scripts/build_offers_codes_staging.py \
  > logs/build_offers_codes_staging.log 2>&1 &

# 2. Bulk import (~22 min)
nohup uv run --no-project --with pymilvus --with pyarrow --with boto3 \
  python scripts/milvus_bulk_import_offers_codes.py \
  > logs/milvus_bulk_import_offers_codes.log 2>&1 &

# 3. Build/restart search-api
cd playground-app && docker compose up -d --build search-api

# 4. Pull PostHog queries (5 min, hits HogQL)
uv run python scripts/fetch_posthog_search_queries.py

# 5. Validate end-to-end
uv run python scripts/validate_hybrid.py

# 6. Unit tests
uv run python -m unittest tests.test_hybrid -v
```
