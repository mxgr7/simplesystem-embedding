# ESCI dataset build scripts

Scripts that produced `/data/datasets/queries_offers_esci/` — an ESCI-style
relevance dataset (775,728 labeled `(query, offer)` pairs over 20,000
queries) for training/evaluating the simplesystem cross-encoder reranker.

The canonical description of the dataset itself (schema, row counts, label
distribution, splits, known limitations) lives in the dataset folder:

  `/data/datasets/queries_offers_esci/README.md`

This README documents the **scripts**: what each one does, the run order,
prerequisites, and how to reproduce the build from scratch.

---

## At a glance

```
posthog_queries.parquet  (1.086M events, 90 days)
        │  curate_esci_queries.py
        ▼
queries.parquet  (20,000 queries, stratified)
        │  retrieve_esci_candidates.py     ← needs search-api at :8001
        ▼
candidates.parquet  (644,083 rows)
        │  materialize_esci_dataset.py     (DuckDB join)
        ▼
queries_offers_merged.parquet
        │  annotatorv3 (xAI Batch, grok-4-1-fast-reasoning, ~$135)
        │   experiment: simplesystem_batch.yaml
        ▼
annotations-full.jsonl
        │  extract_esci_labels.py
        ▼
queries_offers_merged_labeled.parquet  (644,083 rows, label column)
        │  split_esci_dataset.py
        ▼
+ split column

——— Hard-negatives addendum ———

mono-{Exact,Substitute,Complement} queries  (4,555 queries)
        │  retrieve_mono_label_negatives.py   ← needs search-api at :8001
        ▼
candidates_mono_label_negatives.parquet  (131,670 rows, 4,389 queries)
candidates_mono_label_negatives_no_below.json  (166 skipped)
        │  materialize_mono_label_negatives.py
        ▼
queries_offers_merged_mono_label_negatives.parquet
        │  annotatorv3 (xAI Batch, same model+prompt, ~$30)
        ▼
annotations-mono-neg-full.jsonl
        │  append_mono_label_negatives.py
        ▼
queries_offers_merged_labeled.parquet  (775,728 rows)  ← final
```

Helper scripts that are not part of the build chain but were used during
development:
- `sample_esci_dataset.py` — stratified 100-row sample for prompt iteration
- `compare_annotation_iterations.py` — diff two annotator runs on the same rows
- `inspect_esci_dataset.py` — sanity stats for queries + candidates
- `explore_esci_labels.py` — quality stats over the labeled parquet

---

## Scripts

| Script | Purpose | Inputs | Outputs |
| --- | --- | --- | --- |
| `curate_esci_queries.py` | Aggregate 90 days of `search_performed` events into a 20K-query stratified sample. Dedups by `lower(normalizedQueryTerm)`, drops noise + bots, stratifies by `frequency_band × hit_band`, forces ≥2K MPN-shaped queries. | `/data/datasets/posthog_queries.parquet/*.parquet` | `queries.parquet` |
| `retrieve_esci_candidates.py` | Per query, hit search-api 3× (`hybrid_classified` k=30, `vector` k=20, `bm25` k=20), union by candidate id. Persists per-leg rank/score and `source_legs`. | `queries.parquet`, search-api at `localhost:8001` | `candidates.parquet`, `retrieval_failures.json` |
| `materialize_esci_dataset.py` | DuckDB join `candidates ⋈ queries ⋈ offers_embedded_full` on `id`. Excludes `offer_embedding`. Emits `example_id` 1..N. | `candidates.parquet`, `queries.parquet`, `offers_embedded_full` | `queries_offers_merged.parquet` |
| `extract_esci_labels.py` | Parse the annotator JSONL: prints sanity stats (records, parse failures, label dist, finish-reasons, cost), builds slim labels parquet, joins it onto the merged parquet. | `annotations-full.jsonl`, `queries_offers_merged.parquet` | `esci_labels.parquet`, `queries_offers_merged_labeled.parquet` (without split) |
| `split_esci_dataset.py` | Add stratified 80/10/10 `split` column at the `query_id` level. Stable hash seeded with `"esci-split-seed-42"`. | `queries_offers_merged_labeled.parquet` | rewrites parquet in place with `split` |
| `retrieve_mono_label_negatives.py` | Find queries whose entire labeled candidate set has one ESCI label (mono-{E,S,C}); for each, run dense retrieval at k=10000 and keep candidates with cosine < 0.70. Random-sample 30 per query (seed `f"42-{query_id}"`). Raises `NoBelowThreshold` for queries with zero sub-threshold hits — those ids are written to a skip-log. | `queries_offers_merged_labeled.parquet`, search-api at `localhost:8001` | `candidates_mono_label_negatives.parquet`, `candidates_mono_label_negatives_no_below.json` |
| `materialize_mono_label_negatives.py` | DuckDB join, mirror of `materialize_esci_dataset.py` for the addendum candidates. Continues `example_id` from `MAX(example_id) + 1` so IDs stay globally unique. | `candidates_mono_label_negatives.parquet`, `queries.parquet`, `offers_embedded_full`, `queries_offers_merged_labeled.parquet` (for max example_id) | `queries_offers_merged_mono_label_negatives.parquet` |
| `append_mono_label_negatives.py` | Parse addendum annotations, build slim sidecar (`esci_labels_mono_label_negatives.parquet`), `UNION ALL` into the canonical labeled parquet. Inherits `split` from the parent query. | `annotations-mono-neg-full.jsonl`, `queries_offers_merged_mono_label_negatives.parquet`, `queries_offers_merged_labeled.parquet` | rewrites canonical parquet in place via `.tmp + replace`; `esci_labels_mono_label_negatives.parquet` |
| `sample_esci_dataset.py` | 100-row stratified sample (20 per `hit_band`) for annotator-prompt iteration. | `queries_offers_merged.parquet` | `queries_offers_merged_sample100.parquet` |
| `compare_annotation_iterations.py` | Confusion matrix between two annotator runs over the same example_ids — used to evaluate prompt versions. | two `annotations-*.jsonl` files | stdout |
| `inspect_esci_dataset.py` | Quick stats over `queries.parquet` and `candidates.parquet` (counts, distributions, retrieval coverage). | `queries.parquet`, `candidates.parquet` | stdout |
| `explore_esci_labels.py` | Quality stats over the labeled parquet: label dist by stratum, rank-vs-label correlation, multi-leg-vs-Exact-share, mono-label query counts. | `queries_offers_merged_labeled.parquet` | stdout |

---

## Reproducing the build

### Prerequisites

- Python env with the project `pyproject.toml` (run scripts via `uv run`).
- `/data/datasets/posthog_queries.parquet/` with 90 daily files. Produced
  out-of-band; not checked into this folder. The window used for the
  current dataset was 2026-01-28 → 2026-04-27.
- A running search-api at `localhost:8001/offers/_search` over the
  production index (`offers-prod`). `SEARCH_API_KEY` from
  `~/shared/.env`. The build hits the api hundreds of thousands of times,
  so colocate the script and the api.
- `offers_embedded_full` parquet shards for the join (path is hard-coded
  inside `materialize_esci_dataset.py` — adjust if it has moved).
- annotatorv3 checkout with the experiment yamls:
  - `simplesystem_batch.yaml` — xAI Batch path (production)
  - `simplesystem_sync.yaml` — OpenRouter sync path (cross-check / dev)
- xAI API credit (~$135 for the initial 644K rows + ~$30 for the
  131K-row addendum at the time of writing, with `reasoning.effort=high`).

### Step-by-step

All paths are hard-coded inside the scripts; edit constants at the top of
each file before running if you want to point them elsewhere.

```bash
# --- core pipeline ---
uv run python scripts/esci_dataset/curate_esci_queries.py
uv run python scripts/esci_dataset/inspect_esci_dataset.py        # optional
uv run python scripts/esci_dataset/retrieve_esci_candidates.py
uv run python scripts/esci_dataset/materialize_esci_dataset.py
uv run python scripts/esci_dataset/sample_esci_dataset.py         # optional, 100-row sample for prompt iteration

# --- annotator: initial 644K rows ---
# from ~/annotatorv3:
uv run annotator experiment=simplesystem_batch                    # writes annotations-full.jsonl

uv run python scripts/esci_dataset/extract_esci_labels.py <annotations-full.jsonl>
uv run python scripts/esci_dataset/split_esci_dataset.py
uv run python scripts/esci_dataset/explore_esci_labels.py         # optional QA

# --- hard-negatives addendum ---
uv run python -u scripts/esci_dataset/retrieve_mono_label_negatives.py | tee retrieve.log
#   ^ -u: stdout is line-buffered; without it the progress log block-buffers
#         to a pipe and you see nothing for ~30 minutes.
uv run python scripts/esci_dataset/materialize_mono_label_negatives.py

# annotator again on the addendum (same prompt, same model, different data file)
# the experiment yaml's `data_files` already points at queries_offers_merged_mono_label_negatives.parquet
uv run annotator experiment=simplesystem_batch                    # writes annotations-mono-neg-full.jsonl

uv run python scripts/esci_dataset/append_mono_label_negatives.py <annotations-mono-neg-full.jsonl>
```

After the last step, `queries_offers_merged_labeled.parquet` is the final
canonical dataset (775,728 rows, `split` column set, addendum rows have
`example_id > 644,083`).

### Wallclock and cost reference (one full re-run)

| Stage | Time | Cost |
| --- | --- | --- |
| `curate_esci_queries.py` | ~1 min (DuckDB) | — |
| `retrieve_esci_candidates.py` (3 legs × 20K queries, 8 workers) | ~30 min | — |
| `materialize_esci_dataset.py` | ~2 min | — |
| Initial annotation (xAI Batch, 644K rows) | ~24 h | $135.27 |
| `extract_esci_labels.py` + `split_esci_dataset.py` | <1 min | — |
| `retrieve_mono_label_negatives.py` (k=10000 vector, 4,555 queries) | ~30 min | — |
| `materialize_mono_label_negatives.py` | <1 min | — |
| Addendum annotation (xAI Batch, 131,670 rows) | ~6 h | $29.80 |
| `append_mono_label_negatives.py` | <1 min | — |

---

## Determinism / seeds

| Stage | Seed |
| --- | --- |
| query curation (random subsampling) | `0` |
| 100-row prompt-iteration sample | `42` |
| train/val/test split | hash with salt `"esci-split-seed-42"` inside `NTILE(100)` |
| addendum 30-per-query selection | `f"42-{query_id}"` (per-query) |

Annotator output is **not** deterministic (LLM sampling). Iterations 1–3
on the prompt produced different labels on ~5% of the 100-row sample; the
final prompt v3 lives in `simplesystem_batch.yaml`.

---

## Common gotchas

- **Search-api `dense_limit` cap.** The `vector` mode of the search-api
  defaults to `dense_limit=200` even when you ask for `k=10000`. The
  retrieval scripts pass both `dense_limit=k` and `num_candidates=k` (HNSW
  `efSearch`) explicitly — don't strip those if you re-tune k.
- **Block-buffered stdout.** Long-running retrieval scripts buffer their
  progress log when stdout is a pipe (e.g. `… | tee log`). Run them with
  `python -u` or `PYTHONUNBUFFERED=1` if you want to watch progress.
- **`extract_esci_labels.py` finish_reason.** A non-empty `finish_reason`
  indicates the model returned a usable response; an empty string flags a
  truncated/API-level failure. The initial pass had 25 such failures
  (0.004%) which were dropped from the canonical labeled parquet — their
  `example_id`s show up as gaps in the otherwise-monotone sequence.
- **Mono-label queries that survive the addendum.** 166 queries have zero
  candidates with cosine < 0.70 even at k=10000 (`whiteboard`,
  `kugelschreiber`, `bürostuhl`, …): the catalog genuinely has thousands
  of legitimate matches and there is no out-of-cluster contrast to mine.
  Their ids are persisted in
  `candidates_mono_label_negatives_no_below.json`. A further 437 queries
  found < 30 sub-threshold candidates after dedup; total residual
  mono-{E,S,C} = 603 (was 4,555 before the addendum). For these, in-batch
  negatives at training time are the simplest treatment.
- **Mono-Irrelevant queries (1,579) untouched.** The addendum mines hard
  *negatives* for mono-{E,S,C}; mono-Irrelevant queries need positive
  mining (different problem, out of scope).
