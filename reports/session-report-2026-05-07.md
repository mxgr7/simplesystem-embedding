# Session Report — 2026-05-07

## Summary

This session had two major parts:

1. Build and validate a very fast Rust converter for MongoDB `offers/*.json.gz` into projected parquet.
2. Build a new fully-Rust end-to-end materialization pipeline for the local MongoDB export that produces final `articles/*.parquet` and `offer_rows/*.parquet` outputs without embeddings.

The first part reached production-scale performance and correctness targets for the projected-offers converter. The second part produced a working, memory-bounded Rust pipeline with exact parity on small and real-shard smoke runs, plus benchmark data that identified the remaining bottlenecks.

---

## Environment and constraints

- Working directory: `/workspace`
- Date: `2026-05-07`
- Python package management: `uv`
- `.venv` expected and used for Python-based verification
- No type annotations required for project work
- Available CPUs in the container: `16`
- Memory budget requested for the new Rust pipeline: `80 GB`

Primary source dataset:
- `/data/mongodb-export-2026-03-04/offers/*.json.gz`
- `/data/mongodb-export-2026-03-04/pricings/*.json.gz`
- `/data/mongodb-export-2026-03-04/coreArticleMarkers/*.json.gz`
- `/data/mongodb-export-2026-03-04/customerArticleNumbers/*.json.gz`

---

## Part 1 — Fast Rust `offers -> projected parquet` converter

### Goal

Build the fastest possible single-machine converter from MongoDB `offers/*.json.gz` to projected parquet, verify correctness against DuckDB using `offer_id` as the primary key, and add computed `s2class_code`.

### Implemented crate

Path:
- `rust/offers-to-offer-projected/`

Files added/updated:
- `rust/offers-to-offer-projected/src/main.rs`
- `rust/offers-to-offer-projected/Cargo.toml`
- `rust/offers-to-offer-projected/.cargo/config.toml`
- `rust/offers-to-offer-projected/README.md`
- `rust/offers-to-offer-projected/.gitignore`

### Key implementation details

- Streaming gzipped NDJSON parsing instead of whole-file parsing
- Fast typed borrowed deserialization on the hot path
- Configurable CLI including:
  - `--selection-mode lexical|largest`
  - `--input-format ndjson|auto`
  - `--compression snappy|uncompressed`
- `manifest.tsv` output mapping `source_file_id -> path`
- Actual real export shape support:
  - top-level NDJSON rows
  - nested `offer.offerParams`
  - Extended JSON numeric wrappers in `eclassGroups`

### Performance result

Command shape used for the successful target run:

```bash
/workspace/rust/offers-to-offer-projected/target/release/offers-to-offer-projected \
  --input-glob '/data/mongodb-export-2026-03-04/offers/*.json.gz' \
  --output-dir /data/mongodb-export-2026-03-04/offers_parquet_projected \
  --limit 2048 \
  --selection-mode lexical \
  --input-format ndjson \
  --threads 16 \
  --shards 64 \
  --batch-rows 32768 \
  --row-group-rows 262144 \
  --compression snappy \
  --overwrite
```

Result for the first `2048` lexical offer files:
- rows: `74,454,646`
- compressed input: `23.33 GB`
- parquet output: `12.79 GB`
- wall time: `46.57s`

This met the original target of converting `2048` chunks in under `60s`.

### Correctness verification

DuckDB verification on the first `128` files showed:
- source rows: `4,686,953`
- parquet rows: `4,686,953`
- duplicate `offer_id` in source: `0`
- duplicate `offer_id` in parquet: `0`
- missing in parquet: `0`
- missing in source: `0`
- column mismatches: `0`

### Commit created

Committed baseline converter work:
- commit: `fb429f7`
- message: `Add fast Rust offers to parquet converter`

---

## Part 1b — Add `s2class_code` to the Rust converter

### Requirement

Implement `s2class_code` to match legacy indexer behavior:
- ignore source `S2CLASS`
- choose the highest available non-S2 eClass version
- map through `indexer/classification_mapping/{version}-s2.bin.gz`
- fall back to `90909090`
- expand to ancestor hierarchy and store sorted unique `i32` values

### Implementation

Added in:
- `rust/offers-to-offer-projected/src/main.rs`

Main pieces:
- `init_s2_mappings()`
- `load_s2_mappings()`
- `default_s2class_hierarchy()`
- `derive_s2class_hierarchy(...)`
- `derive_s2class_hierarchy_from_value(...)`
- `expand_eclass_hierarchy(...)`
- `json_code_i32(...)`
- `BatchBuilder::append_s2class_codes(...)`

Schema update:
- new field after `customer_article_number_count`
- Arrow type: `List<Int32>`
- DuckDB type: `INTEGER[]`

### Tests

Rust unit tests covered:
- highest-version wins
- source `S2CLASS` ignored
- no fallback to lower version if the higher version exists but maps nothing
- default fallback behavior
- real mapping file behavior for `ECLASS_5_1`

Command:
```bash
cd /workspace/rust/offers-to-offer-projected && cargo test --release
```

Result:
- `5 passed`

### Performance impact

Small benchmark on `256` files:
- before `s2class_code`: `5.844s`
- after `s2class_code`: `6.132s`
- slowdown: `+0.288s` (~`4.9%`)

Full `2048`-file rebuild with `s2class_code`:
- rows: `74,454,646`
- input: `23.33 GB`
- output: `12.90 GB`
- wall: `48.896s`

### Validation status

- Schema confirmed in DuckDB
- Real parquet rows spot-checked and showed plausible expanded hierarchies
- Verification confidence was good but initially lacked a full automated end-to-end parity check on real source vs parquet for `s2class_code`

---

## Part 1c — High-confidence `s2class_code` verifier

### Goal

Build and run a high-confidence verifier that compares recomputed `s2class_code` from raw source against generated parquet across all `2048` generated offer files.

### Added

Script:
- `scripts/verify_offer_projected_s2class.py`

Report:
- `reports/validation/verify_offer_projected_s2class_2048.json`

### Implementation notes

The verifier:
- reads source files listed in the converter `manifest.tsv`
- recomputes `s2class_code` from raw source `offer.offerParams.eclassGroups`
- uses real binary mapping tables from `indexer/classification_mapping/*.bin.gz`
- ignores source `S2CLASS`
- joins against parquet on `offer_id`
- checks:
  - row counts
  - duplicate `offer_id`
  - missing rows on either side
  - exact `s2class_code` mismatches

An early version used slow per-row inserts into DuckDB for `s2map`; this was optimized by writing a CSV and loading it with DuckDB, reducing mapping-load time from hundreds of seconds to sub-second.

### Full verification result

Run against all `2048` files:
- source rows: `74,454,646`
- parquet rows: `74,454,646`
- source duplicate `offer_id`: `0`
- parquet duplicate `offer_id`: `0`
- missing in parquet: `0`
- missing in source: `0`
- mismatched `s2class_code`: `0`

Conclusion:
- no failures found
- no converter fixes were needed

---

## Part 2 — New fully-Rust end-to-end materialization pipeline

### Goal

Implement a new fully-Rust pipeline that reproduces the current `scripts/indexer_bulk_local_json.py` materialization behavior for final outputs only, excluding embeddings, and keeping memory bounded at a configurable level (`80 GB` for now).

### Canonical reference

The Python/DuckDB reference pipeline was traced through:
- `scripts/indexer_bulk_local_json.py`
- `indexer.bulk.run_bulk_indexer(...)`
- `indexer.duckdb_projection`
- `indexer.projection`

Target final outputs:
- `articles/*.parquet`
- `offer_rows/*.parquet`

Excluded:
- TEI embeddings
- Redis
- Milvus
- any sink behavior beyond parquet materialization

### Implemented crate

Path:
- `rust/indexer-json-to-parquet/`

Files added:
- `rust/indexer-json-to-parquet/src/main.rs`
- `rust/indexer-json-to-parquet/Cargo.toml`
- `rust/indexer-json-to-parquet/.cargo/config.toml`
- `rust/indexer-json-to-parquet/.gitignore`
- `rust/indexer-json-to-parquet/README.md`

### Architecture implemented

Current implementation is a bounded-memory spill-based hash-partition pipeline:

1. Parse and preproject `offers`
   - compute offer-only derived fields early
   - hash-partition by `(vendor, articleNumber)`
2. Parse `pricings`
   - compactly project pricing rows
   - hash-partition by `(vendor, articleNumber)`
3. Parse `coreArticleMarkers`
   - hash-partition by `(vendor, articleNumber)`
4. Parse `customerArticleNumbers`
   - hash-partition by `(vendor, articleNumber)`
5. Process one join bucket at a time
   - join the 4 collections on `(vendor, articleNumber)`
   - write final `offer_rows/*.parquet`
   - emit article partials
6. Repartition article partials by `article_hash`
7. Reduce article partitions into final `articles/*.parquet`

The implementation includes:
- real `s2class_code` mapping logic
- article-hash computation compatible with the Python/DuckDB logic
- final schema materialization for both output datasets
- input manifests written under `<output-root>/inputs/`
- run stats written to `<output-root>/run_stats.json`

### Memory control

CLI parameter:
- `--memory-limit-gb 80`

This currently influences:
- default join-bucket count
- article-bucket count
- worker fanout defaults

It is not a hard allocator-enforced cap, but the pipeline design is intended to keep working sets bounded by partitioning.

---

## Part 2b — Verification tooling for the Rust materializer

### Added

Script:
- `scripts/verify_indexer_materialized_rust.py`

### Verification strategy

The verifier:
- rebuilds canonical expected `expected_offers` and `expected_articles` using DuckDB SQL from raw input
- loads actual parquet outputs from the Rust pipeline
- canonicalizes list ordering and nested structs where needed
- compares full row sets keyed by:
  - `id` for `offer_rows`
  - `article_hash` for `articles`

An initial slow verifier path used full `init_macros()` and slow s2map loading; this was optimized by loading macros directly and building `s2map` through a fast CSV path, reducing verifier runtime dramatically.

---

## Part 2c — Fast iteration runs and parity results

The user requested that verification runs remain under `60s` during development. After optimizing the verifier, that was achieved.

### Tiny trimmed subset

Temporary trimmed dataset created under:
- `/tmp/indexer_mini/`

Composition:
- `1000` offer rows
- `10000` pricing rows
- `10000` marker rows
- `10000` customer-article-number rows

Pipeline run:
- runtime: `0.08s`

Verifier run:
- runtime: `0.77s`

Result:
- exact parity for both `offer_rows` and `articles`

### One real shard per collection

Output:
- `/tmp/indexer_parquet_rust_smoke2`

Pipeline runtime:
- `3.05s`

Rows:
- `offer_rows`: `34,979`
- `articles`: `27,718`

Verifier runtime:
- `8.32s`

Result:
- exact parity for both `offer_rows` and `articles`

---

## Part 2d — Benchmarks run for the Rust materializer

### Benchmark 1

Configuration:
- `offer-file-limit 1`
- `pricing-file-limit 100`
- markers unlimited
- cans unlimited
- `--threads 16`

Resolved files:
- offers: `1`
- pricings: `100`
- markers: `2`
- cans: `46`

Wall time:
- `90.95s`

Stage timings:
- partition offers: `1.52s`
- partition pricings: `42.77s`
- partition markers: `4.28s`
- partition cans: `36.58s`
- process join buckets: `4.22s`
- repartition article partials: `0.26s`
- materialize articles: `0.08s`

Rows:
- offers rows: `34,979`
- pricing rows: `30,002,658`
- marker rows: `1,115,453`
- can rows: `27,196,731`
- offer_rows rows: `34,979`
- article rows: `27,718`

Artifacts:
- `/tmp/indexer_bench_o1_p100_t16`
- `/tmp/indexer_bench_o1_p100_t16/run_stats.json`

### Benchmark 2

Configuration:
- `offer-file-limit 1`
- all pricings
- markers unlimited
- cans unlimited
- `--threads 16`

Resolved files:
- offers: `1`
- pricings: `4116`
- markers: `2`
- cans: `46`

Wall time:
- `1963.78s` (~`32.7 min`)

Stage timings:
- partition offers: `0.56s`
- partition pricings: `1717.44s`
- partition markers: `1.86s`
- partition cans: `34.98s`
- process join buckets: `208.06s`
- repartition article partials: `0.39s`
- materialize articles: `0.07s`

Rows:
- offers rows: `34,979`
- pricing rows: `1,205,698,532`
- marker rows: `1,115,453`
- can rows: `27,196,731`
- offer_rows rows: `34,979`
- article rows: `27,718`

Artifacts:
- `/tmp/indexer_bench_o1_pall_t16`
- `/tmp/indexer_bench_o1_pall_t16/run_stats.json`

---

## Bottleneck analysis

### Observed dominant bottleneck

For the current Rust materializer, the dominant cost is overwhelmingly:
- **full pricings ingestion / decompression / JSON parse / partition spill**

On the full-pricings benchmark, this stage alone took:
- `1717.44s`

### Secondary bottlenecks

- join-bucket processing (`208.06s` in the full-pricings benchmark)
- cans ingestion (`34.98s`)
- full-offers ingestion is expected to become a major bottleneck on a true all-offers run

### Not major bottlenecks

- markers ingestion
- article-partial repartition
- final article aggregation

### Full-offers estimate with current implementation

Estimated total runtime without limiting offers:
- optimistic: `~3.5h`
- likely: `~4h`
- pessimistic: `~5h`

This estimate was based on:
- the full-pricings benchmark
- expected scaling of offer parsing and downstream materialization across all offer files

---

## Key design discussion and conclusions

### Referenced side rows

It was clarified that for subset runs, many side-collection rows are irrelevant because their `(vendor, articleNumber)` key is absent from the selected offers set. This means side-key filtering can help subset runs a lot.

For the **full offers corpus**, however, it appears that the referenced pricing count is effectively the whole pricing corpus:
- total pricing rows observed: `1,205,698,532`

Conclusion:
- side-key filtering is highly useful for subset runs
- it likely saves little on the full-offers full-pricings case

### Can more parallelism reduce spill?

Conclusion:
- more parallelism can improve throughput
- it does **not** fundamentally reduce spill volume
- with a fixed `80 GB` budget, aggressive parallelism can even increase pressure and force smaller partitions / more temp files

### Better next architecture

A better longer-term design was discussed:

1. Parse, project, and join-key partition `offers`
2. Parse, project, and join-key partition `pricings`
3. Parse/group `coreArticleMarkers` and likely `customerArticleNumbers` compactly, keeping them in memory if feasible
4. Process one `(vendor, articleNumber)` join partition at a time
5. Write final `offer_rows` directly
6. Emit article partials partitioned by `article_hash`
7. Aggregate articles by `article_hash`

This architecture keeps the same broad bounded-memory shape, but aims to reduce spill overhead and simplify locality by making the join partitioning more explicit and reusable.

---

## Files and artifacts created in this session

### Rust crates
- `rust/offers-to-offer-projected/`
- `rust/indexer-json-to-parquet/`

### Python verification scripts
- `scripts/verify_offer_projected_s2class.py`
- `scripts/verify_indexer_materialized_rust.py`

### Reports
- `reports/validation/verify_offer_projected_s2class_2048.json`
- `reports/session-report-2026-05-07.md` (this file)

### Temporary benchmark outputs
- `/tmp/indexer_parquet_rust_smoke`
- `/tmp/indexer_parquet_rust_smoke2`
- `/tmp/indexer_bench_o1_p100_t16`
- `/tmp/indexer_bench_o1_pall_t16`
- `/tmp/indexer_mini_out`

---

## Current status at session end

### Completed
- fast Rust projected-offers converter built and committed
- `s2class_code` support implemented and validated
- high-confidence full `s2class_code` parity check run on all `2048` offer files with zero mismatches
- new fully-Rust final materialization pipeline implemented
- exact parity achieved on:
  - trimmed mini subset
  - 1 real shard per collection
- benchmark data collected for constrained-offer/full-pricings scenarios

### Open
- the new full Rust materializer is correct on smoke runs but not yet optimized enough for a truly fast full-corpus run
- main remaining work is performance redesign around pricing ingestion and spill reduction
- no full all-offers production benchmark was run in this session

---

## Recommended next steps

1. Redesign the full Rust materializer around a cleaner reusable join-key partition cache.
2. Pre-project pricings as early and compactly as possible.
3. Investigate keeping markers and possibly grouped customer-article-numbers in memory.
4. Benchmark different partition counts and denser intermediate formats.
5. Run larger but still bounded validations after each optimization step.
6. Once throughput improves, benchmark the full all-offers corpus.
