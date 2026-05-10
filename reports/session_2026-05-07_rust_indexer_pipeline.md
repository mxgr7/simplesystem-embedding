# Session Report — Rust Mongo Export → Parquet Pipeline

Date: 2026-05-07

## Goal

Build and benchmark a fully Rust, memory-bounded pipeline that reads the local MongoDB export collections:

- `offers`
- `pricings`
- `coreArticleMarkers`
- `customerArticleNumbers`

and materializes final parquet outputs:

- `offer_rows/*.parquet`
- `articles/*.parquet`

while keeping the pipeline phaseable, fast, and reusable via persisted intermediate artifacts.

---

## Earlier completed work carried into this session

### 1. Fast Rust offer-projection converter

A separate crate was built in:

- `/workspace/rust/offers-to-offer-projected`

It converts gzipped Mongo offer exports to projected parquet and includes `s2class_code` derivation matching legacy semantics.

Key already-established results:

- full 2048-file projected-offers run: about **48.9s**
- rows: **74,454,646**
- projected parquet parity verified with DuckDB
- `s2class_code` parity verified on the full 2048-file run

Existing commit for the baseline converter work:

- `fb429f7` — `Add fast Rust offers to parquet converter`

### 2. Initial full Rust materializer

A full-pipeline Rust crate existed in:

- `/workspace/rust/indexer-json-to-parquet`

It already supported:

- reading all 4 gzipped NDJSON collections
- partitioning by join key
- writing final `offer_rows` and `articles`
- bounded-memory operation
- small-subset verification against DuckDB

---

## Main work completed in this session

## A. Re-profiled and optimized offers phase 1.a

### What phase 1.a means

Offers-only phase:

- parse offer NDJSON
- preproject offer fields
- hash by `(vendor, articleNumber)`
- persist join-partition artifacts

### What was measured before the rewrite

On 512 offer files (`17,813,803` rows):

- old baseline (normal storage, old writer): **44.46s**
- old all-in-shm: **33.16s**
- old all-in-shm with fewer buckets:
  - 256 buckets: **22.23s**
  - 64 buckets: **19.54s**

Main finding:

- the old design spent a lot of time doing tiny scattered writes
- write syscall count was extremely high

### Rewrite implemented

Offers partitioning was rewritten to use:

- buffered per-partition appends
- memory-budget-derived flushes
- persisted partition artifacts
- fewer effective write calls

File changed:

- `/workspace/rust/indexer-json-to-parquet/src/main.rs`

### Post-rewrite offers benchmarks

Using `--offers-only --threads 16`:

- 16 files: **0.69s**
- 128 files: **5.20s**
- 512 files: about **21.8–22.2s**

Improvement versus earlier measurements:

- 16 files: about **56.9% faster**
- 128 files: about **39.5% faster**
- 512 files: about **50% faster** than the original baseline path

### Full-offers estimate from the improved path

Estimated before full run:

- phase 1.a full offers: about **10.5–11.5 min**

Later real full run result was very close:

- **11.31 min**

---

## B. Added temp-artifact compression and selected zstd level 1

### Motivation

Phase 1 artifacts were uncompressed and large. Sampling showed they were highly compressible.

### Implementation

Added temp artifact options:

- `--temp-compression uncompressed|zstd`
- `--temp-zstd-level <n>`

Files changed:

- `/workspace/rust/indexer-json-to-parquet/Cargo.toml`
- `/workspace/rust/indexer-json-to-parquet/src/main.rs`

### Measured 512-file offers temp compression tradeoff

With `join_buckets=256`:

- uncompressed:
  - time: **22.16s**
  - temp size: **21.465 GB**
- zstd level 1:
  - time: **23.50s**
  - temp size: **3.366 GB**
- zstd level 3:
  - time: **25.52s**
  - temp size: **2.944 GB**

Decision:

- use **zstd level 1** from here on out
- it provides a very strong footprint reduction with only a modest runtime penalty

The crate default was changed to:

- `--temp-compression zstd`
- `--temp-zstd-level 1`

---

## C. Reshaped the program into explicit reusable phases

The pipeline was converted from a mostly monolithic run shape into distinct subcommands that persist and consume artifacts.

### Current commands

- `run`
- `partition-offers`
- `partition-pricings`
- `partition-markers`
- `partition-cans`
- `process-join-buckets`
- `repartition-article-partials`
- `materialize-articles`

### Artifact semantics

Artifact root is now explicit via:

- `--artifact-root`

(`--temp-dir` remains as an alias)

### Persisted metadata added

To ensure downstream phases are compatible with upstream artifacts:

- `partition_artifact.json`
- `article_buckets_artifact.json`

These record things like:

- `join_buckets`
- `article_buckets`
- temp compression mode
- temp zstd level

### Verified split-phase execution

A full split phase chain was executed successfully on a 1-file-per-collection smoke test and then verified against DuckDB with zero mismatches.

Files updated:

- `/workspace/rust/indexer-json-to-parquet/src/main.rs`
- `/workspace/rust/indexer-json-to-parquet/README.md`

---

## D. Added progress reporting

### Phase 2 progress

Added 10-second progress updates to:

- `process-join-buckets`

Progress line includes:

- completed buckets / total buckets
- percent complete
- `offer_rows` written so far
- `article_partial_rows` written so far
- elapsed
- ETA

### Phase 3 progress

Added 10-second progress updates to:

- `repartition-article-partials`

Progress line includes:

- completed partial files / total partial files
- `article_partial_rows` processed so far
- elapsed
- ETA

### Phase 4 progress

Added 10-second progress updates to:

- `materialize-articles`

Progress line includes:

- completed article buckets / total buckets
- `article_rows` written so far
- elapsed
- ETA

All progress reporting was built and tested successfully.

---

## Benchmarks and runs performed in this session

## 1. Phase 1 end-to-end for first 128 offer files

Interpretation used:

- phase 1 = `partition-offers` + `partition-pricings` + `partition-markers` + `partition-cans`
- offers limited to first 128 lexical files
- side collections full
- temp compression zstd level 1

Artifact root:

- `/tmp/indexer_phase1_128_artifact`

Results:

- total phase 1 wall time: **186.92s** (**3.12 min**)
- `partition-offers`: **5.85s**
- `partition-pricings`: **177.88s**
- `partition-markers`: **0.49s**
- `partition-cans`: **2.08s**

Key conclusion:

- phase 1 is overwhelmingly dominated by `partition-pricings`

---

## 2. Full offers phase 1.a only

Command:

- `partition-offers`

Artifact root:

- `/data/mongodb-export-2026-03-04/indexer_artifact_all_offers_j1024_zstd1`

Settings:

- `join_buckets=1024`
- `temp_compression=zstd`
- `temp_zstd_level=1`
- `threads=16`

Results:

- offer files: **15,448**
- offer rows: **521,763,985**
- wall time: **678.53s** (**11.31 min**)
- artifact size after offers phase: **100.753 GB**

This persisted the full offers partition artifact for later reuse.

---

## 3. Reused side artifacts instead of recomputing them

Previously created phase-1 side artifacts still existed under:

- `/tmp/indexer_phase1_128_artifact`

Those compatible directories were moved into the full-offers artifact root:

- `partition/pricings`
- `partition/markers`
- `partition/cans`
- corresponding `inputs/*.txt`

Compatibility was checked via matching metadata:

- `join_buckets=1024`
- `temp_compression=Zstd`
- `temp_zstd_level=1`

After the move, full phase 1 artifacts all lived under:

- `/data/mongodb-export-2026-03-04/indexer_artifact_all_offers_j1024_zstd1`

with sizes:

- `partition/offers`: **100.752 GB**
- `partition/pricings`: **23.666 GB**
- `partition/markers`: **0.011 GB**
- `partition/cans`: **0.294 GB**

---

## 4. Full phase 2

Command:

- `process-join-buckets`

Artifact root:

- `/data/mongodb-export-2026-03-04/indexer_artifact_all_offers_j1024_zstd1`

Output root:

- `/data/mongodb-export-2026-03-04/indexer_output_all_offers_j1024_zstd1`

Results:

- wall time: **1358.17s** (**22m 38s**)
- `offer_rows_rows`: **521,763,985**
- `article_partial_rows`: **179,451,902**

The new 10-second progress reporting was active throughout the run.

---

## 5. Full phase 3

Command:

- `repartition-article-partials`

Artifact root:

- `/data/mongodb-export-2026-03-04/indexer_artifact_all_offers_j1024_zstd1`

Results:

- wall time: **322.02s** (**5m 22s**)

This produced the article-bucket artifact needed by phase 4.

---

## 6. Full phase 4

Command:

- `materialize-articles`

Artifact root:

- `/data/mongodb-export-2026-03-04/indexer_artifact_all_offers_j1024_zstd1`

Output root:

- `/data/mongodb-export-2026-03-04/indexer_output_all_offers_j1024_zstd1`

Results:

- wall time: **111.48s** (**1m 51s**)
- `article_rows_rows`: **159,228,293**

The new 10-second progress reporting was active throughout the run.

---

## Final produced artifacts and outputs

## Artifact root

- `/data/mongodb-export-2026-03-04/indexer_artifact_all_offers_j1024_zstd1`

Contains:

- `partition/offers/`
- `partition/pricings/`
- `partition/markers/`
- `partition/cans/`
- `article_partials/`
- `article_buckets/`
- `partition_artifact.json`
- `article_buckets_artifact.json`

## Final parquet output root

- `/data/mongodb-export-2026-03-04/indexer_output_all_offers_j1024_zstd1`

Contains:

- `offer_rows/*.parquet`
- `articles/*.parquet`

---

## Sanity checks on the final outputs

A structural and aggregate sanity report was generated:

- `/workspace/reports/validation/indexer_output_all_offers_j1024_zstd1_sanity.json`

### File counts and sizes

- `offer_rows/*.parquet`
  - **1024 files**
  - **106.105 GB**
- `articles/*.parquet`
  - **1024 files**
  - **40.158 GB**

### Row counts

- `offer_rows` parquet rows: **521,763,985**
  - matches phase 2 output exactly
- `articles` parquet rows: **159,228,293**
  - matches phase 4 output exactly

### Required keys populated

Offer rows:

- empty `id`: **0**
- empty `article_hash`: **0**
- empty `article_number`: **0**
- empty `vendor_id`: **0**
- empty `catalog_version_id`: **0**

Articles:

- empty `article_hash`: **0**
- empty `name`: **0**
- empty `text_codes`: **0**

### Non-degenerate content checks

Offer rows:

- rows with prices: **521,763,196**
- rows with currencies: **521,763,196**
- rows with features: **356,705,432**
- rows with markers: **6,236,923**

Articles:

- rows with category_l1: **149,564,788**
- rows with eclass5_code: **146,187,909**
- rows with eclass7_code: **46,203,729**
- rows with s2class_code: **159,228,293**
- rows with customer article numbers: **11,037,186**

### Notable oddity

- `delivery_time_days_max` max observed in `offer_rows`: **94,950**

This likely indicates dirty source data rather than a clear pipeline bug, but it should be noted.

### Important caveat

These were **sanity/structural checks**, not a full semantic parity job on the entire all-offers output.

---

## Current command/phase structure

## Phase 1 — source partitioning

Commands:

- `partition-offers`
- `partition-pricings`
- `partition-markers`
- `partition-cans`

Outputs:

- join-key partition artifacts under `--artifact-root`

## Phase 2 — join-bucket processing

Command:

- `process-join-buckets`

Consumes:

- partitioned offers/pricings/markers/cans artifacts

Writes:

- final `offer_rows/*.parquet`
- intermediate `article_partials/*`

## Phase 3 — article partial repartition

Command:

- `repartition-article-partials`

Consumes:

- `article_partials/*`

Writes:

- `article_buckets/*`

## Phase 4 — final article materialization

Command:

- `materialize-articles`

Consumes:

- `article_buckets/*`

Writes:

- final `articles/*.parquet`

## End-to-end wrapper

Command:

- `run`

---

## Approximate end-to-end timings observed for the full all-offers path

- Phase 1 offers partition: **11.31 min**
- Phase 2 join processing: **22.63 min**
- Phase 3 partial repartition: **5.37 min**
- Phase 4 article materialization: **1.86 min**

Notes:

- full side partition artifacts were reused rather than recomputed for the final all-offers run
- full side partitioning had earlier measured roughly **~180s** total for:
  - pricings
  - markers
  - cans

---

## Key code paths changed during this session

- `/workspace/rust/indexer-json-to-parquet/src/main.rs`
- `/workspace/rust/indexer-json-to-parquet/Cargo.toml`
- `/workspace/rust/indexer-json-to-parquet/README.md`

---

## Recommended next steps

1. Run higher-confidence semantic validation on larger/full final outputs if required.
2. Optimize `partition-pricings` further only if another significant speed gain is needed; current runtime is already much better than the earlier implementation.
3. Consider checking for article row count semantics more deeply, e.g. compare `count(distinct article_hash)` in `offer_rows` against `articles` row counts if needed.
4. If desired, commit the `indexer-json-to-parquet` phase-command, temp-compression, and progress-reporting changes.

---

## Bottom line

By the end of this session:

- the Rust full pipeline was reworked into reusable artifact-driven phases
- zstd level 1 temp artifact compression was adopted as the default
- progress reporting was added for long phases
- full all-offers phases 1–4 were run successfully
- final `offer_rows` and `articles` parquet outputs were produced on disk
- sanity checks indicate the outputs are structurally sound and internally consistent
