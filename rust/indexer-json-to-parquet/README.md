# indexer-json-to-parquet

Rust materializer for the local MongoDB export.

It reads the 4 gzipped NDJSON collections:
- `offers/`
- `pricings/`
- `coreArticleMarkers/`
- `customerArticleNumbers/`

and writes final parquet outputs:
- `articles/*.parquet`
- `offer_rows/*.parquet`

## Properties

- fully Rust
- memory-bounded via hash partitioning
- no TEI / Redis / Milvus / embeddings
- reproduces the current DuckDB materialization semantics for final outputs
- `s2class_code` uses the real binary mappings in `indexer/classification_mapping/`

## How it stays memory-bounded

It does not build the full join in memory.

Stages:
1. partition each source collection by `(vendor, articleNumber)` hash
2. process one join bucket at a time into final `offer_rows` parquet plus article partials
3. repartition article partials by `article_hash`
4. reduce one article bucket at a time into final `articles` parquet

`--memory-limit-gb` drives the default bucket counts and worker fanout.
For now the intended default is `80` GB.

## Build

```bash
cd /workspace/rust/indexer-json-to-parquet
/home/agent/.cargo/bin/cargo build --release
```

## Run

Full pipeline:

```bash
/workspace/rust/indexer-json-to-parquet/target/release/indexer-json-to-parquet \
  run \
  --source-root /data/mongodb-export-2026-03-04 \
  --output-root /data/mongodb-export-2026-03-04/indexer_parquet_rust \
  --artifact-root /data/mongodb-export-2026-03-04/indexer_parquet_rust_tmp \
  --memory-limit-gb 80 \
  --overwrite
```

Default temp-artifact compression is now:
- `--temp-compression zstd`
- `--temp-zstd-level 1`

Useful knobs:
- `--join-buckets`
- `--article-buckets`
- `--threads`
- `--parser-workers`
- `--join-bucket-workers`
- `--article-bucket-workers`
- `--offer-file-limit`, `--pricing-file-limit`, `--marker-file-limit`, `--can-file-limit`

## Phase commands

The binary now supports running each phase in isolation against an artifact root.

Example sequence:

```bash
/workspace/rust/indexer-json-to-parquet/target/release/indexer-json-to-parquet \
  partition-offers \
  --artifact-root /tmp/indexer_artifact \
  --offer-file-limit 128 \
  --overwrite

/workspace/rust/indexer-json-to-parquet/target/release/indexer-json-to-parquet \
  partition-pricings \
  --artifact-root /tmp/indexer_artifact \
  --pricing-file-limit 128 \
  --overwrite

/workspace/rust/indexer-json-to-parquet/target/release/indexer-json-to-parquet \
  partition-markers \
  --artifact-root /tmp/indexer_artifact \
  --overwrite

/workspace/rust/indexer-json-to-parquet/target/release/indexer-json-to-parquet \
  partition-cans \
  --artifact-root /tmp/indexer_artifact \
  --overwrite

/workspace/rust/indexer-json-to-parquet/target/release/indexer-json-to-parquet \
  process-join-buckets \
  --artifact-root /tmp/indexer_artifact \
  --output-root /tmp/indexer_output \
  --overwrite

/workspace/rust/indexer-json-to-parquet/target/release/indexer-json-to-parquet \
  repartition-article-partials \
  --artifact-root /tmp/indexer_artifact \
  --overwrite

/workspace/rust/indexer-json-to-parquet/target/release/indexer-json-to-parquet \
  materialize-articles \
  --artifact-root /tmp/indexer_artifact \
  --output-root /tmp/indexer_output \
  --overwrite
```

Artifacts written under `--artifact-root`:
- `partition/offers/`
- `partition/pricings/`
- `partition/markers/`
- `partition/cans/`
- `article_partials/`
- `article_buckets/`
- `partition_artifact.json`
- `article_buckets_artifact.json`

## Fast verification

Small real-data smoke run:

```bash
/workspace/rust/indexer-json-to-parquet/target/release/indexer-json-to-parquet \
  run \
  --source-root /data/mongodb-export-2026-03-04 \
  --output-root /tmp/indexer_parquet_rust_smoke \
  --artifact-root /tmp/indexer_parquet_rust_smoke_tmp \
  --offer-file-limit 1 \
  --pricing-file-limit 1 \
  --marker-file-limit 1 \
  --can-file-limit 1 \
  --overwrite
```

Verify against the DuckDB reference:

```bash
cd /workspace
.venv/bin/python scripts/verify_indexer_materialized_rust.py \
  --offers-glob '/data/mongodb-export-2026-03-04/offers/atlas-fkxrb3-shard-0.0.json.gz' \
  --pricings-glob '/data/mongodb-export-2026-03-04/pricings/atlas-fkxrb3-shard-0.0.json.gz' \
  --markers-glob '/data/mongodb-export-2026-03-04/coreArticleMarkers/atlas-fkxrb3-shard-0.0.json.gz' \
  --cans-glob '/data/mongodb-export-2026-03-04/customerArticleNumbers/atlas-fkxrb3-shard-0.0.json.gz' \
  --output-root /tmp/indexer_parquet_rust_smoke
```
