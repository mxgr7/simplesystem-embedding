# offers-to-offer-projected

Fast Rust converter for Mongo-style `*.json.gz` offer exports -> parquet shards.

## What it writes

A projected parquet dataset with hot scalar columns such as:

- `offer_id`
- `article_number`
- `vendor_id`
- `catalog_version_id`
- `import_epoch`
- `name`
- `description`
- `ean`
- `manufacturer_name`
- `manufacturer_article_number`
- `open_price`, `open_currency`
- `closed_price`, `closed_currency`
- array counts for `keywords`, `features`, `images`, `downloads`, `pricings`, `markers`

Optional flags can also include compact JSON blobs and full raw JSON.

## Build

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /workspace/rust/offers-to-offer-projected
cargo build --release
```

The crate is configured for `target-cpu=native`, LTO, symbol stripping, and `zlib-rs` gzip decoding.

## Fastest run

For a 32-core box:

```bash
/workspace/rust/offers-to-offer-projected/target/release/offers-to-offer-projected \
  --input-glob '/data/mongodb-export-2026-03-04/offers/*.json.gz' \
  --output-dir /data/mongodb-export-2026-03-04/offers_parquet_projected \
  --limit 2048 \
  --threads 32 \
  --shards 128 \
  --batch-rows 8192 \
  --row-group-rows 131072 \
  --compression snappy \
  --overwrite
```

If you need absolute max conversion speed and can afford larger parquet output:

```bash
/workspace/rust/offers-to-offer-projected/target/release/offers-to-offer-projected \
  --input-glob '/data/mongodb-export-2026-03-04/offers/*.json.gz' \
  --output-dir /data/mongodb-export-2026-03-04/offers_parquet_projected \
  --limit 2048 \
  --threads 32 \
  --shards 128 \
  --batch-rows 8192 \
  --row-group-rows 131072 \
  --compression uncompressed \
  --overwrite
```

## Notes

- Default mode is the fast path: projected columns only.
- `--include-json-blobs` and `--include-raw-json` are slower and write more data.
- The converter supports:
  - NDJSON inside gzip
  - top-level JSON arrays inside gzip
  - top-level JSON objects with a `records` array
- Throughput depends mostly on total compressed bytes, not just file count.

## Local benchmark in this workspace

Using 2048 copies of the sample gzip fixture on this workspace host:

- input: `627.83MB` compressed
- rows: `409600`
- output: `8.89MB` with `--compression snappy`
- wall time: `9.553s`

So the `< 60s for 2048 chunks` target is realistic for small-to-medium gzip shards, but the real limit is total compressed input size, gzip ratio, and storage throughput.
