# Full Catalog Embedding Run

**Date:** 2026-04-14/15
**Checkpoint:** `useful-cub-58/best-step=4880-val_full_catalog_ndcg_at_5=0.7379.ckpt`

## Configuration

| Parameter | Value |
|---|---|
| Mode | offer |
| Model | multilingual-e5-base (fine-tuned) |
| Embedding dim | 128 |
| Embedding precision | float16 |
| Encode batch size | 128 |
| Read batch size | 1024 |
| Num workers | 4 |
| Device | CUDA |
| Compression | zstd |

Column renames applied: `manufacturerName`, `categoryPaths`, `manufacturerArticleNumber`, `manufacturerArticleType`.

## Input

- **Source:** `data/offers_grouped.parquet/` (16 partitions, `bucket=00` through `bucket=15`)
- **Total rows:** 159,275,274

## Output

- **Destination:** `data/offers_embedded.parquet/` (16 partitions, matching source)
- **Columns:** `row_number` (int64), `id` (string), `offer_embedding` (list\<float16\>, dim 128)
- **Total rows written:** 159,275,274 (0 skipped)
- **Total size:** 41 GB (~2.6 GB per partition)

## Performance

| Bucket | Throughput (rows/sec) |
|---|---|
| 00 | 5,617.7 |
| 01 | 5,589.6 |
| 02 | 5,582.5 |
| 03 | 5,464.7 |
| 04 | 5,300.3 |
| 05 | 5,248.6 |
| 06 | 5,280.0 |
| 07 | 5,384.9 |
| 08 | 5,685.1 |
| 09 | 5,405.3 |
| 10 | 5,619.8 |
| 11 | 5,375.3 |
| 12 | 5,592.4 |
| 13 | 5,466.8 |
| 14 | 5,689.5 |
| 15 | 5,392.9 |

- **Mean throughput:** ~5,478 rows/sec
- **Wall time:** ~8 hours (20:47 UTC to 04:56 UTC)
- **~30 min per partition** (~10M rows each)

## Size breakdown

Per row (uncompressed): 256 bytes (embedding) + ~32 bytes (id) + 8 bytes (row_number) = ~296 bytes.
159M rows x 296 bytes = ~44 GB uncompressed. Float16 embeddings are high-entropy, so zstd achieves only ~7% compression, yielding 41 GB on disk.
