# Task: Export offer embeddings for Milvus bulk import

## Goal

Create embeddings for all records in `../../data/offers_grouped.parquet/` (16 shards, ~160M rows) and produce Parquet suitable for Milvus bulk import.

## Output columns

- `id` — SHA256 hash (truncated to 16 hex chars) of the 8 identity fields (`name`, `manufacturerName`, `description`, `categoryPaths`, `ean`, `article_number`, `manufacturerArticleNumber`, `manufacturerArticleType`)
- `offer_embedding` — float32 vector (dim=128)
- `vendor_listings` — JSON string (serialized from the source's nested `list<struct>`)

## Checkpoint

`checkpoints/useful-cub-58/best-step=4880-val_full_catalog_ndcg_at_5=0.7379.ckpt`

## Approach

### Step 1: Embed with `embedding-infer-parallel`

The source data uses camelCase column names but the renderer templates expect snake_case. Use the `column_rename` data config option to remap them. Either pass a Hydra override or create a data config variant with:

```yaml
column_rename:
  manufacturerName: manufacturer_name
  categoryPaths: category_paths
  manufacturerArticleNumber: manufacturer_article_number
  manufacturerArticleType: manufacturer_article_type
```

Run inference across all 8 GPUs, using `--use-source-shards` to skip re-splitting the already-partitioned input:

```bash
embedding-infer-parallel \
  --checkpoint checkpoints/useful-cub-58/best-step=4880-val_full_catalog_ndcg_at_5=0.7379.ckpt \
  --input ../../data/offers_grouped.parquet/ \
  --output ../../data/offers_embedded/ \
  --num-shards 8 \
  --use-source-shards \
  --embedding-precision float32 \
  --copy-columns vendor_listings \
  --overwrite
```

### Step 2: Post-process into Milvus format

Transform the output into the final 3-column schema. Can be done with DuckDB or a small script:

- Compute `id`: SHA256 hash (first 16 hex chars) of the 8 identity fields joined with `|`, each JSON-serialized
- Serialize `vendor_listings` to a JSON string
- Keep `offer_embedding` as-is (already float32 list)

## Blocker

CUDA error 46 (`cudaErrorDevicesUnavailable`) — all 8 H100s are visible to `nvidia-smi` but `cudaSetDevice` fails. This is a system-level issue (cgroup/driver/fabric manager). Must be resolved before running.
