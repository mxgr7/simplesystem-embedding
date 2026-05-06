# Milvus Collections

| Collection | Rows | Source | Notes |
|---|---|---|---|
| `articles_v7` | 5,901,369 | ES export (`articles.ndjson`, Freund data) via `convert_es_to_milvus_parquet.py` | Current. Embedded via sstei.maxgerer.com. |
| `offers_v7` | 27,098,336 | ES export (`articles.ndjson`, Freund data) via `convert_es_to_milvus_parquet.py` | Current. |
| `articles_v6` | 982,753 | Partial F9 indexer run | Outdated due to changes since. |
| `offers_v6` | 1,963,499 | Partial F9 indexer run | Outdated due to changes since. |
| `articles_v6_dev` | 290 | Dev/test fixture | |
| `offers_v6_dev` | 297 | Dev/test fixture | |
| `articles_v8` | 290 | Dev/test fixture | |
| `offers_v8` | 297 | Dev/test fixture | |
| `offers` | 159,275,274 | Deprecated legacy import | For playground app only. |
| `offers_codes` | 158,269,705 | Deprecated legacy import | For playground app only. |
| `offers_playground` | 18,330,690 | Legacy import | For the other playground app. |
| `offers_playground_trimmed` | 196,821 | Legacy import | Trimmed version of `offers_playground`. |
