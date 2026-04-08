# Embedding Model Benchmark Report

## Summary

We benchmarked three publicly available Hugging Face models on `/Users/max/Clients/simplesystem/data/queries_offers_eval.parquet` using the new `embedding-catalog-benchmark` script.

The benchmark uses dense embedding retrieval only:

- embed queries
- embed the deduplicated product catalog
- score every query against every product with exact exhaustive similarity
- rank products by score
- report `nDCG@K`, `Recall@K`, `MRR`, and `Precision@K`

The strongest zero-shot model in this setup was `intfloat/multilingual-e5-base`.

## Dataset

- Input rows: `4,913`
- Unique queries: `500`
- Deduplicated catalog size: `4,898`
- Queries with at least one `Exact` label: `465`
- Queries with at least one positive-gain label (`Exact`/`Substitute`/`Complement`): `487`

Label distribution in the input file:

- `Exact`: `3,889`
- `Irrelevant`: `337`
- `Complement`: `375`
- `Substitute`: `312`

## Models Evaluated

### 1. `microsoft/mdeberta-v3-base`

- Parameters: `278,218,752`
- Use case in this repo: generic encoder, not retrieval-tuned
- Result: extremely poor zero-shot retrieval quality

### 2. `BAAI/bge-m3`

- Parameters: `567,754,752`
- Strengths: multilingual retrieval model, supports dense, sparse, and multi-vector retrieval
- Important note: this benchmark only used its dense embeddings

### 3. `intfloat/multilingual-e5-base`

- Parameters: `278,043,648`
- Strengths: multilingual retrieval-tuned dense model
- Important note: evaluated with E5-style prefixes

## Commands Used

### `BAAI/bge-m3`

```bash
uv run embedding-catalog-benchmark \
  --input "/Users/max/Clients/simplesystem/data/queries_offers_eval.parquet"
```

This used the repo defaults after updating:

- `configs/model/base.yaml` -> `BAAI/bge-m3`
- query template -> `Represent this sentence for retrieving relevant products: {{ query_term }}`

### `intfloat/multilingual-e5-base`

```bash
uv run embedding-catalog-benchmark \
  --input "/Users/max/Clients/simplesystem/data/queries_offers_eval.parquet" \
  --model-name "intfloat/multilingual-e5-base" \
  --query-template "query: {{ query_term }}" \
  --offer-template "passage: Article Name: {{ name }}{% if ean %} EAN: {{ ean }}{% endif %}{% if article_number %} Article Number: {{ article_number }}{% endif %}{% if manufacturer_article_number %} Article Number (Manufacturer): {{ manufacturer_article_number }}{% endif %}{% if category_text %} Category: {{ category_text }}{% endif %}{% if manufacturer_article_type %} Article Type: {{ manufacturer_article_type }}{% endif %}{% if manufacturer_name %} Brand: {{ manufacturer_name }}{% endif %}{% if clean_description %} Description: {{ clean_description }}{% endif %}"
```

### `microsoft/mdeberta-v3-base`

```bash
uv run embedding-catalog-benchmark \
  --input "/Users/max/Clients/simplesystem/data/queries_offers_eval.parquet" \
  --model-name "microsoft/mdeberta-v3-base" \
  --query-template "{{ query_term }}"
```

## Results

| Model | Params | Wall Time | MRR | nDCG@10 | Recall@10 | Recall@100 | Precision@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `intfloat/multilingual-e5-base` | 278,043,648 | 85.67s | 0.764661 | 0.701559 | 0.853763 | 0.924731 | 0.448602 |
| `BAAI/bge-m3` | 567,754,752 | 125.45s | 0.515293 | 0.411176 | 0.679570 | 0.827957 | 0.253548 |
| `microsoft/mdeberta-v3-base` | 278,218,752 | 63.10s | 0.005601 | 0.002187 | 0.006452 | 0.066667 | 0.001720 |

Additional metrics:

### `intfloat/multilingual-e5-base`

- `nDCG@1`: `0.695257`
- `nDCG@5`: `0.689473`
- `nDCG@10`: `0.701559`
- `nDCG@100`: `0.757864`
- `Recall@1`: `0.713978`
- `Recall@5`: `0.827957`
- `Recall@10`: `0.853763`
- `Recall@100`: `0.924731`
- `Precision@1`: `0.713978`
- `Precision@5`: `0.547527`
- `Precision@10`: `0.448602`
- `Precision@100`: `0.077376`

### `BAAI/bge-m3`

- `nDCG@1`: `0.430493`
- `nDCG@5`: `0.409578`
- `nDCG@10`: `0.411176`
- `nDCG@100`: `0.495983`
- `Recall@1`: `0.432258`
- `Recall@5`: `0.606452`
- `Recall@10`: `0.679570`
- `Recall@100`: `0.827957`
- `Precision@1`: `0.432258`
- `Precision@5`: `0.318280`
- `Precision@10`: `0.253548`
- `Precision@100`: `0.056495`

### `microsoft/mdeberta-v3-base`

- `nDCG@1`: `0.002074`
- `nDCG@5`: `0.001840`
- `nDCG@10`: `0.002187`
- `nDCG@100`: `0.006210`
- `Recall@1`: `0.002151`
- `Recall@5`: `0.004301`
- `Recall@10`: `0.006452`
- `Recall@100`: `0.066667`
- `Precision@1`: `0.002151`
- `Precision@5`: `0.001720`
- `Precision@10`: `0.001720`
- `Precision@100`: `0.001183`

## Interpretation

### `mdeberta-v3-base`

`mdeberta-v3-base` is not a viable zero-shot dense retrieval baseline for this task. Its metrics are close to unusable for product search, and spot checks showed obviously irrelevant top results for natural queries.

### `BAAI/bge-m3`

`bge-m3` is much better than `mdeberta-v3-base`, but it underperformed `multilingual-e5-base` in this benchmark despite being larger.

The likely reason is methodological rather than architectural:

- `bge-m3` is designed to support dense, sparse, and multi-vector retrieval
- this benchmark only uses dense embeddings
- so the benchmark does not exploit one of `bge-m3`'s major strengths

### `intfloat/multilingual-e5-base`

`multilingual-e5-base` was the best dense-only zero-shot model in this evaluation.

It delivered:

- the best ranking quality
- the best retrieval quality
- lower wall time than `bge-m3`
- roughly half the parameter count of `bge-m3`

## Recommendation

For this repository's current benchmark and retrieval setup, prefer `intfloat/multilingual-e5-base` as the default public zero-shot baseline.

Reason:

- it is retrieval-tuned
- it is multilingual
- it performs best in the current dense-only benchmark
- it is smaller and faster than `bge-m3`

## Follow-Up Options

1. Switch the default base model to `intfloat/multilingual-e5-base`.
2. Add a BM25 baseline to compare lexical retrieval against dense retrieval.
3. Add a hybrid benchmark to test dense + lexical retrieval.
4. Add a top-K output file option so failure cases can be inspected query by query.
