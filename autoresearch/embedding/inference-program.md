# inference-program: maximize embedding throughput

Autonomous optimization program for embedding inference speed on a single H100 80GB GPU.

## Objective

Optimize the `embedding-infer` pipeline so that it can embed the full `offers_grouped.parquet` dataset (~159M rows, 16 Hive-partitioned shards, 35 GB on disk) as fast as possible on a single H100 GPU. The full run will happen later — your job is to build and benchmark the fastest pipeline, measured on a 100k-row subset and extrapolated.

The model is `intfloat/multilingual-e5-base` (12 layers, 768-dim hidden, 128-dim projection head). Output precision is `float16`. Checkpoint: `checkpoints/useful-cub-58/best-step=4880-val_full_catalog_ndcg_at_5=0.7379.ckpt`.

## Fixed constraints

- **1 GPU**: NVIDIA H100 80GB HBM3. No multi-GPU parallelism.
- **Correctness**: output embeddings must be bit-identical to the current `embedding-infer` output at `--embedding-precision float16`. Any optimization that changes the output is invalid. Verify by diffing against the unoptimized baseline output on the benchmark subset.
- **Checkpoint**: use `checkpoints/useful-cub-58/best-step=4880-val_full_catalog_ndcg_at_5=0.7379.ckpt` for all runs.
- **Dependencies**: new packages may be added to `pyproject.toml` if they help (e.g. ONNX Runtime, TensorRT, `optimum`, `bettertransformer`). Justify each addition.

## Column mapping

`offers_grouped.parquet` uses camelCase column names. The inference CLI needs snake_case. Use `--column-rename` with the following mapping:

```
manufacturerName=manufacturer_name,categoryPaths=category_paths,manufacturerArticleNumber=manufacturer_article_number,manufacturerArticleType=manufacturer_article_type
```

The dataset has no `query_term`, `query_id`, `offer_id_b64`, or `label` columns — that's fine for `--mode offer`. Use `--copy-columns name` (or whichever columns you want in the output) rather than relying on the defaults.

## Measuring throughput

### The benchmark subset

Do **not** run the full 159M-row dataset for every experiment. Use a reproducible subset via `--limit-rows`. Start with 100k; increase if needed (see below).

```bash
LIMIT=100000

uv run embedding-infer \
  --checkpoint checkpoints/useful-cub-58/best-step=4880-val_full_catalog_ndcg_at_5=0.7379.ckpt \
  --input ../../data/offers_grouped.parquet \
  --output /tmp/infer-bench/baseline.parquet \
  --mode offer \
  --embedding-precision float16 \
  --encode-batch-size 128 \
  --read-batch-size 1024 \
  --limit-rows $LIMIT \
  --column-rename 'manufacturerName=manufacturer_name,categoryPaths=category_paths,manufacturerArticleNumber=manufacturer_article_number,manufacturerArticleType=manufacturer_article_type' \
  --copy-columns name \
  --overwrite
```

100k rows should be enough to get a stable rows/sec number (model forward passes dominate; startup is amortized). If you find that variance between runs is too high or that one-time costs (e.g. `torch.compile` warmup) dominate the measurement, increase the subset size (e.g. 500k, 1M) until the throughput number stabilizes. Document which subset size you're using in each `inference-results.tsv` row.

The benchmark metric is:

```
throughput = written_rows / wall_clock_seconds
```

Read `wall_clock_seconds` from the process wall time (wrap with `time` or read the phase-times output). Extrapolate to the full dataset:

```
estimated_full_time = 159_275_274 / throughput
```

### The correctness check

After each change, run the optimized version on the same subset and compare:

```bash
python -c "
import pyarrow.parquet as pq, numpy as np
a = pq.read_table('/tmp/infer-bench/baseline.parquet')
b = pq.read_table('/tmp/infer-bench/optimized.parquet')
emb_a = np.stack(a['offer_embedding'].to_pylist())
emb_b = np.stack(b['offer_embedding'].to_pylist())
print('max_diff:', np.max(np.abs(emb_a.astype(float) - emb_b.astype(float))))
print('rows_match:', len(a) == len(b))
"
```

A `max_diff` of 0 is ideal. Small numerical differences (< 1e-3 in float16 units) from reordered ops or fused kernels are acceptable — flag and document them.

## What you CAN change

- `src/embedding_train/infer.py` — the inference pipeline, batching, I/O, prefetching, async writes.
- `src/embedding_train/infer_parallel.py` — if useful for single-GPU pipelining or process-level tricks.
- `src/embedding_train/model.py` — inference path only (`encode()`, `forward()`, model loading). Do not break training.
- `src/embedding_train/rendering.py` — text rendering, Jinja2 template compilation, column mapping.
- `src/embedding_train/text.py` — text normalization, HTML cleaning.
- `src/embedding_train/precision.py` — quantization / serialization hot path.
- `pyproject.toml` — to add dependencies.
- Any new file under `src/embedding_train/` if needed.

## What you CANNOT change

- The model weights or architecture (no pruning, no distillation, no layer dropping).
- The output format: same Parquet schema, same `offer_embedding` column, same float16 values.
- `src/embedding_train/eval.py`, `metrics.py`, `catalog_benchmark.py` — read-only.
- Training code paths in `model.py` (`training_step`, `validation_step`, etc.) — inference-only changes.

## Logging results

Maintain `inference-results.tsv` (tab-separated) with this schema:

```
commit	rows_per_sec	estimated_full_minutes	bench_rows	status	description
```

1. git commit hash (short, 7 chars)
2. `rows_per_sec` on the benchmark subset (e.g. `12345.6`)
3. `estimated_full_minutes` for 159M rows (e.g. `215.0`)
4. `bench_rows` — subset size used (e.g. `100000`)
5. status: `keep`, `discard`, or `crash`
6. short description of what this experiment tried

Example:

```
commit	rows_per_sec	estimated_full_minutes	bench_rows	status	description
a1b2c3d	3200.0	829.4	100000	keep	baseline (encode_batch_size=128)
b2c3d4e	5800.0	457.6	100000	keep	encode_batch_size=1024 + torch.compile
c3d4e5f	0.0	0.0	0	crash	ONNX export fails on projection head
```

## The experiment loop

The experiment runs on the `speedy-inference` branch.

LOOP FOREVER:

1. Look at the current git state.
2. Make a change (config tweak, code edit, new dependency).
3. `git commit` the change.
4. Run the benchmark subset and record wall time + rows/sec.
5. Run the correctness check against the baseline output (re-generate the baseline at the new subset size if you increased it).
6. If `rows_per_sec` improved and correctness passes: **keep** the commit.
7. If it's equal, worse, or correctness fails: `git reset --hard` to the previous keep point.
8. Log the result in `inference-results.tsv` (do NOT commit this file).
9. Repeat.

**The first run** is always the unoptimized baseline — run `embedding-infer` as-is with default batch sizes to establish the starting throughput.

**Crashes**: fix if trivial, skip if not. Log as `crash`.

**Correctness failures**: always discard. No exceptions.

**NEVER STOP**: continue indefinitely until manually interrupted. If you run out of ideas, re-read the source files, profile the pipeline, look for new bottlenecks. The loop runs until the human stops you.
