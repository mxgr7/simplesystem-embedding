# autoresearch (cross-encoder edition)

This is an experiment to have the LLM do its own research on the cross-encoder fine-tuning project in this repo (`src/cross_encoder_train/`). It mirrors the embedding-train autoresearch program (`program.md`), but adapted to the pair-classification / pair-ranking setup.

The model is a HuggingFace `AutoModel` encoder (default `deepset/gelectra-base`) with a `[CLS]`-pooled linear head over 4 relevance classes (`Irrelevant`, `Complement`, `Substitute`, `Exact`). Validation derives a per-pair score from the softmax probabilities and the `GAIN_VECTOR`, then computes catalog-style ranking metrics on it. The success metric is `val/rank/ndcg_at_5` (higher is better).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr28-ce`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Read the in-scope files**. Read for full context:
   - `README.md` — repository context and CLIs.
   - `configs/cross_encoder.yaml` and the per-group files `configs/model/cross_encoder.yaml`, `configs/data/cross_encoder.yaml`, `configs/trainer/cross_encoder.yaml`, `configs/optimizer/adamw.yaml`, `configs/logger/mlflow.yaml` — all tunable configuration.
   - `src/cross_encoder_train/train.py` — Hydra entry point and Lightning Trainer wiring.
   - `src/cross_encoder_train/model.py` — encoder, classification head, loss, validation row construction.
   - `src/cross_encoder_train/data.py` — datamodule, tokenizer, query-id-based train/val split, class weights.
   - `src/embedding_train/rendering.py` — `RowTextRenderer` used to materialize `query_text`/`offer_text` from the parquet via Jinja templates in `configs/data/cross_encoder.yaml`. **You can edit this**, but it is shared with embedding-train, so behave carefully.
   - **Read-only** (the eval harness; do not modify):
     - `src/cross_encoder_train/metrics.py` — classification metrics.
     - `src/cross_encoder_train/labels.py` — `LABEL_ORDER`, `GAIN_VECTOR`, `NUM_CLASSES`. These tie directly into the eval harness.
     - `src/embedding_train/metrics.py` — `compute_ranking_metrics` and `compute_binary_retrieval_metrics` are the ground-truth metric implementations.
4. **Verify data exists**: confirm the labeled parquet referenced by `configs/data/cross_encoder.yaml` (`path: ../../data/queries_offers_labeled.parquet`) is readable, and that the MLflow server at the URI in `configs/logger/mlflow.yaml` (`http://127.0.0.1:5001`) is reachable (`curl -s <uri>`). If either is missing, stop and tell the human.
5. **Verify GPU**: this host has a CUDA driver / cuda-13.0 compat-layer mismatch. Every invocation that touches `torch.cuda` MUST be prepended with `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`. One-line check:
   ```bash
   LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
     uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Should print `True NVIDIA H100 80GB HBM3`.
6. **Initialize results.tsv**: create `results.tsv` (project root) with just the header row (if it doesn't exist yet). The baseline will be recorded after the first run. **Note**: the embedding-train autoresearch uses the same filename. If you're running both projects on the same checkout, name the cross-encoder file `results-cross-encoder.tsv` instead and use that throughout this document.
7. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed wall-clock budget** set via `trainer.max_time`. The default is **20 minutes** (`00:00:20:00`). One full pass over the 204k-row dataset at the default `batch_size=32` takes ~6 min on the H100, so a 20-min budget yields ~3 full epochs and 3 val checkpoints (with `val_check_interval=1.0`). If you want denser val signal, set `trainer.val_check_interval=0.5` (val twice per epoch) — but each val pass costs ~30s of training time. Launch:

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
  uv run cross-encoder-train trainer.max_time=00:00:20:00 trainer.max_epochs=1000 > run.log 2>&1
```

(The high `max_epochs` is there so the time budget is the binding constraint.)

**What you CAN do:**
- Edit anything under `configs/` for the cross-encoder group — base model, head dropout, label smoothing, use_class_weights, gradient checkpointing, optimizer LR / weight decay / scheduler / warmup, batch size, max_pair_length, val fraction, query/offer Jinja templates, num_workers.
- Edit `src/cross_encoder_train/train.py`, `model.py`, `data.py` — model architecture, pooling (try mean/max/attention pool instead of `[CLS]`), loss (focal loss, ordinal regression head, multi-task auxiliary heads), batching, training loop, augmentations.
- Edit `src/embedding_train/rendering.py` if you want to change how query/offer text is built (e.g. add HTML stripping, different category-path rendering, special tokens) — but remember it's shared with embedding-train.
- Try different base encoders: e.g. `deepset/gelectra-large`, `xlm-roberta-base`, `microsoft/mdeberta-v3-base`, German-specific models. Keep VRAM reasonable (see below).
- Change how the per-pair ranking score is derived from logits inside `validation_step` (e.g. logit of the `Exact` class, temperature-scaled softmax, learned scoring layer). The score function is fair game; what gets done with the score downstream is not.

**What you CANNOT do:**
- Modify `src/cross_encoder_train/metrics.py`, `src/cross_encoder_train/labels.py`, or `src/embedding_train/metrics.py`. These define the ground-truth metric and the label/gain mapping.
- Change `LABEL_ORDER`, `GAIN_VECTOR`, `NUM_CLASSES`, or remap raw labels.
- Change `trainer.monitor_metric` (must stay `val/rank/ndcg_at_5`) or `trainer.monitor_mode` (must stay `max`).
- Change the dataset path to a different parquet, or change the `column_mapping` keys, or alter how validation queries are selected (`val_fraction`, the query-id-based split in `data.py`). You can change `seed` but realize it perturbs the val split.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.

**The goal is simple: get the highest `val/rank/ndcg_at_5`** on this dataset. Since the time budget is fixed, don't worry about training time — every run is bounded by `trainer.max_time`. The only constraint is that the code runs without crashing and finishes within the budget.

**Headroom note**: The baseline (single epoch, default config) reaches `val/rank/ndcg_at_5 ≈ 0.978`, so there's not much ceiling left and the noise floor is unknown. Treat improvements <0.002 with skepticism — replicate with a different seed before committing as a `keep`. Big swings are more likely to come from architectural changes (different base model, different loss, different prompt template) than from LR/optimizer micro-tuning.

**VRAM** is a soft constraint. Baseline uses ~7.5 GB on the H100 with `bf16-mixed` and batch 32 — there's lots of headroom. Some increase is acceptable for meaningful gains, but it should not blow up dramatically. If you need more compute, prefer larger models or larger `max_pair_length` over absurdly large batch sizes that won't generalize.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. A ~0.001 improvement from deleting code is worth keeping; the same improvement from 20 lines of hacks is not.

**The first run**: Your very first run should always be to establish the baseline, so run the training script as-is (no config overrides except `trainer.max_time` and `trainer.max_epochs` as above).

## Data-side knobs (no negative-mining loop here)

Unlike embedding-train, there is no hard/semi-hard negative mining workflow for the cross-encoder — it consumes pre-labeled pairs. But the data side still has real levers:

- **Prompt templates** (`data.query_template`, `data.offer_template` in `configs/data/cross_encoder.yaml`): which fields to include, ordering, separators, special tokens. The current offer template includes name, EAN, article numbers, category, manufacturer, description.
- **`max_pair_length`**: default 384. Longer pairs preserve more description text but cost FLOPs quadratically; shorter pairs train faster and might reduce noise from boilerplate. Worth sweeping (e.g. 256, 384, 512).
- **Class balancing**: `model.use_class_weights` is on by default with weights derived from inverse frequency. Try off (let the loss focus on the majority `Exact` class), or try focal loss / label smoothing alternatives.
- **`limit_rows`**: only useful for fast-iteration debugging, not for real runs.

If you want to experiment with custom data preprocessing (e.g. cleaned descriptions, deduplicated pairs, query normalization), do it in `data.py:_prepare_records` or `rendering.py`.

## Reading results

Training logs to MLflow at the URI in `configs/logger/mlflow.yaml` (`http://127.0.0.1:5001`, experiment `simplesystem-embedding` by default — feel free to set `logger.experiment_name=cross-encoder-autoresearch` for clarity).

After a run finishes, pull the final metric:

```bash
grep -E "val/rank/ndcg_at_5|val/loss|peak" run.log | tail -20
```

Or via the MLflow REST API:

```bash
EXP_ID=$(curl -sS "http://127.0.0.1:5001/api/2.0/mlflow/experiments/get-by-name?experiment_name=cross-encoder-autoresearch" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['experiment']['experiment_id'])")
curl -sS -X POST "http://127.0.0.1:5001/api/2.0/mlflow/runs/search" \
  -H "Content-Type: application/json" \
  -d "{\"experiment_ids\":[\"$EXP_ID\"],\"max_results\":1}" \
  | python3 -m json.tool
```

For peak VRAM, grep `nvidia-smi` lines you logged yourself, or read it back from `torch.cuda.max_memory_allocated()` if you instrument the trainer.

If `grep` output is empty or the run obviously crashed, run `tail -n 80 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after a few attempts, give up on that idea.

## Logging results

When an experiment is done, log it to `results.tsv` (or `results-cross-encoder.tsv` if you're sharing the checkout with embedding-train autoresearch — see Setup step 6).

The TSV has a header row and 5 columns:

```
commit	val_ndcg_at_5	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `val/rank/ndcg_at_5` achieved (e.g. `0.978100`) — use `0.000000` for crashes
3. peak VRAM in GB, `.1f` (e.g. `7.5`) — use `0.0` for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried

Example:

```
commit	val_ndcg_at_5	memory_gb	status	description
a1b2c3d	0.977900	7.5	keep	baseline (gelectra-base, bf16-mixed, batch 32, 20-min budget)
b2c3d4e	0.979100	7.6	keep	raise LR to 3e-5 and warmup_ratio 0.1
c3d4e5f	0.976200	7.5	discard	switch loss to focal gamma=2
d4e5f6g	0.000000	0.0	crash	gelectra-large with batch 32 and gradient_checkpointing=false (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr28-ce` or `autoresearch/apr28-ce-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune configs and/or source files with an experimental idea by directly hacking the code.
3. `git commit` the change.
4. Run:
   ```bash
   LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
     uv run cross-encoder-train trainer.max_time=00:00:20:00 trainer.max_epochs=1000 > run.log 2>&1
   ```
5. Read out the results via MLflow and/or `grep` on `run.log`.
6. If the grep output is empty / the MLflow run is missing, the run crashed. Read `tail -n 80 run.log` and attempt a fix. If you can't, skip it.
7. Record the row in `results.tsv` (NOTE: do not commit your changes to `results.tsv`, that's up to the human to review & commit).
8. Add and/or update any notable insights in `NOTES.md` (also do not commit your changes to `NOTES.md`).
9. If `val/rank/ndcg_at_5` improved (higher), "advance" the branch — keep the commit.
10. If it's equal or worse, `git reset --hard` back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. You're advancing the branch so you can iterate. You can rewind further if you feel stuck, but do this very sparingly.

**Timeout**: Each experiment should take ~`trainer.max_time` plus a minute or so for startup, validation, and teardown. If a run exceeds `2 × max_time + 3min` (i.e. ~43 min for a 20-min budget), kill it and treat it as a failure.

**Crashes**: Use judgment. If it's something dumb and easy to fix (typo, missing import, OOM that's solvable by halving batch size), fix it and re-run. If the idea is fundamentally broken, log `crash` in the TSV and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. The human might be asleep and expects you to continue *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, read the existing `NOTES.md` and `reports/` for prior findings, try combining previous near-misses, try more radical changes (different base model, different loss, different pooling, prompt template overhaul, label smoothing, layer freezing / unfreezing schedules, head architecture, two-stage fine-tuning, contrastive auxiliary objective on top of CE). The loop runs until the human interrupts you, period.
