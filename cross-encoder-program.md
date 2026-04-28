# autoresearch (cross-encoder edition)

This is an experiment to have the LLM do its own research on the cross-encoder fine-tuning project in this repo (`src/cross_encoder_train/`).

The model is a HuggingFace `AutoModel` encoder (default `deepset/gelectra-base`) with a `[CLS]`-pooled linear head over 4 relevance classes (`Irrelevant`, `Complement`, `Substitute`, `Exact`). Validation logs both classification metrics (accuracy / per-class F1 / macro-F1) and ranking metrics (ndcg@k, MRR of Exact).

**Success metrics are `val/cls/micro_f1` AND `val/cls/macro_f1`** (higher is better on both). Micro-F1 (= accuracy in the single-label multi-class case) is the primary metric — it matches Wu et al. (2022) Task 2 on Amazon ESCI, which uses the same 4-label scheme. Macro-F1 is co-equal: it's there because the dataset is 80% `Exact`, and a model can buy +0.01 micro-F1 by squeezing the majority class while doing nothing useful for `Substitute` / `Complement`. Tracking macro-F1 forces minority-class quality.

**Keep/discard rule with two metrics**: a run is `keep` if **at least one of (micro_f1, macro_f1) strictly improves and the other does not regress beyond the noise floor (~0.002)**. If one metric improves and the other regresses meaningfully, it's `discard`. If both improve, obviously `keep`. The Lightning `ModelCheckpoint` still selects on `val/cls/micro_f1` (mechanical — one number must win) but the keep decision in `results.tsv` looks at both.

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
- Change how the per-pair ranking score is derived from logits inside `validation_step` — but note this only affects the secondary ranking metrics, not the success metric (`val/cls/micro_f1`), which depends solely on `argmax`-based predictions.
- **Explore the data.** Open the labeled parquet, slice it, look at label distributions per query, look at character/token length distributions, sample concrete (query, offer, label) rows, look for duplicates, suspicious labels, locale-specific quirks, etc. This kind of exploration is **encouraged** — every well-grounded experiment idea starts with knowing what's actually in the data. Exploration time does **not** count towards the per-run training budget. Persist anything notable (distributions, surprising findings, hypotheses worth testing) in `data-insights.md` at the project root, so future iterations can build on prior observations instead of rediscovering them.

**What you CANNOT do:**
- Modify `src/cross_encoder_train/metrics.py`, `src/cross_encoder_train/labels.py`, or `src/embedding_train/metrics.py`. These define the ground-truth metric and the label/gain mapping.
- Change `LABEL_ORDER`, `GAIN_VECTOR`, `NUM_CLASSES`, or remap raw labels.
- Change `trainer.monitor_metric` (must stay `val/cls/micro_f1`) or `trainer.monitor_mode` (must stay `max`).
- Change the dataset path to a different parquet, or change the `column_mapping` keys, or alter how validation queries are selected (`val_fraction`, the query-id-based split in `data.py`). You can change `seed` but realize it perturbs the val split.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.

**The goal is simple: push both `val/cls/micro_f1` and `val/cls/macro_f1` higher** on this dataset. Since the time budget is fixed, don't worry about training time — every run is bounded by `trainer.max_time`. The only constraint is that the code runs without crashing and finishes within the budget.

**Headroom note**: Baseline (single epoch, default config) is `micro_f1 ≈ 0.853, macro_f1 ≈ 0.654`. Per-class F1: exact 0.93, complement 0.65, irrelevant 0.60, substitute 0.44.

- **Micro-F1 has tight headroom.** The trivial "always Exact" baseline gets micro_f1 ≈ 0.825 on this val split, so we're only 2.8 points above the dumbest possible model. Expect total micro-F1 headroom around +0.05 (target ~0.90). For reference Wu et al. (2022) reached 0.831 on Amazon ESCI Task 2 with ensembles of large transformers — but that's a harder dataset (multilingual, only 44% Exact), so we should plausibly beat them with a single well-tuned model.
- **Macro-F1 has much more headroom.** Substitute and Complement are where the real gains live: F1(substitute)=0.44 today. Lifting macro to 0.75+ is plausible and unlocks product value that micro-F1 hides.
- Treat improvements <0.002 on either metric with skepticism — replicate with a different seed before committing as a `keep`.

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
grep -E "val/cls/micro_f1|val/cls/macro_f1|val/loss|peak" run.log | tail -20
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

The TSV has a header row and 6 columns:

```
commit	val_micro_f1	val_macro_f1	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `val/cls/micro_f1` achieved (e.g. `0.853000`) — use `0.000000` for crashes
3. `val/cls/macro_f1` achieved (e.g. `0.654000`) — use `0.000000` for crashes
4. peak VRAM in GB, `.1f` (e.g. `7.5`) — use `0.0` for crashes
5. status: `keep`, `discard`, or `crash`
6. short description of what this experiment tried

Example:

```
commit	val_micro_f1	val_macro_f1	memory_gb	status	description
a1b2c3d	0.853000	0.654000	7.5	keep	baseline (gelectra-base, bf16-mixed, batch 32, 20-min budget)
b2c3d4e	0.861400	0.671200	7.6	keep	raise LR to 3e-5 and warmup_ratio 0.1
c3d4e5f	0.857200	0.638100	7.5	discard	+0.004 micro but -0.016 macro: minority classes regressed
e5f6g7h	0.851000	0.689000	7.5	keep	+0.035 macro at -0.002 micro (within noise): focal loss gamma=2
d4e5f6g	0.000000	0.000000	0.0	crash	gelectra-large with batch 32 and gradient_checkpointing=false (OOM)
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
9. Apply the keep/discard rule (see "Success metrics" in the intro): if at least one of `val/cls/micro_f1` and `val/cls/macro_f1` strictly improves and the other is within the noise floor (±0.002), "advance" the branch — keep the commit.
10. Otherwise, `git reset --hard` back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. You're advancing the branch so you can iterate. You can rewind further if you feel stuck, but do this very sparingly.

**Timeout**: Each experiment should take ~`trainer.max_time` plus a minute or so for startup, validation, and teardown. If a run exceeds `2 × max_time + 3min` (i.e. ~43 min for a 20-min budget), kill it and treat it as a failure.

**Crashes**: Use judgment. If it's something dumb and easy to fix (typo, missing import, OOM that's solvable by halving batch size), fix it and re-run. If the idea is fundamentally broken, log `crash` in the TSV and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. The human might be asleep and expects you to continue *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, read the existing `NOTES.md` and `reports/` for prior findings, try combining previous near-misses, try more radical changes (different base model, different loss, different pooling, prompt template overhaul, label smoothing, layer freezing / unfreezing schedules, head architecture, two-stage fine-tuning, contrastive auxiliary objective on top of CE). The loop runs until the human interrupts you, period.
