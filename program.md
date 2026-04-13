# autoresearch (embedding-train edition)

This is an experiment to have the LLM do its own research on the embedding fine-tuning project in this repo.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr13`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Read the in-scope files**. The training code lives in a Hydra + Lightning package. Read for full context:
   - `README.md` — repository context and CLIs.
   - `configs/config.yaml` and the files under `configs/model/`, `configs/data/`, `configs/trainer/`, `configs/optimizer/`, `configs/logger/` — all tunable configuration.
   - `src/embedding_train/train.py` — Lightning entry point and training loop.
   - `src/embedding_train/model.py` — encoder, pooling, projection head.
   - `src/embedding_train/losses.py` — contrastive / triplet losses.
   - `src/embedding_train/batching.py` — batch construction, negative mixing.
   - `src/embedding_train/data.py` — datamodule, hard/semi-hard negative wiring.
   - `src/embedding_train/eval.py` and `metrics.py` — **read-only**: this is the evaluation harness and must not be modified.
4. **Verify data exists**: confirm the labeled parquet referenced by `configs/data/default.yaml` is readable, and that the MLflow server at the URI in `configs/logger/mlflow.yaml` is reachable (`curl -s <uri>` or use the `mlflow` skill). If either is missing, stop and tell the human.
5. **Initialize results.tsv**: create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed wall-clock budget** set via `trainer.max_time`. The default is **20 minutes** (`00:00:20:00`), chosen from the existing MLflow run catalog: at 20 min you get ~4–5 full-catalog val checkpoints, the final val fires ~16 min in, and cross-run deltas at this budget are ~0.01–0.02 `ndcg@5` vs a ~±0.003 noise floor. That's a clean signal-to-noise ratio for keep/discard decisions. Launch it as:

```bash
uv run embedding-train trainer.max_time=00:00:20:00 trainer.max_epochs=1000
```

(The high `max_epochs` is there so the time budget is the binding constraint, not the epoch count.)

**What you CAN do:**
- Edit anything under `configs/` — model choice, loss, optimizer, LR schedule, batch size, pooling, projection dim, gradient checkpointing, etc.
- Edit `src/embedding_train/train.py`, `model.py`, `losses.py`, `batching.py`, `data.py` — architecture, training loop, negative mining, loss math.
- Add or rewire existing hard/semi-hard negative files via `data.hard_negatives_path` / `data.semi_hard_negatives_path`.
- Run the supporting CLIs documented in `README.md` to generate new training inputs:
  - `uv run embedding-index-build --checkpoint <ckpt> --input <offers.parquet> --output ../../data/offer-index-<run-name>`
  - `uv run embedding-mine-hard-negatives --checkpoint <ckpt> --index ../../data/offer-index-<run-name> --input ../../data/queries_offers_labeled.parquet --output ../../data/hard_negatives-<run-name>.parquet`
  - Same CLI with `--rank-start/--rank-end/--provenance semi_hard_negative` for semi-hard mining.
  - You may also use `embedding-infer` for sanity-checking a checkpoint's embeddings.

**What you CANNOT do:**
- Modify `src/embedding_train/eval.py`, `metrics.py`, `catalog_benchmark.py`, or `faiss_index.py`. These define the ground-truth metric.
- Modify `configs/trainer/default.yaml` fields that control evaluation: `validation_mode`, `validation_metric`, `validation_similarity`, `validation_relevant_labels`, `validation_catalog_sample`. You may override other trainer fields freely.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Change the evaluation parquet path or label mapping.
- Run `embedding-eval`, `embedding-catalog-benchmark`, or `embedding-index-search` as part of the keep/discard decision. These are useful for investigation but are NOT the ground-truth signal — MLflow `val/full_catalog/ndcg_at_5` from the training run is.

**The goal is simple: get the highest `val_ndcg_at_5`** (note: higher is better, opposite direction from the original bpb setup). Since the time budget is fixed, you don't need to worry about training time — every run is bounded by `trainer.max_time`. The only constraint is that the code runs without crashing and finishes within the budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. A ~0.001 improvement from deleting code is worth keeping; the same improvement from 20 lines of hacks is not.

**The first run**: Your very first run should always be to establish the baseline, so run the training script as-is (no config overrides except `trainer.max_time` and `trainer.max_epochs` as above).

## Mining new negatives (occasional, not every iteration)

Once you have a decent checkpoint, you can build an index from it and mine a fresh pool of hard or semi-hard negatives for subsequent training runs. `NOTES.md` documents this as the "Re-mining iteration" strategy and explains why it's high-leverage: the current SOTA was trained against a pool mined from a much weaker encoder, so the pool is almost certainly stale.

**When to do this**: when you've plateaued on the current negative pool (say, 3–4 consecutive runs without improvement) and haven't re-mined in a while. Not every iteration — each mining cycle costs extra wall-clock on top of a normal training run and makes the comparison to previous runs less clean.

**Workflow**:

1. Pick the best checkpoint so far (`checkpoints/<run-name>/best-step=...ckpt`).
2. Build an offer index from it:
   ```bash
   uv run embedding-index-build \
     --checkpoint checkpoints/<run-name>/best-...ckpt \
     --input ../../data/queries_offers_labeled.parquet \
     --output ../../data/offer-index-<run-name>
   ```
3. Mine hard or semi-hard negatives against that index:
   ```bash
   uv run embedding-mine-hard-negatives \
     --checkpoint checkpoints/<run-name>/best-...ckpt \
     --index ../../data/offer-index-<run-name> \
     --input ../../data/queries_offers_labeled.parquet \
     --output ../../data/semi_hard_negatives-<run-name>.parquet \
     --top-k 100 --rank-start 20 --rank-end 60 \
     --max-negatives-per-query 10 \
     --provenance semi_hard_negative
   ```
4. Point the next training run at it:
   ```bash
   uv run embedding-train \
     trainer.max_time=00:00:20:00 trainer.max_epochs=1000 \
     data.semi_hard_negatives_path=../../data/semi_hard_negatives-<run-name>.parquet
   ```
5. In `results.tsv`, mention the pool in the description (e.g. `contrastive + semi-hard pool mined from <run-name>`) so you can tell apart runs trained against different pools.

**Do not** commit mined parquets or built indexes to git. They go under `../../data/` (same tree as the labeled parquet) and stay untracked. A timing-out or crashed mine does not count as an experiment — just retry or skip it.

## Reading results

Training logs to MLflow at the URI in `configs/logger/mlflow.yaml`. After a run finishes, pull the final metric and peak VRAM via the `mlflow` skill — search the `simplesystem-embedding` experiment for the most recent run and read:

- `val_ndcg_at_5` (or whatever the configured `validation_metric` is)
- `peak_vram_mb` if logged; otherwise read it from the run log

The run command should redirect stdout and stderr to a log file so it doesn't flood your context:

```bash
uv run embedding-train trainer.max_time=00:00:20:00 trainer.max_epochs=1000 > run.log 2>&1
```

Key lines can then be grepped:

```bash
grep -E "val_ndcg_at_5|peak" run.log | tail -20
```

If the grep output is empty or the run obviously crashed, run `tail -n 80 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after a few attempts, give up on that idea.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_ndcg_at_5	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `val_ndcg_at_5` achieved (e.g. `0.612340`) — use `0.000000` for crashes
3. peak VRAM in GB, `.1f` (e.g. `12.3`) — use `0.0` for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried

Example:

```
commit	val_ndcg_at_5	memory_gb	status	description
a1b2c3d	0.612340	14.2	keep	baseline
b2c3d4e	0.618900	14.4	keep	raise LR to 3e-5 and warmup 200 steps
c3d4e5f	0.609100	14.2	discard	switch loss_type to triplet margin=0.3
d4e5f6g	0.000000	0.0	crash	output_dim=1024 with gradient_checkpointing=false (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr13` or `autoresearch/apr13-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune configs and/or source files with an experimental idea by directly hacking the code.
3. `git commit` the change.
4. Run: `uv run embedding-train trainer.max_time=00:00:20:00 trainer.max_epochs=1000 > run.log 2>&1`.
5. Read out the results via MLflow and/or `grep` on `run.log`.
6. If the grep output is empty / the MLflow run is missing, the run crashed. Read `tail -n 80 run.log` and attempt a fix. If you can't, skip it.
7. Record the row in `results.tsv` (NOTE: do not commit `results.tsv`, leave it untracked).
8. If `val_ndcg_at_5` improved (higher), "advance" the branch — keep the commit.
9. If it's equal or worse, `git reset --hard` back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. You're advancing the branch so you can iterate. You can rewind further if you feel stuck, but do this very sparingly.

**Timeout**: Each experiment should take ~`trainer.max_time` plus a minute or so for startup, validation, and teardown. If a run exceeds `2 × max_time + 3min`, kill it and treat it as a failure.

**Crashes**: Use judgment. If it's something dumb and easy to fix (typo, missing import), fix it and re-run. If the idea is fundamentally broken, log `crash` in the TSV and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. The human might be asleep and expects you to continue *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, read the existing `NOTES.md` and `reports/` for prior findings, try combining previous near-misses, try more radical changes (different base model, different loss, different pooling, projection head sizes, harder negatives, learning rate schedules, weight decay, dropout, layer freezing, contrastive temperature). The loop runs until the human interrupts you, period.
