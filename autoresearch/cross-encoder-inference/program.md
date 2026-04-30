# autoresearch (cross-encoder inference edition)

This is an experiment to have the LLM autonomously squeeze inference latency and hosting cost out of the production cross-encoder reranker (`src/cross_encoder_serve/`) without sacrificing quality below a hard floor.

The starting model is the released **Soup CE** at `../../../checkpoints/cross-encoder/releases/v1.0-2026-04-29/soup.ckpt` — a uniform weight average of 3 `deepset/gelectra-large` (334.7M params) checkpoints with a `Linear(1024, 4)` head over `(Irrelevant, Complement, Substitute, Exact)`. Today this model is served via `cross_encoder_serve.server:app` (FastAPI + Lightning + HF tokenizer + bf16 autocast on CUDA, eager mode). The training-side context lives in `../cross-encoder/program.md` and `../cross-encoder/NOTES.md`; read those for background but do **not** modify the training program's `results.tsv` or `NOTES.md`.

## The goal — exactly one number, one floor

**Hardware: NVIDIA RTX 4090 24 GB (production target).** Optimizing for any other card is out of scope.

**Primary target — latency:** the **end-to-end `POST /rerank` p95 latency**, **warm**, for a request with **1 query × 2000 offers, every offer padded/truncated to `max_pair_length = 512` tokens (worst case)**, must be **< 1000 ms**. End-to-end means request-in → response-out: tokenization, GPU forward, calibration, all of it. Steady-state, not cold start.

**Quality floor — must not be violated:** on the original training val split (`configs/data/cross_encoder.yaml::path` → `data/queries_offers_esci/queries_offers_merged_labeled.parquet`, query-id-based split with the configured `val_fraction` and `seed`), CE-alone (no LGBM stack), the metric outputs from `src/cross_encoder_train/metrics.py::compute_classification_metrics` must satisfy:

- `val/cls/micro_f1 ≥ 0.890`
- `val/cls/macro_f1 ≥ 0.770`

Strictly. A run that lands at 0.8899 micro is a floor violation.

**API contract:** only `p_exact_calibrated` (the temperature-scaled `softmax(logits/T)[Exact]`) matters to downstream. The other 3 class probs and `predicted_label` exist in the response schema for compatibility, but downstream consumers only look at `p_exact_calibrated`. You may drop or zero-fill the others if it buys speed; you may also drop the LGBM stack entirely (it is gone from the success criterion).

**Hosting cost** is implicit in the latency target on a 4090 — once you fit the request inside 1000 ms p95 on a 4090, hosting cost per request is set by 4090 hourly cost / requests-per-second. So lower latency = lower cost. There is no separate cost metric.

## Reality check — where the baseline likely sits

The release manifest reports H100 latency: `n=2000_offers_median_ms: 681`. The 4090 has roughly half the FP16 TFLOPS of an H100 and roughly 1/3 the memory bandwidth, so expect baseline p95 on a 4090 in the 1300–2000 ms range — i.e. the goal is **not free**, but compilation + lower precision + dropping unnecessary work might be enough on their own. Distillation is the bigger lever if those don't suffice.

## Setup

To set up:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr30-ce-infer`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `src/cross_encoder_serve/server.py`, `src/cross_encoder_serve/inference.py` — the actual production serving path. **This is the primary edit zone.**
   - `src/cross_encoder_train/model.py`, `train.py`, `data.py` — needed for distillation (you will likely add a `DistillationModule` and a CLI that mirrors `cross-encoder-train`).
   - `configs/cross_encoder.yaml` and the per-group files under `configs/{model,data,trainer,optimizer,logger}/cross_encoder.yaml`.
   - `configs/model/distill.yaml` exists from the embedding-train side (different model, different loss) — useful as inspiration for distillation config layout, not for direct reuse.
   - `pyproject.toml` — current dep set; you can add new ones (see "What you CAN do").
   - `../cross-encoder/program.md`, `../cross-encoder/NOTES.md`, `../cross-encoder/data-insights.md` — background on the training program and the data. Read but do not modify.
   - **Read-only** (the eval harness; do not modify):
     - `src/cross_encoder_train/metrics.py` — `compute_classification_metrics`. The ground-truth quality measurement.
     - `src/cross_encoder_train/labels.py` — `LABEL_ORDER`, `NUM_CLASSES`. Tied to the eval harness.
4. **Verify the teacher checkpoint** exists at `../../../checkpoints/cross-encoder/releases/v1.0-2026-04-29/soup.ckpt` and loads via `CrossEncoderModule.load_from_checkpoint(...)` (see how `Reranker.__init__` does it in `inference.py`).
5. **Verify data exists**: confirm the labeled parquet referenced by `configs/data/cross_encoder.yaml` is readable. If missing, stop and tell the human.
6. **Verify GPU**: this host has the same CUDA compat-layer mismatch as the training program. Every invocation that touches `torch.cuda` MUST be prepended with `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`. One-line check:
   ```bash
   LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
     uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Should print `True NVIDIA GeForce RTX 4090`.
7. **Verify MLflow** (optional but recommended for distillation runs): `curl -s http://127.0.0.1:5001` should respond. Use experiment name `cross-encoder-inference` for clarity.
8. **Artifacts already initialized** in this directory (`autoresearch/cross-encoder-inference/`):
   - `program.md` — this file.
   - `results.tsv` — per-experiment log; header only at start. See "Logging results".
   - `NOTES.md` — accumulated insights. Newest at the top.
   - `bench_rerank.py` — the latency benchmark harness. See "Benchmark methodology".
   - `eval_val.py` — the quality eval harness. See "Quality eval methodology".
   - `fixture_2000x512.json` — the frozen benchmark fixture (created on first `bench_rerank.py` run; do not regenerate without re-anchoring).
9. **Confirm and go**: confirm setup looks good, then start.

## Benchmark methodology — `bench_rerank.py`

The latency benchmark is `bench_rerank.py` (in this directory). It is the **canonical latency number** — every keep/discard decision goes through it. Do not optimize against a proxy (e.g. raw GPU forward time) without also running the end-to-end benchmark.

What it does:

1. Loads or builds the fixture: 1 query + 2000 offer dicts sampled from the labeled parquet (deterministic seed=0), with each offer's `description` padded so that `tokenizer(query, offer_text, truncation="only_second", max_length=512)` returns exactly 512 tokens. Saved to `fixture_2000x512.json` — **frozen across runs**.
2. Spawns `cross_encoder_serve.server:app` as a subprocess (passes `CKPT`, `SERVE_DTYPE`, `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:...`; explicitly unsets `LGBM`).
3. Polls `GET /health` until `status: ok` (240 s timeout to absorb cold model load).
4. Sends 2 warmup `POST /rerank` requests (discarded), then 10 measurement requests sequentially. Single client — the goal is "1 batch in < 1000 ms", not concurrent throughput.
5. Polls `nvidia-smi --query-gpu=memory.used` at 100 ms cadence during the measurement phase, takes the max → `peak_vram_gb`.
6. Prints machine-greppable lines: `p50_ms=`, `p95_ms=`, `p99_ms=`, `peak_vram_gb=`, plus health metadata (`autocast_dtype`, `attn_implementation`).
7. Tears the server down.

Run:
```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
  uv run --extra train python autoresearch/cross-encoder-inference/bench_rerank.py \
  --ckpt ../checkpoints/cross-encoder/releases/v1.0-2026-04-29/soup.ckpt \
  --serve-dtype bf16 \
  > bench.log 2>&1
grep -E "^(p50_ms|p95_ms|p99_ms|peak_vram_gb|health\\.|n_warmup)" bench.log
```

**Don't change the fixture** unless you have a strong reason. If you do (e.g. different padding strategy or different `max_pair_length` semantics), every prior `results.tsv` row is invalidated — re-run the previous branch tip to re-anchor.

**Tighten the measurement as latency drops.** Defaults (`--n-warmup 2 --n-measure 10`) are tuned for the slow early-baseline regime where each request is ~9 s and total bench cost is ~2.5 min. With only 10 measurements, p95 is essentially `max(samples)` — fine when the run-to-run jitter is small relative to the gap you're trying to detect. As latency falls and per-request cost shrinks (e.g. 1 s/req at the goal), bump to `--n-warmup 5 --n-measure 30` (~30 s total) — at that point a tighter p95 estimate is cheap and lets you distinguish smaller wins. Once near the goal, even `--n-measure 100` (~100 s) is reasonable for the final confirmation run.

## Quality eval methodology — `eval_val.py`

The quality eval is `eval_val.py` (in this directory). It is the **canonical quality measurement** — both F1s in `results.tsv` come from it. Floor violation = `discard`, no exception.

What it does:

1. Loads `configs/cross_encoder.yaml` via Hydra, overrides `data.path` to the absolute path of the labeled parquet (default: `../../data/queries_offers_esci/queries_offers_merged_labeled.parquet` resolved from the repo root).
2. Instantiates `CrossEncoderDataModule(cfg).setup("fit")` so the val split is **bit-identical** to the one `cross-encoder-train` uses (same `val_fraction`, same `seed`, same query-id-based split).
3. Loads a Lightning checkpoint via `CrossEncoderModule(cfg=cfg).load_state_dict(...)`, mirroring the production loader (`Reranker.__init__` in `src/cross_encoder_serve/inference.py`): forces `model.compile=false`, strips the `_orig_mod.` prefix.
4. Runs `model(inputs)` over the val dataloader under `torch.autocast(device_type="cuda", dtype=...)` (default: bf16), collects argmax predictions.
5. Calls `compute_classification_metrics` from the **read-only** `src/cross_encoder_train/metrics.py` — never reimplement the metric.
6. Prints machine-greppable lines: `val/cls/micro_f1=`, `val/cls/macro_f1=`, `val/cls/f1_{irrelevant,complement,substitute,exact}=`, `val/cls/evaluated_pairs=`.

Run (default: 10k random-uniform subset, deterministic seed=0):
```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
  uv run --extra train python autoresearch/cross-encoder-inference/eval_val.py \
  --ckpt ../checkpoints/cross-encoder/releases/v1.0-2026-04-29/soup.ckpt \
  > eval.log 2>&1
grep "^val/cls/" eval.log
```

**Subset vs full val**: by default `eval_val.py` runs on a 10k uniform-random subset of the 76,048-row val split (deterministic, seed=0). Iteration time on a 4090: ~3-4 min. Estimated noise vs full val: micro_f1 ±0.005, macro_f1 ±0.013 (driven by Complement, the rarest class at 1.8% of val). This is fine for iteration since the floors (0.890 / 0.770) sit far enough from a healthy model that 0.013 noise won't flip a keep/discard decision in the typical case. **Before declaring the latency goal hit and stopping the loop, re-run with `--subset-rows 0` (full val, ~25 min) to confirm the floors hold on the canonical split.**

**Only Lightning `.ckpt` is supported out of the box.** If you produce a non-Lightning artifact (ONNX, TensorRT, GPTQ-quantized state dict, distilled student saved with a different module), extend the `load_model` function in `eval_val.py` rather than writing a parallel script — the eval pipeline (split, subset draw, batching, metric) must stay identical across variants.

## Keep/discard rule

The goal is two-sided: drive **latency p95 down** while keeping **(micro_f1, macro_f1) at or above their floors**. Track best-so-far for `latency_p95_ms` on the branch.

A run is `keep` if **both**:

1. Quality floors hold on the 10k subset: `micro_f1 ≥ 0.890` AND `macro_f1 ≥ 0.770` (with subset noise budget — see below).
2. It Pareto-improves vs. branch tip on `(latency_p95_ms ↓, micro_f1 ↑, macro_f1 ↑)` — i.e. latency strictly improves and neither F1 regresses below the previous run beyond the subset noise floor (micro ~0.005, macro ~0.013), OR latency holds and at least one F1 strictly improves beyond noise.

Else: `discard` (`git reset --hard`).

Special cases:
- **Baseline run**: always `keep` regardless of whether it hits the targets — it sets the anchor.
- **Floor violation** (any F1 below floor on the subset): `discard`, even if latency improved dramatically. If a candidate lands within ~0.005 micro / ~0.015 macro of the floor on the subset, re-run with `--subset-rows 0` (full val) before committing the discard — subset noise can flip a borderline call.
- **Latency target met (< 1000 ms p95) and subset floors held**: re-run quality with `--subset-rows 0` (full val) to confirm. If full-val F1s also hold, the goal is met — but **keep iterating** for headroom (lower p95 = comfortable budget for traffic spikes, sequence-length variance, lower hosting cost).
- Improvements < 5 ms on p95 are within benchmark noise; treat them with skepticism (re-run the benchmark before committing as `keep`).

## What you CAN do

- Edit anything under `src/cross_encoder_serve/` — that's the production serving path. Tokenization strategy, batching/sub-batching, autocast dtype, attention impl, output schema, calibration math, etc.
- Edit anything under `src/cross_encoder_train/` to support distillation (new module, new loss, new CLI). Don't break the existing `cross-encoder-train` CLI without good reason.
- Edit `configs/` freely.
- Add new dependencies to `pyproject.toml` and `uv sync` them. Specifically allowed (and likely useful): `onnxruntime-gpu`, `tensorrt`, `optimum`, `flash-attn`, `bitsandbytes`, `auto-gptq`, `awq`, `torch_tensorrt`. Pin versions and justify each addition in a commit message.
- Use `torch.compile` / CUDA graphs / SDPA backend selection / FlashAttention freely.
- Quantize: dynamic INT8, static INT8 with calibration on the train split, INT4 (GPTQ/AWQ), bf16/fp16 weights (not just autocast). Recalibrate temperature `T` afterwards if calibration drifts (re-fit by minimizing NLL on val per `manifest.json::calibration` recipe — this is allowed because `T` is a serving-time hyperparameter, not a model weight).
- Distill: train a smaller student (e.g. `deepset/gelectra-base` ~110M, `xlm-roberta-base` ~278M, `microsoft/Multilingual-MiniLM-L12-H384` ~117M, `microsoft/mdeberta-v3-base`, or even a 6-layer pruned version of gelectra-large) against teacher logits from `soup.ckpt`. Standard recipe: KL divergence on softened logits + optional MSE on hidden states. The teacher's logits are computed once over the train split and cached.
- Drop layers / heads from the teacher (structural pruning) and fine-tune to recover.
- Drop the LGBM stack entirely — it's outside the success criterion.
- Slim the API response schema if it helps (e.g. compute only `logit[EXACT]` instead of full 4-class softmax; though this is a tiny win).
- Pre-tokenize on a worker thread / use `tokenizers`' batch_encode_plus with parallelism. Pinned host memory + non-blocking H2D transfers.
- Sub-batch the 2000 offers internally (e.g. forward 4×500 instead of one 2000) if it helps the kernel mix or fits memory better. The API contract stays "2000 offers in one request"; sub-batching is internal.
- Recompute the calibration temperature `T` on val for any new (distilled / quantized) model. Update `manifest.json` style metadata or just hard-code in the candidate's serving config.

## What you CANNOT do

- Modify `src/cross_encoder_train/metrics.py` or `src/cross_encoder_train/labels.py`. They define the quality floor measurement.
- Change `LABEL_ORDER`, `NUM_CLASSES`, or remap raw labels.
- Change the val split definition: same `val_fraction`, same `seed`, same query-id-based split as today. The floors are anchored to today's split.
- Change `max_pair_length = 512` or otherwise reduce the worst-case input length the benchmark measures. The user explicitly said "worst case 512".
- Change the benchmark fixture after the baseline (see "Benchmark methodology"). New fixture = invalidated TSV.
- Optimize for hardware other than the 4090. No CPU-only path, no multi-GPU.
- Remove `p_exact_calibrated` from the `/rerank` response. It's the one field downstream actually consumes — it must stay. (You may zero-fill or drop the other 4-class fields and `predicted_label` if it buys speed.)
- Change `POST /rerank` to require batched-by-pair input or any new endpoint shape that breaks existing clients. Internal sub-batching is fine; external schema change is not.
- Modify the released `soup.ckpt`. Treat it as a frozen reference (read for distillation, never overwrite).

## The lever menu (no prescribed order — pick by expected impact × risk)

Rough expected-impact ladder, lightest to heaviest:

1. **Free-ish wins on the existing model**: `torch.compile(mode="reduce-overhead")` on the encoder (note `inference.py` deliberately disables it today; revisit on a 4090 in inference-only mode where it may behave differently from training). FlashAttention-2 if the gelectra config accepts it. CUDA graphs for the fixed (B, S) = (2000, 512) shape — but watch for sub-batching interaction. fp16 autocast instead of bf16 (slightly faster on Ada).
2. **Drop work**: only compute logit at `EXACT` index instead of full 4-class softmax (minor); skip building unused response fields; remove LGBM-related code paths from the serve hot path; pre-tokenize during request validation in parallel with model loading on first request.
3. **Lower precision on weights, not just autocast**: load the model in bf16 weights; INT8 dynamic quantization on the linear layers; INT8 static quantization with a calibration step on a train subset.
4. **Aggressive quantization**: INT4 GPTQ / AWQ on the encoder. Check quality carefully — minority classes (Substitute, Complement) are most fragile.
5. **Compile-and-export**: ONNX export → ONNX Runtime CUDA EP with IO binding (eliminates a lot of Python overhead). TensorRT EP or `torch_tensorrt` for fused kernels and aggressive precision.
6. **Distillation to a smaller encoder**: gelectra-large (334M) → gelectra-base (110M) or MiniLM-L12 (117M). Roughly 3× speedup if it holds quality. Combine with bf16/fp16 + compile + ONNX/TRT for compounding wins. Use teacher-logits KL on the labeled parquet (the user confirmed: only this dataset, no extra unlabeled corpus).
7. **Layer pruning + recover**: drop top-N layers of gelectra-large, distill the remaining stack against the full teacher's logits. Cheaper than from-scratch distillation, often very effective.
8. **Architecture-level rethink** (highest risk, highest reward): a smaller bespoke encoder, late-interaction cross-encoder variants, or a query-prefix-cached forward pass. Treat as a last resort — the smaller-encoder distillation path subsumes most of these in practice.

A useful sequencing instinct: **distill once, then stack kernel/precision wins on top.** A distilled student that meets quality unlocks 2–3× compounding from compile + ONNX/TRT + INT8 that the same wins on the 334M teacher would only deliver at risk of breaking the floor.

## Logging results

After every experiment (including crashes), append a row to `results.tsv` (this directory). The TSV has a header row and 8 columns:

```
commit	latency_p95_ms	latency_p50_ms	val_micro_f1	val_macro_f1	peak_vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. p95 end-to-end `/rerank` latency in ms (e.g. `1547.3`) — `0.0` for crashes
3. p50 end-to-end latency in ms (for context; not gating)
4. `val/cls/micro_f1` (e.g. `0.8945`) — `0.0000` for crashes
5. `val/cls/macro_f1` (e.g. `0.7723`) — `0.0000` for crashes
6. peak VRAM in GB during the benchmark, `.1f`
7. status: `keep`, `discard`, or `crash`
8. short description of what this experiment tried

Example:

```
commit	latency_p95_ms	latency_p50_ms	val_micro_f1	val_macro_f1	peak_vram_gb	status	description
a1b2c3d	1742.0	1690.5	0.8951	0.7782	7.8	keep	baseline (soup.ckpt, bf16-autocast, eager, 4090)
b2c3d4e	1480.2	1455.1	0.8950	0.7780	7.6	keep	+ torch.compile(reduce-overhead) on encoder
c3d4e5f	1390.7	1370.3	0.8947	0.7779	5.1	keep	+ fp16 weights (not just autocast)
d4e5f6g	1180.4	1160.0	0.8920	0.7745	5.1	keep	+ INT8 dynamic quantization on Linear layers
e5f6g7h	820.1	810.5	0.8943	0.7791	2.6	keep	distilled to gelectra-base + bf16 + compile + ONNX RT
f6g7h8i	0.0	0.0	0.0000	0.0000	0.0	crash	tensorrt build failed: unsupported op TokenTypeIds
g7h8i9j	1175.0	1158.0	0.8881	0.7702	5.1	discard	INT8 static (cal=1k) — micro_f1 below floor
```

Do **not** commit changes to `results.tsv` or `NOTES.md`. Those are for human review.

## Reading harness output

If either run produces no greppable lines, the run crashed — `tail -n 80 bench.log` (or `eval.log`) and try to fix. If unfixable in a few attempts, log `crash` and move on.

## The experiment loop

Run on a dedicated branch (`autoresearch/<tag>`).

LOOP FOREVER:

1. Note the current branch/commit.
2. Pick an idea from the lever menu (or invent one — combine prior wins, look at profiler output to find the actual bottleneck, etc.).
3. Implement it: edit `src/cross_encoder_serve/`, `src/cross_encoder_train/`, `configs/`, add deps, etc.
4. `git commit` the change.
5. Run `eval/eval_val.py` on the resulting model. **If quality floor is violated, `git reset --hard` immediately — no point benchmarking.**
6. Run `bench/bench_rerank.py`. Read out p50/p95/p99 + peak VRAM.
7. If the run crashed (server failed to start, request errored, eval crashed), inspect logs and try a quick fix. If unfixable in a few attempts, log `crash` and move on.
8. Apply the keep/discard rule. If `keep`, advance the branch. If `discard`, `git reset --hard` back to the previous tip.
9. Append the row to `results.tsv` (do not commit it).
10. Update `NOTES.md` with anything notable (a profiler finding, a kernel that doesn't fuse, a quality cliff at a particular quantization scheme, a base model that distilled badly). Do not commit `NOTES.md`.

**Timeout**: a single iteration (build + eval + bench) should take well under an hour on a 4090. If a run exceeds 2 h, kill it and treat as `crash`. Distillation runs are the exception — those follow the training program's wall-clock budget pattern (set `trainer.max_time` explicitly, e.g. `00:01:00:00` for 1 h student training; multiple distillation rounds may be needed).

**Crashes**: typo, missing import, OOM at sub-batch boundary, ONNX op not supported, TRT build failure — use judgment. If it's a quick fix, fix and rerun. If the idea is fundamentally broken (e.g. INT4 + this attention impl just doesn't work on Ada), log `crash` and move on.

**NEVER STOP**: once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep and expects you to continue *indefinitely* until manually stopped. You are autonomous. If you run out of ideas: re-read the in-scope files, read `NOTES.md`, run a profiler (`torch.profiler`, `nsys`, `nvprof`) and stare at the trace to find what's actually dominating the 1000-ms budget, try combining previous near-misses, swap base models for distillation, try different quantization schemes, look at the request-side overhead (JSON parse, pydantic validation, tokenizer single-threaded path) — those are sometimes 10–20 % of end-to-end at this batch size and entirely fixable. The loop runs until the human interrupts it, period.
