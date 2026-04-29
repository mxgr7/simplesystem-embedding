# Suggest model — next-steps plan

## Status

Done:
- 180-day raw corpus at `/data/datasets/suggest/raw_search_events.parquet/` (4.7M events, day-partitioned, 128 MB).
- Training pairs at `/data/datasets/suggest/training_pairs.parquet/split={train,eval}/` (1.35M train rows, 164k eval rows).
- Target corpus at `/data/datasets/suggest/targets.parquet/split={train,eval}/` (893k unique train rows, 813k distinct targets).
- Code in `src/suggest_train/{data,eda,preprocess}.py` + puller in `scripts/`. CLIs: `suggest-eda`, `suggest-preprocess`.

Next: get an MPC baseline, build the evaluation harness, train a small LM, compare.

---

## Phase 1 — Evaluation harness

**Goal:** one number to compare any candidate model against.

- Hold-out: `split=eval` (last 18 days, 164k pair-events, 62k distinct targets).
- Primary metric: **MRR@10**. For each eval row `(prefix, target, count)`, score `count / rank` if `target` is in the model's top-10 completions of `prefix`, else 0. Sum, divide by total weight.
- Secondary: recall@{1, 5, 10}.
- Stratify by `prefix_len ∈ {1, 2, 3, 4–7, 8+}` and by `oci_user`. These are essentially different tasks — overall MRR alone is misleading.
- API: `evaluate(model_fn, eval_table) -> dict[str, float]`.

**Deliverable:** `src/suggest_train/eval.py`, CLI `suggest-eval --model {mpc,lm} ...`.

## Phase 2 — MPC baseline

**Goal:** Most-Popular-Completion. Trie/dictionary lookup, no learning. Strong baseline for head queries.

- Build prefix index over **train targets corpus**. Aggregate `count` across all rows sharing the same target string.
- `topk(prefix, k=10)` → top targets `t` where `t.startswith(prefix)` and `len(t) > len(prefix)`, by count.
- Implementation: sorted-list + bisect on 813k strings is fine for now. Swap to `marisa-trie` or `pygtrie` if we need μs latency later.
- Train two variants:
  - Pooled (all OCI / all search modes).
  - Filtered to non-OCI + `search_articles_by = STANDARD` (likely cleaner head distribution).

**Deliverable:** `src/suggest_train/mpc.py` with `class MPC.topk(prefix, k)`, CLI `suggest-mpc --build`.

**Expected:** MRR@10 ≈ 0.3–0.5 for prefix_len ≥ 4; weak (<0.1) at len=1 (huge fanout).

## Phase 3 — Tokenizer

- SentencePiece BPE trained on train targets, weighted by `count`.
- Vocab: **8k**. Queries are short — broader vocab buys nothing.
- Specials: `<s>`, `</s>`, `<pad>`, `<sep>` (prefix/completion separator).
- Save to `data/datasets/suggest/tokenizer/`.

**Deliverable:** `src/suggest_train/tokenizer.py`.

## Phase 4 — Small decoder-only LM (from scratch)

**Architecture (starting point):**
- 6 layers × 384 hidden × 6 heads × 4× FFN ≈ **~15M params**
- Context: 64 tokens
- RoPE positions, tied I/O embeddings
- Mixed-precision training

**Training data — two options:**

  - **(a) Targets-only:** `<s> target </s>`, sample weighted by `count`, standard next-token CE.
  - **(b) Prefix-conditioned pairs:** `<s> prefix <sep> completion </s>`, loss only on completion tokens.

  Start with **(a)** — simpler, larger effective corpus, learns the full-query distribution. Move to (b) if (a) underperforms.

**Training:**
- AdamW, lr 3e-4, cosine schedule, 1k warmup
- Batch 256–512, ~50 epochs, early-stop on eval MRR@10
- Lightning module + MLflow logging (mirror `embedding_train` patterns)
- Hydra config in `configs/suggest/`
- Time: **~30 min on a single RTX 4090** (vast.ai)

**Inference:**
- Beam search width 10–20 conditioned on `prefix`.
- Constrained decoding to enforce `output.startswith(prefix)` at the byte level — or as fallback, post-filter. (Constrained is cleaner; fallback is faster to build.)

**Deliverables:** `src/suggest_train/{model,train,infer}.py`, CLIs `suggest-train`, `suggest-infer`.

## Phase 5 — Hybrid + iteration

Once both MPC and the LM exist:

- **Hybrid candidate generation + ranker**: union top-50 MPC candidates and top-50 LM candidates, rerank with a LightGBM LTR on features `(lm_logprob, mpc_count, prefix_len, target_len, oci_user, search_articles_by, ...)`. Standard production pattern.
- **Conditioning**: add `oci_user` and `search_articles_by` as conditioning tokens / soft prompts to the LM. Measure stratified MRR.
- **Refresh**: re-pull PostHog weekly. The puller is resumable; `--days 7` extends the corpus by the missing tail.

## Phase 6 — Deployment (later)

- Add `/suggest?q=` endpoint to existing `search-api/`.
- Latency budget: <30 ms p95.
- 15M-param LM via ONNX Runtime is comfortably under that on CPU; MPC alone is sub-ms.

---

## Open decisions (resolve at the relevant phase)

1. **MPC scope** (Phase 2): pool all data or filter to non-OCI / STANDARD only? *Default: train both, pick on eval MRR.*
2. **LM input format** (Phase 4): targets-only (a) or prefix-conditioned pairs (b)? *Default: start with (a).*
3. **Diacritic handling**: trained on raw `query_term` (kept `ä ö ü ß`). MPC lookup may need a normalization fallback for users who type `ae` for `ä`. *Defer until Phase 5.*
4. **Frequency reweighting** (Phase 4): linear or log weighting on `count`? *Default: linear (sample proportional to count).*
5. **Eval freshness**: 18 days includes the freshest data. Add a "yesterday-only" drift-monitoring slice? *Defer to deployment.*
