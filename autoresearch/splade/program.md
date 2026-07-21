# autoresearch (SPLADE edition)

You are autonomously improving our German B2B product-search SPLADE model
(learned sparse retrieval: gbert-base backbone, SPLADE-max pooling, FLOPS
regularization). The current best models were trained 2026-07-21 with
"kitchen-sink" doc fields and symmetric text folding; your job is to push
retrieval quality further while keeping the model deployably sparse.

**Success metrics are seg R@100(E) AND gold R@100(E)** — recall@100 for
Exact-labeled docs on two frozen retrieval pools, measured by the harness at
top-256 doc pruning. seg = 984 held-out id_plus_text/text_spec terms (the
production segments we historically underserved); gold = 1,000-term frozen ESCI
gold test split. Both matter: seg is the targeted-quality metric, gold is the
regression guard (models have improved seg while quietly losing gold before).
Single-seed noise on both is ≈ ±0.003.

**Sparsity guardrail (hard)**: FLOPS_w ≤ 6.0 on both evals. Sparse serving cost
is a deployment requirement. Current models sit at ~1.5–2.7 — you have headroom,
but a dense-ish model that wins recall is NOT a keep.

**Keep/discard rule**: keep iff mean(Δseg, Δgold) ≥ +0.002 vs the current best
keep AND neither metric drops by more than 0.004 AND FLOPS_w ≤ 6.0 on both.
Otherwise `git reset --hard` to the last keep.

## Baselines (already trained; beat these)

| model | seg R@100 | gold R@100 | seg R@10 | gold R@10 | FLOPS_w | qnnz |
|---|--:|--:|--:|--:|--:|--:|
| fold_raw  | .9635 | .8999 | .4316 | .4866 | 1.6–2.2 | 43–44 |
| fold_b50  | .9523 | .9101 | .4209 | .4927 | 2.1–2.7 | 42 |
| soup_fold | .9588 | .9040 | .4293 | .4925 | 1.6–2.3 | 39 |

Reference recipe (fold_raw): warm-start `data/v1a_best.ckpt` (domain-pretrained
SPLADE backbone — training gbert from scratch on these sets DEGENERATES, never
skip the warm start), `data=splade_sink`, flops_lambda_q=5e-4,
flops_lambda_d=3e-4, flops_warmup_steps=300, lr=1.5e-5, 8 epochs, batch 256 +
accumulate_grad_batches=2 (batch 512 @ seq 512 OOMs the H100 — SPLADE logits
are [B,seq,vocab] ≈ 32 GiB), encode_batch_size=32.

## Setup

1. You are in a git worktree at `/workspace/autoresearch-splade-wt`, branch
   `autoresearch/splade-jul22`. Work here; commit code changes to this branch.
2. Training/eval runs execute on the H100 box `vastai0` via
   `autoresearch/splade/run_remote.sh` and `eval_remote.sh` (read both scripts
   now — they encode the ssh/staging mechanics; do not reimplement them).
   ssh access: `ssh -F /workspace/.ssh/vastai.conf vastai0`.
3. Read in scope: `README.md`, `AGENTS.md`, `configs/**`,
   `src/embedding_train/**`, `scripts/splade_flops_stoplist.py`,
   `scripts/soup.py`, `autoresearch/splade/NOTES.md` (pre-seeded with
   everything already tried — read it FIRST, do not rediscover it).
4. Verify remote data exists (run
   `ssh -F /workspace/.ssh/vastai.conf vastai0 'ls ~/simplesystem-embedding/data/splade_train_raw_fold.parquet ~/simplesystem-embedding/data/v1a_best.ckpt'`).
5. Verify the GPU is free before EVERY launch (run_remote.sh checks, but you
   own the etiquette): another team job may be running; if busy, wait — never
   kill processes you did not start.
6. `results.tsv` here is pre-seeded with the baselines. Append; never rewrite.
7. First run: reproduce the fold_b50 screen baseline (45-min budget, seg eval)
   to validate the whole loop before touching any knob.

## Experimentation

- **Screens**: train on `data/splade_train_b50_fold.parquet` with
  `trainer.max_time=00:00:45:00`, then seg eval only. A screen that clearly
  beats the screen baseline graduates.
- **Keeps**: full 8-epoch run on `data/splade_train_raw_fold.parquet`
  (`trainer.max_time` unset), then BOTH evals. Only full runs can be keeps.
- Example screen (overrides are plain `key=value` tokens, no spaces/quotes —
  they are passed through ssh; data paths must be absolute box paths):
  `autoresearch/splade/run_remote.sh myrun data.path=/home/max/simplesystem-embedding/data/splade_train_b50_fold.parquet trainer.max_time=00:00:45:00 optimizer.lr=1.5e-5 trainer.max_epochs=8`
  then `autoresearch/splade/eval_remote.sh myrun seg`.
- The known-good recipe is baked into the branch configs (warm start, FLOPS
  λs/warmup, batch 256; `trainer.accumulate_grad_batches=2` and
  `encode_batch_size=32` are fixed by run_remote.sh) — a run with only the
  overrides in the example above IS the reference recipe.
- Timeout: kill a run at 2× its budget + 10 min (`run_remote.sh` prints the
  remote pid; screens that hang are discards).

**What you CAN do**
- Any `configs/**` and `src/embedding_train/**` change: losses (contrastive
  temperature/margin, MarginMSE/KL distillation — CE teacher scores are at
  `~/simplesystem-embedding/data/ce_scores*.parquet` on the box; the loader
  supports `data.ce_scores_path`), FLOPS regularizer shape/schedule, stopword
  masking (`model.stopword_mask_ids`), pooling, field dropout
  (`data.field_dropout_p` — code is in this branch, verified), sampling
  (n_pos/n_neg, batching mode), template/field-order edits in
  `configs/data/splade_sink.yaml`, tokenizer max lengths.
- Backbone swaps: gbert-large checkpoints exist on the box
  (`~/simplesystem-embedding/checkpoints/splade-large-pretrain/` and
  `splade-large-gold-ft/`); mind VRAM at seq 512.
- Warm-start from any existing checkpoint (fold_*/soup_fold included) for
  continued training.
- Model soups over your own keeps (`scripts/soup.py`).
- Self-mined hard negatives FROM THE PROVIDED TRAIN PARQUETS ONLY (e.g. encode
  the train pool with a checkpoint, mine in-domain negatives, feed via
  `data.semi_hard_negatives_path`).
- Query-side augmentation derived from the provided train queries.

**What you CANNOT do**
- Touch `scripts/splade_flops_stoplist.py` metrics logic, the `*_fold` eval
  jsonls, or anything under `~/simplesystem-embedding` on the box other than
  reading `data/` and `checkpoints/` (your runs live in `~/ar_splade`).
- Introduce new supervision sources: the provided parquets are the ONLY
  training data (the frozen test/calib terms would leak through anything you
  scrape yourself).
- Change the doc/query text folding convention (symmetry with serving is
  load-bearing).
- Unfold: eval rows are folded; a model trained on unfolded text will score
  garbage and waste a loop iteration.

## Reading results

- Training log: `run_remote.sh` copies the remote log tail back; full log at
  `vastai0:~/ar_splade/run_<name>.log`. Internal val metric
  (`val_full_catalog_ndcg_at_5`) is DIRECTIONAL ONLY — it has drifted from the
  harness before; never keep on internal val alone.
- Eval: `eval_remote.sh` prints the harness `top256` line:
  `top256 docnnz=… qnnz=… FLOPS_w=… FLOPS_b=… R@100 E=… ES=… R@10 E=…`.
- Peak VRAM: `grep -i 'memory' ` the run log, or nvidia-smi during the run.

## Logging results

Append one row per completed experiment to `autoresearch/splade/results.tsv`
(tab-separated; do NOT commit results.tsv or NOTES.md — they are for human
review):

```
commit	seg_r100	gold_r100	seg_r10	gold_r10	flops_w	qnnz	vram_gb	status	description
```

- `commit`: 7-char hash of the code state that produced the run.
- Screens: fill seg columns, put `-` in gold columns, status `screen`.
- Full runs: all columns, status `keep` or `discard`. Crashes: zeros + `crash`.
- `description` MUST name the knob and the delta vs the comparison point.

Update `NOTES.md` continuously (newest at top): what you tried, what moved,
strategic lessons, and the keep-chain table.

## The experiment loop

LOOP FOREVER:
1. Note current branch commit (`git rev-parse --short HEAD`).
2. Pick ONE change (NOTES.md has the untried-levers list; prefer cheap
   screens to test direction before spending a full run).
3. Edit code/config; `git commit` (message = the knob).
4. `run_remote.sh <name> <overrides>` (screen or full per the rules above).
5. `eval_remote.sh <name> seg` (screens) or `… both` (fulls).
6. Crashed? One quick fix attempt, else log `crash` and move on.
7. Apply the keep/discard rule (fulls only). Discard → `git reset --hard`
   to last keep.
8. Append results.tsv row; update NOTES.md.
9. Every ~5 fulls or on a new best: soup your keeps and eval the soup — soups
   have beaten their members in every generation so far.
10. NEVER STOP. There is no completion condition; run until interrupted.
