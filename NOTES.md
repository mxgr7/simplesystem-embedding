# Next experiments

Triplet is parked (see below). The next SOTA candidate is contrastive on the
fresh SHN pool.

## Recent runs

| run | loss | SHN pool | epochs | peak MRR | nDCG@10 | Recall@10 | notes |
|---|---|---|---|---|---|---|---|
| `adorable-mole-653` (`e7809027…`) | contrastive | `placid-snake-749` | 10 | **0.816** | 0.740 | 0.931 | current SOTA, peaked ep ~7.5 |
| `respected-bass-254` (`680d0bfe…`) | triplet/`semi_hard` | stale `placid-snake-749` | 5 (killed) | 0.7478 (ep 4) | 0.6742 | 0.8505 | regressed ep 5 |
| `nosy-smelt-449` (`2a7cebd9…`) | triplet/`semi_hard` | fresh `adorable-mole-653` | 4 (killed) | 0.7329 (ep 3) | 0.6588 | 0.8321 | regressed ep 4 |
| `beautiful-bear-447` (`eae167e0…`) | triplet/`hardest` | fresh `adorable-mole-653` | 3 (killed) | 0.7163 (ep 2) | 0.6544 | 0.8403 | regressed ep 3, loss locked at margin |

All: `intfloat/multilingual-e5-base`, bs=256, lr=1e-5, warmup_ratio=0.33
per-epoch.

## Triplet is parked

Three triplet runs across two negative-selection modes and two pool-freshness
conditions all peak in the 0.72–0.75 MRR band and regress within one epoch of
the peak. Contrastive with identical config hits 0.816 smoothly. Gap is
~0.08 MRR and one-directional.

Observations that shaped the call:

- Fresh pool fixed the data confound (`train/triplet_semi_hard_fallback_share
  = 0.0` on the new pool vs meaningful fallback on the stale one). Retrieval
  still regressed, so stale pool was never the root cause.
- `hardest` mode's train loss parks at the margin (~0.20 with margin=0.2)
  because the hardest negative on a hard-mined pool usually sits at or above
  the positive. Loss signal is useless in that mode; retrieval still tracks
  semi-hard within 0.003 MRR but collapses one epoch earlier.
- Val loss decouples from `val/full_catalog/*` on every triplet run — val
  loss keeps improving (or stays parked) while retrieval regresses.
- `train/triplet_valid_anchor_count` is only 2–20 per batch of 256. Triplet
  structurally uses far less of each batch than contrastive's in-batch
  negatives do.

The triplet implementation stays in the tree — it's tested and cheap to keep.
Revisit only on a different base model, a different dataset, or a
qualitatively different batching format (not another tuning knob on this
setup).

## Tooling — close before the next launch

- ~~`triplet_semi_hard_fallback_share` metric~~ — done in `2c537fe`. Triplet
  runs log fallback share/count + valid-anchor count (train and val).
- **`ModelCheckpoint`** with `save_top_k>=1`, monitor `val/full_catalog/mrr`.
  Every overshoot so far has lost its peak weights.
- **`EarlyStopping`** on the same monitor, patience ~2 epochs. Stops the
  regression tail automatically and pays off on contrastive runs too.

## Experiments

In priority order. All inherit `adorable-mole-653`'s config (e5-base, bs=256,
lr=1e-5, warmup_ratio=0.33) unless noted.

### 1. Contrastive on the fresh SHN pool

The pool mined from `adorable-mole-653` is ready
(`semi_hard_negatives-adorable-mole-653.parquet`, 21k queries / 210k records).
Re-train contrastive on it — most likely source of lift, since the current
SOTA was trained on a pool mined from a 6-MRR-points-weaker encoder and its
`batch_semi_hard_negative_share ≈ 0.04` suggested the old pool barely
contributed.

### 2. Schedule audit: 8 vs 10 epochs

`adorable-mole-653` peaked ~ep 7.5 with the tail flat. Re-run #1 at
`trainer.max_epochs=8`. If gap < 0.005 MRR, make 8 the default — saves 20%
on every subsequent run.

### 3. Warmup sweep

`warmup_ratio ∈ {0.1, 0.2, 0.33, 0.5}` once #1 and #2 have landed.

### Deferred (only if the above leaves a clear gap)

- **Batch-size sensitivity** at production batch size.
- **LR sensitivity** (`{5e-6, 1e-5, 2e-5}`).
- **Regularization** (`weight_decay=0.1` or LR cooldown for last 2 ep).
- **Re-mining iteration**: mine a second pool from the #1 winner if that run
  beats current SOTA by more than the pool-staleness-implied gap.
- **Triplet revisit**: only if a new base model or dataset changes the
  structural picture above.
