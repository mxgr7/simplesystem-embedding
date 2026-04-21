# Progressive Field Import — Results

Single-bucket sweep (`bucket=00.parquet`, 9,954,348 rows) on Hetzner box
(48 cores / 184 GB / 512 GB NVMe). Each row = one full
drop → create → stream-insert → flush → IVF_FLAT + INVERTED →
compaction-settle → load → measure cycle. Cumulative field set per iter.

Full data: `logs/progressive_sweep.csv` · detail log:
`logs/progressive_sweep.log` · orchestrator: `scripts/progressive_import.py`.

## Summary table

All sizes are decimal GB (df/mc-style). `Δ RSS` and `Δ MinIO` are the
delta to the previous iteration, i.e. the *marginal* cost of the field
added in that iter. Insert rate is `rows/s` averaged over the entire
insert phase. Wall time is full iter wall (cleanup + insert + indexes +
compaction + load + measure).

| # | Field added | Schema type + index | Insert rate | Δ Insert s | Loaded RSS | Δ RSS | MinIO | Δ MinIO | Wall min |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| 0 | *(baseline: id, offer_embedding)* | VARCHAR PK + FLOAT16_VECTOR(128, IVF_FLAT) | 45,741 | — | 6.98 | — | 8.8 | — | 13.7 |
| 1 | `n` | INT64 | 47,557 | −8 | 7.07 | +0.10 | 11.8 | +3.0 | 13.2 |
| 2 | `ean` | VARCHAR(32) | 46,963 | +3 | 7.27 | +0.20 | 15.0 | +3.2 | 13.5 |
| 3 | `article_number` | VARCHAR(64) | 46,877 | 0 | 7.39 | +0.12 | 18.3 | +3.2 | 14.5 |
| 4 | `manufacturerName` | VARCHAR(128) | 45,213 | +8 | 7.60 | +0.21 | 21.5 | +3.2 | 14.6 |
| 5 | `manufacturerArticleNumber` | VARCHAR(128) | 42,182 | +16 | 7.67 | +0.07 | 25.8 | +4.3 | 15.4 |
| 6 | `manufacturerArticleType` | VARCHAR(512) | 39,272 | +18 | 7.78 | +0.11 | 29.0 | +3.2 | 14.5 |
| 7 | `name` | VARCHAR(256) | 39,312 | 0 | 8.50 | +0.72 | 34.4 | +5.4 | 14.8 |
| 8 | `vendor_ids` | ARRAY<VARCHAR>(cap=32) + INVERTED | 30,227 | +76 | 9.06 | +0.56 | 38.7 | +4.3 | 15.6 |
| 9 | `category_l1..l5` (5 fields) | 5× ARRAY<VARCHAR>(cap=64) + 5× INVERTED | 15,997 | +293 | 11.85 | +2.79 | 46.2 | +7.5 | 21.1 |
| 10 | `catalog_version_ids` | ARRAY<VARCHAR>(cap=2048) + INVERTED | 14,054 | +86 | 13.35 | +1.49 | 54.8 | +8.6 | 22.8 |
| 11 | `description` | VARCHAR(65535), avg 499 B | 5,238 | +1192 | 18.55 | +5.21 | 69.8 | +15.0 | 42.0 |

Totals: 18.55 GB loaded RSS (+11.57 vs baseline), 69.8 GB MinIO
(+61.0 vs baseline), 215 min wall (3.6 h), never approached the
184 GB RAM cap or the 30 GB-free disk abort threshold (min free
observed: 290 GB).

## Per-field analysis

**`n` (INT64)** — essentially free. +0.10 GB RSS, +3.0 GB MinIO (raw
columnar replay log for 10M ints @ 8 B = 80 MB compressed — the rest is
flush overhead from the 8-byte-per-row fixed record appearing in each
segment's binlog). Insert rate actually nudged *up*, within noise.

**`ean` / `article_number` / `manufacturerName` /
`manufacturerArticleNumber` / `manufacturerArticleType`** — small
UTF-8 VARCHARs (avg 2–12 B per row per plan's prior scan). Each adds
+0.07 to +0.21 GB RSS and +3.2 to +4.3 GB MinIO. The MinIO hit is
dominated by insert_log columnar persistence (raw bytes + segment
metadata), not the in-RAM cost. Insert rate slowly degrades
(46.9k → 39.3k as we cross from 4 to 8 fields) — this is gRPC
payload overhead, not Milvus compute.

**`name` (VARCHAR(256), avg 57 B)** — +0.72 GB RSS, +5.4 GB MinIO.
Larger per-row footprint than the other text fields and it shows in
both dimensions. Still small in absolute terms.

**`vendor_ids` (ARRAY<VARCHAR>, cap=32 + INVERTED)** — +0.56 GB RSS,
+4.3 GB MinIO. First ARRAY + first INVERTED index. Insert rate drops
sharply from 39k to 30k rows/s: the columnar-array path is wider, and
the INVERTED index adds background work that shows up as slower
compaction. Still cheap relative to the filter value it provides.

**`category_l1..l5` (5 ARRAYs, cap=64 each, 5 INVERTEDs)** — +2.79 GB
RSS, +7.5 GB MinIO, +293 s insert, +5 extra INVERTED indexes finishing
post-flush. This is the largest *structural* step: insert rate halves
(30k → 16k rows/s) because each row now carries 5 additional variable-
length string arrays. Per-field this averages to +0.56 GB RSS / +1.5 GB
MinIO — roughly the same marginal profile as `vendor_ids`. Not a cliff;
scales linearly with array count.

**`catalog_version_ids` (ARRAY<VARCHAR>, cap=2048 + INVERTED)** —
+1.49 GB RSS, +8.6 GB MinIO, +86 s insert. The `max_capacity=2048` is
65× wider than `vendor_ids`, but the marginal RSS hit (+1.49) is only
2.7× and marginal MinIO (+8.6) is 2× the single-INVERTED ARRAYs above.
Real row-level capacity stays small in practice; the cap is headroom,
not materialized cost. This field had the reputation of being a
problem, but on this bucket it isn't one — it's ~same cost per byte as
the category fields.

**`description` (VARCHAR(65535), avg 499 B)** — +5.21 GB RSS, +15.0 GB
MinIO, +1192 s insert. The single largest marginal cost in the sweep
and the source of every real bottleneck. A 50k-row batch blew the
gRPC 64 MB message cap (observed: 67.5 MB for the failing batch),
forcing a 10× shrink to 5k/batch — that, together with the raw payload
size, is why insert rate drops to 5.2k rows/s. Post-load RSS grew
+5.2 GB (the data is held in the columnar store for return-as-output;
`load_fields` does not help, only `mmap` can disk-back it — see
`APRIL_20.md`). MinIO +15 GB captures the compressed binlog of ~5 GB of
raw description data. Still well within RAM and disk budget on this
bucket, but at full 16× scale this extrapolates to ~83 GB RSS and
~240 GB MinIO just for description — worth keeping on disk via
`mmap.enabled` from day one at 159M rows, or excluding it entirely.

## Interpretation

For a 1-bucket (~10M rows) load with everything loaded into RAM, the
RSS budget per field-category is roughly:

- Scalar primitives (INT64, small VARCHAR ≤128 B): +0.1–0.2 GB each
- Medium VARCHAR (name, mfg article type, up to 512 B cap, avg ≤60 B):
  +0.2–0.7 GB each
- Small ARRAY<VARCHAR> + INVERTED (cap ≤64): +0.5–0.6 GB each
- Wide ARRAY<VARCHAR> + INVERTED (cap=2048, actual rows still small):
  +1.5 GB
- Large-body VARCHAR (description, avg 500 B, cap 64 KB): +5+ GB

The non-linear step is **not** between field types — it's between
"fields with bounded per-row bytes" (everything before description) and
"fields where the per-row content itself is heavy" (description). The
category-array step is linear in array count and proportionally the
cheapest per-field on an RSS basis.

Insert throughput correlates almost perfectly with per-row gRPC payload
size: 45k rows/s at 2 fields collapses to 5k rows/s once description is
in. Batch-sizing at insert time (shrinking when wide fields are
present) is mandatory — the hard-coded 50k batch hits the 64 MB cap
immediately with description. Consider per-field heuristic batching in
future ingestion tooling.

## Caveats

- Results are from 1 bucket (10M rows). RSS scales near-linearly with
  row count but index-build and compaction overhead scale super-
  linearly, so full-scale (159M rows) numbers will likely be >16× the
  marginal costs above for index-heavy fields.
- All indexes fresh-built per iter — no incremental-insert shape.
- `description` sizing here was after byte-safe truncation at 65,535
  bytes; if truncation hit many rows on other buckets, per-field cost
  would change.
- MinIO numbers include both `insert_log` (raw columnar replay) and
  `index_files` (serialized index structures). `index_files` stayed at
  ~3.1–3.3 GB throughout (dominated by the vector index), confirming
  the scalar insert_log is where all the growth is.
- Compaction wait never exceeded the 30 min cap — all iters settled
  within 30–330 s with two consecutive clean polls.

## Extrapolation to full dataset (159.3M rows, 16×)

The bucket sweep is exactly 1/16 of the full corpus
(159,275,274 / 9,954,348 = 16.001). Per-field RSS is dominated by
per-row data (the Milvus runtime heap is ~1 GB regardless of row
count), so extrapolation is "scale per-row data by 16×, then re-add
the fixed heap".

| Iter | Field added | Δ RSS @ 1B | Δ RSS @ full | **Cumulative @ full** |
|---|---|--:|--:|--:|
| 0 | baseline (id + IVF_FLAT vec) | 5.98 (data) | 95.7 | **96.7** |
| 1 | `n` | +0.10 | +1.6 | 98.3 |
| 2 | `ean` | +0.20 | +3.2 | 101.5 |
| 3 | `article_number` | +0.12 | +1.9 | 103.4 |
| 4 | `manufacturerName` | +0.21 | +3.4 | 106.8 |
| 5 | `manufacturerArticleNumber` | +0.07 | +1.1 | 107.9 |
| 6 | `manufacturerArticleType` | +0.11 | +1.8 | 109.6 |
| 7 | `name` | +0.72 | +11.5 | 121.2 |
| 8 | `vendor_ids` | +0.56 | +9.0 | 130.1 |
| 9 | `category_l1..l5` (5 fields) | +2.79 | +44.6 | **174.8** |
| 10 | `catalog_version_ids` | +1.49 | +23.8 | **198.6** ⚠ |
| 11 | `description` | +5.21 | +83.4 | **282.0** ⚠⚠ |

⚠ = exceeds 90% threshold of 184 GB host RAM (queryCoord limit ≈
166 GB). ⚠⚠ = exceeds host RAM cap entirely.

### What this means for the 184 GB host

This config (IVF_FLAT, all-loaded, no mmap) **does not fit**
at full scale beyond iter 9. A literal repeat of the sweep on the full
corpus would OOM somewhere during iter 10 (`catalog_version_ids`).

The three knobs that bring it back inside the budget — each
independently large enough to matter — are exactly those identified in
`APRIL_20.md`:

1. **HNSW instead of IVF_FLAT** (saves ~45 GB at full scale). IVF_FLAT
   in Knowhere upcasts fp16 → fp32 internally for build and search,
   doubling vector RAM. HNSW keeps fp16 native end-to-end. Halves the
   baseline.
2. **`mmap.enabled=true` on scalar fields and their INVERTED indexes**
   (saves ~30–50 GB). Only mmap pushes scalar payloads to disk —
   `load_fields` does not restrict scalar INVERTED loading.
3. **Disk-back `description`** (saves ~83 GB). Either exclude from
   `load_fields` (insert_log still resident, but query-time materialize
   from disk) or mmap the field.

Cross-check: APRIL_20.md reports the actual full-dataset run at
**94.9 GB RSS** with HNSW + scalar mmap + `load_fields=[id,
offer_embedding]`. That matches an HNSW-halved baseline (~50 GB
vector + ~1 GB heap) plus residual mmap'd-but-not-fully-evicted
scalar pages (~45 GB) — close to expected.

Recommended full-scale config (predicted RSS in parentheses):

- HNSW vector index (~50 GB)
- mmap.enabled on all scalar fields + INVERTED indexes (~10–20 GB
  resident)
- mmap.enabled on `description` field (~5–10 GB resident)
- → predicted total: **80–100 GB RSS**, leaving ~80 GB headroom for
  query traffic, segment compaction, Go heap slack.

## Reproduce

```
uv run --no-project --with pymilvus --with pyarrow --with numpy \
  --with boto3 python scripts/progressive_import.py \
  --bucket bucket=00.parquet \
  --csv-out logs/progressive_sweep.csv \
  --log-out logs/progressive_sweep.log
```

Resume / spot-run a single iter:

```
... python scripts/progressive_import.py --iters 11-11 ...
```
