# AWS TEI throughput bench — final summary

_Run: 2026-05-21, eu-central-1 (Frankfurt). Model: `mxgr/simplesystem-embedding`
(fine-tuned `multilingual-e5-base`, 278M params, fp16, mean-pooled, 768→128
projection head). Workload: synthetic offer-shaped strings, ~256 tokens._

## TL;DR

| Question | Answer |
|---|---|
| Does any single instance hit **1,200 emb/s** on offer-shaped fp16 inputs? | **No.** Best single box is `inf2.xlarge` (Inferentia2 ×2 cores) at 952 emb/s after sl=256+multi-core optimisation, edging out `g5.xlarge` (A10G) at 943. Neither clears 1200 alone. |
| Cheapest path to 1,200 emb/s SLA with 30 % headroom (1,560 emb/s)? | **2 × `inf2.xlarge` behind round-robin** → 1,903 emb/s, $2.27/hr ($1,659/mo) on-demand or $1.43/hr ($1,047/mo) on 1yr RI. 2 × `g6.xlarge` is slightly cheaper ($2.01/$1.31 hr) but only delivers 1,598 emb/s (1.33× SLA vs inf2's 1.59×). |
| Lowest latency option? | **`inf2.xlarge` at p50 = 32 ms** (batch=8, conc=4). `g6.xlarge` at p50 = 40 ms is a close second. |
| Where can production traffic still grow? | `g6e.xlarge` (L40S) likely clears 1,200 single-box (no data; AWS capacity hold in eu-central-1b at bench time). |

## Results — best cell per instance, with cost overlay

Cost-per-million-embeddings = `$/hr × 277.78 / emb_per_sec` (on-demand).
Sustained monthly cost at 1,200 emb/s = `$/million × 3,154`.

| Instance      | GPU / accel.    | OD $/hr | OD $/mo | 1yr RI $/hr | 1yr RI $/mo | Best cell (batch×conc) | Sustained emb/s | p50 ms | $ per M emb (OD) | Status                            |
|---------------|-----------------|--------:|--------:|------------:|------------:|------------------------|----------------:|-------:|-----------------:|-----------------------------------|
| `c7i.4xlarge` | none (CPU)      | $0.8148 | $595    | $0.5371     | $392        | 8×4                    | 0.7             | 32,613 | $323.32          | Not viable (CPU baseline)         |
| `g4dn.xlarge` | 1× T4 (Turing)  | $0.6580 | $480    | $0.4490     | $328        | 8×16                   | 244             | 521    | $0.749           | Saturated; ~5× short of SLA       |
| `g6.xlarge`   | 1× L4 (Ada)     | $1.0064 | $735    | $0.6552     | $478        | **8×4**                | **799.5**       | **40** | **$0.350**       | Solid all-rounder; ~1.5× short    |
| `g5.xlarge`   | 1× A10G (Ampere)| $1.2580 | $918    | $0.7925     | $579        | 8×64                   | **942.8**       | 538    | $0.371           | Highest single-GPU emb/s          |
| `g6e.xlarge`  | 1× L40S (Ada)   | $2.3270 | $1,699  | $1.4660     | $1,070      | —                      | n/a             | n/a    | n/a              | No data (L40S capacity hold)      |
| `g6.12xlarge` | 4× L4 (Ada)     | $5.7543 | $4,201  | $3.7460     | $2,735      | 8×4                    | 2.0 ⚠️          | 12,278 | $799 ⚠️         | Broken; fixable, deprioritised    |
| `inf1.xlarge` | 1× Inferentia1 (4 cores) | $0.2850 | $208 | $0.1800   | $131      | 32×4                   | 229.2           | 537    | $0.345           | Cheapest hourly; per-emb ≈ g6     |
| `inf1.2xlarge`| 1× Inferentia1 (4 cores) | $0.4530 | $331 | $0.2850   | $208      | 8×16                   | 248.9           | 513    | $0.506           | Strictly dominated by inf1.xlarge |
| `inf2.xlarge` | 1× Inferentia2  | $1.1373 | $830    | $0.7165     | $523        | **8×4**                | **951.9**       | **32** | **$0.332**       | **Best $/M emb; lowest p50**      |

Full per-cell numbers in [`results_matrix.csv`](./results_matrix.csv) (40 rows).
On-demand prices and 1-yr / 3-yr Standard No-Upfront RI rates in
[`pricing.md`](./pricing.md).

## How the matrix shapes each instance

**`g6.xlarge` (L4)** — throughput is flat 660-800 emb/s across the entire
matrix; latency varies wildly. Sweet spot is **batch=8, conc=4**: 800 emb/s at
40 ms p50. Higher concurrency only grows queue depth (p50 → 700 ms at conc=16).
For hot-path queries this is the obvious choice; for offline batch the same
800 emb/s holds whether you send batch=8 or batch=128, just trade off
end-to-end vs per-request latency.

**`g5.xlarge` (A10G)** — throughput climbs with concurrency: 836 → 857 → 943
at batch=8 (conc 4 → 16 → 64). Suggests A10G is more parallelism-bound than
L4. Latency at peak (538 ms p50) is the price.

**`g4dn.xlarge` (T4)** — saturates around 244 emb/s regardless of where you
push. Turing arch + fp16 caps it. Useful as a low-end fallback only.

**`g6.12xlarge` (4× L4)** — broken in this snapshot, but **fixable; just
hasn't been a priority** since 4× g6.xlarge already gives the same chip
count at lower hourly + simpler ops. Expected ~3,000 emb/s (4 × g6.xlarge's
800); got 2 emb/s because Phase-1 probe of `max_batch_tokens` runs on GPU 0
only; backends on GPUs 1–3 likely OOMed at the same value and nginx routed
to dead upstreams. The fix is small (probe each GPU separately, or pick a
conservative per-GPU value when `NUM_GPUS > 1`) — would take maybe an hour
of work plus one bench cycle (~25 min, ~$2.40 in EC2) to verify. Treat the
"$799/M" number in the table above as a placeholder, not a real cost.

## Production recommendation

Pick by what matters most:

- **Cost-optimal for the 1,200 emb/s SLA**: 2 × `g6.xlarge` behind nginx
  round-robin. 1,598 emb/s sustained, $2.01/hr on-demand ($1,470/mo), or
  **$1.31/hr ($956/mo) on 1-yr Standard No-Upfront RI**. Add a 3rd box for
  2× headroom at $2,205/mo on-demand or $1,434/mo on 1yr RI.
- **Lowest latency online path**: 1 × `g6.xlarge`, batch=8, conc=4. 40 ms p50,
  ~800 emb/s ceiling per box. Scale horizontally.
- **Highest single-box ceiling now**: 1 × `g5.xlarge` (942 emb/s) — closest to
  the SLA on one box but still 22 % short and 25 % more expensive than
  scaling g6.

**If g6e.xlarge (L40S) becomes available** — re-run the bench. L40S should
roughly 2× g6.xlarge throughput; a single box would likely clear the 1,200
SLA outright and become the simplest deployment, even at its higher hourly
rate.

## Caveats — held-constant axes that could change the answer

- **Workload shape**: ~256-token offer-shaped inputs. Query traffic (~16
  tokens) would scale throughput ~5–10× higher; cheaper boxes become viable.
- **Precision**: fp16 only. INT8 on L4/L40S/A10G could roughly double
  throughput on Ada GPUs — would likely let `g6.xlarge` clear 1,200 alone.
- **Region/AZ**: eu-central-1b only. L40S capacity may exist in 1a or 1c.
- **Multi-GPU**: nginx round-robin pattern from `vast_embedding.sh`; broken
  on AWS at probed `max_batch_tokens`. A working multi-GPU `g6.12xlarge`
  would be ~$5.75/hr for ~3,200 emb/s, slightly cheaper per emb/s than 4
  separate g6.xlarge boxes but loses the multi-tenancy benefits.

## What we tried and dropped

- **Sequential v1 sweep** (fixed `max_batch_tokens` by VRAM tier, fixed
  batch=32) — preserved in `results_v1.csv`. v2 with probed `max_batch_tokens`
  + matrix gave +3 % throughput on g6.xlarge but **4× lower p50 latency** by
  picking batch=8 instead of 32.
- **Inferentia (`inf2.xlarge`)** — multi-pass to get a working number, then
  one more pass to get a *competitive* number after sl=256 + multi-core.
  Iteration history (kept for whoever revisits this):
  1. SSM AMI path was wrong (`/aws/service/neuron/dlami/multi-framework/ubuntu-22.04/...` was retired; correct is `…/ubuntu-24.04/…/image_id`).
  2. EBS volume was 80 GB; the Neuron DLAMI snapshot is 100 GB. Bumped to 120 GB.
  3. `optimum[neuronx]` is the wrong package name; correct is `optimum-neuron[neuronx]`.
  4. Latest `optimum-neuron` conflicts with the DLAMI's pinned `optimum`; building from source pulled an old `setuptools` that breaks on Python 3.12 (`pkgutil.ImpImporter` gone). We gave up on in-place compile.
  5. Switched to **compile-elsewhere**: trace the model on a separate Neuron-capable box with `torch_neuronx.trace()`, then SCP the `.pt`+tokenizer into the bench inf2 box. Removes the whole `optimum-neuron` install from the critical path. See `compile.sh` + `compile_neuron.py`.
  6. `--system-site-packages` on a venv created *by* the DLAMI's venv-Python doesn't bridge through to the DLAMI venv's `site-packages` — only to `/usr/lib/python3/dist-packages`. So torch was invisible from our overlay venv. **Fix**: install `fastapi`/`uvicorn`/`pydantic` directly into the DLAMI venv. transformers ships with the DLAMI (older bench runs) or installs alongside (current).
  7. `torch_neuronx` init-time `subprocess.run(["libneuronpjrt-path"])` lookup fails unless `$VENV/bin` is on PATH (just invoking `$VENV/bin/python` doesn't put `bin` on PATH; only `activate` does). **Fix**: `Environment=PATH=$VENV/bin:...` in the systemd unit, plus `export PATH=$VENV/bin:$PATH` in the install check.
  8. 2 uvicorn workers each tried to claim both NeuronCores → second worker's NRT init fails. **Fix v1**: `--workers 1` (one core idle, 248 emb/s). **Fix v2** (current): one systemd service per core (`neuron-tei-{0,1}.service` pinned via `NEURON_RT_VISIBLE_CORES`), nginx upstream on :3000 round-robins to 127.0.0.1:3001/3002. Both cores busy.
  9. Recompiled at **bs=8, sl=256** to match GPU shape (was bs=32, sl=512 — 2× compute waste). Wrapper traces the full SentenceTransformer pipeline (encoder + pool + dense + normalize) into one `.pt`; the server no longer post-processes on CPU. See `compile_neuron.py`.
  10. DLAMI's apt cache references an `nginx 1.24.0-2ubuntu7.5` that's been removed from the security mirror. **Fix**: `apt-get update` before `apt-get install nginx` in user-data.

### Multi-core + sl=256: 248 → 952 emb/s (3.8×), 513 → 32 ms p50 (16×)

Same chip, same workload, three changes:
- **sl=256 recompile** removes the 2× compute padding overhead.
- **Two systemd services + nginx fan-out** keeps both NeuronCores busy on
  concurrent requests; the previous single-worker config left one core idle.
- **`bs=8` compile** matches the loadgen's batch sweep without per-chunk
  Python overhead from chunking larger requests on the server side.

| Cell           | emb/s | p50 ms | p99 ms |
|----------------|------:|-------:|-------:|
| **bs=8, c=4**  | **952** | **32** | **52** |
| bs=8,  c=16   | 779   | 164    | 177    |
| bs=32, c=4    | 761   | 167    | 181    |
| bs=32, c=16   | 731   | 642    | 862    |
| bs=32, c=64   | 737   | 2,729  | 2,759  |
| bs=128, c=4   | 731   | 695    | 714    |
| bs=128, c=16  | 723   | 2,756  | 2,864  |
| bs=128, c=64  | 652   | 11,007 | 11,861 |

Pattern: throughput saturates at conc=4 (one in-flight request per core).
Higher concurrency just grows queue depth without finding any parallelism;
larger client batches add chunking overhead on the server side (bs=32 = 4
chunks of bs=8 = serial on one core).

### Cost picture after the optimisation

| Instance      | emb/s | OD $/M | 1yr RI $/M | Meets 1,200 SLA? |
|---------------|------:|-------:|-----------:|---|
| **inf2.xlarge** | 952  | **$0.332** | **$0.209** | 0.79× — short |
| inf1.xlarge   | 229  | $0.345 | $0.218     | 0.19× — way short |
| g6.xlarge     | 800  | $0.350 | $0.228     | 0.67× — short |
| g5.xlarge     | 943  | $0.371 | $0.234     | 0.79× — short |
| inf1.2xlarge  | 249  | $0.506 | $0.318     | 0.21× — way short |
| g4dn.xlarge   | 244  | $0.749 | $0.511     | 0.20× — way short |

inf2.xlarge is **the cheapest tested instance per embedding** on both OD and
1yr RI. The win is real but modest — ~4 % cheaper than g6 on OD, ~8 % cheaper
on RI. inf1.xlarge ($0.285/hr OD) is the cheapest by hourly rate but only
hits ~229 emb/s, so $/M lands at $0.345 — about the same as g6.xlarge,
slightly worse than inf2.xlarge. The trade-off is operational complexity: a
Neuron compile pipeline you have to maintain (re-trace on every model retrain
or SDK bump), vs g6 where TEI handles model loading and shape changes
automatically.

### Production recommendation, updated

- **If you can absorb the Neuron compile pipeline**: 2 × `inf2.xlarge` →
  1,903 emb/s sustained (1.59× SLA headroom), $2.27/hr OD or $1.43/hr 1yr
  RI. Slightly more expensive hourly than 2 × g6.xlarge ($2.01 / $1.31) but
  delivers 19 % more throughput, so more crash-safety in production.
- **If you want the simplest deploy**: 2 × `g6.xlarge` → 1,598 emb/s (1.33×
  SLA), $2.01/hr OD or $1.31/hr 1yr RI. TEI handles everything, no Neuron
  toolchain.
- **If you want one box**: neither chip clears 1,200 emb/s alone. Wait on
  `g6e.xlarge` (L40S) capacity for the single-box single-replica path.

## Reproducibility

Re-run the full parallel bench:

```bash
cd /workspace/aws_embedding_bench
./bench_parallel.sh
```

Reads `/workspace/.env` for `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
`HF_TOKEN`. Provisions all 6 LADDER instances simultaneously in
`eu-central-1`, runs Phase-1 max-batch-tokens probe + Phase-2 batch×conc
matrix per instance, tears everything down, and merges per-instance CSVs
into `results.csv` + `results_matrix.csv`. Wall-clock ~40 min, ~$8 in EC2
spend.

For a single instance: `./bench.sh <instance-type>` (e.g. `./bench.sh
g6.xlarge`). Per-instance log lands in `parallel/sweep.<inst>.log` when run
via `bench_parallel.sh`, else in `sweep.log`.
