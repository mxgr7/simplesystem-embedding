# aws_embedding_bench

Benchmark TEI throughput for `mxgr/simplesystem-embedding` across AWS GPU,
CPU, and Inferentia instance types. Built to answer: **what's the cheapest
AWS instance that can sustain 1,200 embeddings/sec at ~256-token (offer-
shaped) inputs?**

Pulumi (Python) provisions a TEI server + a co-located `c7i.large` load-gen
box; the load-gen runs `loadgen.py` and writes one CSV row per instance to
`results.csv`. Everything is torn down after each run.

## Prereqs

- Pulumi CLI installed (`curl -fsSL https://get.pulumi.com | sh`); ≥ 3.130
  required for the `toolchain: uv` runtime option in `Pulumi.yaml`.
- `uv` installed (`curl -fsSL https://astral.sh/uv/install.sh | sh`).
  Pulumi uses it to create/manage the local venv and install
  `pyproject.toml` deps on first `pulumi up`. EC2 user-data also installs
  uv to speed up runtime pip installs.
- `aws` CLI (only for the quota pre-flight check; the bench itself uses
  the AWS SDK via pulumi-aws)
- `/workspace/.env` populated with `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `HF_TOKEN`. `AWS_REGION` defaults to
  `eu-central-1` (override in `.env`).

The IAM user needs `AmazonEC2FullAccess` (or equivalent). The first
`pulumi up` will fail loudly if permissions are missing.

### Quota requirements

AWS gates GPU and Inferentia instances behind service quotas. Confirm in
the Service Quotas console (region eu-central-1):

| Quota                                         | Required for                  |
|-----------------------------------------------|-------------------------------|
| Running On-Demand G and VT instances (≥ 48)   | every G-family in the ladder  |
| Running On-Demand Inf instances (≥ 4)         | `inf2.xlarge`                 |

`bench.sh` runs `aws service-quotas get-service-quota` on launch and warns
if either looks low.

## Usage

```bash
# One instance
./bench.sh g6.xlarge

# Full ladder
./bench.sh --sweep

# Subset of the ladder
./bench.sh --sweep g6.xlarge g6e.xlarge
```

Each instance run takes ~12 min (boot + warmup + measurement + destroy).
The full 7-instance sweep is ~90 min wall time and ~$4 in compute.

## Output: `results.csv`

| column         | meaning                                                       |
|----------------|---------------------------------------------------------------|
| `instance_type`| e.g. `g6.xlarge`                                              |
| `concurrency`  | concurrency level of the best-throughput step                 |
| `batch`        | inputs per request (default 32)                               |
| `target_tokens`| approx token count per input (default 256)                    |
| `emb_per_sec`  | **the number to look at** — sustained embeddings/sec          |
| `req_per_sec`  | sustained requests/sec                                        |
| `p50_ms`, `p99_ms` | per-request latency at the best-throughput step           |
| `errors`       | transport + non-2xx tally during the measurement window       |
| `duration_s`, `warmup_s`, `timestamp` | bookkeeping                            |

Production decision rule: cheapest instance with `emb_per_sec ≥ 1560`
(1,200 × 1.3 headroom for the SLA).

## The ladder

| Instance       | Compute                  | ~$/hr (eu-c-1, OD) | Serving stack      |
|----------------|--------------------------|--------------------|--------------------|
| `c7i.4xlarge`  | 16 vCPU AVX-512, no GPU  | ~$0.85             | TEI (CPU image)    |
| `g4dn.xlarge`  | 1× T4 (Turing)           | ~$0.65             | TEI (`turing-1.9`) |
| `inf2.xlarge`  | 1× Inferentia2 (2 cores) | ~$0.95             | custom FastAPI     |
| `g6.xlarge`    | 1× L4                    | ~$0.95             | TEI (`89-1.9`)     |
| `g5.xlarge`    | 1× A10G                  | ~$1.30             | TEI (`86-1.9`)     |
| `g6e.xlarge`   | 1× L40S                  | ~$2.20             | TEI (`89-1.9`)     |
| `g6.12xlarge`  | 4× L4                    | ~$5.50             | TEI (`89-1.9`)     |

Edit `LADDER` in `bench.sh` and `_INSTANCE_TABLE` in `__main__.py` to add
more.

## Inferentia notes

TEI doesn't ship a Neuron build. For `inf2.xlarge` we:
1. Compile the model on first boot via
   `optimum-cli export neuron --model mxgr/simplesystem-embedding ...`
   (takes ~10 min)
2. Serve via `neuron_server.py` — a thin FastAPI app that exposes
   `POST /embed` with the same wire format TEI uses, so `loadgen.py`
   doesn't notice the difference.

If compile fails (custom op, etc.), the row is written as
`emb_per_sec=0.0, errors=1` and the sweep continues. Check
`/var/log/neuron-compile.err` on the server box before teardown for the
root cause.

## File layout

```
aws_embedding_bench/
├── README.md                  this file
├── Pulumi.yaml                project descriptor (toolchain: uv)
├── pyproject.toml             pulumi + pulumi-aws (uv-managed deps)
├── __main__.py                Pulumi program (server + loadgen EC2s)
├── tei_userdata.sh            EC2 user-data: TEI in Docker + nginx
├── inferentia_userdata.sh     EC2 user-data: neuron compile + FastAPI
├── neuron_server.py           TEI-compat FastAPI for Inferentia
├── loadgen.py                 client-side throughput driver
├── bench.sh                   orchestrator (pulumi up → loadgen → destroy)
└── results.csv                appended-to as runs complete (gitignored)
```

## Caveats / not yet wired up

- **Cloudflare tunnel + Prometheus exporters**: present in
  `vast_embedding.sh` for production, omitted here. The benchmark hits
  TEI on the private VPC IP from the co-located load-gen, so no public
  exposure is needed.
- **Inputs are synthetic**, not real parquet rows. The production
  embedding parquet lives on a separate mount; for throughput
  measurement the token-count distribution is what matters, not the
  literal text. Swap the corpus loader in `loadgen.py` if you want real
  data.
- **Spot pricing**: not used. Benchmark runs are short and the certainty
  of on-demand availability is worth more than the ~70% saving.
