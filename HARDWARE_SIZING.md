# Hardware sizing & cloud cost analysis

Snapshot of the current single-box deployment (Milvus standalone + MinIO + etcd + TEI CPU embedder + FastAPI playground), translated into minimum and comfortable hardware budgets and then priced on AWS and Hetzner under a weekday-business-hours schedule.

## 1. Current resident footprint

Measured on the existing box (184 GiB RAM, 48 cores, 902 GB root NVMe + 512 GB attached volume).

| Component              | RAM (RSS) | Disk                                   |
|------------------------|-----------|----------------------------------------|
| Milvus standalone      | ~146 GB   | `docker_volumes/milvus`: 160 GB        |
| MinIO (segment store)  | ~1.3 GB   | `docker_volumes/minio`: 228 GB         |
| etcd                   | ~65 MB    | `docker_volumes/etcd`: <1 MB           |
| TEI (CPU, ST model)    | ~12 GB    | model dir: 1.1 GB                      |
| uvicorn playground     | ~170 MB   | negligible                             |
| **Total**              | **~160 GB** | **~390 GB**                          |

Milvus anon heap is ~73 GB; the other ~74 GB of its RSS is mmap'd index pages — reclaimable under memory pressure but search latency degrades when they page out.

## 2. Hardware requirements

| Resource | Bare minimum                              | Comfortable buffer                       |
|----------|-------------------------------------------|------------------------------------------|
| CPU      | 8 cores / 16 threads, AVX2                | 16–24 cores                              |
| RAM      | 96 GB                                     | 192 GB (256 GB if corpus grows >50%)     |
| Disk     | 512 GB NVMe SSD                           | 1–2 TB NVMe SSD                          |
| Network  | 1 Gbps                                    | 1–10 Gbps                                |
| Notes    | mmap will page under load; no swap        | Full index stays hot in page cache       |

HDDs are not viable — Milvus mmaps index segments, so the storage layer must be NVMe.

## 3. Schedule assumption

Mon–Fri, 07:00–20:00 CET, automated stop/delete during off-hours.

- 13 h/day × 5 days = 65 h/week
- 65 h × 52 weeks / 12 months ≈ **281.67 active h/month**
- Idle hours saved: 730 − 281.67 ≈ **448 h/month (61 %)**

All prices below are net (excl. VAT), April 2026 list rates, EUR conversion at €1 ≈ $1.08.

## 4. AWS (eu-central-1, Frankfurt)

Strategy: **stop** the instance during off-hours (compute billing pauses; EBS keeps billing 24/7). Termination is avoided so the EBS volume persists without snapshot-restore ceremony.

| Tier        | Instance       | vCPU | RAM    | On-demand / h | Compute (281.67 h) | EBS gp3             | **Monthly total** |
|-------------|----------------|------|--------|---------------|--------------------|---------------------|-------------------|
| Minimum     | r6i.4xlarge    | 16   | 128 GB | ~$1.10        | ~$310              | 512 GB → ~$49       | **~$359 (~€332)** |
| Comfortable | r6i.8xlarge    | 32   | 256 GB | ~$2.20        | ~$620              | 1 TB → ~$97         | **~$717 (~€664)** |

Notes:
- r7i is ~5 % more expensive for modest gains; r5 is ~5 % cheaper on older Ice Lake.
- 3-year Savings Plan cuts compute ~40 %, but the scheduled-off-hours approach already avoids 61 % of runtime — Savings Plans stack poorly with shutdown schedules.
- Egress not included: $0.09/GB after 100 GB/month free tier.
- An always-on equivalent would be ~$800 / ~$1 600 per month, so shutdown saves roughly the same 55–60 %.

## 5. Hetzner

Two practical paths. Cloud bills hourly and lets you realise the shutdown savings; dedicated (AX line) is flat monthly and ignores the schedule.

### 5a. Hetzner Cloud (CCX dedicated vCPU, hourly, delete + restore from snapshot)

Important: a *stopped* Hetzner Cloud server still bills. To realise shutdown savings you must **delete** the server and recreate it from a snapshot each morning (automatable via `hcloud` CLI + systemd timer). Snapshot storage bills €0.012/GB/month.

| Tier        | Plan   | vCPU | RAM    | Local NVMe | Hourly   | Compute (281.67 h) | Snapshot (~400 GB) | **Monthly total** |
|-------------|--------|------|--------|------------|----------|--------------------|--------------------|-------------------|
| Minimum     | CCX53  | 32   | 128 GB | 600 GB     | €0.2873  | €81                | ~€5                | **~€86**          |
| Comfortable | CCX63  | 48   | 192 GB | 950 GB     | €0.4165  | €117               | ~€5                | **~€122**         |

Snapshot/restore adds ~3–5 min at start-up and a couple of minutes at shutdown; acceptable for a 07:00 start. The monthly cap (price ceiling for 24/7 use) would be €172 / €250 — so the schedule saves ~50 %.

### 5b. Hetzner dedicated (AX line, flat monthly, schedule gives no savings)

Dedicated servers bill by the month regardless of power state — listed for completeness, since they're often cheaper than Cloud at 24/7 usage.

| Tier        | Server                            | Cores / CPU              | RAM    | Disk                  | **Monthly** |
|-------------|-----------------------------------|--------------------------|--------|-----------------------|-------------|
| Minimum     | AX102                             | 16C Ryzen 9 7950X3D      | 128 GB | 2 × 1.92 TB NVMe      | **~€129**   |
| Comfortable | AX162-R (custom) or auction 256 GB| 24–48C EPYC / Xeon       | 192–256 GB | 2 × 1.92 TB NVMe | **~€180–230** |

Setup fee €99–129 applies on new orders; waived on auction listings.

## 6. Headline comparison

281.67 active hours/month, storage sized to match requirements, EUR net:

| Tier        | AWS (schedule) | Hetzner Cloud (schedule) | Hetzner dedicated (24/7) |
|-------------|----------------|--------------------------|---------------------------|
| Minimum     | ~€332          | **~€86**                 | ~€129                     |
| Comfortable | ~€664          | **~€122**                | ~€180–230                 |

Hetzner Cloud with scripted delete+snapshot is ~**4× cheaper** than AWS at both tiers. If the delete-and-restore dance is unwelcome, a flat-rate AX102 (~€129) still beats scheduled AWS minimum and removes all state-management complexity.

## 7. Caveats

- Prices change; rerun the math before committing.
- AWS r6i/r7i assume default EBS optimisation; add a few % for gp3 IOPS beyond 3 000 baseline if Milvus compaction needs it.
- Hetzner Cloud snapshot restore throughput is ~100 MB/s — 400 GB ≈ 70 min worst case; in practice Milvus warm-up dominates. Test the timing before relying on a 07:00 hard start.
- If the corpus grows, RAM is the binding constraint; upgrade the tier before the p99 latency collapses from mmap paging.
