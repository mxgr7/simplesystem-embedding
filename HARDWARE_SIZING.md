# Hardware sizing & cloud cost analysis

Single-box deployment: Milvus standalone + MinIO + etcd + TEI CPU embedder + FastAPI playground.

## Current resident footprint

Measured on the existing box (184 GiB RAM, 48 cores).

| Component          | RAM (RSS) | Disk                              |
|--------------------|-----------|-----------------------------------|
| Milvus standalone  | ~146 GB   | 160 GB                            |
| MinIO              | ~1.3 GB   | 228 GB                            |
| TEI (CPU)          | ~12 GB    | 1.1 GB (model)                    |
| etcd + playground  | <250 MB   | negligible                        |
| **Total**          | **~160 GB** | **~390 GB**                     |

Milvus anon heap is ~73 GB; the other ~74 GB is mmap'd index pages — reclaimable, but search latency degrades when they page out.

## Hardware requirements

| Resource | Bare minimum                       | Comfortable buffer                  |
|----------|------------------------------------|-------------------------------------|
| CPU      | 8 cores / 16 threads, AVX2         | 16–24 cores                         |
| RAM      | 96 GB                              | 192 GB                              |
| Disk     | 512 GB NVMe SSD                    | 1–2 TB NVMe SSD                     |
| Notes    | mmap pages under load; no swap     | full index stays hot in page cache  |

HDDs are not viable — Milvus mmaps index segments.

## Cost analysis

Schedule: Mon–Fri 07:00–20:00 CET with automated stop/delete off-hours ≈ **282 active h/month** (61 % idle). Prices are net, April 2026 list rates, €1 ≈ $1.08.

### AWS (eu-central-1, stop during off-hours)

Instance is stopped off-hours so compute pauses; EBS keeps billing 24/7.

| Tier        | Instance     | vCPU | RAM    | $/h   | Compute (282 h) | EBS gp3       | **Monthly**       |
|-------------|--------------|------|--------|-------|-----------------|---------------|-------------------|
| Minimum     | r6i.4xlarge  | 16   | 128 GB | 1.10  | ~$310           | 512 GB → ~$49 | **~$359 (~€332)** |
| Comfortable | r6i.8xlarge  | 32   | 256 GB | 2.20  | ~$620           | 1 TB → ~$97   | **~$717 (~€664)** |

### Hetzner Cloud (CCX, delete + restore from snapshot)

A *stopped* Hetzner Cloud server still bills. To realise shutdown savings, **delete** the server nightly and recreate from snapshot at 07:00 (scriptable via `hcloud` CLI). Snapshot storage: €0.012/GB/month.

| Tier        | Plan   | vCPU | RAM    | NVMe   | €/h    | Compute (282 h) | Snapshot (~400 GB) | **Monthly** |
|-------------|--------|------|--------|--------|--------|-----------------|--------------------|-------------|
| Minimum     | CCX53  | 32   | 128 GB | 600 GB | 0.2873 | €81             | ~€5                | **~€86**    |
| Comfortable | CCX63  | 48   | 192 GB | 950 GB | 0.4165 | €117            | ~€5                | **~€122**   |

Restore adds ~3–5 min at start; plan for a 06:55 cron.

### Headline comparison

| Tier        | AWS   | Hetzner Cloud |
|-------------|-------|---------------|
| Minimum     | ~€332 | **~€86**      |
| Comfortable | ~€664 | **~€122**     |

Hetzner Cloud with scripted delete+snapshot is ~4× cheaper at both tiers.

## Caveats

- Prices change — rerun the math before committing.
- Hetzner snapshot restore ~100 MB/s; 400 GB is ~70 min worst case. Test before relying on a hard 07:00 start.
- RAM is the binding constraint as the corpus grows; upgrade before p99 collapses from mmap paging.
