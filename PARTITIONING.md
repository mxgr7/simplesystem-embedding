# Partitioning analysis — offers_flat.parquet

Dataset: `/mnt/HC_Volume_105463954/simplesystem/data/offers_flat.parquet/`
(16 existing buckets `bucket=00..15.parquet`, ~9.95M rows each, ~159.3M offer rows total)

## ID semantics

Both `vendor_id` and `catalog_version_id` are **UUID v4** (36-char, version nibble = `4` in 100% of samples, variant nibbles `8/9/a/b` uniformly distributed). No timestamp, no MAC, no sequence — pure random bits. Any hex prefix is already a perfect uniform hash.

## Functional dependency

`catalog_version_id → vendor_id` holds.

- 4,953 distinct catalog_version_ids
- 4,953 distinct (vendor_id, catalog_version_id) pairs
- 738 distinct vendor_ids (avg ~6.7 catalog versions per vendor)
- 0 catalog_version_ids mapping to >1 vendor

## Row distribution per catalog_version_id

Strong right-skew (exploded offer×cv rows, total 521.8M).

| stat | offers/cv |
|------|-----------|
| min | 1 |
| p10 | 67 |
| p25 | 2,369 |
| median | 28,793 |
| mean | 105,343 |
| p75 | 110,718 |
| p90 | 256,161 |
| p99 | 1,067,346 |
| max | 1,663,653 |
| stddev | 197,712 |

- 63 catalog versions have ≥1M offers each (~74M rows, ~14% of explosion)
- 171 catalog versions have <10 offers
- Top ~10 heaviest cv's belong to just ~5 mega-vendors (`fa3dd87c-…` alone owns 5 of them)

Direct value-level partitioning by `catalog_version_id` is unusable: file sizes would range 1 → 1.66M rows.

## Hash-prefix bucketing

### 2-hex prefix of `id` (offer-id), 256 buckets, 159.3M rows

| stat | rows/bucket |
|------|-------------|
| min | 620,016 |
| max | 624,400 |
| mean | 622,169 |
| stddev | 793 |
| max/min | **1.007×** |

Effectively perfect balance. Recommended for offer-id access and parallel scans.

### 2-hex prefix of `catalog_version_id`, 256 buckets, 521.8M exploded rows

| stat | rows/bucket |
|------|-------------|
| min | 276,700 (`31`) |
| p5 | 630,025 |
| median | 1,842,742 |
| mean | 2,038,141 |
| p95 | 4,035,068 |
| max | 5,382,818 (`c7`) |
| stddev | 1,033,856 |
| max/min | **19.4×** |
| max/mean | **2.6×** |

Moderately skewed. Root cause is the per-cv heavy tail — a mega-cv (up to 1.66M rows) fully lands in a single prefix bucket, so an unlucky prefix absorbs 3× the mean. Going to 3-hex / 4096 buckets worsens it; salting or sub-splitting the heavy cv's would be needed.

## Recommendations

- **Partition by `id` prefix** if queries are offer-id lookups or fully parallel scans — free, near-perfect balance.
- **Partition by `catalog_version_id` or `vendor_id` prefix** only if queries routinely filter by those keys. Accept ~20× bucket imbalance, or mitigate by splitting heavy cv's into multiple sub-files.
- **Do not partition by UUID value** (one file per cv/vendor) — distribution is too long-tailed.
- **Do not expect time-based partitioning** from any UUID here — v4 carries no timestamp.
