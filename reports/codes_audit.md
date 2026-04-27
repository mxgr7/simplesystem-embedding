# offers_codes — staging audit

- Source: `/data/datasets/offers_embedded_full.parquet`
- Output: `/data/datasets/offers_codes_staging.parquet/bucket=NN.parquet` (16 files)
- Frequency cap: > 500 occurrences
- Total wall time: 232.3s

## Row-level survival

- rows in source: 159,275,274
- rows kept (≥1 surviving identifier): 158,269,705
- rows skipped (all three dropped): 1,005,569

## Per-field survival (counted on kept rows; one row may contribute to multiple fields)

- rows with surviving EAN: 109,171,626
- rows with surviving MPN: 128,544,372
- rows with surviving article_number: 146,431,835

## Distinct values indexed

- distinct EAN values: 14,975,546
- distinct MPN values: 22,011,435
- distinct article_number values: 49,584,819

## Frequency-cap denylist

- size: 81
- top 30 by occurrence count:

  | value | count |
  |---|---:|
  | `4031100000000` | 19,355 |
  | `4015450000000` | 8,164 |
  | `4547360000000` | 8,017 |
  | `3413530000000` | 5,154 |
  | `3413520000000` | 5,000 |
  | `4047322397339` | 3,449 |
  | `3665350000000` | 3,272 |
  | `4050362046105` | 2,507 |
  | `3665788058918` | 2,139 |
  | `4960999904580` | 2,014 |
  | `3701300000000` | 1,459 |
  | `4050362046167` | 1,308 |
  | `4007220000000` | 1,175 |
  | `1463250-c` | 1,162 |
  | `1463770-c` | 1,094 |
  | `1463475-c` | 1,090 |
  | `4056710000000` | 1,089 |
  | `1463495-c` | 1,073 |
  | `1461995-c` | 1,072 |
  | `4010890000000` | 1,062 |
  | `4013902050253` | 945 |
  | `4013290000000` | 872 |
  | `10010` | 846 |
  | `ölflex-classic-fd810` | 827 |
  | `180704` | 812 |
  | `10020` | 772 |
  | `4054770000000` | 770 |
  | `l9606` | 755 |
  | `180702` | 717 |
  | `4014860000000` | 707 |

## Sanity probes (BM25 query → match in materialised text_codes)

  | query | rows containing as token |
  |---|---:|
  | `00000000` | 0 |
  | `4031100000000` | 0 |
  | `n/a` | 0 |
  | `magnet` | 0 |
