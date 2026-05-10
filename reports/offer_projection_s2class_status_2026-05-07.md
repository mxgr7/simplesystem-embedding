# Offer projection / S2CLASS status — 2026-05-07

## Summary
We benchmarked the DuckDB-based offers/article projection pipeline, aligned `S2CLASS` derivation with legacy behavior, optimized the default DuckDB S2 path, and added an alternative dedup-based offer projection builder for A/B comparison.

## Pipeline findings
- Main SQL lives in `indexer/duckdb_projection.py`.
- Flow: `flat` -> `projected` -> `finalized` -> `articles` / `offers`.
- Estimated full offers runtime:
  - `json.gz -> parquet`: ~15 min
  - `parquet -> offer_projected`: ~74–76 min
  - combined: ~89–91 min
- Multi-file parquet output improved conversion throughput materially, but offer-only projection remained CPU-underutilized.

## Legacy S2CLASS alignment
Legacy behavior is:
- ignore source-provided `S2CLASS`
- choose the highest available non-`S2CLASS` eclass version
- map only that version
- if that chosen version has no mapping hit, fall back to `90909090`

Changes made:
- expanded supported mapping versions from only `ECLASS_5_1` / `ECLASS_8` to `ECLASS_5_1`, `6`, `7_1`, `8` … `16`
- copied mapping files into `indexer/classification_mapping/`
- updated Python and DuckDB logic to match legacy fallback semantics

Commit:
- `f818787` — `Align S2CLASS derivation with legacy mapping`

## Default DuckDB S2 optimization
Changed the offer projection from a repeated-branch correlated-subquery pattern to:
- extract eclass arrays once
- select the winning source version once
- do one lateral join to `s2map`
- `COALESCE` to the default expanded S2 hierarchy

Quick synthetic benchmark (~2M rows):
- before: ~32–34s
- after: ~10–13s

Commit:
- `4a9cb71` — `Optimize DuckDB S2CLASS offer projection`

## Alternative dedup-based builder
Added a second builder that keeps the original path intact for comparison.

New pieces:
- `indexer.duckdb_projection.offer_projected_build_sql_dedup_eclass(...)`
- `scripts/build_offer_projected_dedup_eclass.py`

What it does:
- computes `to_json(offerParams.eclassGroups)`
- deduplicates identical `eclassGroups`
- derives `eclass5_code`, `eclass7_code`, `s2class_code` once per unique blob
- joins the derived classification arrays back to the full offer stream

Quick synthetic benchmark (~200k rows, 167 unique `eclassGroups`):
- existing builder: ~1.0–1.3s
- dedup builder: ~0.29s
- speedup: ~3.6x–4.5x

Validation:
- dedup builder matched the default builder on the 200-row fixture seed

Commit:
- `f5b0e77` — `Add deduped eclass offer projection builder`

## Key files
- `indexer/duckdb_projection.py`
- `indexer/s2class_mapper.py`
- `scripts/build_offer_projected.py`
- `scripts/build_offer_projected_dedup_eclass.py`
- `tests/test_projection.py`
- `tests/test_duckdb_offer_projected_parity.py`

## Recommended next step
Run the original and dedup builders on the same real parquet sample/chunks and compare:
- wall time
- CPU utilization
- output size
- number of unique `eclassGroups` per chunk
