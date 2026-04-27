# F5 — Aggregations / summaries

**Category**: ftsearch (`./search-api/`)
**Depends on**: F1, F2, F3 (and F4 for the `SUMMARIES_ONLY` / `BOTH` mode plumbing)
**Unblocks**: A3, A6

References: spec §3 (response shape), §4.4.

## Scope

Compute every summary kind §4.4 lists over the full filtered hit set, returning them in the response shape the ACL will pass through.

The 10 summary kinds: `VENDORS`, `MANUFACTURERS`, `FEATURES`, `PRICES`, `CATEGORIES`, `ECLASS5`, `ECLASS7`, `S2CLASS`, `PLATFORM_CATEGORIES`, `ECLASS5SET`.

## In scope

- Each kind, per the §4.4 table:
  - **`VENDORS`** — Milvus `group_by_field=vendor_id` over the filtered hit-set, or post-aggregate in Python after fetching `vendor_id` for every hit.
  - **`MANUFACTURERS`** — same on `manufacturerName`.
  - **`FEATURES`** — fetch `features` for every hit, count per `name=value` token, group by `name`, list values per name with counts. The `name=value` separator is `=`.
  - **`PRICES`** — for each hit, run the F3 price-resolution module under request `currency` × `sourcePriceListIds`, take min/max, group by currency. Returns one entry per request currency in practice (the array shape is preserved on the wire).
  - **`CATEGORIES`** — hierarchical. Given `currentCategoryPathElements` (the level the user is browsing), compute:
    - `sameLevel`: counts grouped by paths at the same depth as `currentCategoryPathElements`.
    - `children`: counts grouped by paths one level deeper.
    Use `category_l1..l5` with the `¦` separator.
  - **`ECLASS5`** / **`ECLASS7`** / **`S2CLASS`** — same hierarchical pattern on the eClass code tree (`selectedEClassGroup`, `sameLevel`, `children`). Implementation needs a static eClass hierarchy lookup so that "siblings" and "children" can be derived from the integer codes; load this from a checked-in JSON or YAML, NOT MongoDB.
  - **`PLATFORM_CATEGORIES`** — alias of `CATEGORIES`, OR `S2CLASS` when the request flag `s2ClassForProductCategories: true`.
  - **`ECLASS5SET`** — for each `eClassesAggregations[]` entry on the request, count hits whose `eclass5_code ∈ entry.eClasses`; return `[{id, count}, …]`.
- Aggregations run over the *full filtered hit set* (not just the page). Use the F4 hitcount-style pass (filter expr + output fields = the columns each summary needs) with the same safety cap as `hitCount`. Document where summaries are clipped at the cap.
- Mode interaction:
  - `HITS_ONLY` — skip aggregations entirely; `summaries` is empty/omitted in the response (define which — recommendation: omit, matches legacy).
  - `BOTH` — compute aggregations AND paginate articles.
  - `SUMMARIES_ONLY` — compute aggregations, skip article hydration.
- Wiring: aggregations consume the request's `summaries: [...]` list — only compute the kinds requested.

## Out of scope

- Hierarchy data ingestion — eClass / S2Class trees are checked in as static reference data in this packet. If they need to come from MongoDB, that's a separate workstream and should be flagged.
- Article hydration for `BOTH` (already covered by F4).

## Deliverables

- Aggregations module under `search-api/`.
- Static eClass / S2Class hierarchy reference data checked in (or wired from `configs/`).
- Tests for each summary kind on a fixture: shape, counts, and that `sum(group_counts) ≤ metadata.hitCount`.
- Tests for hierarchical summaries that exercise sameLevel/children logic at each depth.

## Acceptance

- Each of the 10 summary kinds returns the response shape from §3.
- Counts are internally consistent: `Σ group counts ≤ metadata.hitCount`.
- `PLATFORM_CATEGORIES` aliases correctly under both `s2ClassForProductCategories` settings.
- Aggregations honour the request `summaries` list — kinds not requested are not computed.

## Open questions for this packet

- eClass / S2Class hierarchy source: confirm a static checked-in tree is acceptable (it doesn't change often). If it's tenant-overrideable, this packet doesn't cover that — flag a deviation.
- Milvus `group_by_field` capability for the version in use: confirm before relying on it; fall back to "fetch + Python count" if not.
