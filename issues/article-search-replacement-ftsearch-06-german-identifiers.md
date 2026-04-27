# F6 — German identifier tokenization + classifier hardening

**Category**: ftsearch (`./search-api/`)
**Depends on**: —
**Unblocks**: better quality on `STANDARD` queries; precondition for retiring legacy

References: spec §2.1 (consequences), §4.1, §4.5.

## Scope

`STANDARD` is the only mode now (§2.1), so identifier-vs-text routing happens entirely inside ftsearch's classifier. Tighten the classifier so it reliably catches numeric EANs and SKU-style tokens against a German query corpus, and bring the BM25 leg's tokenization closer to legacy's `article_number_normalized_*` analyzer so partial matches keep working.

## In scope

- **Classifier hardening** in `search-api/hybrid.py`:
  - Audit `is_strict_identifier` against a German query corpus (mine logs or use the captured PostHog dataset under `scripts/fetch_posthog_search_queries.py`).
  - Extend the `GENERIC_TOKENS` denylist with German-specific generic tokens that pass the shape check today (e.g. "Bohrer", "Schrauben", "Kabel", "Klemme", … — empirically grounded).
  - Confirm classifier behaviour on the strict-identifier examples called out in the spec (EANs, vendor SKUs, article numbers).
- **Identifier tokenization** for the BM25 leg:
  - Apply pattern-replace + n-gram tokenization equivalent to legacy's `article_number_normalized_*` analyzer before BM25 search. This means the codes ingestion pipeline (the `offers_codes` collection's text field) needs the same tokenisation applied at index time, AND the query side needs the same normalisation.
  - Match on partial article numbers (e.g. searching `"4006381333"` should still hit `"4006381333931"`).
  - Decide where the tokenisation lives: as a pre-processing step in `_bm25_search` for queries (yes), and as a step in the codes import pipeline (`scripts/build_offers_codes_staging.py`) for documents.
- **German compound decompounding** for text-classified queries:
  - Decompound German compounds before passing to the BM25 leg so that "Akkubohrmaschine" matches articles indexed as "Akku Bohrmaschine".
  - Pick a library or a precomputed compound dictionary (do not roll your own); document the choice and the failure mode for OOV compounds.
  - Apply on both sides — at codes ingestion and at query time — for consistency.
- Reindex of `offers_codes` after the new tokenisation lands; coordinate with operations.
- Tests:
  - Classifier unit tests over a representative corpus (positive + negative cases).
  - BM25 path tests showing partial-EAN and partial-SKU queries return the expected hit, before and after the change.
  - Compound decompounding tests on a small German vocab.

## Out of scope

- Replacing the embedding model itself — `useful-cub-58` already handles most free-text German.
- Customer-article-number search — dropped per §2.1.
- Changes to the dense leg.

## Deliverables

- Updated classifier in `search-api/hybrid.py`.
- Updated `_bm25_search` (or a wrapper) applying identifier tokenisation at query time.
- Updated codes-staging script applying the same tokenisation at index time.
- Compound-decompounding integration in both ingestion and query.
- Reindex of the `offers_codes` collection (operational change, sequenced).
- Tests as above.

## Acceptance

- Classifier no longer routes the documented German generic tokens to strict.
- Numeric-EAN and hyphenated-SKU queries land on the strict path with a high hit rate.
- A documented set of German compound queries that previously missed now hit.
- No regression on the existing classifier corpus.

## Open questions for this packet

- Compound-decompounding library: pick one (e.g. `compound-split`, `german-compound-splitter`, custom dictionary). Document the choice and accept that imperfect splits will remain.
- Whether to apply the new analyzer to the existing dense vector pipeline too — recommendation is no, the embedding model already absorbs compounds.
