# F6 — German identifier tokenization + classifier hardening

**Category**: ftsearch (`./search-api/`)
**Depends on**: —
**Unblocks**: better quality on `STANDARD` queries; precondition for retiring legacy

References: spec §2.1 (consequences), §4.1, §4.5.

**Legacy reference** (next-gen): existing analyzer config in `article/search/commons/src/main/resources/es/settings-articles.json:12-139` (snowball German2 stemmer, `german_text` word-delimiter + edge-ngram, decompounder dict at `analysis/dictionary-de.txt`). We do NOT port the decompounder — semantic search handles compounds. We DO mirror the `article_number_normalized_*` pattern-replace + n-gram for SKU/EAN partial matches.

## Scope

`STANDARD` is the only mode now (§2.1), so identifier-vs-text routing happens entirely inside ftsearch's classifier. Tighten the classifier so it reliably catches numeric EANs and SKU-style tokens against a German query corpus, and bring the BM25 leg's tokenization closer to legacy's `article_number_normalized_*` analyzer so partial matches keep working.

## In scope

- **Classifier hardening** in `search-api/hybrid.py`:
  - Audit `is_strict_identifier` against the captured PostHog query corpus (`scripts/fetch_posthog_search_queries.py`).
  - **Derive the `GENERIC_TOKENS` denylist empirically from PostHog**: take the most-frequent `STANDARD` queries that currently misroute as strict identifiers, manually triage, and check in the resulting word list as data (e.g. `search-api/data/german_generic_tokens.txt`). Document the curation procedure so it can be re-run when the corpus drifts.
  - Confirm classifier behaviour on the strict-identifier examples called out in the spec (EANs, vendor SKUs, article numbers).
- **Identifier tokenization** for the BM25 leg:
  - Apply pattern-replace + n-gram tokenization equivalent to legacy's `article_number_normalized_*` analyzer before BM25 search. This means the codes ingestion pipeline (the `offers_codes` collection's text field) needs the same tokenisation applied at index time, AND the query side needs the same normalisation.
  - Match on partial article numbers (e.g. searching `"4006381333"` should still hit `"4006381333931"`).
  - Decide where the tokenisation lives: as a pre-processing step in `_bm25_search` for queries (yes), and as a step in the codes import pipeline (`scripts/build_offers_codes_staging.py`) for documents.
- **German compound decompounding**: NOT in scope. Decompounding is left to the dense semantic-search leg (the embedding model already absorbs compounds — `useful-cub-58` covers "Akkubohrmaschine" ↔ "Akku Bohrmaschine"). Do not port the legacy Lucene decompounder.
- Reindex of `offers_codes` after the new tokenisation lands; coordinate with operations.
- Tests:
  - Classifier unit tests over a representative corpus (positive + negative cases).
  - BM25 path tests showing partial-EAN and partial-SKU queries return the expected hit, before and after the change.

## Out of scope

- Replacing the embedding model itself — `useful-cub-58` already handles most free-text German.
- Customer-article-number search — dropped per §2.1.
- Changes to the dense leg.

## Deliverables

- Updated classifier in `search-api/hybrid.py`.
- Updated `_bm25_search` (or a wrapper) applying identifier tokenisation at query time.
- Updated codes-staging script applying the same tokenisation at index time.
- Reindex of the `offers_codes` collection (operational change, sequenced).
- Tests as above.

## Acceptance

- Classifier no longer routes the documented German generic tokens (curated from PostHog) to strict.
- Numeric-EAN and hyphenated-SKU queries land on the strict path with a high hit rate.
- No regression on the existing classifier corpus.

## Open questions for this packet

(none — denylist source confirmed (PostHog), decompounding explicitly out of scope.)
