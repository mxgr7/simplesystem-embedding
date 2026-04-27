# Hybrid search v0 — addressing the EAN/MPN weakness

## Goal

Our fine-tuned e5 dense model is strong on free-text product search but weak on EAN/GTIN/MPN identifier queries — an expected limitation of dense models on opaque numeric strings. We want to ship dense to production for an A/B test, with the identifier weakness covered by a second, lexical retrieval path.

## Volume targeted by the fix

From 30 days of `search_performed` (PostHog, 608,804 events):

| Pattern | Events | % of volume | Distinct terms | Routing |
|---|---:|---:|---:|---|
| EAN (`^\d{8}$\|^\d{12,14}$`) | 20,471 | 3.4% | 15,654 | strict path |
| Hyphenated code with digit | 19,998 | 3.3% | 15,761 | strict path |
| Alpha-then-digit (`cr2032`, `rj45`) | 15,400 | 2.5% | 11,505 | strict path |
| **Strict identifier total** | **55,869** | **9.2%** | **42,919** | BM25-only, high limit |
| Ambiguous identifier-shape (digit_other, etc.) | ~30k | ~5% | ~15k | hybrid path |
| Free text | rest | ~85% | rest | hybrid path |

## Architecture

**Two Milvus collections + a regex query classifier that routes to one of two paths.**

```
                                ┌──────────────────┐
                          ┌────→│ classify(query)  │────┐
                          │     └──────────────────┘    │
                          │                             │
                          │                  ┌──────────┴──────────┐
                          │                  ▼                     ▼
                          │       strict identifier?          else (hybrid)
                          │                  │                     │
                          │                  ▼                     ▼
                          │      ┌────────────────────┐  ┌────────────────────┐
                          │      │ offers_codes       │  │ offers_codes (k=20)│
                          │      │ BM25 only          │  │  ─┐                │
                          │      │ limit=500          │  │   │                │
                          │      └────────────────────┘  │ + offers_dense     │
                          │                  │           │   (k=200)          │
                          │                  │           │  ─┘                │
                          │                  │           └────────┬───────────┘
                          │                  │                    │
                          │                  ▼                    ▼
                          │          (no fusion)         client-side RRF (k=60)
                          │                  │                    │
                          │                  └─────┬──────────────┘
                          │                        ▼
                          │             ┌────────────────────┐
                          │             │ result.length == 0?│
                          │             └─────┬──────────────┘
                          └─────yes (fall back)│
                                               ▼ no
                                          top 24 results
```

**Why a separate codes collection:**

- The dense collection is ~650 GB and was expensive to build — we don't want to touch it.
- The codes collection has no embeddings (text + sparse inverted index only). Estimated build: ~1–2 hours.
- The codes ingest pipeline can be iterated independently (re-tokenize, change analyzer, tighten filters) without disturbing dense.
- Failure domains are decoupled.

**Why the strict-path classifier:**

- Pure BM25 with a high limit returns *all* matching offers — popular EANs can have 100+ vendor listings; capping at 20 in hybrid would truncate them.
- Dense ANN over an opaque numeric string is noise, not signal. Skipping it also saves ~10–50 ms per query (no embedding, no ANN).
- Tied scores within a single EAN match are visible, forcing a deliberate decision about secondary sort (price, vendor, freshness) instead of accepting hidden tied-score random ordering inside hybrid.

**Primary key constraint:** `offers_codes.id` MUST equal `offers_dense.id` (use the existing offer `id`). Client-side merge depends on it.

## `offers_codes` schema

```python
schema.add_field("id",          DataType.VARCHAR, max_length=64, is_primary=True)
schema.add_field("text_codes",  DataType.VARCHAR, max_length=512,
                 enable_analyzer=True,
                 analyzer_params={
                     "tokenizer": "whitespace",
                     "filter": [
                         "lowercase",
                         {"type": "length", "min": 4, "max": 40},  # token-level safety net
                     ],
                 })
schema.add_field("sparse_codes", DataType.SPARSE_FLOAT_VECTOR)

schema.add_function(Function(
    name="bm25_codes",
    function_type=FunctionType.BM25,
    input_field_names=["text_codes"],
    output_field_names=["sparse_codes"],
))

idx = client.prepare_index_params()
idx.add_index("sparse_codes", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
client.create_collection("offers_codes", schema=schema, index_params=idx)
```

**Why whitespace tokenizer (not standard):** identifiers must be matched as atomic tokens. `"RJ45"` should match `rj45`, not be split into sub-tokens. Standard tokenizers also split on hyphens (`"TZE-231"` → `["tze", "231"]`), which we don't want.

## `text_codes` ingest filter (CRITICAL)

Source: `/data/offers_embedded_full.parquet/`. The catalog is dirty — naive ingest will pollute the index with high-DF junk that creates false-positive matches.

For each row, build `text_codes` by concatenating the surviving values of `ean`, `manufacturerArticleNumber`, and `article_number`, joined by single spaces. **If all three are dropped by the filters below, skip the row entirely** — a blank row would either index nothing (waste) or accidentally match the empty query.

### Universal denylist (applies to all three fields)

Drop the value if any of:
- empty, or `length < 4`, or `length > 40`
- matches `^0+$` (all zeros — covers `'00000000'` ~49M rows, `'0000000000000'`, etc.)
- matches `^9+$` (all nines — `'9999999999999'` placeholder convention)
- matches `^[-_.\s]+$` (only punctuation/whitespace — `'----'`, `'   '`, etc.)
- in `{"k.a.", "n/a", "n.a.", "null", "aucune donnée", "#ref!"}` (case-insensitive multilingual placeholders observed in the data)
- **contains internal whitespace** (matches `\s`) — real codes are atomic (`rj45`, `4031100000000`, `h07v-k`, `tze-231`); values with spaces are overwhelmingly description fragments (`'DC Brushless Fan'`, `'FACHBÖDEN GELOCHT'`), multi-part SKU concatenations (`'500601 637790 0001'`), brand strings (`'ATLAS Schuhfabrik GmbH & Co. KG'`), or standards refs (`'DIN 1837 A'`). Without this rule, the whitespace tokenizer splits these into short sub-tokens (`1` appears 4.7M times, `100` 1.2M times, `nr:` 1.2M times) that flood the index with low-IDF noise. Drops ~28M field values; products still survive via their other fields.

> We deliberately do **not** drop other repeated-digit values like `'4444'` or `'7777'` — those can be real short SKUs. Only `0` and `9` carry placeholder semantics. The frequency cap below catches any other repeated patterns that turn out to be vendor-specific placeholders.

### Frequency cap (applies to all three fields)

After applying the structural filters above, compute the per-value occurrence count and drop any value appearing more than **500 times**. Critically, **dedupe per row before counting** — when a vendor uses the EAN as the article_number (common for resellers), the same value appears in two columns of the same row; counting it twice falsely inflates frequency for legitimate products.

```sql
WITH per_row AS (
  -- one row per (offer, distinct identifier value) — dedupes within row
  SELECT DISTINCT id, lower(v) AS v FROM (
    SELECT id, ean AS v FROM offers_filtered_ean
    UNION ALL SELECT id, manufacturerArticleNumber FROM offers_filtered_mpn
    UNION ALL SELECT id, article_number FROM offers_filtered_art
  )
),
freq AS (SELECT v, COUNT(*) AS n FROM per_row GROUP BY v)
SELECT v FROM freq WHERE n > 500;  -- ingest denylist
```

**Threshold rationale (from audit):** the per-value occurrence distribution has a sharp gap. The 100–500 range (~75k distinct values) is dominated by *legitimate popular products* — Bosch power tools, DYMO label makers, 3M PPE, Klingspor abrasives — exactly what users search EANs for. Clearly-placeholder territory (`4031100000000`, `atlas schuhfabrik gmbh & co. kg`, `bez údajů`, brand-prefix-then-zeros patterns) lives at >1000 occurrences. A threshold of 500 catches ~117 distinct junk values while preserving popular real products. Do **not** lower it without re-running the audit on `/data/offers_embedded_full.parquet/` — see the "popular products dropped at threshold=100" finding.

### EAN-specific (additional)

- length not in `{8, 12, 13, 14}`

### MPN- and article_number-specific (additional)

- matches `^[a-zäöüß]+$` after lowercasing — pure-letter words like `'magnet'`, `'atlas'`, `'kugelschreiber'`. The catalog has many MPN values that are actually category words; matching them in BM25 would push unrelated products to the top of the fused list whenever a user types one of those words.
- contains `e+` after lowercasing **and** matches `^[\d,.eE+\-]+$` — Excel-mangled scientific notation like `'8,45601E+11'` (EANs that lost precision through Excel; useless as either EAN or MPN).

### Result (measured against `/data/offers_embedded_full.parquet/`)

- **158,269,705 rows kept** (99.37%) — 1,005,569 rows dropped entirely (their only identifiers were whitespace-containing description fragments / multi-part SKUs).
- **Field survival**: 109.2M rows have a usable EAN, 128.5M have a usable MPN (down from 144.3M before the whitespace rule — 15.8M whitespace-containing MPN values dropped), 146.4M have a usable article_number (down from 159.2M — 12.8M whitespace article_numbers dropped).
- **Distinct values indexed**: 15.0M EANs, 22.0M MPNs, 49.6M article_numbers.
- **Frequency cap dropped 81 distinct values** — fewer than before because most multi-word junk was already removed by the whitespace rule upstream.
- **Verified post-filter**: 0 surviving identifier values contain whitespace, so the BM25 index will not be flooded with short sub-tokens like `1`, `100`, `nr:`, `10` (which would otherwise occur 4.7M / 1.2M / 1.2M / 1.0M times respectively).
- Common placeholder queries (`00000000`, `0`, `n/a`, `magnet`) return zero codes hits, contributing zero noise to the fused result.

## Query classifier

```python
import re

# All patterns are matched case-insensitively against the lowercased, trimmed query.
ID_PATTERNS = [
    r"\d{8}",                                  # EAN-8
    r"\d{12,14}",                              # UPC-A / EAN-13 / GTIN-14
    r"(?=.*\d)[a-z0-9]+(?:-[a-z0-9]+)+",       # hyphenated WITH at least one digit
                                               # matches: tze-231, h07v-k, wd-40, 221-413, gtb6-p5211
                                               # rejects: post-it, t-shirt, u-power, o-ringe, uni-ball
    r"[a-z]+\d+[a-z0-9]*",                     # alpha-then-digit
                                               # matches: cr2032, rj45, lr44, ffp2, dtw300, e1987303
]
ID_RE = re.compile("|".join(f"^{p}$" for p in ID_PATTERNS), re.IGNORECASE)

# Length floor on alpha-then-digit prevents trivial 2-3 char matches.
def is_strict_identifier(q: str) -> bool:
    q = q.strip()
    if not (4 <= len(q) <= 40):
        return False
    return bool(ID_RE.fullmatch(q))
```

**Asymmetric error handling.** False positives (free text classified as identifier) skip dense and may return 0 results — caught by the 0-result fallback below. False negatives (real identifier classified as free text) fall through to hybrid, where BM25 still picks them up. So the classifier is **high-precision by design** — only the patterns above, no "maybe identifier" cases. Ambiguous shapes (pure digits 5–7 or 9–11, pure-letter words) intentionally stay in the hybrid path.

## Query path

```python
import asyncio
from collections import defaultdict

def rrf_merge(result_lists, k=60, top_n=24):
    scores = defaultdict(float)
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_n]

async def hybrid_search(query_text: str, query_vec: list[float]):
    dense_task = asyncio.to_thread(
        dense_client.search, "offers_dense",
        data=[query_vec], anns_field="dense", limit=200,
        search_params={"metric_type": "COSINE"})
    codes_task = asyncio.to_thread(
        codes_client.search, "offers_codes",
        data=[query_text.lower()], anns_field="sparse_codes", limit=20,
        search_params={"metric_type": "BM25"})
    dense_hits, codes_hits = await asyncio.gather(dense_task, codes_task)
    return rrf_merge([dense_hits[0], codes_hits[0]])

async def strict_search(query_text: str):
    hits = codes_client.search(
        "offers_codes",
        data=[query_text.lower()], anns_field="sparse_codes",
        limit=500,                                # covers all popular EANs (max ~463)
        search_params={"metric_type": "BM25"})
    return [(h.id, h.score) for h in hits[0]]

async def search(query_text: str, embed_fn):
    if is_strict_identifier(query_text):
        results = await strict_search(query_text)
        if results:
            return results[:24]
        # 0-result fallback: rescues off-catalog EANs, DIN/ISO standard refs
        # like `din912`, typos, etc. Costs latency only on the empty subset.
    query_vec = await embed_fn(query_text)
    return await hybrid_search(query_text, query_vec)
```

**Defer embedding until after the classifier** — strict-path queries skip embedding entirely, saving ~10–50 ms per query.

**Tied scores in `strict_search`.** When a query matches an EAN with N vendor listings, BM25 returns all N with identical scores. Order within a tied group is implementation-defined; the response layer must decide a secondary sort (price ascending, vendor preference, listing freshness) — see "Out of scope".

## Parameter choices (and rationale)

### Strict path (BM25-only)

| Param | Value | Rationale |
|---|---|---|
| `codes.limit` | 500 | Covers the top of the catalog distribution: max real popular EAN has 463 vendor listings (Paperflow, Bosch power tools, etc.). All matches returned, no truncation. |

### Hybrid path

| Param | Value | Rationale |
|---|---|---|
| `dense.limit` | 200 | ~5–10× page size — gives RRF enough candidate pool |
| `codes.limit` | 20 | The strict path handles EAN/code queries; here codes only sees ambiguous-shape queries where match counts are small (mostly 0–5). Worst-case fused rank ~28. |
| RRF `k` | 60 | Standard. Codes hit at rank 1 contributes `1/61` — ties with dense rank 1, naturally placing exact-identifier matches near top of fused list. |

### Both paths

| Param | Value | Rationale |
|---|---|---|
| Final `top_n` | 24 | Match existing page size |
| Strict-path 0-result fallback | enabled | Rescues off-catalog EANs (~17% of EAN-shape queries don't match catalog), DIN/ISO standard references like `din912` not stored as codes, and rare typos. |

## What NOT to do

- **Don't put codes into the dense collection's text field with a standard analyzer.** It re-introduces the pure-word pollution problem and prevents independent iteration.
- **Don't broaden the classifier to ambiguous shapes** like pure digits 5–7 (`114150`), pure-letter words (`atlas`), or hyphenated words without digits (`post-it`, `t-shirt`, `o-ringe`). The current patterns were validated against 30 days of PostHog data — see "Volume".
- **Don't add a `sparse_main` (BM25 over title/description) right now.** The e5 model is qualitatively good on free text; adding BM25 there solves a problem we don't have. Keep the path additive: ship dense + codes, add `sparse_main` later only if the A/B reveals specific lexical-failure modes.
- **Don't use Milvus's `hybrid_search`.** It only works within one collection. Two collections + client-side RRF is the whole point of this design.
- **Don't skip the 0-result fallback.** It's the safety valve for classifier false positives and off-catalog identifiers.

## Validation before shipping

1. **Classifier precision spot-check.** Run the classifier on the 200 most-frequent PostHog queries (past 30 days). For each query flagged as strict-identifier, manually verify it really is one. Target: 100% precision on the head distribution. Adjust patterns if any free-text terms slip through.
2. **Codes recall@5 spot-check** on a held-out set of 100 known EAN/MPN queries — confirm the right product surfaces from the strict path.
3. **Free-text regression check** on a held-out set of 100 free-text queries — confirm the classifier rejects them, codes contributes 0 (or near-0) in hybrid, and the fused result equals the dense-only result.
4. **0-result fallback check** — verify that `din912`, an off-catalog EAN, and a typo'd identifier all produce results (via fallback to hybrid) rather than empty pages.
5. **Top-of-fused-page noise check** on the hybrid path for ambiguous-shape queries — manually inspect top 10 results for 50 such queries; flag any cases where a polluted MPN slipped past the filter and pushed real hits down.
6. **Latency**: strict-path target is BM25-only round-trip; hybrid-path target is `max(dense_lat, codes_lat) + RRF`. Both should be under existing SLO.

## Things to monitor in the A/B

- **Strict-path classification rate** — should be ~9% of queries; large deviations suggest the regex is over- or under-firing.
- **0-result rate per path:**
  - Strict path: was 26.6% for EAN-shape queries pre-fix. Should drop sharply (the 17% off-catalog cases get rescued by fallback; the 62% EAN-match cases now resolve cleanly).
  - Hybrid path: should be unchanged from baseline.
- **CTR on strict-path queries** — primary success metric for this change.
- **CTR on free-text queries** — should be unchanged (regression guard).
- **Strict-path fallback rate** — fraction of strict-classified queries hitting the empty-result fallback. High rate (> ~25%) means the classifier is catching too many off-catalog identifiers; investigate.
- **Distribution of which collection contributed the clicked result in hybrid** — if codes never wins on any free-text query, ingest filter is working; if it sometimes does, investigate.

## Out of scope (deliberately)

- Reindexing the dense collection
- Cross-encoder reranking
- Query rewriting / synonyms / typo tolerance
- A `sparse_main` BM25 over product titles/descriptions
- Per-source weight tuning (`WeightedRanker`) — start with parameter-free RRF, tune after data
- **Secondary sort within tied BM25 scores** (price, vendor reputation, freshness). When the strict path matches a popular EAN with N vendor listings, all N tie on BM25 score. v0 returns them in implementation-defined order; the response/UI layer should pick a sort key. If a sort key isn't available there, this becomes a real product question to resolve before launch — flag separately, not in this PR.
