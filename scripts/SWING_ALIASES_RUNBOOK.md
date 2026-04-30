# Alias-swing operator runbook

Operating `scripts/swing_aliases.py` — the F9 paired alias-swing CLI
that atomically points the public `articles` + `offers` aliases at a
freshly-built collection pair.

For the conceptual "why two aliases" / "why swing in this order"
material, see `MILVUS_ALIAS_WORKFLOW.md`. This doc is the operator-
facing reference: pre-flight, invocation recipes, validation knobs,
rollback, failure modes.

## Quick reference

| Concern | Where |
| --- | --- |
| Code | `scripts/swing_aliases.py` |
| Conceptual workflow | `scripts/MILVUS_ALIAS_WORKFLOW.md` |
| F9 packet | `issues/article-search-replacement-ftsearch-09-article-dedup.md` |
| Test surface | `tests/test_swing_aliases.py` (5 tests, live Milvus) |

## What the script does

Atomically swings two Milvus aliases (`articles` + `offers`) to a
fresh collection pair. The swing sequence is **articles first, offers
last** — so a partial-failure window leaves prior offer rows readable
against the new articles, instead of orphaning offer rows pointing at
no-longer-aliased article hashes.

Pre-flight validation catches the two failure modes a manual swing
typically misses:

  - **Half-populated bulk runs**: row count below
    `--min-rows-articles` / `--min-rows-offers` (defaults `1`,
    operators should bump for production).
  - **Join-key drift**: 200 random offers sampled, their
    `article_hash` resolved against the target articles collection.
    Catches the F9 worst-case bug where bulk articles + offers were
    written from different DuckDB stream invocations and the hashes
    drifted.

Both validations can be skipped with `--no-validate` if validated by
hand already.

## Pre-flight

Before running the script:

  - **Both target collections exist + are populated**. Use
    `scripts/create_{articles,offers}_collection.py` and
    `scripts/indexer_bulk.py` first; see `indexer/RUNBOOK.md` for the
    bulk pipeline.
  - **Server-side flush has happened**. The script reads row counts
    from `get_collection_stats`, which only sees sealed segments.
    `indexer_bulk.py` issues a final `flush()` automatically, but if
    you're swinging onto a hand-loaded collection, flush manually
    first.
  - **Operator knows the prior alias targets**. The script captures
    them and prints a `--rollback-to` hint after a successful swing,
    but if the operator wants to revert later they need that hint.

## Standard invocation

```sh
uv run python scripts/swing_aliases.py \
    --articles-target articles_v4 \
    --offers-target offers_v5 \
    --milvus-uri http://localhost:19530
```

Output (success):

```
Target pair: articles='articles_v4', offers='offers_v5'
Pre-flight validation:
  rows: articles=130000000, offers=510000000
  join-key check OK: 200/200 sampled article_hashes resolve
Prior alias state (for rollback):
  'articles' → 'articles_v3'
  'offers' → 'offers_v4'
Swing 1/2: articles alias
  swinging alias 'articles': 'articles_v3' → 'articles_v4'
Swing 2/2: offers alias
  swinging alias 'offers': 'offers_v4' → 'offers_v5'
Both aliases swung. Final state:
  'articles' → 'articles_v4'
  'offers' → 'offers_v5'
Rollback hint: re-run with `--rollback-to articles_v3,offers_v4` to revert.
```

## Production-scale validation

The defaults (`--min-rows-articles=1`, `--min-rows-offers=1`) are for
smoke runs. Set them to expected catalog scale on production:

```sh
uv run python scripts/swing_aliases.py \
    --articles-target articles_v4 \
    --offers-target offers_v5 \
    --milvus-uri http://milvus.internal:19530 \
    --min-rows-articles 100000000 \
    --min-rows-offers 400000000 \
    --join-key-sample 1000
```

`--join-key-sample 1000` raises the random sample from 200 to 1000
offers — costs one extra Milvus query, catches narrower drift bands.

## Dry-run

```sh
uv run python scripts/swing_aliases.py \
    --articles-target articles_v4 --offers-target offers_v5 --dry-run
```

Runs every pre-flight validation against the live Milvus, prints the
plan, and exits without mutating any alias. Use this to sanity-check
target collection state before the real run.

## Rollback

### Same-run rollback (automatic)

If swing 1 succeeds and swing 2 fails, the script automatically rolls
swing 1 back to its prior target and re-raises the failure. The system
ends up on the prior consistent pair. No operator action needed beyond
investigating the swing-2 failure.

If the rollback itself fails (rare — both alias mutations are single
metaserver writes), the script logs the precise `MilvusClient.alter_alias`
call to run by hand and re-raises.

### Post-cutover rollback (manual)

If both swings succeed but a data-quality issue surfaces post-cutover,
swing back with `--rollback-to`:

```sh
uv run python scripts/swing_aliases.py \
    --rollback-to articles_v3,offers_v4 \
    --milvus-uri http://milvus.internal:19530
```

`--rollback-to` skips validation (you're going back to a known-good
pair). The script swings articles → `articles_v3` then offers →
`offers_v4` in the same order.

## Validation knobs

```
--no-validate            Skip row-count + join-key checks. Use only
                         after a manual validation pass.
--min-rows-articles N    Floor on articles target row count. Default 1
                         (smoke). Production: catalog scale (~100M+).
--min-rows-offers N      Same, for offers target. Production: ~400M+.
--join-key-sample N      Random offers to sample for hash-resolution
                         (default 200). 0 = skip the join-key check.
                         Larger = stronger drift coverage, one extra
                         Milvus query worth of cost.
```

## Failure modes

### `Target collection 'articles_v4' does not exist`

The bulk indexer didn't create the collection. Run
`scripts/create_articles_collection.py --version 4 --no-alias` first.

### `articles target 'articles_v4' has 0 rows, below --min-rows-articles=1`

Bulk indexer crashed mid-run, or `flush()` wasn't called yet. Check
the indexer logs + the `--bulk-insert-checkpoint` file. Resume the
bulk run rather than swinging onto an empty collection.

### `join-key drift: 47/200 sampled offer article_hashes not found`

Bulk articles and offers came from different indexer invocations. The
two streams are emitted from a single DuckDB query in `bulk.py`, so
this should never happen — but if it does, **do not swing**. Re-run
the bulk indexer to regenerate a consistent pair.

### `Second swing FAILED: ... — rolling back first swing`

Network blip or Milvus metaserver hiccup mid-swing. The automatic
rollback log line will say either:

  - `Rollback complete. System is back on the prior pair.` —
    investigate why the second swing failed; safe to retry.
  - `ROLLBACK ALSO FAILED ...` — run the printed
    `MilvusClient.alter_alias(...)` call manually to restore the
    prior articles target.

### `--rollback-to must be 'articles_vM,offers_vN'`

The `--rollback-to` value is parsed as two comma-separated collection
names. No spaces.

## Soak window

The script does **not** drop the previous collections — that's by
design. Operators drop them after a soak window (default 7 days) once
they're confident no in-flight requests still reference them:

```sh
uv run python -c "
from pymilvus import MilvusClient
c = MilvusClient(uri='http://localhost:19530')
c.drop_collection('articles_v3')
c.drop_collection('offers_v4')
"
```

Milvus releases by collection ID rather than name, so the drop is
safe immediately after the alias swings — the soak window is purely
to keep a rollback target available in case a regression surfaces.
