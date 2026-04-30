# Milvus collection: alias-based versioning workflow

Operator notes for `scripts/create_articles_collection.py`,
`scripts/create_offers_collection.py`, and the zero-downtime contract in
spec §4.8.

Two collections, two aliases (post-F9 topology):

| Collection family | Versioned name      | Public alias | Builder script                       |
| ----------------- | ------------------- | ------------ | ------------------------------------ |
| Articles          | `articles_v4`, `…5` | `articles`   | `scripts/create_articles_collection.py` |
| Offers            | `offers_v5`, `…6`   | `offers`     | `scripts/create_offers_collection.py`   |

Versions are intentionally numerically aligned past the F9 PR2b cutover —
articles jumped from v1 to v4 to pair with the post-F8 `offers_v4 → v5`
bump, so a single integer N picks out the consistent pair (`articles_v{N}`,
`offers_v{N+1}`). Pre-cutover history (articles_v1..v3, offers_v3..v4) is
defunct and not referenced by any current script or test.

The version number is operator-supplied (`--version N`); each script
refuses to overwrite an existing collection.

ftsearch never embeds a versioned name. `MilvusClient.search` and
`MilvusClient.has_collection` accept an alias for `collection_name`
transparently — the resolution happens server-side. No client change is
needed when an alias swings.

Article and offer rows reference each other through `article_hash` (the
join key). A consistent search response requires *both* aliases to point
at collections built from the same indexer run — see the "Paired alias
swing" section below.

## Bring up a new collection version

```bash
# Pick N higher than the current version on each. Steady-state pairs
# articles_v{N} with offers_v{N+1}.
uv run python scripts/create_articles_collection.py --version 4 --alias articles
uv run python scripts/create_offers_collection.py   --version 5 --alias offers
```

Each script creates the versioned collection, builds the vector + scalar
indexes, loads the collection, and (unless `--no-alias`) creates or
atomically swings the alias to point at it.

For a pre-population staging run, suppress the alias on both:

```bash
uv run python scripts/create_articles_collection.py --version 4 --no-alias
uv run python scripts/create_offers_collection.py   --version 5 --no-alias
# ... populate both via the F9 indexer two-stream emitter ...
# ... validate end-to-end ...
# Then perform the paired alias swing (see below).
```

`--dry-run` prints the plan without contacting Milvus.

## First-cutover caveat (one-time, I3 territory)

Today the prod collection is named `offers` (no alias). Milvus
disallows an alias whose name matches an existing collection — so the
alias `offers` cannot be created until the legacy `offers` collection
is renamed or dropped.

Sequence at first cutover:

1. Build `articles_v4` and `offers_v5` with `--no-alias` and populate
   them via the F9 indexer (the same bulk run emits both row streams).
2. Validate end-to-end against the versioned names (e.g. via temporary
   aliases like `articles_staging` / `offers_staging` that ftsearch can
   be pointed at from a staging config).
3. Stop the indexer + drain in-flight searches.
4. Rename or drop the legacy `offers` collection (the destructive step
   — pick rename over drop until parity is signed off).
5. Create both production aliases (paired swing, see below).
6. Resume traffic.

For *every subsequent* version bump, step 4 is unnecessary and the
paired alias swing is the steady-state zero-downtime contract.

## Paired alias swing (steady state)

`articles` and `offers` aliases must be swung together because the
join key (`article_hash`) is only consistent within a single indexer
run. A response that mixes article rows from one bulk run with offer
rows from another can have hashes on the offer side that no longer
exist on the article side — observable as missing articles or empty
result pages.

Milvus has no native multi-alias transaction, so the operator drives
the pair manually with explicit rollback. The recommended sequence:

1. Build `articles_v{M+1}` and `offers_v{N+1}` with `--no-alias`.
2. Populate both via the same indexer bulk run (F9 PR2's orchestrator
   guarantees consistent hashes across the two streams).
3. Validate against the versioned names (or staging aliases).
4. Capture the current alias targets for rollback:
   ```python
   from pymilvus import MilvusClient
   c = MilvusClient(uri="http://localhost:19530")
   prev_articles = c.describe_alias(alias="articles")["collection_name"]
   prev_offers   = c.describe_alias(alias="offers")["collection_name"]
   ```
5. Swing both aliases. Order matters for failure recovery — swing the
   *consumer-side* alias (`offers`, the lookup target during search
   resolve) **last**, so a partial-failure window leaves the old offers
   readable against the new articles rather than the other way around:
   ```bash
   uv run python scripts/create_articles_collection.py --version M+1 --alias articles
   uv run python scripts/create_offers_collection.py   --version N+1 --alias offers
   ```
6. **If step 5 second swing fails**, immediately roll the first one
   back:
   ```python
   c.alter_alias(collection_name=prev_articles, alias="articles")
   ```
   The system is back on the previous consistent pair. Investigate the
   swing failure before retrying.
7. Drop the old `articles_v{M}` and `offers_v{N}` collections once
   you're confident no in-flight requests still reference them (Milvus
   releases by collection ID, not name, so dropping is safe
   immediately after the swing).

**Recommended path**: use `scripts/swing_aliases.py` instead of the
manual sequence above — it captures the prior alias state for
rollback, validates row counts + join-key consistency, swings both
aliases in the right order, and auto-rolls-back on second-swing
failure. See `scripts/SWING_ALIASES_RUNBOOK.md` for the operator
recipe. The manual sequence above remains valid as a fallback when
the script can't be run (e.g. partial-rollback scenarios outside its
scope).

The full I3 packet (orchestration + dual-write window + Kafka cutover)
is still pending — `swing_aliases.py` is its alias-mechanics half.

## Swap during steady state (single alias)

The single-alias procedure (used during partial migrations or schema
fixes that touch only one collection):

1. Build the new versioned collection with `--no-alias`.
2. Populate via the indexer.
3. Validate (smoke against the versioned name directly, or via a
   secondary alias).
4. Re-run the create script with `--alias <name>` — `alter_alias` is
   atomic at the metaserver; readers see either the old target or the
   new target, never neither.
5. Drop the old versioned collection.

Note: an `articles_v{N}`-only swap without a matching `offers_v{N}`
swap is only safe when the article hashes have not changed (e.g. a
schema-only fix that re-derives identical hashes from the same
embedded-field tuple). Any change that affects the hash function,
embedded-field set, or canonicalisation must be a paired swing.

## Inspecting current state

```python
from pymilvus import MilvusClient
c = MilvusClient(uri="http://localhost:19530")
c.list_collections()                            # all collections
c.list_aliases(collection_name="articles_v4")   # aliases pointing at it
c.describe_alias(alias="articles")              # which collection 'articles' resolves to
c.describe_alias(alias="offers")                # which collection 'offers' resolves to
```

## Schema reference

- `scripts/create_articles_collection.py` (`build_schema`) — `articles_v{N}`:
  vector + BM25 + article-level scalars + per-currency envelope.
  `SCALAR_INDEX_FIELDS` and `CATALOG_CURRENCIES` define the index set.
- `scripts/create_offers_collection.py` (`build_schema`) — `offers_v{N}`:
  per-offer scalars + `article_hash` join key. No vectors (a
  `_placeholder_vector` field satisfies Milvus's at-least-one-vector
  requirement). `SCALAR_INDEX_FIELDS` lists the indexed fields.
- F9 packet (`issues/article-search-replacement-ftsearch-09-article-dedup.md`)
  is the canonical design reference for the two-collection topology.

`tests/test_offers_collection_schema.py` asserts the offers schema and
runs every F3-bound filter expression against the live collection on
each test run; the equivalent articles-schema test lands with F9 PR2.
