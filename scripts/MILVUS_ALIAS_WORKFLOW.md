# Milvus offers-collection: alias-based versioning workflow

Operator notes for `scripts/create_offers_collection.py` and the
zero-downtime contract in spec §4.8.

## Naming

- **Collections** are versioned: `offers_v1`, `offers_v2`, ...
- **Alias** `offers` (configurable via `--alias`) is what ftsearch hits.
- The version number is operator-supplied (`--version N`); the script
  refuses to overwrite an existing collection.

ftsearch never embeds a versioned name. `MilvusClient.search` and
`MilvusClient.has_collection` accept an alias for `collection_name`
transparently — the resolution happens server-side. No client change is
needed when the alias swings.

## Bring up a new collection version

```bash
# Pick N = (current version) + 1.
uv run python scripts/create_offers_collection.py --version 3 --alias offers
```

The script creates `offers_v3`, builds the vector + scalar indexes,
loads the collection, and (unless `--no-alias`) creates or atomically
swings the `offers` alias to point at it.

For a pre-population staging run, suppress the alias:

```bash
uv run python scripts/create_offers_collection.py --version 3 --no-alias
# ... populate offers_v3 via the indexer (I1) ...
# ... validate ...
# Then swing the alias:
uv run python scripts/create_offers_collection.py --version 3 --alias offers   # idempotent if already v3
```

`--dry-run` prints the plan without contacting Milvus.

## First-cutover caveat (one-time, I3 territory)

Today the prod collection is named `offers` (no alias). Milvus
disallows an alias whose name matches an existing collection — so the
alias `offers` cannot be created until the legacy `offers` collection
is renamed or dropped.

Sequence at first cutover:

1. Build `offers_v2` with `--no-alias` and populate it (I1).
2. Validate end-to-end against `offers_v2` (e.g. via a temporary alias
   like `offers_v_alias` that ftsearch can be pointed at from a staging
   config).
3. Stop the indexer + drain in-flight searches.
4. Rename or drop the legacy `offers` collection (the destructive step
   — pick rename over drop until parity is signed off).
5. Run `create_offers_collection.py --version 2 --alias offers`. The
   script's swing path is a no-op for the create-from-scratch case
   (alias did not exist), so it just creates the alias.
6. Resume traffic.

For *every subsequent* version bump (`offers_v2 → offers_v3`), step 4
is unnecessary and the alias swing is atomic — that's the steady-state
zero-downtime contract.

## Swap during steady state

Once `offers` is an alias (post first cutover), the swap is:

1. Build `offers_v{N+1}` with `--no-alias`.
2. Populate via the indexer.
3. Validate (smoke against the versioned name directly, or via a
   secondary alias).
4. Run `create_offers_collection.py --version N+1 --alias offers` —
   `MilvusClient.alter_alias` is atomic at the metaserver; readers see
   either the old target or the new target, never neither.
5. Drop the old `offers_v{N}` collection once you're confident no
   in-flight requests still reference it (Milvus releases by collection
   ID, not name, so dropping `offers_v{N}` is safe immediately after
   the swing).

## Inspecting current state

```python
from pymilvus import MilvusClient
c = MilvusClient(uri="http://localhost:19530")
c.list_collections()                          # all collections
c.list_aliases(collection_name="offers_v3")   # aliases pointing at it
c.describe_alias(alias="offers")              # which collection 'offers' resolves to
```

## Schema reference

The schema fields are defined in `scripts/create_offers_collection.py`
(`build_schema`) and mirror spec §7. Scalar indexes — driven by F3..F5
filter / group / aggregation needs — are listed in
`SCALAR_INDEX_FIELDS` in the same file. `tests/test_offers_collection_schema.py`
asserts the schema and runs every F3-bound filter expression against
the live collection on each test run.
