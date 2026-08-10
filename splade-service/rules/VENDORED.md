# Vendored preprocessing rules — do not edit here

Every `.py` and `.json` in this directory is a **byte-identical copy** of a file
in the research repo:

| here | source of truth |
| -- | -- |
| `textnorm.py` | `/workspace/pipeline/textnorm.py` |
| `feat_kw_rules.py` | `/workspace/pipeline/feat_kw_rules.py` |
| `category_rules.py` | `/workspace/pipeline/category_rules.py` |
| `catclass_common.py` | `/workspace/pipeline/catclass_common.py` |
| `s2_junk_contract.py` | `/workspace/pipeline/s2_junk_contract.py` |
| `contracts/feat_bool_keys.json` | `/workspace/pipeline/contracts/feat_bool_keys.json` |
| `contracts/s2_junk.json` | `/workspace/pipeline/contracts/s2_junk.json` |

`/workspace/pipeline/tests/test_renderer_parity.py` fails if any of them drifts
by a single byte, and separately fails if the two renderers stop producing the
same field values. **Fix the source file and re-copy; never patch the copy.**

## Why a copy and not an import

This is a separate repository, and the re-encode
(`scripts/reindex_articles_with_splade.py` → `source_assembler.assemble_nul`)
must not depend on a research checkout being present at a particular path.
Before MXG-48 the alternative in force was a hand-written second implementation
of the same rules — that is what produced the divergence MXG-48 exists to
close, so a copy under a byte-identity gate is the weaker of the two evils.
The end state is neither: `field_preprocessing.md` §1–§19 becomes a machine-
readable contract that the Kotlin indexer and this service both satisfy
(MXG-65 / MXG-98).

## Why these five files and no more

They are the transitive import closure at module-import time, and all of it is
stdlib:

```
feat_kw_rules  -> textnorm
category_rules -> s2_junk_contract, catclass_common
```

`textnorm`'s `from common import es_request` sits inside function bodies that
only run when an Elasticsearch analyzer is passed; nothing here passes one.
Each module does `sys.path.insert(0, HERE)` and resolves its contract JSON
relative to `__file__`, so the directory is self-contained by construction.

`category_rules` is copied whole although only its path-hygiene half (§16) is
used. Its s2 half (§17) needs `pipeline/out/category_tree.json`, which does not
exist here — `load_lexicon` would silently degrade to an empty `raw_map`. The
assembler never calls it, and the parity test asserts that.

## Refresh

```
cd /workspace && for f in textnorm.py feat_kw_rules.py category_rules.py \
    catclass_common.py s2_junk_contract.py; do
  cp -p pipeline/$f autoresearch-splade-wt/splade-service/rules/$f
done
for f in feat_bool_keys.json s2_junk.json; do
  cp -p pipeline/contracts/$f autoresearch-splade-wt/splade-service/rules/contracts/$f
done
.venv/bin/python -m pytest pipeline/tests/test_renderer_parity.py -q
```
