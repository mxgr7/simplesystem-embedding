# cross-encoder data insights

Persistent observations from offline exploration. Update as we learn.

## Token-length distribution (gelectra-base tokenizer, 5000-row sample)

```
query tokens         : p50=4    p75=5    p90=7    p95=9    p99=13   max=76
offer tokens         : p50=159  p75=250  p90=380  p95=499  p99=829  max=2397
pair tokens (+3 spc) : p50=167  p75=258  p90=389  p95=504  p99=843  max=2403
```

Queries are tiny (p99=13). The pair length is essentially the offer length plus
~7 tokens for query+specials.

### Truncation at various `max_pair_length`

| L   | % pairs truncated |
|-----|-------------------|
| 256 | 25.3 %            |
| 384 | 10.2 %  ← current default |
| 512 |  4.8 %            |
| 768 |  1.2 %            |

### Pair-length p90 by label (the leverage finding)

| label       | n     | p50 | p90 | p99 | max  |
|-------------|-------|-----|-----|-----|------|
| Irrelevant  |   324 | 151 | 367 | 937 | 1565 |
| Complement  |   348 | 140 | 389 | 767 | 1332 |
| **Substitute** | **325** | **184** | **439** | **884** | **1722** |
| Exact       | 4 003 | 168 | 385 | 827 | 2403 |

**Substitute pairs are the longest** — p90=439 is ~50 tokens above the current
`max_pair_length=384`. So Substitute (already the worst class by F1) is the
class that loses the most context to truncation. Strong reason to expect that
raising `max_pair_length` to 512 (or 768) will lift Substitute F1 and macro_f1.

## Class distribution

Full dataset (204,182 rows total):

| label       | count   | share   |
|-------------|---------|---------|
| Exact       | 164,199 | 80.4 %  |
| Complement  |  14,643 |  7.2 %  |
| Irrelevant  |  13,665 |  6.7 %  |
| Substitute  |  11,675 |  5.7 %  |

Substitute is the smallest class **and** the worst-performing class by F1. The
trivial "always Exact" baseline gets micro≈0.825 on the val split (per program).
Inverse-frequency class weights (cw=on) give weight 3.7×Irrelevant /
3.5×Complement / 4.3×Substitute / 0.31×Exact — i.e. minority classes get
~12-14x more gradient than Exact. Empirically this hurts at bs=32 (see exp
a1f0bd9 in `results.tsv`).

## Per-query structure

- 21,083 distinct `query_id`s.
- Offers per query: p50=7, p90=23, p99=45, max=131, min=1. Most queries are
  small but there are heavy-tail queries with many offers.
- **Distinct labels per query**:
    - 1 label only: 13,674 queries (64.9 %)
    - 2 labels: 5,338 queries (25.3 %)
    - 3 labels: 1,612 queries (7.6 %)
    - 4 labels: 459 queries (2.2 %)
- **59.2 % of queries have ONLY `Exact` offers** — they don't supply any
  within-query contrast for the classifier to learn class boundaries; they
  effectively just teach "this offer matches this query".
- Queries that contain **at least one** offer of a given class:
    - Exact: 92.7 %
    - Complement: 19.5 %
    - Substitute: 17.2 %
    - Irrelevant: 17.8 %

**Implication**: only ~17-20 % of queries discriminate between Exact and any
given minority class. A natural future experiment is to **sample multi-label
queries with higher weight** so each batch carries more contrastive signal —
similar to embedding-train's `random_query_pool` insight (per the embedding
NOTES.md), but adapted to the cross-encoder per-row sampler.

## Duplicate pairs

Zero duplicate `(query_id, offer_id_b64)` rows in the full parquet. No label
noise from duplicates; no need to deduplicate before training.

## Category as a free signal

A naïve classifier that **just predicts the majority class within each
top-level category** achieves accuracy **0.8725** on the full dataset. To put
that on the same scale as our val metric:

| classifier                                  | accuracy / micro_f1 |
|---------------------------------------------|---------------------|
| trivial "always Exact"                      | 0.825 (program note) |
| **majority-class-per-root-category**        | **0.873**           |
| our best so far (exp 0b0a561, lr=2e-5)      | 0.908               |

So **of the 0.083 lift from "always Exact" to our model, ~0.048 is just
category-paths leverage** that any classifier could exploit, and only ~0.035
is text-reading skill on the other fields (name, EAN, art#, description,
query). This is actually evidence in favour of the `max_pair_length=512`
hypothesis: the marginal text-reading signal is what differentiates Exact
from Substitute / Complement, and that lives in the description tail that's
currently being truncated.

A few category-majority distributions are extreme and worth noting:

| top-level category root            | n   | Exact | Sub | Comp | Irrelev |
|------------------------------------|-----|-------|-----|------|---------|
| Stromversorgung > Batterien        | 280 | 100 % | 0 % | 0 %  | 0 %     |
| PSA-Katalog > Arbeitssicherheit    | 306 |  99 % | 1 % | 0 %  | 0 %     |
| Elektrowerkzeuge und Zubehör (one) | 334 |  64 % | 4 % | 10 % | 22 %    |
| Lager > Zubehör Lager > weiteres   | 422 |  55 % | 1 % | 36 % | 7 %     |
| Betrieb > Zubehör Betrieb          | 421 |  60 % | 5 % | 31 % | 4 %     |

Accessory-style categories ("Zubehör") tilt heavily toward `Complement`,
which fits the labelling intuition (an accessory complements another item).
The model can shortcut on category for a lot of Complement decisions; the
hard cases are likely outside these accessory categories.

## Field quality and template noise

### Field presence

| field                          | non-empty | placeholder `00000000` |
|--------------------------------|-----------|------------------------|
| name                           | 100.0 %   | 0                      |
| ean                            | 100.0 %   | **39,471 (19.3 %)**    |
| article_number                 | 100.0 %   | 0                      |
| manufacturer_article_number    |  92.6 %   | 0                      |
| manufacturer_article_type      |  17.3 %   | 0                      |
| category_paths                 | 100.0 %   | 0                      |
| manufacturer_name              |  94.9 %   | 0                      |
| description                    |  98.4 %   | 0                      |

### Template leak: `EAN: 00000000`

The template uses `{% if ean %}` as a truthiness guard, but the literal
string `"00000000"` is a placeholder for "no EAN" — and it's truthy in Jinja.
**On 19.3 % of rows the offer text contains the noise sequence
`EAN: 00000000`** (~6 tokens of pure noise). Cheap, well-justified
experiment: change the template guard to `{% if ean and ean != '00000000' %}`
(or pre-blank these in `_prepare_records`). Expected effect: tiny token
savings on average, plus removing a feature the model has to learn to ignore.
Probably small, but a free win.

### Description size

For the 98.4 % of rows with a description, post-`clean_html`:
p50 = 327 chars, p90 = 1199 chars, p99 = 3299 chars, max = 9455 chars. The
p90 description alone (1199 chars ≈ ~300 tokens) is essentially the entire
budget at `max_pair_length=384`. So when the description is long, only the
description's first ~70 % survives at 384 — and product specs often live near
the end.

## Query features: digits and substring-of-name

From a 20k-row sample, querying through the actual `query_term`:

### Digit-rate by label (article-number-like queries)

| label       | digit-rate | multi-word-rate |
|-------------|-----------:|----------------:|
| Exact       | 29.0 %     | 49.6 %          |
| **Substitute** | **59.5 %** | **63.1 %**     |
| Complement  | 20.9 %     | 46.8 %          |
| **Irrelevant** | **59.5 %** | 39.9 %         |

**Queries with digits are 2× more likely to be `Substitute` or `Irrelevant`**
than `Exact` or `Complement`. This makes sense — users typing article numbers
or model codes often get near-misses (typos, similar SKUs) that the catalog
labels as `Substitute`, or wrong-shop hits that get labelled `Irrelevant`.

### Query is substring of offer name (case-insensitive)

| label       | q ⊂ name rate |
|-------------|--------------:|
| Exact       | 54.8 %        |
| Complement  | 52.8 %        |
| Substitute  | 12.1 %        |
| Irrelevant  | 11.4 %        |

Substring-match cleanly separates `{Exact, Complement}` (above 50 %) from
`{Substitute, Irrelevant}` (around 12 %). The model surely picks this up
implicitly via attention, but the asymmetry explains why Substitute is hard:
**when the query does not appear verbatim in the offer name, the model has
to read deeper offer text** (specs, dimensions in the description tail) to
decide if it's a near-substitute or just irrelevant. That is exactly the
context that gets truncated at `max_pair_length=384`. Two reinforcing
arguments for spending tokens on the description side.

A binary "q⊂name → Exact else not-Exact" classifier scores only **0.587**
on Exact-vs-not, well below the trivial-Exact 0.825 baseline — i.e.
substring alone is not enough; the model has to combine substring evidence
with spec reading.

### Digit-overlap is a strong Substitute / Irrelevant signal

Restricted to queries that contain at least one digit run (about 33 % of all
queries):

| label       | any-digit-overlap with name | query-has-digit-not-in-name |
|-------------|----------------------------:|----------------------------:|
| Exact       | 81.3 %                      | 21.3 %                      |
| Complement  | 83.7 %                      | 24.6 %                      |
| Substitute  | 61.7 %                      | **62.1 %**                  |
| Irrelevant  | 38.8 %                      | **76.5 %**                  |

When **any query digit fails to appear verbatim in the offer name**, the
label is strongly biased away from `Exact`/`Complement` toward `Substitute`/
`Irrelevant`. The model can in principle pick this up via attention, but the
signal is character-level (e.g., `M6x25` vs `M6x16`) and lives at the subword
boundary — worth keeping in mind when considering whether a different
tokenizer (mDeBERTa, XLM-R) would do better, since their digit segmentation
differs from gelectra-base.

## What lives in the truncated tail of long Substitute pairs

Took 8 random Substitute pairs whose pair length exceeds 384 tokens and read
exactly the segment that gets truncated at `max_pair_length=384`. Sample:

| query | pair_len | what's truncated |
|-------|---------:|------------------|
| `TZe-221` | 452 | `Breite: 9 mm, Länge: 8 m, Farbe: weiß/schwarz` (the **exact** physical dimensions that distinguish TZe-221 from other TZe variants) |
| `Hilti Akkuschrauber` | 472 | `Abmessungen 134 x 68 x 204 mm, Gewicht 0.852 kg, Schalldruckpegel 95 dB, Nennspannung 21.6 V` |
| `steckdosenverteiler` | 529 | `Gerätehöhe 384 mm, Gerätebreite 255 mm, RDF 0.73, InA 32, 4×Schutzkontaktsteckdose 16A 230V IP54` |
| `Taschenlampe 500lm` | 506 | `IPX4, 450 mAh Li-Ion, USB-C-Ladeanschluss, Robuster Taschenclip` |
| `USB C kabel 0,5` | 691 | `32.4 Gbps, 7680×4320 @ 60 Hz, DisplayPort DP 8K` |
| `lan kabel 50m` | 470 | `IEEE802.3bt, TIA/EIA 568B Belegung, Temperaturbereich -5–70 °C, CCA-Leitermaterial` |

The truncated tail in every single sample contains the exact numerical /
material specs that are needed to tell a `Substitute` from `Exact`. This is
direct evidence that `max_pair_length=512` (covering p95 = 504 tokens)
should lift Substitute and macro_f1, and 768 (covering p99 = 843) might lift
it further at higher cost.

It also suggests a **template reordering** experiment if 512 is not enough:
the offer template currently puts `Beschreibung:` (description) last, so
truncation always chops description tail first. Putting description higher,
or introducing a "compress description" step that keeps the first N spec-
laden tokens and skips boilerplate, could squeeze more value into the same
budget.

## Inference-time failure analysis (240-row CPU sample, focal=2 keep)

Ran the gelectra-large + focal=2 best checkpoint on a balanced 60-per-class
sample. The 15 most-confidently-wrong `Substitute → Exact` failures (model
predicts Exact with high probability when label is Substitute):

| query                  | offer summary                                  | failure mode                |
|------------------------|------------------------------------------------|-----------------------------|
| `M10x80`               | KLEMMHEBEL (clamp lever) M10x80                | dimensions match, wrong product class |
| `Sicherheitsschuhe S3` | BOSCH Sicherheitsschuhe (rated S1)             | safety rating S1 vs S3      |
| `Druckpapier`          | Photopapier glossy                             | paper, but specialised      |
| `cavo usb c` (×3)      | USB-C → HDMI cables of various lengths         | not pure USB-C cable        |
| `isopropanol`          | Isopropanol-**Tücher** (wipes)                 | liquid vs wipe form factor  |
| `wera HEX 5`           | Wera 950/9 Hex-Plus **Imperial** set           | metric vs imperial mismatch |
| `OR 126,6×3,53 N`      | O-Ring 126,6×3,53 **EPDM**                     | nitrile (`N`) vs EPDM       |
| `950060 1`             | Ersatzschlüssel 950060 **031**                 | near-miss SKU suffix        |
| `Isolierband schwarz`  | **Gewebeband** schwarz (fabric tape)           | tape type wrong             |

And `Irrelevant → Exact` failures show a different mode:

| query           | offer summary                                  | what fooled the model |
|-----------------|------------------------------------------------|------------------------|
| `Loctite 454`   | Loctite LB 8009 Schmiermittel, **Tube 454 g**  | matched "454" as package weight, not article number |
| `Aufsteller`    | Sonnen-**Ständer** for ground mount            | German `Steller`/`Ständer` polysemy |
| `typ 13`        | Feindrahtklammern Typ 58, 12mm × **13** mm     | matched "13" as a dimension |
| `950060 1`      | Schließnummer **030**                          | near-miss SKU |

### Tokenizer comparison (gelectra vs mdeberta on digit strings)

| string             | gelectra (WordPiece)                       | mdeberta (SentencePiece)        |
|--------------------|--------------------------------------------|---------------------------------|
| `950060 030`       | `95 ##00 ##60 03 ##0`                      | `▁95 0060 ▁030`                 |
| `950060 1`         | `95 ##00 ##60 1`                           | `▁95 0060 ▁1`                   |
| `M10x80`           | `M ##10 ##x ##80`                          | `▁M 10 x 80`                    |
| `OR 126,6X3,53 N`  | `OR 126 , 6 ##X ##3 , 53 N`                | `▁OR ▁12 6,6 X 3,5 3 ▁N`        |
| `Loctite 454`      | `Loc ##ti ##te 45 ##4`                     | `▁Loc tite ▁454`                |
| `BVP130`           | `BV ##P ##13 ##0`                          | `▁B VP 130`                     |
| `TZe-221`          | `T ##Ze - 22 ##1`                          | `▁ TZ e - 221`                  |

**Pattern**: mdeberta keeps multi-digit runs as single tokens far more
often (`130`, `221`, `030`, `454`). gelectra fragments them with `##`
continuations (`##13 ##0`, `##22 ##1`, `03 ##0`, `45 ##4`). A whole-number
token is much easier for attention to match against the same number in the
offer side; `##0` ending one number and `1` ending another offer no
structural commonality even though they're meant to be opaque IDs. This
directly supports the failure-mode hypothesis.

### Failure modes (ranked)

1. **Numerical / spec distinction at the character/digit level** — most failures
   come down to: query digits appear in offer in an unrelated context
   (`454` as weight, `13` as dimension), or appear with a near-miss suffix
   (`950060 1` vs `950060 030`). The WordPiece tokenizer in gelectra splits
   numbers into subwords without preserving char-level structure, so the
   model has to learn to compare digit subsequences via attention. This is
   the main argument for trying a different tokenizer: mDeBERTa-v3 and
   XLM-R use SentencePiece with byte-level fallback, which gives finer-
   grained number tokens. **Strong candidate for the next high-leverage
   experiment.**
2. **Form-factor / material substitution** — query for a liquid, offer for a
   wipe; query for a wire connector, offer for a connector kit. These need
   semantic reasoning over short queries and offer descriptions; bigger
   model + more spec context (already done) is the only handle.
3. **Polysemous short German queries** — `sicher`, `Aufsteller`,
   `Druckpapier`. Single-word queries don't carry enough info; the model
   collapses to the dominant offer category. Likely irreducible.
4. **Possible label noise** — `k10f013nch` query exactly matches the offer's
   article number `K10F013NCH`, yet the label is `Irrelevant`. Some labels
   may be wrong, or graded by an unstated criterion. Worth flagging but not
   actionable from the cross-encoder side.

## Qualitative patterns in `Substitute` (worst class)

From a 25-row random sample, the recurring patterns:

- **Dimensional / size variants**: e.g. query `Diebst.Schraube M6x25` vs
  offer `... M6x16`. Same product family, different size.
- **Article-number near-misses**: query `203211 8` vs offer article
  `203211 2` — same line, different SKU variant.
- **Substituted brand or feature**: query `VBMT 160404E` vs offer
  `VBMT160404-UF-YG3010` — same insert spec, different coating/brand.
- **Bundle/kit vs single item**: query `handabroller` vs offer `36 Rollen
  Polypropylen-Band, 1 Handabroller` — the queried item is one component of
  a larger kit.
- **Quantity mismatch**: query `verlängerungskabel 15m` vs offer
  `1.5m USB 3.0 Verlängerungskabel` — orders-of-magnitude length mismatch.
- **Same function, different tool**: query `Seitenschneider` vs offer
  `Russische Zange` — both cutting tools.

Almost all of these require the model to read **fine numerical / spec
details deep in the offer name and description**. Truncation at 384 tokens
disproportionately hurts this class (Substitute p90 pair length = 439). This
is the strongest data-driven argument so far for trying `max_pair_length=512`.
