#!/usr/bin/env python3
"""§14 `features` and §15 `keywords`: the settled per-record union rules.

Extracted from `feat_kw_audit.py` by MXG-67 so a BUILD PATH can call them.
The rules were settled, tested and measured months before anything imported
them: `build_cheap_features.ArticleDoc.from_es` -- the one seam that produces
both the CE's training text (`build_es_text.py` -> `out/ce_data/es_text.jsonl.gz`)
and its serving text -- still took `features` from a single record and unioned
`keywords` with none of §15's drops.

This module is the §14/§15 twin of `id_rules` / `name_rules` / `category_rules`,
and it carries their constraint: **it must import nothing heavier than
`textnorm`.** `feat_kw_audit` cannot be imported from a serving path because it
pulls in `name_bag_audit`, which pulls in an LLM client (`glm_judge_check`).
`tests/test_feat_kw_rules.py` asserts that import hygiene in a subprocess.

`feat_kw_audit` re-exports everything below, so the ten audits/probes/evals and
the two rule test files that already call these functions keep working
unchanged -- which is also the proof that the move was behaviour-preserving.
The control-arm copies of today's pick (`rep_offer`, `render_today`) stay in the
audit: they are not rules.

Evidence for the two operator choices lives in the docstrings below and in
`report/pipeline_v2/feat_kw_audit.md`; what the union costs the cross-encoder is
`report/pipeline_v2/feat_kw_ce_ab.md`.
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textnorm as TN  # noqa: E402
from textnorm import floor, norm_compare as norm, toks_compare as toks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# the rules
# ===========================================================================
# A value that says nothing on its own. Normalized forms (norm() already
# lowercases, strips punctuation and collapses whitespace).
# Deliberately does NOT contain the yes/no tokens: those are handled by the
# boolean-key rule below, which needs to tell "SVHC-frei: ja" (keep the key, it
# is a real property) from "Dimmung DALI: nein" (drop the whole entry).
JUNK_VALUES = {
    "", "0", "00", "000", "k a", "ka", "n a", "na", "keine angabe",
    "keine angaben", "nicht zutreffend", "nicht relevant", "nicht definiert",
    "ohne angabe", "sonstige", "sonstiges", "diverse", "divers",
    "unbekannt", "unknown", "none", "null", "xx", "xxx",
    "-", "--", "---", "entfaellt", "entfallt", "leer", "empty", "tbd",
    "n v", "nv", "9999", "99999", "999999",
}


def is_junk_value(nv):
    if nv in JUNK_VALUES:
        return True
    if len(nv) < 2:
        return True
    # a single repeated character: "....", "0000", "----"
    bare = nv.replace(" ", "")
    return len(set(bare)) <= 1


def listed(value):
    """Wrap a bare scalar so `for x in listed(v)` never walks a string's
    characters. The ES index carries `features`/`keywords` as lists on every
    record sampled (758/758), so this changes no real document — it preserves a
    robustness property `source_assembler` had and its rewrite would otherwise
    have dropped (MXG-48). Only ever adds values; it cannot remove one.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def feature_pairs(offer):
    """One record's `features` -> [(raw_key, raw_value)]. The ES shape is
    [{"name": ..., "values": [...]}]; a bare string is tolerated because
    build_cheap_features and sibling_offer_diffs both guard for it, and a bare
    dict because a single-feature record can arrive unwrapped."""
    out = []
    for f in listed(offer.get("features")):
        if isinstance(f, dict):
            k = str(f.get("name") or "").strip()
            vals = f.get("values") or []
            if not isinstance(vals, list):
                vals = [vals]
            if not vals:
                out.append((k, ""))
            for v in vals:
                out.append((k, str(v).strip()))
        elif isinstance(f, str):
            out.append(("", f.strip()))
    return out


def lost_tokens(member_itoks, pool, count_glued=False):
    """Index tokens a drop would remove from the document, that anyone can miss.

    `count_glued=False` exempts tokens containing `_`, i.e. the ones
    `joined_words_strict` builds by catenating across a punctuation junction.
    §15 established, and probed directly, that losing one cannot break a match:
    the filter emits `vhm_bohrer` at the SAME position as `vhm` and `bohrer`, so
    ES matches the group as synonyms, and the parts survive in the pool by
    construction -- containment is what put the candidate here. Production has
    no phrase clause on `offers.keywords`, so what is left is a scoring nudge on
    hyphenated-compound queries.

    This matters more than it sounds, because the glued tokens are not a
    sideshow: the top genuine losses under the name-subset drop are
    `400ml_weicon`, `0_2kg_weicon`, `10mx19mm_tesafilm` -- tokens that exist
    ONLY because a vendor pasted the product name twice and the analyzer glued
    across the junction. Counting them as losses preserves a duplicated keyword
    for the sake of a token that the duplication itself invented.
    """
    lost = member_itoks - pool
    return lost if count_glued else {t for t in lost if "_" not in t}


def dedup_token_subset(vals, also_substring=True, itoks=None, vetoed=None,
                       pool=frozenset()):
    """Drop a value whose token set is contained in another's, longest
    normalized first (G5 for prose).

    `also_substring` additionally drops a value that is a plain substring of a
    kept one. The two operators are NOT nested, which §4's ladder presentation
    obscures: `stahl` is a substring of `edelstahl` but their token sets are
    disjoint, so token-subset alone keeps both. That makes them a choice rather
    than a ladder, and **neither field takes the substring half any more** --
    `keywords` retired it 2026-08-06 (§15), `features` 2026-08-08 (§14, MXG-54).
    `also_substring=True` survives as a measurement arm: the ladder reports each
    operator separately as well as combined, and that is what prices what the
    retirement gave up.

    `itoks` maps a RAW value to the index tokens ES actually produces for it,
    and turns the whole thing from a claim into a guarantee. Containment in
    comparison space does not imply containment in index space -- the compare
    profile is finer than the analyzer in some places and coarser at a
    non-alphanumeric infix -- and every measured loss from this operator came
    from that gap. With `itoks` supplied, a drop additionally requires that the
    value's index tokens are already carried by the values that survive, so the
    operator cannot remove a token from the document by construction rather
    than by measurement. It is a VETO on a compare-space candidate, never a
    licence to drop something compare-space would keep: the analyzer is coarser
    than the compare form (it folds `Äpfel` and `aepfel` together), so using it
    alone would over-merge, which is the one direction §4.1 rule 2 forbids.
    """
    items = [(raw, nv, set(nv.split())) for raw, nv in vals if nv]
    # `t[0]` is the G5 tie-break, added by MXG-48. Without it the sort is
    # stable on equal normalized forms, so RECORD ORDER decided which raw
    # spelling survived: `Edelstahl,` or `Edelstahl;`, `Stichsaegeblaetter` or
    # `Stichsägeblätter`. Measured at 4.4% of multi-record articles -- the same
    # defect `article_rules`' reshuffle fence exists to prevent, which §14/§15
    # never had. Only breaks ties that were previously arbitrary.
    items.sort(key=lambda t: (-len(t[1]), t[1], t[0]))
    keep, kept_itoks = [], set(pool)
    for raw, nv, ts in items:
        by_tokset = any(ts <= k[2] for k in keep)
        drop = by_tokset or \
            (also_substring and any(nv in k[1] for k in keep))
        if drop and itoks is not None and \
                lost_tokens(itoks.get(raw, set()), kept_itoks):
            drop = False
            if vetoed is not None:
                vetoed.append((raw, "tokset" if by_tokset else "substring"))
        if drop:
            continue
        keep.append((raw, nv, ts))
        if itoks is not None:
            kept_itoks |= itoks.get(raw, set())
    return [(raw, nv) for raw, nv, _ in keep]


# ===========================================================================
# the boolean-key lexicon
# ===========================================================================
# 21,287 of the random frame's feature values are the literal string `nein`.
# A key whose observed values are ALL drawn from {ja, nein, true, false, 0, 1,
# x, -} is a yes/no flag, and a NEGATIVE flag rendered into the document text
# is worse than useless: `Dimmung DALI: nein` puts the token `dali` in the
# article, so an article that explicitly does NOT dim over DALI becomes
# matchable by a DALI query. That is a false positive the catalog told us how
# to avoid, so it is not covered by G3.
#
# Built corpus-wide from the random frame, exactly like §11's ingestion
# frequency blocklist: a per-article view cannot tell a boolean key from a
# one-off, because it sees one value.
BOOL_VALUES = {"ja", "nein", "true", "false", "0", "1", "x", "", "y", "n",
               "yes", "no", "vorhanden", "nicht vorhanden"}
NEG_VALUES = {"nein", "false", "0", "ohne", "no", "n", "-", "",
              "nicht vorhanden"}
BOOL_MIN_OBS = 3

CONTRACT_PATH = os.path.join(HERE, "contracts", "feat_bool_keys.json")
_bool_keys = None


def build_bool_keys(docs):
    """normalized key -> True if every observed value is a yes/no token."""
    kv = defaultdict(Counter)
    for d in docs:
        for o in (d.get("offers") or []):
            for rk, rv in feature_pairs(o):
                kv[norm(rk)][norm(rv)] += 1
    return {k for k, c in kv.items()
            if sum(c.values()) >= BOOL_MIN_OBS and set(c) <= BOOL_VALUES}


def bool_keys():
    """The PINNED lexicon, for callers that render documents.

    `build_bool_keys` needs a corpus. A build path has one article, and a
    serving path has less time than that, so the set is frozen to a tracked
    artifact and both sides read the same one. They must: the flag rule decides
    whether a key renders at all, so training and serving disagreeing about the
    set is a train/serve skew in the document text itself.

    FAILS CLOSED, which is the MXG-50 lesson repeated (`category_rules.
    load_lexicon`'s comment names it): a missing artifact returning an empty set
    would not error, it would silently render the `rule_noflag` arm -- a
    different, unmeasured document text -- and nothing downstream could tell.

    Regenerating the artifact changes rendered text and is therefore a ticket of
    its own, not a side effect of a run. `feat_kw_audit.py --step bool-keys`
    writes it.
    """
    global _bool_keys
    if _bool_keys is None:
        try:
            with open(CONTRACT_PATH, encoding="utf-8") as f:
                art = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"boolean-key lexicon not found at {CONTRACT_PATH}. It is "
                "tracked in git -- run from a full checkout of /workspace, or "
                "rebuild it with `feat_kw_audit.py --step bool-keys`.")
        keys = art.get("keys") or []
        if not keys:
            raise ValueError(
                f"keys is empty in {CONTRACT_PATH}. This module fails closed: "
                "an empty lexicon silently renders the `rule_noflag` arm "
                "instead of the settled §14 rule.")
        _bool_keys = frozenset(keys)
    return _bool_keys


def union_features(offers, bool_keys=(), key_norm=True, drop_junk=True,
                   dedup="token", flag_policy="drop"):
    """The rule under measurement. Returns
    {norm_key: {"disp": display key, "vals": [(raw, norm)], "n_recs": int}}
    with keys in deterministic order.

    `dedup` defaults to `token`, NOT `both` -- owner decision 2026-08-08
    (MXG-54), matching §15's 2026-08-06 change. The argument is not §15's,
    though, and the difference matters when reading the evidence. `keywords` had
    a measured lexical defect: it is analyzed and queried, so a substring drop
    could and did remove a token from the index. `features` is a `keyword`-type
    field with no analyzer and is not in the STANDARD query's field list, so no
    lexical match can break here and there is no veto to build -- the compare-
    vs-index instrument §15 used does not exist for this field. What carries the
    decision instead is symmetry plus a measured price: the same operator
    deletes the same German head nouns (`handschuh` inside `schutzhandschuhe`)
    from the text SPLADE and the CE read, and turning it off costs under 1% of
    rendered characters, because the key merge -- not the operator -- is what
    shrinks this field (§14 *The operator is not the lever*).

    `flag_policy` is the MXG-108 per-consumer fork, and it is the ONLY place
    the two consumers' feature CONTENT may differ:

      "drop"      the settled §14 rule (search/SPLADE/enrich): a negative
                  bool value deletes the entry, a positive one keeps the bare
                  key. The rationale is lexical false positives -- `Dimmung
                  DALI: nein` makes a non-DALI article matchable by a DALI
                  query -- which is an argument about MATCHING.
      "keep_raw"  the CE profile: bool-key entries render as ordinary valued
                  entries with the raw vendor value, BOTH polarities
                  (`Dimmung DALI: nein`, `SVHC-frei: false`, `Rostfrei: ja`).
                  A pairwise reader needs the negation stated, not deleted --
                  MXG-84 measured a fresh student trained on flag-less text
                  scoring near-chance on the negation gate (0.30/0.49) where
                  its old-text twin passed unaided. No canonicalization (emit
                  what the vendor wrote, MXG-47 rule 4) and no junk test
                  inside the bool branch (the pinned lexicon vouches the
                  key); only a raw value that strips to empty is skipped,
                  because there is nothing to emit."""
    if flag_policy not in ("drop", "keep_raw"):
        raise ValueError(f"unknown flag_policy {flag_policy!r}")
    buckets = defaultdict(lambda: {"disp": "", "vals": [], "recs": set(),
                                   "flag": False})
    for i, o in enumerate(offers):
        for rk, rv in feature_pairs(o):
            nk = norm(rk) if key_norm else rk.strip().lower()
            nv = norm(rv)
            if not nk:
                # A key made only of characters the compare profile drops --
                # in practice the bare diameter sign, `Ø: 25 mm`. The old
                # profile bucketed these under the invented word `durchmesser`;
                # expanding a symbol into a word is not normalization (MXG-47
                # rule 4), so instead bucket by the floor-normalized raw key.
                # The entry keeps its value and renders as `Ø: 25 mm`, which is
                # what the vendor wrote and what the models should see.
                nk = floor(rk).lower()
                if not nk:
                    continue
            flag = False
            if nk in bool_keys:
                if flag_policy == "keep_raw":
                    # CE profile: the entry survives with its raw value, both
                    # polarities. An empty raw value has nothing to emit.
                    if not rv.strip():
                        continue
                else:
                    # A negative flag renders its KEY into the document text,
                    # so `Dimmung DALI: nein` makes an article that explicitly
                    # does NOT dim over DALI matchable by a DALI query. Drop
                    # the whole entry. A positive flag keeps the key -- the
                    # property is real -- but drops the value, because `ja` is
                    # not a searchable token in any language a buyer types.
                    if nv in NEG_VALUES:
                        continue
                    flag = True
            elif drop_junk and is_junk_value(nv):
                continue
            b = buckets[nk]
            # display key: the longest original spelling, ties lexicographic
            if (len(rk), rk) > (len(b["disp"]), b["disp"]):
                b["disp"] = rk
            if flag:
                b["flag"] = True
            else:
                b["vals"].append((rv, nv))
            b["recs"].add(i)
    out = {}
    for nk in sorted(buckets):
        b = buckets[nk]
        if dedup == "both":
            vals = dedup_token_subset(b["vals"], also_substring=True)
        elif dedup == "token":
            vals = dedup_token_subset(b["vals"], also_substring=False)
        elif dedup == "exact":
            seen, vals = set(), []
            for raw, nv in sorted(b["vals"], key=lambda t: (-len(t[1]), t[1])):
                if nv and nv not in seen:
                    seen.add(nv)
                    vals.append((raw, nv))
        else:  # substring containment
            seen, vals = [], []
            for raw, nv in sorted(b["vals"], key=lambda t: (-len(t[1]), t[1])):
                if nv and not any(nv in s for s in seen):
                    seen.append(nv)
                    vals.append((raw, nv))
        if not vals and not b["flag"]:
            continue
        out[nk] = {"disp": b["disp"], "vals": vals, "n_recs": len(b["recs"]),
                   "flag": b["flag"]}
    return out


def article_id_forms(doc):
    """Every normalized identifier form the article carries, for the keywords
    cross-field-copy test. Mirrors the MPN/CAN rules' step 3: strip leading
    zeros too, since `0012345` and `12345` are the same part."""
    forms = {"mpn": set(), "artno": set(), "ean": set(), "can": set()}
    for o in (doc.get("offers") or []):
        for key, field in (("mpn", "manufacturerArticleNumber"),
                           ("artno", "articleNumber"), ("ean", "ean")):
            v = o.get(field)
            if not v:
                continue
            n = norm(str(v)).replace(" ", "")
            if len(n) >= 3:
                forms[key].add(n)
                forms[key].add(n.lstrip("0") or n)
    for c in (doc.get("custnos") or []):
        if isinstance(c, dict):
            c = c.get("raw") or c.get("normalized")
        if not c:
            continue
        n = norm(str(c)).replace(" ", "")
        if len(n) >= 3:
            forms["can"].add(n)
            forms["can"].add(n.lstrip("0") or n)
    return forms


def classify_keyword(nv, forms):
    """Which cross-field copy, if any, this keyword member is. Only an EXACT
    whole-member match counts: a keyword that merely contains the MPN also
    carries product words and is not a bare id copy."""
    bare = nv.replace(" ", "")
    stripped = bare.lstrip("0") or bare
    for key in ("ean", "artno", "can", "mpn"):   # precedence as in build_id_eval
        if bare in forms[key] or stripped in forms[key]:
            return key
    return None


# The STANDARD query's field list (replay_search.build_standard_query) with
# each field's INDEX analyzer from the live mapping. This is the pool a dropped
# keyword's tokens may hide in: `cross_fields` matches over the pooled list, so
# a token still produced by the name, brand, vendor or article number under ITS
# OWN analyzer is not lost to the document. `kw_subset_token_loss.py` builds the
# same pool independently on purpose -- it is the verifier, and a verifier that
# imports the rule's pooling code cannot catch a bug in it.
STANDARD_FIELDS = (
    ("name", "german_strict_joined"),
    ("name", "german_strict"),
    ("manufacturerName", "german_company"),
    ("vendorName", "german_company"),
    ("articleNumber", "article_number_segments_analyzer"),
)
KW_ANALYZER = "german_strict"
SEP = " ; "          # breaks token adjacency across concatenated values


def index_pool(doc, members, analyzer=KW_ANALYZER):
    """(index tokens per member, pooled index tokens of every OTHER field).

    One `_analyze` round trip per distinct analyzer, four in total, whatever
    the article's size.
    """
    offs = doc.get("offers") or []

    def joined(key):
        return SEP.join(dict.fromkeys(str(o.get(key) or "").strip()
                                      for o in offs if o.get(key)))

    by_analyzer = defaultdict(list)
    for field, an in STANDARD_FIELDS:
        by_analyzer[an].append(joined(field))
    # Members ride along in the same request, and are partitioned off by INDEX,
    # not by value: a keyword that IS the name is the same string in both roles,
    # and matching on the value put the name's own tokens outside the pool --
    # so the veto fired on precisely the drop that is provably safe.
    n_fields = len(by_analyzer[analyzer])
    by_analyzer[analyzer] = by_analyzer[analyzer] + members

    pool, itoks = set(), {}
    for an, texts in by_analyzer.items():
        got = TN.analyze_batch(texts, an)
        cut = n_fields if an == analyzer else len(texts)
        for s in got[:cut]:
            pool |= s
        if an == analyzer:
            itoks = dict(zip(texts[cut:], got[cut:]))
    return itoks, pool


def union_keywords(doc, drop_ids=True, drop_junk=True, dedup="token",
                   drop_name_exact=True, drop_name_subset=False,
                   analyzer=None):
    """Set union of members across records. Returns (kept, fates) where kept
    is [(raw, norm)] in deterministic order and fates is a Counter.

    Two name-copy drops, deliberately kept apart:
      drop_name_exact   member is a record's name over again. Two forms, both
                        counted separately in `fates` and both dropped by this
                        one flag, because they carry the identical argument:
                          name_exact    normalized forms are EQUAL
                          name_tokset   normalized TOKEN SETS are equal, which
                                        catches the commonest shape of this
                                        junk -- the name concatenated with
                                        itself (`weicon spray 400ml weicon
                                        spray 400ml`), which is not an exact
                                        match. Adopted 2026-08-06.
                        Provably redundant either way -- every token is already
                        in field 1 of the rendering, so the drop can only lower
                        a term frequency from 2 to 1. §4's stated reason for
                        deduping at all. On by default.
      drop_name_subset  member's token SET is CONTAINED IN the name's. A
                        superset of the above that CAN change which tokens the
                        document carries in which order. Measured, declined.

    `dedup` defaults to `token`, NOT `both`. Owner decision 2026-08-06: the
    plain-substring half of the operator is off. It was the whole of the
    measured defect -- 99% of the vetoed drops, `handschuh` deleted because it
    is a substring of `schutzhandschuhe`, in a language that compounds -- and
    turning it off buys back the safety with no serving dependency.

    `analyzer` (an ES analyzer name, normally `german_strict`) is a MEASUREMENT
    PATH, not part of the shipped rule, for the same owner decision: the rule
    must not need a live ES. It switches every containment/equality drop from a
    claim to a guarantee -- the candidate is identified in comparison space as
    before, then vetoed unless its INDEX tokens are already carried by what
    survives PLUS every other field of the STANDARD query, which `cross_fields`
    pools. Four `_analyze` round trips per article, whatever its size. Keep it:
    it is how `kw_subset_token_loss.py --drop veto` proves the residual, and how
    the next rule change gets checked before it ships. See `dedup_token_subset`
    for why the analyzer is a veto and never a licence.
    """
    forms = article_id_forms(doc)
    name_toks, name_forms, name_toksets = set(), set(), set()
    raw_names = []
    for o in (doc.get("offers") or []):
        raw_nm = str(o.get("name") or "").strip()
        nm = norm(raw_nm)
        if nm:
            name_forms.add(nm)
            name_toksets.add(frozenset(nm.split()))
            raw_names.append(raw_nm)
        if drop_name_subset:
            name_toks |= set(toks(raw_nm))
    itoks, pool = None, set()
    if analyzer:
        members = list(dict.fromkeys(
            str(kw).strip() for o in (doc.get("offers") or [])
            for kw in listed(o.get("keywords")) if str(kw).strip()))
        itoks, pool = index_pool(doc, members, analyzer)
    # G5 (MXG-48): two records can carry the same member under different raw
    # spellings -- `at5-`/`at5`, `Stichsaegeblaetter`/`Stichsägeblätter`. The
    # first-seen rule below would let ES array order pick the one the document
    # renders, so the spelling is chosen here instead, once, over every record.
    canon = {}
    for o in (doc.get("offers") or []):
        for kw in listed(o.get("keywords")):
            raw = str(kw).strip()
            nv = norm(raw)
            if nv and (nv not in canon or raw < canon[nv]):
                canon[nv] = raw
    seen, cand, fates = set(), [], Counter()
    for o in (doc.get("offers") or []):
        for kw in listed(o.get("keywords")):
            raw = canon.get(norm(str(kw).strip()), str(kw).strip())
            nv = norm(raw)
            if not nv or nv in seen:
                if nv:
                    fates["dup_across_records"] += 1
                else:
                    fates["empty"] += 1
                continue
            seen.add(nv)
            if drop_junk and is_junk_value(nv):
                fates["junk"] += 1
                continue
            cls = classify_keyword(nv, forms)
            if cls:
                fates[f"id_copy_{cls}"] += 1
                if drop_ids:
                    continue
            # The name drops, weakest first. Each is vetoed if the member's
            # index tokens are not all in the pool already.
            def name_ok():
                if itoks is None:
                    return True
                mine = itoks.get(raw, set())
                if mine - pool:
                    fates["name_drop_would_lose_glued"] += 1
                if not lost_tokens(mine, pool):
                    return True
                fates["name_drop_vetoed"] += 1
                return False

            if nv in name_forms:
                fates["name_exact"] += 1
                if drop_name_exact and name_ok():
                    continue
            elif drop_name_exact and frozenset(nv.split()) in name_toksets:
                fates["name_tokset"] += 1
                if name_ok():
                    continue
            elif drop_name_subset and name_toks and \
                    set(nv.split()) <= name_toks:
                fates["name_subset"] += 1
                if name_ok():
                    continue
            cand.append((raw, nv))
    vetoed = [] if itoks is not None else None
    if dedup == "both":
        kept = dedup_token_subset(cand, True, itoks, vetoed, pool)
    elif dedup == "token":
        kept = dedup_token_subset(cand, False, itoks, vetoed, pool)
    elif dedup == "exact":
        kept = sorted(cand, key=lambda t: (-len(t[1]), t[1]))
    else:
        seen2, kept = [], []
        for raw, nv in sorted(cand, key=lambda t: (-len(t[1]), t[1])):
            if not any(nv in s for s in seen2):
                seen2.append(nv)
                kept.append((raw, nv))
    fates["subsumed"] += len(cand) - len(kept)
    for _, why in (vetoed or []):
        fates[f"subsume_vetoed_{why}"] += 1
    fates["kept"] += len(kept)
    return kept, fates


# --- rendering ---------------------------------------------------------------
# The caps are §14 step 8 (600) and §15 step 6 (300), and they are part of the
# settled rule, not a tuning knob. `feat_kw_ce_ab.md` priced `cap_600 - rule` at
# -0.0000 [-0.0002, +0.0001] with 7 of 625 terms moving. They matter here
# because both renderers put these two fields LAST, so they are the truncation
# tail: what the union adds is paid out of the CE's 188-token budget.
CAP_FEATURES = 600
CAP_KEYWORDS = 300


def render_features(merged, cap_chars=None, terminator=""):
    """`build_cheap_features.ArticleDoc.from_es` renders one record's features
    as `name: v1, v2` joined by spaces. Same shape for the union.

    `terminator` is the punctuation after each entry, and exists because the two
    renderers disagree about it: `build_cheap_features` emits none, while the
    SPLADE path -- `splade-service/source_assembler._render_features` and its
    training-side twin `build_article_extras` -- ends every entry with `.`. The
    rule owns content, order and cap; punctuation belongs to the template, and
    a §14 caller must not silently change the shape of the documents its model
    was encoded from."""
    parts = []
    for m in merged.values():
        if m["vals"]:
            parts.append(f"{m['disp']}: "
                         f"{', '.join(raw for raw, _ in m['vals'])}{terminator}")
        elif m.get("flag"):
            # positive yes/no flag: key only. Serving never emits this shape --
            # it requires values -- so it appears only where §14 is applied.
            parts.append(f"{m['disp']}{terminator}")
    s = " ".join(parts)
    return s[:cap_chars] if cap_chars else s


def render_keywords(kept, cap_chars=None):
    s = " ".join(raw for raw, _ in kept)
    return s[:cap_chars] if cap_chars else s


def custnos_from_source(src):
    """§15's CAN comparands, read off a raw ES `_source`.

    Lives here because it is an INPUT to the rule, not a property of whoever
    is rendering: `union_keywords` cannot drop a keyword that copies the
    article's customer article number unless it is handed the number. A missing
    comparand does not raise, it just stops a drop from firing (MXG-100).

    `splade-service/source_assembler.py` does NOT call this -- it extracts the
    same list inline (`:255-257`) -- so "both renderers call it" was never true,
    and the two accepted different shapes: the assembler takes a bare string as
    the entry, this raised `AttributeError` on one. Production ES always stores
    `{value, versionIds}`, so it never fired there; it fired the moment a
    fixture used the shorter shape (MXG-98). Accept both, like the assembler.
    """
    out = []
    for c in (src.get("customerArticleNumbers") or []):
        v = c.get("value") if isinstance(c, dict) else c
        if isinstance(v, dict):
            v = v.get("raw") or v.get("normalized")
        if v:
            out.append(v)
    return out


def article_parts(offers, custnos=(), flag_policy="drop"):
    """The settled §14/§15 output for one article, before rendering.

    `render_article` is this plus the joins and the caps. Split out for
    `ph.esci_es_enrich` (MXG-100), whose readers want the parts: `train_id_gate`
    matches identifiers against keyword MEMBERS, and the lexical-coverage
    features count tokens -- neither has a token budget the 600/300 caps exist
    to protect, so capping in the stored value would destroy signal for free.

    What a consumer must NOT do is re-derive the parts by calling the two
    unions itself: the arguments below are the arm `feat_kw_ce_ab.py` priced,
    and they are pinned in exactly one place.

    `flag_policy` is deliberately the ONE union argument a caller may set --
    it is the MXG-108 per-consumer fork (see `union_features`), and the CE
    seam (`build_cheap_features.ArticleDoc.from_es`) is its only legitimate
    non-default caller."""
    merged = union_features(offers, bool_keys=bool_keys(),
                            flag_policy=flag_policy)
    kept, _ = union_keywords({"offers": offers, "custnos": list(custnos or ())})
    return merged, kept


def render_article(offers, custnos=(), terminator="", flag_policy="drop"):
    """(features_text, keywords_text) for one article, under the settled rules.

    THE one entry point a renderer should call: it fixes every argument that
    `feat_kw_ce_ab.py` swept as an arm, so a build path cannot accidentally
    ship a different rendering than the one that was measured.

      features   §14, pinned boolean-key lexicon, token-subset dedup, cap 600
      keywords   §15, id/junk/name-exact drops, token-subset dedup, cap 300

    `analyzer=None` is deliberate and load-bearing: §15's index-token veto costs
    four `_analyze` round trips per article, and the owner decision is that the
    rule must not need a live ES. It is also why `dedup` stays `token` -- the
    plain-substring half is only safe WITH the veto (§15 Open).

    Returns raw strings. The caller applies `textnorm.floor` -- in
    `build_cheap_features` that happens once, in `ArticleDoc.__init__`, for
    every field and both adapters.
    """
    merged, kept = article_parts(offers, custnos, flag_policy=flag_policy)
    return (render_features(merged, cap_chars=CAP_FEATURES,
                            terminator=terminator),
            render_keywords(kept, cap_chars=CAP_KEYWORDS))


def render_features_ce(offers, terminator=""):
    """The CE-profile §14 features render (MXG-108): identical pinned
    arguments to `render_article`, with `flag_policy="keep_raw"`. Split out so
    the CE seam (`ArticleDoc.from_es`, which also needs the search render for
    the lexical features) does not pay the §15 keywords union twice."""
    merged = union_features(offers, bool_keys=bool_keys(),
                            flag_policy="keep_raw")
    return render_features(merged, cap_chars=CAP_FEATURES,
                           terminator=terminator)
