#!/usr/bin/env python3
"""Preprocessing rules for the category-type article fields (MXG-17/19/21).

ONE implementation of the operations that are currently reimplemented five
times each across the repo with divergent semantics. The audit, the probes and
the CE A/B all import from here so that every measurement is made against the
same rule.

Scope note: this module is the *spec's* implementation. It deliberately does
NOT patch the deployed renderers (`splade-service/source_assembler.py`,
`build_article_extras.py`) — the resolution fix below changes served documents
and implies a re-encode, so it ships separately with the audit numbers attached.

The two fields:

  offers[].categoryPaths.upToLevel{1..5}
      CUMULATIVE '¦'-joined vendor breadcrumbs — upToLevelN is the full path
      truncated at depth N, so level k is a strict prefix of level k+1. Only
      the deepest non-empty level carries information. Per-vendor taxonomies
      (15,693 distinct L1 strings corpus-wide), multi-language, and carrying
      control chars and vendor numbering prefixes.

  offers[].s2classGroups
      Always the full ANCESTOR CLOSURE of one assigned 8-digit eCl@ss-style
      code ({XX000000, XXYY0000, XXYYZZ00, XXYYZZWW}), so only the maximal
      chain carries information. Codes only, no names — resolved through
      /data/s2class-categories.json (4,381 entries).

Three defects in the deployed path that this module fixes:

  1. RESOLUTION. `_s2_leaf_labels` does an exact dict lookup, but only 20.2%
     of the 21,687 observed codes are in the 4,381-entry map. Measured on a
     random slice: the `Classification:` line is non-empty on 76% of
     s2-carrying docs today, and on ~100% once an unmapped leaf falls back to
     its deepest NAMED ancestor. Moot for the RENDER since MXG-48 -- §17 rule
     6 removed it -- and it turned out to be moot for the judge prompt too:
     `s2classGroups` is stored as a full closure, so `esci_v4.s2_type`'s
     deepest-first exact lookup already found a named ancestor. Measured over
     8.09M enrich articles when it moved onto `resolve_s2`: 0 gained a name.
     The rollup pays off only where a code arrives WITHOUT its ancestors.
  2. INTERIM CODES. The deployed path filters `S2_JUNK` but has no interim
     filter, so ~5% of served documents carry the contentless string
     "Interimsklassifikation (Sonstige, nicht spezifiziert)".
  3. LEAF SELECTION. `sig()` + `startswith` is a string-prefix test where an
     ancestor test is meant. `catclass_common.maximal_chains` is used instead —
     but honestly: measured over 60,902 records on the random frame and 7,827
     on the gold frame, the two NEVER disagree. Adopted for correctness, not
     because it fixes anything observed. (1) and (2) are the real defects.

One lexicon, three granularities (MXG-50). The junk list is shared; what a
site DOES with a match is not, and widening the list without knowing which
one you are in is a trap:

  record drop   `resolve_s2` below — any junk code, raw or after resolution,
                drops the record's WHOLE classification. Correct (§17 rule 1):
                the junk node's parents are the same junk.
  element filter the behavioural SQL over `ph.article_catalog.s2class`, which
                stores the full ancestor CLOSURE — so removing the code from
                the array leaves its junk parents behind. Fixed in MXG-50 to
                `if(hasAny(col, junk), [], col)`.
  leaf filter   the deployed renderers — drop the code, then leaf-select,
                which PROMOTES the junk parent to leaf and renders its name.
                Still outstanding; MXG-48 owns it with the re-encode.

Public entry points: `load_lexicon`, `deepest_paths`, `clean_path`,
`subsume_paths`, `resolve_s2`, `s2_names`, `union_category`.

    python3 pipeline/category_rules.py --selftest
"""
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import s2_junk_contract as SJ                                           # noqa: E402
from catclass_common import (ARTIFACT_PATH, TREE_PATH, code_ancestors,  # noqa: E402
                             code_depth, maximal_chains, resolve_raw)

PATH_SEP = "¦"
RENDER_SEP = " > "          # within a path, as build_cheap_features._norm_catpath
VALUE_SEP = " | "           # between paths / between class names

# G6 per-field character caps. Drawn from the measured distributions (catpath
# mean 82 chars / p90 132; s2 leaf render mean 36) — these are tail guards, not
# rations: the CE leaves a mean 143 of its 188 tokens free after the head
# fields, so neither field is under budget pressure.
CAT_CAP = 300
S2_CAP = 200

# --------------------------------------------------------------- path hygiene

# Vendor numbering prefixes: "06 - Elektromechanische Bauelemente", "1. Werkzeug".
# A separator is REQUIRED so that "24 mm Rohr" and "3M Klebeband" survive.
NUM_PREFIX_RE = re.compile(r"^\d{1,3}\s*[-._)\]]\s+")
CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
WS_RE = re.compile(r"\s+")

# Contentless leaf segments. Dropping one promotes the path to its parent; a
# path all of whose segments are contentless is dropped entirely.
JUNK_SEGMENTS = {
    "sonstige", "sonstiges", "sonstiger", "diverse", "diverses", "verschiedenes",
    "allgemein", "allgemeines", "nicht spezifiziert", "nicht zugeordnet",
    "ohne kategorie", "ohne zuordnung", "restposten", "misc", "miscellaneous",
    "other", "others", "unspecified", "uncategorized", "n a", "k a", "keine",
    "keine angabe", "unbekannt", "unknown", "default", "root", "-", "--",
}


def _norm_seg(seg):
    """Comparison form for a path segment: NFKC, control chars out, folded."""
    s = unicodedata.normalize("NFKC", str(seg or ""))
    s = CTRL_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip().lower()
    return re.sub(r"[^0-9a-zäöüß ]+", " ", s).strip()


def clean_segment(seg):
    """One path segment -> cleaned display form, or None if contentless.

    Strips control characters, the vendor numbering prefix, and surrounding
    whitespace. Returns None for junk-lexicon members, pure-numeric segments
    and segments under 2 characters.
    """
    s = unicodedata.normalize("NFKC", str(seg or ""))
    s = CTRL_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    s = NUM_PREFIX_RE.sub("", s).strip()
    if len(s) < 2:
        return None
    n = _norm_seg(s)
    if not n or n in JUNK_SEGMENTS or WS_RE.sub("", n).isdigit():
        return None
    return s


def clean_path(path, vendor_norms=()):
    """Raw '¦'-joined path -> list of cleaned segments (may be empty).

    Junk segments are dropped wherever they occur, which promotes a path with
    a contentless leaf to its parent. A leading segment equal to the article's
    own vendor name is dropped as a cross-field copy (same argument as §15
    step 3: the vendor is field 7 of the SPLADE render already).
    """
    segs = []
    for raw in str(path or "").split(PATH_SEP):
        c = clean_segment(raw)
        if c is not None:
            segs.append(c)
    while segs and _norm_seg(segs[0]) in vendor_norms:
        segs.pop(0)
    return segs


def deepest_paths(cp):
    """categoryPaths {upToLevel1..5} -> raw path strings at the deepest level.

    Settles the five divergent copies in favour of the `build_article_extras` /
    `source_assembler` semantics: return ALL paths at the deepest non-empty
    level, not just `v[0]` (which `esci_v4`, `build_class_raw` and
    `build_cheap_features` do, silently dropping an article's other branches).
    """
    if not cp:
        return []
    if isinstance(cp, list):            # older shape: list of such dicts
        out = []
        for e in cp:
            for p in deepest_paths(e):
                if p not in out:
                    out.append(p)
        return out
    for lvl in range(5, 0, -1):
        vals = [p for p in (cp.get(f"upToLevel{lvl}") or []) if p]
        if vals:
            return vals
    return []


def subsume_paths(paths):
    """R1/R2 across paths: drop any path whose segments are a prefix of another.

    Operates on segment lists (from `clean_path`), compared case-insensitively.
    Deterministic output order (G5): deepest first, then lexicographic.
    """
    uniq = {}
    for segs in paths:
        if not segs:
            continue
        key = tuple(_norm_seg(s) for s in segs)
        # G5 (MXG-48): two records can carry the same path under different raw
        # spellings -- `NL-Gesamtkatalog 2026` and `NL_Gesamtkatalog 2026`
        # normalize alike. `setdefault` kept whichever record ES listed first,
        # so the rendered document depended on array order. Deterministic
        # choice, made where the arbitrary order enters.
        if key not in uniq or RENDER_SEP.join(segs) < RENDER_SEP.join(uniq[key]):
            uniq[key] = segs
    keys = list(uniq)
    kept = [k for k in keys
            if not any(o != k and len(o) > len(k) and o[:len(k)] == k
                       for o in keys)]
    kept.sort(key=lambda k: (-len(k), k))
    return [uniq[k] for k in kept]


def render_paths(paths, cap=CAT_CAP):
    """Cleaned+subsumed segment lists -> the rendered `Category:` string."""
    out = VALUE_SEP.join(RENDER_SEP.join(segs) for segs in paths)
    return out[:cap] if cap else out


def render_path_strings(paths, cap=CAT_CAP):
    """`render_paths` for paths ALREADY joined into strings -- the shape
    `article_rules.build_row` stores in `category_paths`.

    Exists so the two document renderers cannot each grow their own idea of
    what "the category" is. Both used to take `category_paths[0]`, i.e. one
    branch of a §16 union that deliberately keeps them all (MXG-103)."""
    out = VALUE_SEP.join(p for p in paths if p)
    return out[:cap] if cap else out


def top_level_1(rendered):
    """Level 1 of the top-ranked path, from a rendered multi-path string.

    `rendered.split(RENDER_SEP)[0]` is what the callers used to do, and it is
    wrong the moment there is more than one path: for `A | C > D` it returns
    `A | C`. Split on the PATH separator first."""
    if not rendered:
        return ""
    return rendered.split(VALUE_SEP)[0].split(RENDER_SEP)[0]


def path_depth(rendered):
    """Deepest level count across the paths in a rendered string.

    `rendered.count(">") + 1` counts separators across the whole string, so
    two two-level paths read as depth 3. §16 takes every path at ONE level,
    but `clean_path` can shorten some of them, so this is a max rather than
    the first path's length."""
    if not rendered.strip():
        return 0
    return max(p.count(RENDER_SEP.strip()) + 1
               for p in rendered.split(VALUE_SEP))


# ------------------------------------------------------------------- lexicon

class Lexicon:
    """Junk/interim/name lexicon.

    `junk` comes from `contracts/s2_junk.json` (MXG-50), which is tracked in
    git and fails closed; `interim`, `raw_map` and `version` come from the
    built artifact, which is not. Seven codes: six ranked by
    `build_category_tree.py --step junk` (name-token heterogeneity +
    vendor-catpath disagreement, min support 200, depth>=3) and then
    HUMAN-CONFIRMED — ten codes inside the same z band were rejected, so the
    z-score ranks and does not decide — plus one hand-found seed the scan
    cannot see.
    """

    def __init__(self, tree, junk, interim, raw_map, version):
        self.tree = tree
        self.junk = frozenset(junk)
        self.interim = frozenset(interim)
        self.raw_map = raw_map or {}
        self.version = version

    def resolve(self, code):
        """Observed code -> deepest NAMED tree ancestor (or None)."""
        code = str(code)
        hit = self.raw_map.get(code)
        return hit if hit else resolve_raw(code, self.tree)

    def name(self, code):
        return self.tree.get(str(code))

    def is_interim(self, code):
        if str(code) in self.interim:
            return True
        n = self.tree.get(str(code)) or ""
        return n.startswith("Interimsklass")


def load_lexicon(artifact=ARTIFACT_PATH, tree_path=TREE_PATH):
    tree = json.load(open(tree_path))
    # Junk from the CONTRACT, which is tracked in git and raises when it is
    # missing or empty. Reading it from the artifact used to fail OPEN: no
    # artifact -> empty junk set -> every filter silently a no-op (MXG-50).
    junk = SJ.junk_codes()
    interim, raw_map, version = set(), {}, "contract+tree-only"
    if os.path.exists(artifact):
        art = json.load(open(artifact))
        raw_map = art.get("raw_map") or {}
        version = art.get("version") or version
        for code, node in (art.get("nodes") or {}).items():
            if node.get("status") == "interim":
                interim.add(code)
    # Fallback + belt-and-braces: the widened prefix. `esci_v4.s2_type` used
    # the narrow "Interimsklassifikation", which misses "Interimsklasse (nicht
    # spezifiziert)" -- 4.5% of the enrich frame. It calls `resolve_s2` since
    # MXG-48, so this is now the only prefix in the repo.
    for code, name in tree.items():
        if name.startswith("Interimsklass"):
            interim.add(code)
    return Lexicon(tree, junk, interim, raw_map, version)


# ------------------------------------------------------------------- s2class

def resolve_s2(codes, lex, drop_junk_offer=True):
    """Raw s2classGroups of ONE record -> (leaf codes, per-code fates).

    1. If any code is junk (raw or resolved), the whole record's
       classification is dropped — its parent codes are the same junk
       (`esci_v4.py:53-58`). Blacklisting only the leaf is not enough.
    2. Every code resolves to its deepest NAMED tree ancestor. This is the
       fix for the 24% empty-line population; an unmapped vendor code is a
       resolution failure, not junk.
    3. Interim codes are dropped after resolution, so a record classified only
       into the 90* family contributes nothing rather than a contentless name.
    4. Leaf-select with `maximal_chains` (correct ancestor test, unlike the
       deployed `sig()`+`startswith`).
    """
    fates = {}
    raw = [str(c) for c in (codes or []) if str(c).strip()]
    if not raw:
        return [], fates
    if drop_junk_offer:
        for c in raw:
            if c in lex.junk or lex.resolve(c) in lex.junk:
                return [], {c: "junk_offer" for c in raw}
    resolved = []
    for c in raw:
        r = lex.resolve(c)
        if r is None:
            fates[c] = "unmapped"
            continue
        if lex.is_interim(r):
            fates[c] = "interim"
            continue
        fates[c] = "leaf_named" if r == c else "ancestor_named"
        resolved.append(r)
    return maximal_chains(set(resolved)), fates


def s2_names(leaves, lex, max_depth=None):
    """Leaf codes -> German class names, deterministic order (G5).

    `max_depth` truncates each leaf to that tree depth before naming — the
    render-depth lever. It matters because catalog accuracy collapses with
    depth (catclass_gold_v1: L1 0.666 / L2 0.426 / L3 0.216 / leaf 0.094), so
    rendering the leaf emits the least reliable node in the chain.
    """
    out = []
    for code in leaves:
        c = code
        if max_depth and code_depth(code) > max_depth:
            c = code_ancestors(code)[max_depth - 1]
        if lex.is_interim(c):
            continue
        n = lex.name(c)
        if n and n not in out:
            out.append(n)
    return sorted(out)


def render_s2(names, cap=S2_CAP):
    out = VALUE_SEP.join(names)
    return out[:cap] if cap else out


# --------------------------------------------------------------- the G1 union

# Best -> worst. One code can have different fates in different records: junk
# in the offer that got dropped wholesale (rule 1), resolvable in another. This
# used to be `dict.update`, so whichever record came LAST won and two
# reshuffles of the same article returned different fates -- a G5 violation
# found by MXG-65's order-invariance fence.
#
# Owner decision 2026-08-10: the record that says the code is FINE wins. That
# is what the emitted facets already do -- a junk record contributes no leaves
# and a good record's leaves are kept regardless -- so this only stops the
# provenance from contradicting the output it describes.
_FATE_RANK = ("leaf_named", "ancestor_named", "interim", "unmapped",
              "junk_offer")


def _merge_fates(into, new):
    for code, fate in new.items():
        cur = into.get(code)
        if cur is None or _rank(fate) < _rank(cur):
            into[code] = fate


def _rank(fate):
    return _FATE_RANK.index(fate) if fate in _FATE_RANK else len(_FATE_RANK)


def union_category(offers, vendor_names=(), lex=None, max_depth=None,
                   cat_cap=CAT_CAP, s2_cap=S2_CAP):
    """Fold one article's records into the category representation.

    Returns a dict with the rendered strings plus the G4 provenance the spec
    requires: per value, how many records carried it, and the cardinality of
    what was folded.

    G2 note: this is a union with NO conflict resolution — no majority, no
    recency. The measured decomposition says ~82% of cross-record s2
    disagreement is genuine branch conflict (only 2% subsumption), so the
    union of a single-label field renders contradictory type assertions. That
    is why `s2_names` is a *separable* output here: the spec's recommendation
    is that raw s2 codes stop being rendered and remain structured-only, and
    this function is built so the caller chooses.
    """
    lex = lex or load_lexicon()
    vendor_norms = {_norm_seg(v) for v in vendor_names if v}

    path_rec, s2_rec, all_fates = {}, {}, {}
    leaf_codes, s2_raw = set(), set()
    n_path_records = n_s2_records = 0
    for off in offers or []:
        raws = deepest_paths(off.get("categoryPaths"))
        cleaned = [clean_path(p, vendor_norms) for p in raws]
        cleaned = [c for c in cleaned if c]
        if cleaned:
            n_path_records += 1
        for segs in cleaned:
            # Same G5 choice `subsume_paths` makes -- but it must ALSO happen
            # here: paths reach `subsume_paths` through this dict with only one
            # spelling left per key, so keeping the first-seen one (MXG-105
            # found a `setdefault` here) made the CE render depend on
            # `offers[]` order for exactly the two-spelling articles the
            # tie-break exists for. The SPLADE seam feeds `subsume_paths`
            # directly and never had the hole.
            key = tuple(_norm_seg(s) for s in segs)
            rec = path_rec.setdefault(key, [segs, 0])
            if RENDER_SEP.join(segs) < RENDER_SEP.join(rec[0]):
                rec[0] = segs
            rec[1] += 1

        leaves, fates = resolve_s2(off.get("s2classGroups"), lex)
        _merge_fates(all_fates, fates)
        leaf_codes |= set(leaves)
        # the RAW ancestor closure of every record rule 1 did not drop. Not
        # part of §17's emission -- it is what `ph.article_catalog.s2class`
        # has always held, and the L1/L2/L3 backoff in the behavioural
        # features arrayJoins those ancestors (MXG-60 keeps that contract and
        # only removes the record pick).
        if not (fates and set(fates.values()) == {"junk_offer"}):
            s2_raw |= {str(c) for c in (off.get("s2classGroups") or [])
                       if str(c).strip()}
        if leaves:
            n_s2_records += 1
        for n in s2_names(leaves, lex, max_depth=max_depth):
            s2_rec[n] = s2_rec.get(n, 0) + 1

    kept_paths = subsume_paths([v[0] for v in path_rec.values()])
    names = sorted(s2_rec)

    return {
        "category_leaf_text": render_paths(kept_paths, cat_cap),
        "s2class_text": render_s2(names, s2_cap),
        # the structured form, G5-ordered. Added for MXG-60: a consumer that
        # stores paths (rather than renders them) must not have to re-parse
        # `category_leaf_text`, whose separators can occur inside a segment.
        "paths": [RENDER_SEP.join(segs) for segs in kept_paths],
        "s2_leaf_codes": sorted(leaf_codes),
        "s2_leaf_names": names,
        "s2_raw_closure": sorted(s2_raw),
        # G4 provenance / cardinality
        "n_records": len(offers or []),
        "n_path_records": n_path_records,
        "n_s2_records": n_s2_records,
        "n_paths_before_subsume": len(path_rec),
        "n_paths": len(kept_paths),
        "n_s2_names": len(names),
        "path_support": {RENDER_SEP.join(segs):
                         path_rec[tuple(_norm_seg(s) for s in segs)][1]
                         for segs in kept_paths},
        "s2_support": dict(s2_rec),
        "s2_fates": all_fates,
    }


# ------------------------------------------------------------------ selftest

def _selftest():
    lex = load_lexicon()
    fail = []

    def check(name, got, want):
        if got != want:
            fail.append(f"{name}\n    got  {got!r}\n    want {want!r}")

    # 1. ancestor closure -> the maximal chain only.
    #    Real article `7NGO6InZWfDZCqkaXYFftr:ODAwNDkwMDI0MA` (a
    #    Drehwendeschneidplatte). Its assigned leaf 21182003 is OFF-TREE, so
    #    this doubles as the rollup case: today the deployed exact-lookup
    #    renders an empty Classification: line, and the walk recovers
    #    "Schneidplatte, geklemmt".
    closure = ["21000000", "21182003", "21182000", "21180000"]
    leaves, fates = resolve_s2(closure, lex)
    check("closure leaf", leaves, ["21182000"])
    check("closure name", s2_names(leaves, lex), ["Schneidplatte, geklemmt"])
    check("off-tree leaf rolled up", fates["21182003"], "ancestor_named")
    check("on-tree codes named directly", fates["21182000"], "leaf_named")

    # 2. unmapped leaf -> deepest NAMED ancestor (the 24% fill-rate fix)
    deep = [c for c in lex.raw_map if lex.raw_map[c] != c and c not in lex.tree]
    if deep:
        c = deep[0]
        leaves, fates = resolve_s2([c], lex)
        check(f"rollup {c} resolves", bool(leaves and lex.name(leaves[0])), True)
        check(f"rollup {c} fate", fates[c], "ancestor_named")
    else:
        fail.append("no unmapped code in raw_map to exercise the rollup")

    # 3. junk drops the whole record, parents included
    leaves, fates = resolve_s2(["27274091", "27274000", "27270000"], lex)
    check("junk offer dropped", leaves, [])
    check("junk fate", set(fates.values()), {"junk_offer"})

    # 4. an all-interim record renders nothing
    leaves, _ = resolve_s2(["90000000", "90900000", "90909000", "90909090"], lex)
    check("all-interim leaves", leaves, [])
    check("all-interim names", s2_names(leaves, lex), [])

    # 5. cumulative path levels: only the deepest is read
    cp = {"upToLevel1": ["Zerspanung"],
          "upToLevel2": ["Zerspanung¦Wendeplattenwerkzeuge"],
          "upToLevel3": ["Zerspanung¦Wendeplattenwerkzeuge¦Wechselschneidplatten"],
          "upToLevel4": [], "upToLevel5": []}
    check("deepest level", deepest_paths(cp),
          ["Zerspanung¦Wendeplattenwerkzeuge¦Wechselschneidplatten"])

    # 6. path hygiene: control chars, numbering prefix, junk leaf, vendor root
    check("ctrl char", clean_path("Scheibenhandräder \raus Aluminium"),
          ["Scheibenhandräder aus Aluminium"])
    check("num prefix", clean_path("06 - Elektromechanische Bauelemente"),
          ["Elektromechanische Bauelemente"])
    check("junk leaf promotes to parent",
          clean_path("Werkzeug¦Zerspanung¦Sonstige"), ["Werkzeug", "Zerspanung"])
    check("unit-looking segment survives", clean_path("24 mm Rohr"), ["24 mm Rohr"])
    check("vendor root dropped",
          clean_path("RS Components¦Kabel", {_norm_seg("RS Components")}),
          ["Kabel"])
    check("all-junk path empty", clean_path("Sonstige¦Diverse"), [])

    # 7. prefix subsumption across records
    check("subsume", subsume_paths([["A", "B"], ["A"], ["A", "B", "C"], ["D"]]),
          [["A", "B", "C"], ["D"]])

    # 8. depth truncation is the render-depth lever
    check("depth truncation", s2_names(["21182003"], lex, max_depth=2),
          [lex.name("21180000")])

    # 9. the whole union, on two records that disagree on a branch
    offers = [{"categoryPaths": {"upToLevel2": ["Zerspanung¦Fräser"]},
               "s2classGroups": ["21000000", "21180000"]},
              {"categoryPaths": {"upToLevel3": ["Zerspanung¦Fräser¦Schaftfräser"]},
               "s2classGroups": ["27000000", "27270000"]}]
    u = union_category(offers, lex=lex)
    check("union subsumes the shorter path", u["n_paths"], 1)
    check("union keeps both class claims", u["n_s2_names"], 2)
    check("union provenance", u["n_s2_records"], 2)

    if fail:
        print(f"FAIL ({len(fail)})")
        for f in fail:
            print("  " + f)
        return 1
    print(f"ok — lexicon {lex.version}: {len(lex.tree)} names, "
          f"{len(lex.junk)} junk, {len(lex.interim)} interim, "
          f"{len(lex.raw_map)} observed codes mapped")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else
             print(__doc__.strip()) or 0)
