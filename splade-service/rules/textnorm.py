#!/usr/bin/env python3
"""Text normalization profiles, one per CONSUMER (MXG-47).

WHY WE NORMALIZE
    Exactly one reason: so two strings that mean the same thing compare equal.
    Normalization is a tool for deciding whether two catalog values are
    duplicates. It is not a tool for producing text.

WHERE WE NORMALIZE
    Merging records into one article -- that is ~all of it. Every downstream
    consumer already normalizes for itself: Elasticsearch has the analyzer
    chain, SPLADE has fold_de + its wordpiece tokenizer, the cross-encoder has
    its tokenizer. There is almost nothing left for preprocessing to do.

THE RULES
    1. Normalize to COMPARE, never to EMIT. The emitted value keeps the
       vendor's original spelling; only the floor (encoding hygiene) touches it.
       The floor holds exactly two REPAIRS, which are allowed to change an
       emitted value because the vendor's spelling there is already corrupt:
       `demojibake` (MXG-52) and `strip_markup` (MXG-64). A repair is not a
       comparison convenience -- it fixes text nothing downstream can read.
    2. The comparison form may be FINER than the consumer's own normalizer
       (under-merges: keeps a redundant value, costs bytes) but NEVER COARSER
       (over-merges: licenses a dedup drop that deletes a token nothing else
       carries). See --selftest for the measured counterexample.
    3. Don't do the downstream's job. `german_strict` already lowercases and
       folds umlauts; doing it earlier buys zero index tokens and can only cost.
    4. Adding text is not normalization. `⌀`->durchmesser, `°`->grad,
       `verz.`->verzinkt are EXPANSION -- a different decision with a different
       owner and its own measurement. They do not belong in a normalizer.

NOT idnorm.py
    idnorm normalizes IDENTIFIERS and chases byte-parity with the ES analyzer
    `article_number_normalized` (keyword tokenizer -> exactly one token), where
    parity is achievable and is a maintained contract, verified by
    `idnorm.py --selftest`.

    Text is read by `german_strict` (whitespace tokenizer + keyword_repeat +
    joined_words_strict), which emits several tokens per input word AT THE SAME
    POSITION and replaces punctuation with `_` rather than dropping it. No
    str->str function can produce that, so text/analyzer parity is not a goal
    here and never will be. `textnorm.py --selftest` proves it rather than
    asserting it.

Usage
  textnorm.py --selftest [--write-snapshot]
"""
import argparse
import bisect
import json
import os
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
# step registry
#
# Every transformation is registered by name and profiles are ordered tuples
# of those names. The tuple IS the machine-readable form of the §4 matrix in
# report/pipeline_v2/field_preprocessing.md -- adding or removing a step is a
# one-line, reviewable diff.
#
# There are ZERO boolean parameters in this module, and there never will be.
# New behaviour is a new named profile, not a flag.
# ==========================================================================
_STEPS = {}


def _step(name):
    def deco(fn):
        _STEPS[name] = fn
        return fn
    return deco


def _run(steps, s):
    s = str(s or "")
    for name in steps:
        s = _STEPS[name](s)
    return s


# ---- tables ---------------------------------------------------------------
TM = dict.fromkeys(map(ord, "®™©"), None)
UMLAUT = {ord("ä"): "ae", ord("ö"): "oe", ord("ü"): "ue",
          ord("ß"): "ss", ord("Ä"): "ae", ord("Ö"): "oe",
          ord("Ü"): "ue"}

# Abbreviations expanded before punctuation is stripped, so only the
# period-terminated form fires ("St." -> stueck, but the token "st" in a part
# code is left alone). Mined from the trailing-period token histogram of the
# 3,900-article fetch; ambiguous ones (d., s., a., i., n., z.) are NOT expanded.
# DELETED from the compare profile -- see _COMPARE_STEPS.
ABBREV = {
    "st": "stueck", "stk": "stueck", "stck": "stueck", "pack": "packung",
    "gr": "groesse", "max": "maximal", "min": "minimal", "f": "fuer",
    "u": "und", "o": "ohne", "inkl": "inklusive", "ca": "circa", "tlg": "teilig",
    "nr": "nummer", "bzw": "beziehungsweise", "ausf": "ausfuehrung",
    "kpl": "komplett", "verz": "verzinkt", "durchm": "durchmesser",
    "mtr": "meter", "elektr": "elektrisch", "autom": "automatisch",
    "transp": "transparent", "zub": "zubehoer", "bl": "blau",
    "montiert": "montiert", "industr": "industrie", "dig": "digital",
    "pol": "polig", "kat": "kategorie",
}
ABBREV_RE = re.compile(r"\b(" + "|".join(sorted(ABBREV, key=len, reverse=True))
                       + r")\.", re.IGNORECASE)

UNITS = ("mm", "cm", "dm", "km", "m", "mg", "kg", "g", "ml", "cl", "l",
         "kw", "w", "kv", "v", "ma", "a", "ah", "nm", "bar", "hz", "khz",
         "mhz", "stueck", "stk", "st", "teilig", "tlg", "polig", "zoll",
         "grad", "min", "h", "sek", "s", "mikron", "my", "µm", "um")
UNIT_RE = re.compile(r"(?<=\d)\s+(" + "|".join(sorted(UNITS, key=len,
                                                      reverse=True)) + r")\b")

# C0 and C1 control characters, minus the whitespace collapse handles anyway.
_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# ---- markup (MXG-64) ------------------------------------------------------
# The normative form of all four tables is contracts/markup.json, which is what
# /next-gen implements from; these are literals so this module stays free of
# file IO, and tests/test_markup_rule.py asserts the two agree. Same
# arrangement test_index_contract.py uses for feat_kw_audit.STANDARD_FIELDS.
#
# SPACE is the DEFAULT disposition for a whitelisted tag: deleting welds two
# words, measured on `iBox 1<strong>1x K.57` -> `1x K.571x`. DELETE is the
# exception for tags that wrap PART of a word, where deleting is exactly what
# preserves the meaning: `H<sub>2</sub>O` -> `H2O`, `mm<sup>2</sup>` -> `mm2`.
_TAG_SPACE = (
    "br p li ul ol dl dt dd hr pre blockquote div table thead tbody tr td th "
    "h1 h2 h3 h4 h5 h6 b strong i em u s small big a span font "
    # not HTML: one vendor's own markup DSL. Stripped, never decoded -- see the
    # contract, and MXG-64 for why `<lt/>`->`<` buys nothing at the index.
    "bullet bulletlist level2 tabvalsep lt gt bis max emdash"
).split()
_TAG_DELETE = "sub sup prefix suffix".split()
MARKUP_TAGS = dict.fromkeys(_TAG_SPACE, " ")
MARKUP_TAGS.update(dict.fromkeys(_TAG_DELETE, ""))

# `<style>`/`<script>` drop WITH their content -- a CSS body is not product
# text, and stripping the tag alone would index it.
MARKUP_CONTAINERS = ("style", "script")

# THE LOAD-BEARING RULE is the whitelist itself. In this catalog `<` and `>`
# are comparison operators far more often than tag delimiters -- 2,700 (v2.1) /
# 2,143 (random) non-tag contexts (`Stahl < 1000 N/mm²`, `Alu > 8% Si`,
# `<10 min`) against ~1,500 / ~2,900 tag occurrences -- so a greedy
# `<[^>]*>` eats real text between a comparison `<` and a later `>`. The
# attribute grammar is spelled out for the same reason: `Wert <b 5 und >10`
# matches `<b[^<>]*>` but not an attribute that must start with a name char.
_ATTR = r"[A-Za-z_:][-A-Za-z0-9_:.]*(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'=<>]+))?"
_TAG_RE = re.compile(
    r"</?\s*(" + "|".join(sorted(MARKUP_TAGS, key=len, reverse=True))
    + r")((?:\s+" + _ATTR + r")*)\s*/?>", re.IGNORECASE)
_CONTAINER_RE = re.compile(
    r"<\s*(" + "|".join(MARKUP_CONTAINERS) + r")\b[^<>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL)
_CDATA_RE = re.compile(r"<!\[CDATA\[|\]\]>")

# Entity names are matched CASE-SENSITIVELY, as HTML requires: `&Uuml;` and
# `&uuml;` are different characters. A CLOSED table on purpose -- a Java
# implementation must not need an HTML library whose set is larger and whose
# numeric handling differs.
MARKUP_ENTITIES = {
    "lt": "<", "gt": ">", "amp": "&", "quot": '"', "apos": "'",
    "nbsp": " ",
    "auml": "ä", "Auml": "Ä", "ouml": "ö", "Ouml": "Ö",
    "uuml": "ü", "Uuml": "Ü", "szlig": "ß",
    "oslash": "ø", "Oslash": "Ø", "aring": "å", "Aring": "Å",
    "eacute": "é", "Eacute": "É", "egrave": "è", "agrave": "à",
    "ccedil": "ç", "ntilde": "ñ", "uacute": "ú", "iacute": "í",
    "oacute": "ó", "aacute": "á",
    "deg": "°", "plusmn": "±", "micro": "µ", "sup2": "²", "sup3": "³",
    "frac12": "½", "frac14": "¼", "frac34": "¾",
    "times": "×", "divide": "÷", "minus": "−",
    "trade": "™", "reg": "®", "copy": "©",
    "laquo": "«", "raquo": "»", "bdquo": "„",
    "ldquo": "“", "rdquo": "”", "lsquo": "‘", "rsquo": "’",
    "ndash": "–", "mdash": "—", "hellip": "…",
    "bull": "•", "middot": "·", "sect": "§", "para": "¶",
    "euro": "€", "pound": "£", "yen": "¥", "cent": "¢",
    "ordm": "º", "ordf": "ª", "shy": "­",
    "ensp": " ", "emsp": " ", "thinsp": " ",
    "acute": "´", "cedil": "¸", "uml": "¨", "sup1": "¹",
    "iquest": "¿", "iexcl": "¡", "curren": "¤", "brvbar": "¦",
    "not": "¬", "macr": "¯", "Oacute": "Ó", "Uacute": "Ú",
    "Aacute": "Á", "Iacute": "Í", "Ccedil": "Ç",
}

# 0x80-0x9F are C1 controls in Unicode and printable characters in cp1252.
# Everything that emits them means cp1252 -- `&#148;` occurs 49 times and is a
# right double quote, not a control. This is the HTML5 rule. The five slots
# cp1252 leaves unassigned are absent, and an entity naming one stays verbatim.
MARKUP_CP1252 = {
    128: "€", 130: "‚", 131: "ƒ", 132: "„", 133: "…", 134: "†", 135: "‡",
    136: "ˆ", 137: "‰", 138: "Š", 139: "‹", 140: "Œ", 142: "Ž", 145: "‘",
    146: "’", 147: "“", 148: "”", 149: "•", 150: "–", 151: "—", 152: "˜",
    153: "™", 154: "š", 155: "›", 156: "œ", 158: "ž", 159: "Ÿ",
}

# `u.` is the German abbreviation for `und`: an upstream hop rewrote `&` -> `u.`
# and hit entity ampersands as collateral (557 occurrences / 91 articles). Only
# the full `u.NAME;` form is recognised -- a bare `u.` is legitimate text
# (`Schrauben u. Muttern`) and must never be touched. The real fix is upstream.
_ENTITY_RE = re.compile(
    r"(?:&(?:#(\d{1,7})|#[xX]([0-9a-fA-F]{1,6})|([A-Za-z][A-Za-z0-9]{1,10}))"
    r"|u\.([A-Za-z][A-Za-z0-9]{1,10}));")

# Latin letters that carry no combining mark to strip, so NFKD leaves them
# alone. Every entry is what `icu_folding` inside german_strict was MEASURED to
# do with it (`--selftest` re-checks all of them against the live index).
_NO_DECOMPOSITION = {
    "ø": "o", "ł": "l", "œ": "oe", "æ": "ae", "ð": "d", "ƒ": "f",
    "đ": "d", "þ": "th", "ħ": "h", "ı": "i", "ŋ": "n", "ŧ": "t",
    "ſ": "s", "ĳ": "ij",
    "ĸ": "q",   # not a typo: ICU folds LATIN SMALL LETTER KRA to `q`, and the
                # contract of this table is to say what icu_folding says.
    "Ø": "o", "Ł": "l", "Œ": "oe", "Æ": "ae", "Ð": "d", "Đ": "d", "Þ": "th",
    "Ħ": "h", "Ŋ": "n", "Ŧ": "t", "Ĳ": "ij",
}

# `fold_umlaut` owns these and runs first; folding ä->a here would silently
# undo the ae/oe/ue/ss spelling the German fold exists to produce.
_GERMAN = set("äöüßÄÖÜ")


class _AccentFold(dict):
    """Lazily-filled str.translate table: accented Latin letter -> ASCII base.

    Populated on demand rather than by sweeping the code space, so import stays
    free and any letter the catalog actually contains gets folded.
    """

    def __missing__(self, cp):
        ch = chr(cp)
        if ch.isascii() or not ch.isalpha() or ch in _GERMAN:
            v = ch
        elif ch in _NO_DECOMPOSITION:
            v = _NO_DECOMPOSITION[ch]
        else:
            base = "".join(c for c in unicodedata.normalize("NFKD", ch)
                           if not unicodedata.combining(c))
            # Greek, Cyrillic, Katakana and the like have no ASCII base. Leave
            # them for `charclass`, whose behaviour on them is unchanged.
            v = base if base.isascii() and base.isalpha() else ch
        self[cp] = v
        return v


ACCENTS = _AccentFold()


# ---- steps ----------------------------------------------------------------
@_step("nfkc")
def _nfkc(s):
    return unicodedata.normalize("NFKC", s)


@_step("nfc")
def _nfc(s):
    return unicodedata.normalize("NFC", s)


@_step("demojibake")
def _demojibake(s):
    """'WÃ¼rth' -> 'Würth' when the utf-8 round-trip through a byte codec is
    clean.

    cp1252 is tried after latin-1 because the two differ exactly on 0x80-0x9F,
    and one of those, `Ÿ` (U+0178, cp1252 0x9F), is what `ß` turns into when
    double-encoded -- so `GrÃ¶ÃŸe` raises UnicodeEncodeError under latin-1 and
    was returned unrepaired. 0.028% of the 50k golden strings (MXG-52).

    A false repair needs the mis-encoded bytes to happen to form valid UTF-8,
    which is why the round-trip itself is the test rather than a heuristic.
    """
    for codec in ("latin-1", "cp1252"):
        try:
            fixed = s.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if fixed != s and "�" not in fixed:
            return fixed
    return s


@_step("demojibake_v0")
def _demojibake_v0(s):
    """latin-1 only. Frozen: `_COMPARE_V0_STEPS` is a byte-exact fence against
    a golden file captured before MXG-47, so improving the live step must not
    be able to move it."""
    try:
        fixed = s.encode("latin-1").decode("utf-8")
        return fixed if fixed != s and "�" not in fixed else s
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _entity_char(m):
    dec, hexa, name, mangled = m.groups()
    if name is not None or mangled is not None:
        return MARKUP_ENTITIES.get(name if name is not None else mangled,
                                   m.group(0))
    cp = int(dec) if dec is not None else int(hexa, 16)
    if 0x80 <= cp <= 0x9F:
        return MARKUP_CP1252.get(cp, m.group(0))
    if cp == 0 or 0xD800 <= cp <= 0xDFFF or cp > 0x10FFFF:
        return m.group(0)
    return chr(cp)


@_step("strip_markup")
def _strip_markup(s):
    """Markup out, product text intact. Contract: contracts/markup.json.

    THE ONLY SEMANTIC STEP THAT TOUCHES AN EMITTED VALUE. Rule 1 of the module
    docstring says normalize to compare, never to emit -- this step breaks it
    deliberately, because `render_features` and `text_es` emit the RAW value:
    a strip living only in the compare profile would clean the dedup
    comparisons and still ship `H<sub>2</sub>O` to SPLADE and the CE and still
    index `reiniger_br_universal`. It is a repair, like demojibake, not a
    comparison convenience (MXG-64, and MXG-13 for the same shape of decision).

    Entities decode BEFORE tags, so a double-escaped `&lt;p&gt;` reaches the
    tag step as `<p>`. Exactly one pass, and replacements are never rescanned:
    `&amp;lt;` becomes the literal text `&lt;` and stops, because the vendor
    escaped that on purpose.

    Whitespace is NOT collapsed here -- both profiles already end in
    collapse_ws, and a step that collapsed on its own would not compose.
    """
    if "<" not in s and "&" not in s and "u." not in s:
        return s
    s = _ENTITY_RE.sub(_entity_char, s)
    if "<" in s:
        s = _CONTAINER_RE.sub(" ", s)
        s = _CDATA_RE.sub(" ", s)
        s = _TAG_RE.sub(lambda m: MARKUP_TAGS[m.group(1).lower()], s)
    return s


@_step("strip_controls")
def _strip_controls(s):
    return _CONTROLS.sub(" ", s.replace(" ", " "))


@_step("strip_tm")
def _strip_tm(s):
    return s.translate(TM)


@_step("lower")
def _lower(s):
    return s.lower()


@_step("expand_abbrev")
def _expand_abbrev(s):
    return ABBREV_RE.sub(lambda m: ABBREV[m.group(1).lower()] + " ", s)


@_step("sym_diameter")
def _sym_diameter(s):
    return s.replace("⌀", " durchmesser ").replace("ø", " durchmesser ")


@_step("decimal_comma")
def _decimal_comma(s):
    return re.sub(r"(?<=\d)[,](?=\d)", ".", s)


@_step("times_to_x")
def _times_to_x(s):
    return re.sub(r"[×✕*]", "x", s)


@_step("glue_dim")
def _glue_dim(s):
    return re.sub(r"(?<=\d)\s*x\s*(?=\d)", "x", s)


@_step("sym_inch")
def _sym_inch(s):
    return re.sub(r'(?<=[0-9/])\s*["”″]', " zoll ", s)


@_step("sym_degree")
def _sym_degree(s):
    return s.replace("°", " grad ")


@_step("fold_umlaut")
def _fold_umlaut(s):
    return s.translate(UMLAUT)


@_step("fold_accents")
def _fold_accents(s):
    """'filetées' -> 'filetees'. Non-German accented letters only.

    Without this, `charclass` turns every such letter into a SPACE and splits
    the word around it -- `filetées` -> `filet es` -- which is the one way this
    profile could still be COARSER than the analyzer on a plain word:
    german_strict yields {filetees}, and the shredded form compares equal to a
    genuine two-word `filet es`. Folding is exactly analyzer-granular instead:
    icu_folding inside german_strict produces the same ASCII base letter, per
    character, verified by --selftest.

    `ø` is no longer contested. It used to be claimed by `⌀ ø -> durchmesser`,
    which is deleted (rule 4: expansion is not normalization), so `ø` folds to
    `o` like any other accent -- which is what ES does with it too.
    """
    return s.translate(ACCENTS)


@_step("charclass")
def _charclass(s):
    return re.sub(r"[^0-9a-zäöüß./]+", " ", s)


@_step("prune_period")
def _prune_period(s):
    """'.' survives only as a decimal point."""
    return re.sub(r"(?<=\d)\.(?=\D)|(?<=\D)\.(?=\d)|(?<=\D)\.(?=\D)|\.$",
                  " ", s)


@_step("prune_slash")
def _prune_slash(s):
    """'/' survives only inside a fraction."""
    return re.sub(r"(?<=\D)/|/(?=\D)", " ", s)


@_step("collapse_ws")
def _collapse_ws(s):
    return " ".join(s.split())


@_step("glue_unit")
def _glue_unit(s):
    return UNIT_RE.sub(r"\1", s)


@_step("alias_stueck")
def _alias_stueck(s):
    return re.sub(r"(?<=\d)(stck|stk|st)\b", "stueck", s)


@_step("alias_teilig")
def _alias_teilig(s):
    return re.sub(r"(?<=\d)tlg\b", "teilig", s)


# ==========================================================================
# profiles
# ==========================================================================

# The FLOOR is the only thing ever applied to an EMITTED value: encoding
# hygiene, nothing semantic.
#
# NFC, not NFKC: NFKC rewrites `mm²`->`mm2`, `½`, fullwidth forms and ligatures,
# which in a B2B catalog are frequently the product's actual spelling. SPLADE's
# normalize_text+fold_de chain is NFC, so emitting NFKC would be gratuitous
# train/serve skew against a model that costs a retrain to move.
#
# demojibake runs FIRST. In norm() it ran after NFKC, which decomposes U+00BC
# and destroys the latin-1 round-trip signature -- so the repair silently never
# fired for the common German case ('WÃ¼rth' -> 'w 1 4rth'). The defect is
# preserved in _COMPARE_V0_STEPS for the golden gate; _COMPARE_STEPS carried it
# too until MXG-52 put demojibake first there as well.
#
# strip_markup (MXG-64) is the SECOND repair in the floor, and the reason the
# floor is analyzer-neutral only on WELL-FORMED input: demojibake already
# changes index tokens for `WÃ¼rth`, and this one does for `Sch&uuml;rze`. That
# is the point of both -- see --selftest step (3a), whose probes are all
# well-formed, and step (6), which asserts the repairs move tokens.
#
# It runs after demojibake (a mojibaked entity must be repaired before it can
# be recognised) and BEFORE strip_controls, so a decoded `&nbsp;` -> U+00A0 is
# then stripped by the step that already handles NBSP.
FLOOR_STEPS = ("demojibake", "strip_markup", "nfc", "strip_controls",
               "collapse_ws")

# Every one of norm()'s 19 steps, in its original order. Never called in anger
# -- this exists so the port can be proven byte-exact against the golden file
# captured before the refactor, and stays as a permanent regression fence.
_COMPARE_V0_STEPS = (
    "nfkc", "demojibake_v0", "strip_tm", "lower", "expand_abbrev",
    "sym_diameter", "decimal_comma", "times_to_x", "glue_dim",
    "sym_inch", "sym_degree", "fold_umlaut", "charclass",
    "prune_period", "prune_slash", "collapse_ws",
    "glue_unit", "alias_stueck", "alias_teilig", "collapse_ws",
)

# The 10 survivors, plus fold_accents (MXG-13) and strip_markup (MXG-64).
# Nine steps are DELETED because each is COARSER than
# `german_strict` -- it merges two forms the index keeps apart, which licenses
# a dedup drop that deletes an index token the document has nowhere else:
#
#   expand_abbrev  ES holds {verz_, verz}, never {verzinkt}
#   sym_diameter   ES never holds `durchmesser`; this is expansion, not
#   sym_inch       normalization -- see rule 4 in the module docstring
#   sym_degree
#   times_to_x     ES: `200×200` -> {200_200, 200200}; `200x200` -> {200x200}
#   glue_dim       ES: `8 x 25 mm` -> [8,x,25,mm]; `8x25mm` -> [8x25mm].
#   glue_unit        THE MEASURED DEFECT: 0.50% (random) / 1.07% (v2.1) of
#                    keyword-carrying articles lose an index token, probed
#                    end-to-end as `8x25` HIT -> MISS (kw_subset_token_loss.py)
#   alias_stueck   no analyzer counterpart at all
#   alias_teilig
#
# The survivors are safe because each is finer-or-equal to the analyzer:
# ES folds Apfel/Äpfel/aepfel all to `apfel` while this profile keeps `apfel`
# and `aepfel` apart (under-merge); ES maps `1,5` and `1.5` to the same pair
# {1_5, 15}, so conflating them is exactly analyzer-granular.
#
# demojibake runs before nfkc (MXG-52): NFKC decomposes U+00BC, which is what
# `ü` double-encodes to, so with the port's original order the repair could
# never fire for the modal German case.
#
# fold_accents runs before charclass (MXG-13 §1): otherwise a non-German
# accented letter becomes a space and splits the word around it.
#
# strip_markup is in BOTH profiles (MXG-64). It is in the floor because the
# emitted value is the thing that has to be clean; it is here as well because
# callers apply norm_compare() directly to a raw value -- feat_kw_audit imports
# it as `norm` -- and without it dedup would go on comparing the invented
# tokens `p reduzierung rohr p`. It is the one step whose presence here is not
# a claim about being finer than german_strict: post-strip, the analyzer reads
# the stripped text too, so the two sides stay aligned by construction.
_COMPARE_STEPS = (
    "demojibake", "strip_markup", "nfkc", "strip_tm", "lower", "decimal_comma",
    "fold_umlaut", "fold_accents", "charclass", "prune_period", "prune_slash",
    "collapse_ws",
)


def floor(s):
    """Encoding hygiene. The ONLY normalization applied to an emitted value."""
    return _run(FLOOR_STEPS, s)


def norm_compare(s):
    """The comparison form. NEVER EMITTED.

    Any caller that renders this into indexed, encoded or CE-visible text is a
    bug -- it is deliberately lossy in ways the consumers are not.
    """
    return _run(_COMPARE_STEPS, s)


def toks_compare(s):
    return norm_compare(s).split()


def compare_sorted(s):
    """Order-normalized comparison form, for set-valued fields whose records
    store the same members in different orders. Never emitted."""
    return " ".join(sorted(toks_compare(s)))


def norm_compare_v0(s):
    """Pre-MXG-47 norm(). Regression fence only -- do not call in anger."""
    return _run(_COMPARE_V0_STEPS, s)


# ==========================================================================
# the analyzer side
# ==========================================================================
_ANALYZE_CACHE = {}

# `_analyze` truncates nothing for us; keep the request bounded ourselves.
_ANALYZE_MAXLEN = 30000

# A single `_analyze` request is capped at `index.analyze.max_token_count`
# (10,000 by default, not overridden on this index). Past it ES answers 400 and
# NOTHING in the response says so. Chunk by an expected-token budget with room
# to spare, and let `analyze_chunks` bisect when a chunk still overshoots.
ANALYZE_TOKEN_BUDGET = 3000


def _value_starts(sent):
    """Character offset at which each SENT text begins in `_analyze`'s layout.

    `_analyze` over an array lays the values out end to end with ONE separator
    character between them, and reports every token's offsets in those
    coordinates. Verified 2026-08-09 against all six chains this repo uses
    (seg, german_strict, german_strict_joined, german_company,
    german_text_decompounded, german_stemmed): 0 violations.
    """
    starts, acc = [], 0
    for t in sent:
        starts.append(acc)
        acc += len(t) + 1
    return starts


def _split_by_offset(sent, tokens):
    """Attribute each token to the value it came from, by start_offset.

    This is the ONLY safe partition. The obvious alternative -- counting
    `position_increment_gap` (100) jumps -- cannot see a value that analyzed to
    NO tokens, and zero-token values are ordinary here: `remove_single_characters`
    (length min=2) empties 'a', '-' and 'x' outright. Worse, it had no way to
    count a LEADING empty value at all, so one 'a' at the head of a batch shifted
    every later value's tokens by one (MXG-27). Offsets have no such blind spot.
    """
    out = [set() for _ in sent]
    starts = _value_starts(sent)
    ends = [s + len(t) for s, t in zip(starts, sent)]
    for tok in tokens:
        off = tok["start_offset"]
        i = bisect.bisect_right(starts, off) - 1
        if not (0 <= i < len(out) and tok["end_offset"] <= ends[i]):
            # Our model of the array layout is wrong, which means every token
            # attribution in this response is suspect. Never guess.
            raise RuntimeError(
                f"_analyze offset {off}-{tok['end_offset']} for "
                f"{tok['token']!r} falls outside every sent value; the array "
                f"layout assumption in _value_starts() no longer holds")
        out[i].add(tok["token"])
    return out


def _analyze_error(exc, analyzer, n):
    """A failure message that names the cause instead of hiding it."""
    detail = ""
    try:  # HTTPError is a response: the body says WHY, the status does not
        detail = exc.read().decode("utf-8", "replace")[:400]
    except Exception:
        pass
    return RuntimeError(
        f"_analyze failed for {n} text(s) on {analyzer!r}: "
        f"{type(exc).__name__}: {exc}{(' -- ' + detail) if detail else ''}")


_SPEC_KEYS = ("tokenizer", "filter", "char_filter")


def _analyze_key(analyzer, spec=None):
    """-> (cache key, `_analyze` request fields) for one chain.

    `_analyze` takes either a NAMED analyzer or an inline `{tokenizer, filter,
    char_filter}` chain, and the inline form is what makes a candidate chain
    priceable without building an index for it. Index-scoped `_analyze`
    resolves the filter names against that index's settings, so an inline spec
    can reuse `company_common` and friends by name.

    Two rules the key has to keep:
      * a dict is unhashable, so a spec cannot be the key itself -- it would
        TypeError at the cache lookup before any request is made;
      * a named analyzer and an inline chain must never collide, or a candidate
        would silently read the baseline's tokens. Hence the `~inline:` prefix,
        which no analyzer name can spell, over canonical JSON.
    """
    if spec is None:
        return analyzer, {"analyzer": analyzer}
    bad = set(spec) - set(_SPEC_KEYS)
    if bad or "tokenizer" not in spec:
        raise ValueError(
            f"inline analyzer spec must have a 'tokenizer' and only "
            f"{list(_SPEC_KEYS)}; got keys {sorted(spec)}")
    body = {k: spec[k] for k in _SPEC_KEYS if k in spec}
    return ("~inline:" + json.dumps(body, sort_keys=True, ensure_ascii=False),
            dict(body))


def analyze_batch(texts, analyzer, es_request=None, index=None, strict=False,
                  spec=None):
    """Index tokens for MANY texts in ONE `_analyze` call. Returns one set per
    input text, in order. Memoized on (analyzer, text) like `analyze_tokens`.

    `spec` sends an INLINE chain instead of a named analyzer (see
    `_analyze_key`); `analyzer` then only labels the call in error messages,
    and the cache is keyed on the chain itself so two labels for one chain
    share it and one label for two chains cannot.

    `_analyze` accepts an array and treats it as a multi-valued field; tokens
    are attributed back to their value by OFFSET (see `_split_by_offset`).

    This is what makes an analyzer-side test affordable per keyword member
    rather than per document: one round trip for the whole article.

    `strict=True` raises instead of caching an empty set when the request keeps
    failing. Prefer it whenever an empty token set would be read as a fact
    about the text rather than as a broken request -- a veto that passes
    vacuously, a feature that silently reads 0. It cannot be bolted on from
    outside: the empty set is cached in HERE, so a retry returns the poisoned
    value without touching ES, and `set()` is a legitimate output anyway
    ('a' under length(min=2) analyzes to nothing), so no caller can tell the
    two apart by inspection.
    """
    texts = [str(t or "") for t in texts]
    key, req = _analyze_key(analyzer, spec)
    todo = [t for t in dict.fromkeys(texts)
            if t.strip() and (key, t) not in _ANALYZE_CACHE]
    if todo:
        # `index` must be defaulted even when a caller supplies its own
        # es_request -- otherwise the URL is `/None/_analyze`, every attempt
        # 404s, and the handler below caches an EMPTY token set for every
        # text. Empty tokens make an analyzer veto vacuously pass, so this
        # failure mode is silent in exactly the direction that matters.
        if es_request is None or index is None:
            from common import es_request as _req, ES_INDEX as _idx
            es_request = es_request or _req
            index = index or _idx
        sent = [t[:_ANALYZE_MAXLEN] for t in todo]
        for attempt in range(4):
            try:
                r = es_request(f"/{index}/_analyze", {**req, "text": sent},
                               timeout=120)
                for t, s in zip(todo, _split_by_offset(sent, r["tokens"])):
                    _ANALYZE_CACHE[(key, t)] = s
                break
            except Exception as e:
                if attempt == 3:
                    if strict:
                        raise _analyze_error(e, analyzer, len(sent)) from e
                    for t in todo:
                        _ANALYZE_CACHE[(key, t)] = set()
    return [_ANALYZE_CACHE.get((key, t), set()) for t in texts]


def analyze_chunks(texts, analyzer, es_request=None, index=None,
                   budget=ANALYZE_TOKEN_BUDGET, per_text=None, log=None,
                   spec=None):
    """`analyze_batch` over an arbitrarily long list, in `strict` chunks.

    Chunk size is an expected-TOKEN budget, not a value count: names run ~6-10
    tokens each, so a batch of 2,000 blows the 10,000-token request cap and the
    only symptom is a 400. `per_text` is the expected tokens per value (measure
    it once for your field; the default is deliberately pessimistic).

    On failure the chunk is BISECTED and retried, so an oversized single value
    is named in the error rather than taking the whole run down with it.

    Returns {text: tokens} for the distinct non-blank inputs -- the lookup a
    prefetch wants, so no caller has to reach into the module cache.
    """
    per_text = per_text or 12
    size = max(1, int(budget // per_text))
    key, _ = _analyze_key(analyzer, spec)
    uniq = [t for t in dict.fromkeys(str(t or "") for t in texts) if t.strip()]
    for i in range(0, len(uniq), size):
        _analyze_chunk(uniq[i:i + size], analyzer, es_request, index, spec)
        if log and i and (i // size) % 50 == 0:
            log(f"  analyzed {i:,}/{len(uniq):,} distinct values ({analyzer})")
    return {t: _ANALYZE_CACHE[(key, t)] for t in uniq}


def _analyze_chunk(chunk, analyzer, es_request, index, spec=None):
    try:
        analyze_batch(chunk, analyzer, es_request, index, strict=True,
                      spec=spec)
    except Exception:
        if len(chunk) == 1:
            raise
        mid = len(chunk) // 2
        _analyze_chunk(chunk[:mid], analyzer, es_request, index, spec)
        _analyze_chunk(chunk[mid:], analyzer, es_request, index, spec)


def analyze_tokens(text, analyzer, es_request=None, index=None, strict=False,
                   spec=None):
    """Index tokens ES would actually produce. Memoized on (analyzer, text).

    Any rule that drops a value as "already present elsewhere in the document"
    must be tested HERE, not on norm_compare() tokens -- that is the whole
    lesson of kw_subset_token_loss.py.

    See `analyze_batch` for what `strict` and `spec` buy and why they belong
    in here.
    """
    text = str(text or "")
    if not text.strip():
        return set()
    chain, req = _analyze_key(analyzer, spec)
    key = (chain, text)
    if key in _ANALYZE_CACHE:
        return _ANALYZE_CACHE[key]
    if es_request is None or index is None:
        from common import es_request as _req, ES_INDEX as _idx
        es_request = es_request or _req
        index = index or _idx
    for attempt in range(4):
        try:
            r = es_request(f"/{index}/_analyze",
                           {**req, "text": text[:_ANALYZE_MAXLEN]},
                           timeout=60)
            out = {t["token"] for t in r["tokens"]}
            _ANALYZE_CACHE[key] = out
            return out
        except Exception as e:
            if attempt == 3:
                if strict:
                    raise _analyze_error(e, analyzer, 1) from e
                return set()
    return set()


# ==========================================================================
# --selftest: prove profile B must be the identity
#
# idnorm.py --selftest asserts that Python/analyzer parity IS ACHIEVED.
# This one asserts the opposite, for text: that parity is ILL-TYPED, so no
# non-identity profile B can be correct. It is a proof obligation, not a diff.
# ==========================================================================
CHAINS = ("german_strict", "german_strict_joined", "german_company")
SNAPSHOT = os.path.join(HERE, "tests", "fixtures", "analyzer_snapshot.json")

# (1) DUAL EMISSION -- >=2 tokens at the same position.
# (2) PUNCTUATION BECOMES `_`, NOT NOTHING -- unify_non_letters replaces.
OBSTRUCTION_PROBES = ["verz. Schraube", "VHM-Bohrer", '1/4"', "3 Stk.",
                      "1.000 mm", "25°", "A2-70"]

# (3) THE FLOOR IS ANALYZER-NEUTRAL, and the aggressive profile is not.
NEUTRAL_PROBES = ["Größe 10", "groesse 10", "Kühlschrank", "kuehlschrank",
                  "PHOENIX", "Phoenix", "Äpfel", "aepfel", "Straße",
                  "Dichtring 8 x 25 mm", "Sechskantschraube M8 verzinkt"]

# Strings where norm_compare_v0 is COARSER than the analyzer and therefore
# destroys index tokens. Each must lose at least one token.
LOSS_PROBES = ["Dichtring 8 x 25 mm", "verz. Schraube", '1/4"', "3 Stk.",
               "200×200", "Ø 25 mm", "25°", "VHM-Bohrer"]

# Known residual: `charclass` turns a non-alphanumeric infix into a SPACE while
# `unify_non_letters` turns it into `_` and joined_words_strict catenates, so
# `200×200` -> {200_200, 200200} but the compare form `200 200` -> {200}. The
# whole of position 0 goes. This is the one place the surviving profile is
# still coarser than the analyzer; M2 (kw_subset_token_loss.py) measures how
# often it bites on real data. Listed here so it is a known quantity rather
# than a surprise.
RESIDUAL_COARSE = {"200×200"}

# (5) THE ACCENT FOLD IS ANALYZER-GRANULAR, per character.
# Every non-ASCII letter that occurs in the catalog: the 112 found by a census
# of 126,443 records (the 50k golden strings plus the v21 and rand frames of
# feat_kw_audit), ordered by frequency, plus the rest of _NO_DECOMPOSITION.
# `fold_accents` must agree with icu_folding on each one, or leave it to
# `charclass` where ES has no ASCII base for it either.
# (6) THE MARKUP STRIP IS A REPAIR -- it must MOVE index tokens, and in the
# direction that restores what the clean spelling would have held. Each pair is
# (marked-up form, the spelling the vendor meant); the strip must make the
# first analyze like the second. `mm<sup>2</sup>` is the one where the marked-up
# form loses the unit entirely, and `Sch&uuml;rze` the one where nothing the
# document holds can be queried for at all.
MARKUP_PROBES = [
    ("Reiniger<br>Universal", "Reiniger Universal"),
    ("Stahl<br>verzinkt", "Stahl verzinkt"),
    ("H<sub>2</sub>O Filter", "H2O Filter"),
    ("mm<sup>2</sup>", "mm2"),
    ("Sch&uuml;rze", "Schürze"),
    ("Gr&#246;&#223;e 10", "Größe 10"),
    ("<p>Reduzierung, Rohr</p>", "Reduzierung, Rohr"),
    ("Kabelkanal mm&lt;sup&gt;2&lt;/sup&gt;", "Kabelkanal mm2"),
    ("u.lt;p u.gt;Reduzierung", "Reduzierung"),
]

# The guard behind the whitelist: `<` and `>` are comparison operators in this
# catalog far more often than tag delimiters, so these must analyze IDENTICALLY
# before and after the strip -- the strip must not touch them at all.
MARKUP_GUARD_PROBES = ["Stahl < 1000 N/mm²", "Alu > 8% Si",
                       "Zeit <10 min, dann >480 min", "Wert <b 5 und >10",
                       "Schrauben u. Muttern", "R&B GmbH"]

ACCENT_CENSUS = (
    "éáóØíÃłőęàąÉřýúűâčěśøµšżèžêôćÁΩçツΩůŸľťŠîμňńÂăŚïŘțĺìñÚŻÍŁëČźÓșœŕæŽÎακλû"
    "ﾞΔºŐõﾚȚŰмĎùÀₘĄĘβŹŃÑďÅÌȘÈʺˆŤÔÝĽеƒËðãΜΑåЛΒòﾊ" + "".join(_NO_DECOMPOSITION)
)


def _probe_fold(es_request, index, ch):
    """What german_strict folds a single letter to, or None if it has no ASCII
    base there. `q`/`z` are sentinels: both survive every filter in the chain
    unchanged, so the probe word cannot be confused with its own fold."""
    r = es_request(f"/{index}/_analyze",
                   {"analyzer": "german_strict", "text": f"q{ch}z"}, timeout=60)
    inner = {t["token"][1:-1] for t in r["tokens"]
             if t["token"].startswith("q") and t["token"].endswith("z")}
    if len(inner) != 1:
        return None
    got = inner.pop()
    return got if got.isascii() and got.isalpha() else None


def _positions(es_request, index, text, analyzer):
    r = es_request(f"/{index}/_analyze",
                   {"analyzer": analyzer, "text": text}, timeout=60)
    return [(t["token"], t["position"]) for t in r["tokens"]]


def _selftest(write_snapshot=False):
    from common import es_request, ES_INDEX
    import datetime
    snap, fails = {}, []

    def check(ok, label, detail=""):
        print(f"  {'OK  ' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(label)

    print(f"index = {ES_INDEX}\n")

    print("(1)+(2) parity is ill-typed -- dual emission and `_` punctuation")
    for s in OBSTRUCTION_PROBES:
        toks = _positions(es_request, ES_INDEX, s, "german_strict")
        snap.setdefault(s, {})["german_strict"] = toks
        by_pos = {}
        for t, p in toks:
            by_pos.setdefault(p, []).append(t)
        dual = any(len(v) > 1 for v in by_pos.values())
        under = any("_" in t for t, _ in toks)
        check(dual or under, repr(s),
              f"-> {[t for t, _ in toks]}"
              f"{'  [dual]' if dual else ''}{'  [_]' if under else ''}")

    print("\n(3a) the floor changes NO index token, on any chain")
    for s in NEUTRAL_PROBES:
        for an in CHAINS:
            a = analyze_tokens(s, an, es_request, ES_INDEX)
            b = analyze_tokens(floor(s), an, es_request, ES_INDEX)
            snap.setdefault(s, {}).setdefault(an, sorted(a))
            check(a == b, f"{s!r} / {an}", "" if a == b else f"{a} != {b}")

    print("\n(3b) the analyzer already folds -- a Python fold buys 0 tokens")
    for a, b in [("Größe 10", "groesse 10"), ("Kühlschrank", "kuehlschrank"),
                 ("Äpfel", "aepfel")]:
        ta = analyze_tokens(a, "german_strict", es_request, ES_INDEX)
        tb = analyze_tokens(b, "german_strict", es_request, ES_INDEX)
        check(ta == tb, f"{a!r} == {b!r} under german_strict", f"-> {sorted(ta)}")

    print("\n(3c) the aggressive profile IS coarser -- it destroys these tokens")
    for s in LOSS_PROBES:
        raw = analyze_tokens(s, "german_strict", es_request, ES_INDEX)
        agg = analyze_tokens(norm_compare_v0(s), "german_strict",
                             es_request, ES_INDEX)
        lost = raw - agg
        check(bool(lost), repr(s),
              f"loses {sorted(lost)}  (v0 -> {norm_compare_v0(s)!r})")

    # A lost token is only HARMFUL if nothing survives at its position.
    # keyword_repeat and joined_words_strict emit siblings at one position, so
    # losing `vhm_bohrer` while keeping `vhm` cannot break a conjunction --
    # ES matches them as a synonym group. Losing every token at a position can.
    print("\n(4) the surviving profile empties no position, except the known"
          " residual")
    for s in LOSS_PROBES:
        raw_pos = {}
        for t, p in _positions(es_request, ES_INDEX, s, "german_strict"):
            raw_pos.setdefault(p, set()).add(t)
        new = analyze_tokens(norm_compare(s), "german_strict",
                             es_request, ES_INDEX)
        emptied = {p: sorted(v) for p, v in raw_pos.items() if not (v & new)}
        expected = s in RESIDUAL_COARSE
        check(bool(emptied) == expected, repr(s),
              f"-> {norm_compare(s)!r}  "
              + (f"empties {emptied}" + (" [known residual]" if expected else "")
                 if emptied else "no position emptied"))

    print("\n(6) the markup strip restores the tokens the clean spelling holds")
    for marked, clean in MARKUP_PROBES:
        got = analyze_tokens(floor(marked), "german_strict", es_request,
                             ES_INDEX)
        want = analyze_tokens(clean, "german_strict", es_request, ES_INDEX)
        raw = analyze_tokens(marked, "german_strict", es_request, ES_INDEX)
        for s, an in ((marked, "german_strict"),
                      (marked, "german_strict_joined"),
                      (clean, "german_strict"),
                      (clean, "german_strict_joined")):
            snap.setdefault(s, {}).setdefault(
                an, sorted(analyze_tokens(s, an, es_request, ES_INDEX)))
        check(got == want, repr(marked),
              f"-> {sorted(got)}"
              + ("" if got == want else f"  != clean {sorted(want)}")
              + f"  [marked-up form held {sorted(raw)}]")

    print("\n(6b) ...and leaves a comparison operator alone, on every chain")
    for s in MARKUP_GUARD_PROBES:
        for an in CHAINS:
            a = analyze_tokens(s, an, es_request, ES_INDEX)
            b = analyze_tokens(floor(s), an, es_request, ES_INDEX)
            snap.setdefault(s, {}).setdefault(an, sorted(a))
            check(a == b and floor(s) == _collapse_ws(s), f"{s!r} / {an}",
                  "" if a == b else f"{sorted(a)} != {sorted(b)}")

    print("\n(5) the accent fold matches icu_folding, per character")
    folds, bad = {}, []
    with ThreadPoolExecutor(16) as pool:
        es_folds = dict(pool.map(
            lambda ch: (ch, _probe_fold(es_request, ES_INDEX, ch)),
            sorted(set(ACCENT_CENSUS))))
    for ch in sorted(set(ACCENT_CENSUS)):
        es = es_folds[ch]
        ours = _fold_accents(ch.lower())
        folds[ch] = {"es": es, "ours": ours}
        # ES has an ASCII base  -> we must produce exactly it.
        # ES has none (`_`, '') -> we must leave the character to charclass,
        #                          which is what the profile did before.
        ok = (ours == es) if es is not None else (ours == ch.lower())
        if not ok:
            bad.append((ch, es, ours))
    check(not bad, f"{len(folds)} catalog letters",
          "" if not bad else f"disagree: {bad[:8]}")
    snap["_accent_folds"] = folds

    if write_snapshot:
        os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
        with open(SNAPSHOT, "w") as f:
            json.dump({"index": ES_INDEX,
                       "captured": datetime.date.today().isoformat(),
                       "chains": list(CHAINS), "probes": snap}, f,
                      indent=1, ensure_ascii=False, sort_keys=True)
        print(f"\nwrote {SNAPSHOT}")

    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--write-snapshot", action="store_true")
    a = ap.parse_args()
    if a.selftest or a.write_snapshot:
        sys.path.insert(0, HERE)
        sys.exit(_selftest(a.write_snapshot))
    ap.print_help()


if __name__ == "__main__":
    main()
