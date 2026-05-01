"""Count posthog queries matching each spec-detection rule across the full dataset."""

from __future__ import annotations

import re
from glob import glob
from pathlib import Path

import pyarrow.parquet as pq

DATA_GLOB = "/data/datasets/posthog_queries.parquet/*.parquet"

# Each rule: (name, compiled regex, short description).
# Regexes are applied case-insensitively to the full query string.
RULES: list[tuple[str, re.Pattern[str], str]] = [
    # ---- Formal norms / standards ----
    ("din",         re.compile(r"\bdin\s*(?:en\s*)?(?:iso\s*)?\d", re.I),       "DIN/EN/ISO norm reference"),
    ("iso",         re.compile(r"\biso\s*\d{3,5}\b", re.I),                     "ISO norm number"),
    ("en_norm",     re.compile(r"\ben\s?\d{2,5}\b", re.I),                      "EN norm number"),
    ("ral_color",   re.compile(r"\bral\s?\d{3,4}\b", re.I),                     "RAL color code"),
    ("ip_rating",   re.compile(r"\bip[\s-]?\d{2}\b", re.I),                     "IP ingress rating"),
    ("atex",        re.compile(r"\batex\b", re.I),                              "ATEX zone"),
    ("nato_nema",   re.compile(r"\b(?:nato|nema|nsf)\b", re.I),                 "NATO/NEMA/NSF marker"),

    # ---- Threads / fasteners / pipe sizing ----
    ("thread_m",    re.compile(r"\bm\s?\d{1,3}(?:[.,]\d+)?(?:\s?[x×]\s?\d+)?\b", re.I), "Metric thread Mxx[.x][xL]"),
    ("strength_cls",re.compile(r"\b(?:8\.8|10\.9|12\.9|a2-70|a4-80)\b", re.I),  "Bolt strength class"),
    ("dn_nw_pn",    re.compile(r"\b(?:dn|nw|pn|sw)\s?\d+\b", re.I),             "DN/NW/PN/SW nominal sizing"),
    ("pg_gland",    re.compile(r"\bpg\s?\d+(?:[.,]\d+)?\b", re.I),              "PG cable-gland size"),
    ("g_thread",    re.compile(r"\bg\s?\d+/\d+\b", re.I),                       "G (BSP) pipe thread"),

    # ---- Dimensions / measurements ----
    ("mm",          re.compile(r"\b\d+(?:[.,]\d+)?\s?mm\b", re.I),              "millimetres"),
    ("cm",          re.compile(r"\b\d+(?:[.,]\d+)?\s?cm\b", re.I),              "centimetres"),
    ("metre",       re.compile(r"\b\d+(?:[.,]\d+)?\s?m\b", re.I),               "metres (noisy)"),
    ("micrometre",  re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:µm|um)\b", re.I),       "micrometres"),
    ("inch_zoll",   re.compile(r"\b\d+(?:/\d+)?\s?(?:zoll|\")", re.I),          "inch / zoll"),
    ("dimensions",  re.compile(r"\b\d+(?:[.,]\d+)?\s?[x×]\s?\d+(?:[.,]\d+)?(?:\s?[x×]\s?\d+(?:[.,]\d+)?)?\b", re.I), "AxB(xC) dimensions"),
    ("fraction",    re.compile(r"\b\d+/\d+\b"),                                 "fraction (e.g. 1/2)"),
    ("decimal_de",  re.compile(r"\b\d+,\d+\b"),                                 "German decimal (e.g. 11,6)"),
    ("tolerance",   re.compile(r"\b\d+(?:[.,]\d+)?\s?[hHjJkKmMnN]\d{1,2}\b"),   "tolerance/fit class"),

    # ---- Electrical ----
    ("voltage",     re.compile(r"\b\d+(?:[.,]\d+)?\s?v\b", re.I),               "voltage Vx"),
    ("ampere",      re.compile(r"\b\d+(?:[.,]\d+)?\s?a\b", re.I),               "amperage Ax (noisy)"),
    ("watt",        re.compile(r"\b\d+(?:[.,]\d+)?\s?w\b", re.I),               "wattage Wx"),
    ("ah",          re.compile(r"\b\d+(?:[.,]\d+)?\s?m?ah\b", re.I),            "battery capacity Ah/mAh"),
    ("hz",          re.compile(r"\b\d+(?:[.,]\d+)?\s?[kmg]?hz\b", re.I),        "frequency Hz/kHz/MHz"),
    ("cross_sect",  re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:mm²|qmm)\b", re.I),     "cable cross-section mm²/qmm"),
    ("awg",         re.compile(r"\bawg\s?\d+\b", re.I),                         "AWG"),
    ("cat_rating",  re.compile(r"\bcat\.?\s?\d[a-z]?\b", re.I),                 "Cat.5/6/6a/7"),

    # ---- Fluid / pressure / mass / thermal ----
    ("pressure",    re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:bar|mpa|psi|kpa)\b", re.I), "pressure"),
    ("volume_l",    re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:l|ml|cl)\b", re.I),     "volume L/ml/cl (noisy)"),
    ("mass_kg",     re.compile(r"\b\d+(?:[.,]\d+)?\s?kg\b", re.I),              "mass kg"),
    ("temp_c",      re.compile(r"\b\d+\s?°\s?c\b", re.I),                       "temperature °C"),
    ("kelvin",      re.compile(r"\b\d{3,4}\s?k\b", re.I),                       "color temp / Kelvin"),
    ("lumen_lux",   re.compile(r"\b\d+\s?(?:lm|lux)\b", re.I),                  "lumen / lux"),

    # ---- Material grades ----
    ("stainless",   re.compile(r"\b(?:v[24]a|aisi\s?\d{3}|1\.4\d{3})\b", re.I), "stainless / 1.4xxx / AISI"),
    ("a2_a4",       re.compile(r"\ba[24](?:-\d{2})?\b", re.I),                  "A2/A4 stainless grade (noisy)"),

    # ---- German lexical spec keywords ----
    ("de_keywords", re.compile(r"\b(?:norm|nennweite|gewinde|durchmesser|länge|breite|höhe|stärke|tiefe|außen|innen|ø)\b", re.I), "German spec keyword"),

    # ---- Identifier-shaped (weak spec signal, often pure catalog lookups) ----
    ("ean",         re.compile(r"^\s*\d{12,14}\s*$"),                           "EAN/GTIN"),
    ("art_digits",  re.compile(r"^\s*\d{4,}\s*$"),                              "all-digit article number"),
    ("alnum_punct", re.compile(r"\b[a-z0-9]+(?:[-./_][a-z0-9]+){1,}\b", re.I),  "alnum chunks joined by - . / _"),

    # ---- New (this turn): punctuation- and alpha+digit-shape rules ----
    # any token that mixes letters and digits (e.g. m10, ip55, dsbg40, 5sh5410)
    ("alpha_digit", re.compile(r"\b(?=[a-z0-9]*[a-z])(?=[a-z0-9]*\d)[a-z0-9]+\b", re.I),
                                                                                "token mixing letters and digits"),
    # any token containing internal -, _, /, . between alphanumeric segments (incl. all-numeric, e.g. 25,3x2,4 or k0313.04)
    ("internal_punct", re.compile(r"[A-Za-z0-9][-_/.][A-Za-z0-9]"),             "internal -_/. between alnum"),
    # query contains any of -_/. anywhere
    ("any_spec_punct", re.compile(r"[-_/.]"),                                   "query contains -, _, /, or ."),
]


def main() -> None:
    files = sorted(glob(DATA_GLOB))
    print(f"Files: {len(files)}")

    total = 0
    nonempty = 0
    counts: dict[str, int] = {name: 0 for name, _, _ in RULES}
    union_norms_only = 0   # union of "real" spec rules (excluding the broad punct/alnum rules)
    union_all = 0
    examples: dict[str, list[str]] = {name: [] for name, _, _ in RULES}

    broad_rule_names = {"alnum_punct", "alpha_digit", "internal_punct", "any_spec_punct", "art_digits"}
    narrow_rule_names = [name for name, _, _ in RULES if name not in broad_rule_names]

    for f in files:
        col = pq.read_table(f, columns=["qt"]).column("qt").to_pylist()
        for q in col:
            total += 1
            if not q:
                continue
            nonempty += 1
            qs = q.strip()
            hit_any = False
            hit_narrow = False
            for name, rx, _desc in RULES:
                if rx.search(qs):
                    counts[name] += 1
                    hit_any = True
                    if name in narrow_rule_names:
                        hit_narrow = True
                    if len(examples[name]) < 5:
                        examples[name].append(qs)
            if hit_any:
                union_all += 1
            if hit_narrow:
                union_norms_only += 1

    print(f"Total rows           : {total}")
    print(f"Non-empty queries    : {nonempty}")
    print(f"Union of all rules   : {union_all}  ({100*union_all/nonempty:.1f}% of non-empty)")
    print(f"Union of NARROW rules: {union_norms_only}  ({100*union_norms_only/nonempty:.1f}% of non-empty)")
    print()
    print(f"{'rule':<18} {'count':>10} {'pct_nonempty':>14}   description")
    print("-" * 95)
    for name, _, desc in RULES:
        c = counts[name]
        pct = 100.0 * c / nonempty if nonempty else 0.0
        print(f"{name:<18} {c:>10} {pct:>13.2f}%   {desc}")

    print()
    print("=== examples per rule (up to 5) ===")
    for name, _, _ in RULES:
        ex = examples[name]
        print(f"\n[{name}]")
        for q in ex:
            print("  ", repr(q))


if __name__ == "__main__":
    main()
