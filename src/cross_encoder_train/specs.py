import re


# Each entry: name -> compiled regex that extracts a normalized "spec token"
# from query text. The match's group(0) (lowercased, whitespace-stripped) is
# the canonical spec string we then look for in offer text.
#
# Rules mirror the production-query analysis (90-day distribution).
# Excluded as too noisy: a2_a4, kelvin, metre (\d+m), tolerance.
RULES = {
    "thread_m": re.compile(r"(?i)\bm\d{1,3}(?:[x×]\d+(?:[.,]\d+)?){0,2}\b"),
    "g_thread": re.compile(r"(?i)\bg\d+(?:/\d+)?\b"),
    "dimensions": re.compile(r"\b\d+(?:[.,]\d+)?[x×]\d+(?:[.,]\d+)?(?:[x×]\d+(?:[.,]\d+)?)?\b"),
    "fraction": re.compile(r"\b\d+/\d+\b"),
    "decimal_de": re.compile(r"\b\d+,\d+\b"),
    "mm": re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s*mm\b"),
    "cm": re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s*cm\b"),
    "micrometre": re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s*(?:µm|um)\b"),
    "cross_sect": re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s*mm[²2]\b"),
    "inch_zoll": re.compile(r"(?i)\b\d+(?:/\d+)?\s*(?:zoll|″|\")\b"),
    "voltage": re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s*v(?:olt)?\b"),
    "ampere": re.compile(r"(?i)(?<![a-z0-9])\d+(?:[.,]\d+)?\s*a(?![a-z0-9])"),
    "ah": re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s*ah\b"),
    "watt": re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s*w(?:att)?\b"),
    "hz": re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s*hz\b"),
    "volume_l": re.compile(r"(?i)(?<![a-z0-9])\d+(?:[.,]\d+)?\s*l(?![a-z0-9])"),
    "mass_kg": re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s*kg\b"),
    "pressure": re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s*(?:bar|psi|mpa)\b"),
    "din": re.compile(r"(?i)\bdin\s*\d+\b"),
    "iso": re.compile(r"(?i)\biso\s*\d+\b"),
    "en_norm": re.compile(r"(?i)\ben\s*\d{3,5}\b"),
    "strength_cls": re.compile(r"\b(?:[4-9]|1[0-2])\.[0-9]\b"),
    "stainless": re.compile(r"(?i)\b(?:1\.4\d{3}|v[24]a)\b"),
    "ral_color": re.compile(r"(?i)\bral\s*\d{3,4}\b"),
    "ip_rating": re.compile(r"(?i)\bip[\s-]?\d{2}[a-z]?\b"),
    "cat_rating": re.compile(r"(?i)\bcat[\s.-]?[1-7][a-z]?\b"),
    "dn_nw_pn": re.compile(r"(?i)\b(?:dn|nw|pn)\s*\d+\b"),
    "pg_gland": re.compile(r"(?i)\bpg\s*\d+\b"),
    "awg": re.compile(r"(?i)\bawg\s*\d+\b"),
    "lumen_lux": re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s*(?:lm|lx|lux)\b"),
    "temp_c": re.compile(r"(?i)\b-?\d+(?:[.,]\d+)?\s*°?c\b"),
    "atex": re.compile(r"(?i)\batex\b"),
    "nato_nema": re.compile(r"(?i)\b(?:nato|nema)[\s-]?[a-z0-9]+\b"),
}


def canonicalize(token):
    return "".join(token.lower().split())


def extract(query, rule_names):
    found = []
    seen = set()
    for name in rule_names:
        rule = RULES.get(name)
        if rule is None:
            continue
        for match in rule.finditer(query):
            canonical = canonicalize(match.group(0))
            if canonical and canonical not in seen:
                seen.add(canonical)
                found.append(canonical)
    return found
