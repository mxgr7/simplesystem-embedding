import math
import re
import unicodedata
from collections import Counter

from cross_encoder_train import specs


_DEFAULT_SLOT_ORDER = ("ean", "article", "spec")

_TOKEN_PREFIX = {
    "ean": "EAN",
    "article": "ART",
    "spec": "SPEC",
}

# Per-slot state vocabularies. The `article` slot collapses the former
# `article` (exact) and `shape` (substring) slots into a single 5-state
# variable: EXACT subsumes SUBSTRING_ONLY (every exact match is also a
# substring match), and OFFER_INVALID is now an explicit state rather than a
# policy-controlled fallback to NONE/MISMATCH.
_SLOT_STATES = {
    "ean": ("NONE", "MATCH", "MISMATCH"),
    "article": ("NONE", "EXACT", "SUBSTRING_ONLY", "MISMATCH", "OFFER_INVALID"),
    "spec": ("NONE", "MATCH", "MISMATCH"),
}

_DIGIT_RUN = re.compile(r"\d+")
_ALNUM_TOKEN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/\-]*[A-Za-z0-9])?")


def _cfg_get(section, key, default=None):
    if section is None:
        return default
    if hasattr(section, "get"):
        return section.get(key, default)
    return getattr(section, key, default)


def _slot_enabled(features_cfg, slot):
    section = _cfg_get(features_cfg, slot)
    if section is None:
        return False
    return bool(_cfg_get(section, "enabled", False))


def feature_token_names(features_cfg):
    if features_cfg is None or not bool(_cfg_get(features_cfg, "enabled", False)):
        return []
    slot_order = list(_cfg_get(features_cfg, "slot_order", _DEFAULT_SLOT_ORDER))
    tokens = []
    for slot in slot_order:
        prefix = _TOKEN_PREFIX.get(slot)
        if prefix is None:
            continue
        if not _slot_enabled(features_cfg, slot):
            continue
        for state in _SLOT_STATES[slot]:
            tokens.append(f"[{prefix}_{state}]")
    return tokens


def _norm_unicode(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if not isinstance(value, str):
        value = str(value)
    return unicodedata.normalize("NFKC", value)


def _norm_id(value, strip_chars="-_/. ", strip_leading_zeros=False):
    text = _norm_unicode(value).lower()
    for ch in strip_chars:
        text = text.replace(ch, "")
    if strip_leading_zeros and text:
        stripped = text.lstrip("0")
        text = stripped or text
    return text


def _split_multivalue(value, separators=",;|"):
    text = _norm_unicode(value)
    if not text:
        return []
    pattern = "[" + re.escape(separators) + "]"
    return [p.strip() for p in re.split(pattern, text) if p.strip()]


def _gtin_checksum_ok(digits):
    body = digits[:-1]
    check = int(digits[-1])
    total = 0
    for index, ch in enumerate(reversed(body)):
        weight = 3 if index % 2 == 0 else 1
        total += int(ch) * weight
    return (10 - total % 10) % 10 == check


def _validate_ean(value, require_checksum=True):
    text = _norm_unicode(value)
    digits = "".join(c for c in text if c.isdigit())
    if len(digits) not in (8, 12, 13, 14):
        return None
    if require_checksum and not _gtin_checksum_ok(digits):
        return None
    return digits


def _validate_alnum_id(value, min_len=4, max_len=32):
    text = _norm_id(value)
    if not (min_len <= len(text) <= max_len):
        return None
    if not any(c.isdigit() for c in text):
        return None
    return text


def _query_ean_candidate(query):
    for match in _DIGIT_RUN.finditer(query or ""):
        run = match.group(0)
        if len(run) in (8, 12, 13, 14):
            return run
    return None


def _query_id_candidates(query, min_len=4):
    if not query:
        return []
    seen = set()
    candidates = []
    for token in _ALNUM_TOKEN.findall(query):
        normalized = _norm_id(token)
        if len(normalized) < min_len:
            continue
        if not any(c.isdigit() for c in normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(normalized)
    return candidates


def _emit(prefix, state):
    return f"[{prefix}_{state}]"


def _on_invalid_policy(section):
    return str(_cfg_get(section, "on_offer_invalid", "none")).lower()


class FeatureExtractor:
    def __init__(self, features_cfg):
        self.cfg = features_cfg
        slot_order = list(_cfg_get(features_cfg, "slot_order", _DEFAULT_SLOT_ORDER))
        self.slot_order = [s for s in slot_order if _slot_enabled(features_cfg, s)]

        self.ean_cfg = _cfg_get(features_cfg, "ean")
        self.article_cfg = _cfg_get(features_cfg, "article")
        self.spec_cfg = _cfg_get(features_cfg, "spec")

        self.spec_rule_names = list(_cfg_get(self.spec_cfg, "rules", []) or [])

        normalize_cfg = _cfg_get(features_cfg, "normalize") or {}
        self.strip_leading_zeros = (
            str(_cfg_get(normalize_cfg, "leading_zeros", "keep")).lower() == "strip"
        )
        self.multivalue_separators = str(
            _cfg_get(normalize_cfg, "multivalue_separators", ",;|")
        )

        self.stats = Counter()
        self.rows_seen = 0

    def feature_token_count(self):
        return sum(len(_SLOT_STATES[s]) for s in self.slot_order)

    def token_strings(self):
        tokens = []
        for slot in self.slot_order:
            prefix = _TOKEN_PREFIX[slot]
            for state in _SLOT_STATES[slot]:
                tokens.append(_emit(prefix, state))
        return tokens

    def extract(self, context):
        self.rows_seen += 1
        out = []
        for slot in self.slot_order:
            if slot == "ean":
                out.append(self._extract_ean(context))
            elif slot == "article":
                out.append(self._extract_article(context))
            elif slot == "spec":
                out.append(self._extract_spec(context))
        return out

    def _record(self, slot, key):
        self.stats[f"{slot}/{key}"] += 1

    def _query(self, context):
        return context.get("query_term", "") or ""

    def _field(self, context, name):
        value = context.get(name)
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return value

    def _extract_ean(self, context):
        prefix = _TOKEN_PREFIX["ean"]
        cfg = self.ean_cfg
        offer_field = _cfg_get(cfg, "offer_field", "ean")
        validate_mode = str(_cfg_get(cfg, "validate", "gtin")).lower()
        require_checksum = validate_mode == "gtin"

        query_candidate = _query_ean_candidate(self._query(context))
        if query_candidate is None:
            self._record("ean", "query_none")
            return _emit(prefix, "NONE")
        self._record("ean", "query_present")

        validated = _validate_ean(
            self._field(context, offer_field), require_checksum=require_checksum
        )
        if validated is None:
            self._record("ean", "offer_invalid")
            if _on_invalid_policy(cfg) == "mismatch":
                return _emit(prefix, "MISMATCH")
            return _emit(prefix, "NONE")
        self._record("ean", "offer_valid")

        if validated == query_candidate:
            self._record("ean", "match")
            return _emit(prefix, "MATCH")
        self._record("ean", "mismatch")
        return _emit(prefix, "MISMATCH")

    def _extract_article(self, context):
        prefix = _TOKEN_PREFIX["article"]
        cfg = self.article_cfg
        offer_fields = list(
            _cfg_get(
                cfg,
                "offer_fields",
                ("article_number", "manufacturer_article_number"),
            )
        )
        min_len = int(_cfg_get(cfg, "min_token_len", 4) or 4)

        query_candidates = _query_id_candidates(self._query(context), min_len=min_len)
        if not query_candidates:
            self._record("article", "query_none")
            return _emit(prefix, "NONE")
        self._record("article", "query_present")

        offer_values = self._collect_offer_ids(context, offer_fields)
        if not offer_values:
            self._record("article", "offer_invalid")
            return _emit(prefix, "OFFER_INVALID")
        self._record("article", "offer_valid")

        offer_set = set(offer_values)
        for cand in query_candidates:
            if cand in offer_set:
                self._record("article", "exact")
                return _emit(prefix, "EXACT")

        for cand in query_candidates:
            for offer_id in offer_values:
                if cand in offer_id or offer_id in cand:
                    self._record("article", "substring_only")
                    return _emit(prefix, "SUBSTRING_ONLY")

        self._record("article", "mismatch")
        return _emit(prefix, "MISMATCH")

    def _collect_offer_ids(self, context, fields):
        out = []
        for field in fields:
            raw = self._field(context, field)
            for part in _split_multivalue(raw, self.multivalue_separators):
                validated = _validate_alnum_id(part)
                if validated is None:
                    continue
                if self.strip_leading_zeros:
                    validated = validated.lstrip("0") or validated
                out.append(validated)
        return out

    def _extract_spec(self, context):
        prefix = _TOKEN_PREFIX["spec"]
        cfg = self.spec_cfg
        offer_fields = list(_cfg_get(cfg, "offer_fields", ("name", "description")))

        query_specs = specs.extract(self._query(context), self.spec_rule_names)
        if not query_specs:
            self._record("spec", "query_none")
            return _emit(prefix, "NONE")
        self._record("spec", "query_present")

        haystack_parts = []
        for field in offer_fields:
            value = self._field(context, field)
            if not value:
                continue
            haystack_parts.append(specs.canonicalize(_norm_unicode(value)))
        haystack = " ".join(haystack_parts)
        if not haystack.strip():
            self._record("spec", "offer_invalid")
            if _on_invalid_policy(cfg) == "mismatch":
                return _emit(prefix, "MISMATCH")
            return _emit(prefix, "NONE")
        self._record("spec", "offer_valid")

        if all(token in haystack for token in query_specs):
            self._record("spec", "match")
            return _emit(prefix, "MATCH")
        self._record("spec", "mismatch")
        return _emit(prefix, "MISMATCH")

    def stats_dict(self):
        out = {f"features/{k}": int(v) for k, v in self.stats.items()}
        out["features/rows_seen"] = int(self.rows_seen)
        out["features/token_count"] = int(self.feature_token_count())
        return out
