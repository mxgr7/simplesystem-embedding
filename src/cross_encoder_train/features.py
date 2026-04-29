import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from cross_encoder_train import specs


_NONE = "NONE"
_MATCH = "MATCH"
_MISMATCH = "MISMATCH"
_STATES = (_NONE, _MATCH, _MISMATCH)

_DEFAULT_SLOT_ORDER = ("ean", "article", "shape", "spec", "brand")
_TOKEN_PREFIX = {
    "ean": "EAN",
    "article": "ART",
    "shape": "SHAPE",
    "spec": "SPEC",
    "brand": "BRAND",
}

_DIGIT_RUN = re.compile(r"\d+")
_ALNUM_TOKEN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/\-]*[A-Za-z0-9])?")
_WORD = re.compile(r"\w+", re.UNICODE)


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
        for state in _STATES:
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


def _validate_brand_offer(value, max_tokens=3, min_len=2, max_len=40):
    text = _norm_unicode(value).strip().lower()
    if not (min_len <= len(text) <= max_len):
        return None
    parts = text.split()
    if len(parts) > max_tokens:
        return None
    if not any(c.isalpha() for c in text):
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


def _query_brand_candidate(query, brand_set, min_len=3):
    if not query or not brand_set:
        return None
    text = _norm_unicode(query).lower()
    for token in _WORD.findall(text):
        if len(token) >= min_len and token in brand_set:
            return token
    return None


def load_brand_dictionary(path):
    if not path:
        return set()
    file_path = Path(path)
    if not file_path.exists():
        return set()
    brands = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        token = stripped.split("#", 1)[0].split()[0].strip().lower()
        if token:
            brands.add(token)
    return brands


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
        self.shape_cfg = _cfg_get(features_cfg, "shape")
        self.spec_cfg = _cfg_get(features_cfg, "spec")
        self.brand_cfg = _cfg_get(features_cfg, "brand")

        self.brand_set = load_brand_dictionary(
            _cfg_get(self.brand_cfg, "dictionary_path") if self.brand_cfg else None
        )
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
        return len(self.slot_order) * len(_STATES)

    def token_strings(self):
        tokens = []
        for slot in self.slot_order:
            prefix = _TOKEN_PREFIX[slot]
            for state in _STATES:
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
            elif slot == "shape":
                out.append(self._extract_shape(context))
            elif slot == "spec":
                out.append(self._extract_spec(context))
            elif slot == "brand":
                out.append(self._extract_brand(context))
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
            return _emit(prefix, _NONE)
        self._record("ean", "query_present")

        validated = _validate_ean(
            self._field(context, offer_field), require_checksum=require_checksum
        )
        if validated is None:
            self._record("ean", "offer_invalid")
            if _on_invalid_policy(cfg) == "mismatch":
                return _emit(prefix, _MISMATCH)
            return _emit(prefix, _NONE)
        self._record("ean", "offer_valid")

        if validated == query_candidate:
            self._record("ean", "match")
            return _emit(prefix, _MATCH)
        self._record("ean", "mismatch")
        return _emit(prefix, _MISMATCH)

    def _extract_article(self, context):
        return self._extract_id(
            context,
            slot="article",
            cfg=self.article_cfg,
            substring=False,
        )

    def _extract_shape(self, context):
        return self._extract_id(
            context,
            slot="shape",
            cfg=self.shape_cfg,
            substring=True,
        )

    def _extract_id(self, context, slot, cfg, substring):
        prefix = _TOKEN_PREFIX[slot]
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
            self._record(slot, "query_none")
            return _emit(prefix, _NONE)
        self._record(slot, "query_present")

        offer_values = self._collect_offer_ids(context, offer_fields)
        if not offer_values:
            self._record(slot, "offer_invalid")
            if _on_invalid_policy(cfg) == "mismatch":
                return _emit(prefix, _MISMATCH)
            return _emit(prefix, _NONE)
        self._record(slot, "offer_valid")

        if substring:
            for cand in query_candidates:
                for offer_id in offer_values:
                    if cand in offer_id or offer_id in cand:
                        self._record(slot, "match")
                        return _emit(prefix, _MATCH)
        else:
            offer_set = set(offer_values)
            for cand in query_candidates:
                if cand in offer_set:
                    self._record(slot, "match")
                    return _emit(prefix, _MATCH)
        self._record(slot, "mismatch")
        return _emit(prefix, _MISMATCH)

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
            return _emit(prefix, _NONE)
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
                return _emit(prefix, _MISMATCH)
            return _emit(prefix, _NONE)
        self._record("spec", "offer_valid")

        if all(token in haystack for token in query_specs):
            self._record("spec", "match")
            return _emit(prefix, _MATCH)
        self._record("spec", "mismatch")
        return _emit(prefix, _MISMATCH)

    def _extract_brand(self, context):
        prefix = _TOKEN_PREFIX["brand"]
        cfg = self.brand_cfg
        offer_field = _cfg_get(cfg, "offer_field", "manufacturer_name")
        min_query_len = int(_cfg_get(cfg, "min_query_token_len", 3) or 3)

        query_brand = _query_brand_candidate(
            self._query(context), self.brand_set, min_len=min_query_len
        )
        if query_brand is None:
            self._record("brand", "query_none")
            return _emit(prefix, _NONE)
        self._record("brand", "query_present")

        validated = _validate_brand_offer(self._field(context, offer_field))
        if validated is None:
            self._record("brand", "offer_invalid")
            if _on_invalid_policy(cfg) == "mismatch":
                return _emit(prefix, _MISMATCH)
            return _emit(prefix, _NONE)
        self._record("brand", "offer_valid")

        if query_brand == validated or query_brand in validated.split():
            self._record("brand", "match")
            return _emit(prefix, _MATCH)
        self._record("brand", "mismatch")
        return _emit(prefix, _MISMATCH)

    def stats_dict(self):
        out = {f"features/{k}": int(v) for k, v in self.stats.items()}
        out["features/rows_seen"] = int(self.rows_seen)
        out["features/brand_dict_size"] = int(len(self.brand_set))
        out["features/token_count"] = int(self.feature_token_count())
        return out
