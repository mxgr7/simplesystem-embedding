"""Build prod_soup inputs from an Elasticsearch article ``_source``.

The assembly rules mirror the historical article catalog and SPLADE extras
builders.  Text normalization and German folding remain the renderer's job.
"""

import hashlib
import json

from constants import FIELD_ORDER


S2CLASS_MAPPING_PATH = "/data/s2class-categories.json"
S2CLASS_MAPPING_SHA256 = (
    "900a5ac0c9a9cfcdd578a43770b5981b47eca29f0e874761b98bd8ddc2f4fd87"
)
S2_JUNK = {"27274091"}
IDENTITY_UNION_FIELDS = (
    "ean",
    "article_number",
    "manufacturer_article_number",
    "manufacturer_article_type",
)
AGGREGATION_MODES = ("v1", "v1-v2-all4-cap3", "v3")
UNION_SEPARATOR = " | "


def mapping_sha256(path=S2CLASS_MAPPING_PATH):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_s2_mapping(
    path=S2CLASS_MAPPING_PATH,
    expected_sha256=S2CLASS_MAPPING_SHA256,
):
    with open(path, "rb") as handle:
        raw = handle.read()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            f"S2 mapping SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    mapping = json.loads(raw)
    if not isinstance(mapping, dict):
        raise ValueError("S2 mapping must be a JSON object")
    return mapping


def _text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_text(item) for item in value if item is not None).strip()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _items(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def _append_distinct(values, value):
    if value and value not in values:
        values.append(value)


def _leaf_paths(category_paths):
    if not category_paths:
        return []
    if isinstance(category_paths, list):
        paths = []
        for entry in category_paths:
            paths.extend(_leaf_paths(entry))
        return paths
    if not isinstance(category_paths, dict):
        return []
    for level in range(5, 0, -1):
        paths = category_paths.get(f"upToLevel{level}") or []
        if paths:
            return [path for path in _items(paths) if path]
    return []


def _s2_leaf_labels(codes, mapping):
    distinct = []
    for code in _items(codes):
        code = _text(code)
        if code and code not in S2_JUNK and code not in distinct:
            distinct.append(code)

    def significant(code):
        while code.endswith("00") and len(code) > 2:
            code = code[:-2]
        return code

    leaves = [
        code
        for code in distinct
        if not any(
            other != code and other.startswith(significant(code))
            for other in distinct
        )
    ]
    return [mapping[code] for code in leaves if code in mapping]


def _render_features(features):
    rendered = []
    for feature in _items(features):
        if not isinstance(feature, dict):
            continue
        name = _text(feature.get("name") or "").strip()
        values = [
            _text(value)
            for value in _items(feature.get("values"))
            if _text(value)
        ]
        if name and values:
            rendered.append(f"{name}: {', '.join(values)}.")
    return " ".join(rendered)


def _normalized_ean(value):
    value = _text(value)
    return "" if value == "00000000" else value[:20]


def _identity_from_offer(offer):
    return {
        "name": _text(offer.get("name"))[:400],
        "manufacturer_name": _text(offer.get("manufacturerName"))[:120],
        "ean": _normalized_ean(offer.get("ean")),
        "article_number": _text(offer.get("articleNumber"))[:120],
        "manufacturer_article_number": _text(
            offer.get("manufacturerArticleNumber")
        )[:120],
        "manufacturer_article_type": _text(
            offer.get("manufacturerArticleType")
        ),
    }


def _most_complete_offer(offers):
    """Choose the V3 representative independently of source offer order."""
    def completeness(offer):
        signals = (
            _text(offer.get("name")),
            _text(offer.get("manufacturerName")),
            _normalized_ean(offer.get("ean")),
            _text(offer.get("articleNumber")),
            _text(offer.get("manufacturerArticleNumber")),
            _text(offer.get("manufacturerArticleType")),
            _render_features(offer.get("features")),
        )
        return sum(bool(value) for value in signals)

    def key(offer):
        return (
            -completeness(offer),
            _text(offer.get("name")),
            _text(offer.get("manufacturerName")),
            _text(offer.get("manufacturerArticleNumber")),
            _text(offer.get("articleNumber")),
            _normalized_ean(offer.get("ean")),
            _text(offer.get("manufacturerArticleType")),
            _render_features(offer.get("features")),
        )

    return min(offers, key=key)


def _historical_offer(offers):
    return next(
        (offer for offer in offers if _text(offer.get("manufacturerName"))),
        offers[0],
    )


def _bounded_identity_union(offers, representative, cap=3):
    sources = {
        "ean": ("ean", _normalized_ean),
        "article_number": (
            "articleNumber", lambda value: _text(value)[:120]
        ),
        "manufacturer_article_number": (
            "manufacturerArticleNumber", lambda value: _text(value)[:120]
        ),
        "manufacturer_article_type": (
            "manufacturerArticleType", _text
        ),
    }
    result = {}
    for field in IDENTITY_UNION_FIELDS:
        source_name, normalize = sources[field]
        values = []
        representative_value = normalize(representative.get(source_name))
        _append_distinct(values, representative_value)
        for offer in offers:
            _append_distinct(values, normalize(offer.get(source_name)))
        result[field] = UNION_SEPARATOR.join(values[:cap])
    return result


def assemble_fields(source, s2_mapping, aggregation="v1"):
    """Build an article aggregation variant from the full current source."""
    offers = source.get("offers") or []
    if not offers:
        return None

    if aggregation not in AGGREGATION_MODES:
        raise ValueError(f"unknown aggregation mode: {aggregation}")
    core = _most_complete_offer(offers) if aggregation == "v3" else _historical_offer(offers)
    core_name = _text(core.get("name"))
    feature_source = core if aggregation == "v3" else next(
        (offer for offer in offers if _text(offer.get("name")) == core_name),
        offers[0],
    )

    fields = {field: "" for field in FIELD_ORDER}
    fields.update(_identity_from_offer(core))
    fields["features_text"] = _render_features(feature_source.get("features"))
    if aggregation == "v1-v2-all4-cap3":
        fields.update(_bounded_identity_union(offers, core))

    keywords = []
    categories = []
    vendors = []
    s2_labels = []
    for offer in offers:
        for keyword in _items(offer.get("keywords")):
            _append_distinct(keywords, _text(keyword))
        for raw_path in _leaf_paths(offer.get("categoryPaths")):
            path = " > ".join(
                part.strip() for part in _text(raw_path).split("¦") if part.strip()
            )
            _append_distinct(categories, path)
        vendor = _text(offer.get("vendorName") or "").strip()
        _append_distinct(vendors, vendor)
        for label in _s2_leaf_labels(offer.get("s2classGroups"), s2_mapping):
            _append_distinct(s2_labels, label)

    customer_numbers = []
    for entry in _items(source.get("customerArticleNumbers")):
        value = entry.get("value") if isinstance(entry, dict) else entry
        value = _text(value or "").strip()
        _append_distinct(customer_numbers, value)

    fields.update(
        {
            "customer_artnos_text": " ".join(customer_numbers),
            "vendor_text": UNION_SEPARATOR.join(vendors),
            "category_leaf_text": UNION_SEPARATOR.join(categories),
            "s2class_text": UNION_SEPARATOR.join(s2_labels),
            "keywords_text": " ".join(keywords),
        }
    )
    return fields


def assemble_nul(source, s2_mapping, aggregation="v1"):
    """Return the exact 14-field prod_soup wire input for an article source."""
    fields = assemble_fields(source, s2_mapping, aggregation)
    if fields is None:
        return None
    return "\x00".join(fields[field] for field in FIELD_ORDER)
