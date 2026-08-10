"""Build prod_soup inputs from an Elasticsearch article ``_source``.

The four article-wide text fields — ``features_text``, ``keywords_text``,
``category_leaf_text``, ``s2class_text`` — are produced by the settled
per-field rules of ``field_preprocessing.md`` §14/§15/§16/§17, imported from
``rules/``, which is a byte-identical vendored copy of the research repo's rule
modules (see ``rules/VENDORED.md``).  Before MXG-48 they were re-implemented by
hand here, which is how they drifted: ``features`` came off a name-matched
record pick and ``keywords`` was a bare union carrying none of §15's drops,
while the training-side twin ``pipeline/build_article_extras.py`` had already
moved to the rules.  ``pipeline/tests/test_renderer_parity.py`` now asserts the
two produce the same six values for the same article.

The head identity fields are unchanged: they still come from one representative
record, chosen by ``aggregation``.

``textnorm.floor`` — encoding hygiene, §4.1/§4.3 — is applied here, to the
prose fields only.  German folding and template rendering remain the renderer's
job (``rendering.render_from_nul``); ``floor`` neither lowercases nor folds, so
it composes ahead of them.
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "rules"))

import category_rules as CR  # noqa: E402
import feat_kw_rules as FKR  # noqa: E402
from textnorm import floor  # noqa: E402

from constants import FIELD_ORDER  # noqa: E402


S2CLASS_MAPPING_PATH = "/data/s2class-categories.json"
S2CLASS_MAPPING_SHA256 = (
    "900a5ac0c9a9cfcdd578a43770b5981b47eca29f0e874761b98bd8ddc2f4fd87"
)
# `S2_JUNK` used to live here: the `encoded_in_splade_v1` pin, one code, kept
# narrow because widening it alone would desync serving from what the live
# index was encoded with. §17 rule 6 removed the render it protected, so there
# is no junk list left to apply and no dated successor pin to mint. The
# mapping loader below stays for the callers that still import it; this module
# no longer needs it.
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


def _render_features(features):
    """The pre-§14 rendering of ONE record's features.

    Retained only as a completeness signal for the ``v3`` representative choice
    below — it has to score records, and "carries features at all" is one of the
    seven signals it counts.  It is no longer what gets emitted.
    """
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


def _category_leaf_text(offers, vendors):
    """§16: deepest level, segment hygiene, union, prefix subsumption, cap.

    ``vendors`` is every record's vendor name, not the representative's: rule 4
    drops a path segment equal to *the article's* vendor name, and a path from
    one record can be rooted in a name only another record carries.
    """
    vendor_norms = {CR._norm_seg(v) for v in vendors}
    vendor_norms.discard("")
    paths = []
    for offer in offers:
        for raw_path in CR.deepest_paths(offer.get("categoryPaths")):
            segments = CR.clean_path(raw_path, vendor_norms)
            if segments:
                paths.append(segments)
    return CR.render_paths(CR.subsume_paths(paths), cap=CR.CAT_CAP)


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


def assemble_fields(source, s2_mapping=None, aggregation="v1"):
    """Build an article aggregation variant from the full current source.

    ``s2_mapping`` is accepted and ignored — §17 rule 6 stopped the
    classification being rendered, so nothing here resolves a code to a name.
    The parameter stays because callers pass it positionally.
    """
    offers = source.get("offers") or []
    if not offers:
        return None

    if aggregation not in AGGREGATION_MODES:
        raise ValueError(f"unknown aggregation mode: {aggregation}")
    core = _most_complete_offer(offers) if aggregation == "v3" else _historical_offer(offers)

    fields = {field: "" for field in FIELD_ORDER}
    fields.update(_identity_from_offer(core))
    if aggregation == "v1-v2-all4-cap3":
        fields.update(_bounded_identity_union(offers, core))

    vendors = []
    for offer in offers:
        _append_distinct(vendors, _text(offer.get("vendorName") or "").strip())

    customer_numbers = []
    for entry in _items(source.get("customerArticleNumbers")):
        value = entry.get("value") if isinstance(entry, dict) else entry
        _append_distinct(customer_numbers, _text(value or "").strip())

    # §14/§15. The terminator keeps the `Name: v1, v2.` shape this module has
    # always emitted, so the template does not move — only the content does.
    features_text, keywords_text = FKR.render_article(
        offers, custnos=customer_numbers, terminator="."
    )

    fields.update(
        {
            # NOT floored: identifiers belong to idnorm, and §19 measured the
            # floor's whitespace collapse breaking exact-term resolution
            # 520/566 -> 0/566.
            "customer_artnos_text": " ".join(customer_numbers),
            "vendor_text": floor(UNION_SEPARATOR.join(vendors)),
            "category_leaf_text": floor(_category_leaf_text(offers, vendors)),
            # §17 rule 6: the classification is a structured facet, not text.
            "s2class_text": "",
            "keywords_text": floor(keywords_text),
            "features_text": floor(features_text),
        }
    )
    return fields


def assemble_nul(source, s2_mapping=None, aggregation="v1"):
    """Return the exact 14-field prod_soup wire input for an article source."""
    fields = assemble_fields(source, s2_mapping, aggregation)
    if fields is None:
        return None
    return "\x00".join(fields[field] for field in FIELD_ORDER)
