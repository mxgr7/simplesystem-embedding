import hashlib
import json
from pathlib import Path

import pytest


from conftest import load_flat_service

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "splade-service"
splade = load_flat_service(
    "splade_service", SERVICE, "constants", "source_assembler"
)
constants = splade.constants
assembler = splade.source_assembler

S2_MAPPING = {
    "27100000": "Tools",
    "27102000": "Hand tools",
    "27102010": "Screwdrivers",
    "31000000": "Office",
}


def test_no_offers_returns_none():
    assert assembler.assemble_fields({"offers": []}, S2_MAPPING) is None
    assert assembler.assemble_nul({}, S2_MAPPING) is None


def test_assembles_historical_identity_and_exact_field_order():
    source = {
        "offers": [
            {
                "name": "fallback",
                "manufacturerName": "",
                "features": [{"name": "Wrong", "values": ["offer"]}],
            },
            {
                "name": "Core",
                "manufacturerName": "M" * 121,
                "manufacturerArticleNumber": "P" * 121,
                "manufacturerArticleType": " Type ",
                "articleNumber": "A" * 121,
                "ean": "123456789012345678901",
                "features": [
                    {"name": " Size ", "values": [" 10 ", "", 20, None]},
                    {"name": "Empty", "values": []},
                ],
            },
        ]
    }

    fields = assembler.assemble_fields(source, S2_MAPPING)
    wire = assembler.assemble_nul(source, S2_MAPPING)

    assert tuple(fields) == constants.FIELD_ORDER
    assert wire.split("\x00") == [fields[name] for name in constants.FIELD_ORDER]
    assert len(wire.split("\x00")) == 14
    assert fields["name"] == "Core"
    assert fields["manufacturer_name"] == "M" * 120
    assert fields["manufacturer_article_number"] == "P" * 120
    assert fields["article_number"] == "A" * 120
    assert fields["ean"] == "12345678901234567890"
    assert fields["manufacturer_article_type"] == "Type"
    # §14, MXG-48: BOTH records' features, not the representative's. The old
    # rendering was "Size: 10, 20." — `Wrong: offer.` came off a record the
    # name-match pick discarded.
    assert fields["features_text"] == "Size: 10, 20. Wrong: offer."
    assert fields["description"] == ""
    assert fields["category_paths"] == ""


def test_core_fallback_and_ean_blank():
    """The identity representative is still a pick; only the article-wide text
    fields moved off it. Named `..._feature_selection` until MXG-48, when the
    feature pick it also covered stopped existing."""
    long_name = "N" * 401
    source = {
        "offers": [
            {
                "name": long_name,
                "ean": "00000000",
                "features": [{"name": "Wrong", "values": ["core"]}],
            },
            {
                "name": long_name[:400],
                "features": [{"name": "Selected", "values": ["yes"]}],
            },
        ]
    }

    fields = assembler.assemble_fields(source, S2_MAPPING)

    assert fields["name"] == long_name[:400]
    assert fields["ean"] == ""
    assert fields["features_text"] == "Selected: yes. Wrong: core."


def test_v3_representative_is_complete_and_order_invariant():
    poor = {"name": "z", "manufacturerName": "Maker"}
    rich = {
        "name": "a",
        "manufacturerName": "Maker",
        "ean": "123",
        "articleNumber": "A",
        "manufacturerArticleNumber": "M",
        "features": [{"name": "Size", "values": ["10"]}],
    }
    first = assembler.assemble_fields({"offers": [poor, rich]}, S2_MAPPING, "v3")
    second = assembler.assemble_fields({"offers": [rich, poor]}, S2_MAPPING, "v3")
    assert first == second
    assert first["name"] == "a"
    assert first["features_text"] == "Size: 10."


def test_v2_identifier_union_is_representative_first_distinct_and_cap3():
    source = {"offers": [
        {"name": "poor", "ean": "222", "articleNumber": "B"},
        {
            "name": "rich", "manufacturerName": "M", "ean": "111",
            "articleNumber": "A", "manufacturerArticleNumber": "P",
            "manufacturerArticleType": "T",
        },
        {"name": "third", "ean": "333", "articleNumber": "C"},
        {"name": "fourth", "ean": "444", "articleNumber": "D"},
    ]}
    fields = assembler.assemble_fields(source, S2_MAPPING, "v1-v2-all4-cap3")
    assert fields["ean"] == "111 | 222 | 333"
    assert fields["article_number"] == "A | B | C"
    assert fields["manufacturer_article_number"] == "P"
    assert fields["manufacturer_article_type"] == "T"
    plain = assembler.assemble_fields(source, S2_MAPPING, "v1")
    assert plain["ean"] == "111"


def test_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError, match="aggregation mode"):
        assembler.assemble_fields({"offers": [{"name": "x"}]}, S2_MAPPING, "bad")


def test_v3_tie_break_includes_type_and_features():
    left = {
        "name": "same", "manufacturerName": "M", "manufacturerArticleType": "Z",
        "features": [{"name": "F", "values": ["20"]}],
    }
    right = {
        "name": "same", "manufacturerName": "M", "manufacturerArticleType": "A",
        "features": [{"name": "F", "values": ["10"]}],
    }
    first = assembler.assemble_fields({"offers": [left, right]}, S2_MAPPING, "v3")
    second = assembler.assemble_fields({"offers": [right, left]}, S2_MAPPING, "v3")
    assert first == second
    # the tie-break still decides the IDENTITY fields...
    assert first["manufacturer_article_type"] == "A"
    # ...and no longer decides `features_text`, because §14 keeps both values
    # of a contradicting key rather than letting a pick arbitrate (G2).
    assert first["features_text"] == "F: 10, 20."


def test_scalar_keywords_and_feature_values_are_not_split_into_characters():
    """A bare scalar where the index normally carries a list. Guarded here
    since before MXG-48 and preserved through the move to the shared rules by
    `feat_kw_rules.listed`."""
    source = {"offers": [{
        "name": "x", "keywords": "drill",
        "features": {"name": "Size", "values": "10"},
    }]}
    fields = assembler.assemble_fields(source, S2_MAPPING, "v1")
    assert fields["keywords_text"] == "drill"
    assert fields["features_text"] == "Size: 10."


def test_article_unions_preserve_historical_order_and_shapes():
    source = {
        "offers": [
            {
                "name": "one",
                "keywords": ["first", " spaced ", "first", ""],
                "vendorName": " Vendor A ",
                "categoryPaths": {
                    "upToLevel1": ["Ignored"],
                    "upToLevel5": [" Werkzeug ¦ Zange ", "Werkzeug¦Schere"],
                },
                "s2classGroups": [
                    "27100000",
                    "27102000",
                    "27102010",
                    "27274091",
                ],
            },
            {
                "name": "two",
                "keywords": [" spaced ", "last"],
                "vendorName": "Vendor A",
                "categoryPaths": [
                    {
                        "upToLevel1": ["Ignored too"],
                        "upToLevel3": ["Werkzeug¦Zange"],
                    },
                    {"upToLevel2": ["Elektro ¦ Kabel"]},
                ],
                "s2classGroups": ["27100000"],
            },
            {
                "name": "three",
                "vendorName": " Vendor B ",
                "s2classGroups": ["31000000", "27102010"],
            },
        ],
        "customerArticleNumbers": [
            {"value": " C-1 "},
            "C-2",
            {"value": "C-1"},
            {"value": "   "},
        ],
    }

    fields = assembler.assemble_fields(source, S2_MAPPING)

    assert fields["customer_artnos_text"] == "C-1 C-2"
    assert fields["vendor_text"] == "Vendor A | Vendor B"
    # §16: union across records, then G5 order — deepest first, then
    # lexicographic. Insertion order was the rule until MXG-48.
    assert fields["category_leaf_text"] == (
        "Elektro > Kabel | Werkzeug > Schere | Werkzeug > Zange"
    )
    # §15's own deterministic order, not insertion order.
    assert fields["keywords_text"] == "spaced first last"
    # §17 rule 6 — the classification is a facet, never document text. The
    # junk code 27274091 in the first record used to drop only itself, which
    # promoted its junk parent to leaf; there is nothing left to promote.
    assert fields["s2class_text"] == ""


def test_category_path_hygiene_is_applied():
    """§16 rules 2-4 and 6, which the deployed renderer had none of."""
    source = {"offers": [
        {
            "name": "Zange", "vendorName": "RS Components",
            "categoryPaths": {"upToLevel3": [
                "RS Components¦06 - Elektromechanische Bauelemente¦Sonstige",
                "Werkzeug\r¦Zange",
            ]},
        },
        {"name": "Zange", "categoryPaths": {"upToLevel1": ["Werkzeug"]}},
    ]}
    fields = assembler.assemble_fields(source, S2_MAPPING)
    # vendor-name root dropped, numbering prefix stripped, contentless leaf
    # `Sonstige` promoted to its parent, control character removed, and the
    # bare `Werkzeug` subsumed by `Werkzeug > Zange`.
    assert fields["category_leaf_text"] == (
        "Werkzeug > Zange | Elektromechanische Bauelemente"
    )


def test_markup_and_entities_are_floored_out():
    """§4.1/§4.3. The training-side twin has floored since MXG-64; this side
    did not, so served documents carried `<br>` and `&uuml;` into the encoder.
    """
    source = {"offers": [{
        "name": "Rohr",
        "keywords": ["&uuml;berwurf"],
        "features": [{"name": "Material", "values": ["V2A<br>Stahl"]}],
    }]}
    fields = assembler.assemble_fields(source, S2_MAPPING)
    assert fields["keywords_text"] == "überwurf"
    assert "<br>" not in fields["features_text"]
    assert fields["features_text"] == "Material: V2A Stahl."


def test_mapping_sha_and_validation(tmp_path):
    path = tmp_path / "s2.json"
    raw = json.dumps(S2_MAPPING, sort_keys=True).encode()
    path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()

    assert assembler.S2CLASS_MAPPING_SHA256 == (
        "900a5ac0c9a9cfcdd578a43770b5981b47eca29f0e874761b98bd8ddc2f4fd87"
    )
    assert assembler.mapping_sha256(path) == expected
    assert assembler.load_s2_mapping(path, expected) == S2_MAPPING
    with pytest.raises(ValueError, match="S2 mapping SHA256 mismatch"):
        assembler.load_s2_mapping(path, "0" * 64)
