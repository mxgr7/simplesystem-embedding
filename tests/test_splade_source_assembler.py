import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "splade-service"
sys.path.insert(0, str(SERVICE))

constants = importlib.import_module("constants")
assembler = importlib.import_module("source_assembler")

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
    assert fields["features_text"] == "Size: 10, 20."
    assert fields["description"] == ""
    assert fields["category_paths"] == ""


def test_core_fallback_ean_blank_and_truncated_name_feature_selection():
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
    assert fields["features_text"] == "Wrong: core."


def test_v3_representative_is_complete_and_order_invariant():
    poor = {"name": "z", "manufacturerName": "Maker"}
    rich = {
        "name": "a",
        "manufacturerName": "Maker",
        "ean": "123",
        "articleNumber": "A",
        "manufacturerArticleNumber": "M",
        "features": [{"name": "Size", "values": ["1"]}],
    }
    first = assembler.assemble_fields({"offers": [poor, rich]}, S2_MAPPING, "v3")
    second = assembler.assemble_fields({"offers": [rich, poor]}, S2_MAPPING, "v3")
    assert first == second
    assert first["name"] == "a"
    assert first["features_text"] == "Size: 1."


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
        "features": [{"name": "F", "values": ["2"]}],
    }
    right = {
        "name": "same", "manufacturerName": "M", "manufacturerArticleType": "A",
        "features": [{"name": "F", "values": ["1"]}],
    }
    first = assembler.assemble_fields({"offers": [left, right]}, S2_MAPPING, "v3")
    second = assembler.assemble_fields({"offers": [right, left]}, S2_MAPPING, "v3")
    assert first == second
    assert first["manufacturer_article_type"] == "A"
    assert first["features_text"] == "F: 1."


def test_scalar_keywords_and_feature_values_are_not_split_into_characters():
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
                    "upToLevel5": [" Root ¦ Leaf ", "Root¦Other"],
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
                        "upToLevel3": ["Root¦Leaf"],
                    },
                    {"upToLevel2": ["Second ¦ Path"]},
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
    assert fields["category_leaf_text"] == (
        "Root > Leaf | Root > Other | Second > Path"
    )
    assert fields["keywords_text"] == "first spaced last"
    assert fields["s2class_text"] == "Screwdrivers | Tools | Office"


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
