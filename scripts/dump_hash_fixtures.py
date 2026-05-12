"""Emit parity fixtures: input tuples → hashes via indexer.projection.compute_article_hash.

The Java EmbeddingInputHasherTest reads this JSON and asserts byte-identical hashes
from the Java port. Hand-curated coverage: empty strings, all 8 fields populated,
multiple category paths (verifying sort + RS), non-ASCII (German/umlauts), null fields,
and the ¦ char inside a category path element (which the canonicalizer doesn't escape).
"""
from __future__ import annotations
import json
import sys
sys.path.insert(0, '/home/mgerer/simplesystem-embedding')
from indexer.projection import compute_article_hash


FIXTURES = [
    {
        "case": "all_empty",
        "row": {},
    },
    {
        "case": "all_populated_ascii",
        "row": {
            "name": "Vishay Dünnschicht SMD Festwiderstand",
            "manufacturerName": "Vishay",
            "description": "Precision 0.1% resistor, 1W, 0402.",
            "categoryPaths": [
                {"elements": ["Bauelemente", "Widerstände", "SMD"]},
            ],
            "ean": "5410886234567",
            "article_number": "240-0020",
            "manufacturerArticleNumber": "TNPV0402100R0BEEN",
            "manufacturerArticleType": "TNPV-Series",
        },
    },
    {
        "case": "multiple_categories_unsorted",
        "row": {
            "name": "Item",
            "manufacturerName": "X",
            "categoryPaths": [
                {"elements": ["B", "Sub"]},
                {"elements": ["A", "Sub"]},
                {"elements": ["A", "Other"]},
            ],
            "article_number": "ART-001",
        },
    },
    {
        "case": "umlauts_only",
        "row": {
            "name": "Tüll für Bürste — ÄÖÜß",
            "categoryPaths": [{"elements": ["Bürsten & Tücher"]}],
        },
    },
    {
        "case": "categorypaths_list_form_no_dict",
        "row": {
            "name": "Plain list of elements",
            "categoryPaths": [["A", "B", "C"]],
        },
    },
    {
        "case": "pipe_inside_element_no_escape",
        "row": {
            "name": "Edge case",
            "categoryPaths": [{"elements": ["A|with|pipe", "B"]}],
        },
    },
    {
        "case": "categorypaths_empty_list",
        "row": {
            "name": "no cats",
            "categoryPaths": [],
        },
    },
    {
        "case": "field_with_nul_should_never_happen_but_test",
        "row": {
            "name": "x\x00y",
            "manufacturerName": "z",
        },
    },
    {
        "case": "all_eight_fields_short",
        "row": {
            "name": "n",
            "manufacturerName": "m",
            "description": "d",
            "categoryPaths": [{"elements": ["c1", "c2"]}],
            "ean": "1234567890",
            "article_number": "an",
            "manufacturerArticleNumber": "man",
            "manufacturerArticleType": "mat",
        },
    },
    {
        "case": "single_category_path_one_element",
        "row": {
            "name": "x",
            "categoryPaths": [{"elements": ["only"]}],
        },
    },
]


def main() -> None:
    out = []
    for f in FIXTURES:
        row = f["row"]
        out.append({
            "case": f["case"],
            "name": row.get("name"),
            "manufacturerName": row.get("manufacturerName"),
            "description": row.get("description"),
            # Normalise category paths to a list of {"elements": [...]} for the Java side.
            "categoryPaths": [
                cp if isinstance(cp, dict) else {"elements": cp}
                for cp in (row.get("categoryPaths") or [])
            ],
            "ean": row.get("ean"),
            "articleNumber": row.get("article_number"),
            "manufacturerArticleNumber": row.get("manufacturerArticleNumber"),
            "manufacturerArticleType": row.get("manufacturerArticleType"),
            "expectedHash": compute_article_hash(row),
        })
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
