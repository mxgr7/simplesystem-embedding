"""Unit tests for `indexer/projection.py`.

Mixes (a) targeted assertions on a small number of known records from
`tests/fixtures/mongo_sample/sample_200.json` (parity against real
MongoDB shapes) with (b) synthetic records that exercise the documented
edge cases the 200-row sample does not happen to cover (long PK,
feature-value containing `=`, category element containing `¦`).
"""

from __future__ import annotations

import base64
import json
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.friendly_id import to_uuid
from indexer.projection import project  # noqa: E402

FIXTURE_PATH = REPO_ROOT / "tests/fixtures/mongo_sample/sample_200.json"


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text())["records"]


# ---------- shape against real records -----------------------------------

def test_pk_format_friendlyid_colon_b64url(records: list[dict]) -> None:
    row = project(records[0]).row
    assert ":" in row["id"]
    fid, b64 = row["id"].split(":", 1)
    # FriendlyId is 22 chars and decodes to the vendorId UUID.
    assert len(fid) == 22
    assert str(to_uuid(fid)) == row["vendor_id"]
    # b64url part round-trips to article_number.
    pad = "=" * (-len(b64) % 4)
    assert base64.urlsafe_b64decode(b64 + pad).decode("utf-8") == row["article_number"]


def test_vendor_id_is_canonical_uuid_string(records: list[dict]) -> None:
    row = project(records[0]).row
    parsed = uuid.UUID(row["vendor_id"])
    assert str(parsed) == row["vendor_id"]


def test_catalog_version_ids_single_uuid(records: list[dict]) -> None:
    row = project(records[0]).row
    assert len(row["catalog_version_ids"]) == 1
    uuid.UUID(row["catalog_version_ids"][0])  # parses as UUID


def test_categories_emit_one_entry_per_depth(records: list[dict]) -> None:
    """Record 0 has one category path of depth 5 → one entry per
    category_l{1..5}."""
    row = project(records[0]).row
    for d in range(1, 6):
        assert len(row[f"category_l{d}"]) == 1


def test_features_use_name_equals_value(records: list[dict]) -> None:
    row = project(records[0]).row
    assert row["features"]
    assert all("=" in t for t in row["features"])
    # OfferParams.features in record 0 has a "Werkstoff" feature with value "ST".
    assert "Werkstoff=ST" in row["features"]


def test_prices_include_open_and_closed_when_present(records: list[dict]) -> None:
    """Record 0 has both `pricings.open` and `pricings.closed` → two entries
    with priorities 1 (OPEN) and 2 (CLOSED)."""
    row = project(records[0]).row
    priorities = sorted(p["priority"] for p in row["prices"])
    assert priorities[0] == 1  # OPEN
    assert 2 in priorities  # CLOSED


def test_prices_include_dedicated_from_joined_collection(records: list[dict]) -> None:
    """Record 1 has joined `pricings[]` rows of type DEDICATED (priority 4)."""
    row = project(records[1]).row
    assert any(p["priority"] == 4 for p in row["prices"])


def test_markers_split_into_enabled_and_disabled(records: list[dict]) -> None:
    """Record 1 has two `coreArticleMarker: true` markers → enabled list
    populated, disabled empty."""
    row = project(records[1]).row
    assert len(row["core_marker_enabled_sources"]) == 2
    assert row["core_marker_disabled_sources"] == []


def test_eclass_codes_pull_first_of_set(records: list[dict]) -> None:
    """Record 0's offerParams.eclassGroups has ECLASS_5_1=[23110101] and
    ECLASS_7_1=[23110101]; no S2CLASS."""
    row = project(records[0]).row
    assert row["eclass5_code"] == 23110101
    assert row["eclass7_code"] == 23110101
    assert row["s2class_code"] == 0


def test_delivery_time_projected(records: list[dict]) -> None:
    row = project(records[0]).row
    assert row["delivery_time_days_max"] == 20  # offerParams.deliveryTime in record 0


# ---------- synthetic edge cases the real sample doesn't cover ----------

def _minimal_record(**overrides) -> dict:
    """Build the smallest joined record that `project()` accepts."""
    inner = {
        "_id": {"$binary": {"base64": "AAAAAAAAAAAAAAAAAAAAAA==", "subType": "04"}},
        "catalogVersionId": {"$binary": {"base64": "AAAAAAAAAAAAAAAAAAAAAA==", "subType": "04"}},
        "offerParams": {
            "name": "x", "manufacturerName": "m", "ean": "e",
            "deliveryTime": 0,
            "features": [],
            "categoryPaths": [],
            "eclassGroups": {},
        },
        "pricings": {},
        "relatedArticleNumbers": {"sparePartFor": [], "accessoryFor": [], "similarTo": []},
    }
    outer = {
        "vendorId": {"$binary": {"base64": "AAAAAAAAAAAAAAAAAAAAAA==", "subType": "04"}},
        "catalogVersionId": {"$binary": {"base64": "AAAAAAAAAAAAAAAAAAAAAA==", "subType": "04"}},
        "articleNumber": "ART-1",
        "offer": inner,
    }
    rec = {"offer": outer, "pricings": [], "markers": []}
    if "offerParams" in overrides:
        inner["offerParams"].update(overrides.pop("offerParams"))
    if "articleNumber" in overrides:
        outer["articleNumber"] = overrides.pop("articleNumber")
    if "vendorId" in overrides:
        outer["vendorId"] = overrides.pop("vendorId")
    rec.update(overrides)
    return rec


def test_long_pk_round_trips() -> None:
    """Acceptance: a representative legacy `articleId` (≥80 chars) lands in
    the PK without truncation and round-trips."""
    long_article = "INDUSTRIAL-PART-9999/SUB-VARIANT-LONG-ZZZ-OPTION-CHANNEL-EXT"
    rec = _minimal_record(articleNumber=long_article)
    row = project(rec).row
    assert len(row["id"]) >= 80
    fid, b64 = row["id"].split(":", 1)
    pad = "=" * (-len(b64) % 4)
    assert base64.urlsafe_b64decode(b64 + pad).decode("utf-8") == long_article


def test_feature_with_equals_in_value_is_dropped() -> None:
    rec = _minimal_record(offerParams={
        "features": [
            {"name": "OK", "values": ["plain"]},
            {"name": "BadValue", "values": ["a=b"]},
        ],
    })
    res = project(rec)
    assert "OK=plain" in res.row["features"]
    assert not any(t.startswith("BadValue=") for t in res.row["features"])
    assert ("BadValue", "a=b") in res.dropped_features


def test_category_element_containing_separator_is_escaped() -> None:
    """`Hand¦Maschine` element → `Hand|Maschine` per CategoryPath.asStringPath."""
    rec = _minimal_record(offerParams={
        "categoryPaths": [{"elements": ["Werkzeug", "Hand¦Maschine"]}],
    })
    row = project(rec).row
    assert row["category_l1"] == ["Werkzeug"]
    assert row["category_l2"] == ["Werkzeug¦Hand|Maschine"]


def test_categories_dedup_per_depth() -> None:
    """Two paths sharing a prefix → that prefix appears once at l1."""
    rec = _minimal_record(offerParams={
        "categoryPaths": [
            {"elements": ["Werkzeug", "Hand"]},
            {"elements": ["Werkzeug", "Akku"]},
        ],
    })
    row = project(rec).row
    assert row["category_l1"] == ["Werkzeug"]
    assert sorted(row["category_l2"]) == ["Werkzeug¦Akku", "Werkzeug¦Hand"]


def test_categories_truncate_at_depth_5() -> None:
    """A path deeper than 5 still produces depths 1..5; deeper levels are
    silently dropped (schema only has l1..l5)."""
    deep = ["a", "b", "c", "d", "e", "f", "g"]
    rec = _minimal_record(offerParams={"categoryPaths": [{"elements": deep}]})
    row = project(rec).row
    assert row["category_l1"] == ["a"]
    assert row["category_l5"] == ["a¦b¦c¦d¦e"]


def test_multiple_currencies_round_trip_through_prices_json() -> None:
    """Acceptance: a row with multiple prices across currencies + price
    lists round-trips through the `prices` JSON field unchanged."""
    rec = _minimal_record()
    rec["offer"]["offer"]["pricings"] = {
        "open": {
            "sourcePriceListId": {"$binary": {"base64": "AAAAAAAAAAAAAAAAAAAAAA==", "subType": "04"}},
            "type": "OPEN",
            "prices": {"staggeredPrices": [{"minQuantity": "1", "price": "10.00"}], "currencyCode": "EUR"},
            "priceQuantity": "1",
        },
    }
    rec["pricings"] = [
        {
            "pricingDetails": {
                "sourcePriceListId": {"$binary": {"base64": "AAAAAAAAAAAAAAAAAAAAAA==", "subType": "04"}},
                "type": "DEDICATED",
                "prices": {"staggeredPrices": [{"minQuantity": "1", "price": "12.00"}], "currencyCode": "USD"},
                "priceQuantity": "1",
            },
        },
    ]
    row = project(rec).row
    assert {p["currency"] for p in row["prices"]} == {"EUR", "USD"}
    assert {p["priority"] for p in row["prices"]} == {1, 4}


def test_price_quantity_divides_unit_price() -> None:
    """`CalculatingPrice.singleUnitPrice`: 80.08 / 100 = 0.8008 — record 0
    in the real sample uses priceQuantity=100."""
    rec = _minimal_record()
    rec["offer"]["offer"]["pricings"] = {
        "open": {
            "sourcePriceListId": {"$binary": {"base64": "AAAAAAAAAAAAAAAAAAAAAA==", "subType": "04"}},
            "type": "OPEN",
            "prices": {"staggeredPrices": [{"minQuantity": "1", "price": "80.08"}], "currencyCode": "EUR"},
            "priceQuantity": "100",
        },
    }
    row = project(rec).row
    assert row["prices"] == [{
        "price": 0.8008,
        "currency": "EUR",
        "priority": 1,
        "sourcePriceListId": "00000000-0000-0000-0000-000000000000",
    }]


def test_no_prices_yields_empty_array() -> None:
    rec = _minimal_record()
    row = project(rec).row
    assert row["prices"] == []


def test_disabled_marker_lands_in_disabled_array() -> None:
    rec = _minimal_record(markers=[
        {"coreArticleListSourceId": {"$binary": {"base64": "AAAAAAAAAAAAAAAAAAAAAQ==", "subType": "04"}},
         "coreArticleMarker": True},
        {"coreArticleListSourceId": {"$binary": {"base64": "AAAAAAAAAAAAAAAAAAAAAg==", "subType": "04"}},
         "coreArticleMarker": False},
    ])
    row = project(rec).row
    assert row["core_marker_enabled_sources"] == ["00000000-0000-0000-0000-000000000001"]
    assert row["core_marker_disabled_sources"] == ["00000000-0000-0000-0000-000000000002"]


def test_relationships_projected() -> None:
    rec = _minimal_record()
    rec["offer"]["offer"]["relatedArticleNumbers"] = {
        "accessoryFor": ["A-1", "A-2"],
        "sparePartFor": ["S-1"],
        "similarTo": [],
    }
    row = project(rec).row
    assert row["relationship_accessory_for"] == ["A-1", "A-2"]
    assert row["relationship_spare_part_for"] == ["S-1"]
    assert row["relationship_similar_to"] == []


# ---------- bulk projection sanity --------------------------------------

def test_all_200_records_project_without_error(records: list[dict]) -> None:
    """No record in the prod sample should hit an unhandled shape."""
    for i, rec in enumerate(records):
        try:
            project(rec)
        except Exception as exc:
            pytest.fail(f"record {i} failed to project: {exc}")
