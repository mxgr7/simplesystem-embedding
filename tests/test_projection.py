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

from indexer.projection import (  # noqa: E402
    CATALOG_CURRENCIES,
    HASH_VERSION,
    MAX_PRICE_SENTINEL,
    aggregate_article,
    compute_article_hash,
    group_by_hash,
    project,
    to_offer_row,
)

FIXTURE_PATH = REPO_ROOT / "tests/fixtures/mongo_sample/sample_200.json"


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text())["records"]


# ---------- shape against real records -----------------------------------

def test_pk_format_uuid_colon_b64url(records: list[dict]) -> None:
    """PK shape after dropping friendly_id: `{vendor_uuid_dashed}:{b64url(article_number)}`.
    DuckDB-native projection produces both halves without a UDF."""
    row = project(records[0]).row
    assert ":" in row["id"]
    head, b64 = row["id"].split(":", 1)
    # Head is the canonical hyphenated UUID and matches vendor_id verbatim.
    assert head == row["vendor_id"]
    uuid.UUID(head)  # parses
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


def _first_where(records: list[dict], pred) -> dict:
    """Find the first record satisfying `pred(record)`. Tests that need a
    specific shape from the prod fixture pick the record predicatively
    rather than pinning to a fixed index — the fixture is a `$sample`
    and re-dumps reshuffle which records land at which positions."""
    for rec in records:
        if pred(rec):
            return rec
    pytest.skip("no record in fixture matched the predicate")


def test_categories_emit_one_entry_per_depth(records: list[dict]) -> None:
    """A record with one categoryPath of depth ≥5 → exactly one entry at
    each `category_l{1..5}`. Same path produces one element per depth."""
    rec = _first_where(records, lambda r: any(
        len((p.get("elements") or [])) >= 5
        for p in (r["offer"].get("offer", {}).get("offerParams") or {}).get("categoryPaths") or []
    ))
    row = project(rec).row
    for d in range(1, 6):
        assert len(row[f"category_l{d}"]) >= 1


def test_features_use_name_equals_value(records: list[dict]) -> None:
    """Every projected feature follows the `name=value` format (per
    spec §7). Pin the format invariant, not specific values — the
    sample reshuffles between dumps."""
    rec = _first_where(records, lambda r: (
        r["offer"].get("offer", {}).get("offerParams") or {}
    ).get("features"))
    row = project(rec).row
    assert row["features"]
    assert all("=" in t for t in row["features"])


def test_prices_include_open_and_closed_when_present(records: list[dict]) -> None:
    """Every offer has `pricings.{open, closed}` embedded (`@NonNull` per
    legacy `Pricings.java`) → two priority-1/2 entries always present."""
    row = project(records[0]).row
    priorities = sorted(p["priority"] for p in row["prices"])
    assert priorities[0] == 1  # OPEN
    assert 2 in priorities  # CLOSED


def test_prices_include_dedicated_from_joined_collection(records: list[dict]) -> None:
    """Joined `pricings[]` rows surface as DEDICATED (priority 4) entries
    on the projected row. Pick any record with non-empty joined pricings."""
    rec = _first_where(records, lambda r: r.get("pricings"))
    row = project(rec).row
    assert any(p["priority"] == 4 for p in row["prices"])


def test_markers_split_into_enabled_and_disabled(records: list[dict]) -> None:
    """A record with at least one `coreArticleMarker: true` → that source
    UUID lands in `core_marker_enabled_sources`. Markers are very sparse
    in the fixture (~0.5% of offers) so this is opportunistic — the
    synthetic `test_disabled_marker_lands_in_disabled_array` covers the
    contract more precisely."""
    rec = _first_where(records, lambda r: any(
        m.get("coreArticleMarker") for m in (r.get("markers") or [])
    ))
    row = project(rec).row
    assert row["core_marker_enabled_sources"]


def test_eclass_codes_copied_as_arrays(records: list[dict]) -> None:
    """Wherever an offer has eclass groups, the projection copies the
    legacy arrays verbatim (every level of the hierarchy preserved so
    parent-level filters match via array_contains)."""
    rec = _first_where(records, lambda r: (
        ((r["offer"].get("offer", {}).get("offerParams") or {}).get("eclassGroups") or {})
        .get("ECLASS_5_1")
    ))
    row = project(rec).row
    assert row["eclass5_code"]
    assert all(isinstance(c, int) for c in row["eclass5_code"])


def test_delivery_time_projected(records: list[dict]) -> None:
    """`offerParams.deliveryTime` projects to a non-negative INT32 (schema
    column type). Specific values vary across the sample."""
    row = project(records[0]).row
    assert isinstance(row["delivery_time_days_max"], int)
    assert row["delivery_time_days_max"] >= 0


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


# ---------- F9: hash, split, aggregate ----------------------------------
#
# These tests cover the two-stream emission path that feeds the F9
# `articles_v{N}` + `offers_v{N}` topology. Hash determinism is the
# foundation: same canonicalised embedded-field tuple → same hash, every
# run, forever (until HASH_VERSION bumps).


def test_hash_format_is_32_lowercase_hex(records: list[dict]) -> None:
    h = compute_article_hash(project(records[0]).row)
    assert len(h) == 32
    assert all(c in "0123456789abcdef" for c in h), f"hash {h!r} contains non-hex chars"


def test_hash_deterministic_across_runs(records: list[dict]) -> None:
    """Re-projecting the same record yields the same hash. Foundation for
    idempotent upserts (F9 §"Hash function and embedded-field set")."""
    rec = records[0]
    h1 = compute_article_hash(project(rec).row)
    h2 = compute_article_hash(project(rec).row)
    assert h1 == h2


def test_hash_changes_when_name_changes() -> None:
    rec = _minimal_record()
    base = project(rec).row
    h_base = compute_article_hash(base)
    h_changed = compute_article_hash({**base, "name": base["name"] + "X"})
    assert h_base != h_changed


def test_hash_invariant_to_non_embedded_fields() -> None:
    """Vendor, catalog, prices, EANs, article_numbers, delivery, features,
    relationships, core markers — none of these affect the hash. Two
    offers of the same article share an embedding."""
    rec = _minimal_record()
    base = project(rec).row
    h_base = compute_article_hash(base)
    perturbed = {
        **base,
        "vendor_id": "different-vendor-uuid",
        "catalog_version_ids": ["different-catalog"],
        "prices": [{"price": 999.0, "currency": "EUR", "priority": 1, "sourcePriceListId": ""}],
        "ean": "9999999999999",
        "article_number": "DIFFERENT-NUM",
        "delivery_time_days_max": 99,
        "features": ["X=Y"],
        "core_marker_enabled_sources": ["src-1"],
        "relationship_accessory_for": ["A-1"],
    }
    assert compute_article_hash(perturbed) == h_base


def test_hash_invariant_to_array_order() -> None:
    """Categories at a depth and eclass codes are sets in spec terms;
    array order at projection-time is implementation-defined. The hash
    must canonicalise."""
    base = {
        "name": "x", "manufacturerName": "m",
        "category_l1": ["a", "b", "c"],
        "category_l2": [], "category_l3": [], "category_l4": [], "category_l5": [],
        "eclass5_code": [10, 20, 30],
        "eclass7_code": [],
        "s2class_code": [],
    }
    reordered = {**base, "category_l1": ["c", "a", "b"], "eclass5_code": [30, 10, 20]}
    assert compute_article_hash(base) == compute_article_hash(reordered)


def test_hash_distinguishes_categories_at_different_depths() -> None:
    """Same string at l1 vs at l2 must hash differently — the depth carries
    semantic meaning that the canonical form preserves."""
    at_l1 = {**_blank_hash_input(), "category_l1": ["Werkzeug"]}
    at_l2 = {**_blank_hash_input(), "category_l2": ["Werkzeug"]}
    assert compute_article_hash(at_l1) != compute_article_hash(at_l2)


def test_hash_version_constant_present() -> None:
    """Operational tooling greps for HASH_VERSION to detect a field-set
    change → schedule a full reindex via the alias-swing playbook (I3)."""
    assert HASH_VERSION == "v1"


def test_catalog_currencies_match_script() -> None:
    """All four `CATALOG_CURRENCIES` declarations must agree:

      * `indexer/projection.py` — envelope projection
      * `scripts/create_articles_collection.py` — articles_v{N} schema
      * `scripts/create_offers_collection.py` — offers_v{N} schema (F8)
      * `search-api/prices.py` — filter translator catalogue check (F8)

    Kept in sync by inspection because the script files aren't on the
    test import path; this test documents the contract."""
    import re

    paths = [
        REPO_ROOT / "scripts/create_articles_collection.py",
        REPO_ROOT / "scripts/create_offers_collection.py",
        REPO_ROOT / "search-api/prices.py",
    ]
    pattern = re.compile(r"^CATALOG_CURRENCIES(?:\s*:\s*[^=]+)?\s*=\s*(\(.+\))\s*$")
    for p in paths:
        line = next(
            (ln for ln in p.read_text().splitlines() if pattern.match(ln.strip())),
            None,
        )
        assert line is not None, f"no CATALOG_CURRENCIES literal found in {p.name}"
        rhs = pattern.match(line.strip()).group(1)
        ccys = eval(rhs)  # noqa: S307 - controlled input from repo files
        assert ccys == CATALOG_CURRENCIES, f"{p.name} CATALOG_CURRENCIES mismatch"


# ---- split: to_offer_row ------------------------------------------------

def test_to_offer_row_drops_article_level_fields(records: list[dict]) -> None:
    row = project(records[0]).row
    h = compute_article_hash(row)
    offer = to_offer_row(row, article_hash=h)
    for k in ("name", "manufacturerName", "category_l1", "category_l2",
              "category_l3", "category_l4", "category_l5",
              "eclass5_code", "eclass7_code", "s2class_code"):
        assert k not in offer, f"article-level field {k!r} leaked into offer row"


def test_to_offer_row_keeps_per_offer_fields(records: list[dict]) -> None:
    row = project(records[0]).row
    h = compute_article_hash(row)
    offer = to_offer_row(row, article_hash=h)
    for k in ("id", "vendor_id", "catalog_version_ids", "ean", "article_number",
              "prices", "delivery_time_days_max", "features",
              "core_marker_enabled_sources", "core_marker_disabled_sources",
              "relationship_accessory_for", "relationship_spare_part_for",
              "relationship_similar_to"):
        assert k in offer, f"per-offer field {k!r} missing from offer row"
    assert offer["article_hash"] == h
    assert offer["_placeholder_vector"] == [0.0, 0.0]


# ---- F8: per-offer envelope on to_offer_row -----------------------------

def _b64_uuid_of(hex_byte: str) -> dict:
    """Build a UUID-shaped EJSON binary from a single repeated hex byte
    (e.g. `_b64_uuid_of("11")` → all-`0x11` UUID)."""
    raw = bytes.fromhex(hex_byte * 16)
    return {"$binary": {"base64": base64.b64encode(raw).decode("ascii"), "subType": "04"}}


def _pricing_with_id(currency: str, price: str, *, source_id: dict) -> dict:
    """Variant of `_pricing` that lets the test pin a specific
    sourcePriceListId — needed to verify `price_list_ids` deduplication
    and union semantics."""
    return {
        "open": {
            "sourcePriceListId": source_id,
            "type": "OPEN",
            "prices": {"staggeredPrices": [{"minQuantity": "1", "price": price}], "currencyCode": currency},
            "priceQuantity": "1",
        },
    }


def test_offer_envelope_price_list_ids_dedup_and_sorted() -> None:
    """`price_list_ids` is the sorted dedup'd union of all
    `prices[].sourcePriceListId` on this offer."""
    rec = _minimal_record()
    pl_a = _b64_uuid_of("11")
    pl_b = _b64_uuid_of("22")
    rec["offer"]["offer"]["pricings"] = _pricing_with_id("EUR", "10.00", source_id=pl_a)
    rec["pricings"] = [
        # Repeat pl_a once (dedup target) + add pl_b.
        {"pricingDetails": {
            "sourcePriceListId": pl_a, "type": "GROUP",
            "prices": {"staggeredPrices": [{"minQuantity": "1", "price": "12.00"}], "currencyCode": "EUR"},
            "priceQuantity": "1",
        }},
        {"pricingDetails": {
            "sourcePriceListId": pl_b, "type": "DEDICATED",
            "prices": {"staggeredPrices": [{"minQuantity": "1", "price": "9.00"}], "currencyCode": "EUR"},
            "priceQuantity": "1",
        }},
    ]
    row = project(rec).row
    offer = to_offer_row(row, article_hash="dummyhash")
    assert offer["price_list_ids"] == sorted(
        {str(uuid.UUID(bytes=base64.b64decode(pl_a["$binary"]["base64"]))),
         str(uuid.UUID(bytes=base64.b64decode(pl_b["$binary"]["base64"])))}
    )


def test_offer_envelope_price_list_ids_empty_when_no_prices() -> None:
    rec = _minimal_record()  # no pricings
    row = project(rec).row
    offer = to_offer_row(row, article_hash="h")
    assert offer["price_list_ids"] == []


def test_offer_envelope_currencies_sorted_dedup() -> None:
    rec = _minimal_record()
    pl = _b64_uuid_of("11")
    rec["offer"]["offer"]["pricings"] = _pricing_with_id("EUR", "10.00", source_id=pl)
    rec["pricings"] = [{
        "pricingDetails": {
            "sourcePriceListId": pl, "type": "DEDICATED",
            "prices": {"staggeredPrices": [{"minQuantity": "1", "price": "9.50"}], "currencyCode": "CHF"},
            "priceQuantity": "1",
        },
    }]
    row = project(rec).row
    offer = to_offer_row(row, article_hash="h")
    # Lowercased + sorted.
    assert offer["currencies"] == ["chf", "eur"]


def test_offer_envelope_per_currency_min_max_across_offer_prices() -> None:
    """For each catalogue currency present on the offer, `_min/_max` span
    every price entry in that currency (regardless of priceList /
    priority)."""
    rec = _minimal_record()
    pl = _b64_uuid_of("11")
    rec["offer"]["offer"]["pricings"] = _pricing_with_id("EUR", "5.00", source_id=pl)
    rec["pricings"] = [
        # Two more EUR entries — envelope spans 5..50.
        {"pricingDetails": {
            "sourcePriceListId": pl, "type": "GROUP",
            "prices": {"staggeredPrices": [{"minQuantity": "1", "price": "20.00"}], "currencyCode": "EUR"},
            "priceQuantity": "1",
        }},
        {"pricingDetails": {
            "sourcePriceListId": pl, "type": "DEDICATED",
            "prices": {"staggeredPrices": [{"minQuantity": "1", "price": "50.00"}], "currencyCode": "EUR"},
            "priceQuantity": "1",
        }},
    ]
    row = project(rec).row
    offer = to_offer_row(row, article_hash="h")
    assert offer["eur_price_min"] == 5.0
    assert offer["eur_price_max"] == 50.0


def test_offer_envelope_missing_currency_uses_max_sentinel() -> None:
    """Catalogue currencies the offer has no price in get the sentinel
    pair. Range predicates `_min <= X` / `_max >= Y` are unsatisfiable
    against the sentinels — the row is naturally excluded."""
    rec = _minimal_record()
    rec["offer"]["offer"]["pricings"] = _pricing("EUR", "10.00")
    row = project(rec).row
    offer = to_offer_row(row, article_hash="h")
    assert offer["eur_price_min"] == 10.0
    assert offer["eur_price_max"] == 10.0
    for ccy in ("chf", "huf", "pln", "gbp", "czk", "cny"):
        assert offer[f"{ccy}_price_min"] == MAX_PRICE_SENTINEL
        assert offer[f"{ccy}_price_max"] == -MAX_PRICE_SENTINEL


def test_offer_envelope_unknown_currency_kept_in_currencies_array_dropped_from_min_max() -> None:
    """An out-of-catalogue currency (e.g. USD) shows up in the
    `currencies` array (the array column accepts any observed currency),
    but has no `usd_price_*` column to land in (those are catalogue-only).
    The post-pass on the JSON `prices` column still resolves USD."""
    rec = _minimal_record()
    rec["offer"]["offer"]["pricings"] = _pricing("USD", "5.00")
    row = project(rec).row
    offer = to_offer_row(row, article_hash="h")
    assert offer["currencies"] == ["usd"]
    assert "usd_price_min" not in offer
    # No EUR price → EUR carries sentinel (filter pre-pass would drop).
    assert offer["eur_price_min"] == MAX_PRICE_SENTINEL


def test_offer_envelope_no_prices_row_still_emits_envelope_columns() -> None:
    """A pricings-empty record still gets every catalogue
    `{ccy}_price_*` column (sentinel-filled) so Path B's probe doesn't
    trip on schema-missing fields."""
    rec = _minimal_record()
    row = project(rec).row
    offer = to_offer_row(row, article_hash="h")
    for ccy in CATALOG_CURRENCIES:
        assert f"{ccy}_price_min" in offer
        assert f"{ccy}_price_max" in offer
        assert offer[f"{ccy}_price_min"] == MAX_PRICE_SENTINEL
        assert offer[f"{ccy}_price_max"] == -MAX_PRICE_SENTINEL


# ---- aggregate ----------------------------------------------------------

def test_aggregate_article_single_offer_carries_embedded_fields() -> None:
    rec = _minimal_record(offerParams={
        "name": "Bohrer",
        "manufacturerName": "Bosch",
        "categoryPaths": [{"elements": ["Werkzeug", "Hand"]}],
        "eclassGroups": {"ECLASS_5_1": [23110101]},
    })
    flat = project(rec).row
    article = aggregate_article([flat])
    assert article["name"] == "Bohrer"
    assert article["manufacturerName"] == "Bosch"
    assert article["category_l1"] == ["Werkzeug"]
    assert article["category_l2"] == ["Werkzeug¦Hand"]
    assert article["eclass5_code"] == [23110101]
    assert article["article_hash"] == compute_article_hash(flat)


def test_aggregate_article_text_codes_unions_eans_and_article_numbers() -> None:
    """Two offers under one hash → text_codes contains both EANs and
    both article_numbers (sorted, deduped)."""
    rec_a = _minimal_record(articleNumber="ART-A")
    rec_a["offer"]["offer"]["offerParams"]["ean"] = "1111111111111"
    rec_a["offer"]["offer"]["offerParams"]["name"] = "x"
    rec_a["offer"]["offer"]["offerParams"]["manufacturerName"] = "m"
    rec_b = _minimal_record(articleNumber="ART-B")
    rec_b["offer"]["offer"]["offerParams"]["ean"] = "2222222222222"
    rec_b["offer"]["offer"]["offerParams"]["name"] = "x"
    rec_b["offer"]["offer"]["offerParams"]["manufacturerName"] = "m"
    rows = [project(rec_a).row, project(rec_b).row]
    # Same embedded fields → same hash; sanity-check.
    assert compute_article_hash(rows[0]) == compute_article_hash(rows[1])
    article = aggregate_article(rows)
    tokens = article["text_codes"].split()
    assert "x" in tokens
    assert "m" in tokens
    assert "1111111111111" in tokens
    assert "2222222222222" in tokens
    assert "ART-A" in tokens
    assert "ART-B" in tokens


def test_aggregate_article_envelope_min_max_across_offers() -> None:
    rec_low = _minimal_record(articleNumber="A")
    rec_low["offer"]["offer"]["pricings"] = _pricing("EUR", "10.00")
    rec_high = _minimal_record(articleNumber="B")
    rec_high["offer"]["offer"]["pricings"] = _pricing("EUR", "200.00")
    rows = [project(rec_low).row, project(rec_high).row]
    article = aggregate_article(rows)
    assert article["eur_price_min"] == 10.0
    assert article["eur_price_max"] == 200.0


def test_aggregate_article_envelope_per_currency() -> None:
    """Multi-currency offer → each currency lands in its own min/max."""
    rec = _minimal_record()
    rec["offer"]["offer"]["pricings"] = _pricing("EUR", "12.50")
    rec["pricings"] = [{
        "pricingDetails": {
            "sourcePriceListId": {"$binary": {"base64": "AAAAAAAAAAAAAAAAAAAAAA==", "subType": "04"}},
            "type": "DEDICATED",
            "prices": {"staggeredPrices": [{"minQuantity": "1", "price": "11.00"}], "currencyCode": "CHF"},
            "priceQuantity": "1",
        },
    }]
    article = aggregate_article([project(rec).row])
    assert article["eur_price_min"] == 12.5
    assert article["eur_price_max"] == 12.5
    assert article["chf_price_min"] == 11.0


def test_aggregate_article_missing_currency_uses_max_sentinel() -> None:
    """Currencies in CATALOG_CURRENCIES that no offer prices in get the
    `MAX_PRICE_SENTINEL` pair (`+S` on _min, `-S` on _max). Range
    predicates naturally exclude these rows; sort-by-price browse puts
    them last. F9 doc said NaN — Milvus 2.6 rejects NaN *and* ±Inf on
    FLOAT, so a large finite sentinel is the working substitute."""
    rec = _minimal_record()
    rec["offer"]["offer"]["pricings"] = _pricing("EUR", "1.00")
    article = aggregate_article([project(rec).row])
    assert article["eur_price_min"] == 1.0
    for ccy in ("chf", "huf", "pln", "gbp", "czk", "cny"):
        assert article[f"{ccy}_price_min"] == MAX_PRICE_SENTINEL, \
            f"{ccy}_price_min should be +sentinel, got {article[f'{ccy}_price_min']}"
        assert article[f"{ccy}_price_max"] == -MAX_PRICE_SENTINEL


def test_aggregate_article_unknown_currency_is_dropped() -> None:
    """A currency outside CATALOG_CURRENCIES (e.g. USD in the test suite)
    has no column to land in — skip it silently. This matches the column
    set being operator-controlled (add a currency: bump version)."""
    rec = _minimal_record()
    rec["offer"]["offer"]["pricings"] = _pricing("USD", "5.00")
    article = aggregate_article([project(rec).row])
    # No EUR price either, so EUR carries the +sentinel; USD has nowhere
    # to land.
    assert article["eur_price_min"] == MAX_PRICE_SENTINEL
    assert "usd_price_min" not in article


def test_max_price_sentinel_excludes_under_natural_range_predicate() -> None:
    """Documents the contract the sentinel is engineered for: callers
    use plain range predicates (`_min <= X AND _max >= Y`) without
    special-casing the sentinel. Sentinel rows fall out naturally for
    any plausible price X, Y."""
    for plausible_price in (0.01, 1.0, 1500.0, 1_000_000.0):
        assert not (MAX_PRICE_SENTINEL <= plausible_price)
        assert not (-MAX_PRICE_SENTINEL >= plausible_price)


def test_aggregate_article_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_article([])


# ---- F9 PR2b: customer_article_numbers ---------------------------------
#
# Inverted-by-value shape mirroring legacy ES Nested. Two source legs:
# joined `customerArticleNumbers` rows + the offer's catalog-supplied
# `offerParams.customerArticleNumber` paired with `outer.catalogVersionId`.

def _customer_number_row(value: str, *, version_uuid_hex: str) -> dict:
    """A joined-collection customerArticleNumbers row in the EJSON shape
    `dump_mongo_sample.js` produces."""
    raw = bytes.fromhex(version_uuid_hex * 16)
    return {
        "_id": {"$oid": "0" * 24},
        "vendorId": _b64_uuid_of("00"),
        "articleNumber": "ART-1",
        "customerArticleNumber": value,
        "customerArticleNumbersListVersionId": {
            "$binary": {"base64": base64.b64encode(raw).decode("ascii"), "subType": "04"},
        },
    }


def test_customer_numbers_joined_only() -> None:
    rec = _minimal_record()
    rec["customerArticleNumbers"] = [
        _customer_number_row("BOLT-001", version_uuid_hex="aa"),
        _customer_number_row("BOLT-002", version_uuid_hex="bb"),
    ]
    row = project(rec).row
    cans = row["customer_article_numbers"]
    assert cans == [
        {"value": "BOLT-001", "version_ids": [str(uuid.UUID(bytes=bytes.fromhex("aa" * 16)))]},
        {"value": "BOLT-002", "version_ids": [str(uuid.UUID(bytes=bytes.fromhex("bb" * 16)))]},
    ]


def test_customer_numbers_catalog_supplied_uses_catalog_version_id() -> None:
    """Per legacy `OfferSourceId.asCustomerArticleNumberSourceId()` — the
    catalog-supplied `offerParams.customerArticleNumber` is paired with the
    offer's own `catalogVersionId` UUID as version_id."""
    rec = _minimal_record(offerParams={"customerArticleNumber": "VENDOR-SKU-X"})
    rec["offer"]["catalogVersionId"] = _b64_uuid_of("cc")
    row = project(rec).row
    expected_version = str(uuid.UUID(bytes=bytes.fromhex("cc" * 16)))
    assert row["customer_article_numbers"] == [
        {"value": "VENDOR-SKU-X", "version_ids": [expected_version]},
    ]


def test_customer_numbers_both_sources_disjoint_values_kept_separate() -> None:
    rec = _minimal_record(offerParams={"customerArticleNumber": "FROM-CATALOG"})
    rec["offer"]["catalogVersionId"] = _b64_uuid_of("cc")
    rec["customerArticleNumbers"] = [
        _customer_number_row("FROM-PRICELIST", version_uuid_hex="aa"),
    ]
    row = project(rec).row
    values = sorted(e["value"] for e in row["customer_article_numbers"])
    assert values == ["FROM-CATALOG", "FROM-PRICELIST"]


def test_customer_numbers_same_value_from_both_sources_unions_version_ids() -> None:
    """Legacy mapper folds both into the same Nested entry when a catalog
    SKU happens to match a price-list SKU. Our projection mirrors this:
    one entry, both version_ids present."""
    rec = _minimal_record(offerParams={"customerArticleNumber": "SHARED-SKU"})
    rec["offer"]["catalogVersionId"] = _b64_uuid_of("cc")
    rec["customerArticleNumbers"] = [
        _customer_number_row("SHARED-SKU", version_uuid_hex="aa"),
    ]
    row = project(rec).row
    cans = row["customer_article_numbers"]
    assert len(cans) == 1
    assert cans[0]["value"] == "SHARED-SKU"
    assert sorted(cans[0]["version_ids"]) == sorted([
        str(uuid.UUID(bytes=bytes.fromhex("aa" * 16))),
        str(uuid.UUID(bytes=bytes.fromhex("cc" * 16))),
    ])


def test_customer_numbers_empty_when_neither_source_populated() -> None:
    """No joined rows + no catalog-supplied SKU → empty list, not absent.
    The bulk indexer always emits the column (Milvus rejects rows missing
    declared fields)."""
    rec = _minimal_record()
    row = project(rec).row
    assert row["customer_article_numbers"] == []


def test_customer_numbers_skip_rows_with_missing_value_or_version() -> None:
    rec = _minimal_record()
    # Row missing value, row missing version, row with both: only the third lands.
    rec["customerArticleNumbers"] = [
        {"customerArticleNumber": "", "customerArticleNumbersListVersionId": _b64_uuid_of("aa")},
        {"customerArticleNumber": "ORPHAN", "customerArticleNumbersListVersionId": None},
        _customer_number_row("OK", version_uuid_hex="bb"),
    ]
    row = project(rec).row
    assert row["customer_article_numbers"] == [
        {"value": "OK", "version_ids": [str(uuid.UUID(bytes=bytes.fromhex("bb" * 16)))]},
    ]


def test_aggregate_article_unions_customer_numbers_across_dedup_group() -> None:
    """Two offers in the same hash group: each contributed a different
    version_id for the same SKU value → article row's entry merges both
    version_ids. A separate value present on only one offer is kept as
    its own entry."""
    rec_a = _minimal_record(articleNumber="A")
    rec_a["customerArticleNumbers"] = [_customer_number_row("BOLT-001", version_uuid_hex="aa")]
    rec_b = _minimal_record(articleNumber="B")
    rec_b["customerArticleNumbers"] = [
        _customer_number_row("BOLT-001", version_uuid_hex="bb"),
        _customer_number_row("BOLT-002", version_uuid_hex="cc"),
    ]
    rows = [project(rec_a).row, project(rec_b).row]
    # Same embedded fields → same hash; sanity check.
    assert compute_article_hash(rows[0]) == compute_article_hash(rows[1])
    article = aggregate_article(rows)
    by_value = {e["value"]: sorted(e["version_ids"]) for e in article["customer_article_numbers"]}
    assert by_value == {
        "BOLT-001": sorted([
            str(uuid.UUID(bytes=bytes.fromhex("aa" * 16))),
            str(uuid.UUID(bytes=bytes.fromhex("bb" * 16))),
        ]),
        "BOLT-002": [str(uuid.UUID(bytes=bytes.fromhex("cc" * 16)))],
    }


def test_aggregate_article_emits_empty_customer_numbers_when_none() -> None:
    rec = _minimal_record()
    article = aggregate_article([project(rec).row])
    assert article["customer_article_numbers"] == []


def test_to_offer_row_strips_customer_numbers() -> None:
    """The offer row has no column to receive customer_article_numbers
    (article-level only per F9). `to_offer_row` must strip it; otherwise
    the Milvus offer-collection upsert fails on an unknown field."""
    rec = _minimal_record()
    rec["customerArticleNumbers"] = [_customer_number_row("X", version_uuid_hex="aa")]
    row = project(rec).row
    offer = to_offer_row(row, article_hash="h")
    assert "customer_article_numbers" not in offer


def test_customer_numbers_present_on_real_fixture(records: list[dict]) -> None:
    """At least one record in the prod fixture has joined customer
    numbers (~22% coverage in current dump). Confirms the projection
    threads through end-to-end against real BinData/EJSON shapes — not
    just the synthetic helpers above."""
    rec = _first_where(records, lambda r: r.get("customerArticleNumbers"))
    row = project(rec).row
    cans = row["customer_article_numbers"]
    assert cans
    for entry in cans:
        assert entry["value"]
        assert entry["version_ids"]
        for v in entry["version_ids"]:
            uuid.UUID(v)  # parses as canonical string


# ---- group_by_hash on the real sample ----------------------------------

def test_group_by_hash_yields_dedup_on_real_sample(records: list[dict]) -> None:
    """The 200-doc sample is too small to hit production's 1.22× ratio
    meaningfully, but `group_by_hash` must at least produce ≤ |records|
    groups (and exactly |records| if every record's embedded tuple is
    distinct, which is the typical case in this sample)."""
    rows = [project(r).row for r in records]
    groups = group_by_hash(rows)
    assert 0 < len(groups) <= len(rows)


# ---- helpers ------------------------------------------------------------

def _blank_hash_input() -> dict:
    return {
        "name": "", "manufacturerName": "",
        "category_l1": [], "category_l2": [], "category_l3": [],
        "category_l4": [], "category_l5": [],
        "eclass5_code": [], "eclass7_code": [], "s2class_code": [],
    }


def _pricing(currency: str, price: str) -> dict:
    """Wrap a single staggered price in the legacy `pricings.open` shape."""
    return {
        "open": {
            "sourcePriceListId": {"$binary": {"base64": "AAAAAAAAAAAAAAAAAAAAAA==", "subType": "04"}},
            "type": "OPEN",
            "prices": {"staggeredPrices": [{"minQuantity": "1", "price": price}], "currencyCode": currency},
            "priceQuantity": "1",
        },
    }
