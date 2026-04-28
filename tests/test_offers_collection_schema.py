"""Acceptance smoke for F1 (Milvus collection schema).

Drives the four packet acceptance criteria against a live Milvus:

  1. The script `scripts/create_offers_collection.py` produces a
     collection with every §7 field, the expected indexes, and a
     registered alias.
  2. A representative legacy `articleId` (≥ 80 chars) round-trips
     through the PK without truncation.
  3. The path-param `/{collection}/_search` contract keeps working
     against the alias name (validated at the MilvusClient layer that
     ftsearch wraps — `has_collection` + `search` both accept aliases).
  4. Every scalar filter expression that F3..F5 will rely on parses
     and executes against the empty/seeded collection without error.

Skipped if Milvus is not reachable. Re-uses the synthetic fixture at
`tests/fixtures/offers_schema_smoke.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

MILVUS_URI = "http://localhost:19530"
COLLECTION = "offers_v2"
ALIAS = "offers_v_alias"
DIM = 128

EXPECTED_FIELDS = {
    "id", "offer_embedding",
    "name", "manufacturerName", "ean", "article_number",
    "vendor_id", "catalog_version_ids",
    "category_l1", "category_l2", "category_l3", "category_l4", "category_l5",
    "prices", "delivery_time_days_max",
    "core_marker_enabled_sources", "core_marker_disabled_sources",
    "eclass5_code", "eclass7_code", "s2class_code",
    "features",
    "relationship_accessory_for", "relationship_spare_part_for", "relationship_similar_to",
    "closed_catalog",
}
EXPECTED_SCALAR_INDEXES = {
    "vendor_id", "catalog_version_ids", "closed_catalog",
    "eclass5_code", "eclass7_code", "s2class_code",
    "category_l1", "category_l2", "category_l3", "category_l4", "category_l5",
    "delivery_time_days_max", "features",
    "core_marker_enabled_sources", "core_marker_disabled_sources",
    "relationship_accessory_for", "relationship_spare_part_for", "relationship_similar_to",
    "ean", "article_number",
}

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "offers_schema_smoke.json"


@pytest.fixture(scope="module")
def client() -> MilvusClient:
    c = MilvusClient(uri=MILVUS_URI)
    if not c.has_collection(COLLECTION):
        pytest.skip(f"Collection {COLLECTION!r} missing — run scripts/create_offers_collection.py first")
    return c


@pytest.fixture(scope="module")
def fixture_rows() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text())["rows"]


def _vector(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float16)


def _to_milvus_row(row: dict) -> dict:
    out = {k: v for k, v in row.items() if not k.startswith("_") and k != "vector_seed"}
    out["offer_embedding"] = _vector(row["vector_seed"])
    return out


# --- (1) schema shape -----------------------------------------------------

def test_schema_has_all_section7_fields(client: MilvusClient) -> None:
    info = client.describe_collection(COLLECTION)
    field_names = {f["name"] for f in info["fields"]}
    missing = EXPECTED_FIELDS - field_names
    extra = field_names - EXPECTED_FIELDS
    assert not missing, f"missing §7 fields: {sorted(missing)}"
    assert not extra, f"unexpected fields beyond §7: {sorted(extra)}"


def test_pk_is_varchar_256(client: MilvusClient) -> None:
    info = client.describe_collection(COLLECTION)
    pk = next(f for f in info["fields"] if f["name"] == "id")
    assert pk["is_primary"] is True
    assert pk["params"]["max_length"] == 256


def test_all_expected_scalar_indexes_present(client: MilvusClient) -> None:
    indexes = set(client.list_indexes(COLLECTION))
    missing = EXPECTED_SCALAR_INDEXES - indexes
    assert not missing, f"missing scalar indexes: {sorted(missing)}"
    assert "offer_embedding" in indexes, "vector index missing"


def test_alias_resolves_to_collection(client: MilvusClient) -> None:
    info = client.describe_alias(alias=ALIAS)
    assert info["collection_name"] == COLLECTION
    # And the alias is callable transparently
    assert client.has_collection(ALIAS)


# --- (2) long-PK round-trip ------------------------------------------------

def test_long_pk_round_trips(client: MilvusClient, fixture_rows: list[dict]) -> None:
    long_row = next(r for r in fixture_rows if len(r["id"]) >= 80)
    assert len(long_row["id"]) >= 80, "fixture must include a ≥80-char PK"

    client.upsert(collection_name=COLLECTION, data=[_to_milvus_row(long_row)])
    fetched = client.get(collection_name=COLLECTION, ids=[long_row["id"]], output_fields=["id"])
    assert fetched, "long-PK row not found after insert"
    assert fetched[0]["id"] == long_row["id"], "PK truncated or mangled on round-trip"


# --- (3) /{collection}/_search contract via alias --------------------------

def test_search_via_alias_works(client: MilvusClient) -> None:
    """ftsearch invokes `client.search(collection_name=collection)`. The
    alias must resolve at the server with no client-side awareness."""
    res = client.search(
        collection_name=ALIAS,
        data=[_vector(seed=99).tolist()],
        anns_field="offer_embedding",
        limit=5,
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        output_fields=["id"],
    )
    assert isinstance(res, list) and len(res) == 1, "search via alias did not return one query group"


# --- (4) F3-bound filter expressions parse + execute -----------------------

@pytest.mark.parametrize("expr,description", [
    ('vendor_id == "44444444-4444-4444-4444-444444444444"',     "vendor equality"),
    ('vendor_id in ["aaa", "bbb"]',                             "vendor IN"),
    ('eclass5_code == 23172001',                                "eclass5 equality"),
    ('eclass5_code in [23172001, 27182301]',                    "eclass5 IN"),
    ('eclass7_code > 0',                                        "eclass7 range"),
    ('s2class_code in [1001, 5042]',                            "s2class IN"),
    ('delivery_time_days_max <= 7',                             "delivery range — maxDeliveryTime"),
    ('closed_catalog == true',                                  "boolean equality"),
    ('closed_catalog == false',                                 "boolean equality (negated)"),
    ('array_contains(catalog_version_ids, "aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa")',
                                                                "array membership"),
    ('array_contains_any(features, ["Werkstoff=Stahl", "Größe=M8x40"])',
                                                                "features OR-within / AND-across (one name)"),
    ('array_contains_any(core_marker_enabled_sources, ["11111111-aaaa-aaaa-aaaa-111111111111"])',
                                                                "coreSortimentOnly (enabled sources)"),
    ('not array_contains_any(core_marker_disabled_sources, ["33333333-cccc-cccc-cccc-333333333333"])',
                                                                "coreSortimentOnly (disabled exclusion)"),
    ('array_contains(relationship_accessory_for, "ACC-001")',   "accessoriesForArticleNumber"),
    ('array_contains(relationship_spare_part_for, "BASE-MODEL-A")',
                                                                "sparePartsForArticleNumber"),
    ('array_contains(relationship_similar_to, "SIMILAR-X")',    "similarToArticleNumber"),
    ('array_contains(category_l2, "Werkzeug¦Hand|Maschine")',   "category prefix at depth (currentCategoryPathElements)"),
    ('prices["currency"] == "EUR"',                             "JSON path (smoke; real query path resolves at request time)"),
    ('ean == "4006381000019"',                                  "ean equality"),
    ('article_number == "INDUSTRIAL-PART-9999/SUB-VARIANT-LONG-ZZZ-OPTION"',
                                                                "article_number equality"),
])
def test_f3_filter_expressions_parse_and_execute(client: MilvusClient, expr: str, description: str) -> None:
    """Each expr must parse and execute against the (potentially empty)
    collection. We don't assert hit counts — F3 will assert semantics."""
    try:
        client.query(collection_name=COLLECTION, filter=expr, output_fields=["id"], limit=1)
    except MilvusException as e:
        pytest.fail(f"filter expr failed to parse/execute ({description}): {expr!r} → {e}")
