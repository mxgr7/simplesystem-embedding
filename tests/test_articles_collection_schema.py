"""Schema smoke for `articles_v{N}` (F9 PR1).

Drives the F9 PR1 acceptance criteria for the new article-side
collection against a live Milvus:

  1. The script `scripts/create_articles_collection.py` produces a
     collection whose fields match the F9 article-level scope (vector +
     BM25 + article-level scalars + per-currency envelope).
  2. PK is `article_hash` VARCHAR(32); the BM25 function is wired
     `text_codes` → `sparse_codes`.
  3. Per-currency envelope columns are present with STL_SORT (powers F4
     sort-by-price browse).
  4. Article-level F3 filter expressions (categories, eclass, s2class)
     parse and execute. ANN smoke against `offer_embedding` returns a
     well-formed result.

Skipped if Milvus is not reachable or `articles_v1` is not present.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

MILVUS_URI = "http://localhost:19530"
COLLECTION = "articles_v1"
DIM = 128

# Mirror of `CATALOG_CURRENCIES` in `scripts/create_articles_collection.py`.
# Kept inline (same convention as `test_offers_collection_schema.py`) so the
# test module is importable without scripts/ on sys.path.
CATALOG_CURRENCIES = ("eur", "chf", "huf", "pln", "gbp", "czk", "cny")

SCALAR_FIELDS = {
    "name", "manufacturerName",
    "category_l1", "category_l2", "category_l3", "category_l4", "category_l5",
    "eclass5_code", "eclass7_code", "s2class_code",
    "text_codes",
}
VECTOR_FIELDS = {"offer_embedding", "sparse_codes"}
ENVELOPE_FIELDS = {f"{c}_price_{s}" for c in CATALOG_CURRENCIES for s in ("min", "max")}
EXPECTED_FIELDS = {"article_hash"} | SCALAR_FIELDS | VECTOR_FIELDS | ENVELOPE_FIELDS

EXPECTED_INVERTED_INDEXES = {
    "category_l1", "category_l2", "category_l3", "category_l4", "category_l5",
    "eclass5_code", "eclass7_code", "s2class_code",
}


@pytest.fixture(scope="module")
def client() -> MilvusClient:
    c = MilvusClient(uri=MILVUS_URI)
    if not c.has_collection(COLLECTION):
        pytest.skip(f"Collection {COLLECTION!r} missing — run scripts/create_articles_collection.py first")
    return c


# --- (1) schema shape -----------------------------------------------------

def test_schema_matches_article_level_scope(client: MilvusClient) -> None:
    info = client.describe_collection(COLLECTION)
    field_names = {f["name"] for f in info["fields"]}
    missing = EXPECTED_FIELDS - field_names
    extra = field_names - EXPECTED_FIELDS
    assert not missing, f"missing article-level fields: {sorted(missing)}"
    assert not extra, f"unexpected fields beyond article-level scope: {sorted(extra)}"


def test_pk_is_article_hash_varchar_32(client: MilvusClient) -> None:
    info = client.describe_collection(COLLECTION)
    pk = next(f for f in info["fields"] if f["name"] == "article_hash")
    assert pk["is_primary"] is True
    assert pk["params"]["max_length"] == 32, "article_hash PK must be VARCHAR(32) per F9"


def test_bm25_function_wired(client: MilvusClient) -> None:
    info = client.describe_collection(COLLECTION)
    funcs = info.get("functions", [])
    assert any(
        f.get("input_field_names") == ["text_codes"]
        and f.get("output_field_names") == ["sparse_codes"]
        for f in funcs
    ), "BM25 function text_codes → sparse_codes not wired"


def test_envelope_columns_present_for_every_currency(client: MilvusClient) -> None:
    info = client.describe_collection(COLLECTION)
    field_names = {f["name"] for f in info["fields"]}
    for ccy in CATALOG_CURRENCIES:
        assert f"{ccy}_price_min" in field_names, f"missing {ccy}_price_min"
        assert f"{ccy}_price_max" in field_names, f"missing {ccy}_price_max"


def test_indexes_present(client: MilvusClient) -> None:
    indexes = set(client.list_indexes(COLLECTION))
    missing_inverted = EXPECTED_INVERTED_INDEXES - indexes
    assert not missing_inverted, f"missing INVERTED indexes: {sorted(missing_inverted)}"
    assert "offer_embedding" in indexes, "HNSW vector index missing"
    assert "sparse_codes" in indexes, "BM25 sparse index missing"
    for ccy in CATALOG_CURRENCIES:
        for suffix in ("min", "max"):
            assert f"{ccy}_price_{suffix}" in indexes, f"missing STL_SORT on {ccy}_price_{suffix}"


# --- (2) ANN + BM25 smoke -------------------------------------------------

def test_ann_smoke(client: MilvusClient) -> None:
    """Vector search round-trips against an empty collection — verifies
    the dense leg of the F9 search path is wired correctly."""
    vec = np.random.default_rng(seed=42).standard_normal(DIM).astype(np.float16)
    res = client.search(
        collection_name=COLLECTION,
        data=[vec.tolist()],
        anns_field="offer_embedding",
        limit=5,
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        output_fields=["article_hash"],
    )
    assert isinstance(res, list) and len(res) == 1, "ANN did not return one query group"


def test_bm25_smoke(client: MilvusClient) -> None:
    """Sparse search round-trips against an empty collection — verifies
    the BM25 leg (F6 absorption point) is wired correctly."""
    res = client.search(
        collection_name=COLLECTION,
        data=["bohrmaschine"],
        anns_field="sparse_codes",
        search_params={"metric_type": "BM25"},
        limit=5,
        output_fields=["article_hash"],
    )
    assert isinstance(res, list) and len(res) == 1, "BM25 search did not return one query group"


# --- (3) F3 article-level filter expressions parse + execute --------------

@pytest.mark.parametrize("expr,description", [
    ('article_hash == "abc123"',                                "article_hash equality (Path B → article ANN)"),
    ('article_hash in ["abc", "def", "ghi"]',                   "article_hash IN (Path B bounded probe → ANN)"),
    ('array_contains(eclass5_code, 23172001)',                  "eclass5 hierarchy match"),
    ('array_contains_any(eclass5_code, [23172001, 27182301])',  "eclass5 hierarchy IN"),
    ('array_contains(eclass7_code, 23172090)',                  "eclass7 hierarchy match"),
    ('array_contains_any(s2class_code, [1001, 5042])',          "s2class hierarchy IN"),
    ('array_contains(category_l2, "Werkzeug¦Hand|Maschine")',   "category prefix at depth"),
    ('array_contains_any(category_l1, ["Maschinenbau", "Werkzeug"])',
                                                                "category at root"),
    ('eur_price_min <= 1500.0 and eur_price_max >= 500.0',      "envelope range (sort-by-price browse precondition)"),
    ('chf_price_min <= 1500.0',                                 "envelope range single bound (rare-currency case)"),
])
def test_f3_article_level_filter_expressions_parse_and_execute(client: MilvusClient, expr: str, description: str) -> None:
    """Each expr must parse and execute against the (potentially empty)
    collection."""
    try:
        client.query(collection_name=COLLECTION, filter=expr, output_fields=["article_hash"], limit=1)
    except MilvusException as e:
        pytest.fail(f"filter expr failed to parse/execute ({description}): {expr!r} → {e}")
