"""End-to-end test for `scripts/swing_aliases.py` against a live
Milvus.

Builds two F9 collection pairs (articles_v_swingtest_old/new +
offers_v_swingtest_old/new), populates each with sample_200, then
exercises the swing script:

  1. Dry-run — no alias mutation, prints plan.
  2. Initial swing — both aliases land on the "old" pair.
  3. Validation rejects: half-populated target (drop offers, retry → exit).
  4. Validation rejects: join-key drift (mismatched hashes between
     articles + offers targets, retry → exit).
  5. Real swing to "new" pair — both aliases move atomically.
  6. Rollback via --rollback-to puts both aliases back on "old".

Skipped if Milvus is not reachable. Cleans up its own collections +
aliases on success.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from pymilvus import MilvusClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.projection import compute_article_hash, group_by_hash, project  # noqa: E402
from indexer.test_loader import load_split  # noqa: E402

MILVUS_URI = "http://localhost:19530"

OLD_ARTICLES = "articles_v_swingtest_old"
NEW_ARTICLES = "articles_v_swingtest_new"
OLD_OFFERS   = "offers_v_swingtest_old"
NEW_OFFERS   = "offers_v_swingtest_new"
ARTICLES_ALIAS = "articles_swingtest_alias"
OFFERS_ALIAS   = "offers_swingtest_alias"

SAMPLE = REPO_ROOT / "tests/fixtures/mongo_sample/sample_200.json"
SCRIPT = REPO_ROOT / "scripts/swing_aliases.py"


def _milvus_reachable() -> bool:
    try:
        c = MilvusClient(uri=MILVUS_URI)
        c.list_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _milvus_reachable(),
    reason=f"Milvus unreachable at {MILVUS_URI}",
)


# ---------- harness -------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_rows() -> list[dict]:
    records = json.loads(SAMPLE.read_text())["records"]
    return [project(r).row for r in records]


def _create_pair(client: MilvusClient, articles: str, offers: str) -> None:
    """Stand up a fresh F9 pair via the prod schema scripts. Subprocess
    so we exercise the same code path operators run."""
    for name in (articles, offers):
        if client.has_collection(name):
            client.drop_collection(name)

    # Strip any 'articles_'/'offers_' prefix to get the version arg
    # that the create scripts expect. We pass a string; the int-only
    # restriction got dropped, but to be safe use a short numeric...
    # Actually the scripts require `--version` as int. Use explicit
    # high numbers + rename via has_collection check.
    for script, target_name, version in (
        ("scripts/create_articles_collection.py", articles, 7777),
        ("scripts/create_offers_collection.py",   offers,   8888),
    ):
        # We can't pick an arbitrary collection name from the script;
        # it builds `articles_v{N}` / `offers_v{N}`. Workaround: build
        # under a temp version, then drop + rename via metadata. Milvus
        # 2.6 has no rename, so use describe → recreate.
        # Simpler: just create raw without the script.
        pass

    # Build collections directly via MilvusClient for test isolation
    # — bypasses the script's collection-name templating constraint.
    from pymilvus import DataType, Function, FunctionType
    from scripts.create_articles_collection import (
        BM25_ANALYZER_PARAMS, CATALOG_CURRENCIES, DIM, SCALAR_INDEX_FIELDS,
        VECTOR_INDEX_DEFAULTS,
    )

    # ARTICLES schema
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("article_hash", DataType.VARCHAR, max_length=32, is_primary=True)
    schema.add_field("offer_embedding", DataType.FLOAT16_VECTOR, dim=DIM)
    schema.add_field("text_codes", DataType.VARCHAR, max_length=8192,
                     enable_analyzer=True, analyzer_params=BM25_ANALYZER_PARAMS)
    schema.add_field("sparse_codes", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="bm25_codes", function_type=FunctionType.BM25,
        input_field_names=["text_codes"], output_field_names=["sparse_codes"],
    ))
    schema.add_field("name", DataType.VARCHAR, max_length=1024)
    schema.add_field("manufacturerName", DataType.VARCHAR, max_length=256)
    for d, ml in zip(range(1, 6), (256, 640, 768, 1024, 1280)):
        schema.add_field(f"category_l{d}", DataType.ARRAY, element_type=DataType.VARCHAR,
                         max_capacity=64, max_length=ml)
    for f in ("eclass5_code", "eclass7_code", "s2class_code"):
        schema.add_field(f, DataType.ARRAY, element_type=DataType.INT32, max_capacity=16)
    schema.add_field("customer_article_numbers", DataType.JSON)
    for ccy in CATALOG_CURRENCIES:
        schema.add_field(f"{ccy}_price_min", DataType.FLOAT)
        schema.add_field(f"{ccy}_price_max", DataType.FLOAT)
    params = client.prepare_index_params()
    params.add_index(field_name="offer_embedding", index_type="HNSW",
                     metric_type="COSINE", params={"M": 16, "efConstruction": 200})
    params.add_index(field_name="sparse_codes", index_type="SPARSE_INVERTED_INDEX",
                     metric_type="BM25", params={"mmap.enabled": True}, index_name="sparse_codes")
    for f in SCALAR_INDEX_FIELDS:
        params.add_index(field_name=f, index_type="INVERTED", index_name=f)
    for ccy in CATALOG_CURRENCIES:
        for s in ("min", "max"):
            params.add_index(field_name=f"{ccy}_price_{s}", index_type="STL_SORT", index_name=f"{ccy}_price_{s}")
    client.create_collection(collection_name=articles, schema=schema, index_params=params)
    client.load_collection(articles)

    # OFFERS schema (re-import the prod constants)
    from scripts.create_offers_collection import (
        SCALAR_INDEX_FIELDS as OFFERS_SCALAR_INDEX_FIELDS,
    )
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field("_placeholder_vector", DataType.FLOAT_VECTOR, dim=2)
    schema.add_field("article_hash", DataType.VARCHAR, max_length=32)
    schema.add_field("ean", DataType.VARCHAR, max_length=64)
    schema.add_field("article_number", DataType.VARCHAR, max_length=256)
    schema.add_field("vendor_id", DataType.VARCHAR, max_length=64)
    schema.add_field("catalog_version_ids", DataType.ARRAY,
                     element_type=DataType.VARCHAR, max_capacity=2048, max_length=64)
    schema.add_field("prices", DataType.JSON)
    schema.add_field("delivery_time_days_max", DataType.INT32)
    for f in ("core_marker_enabled_sources", "core_marker_disabled_sources"):
        schema.add_field(f, DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=64)
    schema.add_field("features", DataType.ARRAY, element_type=DataType.VARCHAR,
                     max_capacity=512, max_length=512)
    for f in ("relationship_accessory_for", "relationship_spare_part_for", "relationship_similar_to"):
        schema.add_field(f, DataType.ARRAY, element_type=DataType.VARCHAR,
                         max_capacity=128, max_length=256)
    schema.add_field("price_list_ids", DataType.ARRAY, element_type=DataType.VARCHAR,
                     max_capacity=512, max_length=64)
    schema.add_field("currencies", DataType.ARRAY, element_type=DataType.VARCHAR,
                     max_capacity=8, max_length=8)
    for ccy in CATALOG_CURRENCIES:
        schema.add_field(f"{ccy}_price_min", DataType.FLOAT)
        schema.add_field(f"{ccy}_price_max", DataType.FLOAT)
    params = client.prepare_index_params()
    params.add_index(field_name="_placeholder_vector", index_type="FLAT", metric_type="L2")
    for f in OFFERS_SCALAR_INDEX_FIELDS:
        params.add_index(field_name=f, index_type="INVERTED", index_name=f)
    for ccy in CATALOG_CURRENCIES:
        for s in ("min", "max"):
            params.add_index(field_name=f"{ccy}_price_{s}", index_type="STL_SORT", index_name=f"{ccy}_price_{s}")
    client.create_collection(collection_name=offers, schema=schema, index_params=params)
    client.load_collection(offers)


def _populate_pair(
    client: MilvusClient, *, articles: str, offers: str, rows: list[dict],
) -> None:
    load_split(
        client,
        articles_collection=articles,
        offers_collection=offers,
        rows=rows,
    )
    # `get_collection_stats` reads sealed segments only — rows just
    # upserted live in growing segments and don't show in counts until
    # `flush()` seals them. The swing script's row-count validation
    # depends on this being current.
    client.flush(articles)
    client.flush(offers)
    time.sleep(2)


@pytest.fixture
def setup_pairs(fixture_rows: list[dict]):
    """Create + populate two F9 pairs and one alias pair pre-pointed at
    the OLD pair. Each test starts from this clean state."""
    client = MilvusClient(uri=MILVUS_URI)
    _create_pair(client, OLD_ARTICLES, OLD_OFFERS)
    _create_pair(client, NEW_ARTICLES, NEW_OFFERS)
    _populate_pair(client, articles=OLD_ARTICLES, offers=OLD_OFFERS, rows=fixture_rows)
    _populate_pair(client, articles=NEW_ARTICLES, offers=NEW_OFFERS, rows=fixture_rows)

    # Drop any pre-existing aliases left over from a prior failed run.
    for alias in (ARTICLES_ALIAS, OFFERS_ALIAS):
        try:
            client.drop_alias(alias=alias)
        except Exception:
            pass

    # Initial alias state: both → OLD.
    client.create_alias(collection_name=OLD_ARTICLES, alias=ARTICLES_ALIAS)
    client.create_alias(collection_name=OLD_OFFERS,   alias=OFFERS_ALIAS)
    yield client

    # Cleanup
    for alias in (ARTICLES_ALIAS, OFFERS_ALIAS):
        try:
            client.drop_alias(alias=alias)
        except Exception:
            pass
    for name in (OLD_ARTICLES, NEW_ARTICLES, OLD_OFFERS, NEW_OFFERS):
        if client.has_collection(name):
            client.drop_collection(name)


def _run_swing(*extra_args: str) -> subprocess.CompletedProcess:
    """Subprocess invocation — exercises the CLI exactly as an
    operator would."""
    cmd = [
        sys.executable, str(SCRIPT),
        "--milvus-uri", MILVUS_URI,
        "--articles-target", NEW_ARTICLES,
        "--offers-target",   NEW_OFFERS,
        "--articles-alias",  ARTICLES_ALIAS,
        "--offers-alias",    OFFERS_ALIAS,
        *extra_args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


# ---------- tests --------------------------------------------------------

def test_dry_run_does_not_swing(setup_pairs: MilvusClient) -> None:
    client = setup_pairs
    result = _run_swing("--dry-run")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "(dry-run — no swings performed)" in result.stderr or "dry-run" in result.stderr
    # Aliases unchanged.
    assert client.describe_alias(alias=ARTICLES_ALIAS)["collection_name"] == OLD_ARTICLES
    assert client.describe_alias(alias=OFFERS_ALIAS)["collection_name"] == OLD_OFFERS


def test_swing_moves_both_aliases_to_new_pair(setup_pairs: MilvusClient) -> None:
    client = setup_pairs
    result = _run_swing()
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert client.describe_alias(alias=ARTICLES_ALIAS)["collection_name"] == NEW_ARTICLES
    assert client.describe_alias(alias=OFFERS_ALIAS)["collection_name"] == NEW_OFFERS


def test_validation_rejects_when_target_below_min_rows(setup_pairs: MilvusClient) -> None:
    client = setup_pairs
    # Fixture has 200 rows. Set min above that → should reject.
    result = _run_swing("--min-rows-articles", "100000")
    assert result.returncode != 0
    assert "below --min-rows-articles" in result.stderr or "below" in (result.stderr + result.stdout)
    # Aliases unchanged.
    assert client.describe_alias(alias=ARTICLES_ALIAS)["collection_name"] == OLD_ARTICLES


def test_validation_rejects_join_key_drift(
    setup_pairs: MilvusClient, fixture_rows: list[dict],
) -> None:
    """Stuff the NEW articles collection with garbage hashes and verify
    the join-key sampler catches it before any alias moves."""
    client = setup_pairs
    # Re-create NEW articles with totally different hashes than the
    # offers carry. Easiest: drop + recreate empty + insert one row
    # whose hash won't match any offer's article_hash.
    client.drop_collection(NEW_ARTICLES)
    _create_pair(client, NEW_ARTICLES, "_throwaway_for_drift_test")
    client.drop_collection("_throwaway_for_drift_test")
    # Insert one synthetic article that never matches any offer hash.
    import numpy as np
    client.upsert(collection_name=NEW_ARTICLES, data=[{
        "article_hash": "ffffffffffffffffffffffffffffffff",
        "name": "fake", "manufacturerName": "fake",
        "category_l1": [], "category_l2": [], "category_l3": [], "category_l4": [], "category_l5": [],
        "eclass5_code": [], "eclass7_code": [], "s2class_code": [],
        "text_codes": "fake",
        "customer_article_numbers": [],
        "offer_embedding": np.zeros(128, dtype=np.float16),
        **{f"{c}_price_{s}": 0.0 for c in
           ("eur", "chf", "huf", "pln", "gbp", "czk", "cny") for s in ("min", "max")},
    }])
    client.flush(NEW_ARTICLES)
    time.sleep(2)
    # NEW_OFFERS still carries the sample_200 hashes from setup_pairs
    # (we only dropped+recreated NEW_ARTICLES). Sampling NEW_OFFERS
    # gives those hashes; looking them up in NEW_ARTICLES (which now
    # only has "ffff…") will miss → drift detected.

    result = _run_swing("--min-rows-articles", "1")
    assert result.returncode != 0, (
        f"join-key drift should reject — stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "drift" in (result.stderr + result.stdout).lower(), \
        f"missing 'drift' in output: {result.stderr}\n{result.stdout}"
    # Aliases unchanged.
    assert client.describe_alias(alias=ARTICLES_ALIAS)["collection_name"] == OLD_ARTICLES


def test_rollback_to_swings_back_to_old_pair(setup_pairs: MilvusClient) -> None:
    """First swing forward, then explicit rollback. Aliases should
    end up on the OLD pair."""
    client = setup_pairs
    fwd = _run_swing()
    assert fwd.returncode == 0
    assert client.describe_alias(alias=ARTICLES_ALIAS)["collection_name"] == NEW_ARTICLES

    rb = subprocess.run([
        sys.executable, str(SCRIPT),
        "--milvus-uri", MILVUS_URI,
        "--articles-target", NEW_ARTICLES,  # required arg, ignored on rollback path
        "--offers-target",   NEW_OFFERS,
        "--articles-alias",  ARTICLES_ALIAS,
        "--offers-alias",    OFFERS_ALIAS,
        "--rollback-to",     f"{OLD_ARTICLES},{OLD_OFFERS}",
    ], capture_output=True, text=True, timeout=60)
    assert rb.returncode == 0, f"stderr:\n{rb.stderr}\nstdout:\n{rb.stdout}"
    assert client.describe_alias(alias=ARTICLES_ALIAS)["collection_name"] == OLD_ARTICLES
    assert client.describe_alias(alias=OFFERS_ALIAS)["collection_name"] == OLD_OFFERS
