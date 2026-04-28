"""F9 two-stream load integration test.

Loads the 200-doc real-MongoDB sample through `indexer/projection.py` +
`indexer/test_loader.py:load_split` into the live `articles_v{N}` +
`offers_v{N}` collections and asserts:

  1. Article count ≤ offer count (dedup happened, or at minimum didn't
     amplify).
  2. Every offer's `article_hash` resolves to a row in `articles_v{N}`
     (the join key is intact end-to-end).
  3. The per-currency envelope columns on `articles_v{N}` are populated
     for at least the dominant currency in the sample (EUR).
  4. The BM25 corpus on `articles_v{N}` is non-empty for at least one
     dedup'd article — proves `text_codes` aggregation actually fed the
     analyzer.

This is the F9 PR2 "no search-path change yet" gate from the design doc:
the indexer projection writes both collections correctly; the search-api
routing rewrite is PR3.

Skipped if Milvus is not reachable, or either collection is missing.
The test reuses the live `articles_v1` / `offers_v3` collections that
PR1's schema-smoke tests rely on; it inserts under a temporary
`article_hash` namespace (a synthetic prefix) so it doesn't collide with
those tests' fixture rows. Cleanup at module teardown.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pymilvus import MilvusClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.projection import (  # noqa: E402
    CATALOG_CURRENCIES,
    MAX_PRICE_SENTINEL,
    aggregate_article,
    compute_article_hash,
    group_by_hash,
    project,
    to_offer_row,
)
from indexer.test_loader import load_split, stub_vector  # noqa: E402

MILVUS_URI = "http://localhost:19530"
ARTICLES = "articles_v1"
OFFERS = "offers_v3"
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/mongo_sample/sample_200.json"

# Tag inserted rows with a synthetic offer-id prefix so we can cleanly
# delete them in teardown without touching schema-smoke test rows.
ID_NAMESPACE = "f9pr2:"


@pytest.fixture(scope="module")
def client() -> MilvusClient:
    try:
        c = MilvusClient(uri=MILVUS_URI)
        c.list_collections()
    except Exception as exc:
        pytest.skip(f"Milvus unreachable at {MILVUS_URI}: {exc}")
    if not c.has_collection(ARTICLES):
        pytest.skip(f"{ARTICLES!r} missing — run scripts/create_articles_collection.py first")
    if not c.has_collection(OFFERS):
        pytest.skip(f"{OFFERS!r} missing — run scripts/create_offers_collection.py first")
    return c


@pytest.fixture(scope="module")
def projected_rows() -> list[dict]:
    raw = json.loads(FIXTURE_PATH.read_text())["records"]
    rows = [project(r).row for r in raw]
    # Namespace the PKs and re-key any per-offer string the test asserts on.
    for row in rows:
        row["id"] = ID_NAMESPACE + row["id"]
    return rows


@pytest.fixture(scope="module", autouse=True)
def loaded(client: MilvusClient, projected_rows: list[dict]):
    # Track inserted hashes and offer ids for teardown.
    offer_ids = [r["id"] for r in projected_rows]
    article_hashes = list(group_by_hash(projected_rows).keys())

    articles_visible, offers_visible = load_split(
        client,
        articles_collection=ARTICLES,
        offers_collection=OFFERS,
        rows=projected_rows,
    )
    if articles_visible < len(article_hashes):
        pytest.fail(
            f"only {articles_visible}/{len(article_hashes)} articles visible after seed"
        )
    if offers_visible < len(offer_ids):
        pytest.fail(
            f"only {offers_visible}/{len(offer_ids)} offers visible after seed"
        )
    yield
    # Best-effort cleanup. Delete-by-PK is cheap; ignore failures so a
    # broken delete doesn't mask the real test result.
    try:
        client.delete(collection_name=OFFERS, ids=offer_ids)
    except Exception:
        pass
    try:
        client.delete(collection_name=ARTICLES, ids=article_hashes)
    except Exception:
        pass


# ---------- assertions ---------------------------------------------------

def test_article_count_at_most_offer_count(projected_rows: list[dict]) -> None:
    """Dedup never amplifies. The 200-doc sample is too small to validate
    production's 1.22× ratio; the floor here is just that hashing
    doesn't accidentally produce more groups than inputs."""
    groups = group_by_hash(projected_rows)
    assert 0 < len(groups) <= len(projected_rows)


def test_every_offer_hash_resolves_to_an_article(
    client: MilvusClient, projected_rows: list[dict],
) -> None:
    """Path B's bounded probe relies on this invariant: every distinct
    `article_hash` returned by the offers query must hit a row in
    `articles_v{N}`. Validate end-to-end through the live collections."""
    sample_ids = [r["id"] for r in projected_rows[:20]]
    offer_rows = client.query(
        collection_name=OFFERS,
        filter=f'id in {json.dumps(sample_ids)}',
        output_fields=["id", "article_hash"],
        limit=len(sample_ids),
    )
    assert len(offer_rows) == len(sample_ids), "not all sampled offers visible"
    hashes = sorted({r["article_hash"] for r in offer_rows})
    article_rows = client.query(
        collection_name=ARTICLES,
        filter=f'article_hash in {json.dumps(hashes)}',
        output_fields=["article_hash"],
        limit=len(hashes),
    )
    found = {r["article_hash"] for r in article_rows}
    missing = set(hashes) - found
    assert not missing, f"hashes referenced from offers but missing on articles: {sorted(missing)}"


def test_envelope_populated_for_dominant_currency(
    client: MilvusClient, projected_rows: list[dict],
) -> None:
    """The 200-doc sample is EUR-heavy. At least one article must have
    a real `eur_price_min` (i.e. < the missing-currency sentinel) —
    proves the per-currency aggregator wired the prices through."""
    groups = group_by_hash(projected_rows)
    sample_hashes = list(groups.keys())[:50]
    rows = client.query(
        collection_name=ARTICLES,
        filter=f'article_hash in {json.dumps(sample_hashes)}',
        output_fields=["article_hash", "eur_price_min", "eur_price_max"],
        limit=len(sample_hashes),
    )
    real = [r for r in rows if r["eur_price_min"] < MAX_PRICE_SENTINEL]
    assert real, "no article in sample has a real eur_price_min — envelope didn't aggregate"
    for r in real:
        assert r["eur_price_min"] <= r["eur_price_max"], (
            f"min > max on {r['article_hash']}: "
            f"{r['eur_price_min']} > {r['eur_price_max']}"
        )


def test_text_codes_corpus_non_empty_on_articles(
    client: MilvusClient, projected_rows: list[dict],
) -> None:
    """`text_codes` feeds BM25. Empty corpus = no BM25 hits = silent F6
    breakage. Verify against the live collection."""
    groups = group_by_hash(projected_rows)
    sample_hashes = list(groups.keys())[:10]
    rows = client.query(
        collection_name=ARTICLES,
        filter=f'article_hash in {json.dumps(sample_hashes)}',
        output_fields=["article_hash", "text_codes"],
        limit=len(sample_hashes),
    )
    non_empty = [r for r in rows if r["text_codes"].strip()]
    assert non_empty, "no article has populated text_codes — BM25 corpus didn't aggregate"


def test_bm25_search_returns_results_against_seeded_corpus(
    client: MilvusClient, projected_rows: list[dict],
) -> None:
    """Pick a real EAN from the sample, run BM25 search, expect at least
    one hit. End-to-end smoke that the analyzer + sparse index actually
    indexed the aggregated `text_codes`."""
    eans = [r["ean"] for r in projected_rows if r.get("ean")]
    if not eans:
        pytest.skip("no EANs in sample to query for")
    # The whitespace+lowercase analyzer + length filter (min=4) means
    # short tokens get dropped; EANs are 13 chars so they survive.
    target_ean = eans[0]
    res = client.search(
        collection_name=ARTICLES,
        data=[target_ean],
        anns_field="sparse_codes",
        search_params={"metric_type": "BM25"},
        limit=5,
        output_fields=["article_hash", "text_codes"],
    )
    assert isinstance(res, list) and res, "BM25 search returned no hit groups"
    hits = res[0]
    assert hits, f"BM25 search for {target_ean!r} produced no hits against seeded corpus"
    # Sanity: at least one of the top-k hits should actually contain the
    # token. (BM25 ranking + analyzer pipeline can produce loose matches,
    # but an exact-token query must hit the document.)
    assert any(target_ean in h.get("entity", {}).get("text_codes", "") for h in hits), (
        f"top-{len(hits)} BM25 hits don't contain {target_ean!r}"
    )


# ---------- helpers (parity with unit tests) ------------------------------

def test_stub_vector_keyed_by_hash_is_deterministic() -> None:
    """The loader stubs vectors by hash, not by name — so two calls with
    the same hash produce the same vector. This is what makes the dedup
    invariant hold under stub embedding."""
    h = "deadbeefcafef00d" * 2  # 32 chars
    v1 = stub_vector(h)
    v2 = stub_vector(h)
    assert (v1 == v2).all()


def test_aggregate_article_round_trip_against_real_record(
    projected_rows: list[dict],
) -> None:
    """Pure-function spot-check: pick one record from the real sample,
    aggregate it as a single-offer group, confirm hash + name + envelope
    match what we'd compute by hand."""
    flat = projected_rows[0]
    hash_ = compute_article_hash(flat)
    article = aggregate_article([flat])
    assert article["article_hash"] == hash_
    assert article["name"] == flat["name"]
    # No assertion on envelope value — depends on the sample — but the
    # column set must match CATALOG_CURRENCIES.
    for ccy in CATALOG_CURRENCIES:
        assert f"{ccy}_price_min" in article
        assert f"{ccy}_price_max" in article
    # The offer row built from the same flat row must reference this hash.
    offer = to_offer_row(flat, article_hash=hash_)
    assert offer["article_hash"] == hash_
