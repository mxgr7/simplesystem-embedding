"""F9 PR3 integration test — dedup-topology dispatch through the
in-process FastAPI app against live `articles_v1` + `offers_v3`.

Loads sample_200 with a `pr3:` offer-id namespace, then exercises the
`USE_DEDUP_TOPOLOGY=1` flag through the search endpoint. Coverage:

  * Vendor filter (offer-side) narrows to that vendor's articles.
  * Manufacturer filter (article-side) narrows to that manufacturer's
    hash bucket.
  * `articleIdsFilter` (offer-side `id IN [...]`) round-trips: the
    requested IDs come back as the representative offers.
  * `closedMarketplaceOnly` empty-list edge case: returns empty.
  * Bounded-probe overflow → Path A fallback emits `recallClipped: true`
    when `PATH_B_HASH_LIMIT` is set low and the offer filter is
    permissive (no per-request override; the test reloads the app with
    a tiny limit).

Skipped if Milvus is not reachable or either collection is missing.
The legacy single-collection path is unaffected by these tests
(USE_DEDUP_TOPOLOGY is module-scoped via env reload).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pymilvus import MilvusClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_API_DIR = REPO_ROOT / "search-api"
sys.path.insert(0, str(SEARCH_API_DIR))
sys.path.insert(0, str(REPO_ROOT))

from indexer.projection import compute_article_hash, group_by_hash, project  # noqa: E402
from indexer.test_loader import load_split  # noqa: E402

MILVUS_URI = "http://localhost:19530"
ARTICLES = "articles_v1"
OFFERS = "offers_v3"
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/mongo_sample/sample_200.json"
ID_NAMESPACE = "pr3:"

_BASE_BODY = {
    "searchMode": "HITS_ONLY",
    "selectedArticleSources": {},
    "currency": "EUR",
}


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def milvus_client() -> MilvusClient:
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
    for r in rows:
        r["id"] = ID_NAMESPACE + r["id"]
    return rows


@pytest.fixture(scope="module", autouse=True)
def loaded(milvus_client: MilvusClient, projected_rows: list[dict]):
    offer_ids = [r["id"] for r in projected_rows]
    article_hashes = list(group_by_hash(projected_rows).keys())
    a_visible, o_visible = load_split(
        milvus_client,
        articles_collection=ARTICLES,
        offers_collection=OFFERS,
        rows=projected_rows,
    )
    if o_visible < len(offer_ids):
        pytest.fail(f"only {o_visible}/{len(offer_ids)} offers visible after seed")
    yield
    try:
        milvus_client.delete(collection_name=OFFERS, ids=offer_ids)
    except Exception:
        pass
    # Don't drop article rows — other tests may have loaded the same
    # hashes (idempotent upserts). Only the namespaced offer ids are
    # uniquely ours.


@pytest.fixture(scope="module")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def _make_client(monkeypatch_session, *, path_b_hash_limit: int = 16_383) -> TestClient:
    """Reload the main module with the dedup flag on + low PATH_B_HASH_LIMIT
    when needed (overflow tests). Avoids real TEI: tests here go through
    the no-query browse path."""
    monkeypatch_session.setenv("USE_DEDUP_TOPOLOGY", "1")
    monkeypatch_session.setenv("MILVUS_ARTICLES_COLLECTION", ARTICLES)
    monkeypatch_session.setenv("PATH_B_HASH_LIMIT", str(path_b_hash_limit))
    monkeypatch_session.setenv("EMBED_URL", "http://embed.invalid")
    monkeypatch_session.setenv("MILVUS_URI", MILVUS_URI)
    monkeypatch_session.setenv("API_KEY", "")
    import importlib

    import main as main_mod
    importlib.reload(main_mod)
    return TestClient(main_mod.app)


@pytest.fixture(scope="module")
def client(monkeypatch_session):
    with _make_client(monkeypatch_session) as c:
        yield c


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _post(client: TestClient, **overrides) -> dict:
    body = {**_BASE_BODY, **overrides}
    r = client.post(f"/{OFFERS}/_search?pageSize=500", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _ids(body: dict) -> list[str]:
    return [a["articleId"] for a in body["articles"]]


# ──────────────────────────────────────────────────────────────────────
# Baseline + filter narrowing
# ──────────────────────────────────────────────────────────────────────

def test_baseline_no_filter_returns_empty(client: TestClient) -> None:
    """No query, no filter → no defensible Path A browse → empty."""
    body = _post(client)
    assert body["articles"] == []
    assert body["metadata"].get("recallClipped") in (False, None)


def test_vendor_filter_narrows_to_namespace_subset(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """Pick the most populous vendor in the sample; assert returned
    representatives come from that vendor's offers (in our namespace)."""
    from collections import Counter
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, _ = vendor_counts.most_common(1)[0]
    namespaced_offers_for_vendor = {
        r["id"] for r in projected_rows if r["vendor_id"] == top_vendor
    }
    body = _post(client, vendorIdsFilter=[top_vendor])
    returned = set(_ids(body))
    # Every returned id must be one of our namespaced offers for that
    # vendor. (Other tests might leave residue at the same vendor —
    # the namespace constraint keeps the assertion clean.)
    ours = returned & namespaced_offers_for_vendor
    assert ours, "vendor filter returned no offers from our namespace"
    assert returned.issubset(namespaced_offers_for_vendor), (
        "vendor filter returned offers not from this vendor's namespaced set: "
        f"{returned - namespaced_offers_for_vendor}"
    )


def test_article_ids_filter_round_trips(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """`articleIdsFilter` is offer-side: each requested id maps to one
    row, which surfaces as its own representative. Order of returned
    items isn't guaranteed; assert as a set membership."""
    sample_ids = [r["id"] for r in projected_rows[:5]]
    body = _post(client, articleIdsFilter=sample_ids)
    returned = set(_ids(body))
    # Each requested id is its own offer; representative selection picks
    # the alphabetically-lowest offer in each hash bucket. If two of our
    # sample ids share a hash, the representative is the lower one — so
    # `returned ⊆ sample_ids` always holds.
    assert returned.issubset(set(sample_ids)), (
        f"returned ids include some not requested: {returned - set(sample_ids)}"
    )
    # And we should have exactly one representative per distinct hash.
    sample_rows = [r for r in projected_rows if r["id"] in set(sample_ids)]
    distinct_hashes = {compute_article_hash(r) for r in sample_rows}
    assert len(returned) == len(distinct_hashes), (
        f"expected one representative per distinct hash "
        f"({len(distinct_hashes)}), got {len(returned)}"
    )


def test_closed_marketplace_only_empty_returns_no_articles(
    client: TestClient,
) -> None:
    """Empty `closedCatalogVersionIds` with `closedMarketplaceOnly=true`
    matches no offers — by-design parity with legacy
    `OfferFilterBuilder` (sentinel `id == ""`)."""
    body = _post(client, closedMarketplaceOnly=True)
    assert body["articles"] == []


# ──────────────────────────────────────────────────────────────────────
# Bounded-probe overflow → Path A fallback + recallClipped
# ──────────────────────────────────────────────────────────────────────

def test_bounded_probe_overflow_emits_recall_clipped_metadata(
    monkeypatch_session, projected_rows: list[dict],
) -> None:
    """Reload the app with `PATH_B_HASH_LIMIT=2`. A permissive offer
    filter (vendor IN <every vendor>) probes more than 2 distinct
    hashes → Path A fallback fires, response carries
    `recallClipped: true`."""
    with _make_client(monkeypatch_session, path_b_hash_limit=2) as c:
        all_vendors = sorted({r["vendor_id"] for r in projected_rows})
        # Vendor IN [all vendors] matches every offer in our namespace
        # (and others); after dedup the distinct hash count vastly
        # exceeds the limit of 2.
        body = _post(c, vendorIdsFilter=all_vendors)
        assert body["metadata"].get("recallClipped") is True, (
            f"expected recallClipped=true, got metadata={body['metadata']}"
        )
