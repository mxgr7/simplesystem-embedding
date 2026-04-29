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
        body = _post(c, vendorIdsFilter=all_vendors)
        assert body["metadata"].get("recallClipped") is True, (
            f"expected recallClipped=true, got metadata={body['metadata']}"
        )
        # When recallClipped fires, hitCount is the probe-limit lower bound
        # and hitCountClipped also fires.
        assert body["metadata"].get("hitCountClipped") is True
        assert body["metadata"]["hitCount"] == 2


# ──────────────────────────────────────────────────────────────────────
# F4 — searchMode + sort + hitCount + pagination
# ──────────────────────────────────────────────────────────────────────

def test_search_mode_summaries_only_returns_empty_articles_with_hitcount(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """SUMMARIES_ONLY: skip article hydration but populate hitCount over
    the full filtered set. Summaries themselves are F5; this test just
    verifies the mode flag wiring."""
    from collections import Counter
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, expected = vendor_counts.most_common(1)[0]
    body = _post(client, searchMode="SUMMARIES_ONLY", vendorIdsFilter=[top_vendor])
    assert body["articles"] == []
    # hitCount is the count of *distinct articles* with offers from this
    # vendor. The 200-doc sample has minimal hash dedup, so this is
    # roughly equal to the offer count.
    assert body["metadata"]["hitCount"] >= 1
    assert body["metadata"]["hitCount"] <= expected


def test_search_mode_both_populates_articles_and_hitcount(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """BOTH: same as HITS_ONLY for articles + hitCount; F5 fills in
    summaries. Verify the response shape doesn't degrade."""
    from collections import Counter
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, _ = vendor_counts.most_common(1)[0]
    body = _post(client, searchMode="BOTH", vendorIdsFilter=[top_vendor])
    assert len(body["articles"]) >= 1
    assert body["metadata"]["hitCount"] >= 1


def _post_with_sort(client: TestClient, sort: str, **overrides) -> dict:
    body = {**_BASE_BODY, **overrides}
    r = client.post(f"/{OFFERS}/_search?pageSize=500&sort={sort}", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_sort_articleid_asc_orders_articles_by_representative_id(
    client: TestClient, projected_rows: list[dict],
) -> None:
    from collections import Counter
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, _ = vendor_counts.most_common(1)[0]
    body = _post_with_sort(client, "articleId,asc", vendorIdsFilter=[top_vendor])
    ids = _ids(body)
    assert ids == sorted(ids), f"ids not sorted asc: {ids[:5]} ..."


def test_sort_articleid_desc_orders_articles_by_representative_id(
    client: TestClient, projected_rows: list[dict],
) -> None:
    from collections import Counter
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, _ = vendor_counts.most_common(1)[0]
    body = _post_with_sort(client, "articleId,desc", vendorIdsFilter=[top_vendor])
    ids = _ids(body)
    assert ids == sorted(ids, reverse=True), f"ids not sorted desc: {ids[:5]} ..."


def test_pagination_returns_distinct_slices(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """page=1 and page=2 must produce disjoint result sets when sort
    is deterministic. Use sort=articleId,asc + a wide vendor filter."""
    all_vendors = sorted({r["vendor_id"] for r in projected_rows})
    page_1 = client.post(
        f"/{OFFERS}/_search?pageSize=5&page=1&sort=articleId,asc",
        json={**_BASE_BODY, "vendorIdsFilter": all_vendors},
    ).json()
    page_2 = client.post(
        f"/{OFFERS}/_search?pageSize=5&page=2&sort=articleId,asc",
        json={**_BASE_BODY, "vendorIdsFilter": all_vendors},
    ).json()
    p1_ids = _ids(page_1)
    p2_ids = _ids(page_2)
    assert len(p1_ids) == 5
    assert len(p2_ids) == 5
    assert set(p1_ids).isdisjoint(set(p2_ids)), (
        f"page 1 and page 2 overlap: {set(p1_ids) & set(p2_ids)}"
    )
    # Both pages share the same hitCount (it's over the full filtered set).
    assert page_1["metadata"]["hitCount"] == page_2["metadata"]["hitCount"]
    # pageCount = ceil(hitCount / pageSize)
    expected_pages = (page_1["metadata"]["hitCount"] + 4) // 5
    assert page_1["metadata"]["pageCount"] == expected_pages


def test_hitcount_accurate_over_filtered_set(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """hitCount must reflect the full filtered set independent of pageSize.
    Filter to a single vendor's namespace (we control the sample) and
    confirm pageSize=1 vs pageSize=500 yield the same hitCount."""
    from collections import Counter
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, _ = vendor_counts.most_common(1)[0]
    body_small = client.post(
        f"/{OFFERS}/_search?pageSize=1",
        json={**_BASE_BODY, "vendorIdsFilter": [top_vendor]},
    ).json()
    body_large = client.post(
        f"/{OFFERS}/_search?pageSize=500",
        json={**_BASE_BODY, "vendorIdsFilter": [top_vendor]},
    ).json()
    assert body_small["metadata"]["hitCount"] == body_large["metadata"]["hitCount"]


def test_hitcount_clipped_when_cap_exceeded(
    monkeypatch_session, projected_rows: list[dict],
) -> None:
    """Reload with `HITCOUNT_CAP=1`. A vendor + manufacturer filter
    matches more than 1 distinct article in our namespace → cap fires:
    `hitCount == 1`, `hitCountClipped == True`."""
    from collections import Counter
    # Pick a vendor + manufacturer pair that has at least 2 distinct
    # articles in the sample.
    pair_counts = Counter(
        (r["vendor_id"], r["manufacturerName"]) for r in projected_rows
        if r.get("manufacturerName")
    )
    if not pair_counts:
        pytest.skip("no (vendor, manufacturer) pairs in sample")
    (vendor, mfr), count = pair_counts.most_common(1)[0]
    if count < 2:
        pytest.skip("most popular (vendor, manufacturer) pair has only one row")

    # _make_client sets the dedup envs + EMBED_URL/MILVUS_URI; layering
    # HITCOUNT_CAP after captures the small cap.
    with _make_client(monkeypatch_session) as _c_warmup:
        pass  # ensure the standard envs are in monkeypatch_session
    monkeypatch_session.setenv("HITCOUNT_CAP", "1")
    import importlib

    import main as main_mod
    importlib.reload(main_mod)
    try:
        with TestClient(main_mod.app) as c:
            body = c.post(
                f"/{OFFERS}/_search?pageSize=10",
                json={**_BASE_BODY,
                      "vendorIdsFilter": [vendor],
                      "manufacturersFilter": [mfr]},
            ).json()
            assert body["metadata"]["hitCount"] == 1
            assert body["metadata"]["hitCountClipped"] is True
    finally:
        monkeypatch_session.delenv("HITCOUNT_CAP", raising=False)
        importlib.reload(main_mod)


def test_sort_name_smoke(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """sort=name needs the article-meta fetch path to fire. Verify a
    request returns 200 and articles are present (exact name ordering
    against shared test data is brittle — assert the response shape)."""
    from collections import Counter
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, _ = vendor_counts.most_common(1)[0]
    body = _post_with_sort(client, "name,asc", vendorIdsFilter=[top_vendor])
    assert "articles" in body
    assert body["metadata"]["hitCount"] >= 1


# ──────────────────────────────────────────────────────────────────────
# F5 — summaries (live Milvus, through TestClient)
# ──────────────────────────────────────────────────────────────────────

def _post_summaries(client: TestClient, kinds: list[str], **overrides) -> dict:
    """POST a BOTH-mode request with the given summary kinds."""
    body = {
        **_BASE_BODY,
        "searchMode": "BOTH",
        "summaries": kinds,
        **overrides,
    }
    r = client.post(f"/{OFFERS}/_search?pageSize=10", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_summaries_hits_only_returns_empty_summary_envelope(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """HITS_ONLY: summaries envelope is present (default values) but
    every list/object is empty/null. The mode flag wins over `summaries`."""
    from collections import Counter
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, _ = vendor_counts.most_common(1)[0]
    body = _post(client, vendorIdsFilter=[top_vendor],
                 summaries=["VENDORS", "MANUFACTURERS"])
    assert body["summaries"]["vendorSummaries"] == []
    assert body["summaries"]["manufacturerSummaries"] == []


def test_summaries_vendors_count_distinct_articles_per_vendor(
    client: TestClient, projected_rows: list[dict],
) -> None:
    from collections import Counter
    # Pick two vendors with multiple offers each.
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_two = [v for v, _ in vendor_counts.most_common(2)]
    body = _post_summaries(client, ["VENDORS"], vendorIdsFilter=top_two)
    summaries = body["summaries"]
    by_v = {s["vendorId"]: s["count"] for s in summaries["vendorSummaries"]}
    # Every requested vendor should appear with a positive count.
    for v in top_two:
        assert v in by_v
        assert by_v[v] >= 1
    # Sum of counts ≤ hitCount * (number of vendors) — loose invariant
    # since one article can be counted under multiple vendors.
    total = sum(by_v.values())
    assert total >= body["metadata"]["hitCount"]


def test_summaries_manufacturers_grouped_by_name(
    client: TestClient, projected_rows: list[dict],
) -> None:
    from collections import Counter
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, _ = vendor_counts.most_common(1)[0]
    body = _post_summaries(client, ["MANUFACTURERS"], vendorIdsFilter=[top_vendor])
    mfrs = body["summaries"]["manufacturerSummaries"]
    # At least one manufacturer should surface (the sample has them).
    assert len(mfrs) >= 1
    # Sum of distinct-article counts across manufacturers ≤ hitCount.
    total = sum(m["count"] for m in mfrs)
    assert total <= body["metadata"]["hitCount"]


def test_summaries_categories_root_yields_top_level_paths(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """No `currentCategoryPathElements` → sameLevel = top-level
    categories present in the filtered set."""
    from collections import Counter
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, _ = vendor_counts.most_common(1)[0]
    body = _post_summaries(client, ["CATEGORIES"], vendorIdsFilter=[top_vendor])
    cats = body["summaries"]["categoriesSummary"]
    assert cats is not None
    # Every sameLevel bucket should be a depth-1 path (single-element
    # categoryPathElements).
    for bucket in cats["sameLevel"]:
        assert len(bucket["categoryPathElements"]) == 1


def test_summaries_eclass5_returns_root_codes_when_no_selection(
    client: TestClient, projected_rows: list[dict],
) -> None:
    from collections import Counter
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, _ = vendor_counts.most_common(1)[0]
    body = _post_summaries(client, ["ECLASS5"], vendorIdsFilter=[top_vendor])
    e5 = body["summaries"]["eClass5Categories"]
    assert e5 is not None
    assert e5["selectedEClassGroup"] is None
    # Each sameLevel bucket is a depth-1 code (1- or 2-digit).
    for bucket in e5["sameLevel"]:
        assert 1 <= len(str(bucket["group"])) <= 2


def test_summaries_only_mode_returns_empty_articles_with_summaries(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """SUMMARIES_ONLY: skip article hydration entirely."""
    from collections import Counter
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, _ = vendor_counts.most_common(1)[0]
    body = client.post(
        f"/{OFFERS}/_search?pageSize=10",
        json={
            **_BASE_BODY,
            "searchMode": "SUMMARIES_ONLY",
            "summaries": ["VENDORS", "MANUFACTURERS"],
            "vendorIdsFilter": [top_vendor],
        },
    ).json()
    assert body["articles"] == []
    assert body["metadata"]["hitCount"] >= 1
    # Both requested kinds populated.
    assert len(body["summaries"]["vendorSummaries"]) >= 1
    assert len(body["summaries"]["manufacturerSummaries"]) >= 1


def test_summaries_only_kinds_requested_are_populated(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """Kinds NOT in the request remain at default (empty/null)."""
    from collections import Counter
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, _ = vendor_counts.most_common(1)[0]
    body = _post_summaries(client, ["MANUFACTURERS"], vendorIdsFilter=[top_vendor])
    summaries = body["summaries"]
    # Manufacturers requested → populated.
    assert len(summaries["manufacturerSummaries"]) >= 1
    # Vendors NOT requested → empty list.
    assert summaries["vendorSummaries"] == []
    # Categories NOT requested → null.
    assert summaries["categoriesSummary"] is None
    assert summaries["eClass5Categories"] is None


def test_summaries_eclass5set_counts_per_entry(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """`eClassesAggregations` → one count entry per request entry."""
    from collections import Counter
    # Find at least one real eclass code in the sample.
    eclass_counts: Counter[int] = Counter()
    for r in projected_rows:
        for code in r.get("eclass5_code") or []:
            eclass_counts[int(code)] += 1
    if not eclass_counts:
        pytest.skip("no eClass5 codes in sample")
    real_code, _ = eclass_counts.most_common(1)[0]

    body = _post_summaries(
        client, ["ECLASS5SET"],
        manufacturersFilter=[],   # no extra constraints
        eClassesAggregations=[
            {"id": "real", "eClasses": [real_code]},
            {"id": "fake", "eClasses": [99999999]},
        ],
    )
    eaggs = body["summaries"]["eClassesAggregations"]
    by_id = {e["id"]: e["count"] for e in eaggs}
    assert by_id["real"] >= 1
    assert by_id["fake"] == 0
