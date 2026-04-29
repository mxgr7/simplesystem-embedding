"""F9 routing dispatcher — unit tests against a mock MilvusClient.

Pure-Python tests with no Milvus dependency. Covers:

  * Path A (offer_expr is None): articles ANN ranks, offers attached
    via `hash IN [...]`, representative selection, price post-pass.
  * Path B happy path: bounded probe → article ANN constrained to
    probe hashes → re-attach probe offers.
  * Path B overflow → Path A fallback with `recall_clipped=True`.
  * `articleIdsFilter` round-trip (offer-side `id IN [...]`).
  * Per-vendor blocked_eclass_vendors Python post-pass.
  * Representative-offer pick: lowest articleId among offers passing
    price filter.

`asyncio.run` wraps each `dispatch_dedup` call so the file stays sync —
this repo doesn't have pytest-asyncio. Live-Milvus integration is
exercised in `tests/test_search_dedup_integration.py`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_API_DIR = REPO_ROOT / "search-api"
sys.path.insert(0, str(SEARCH_API_DIR))

from models import (  # noqa: E402
    BlockedEClassGroup,
    BlockedEClassVendorsFilter,
    EClassVersion,
    PriceFilter,
    SearchMode,
    SearchRequest,
    SelectedArticleSources,
)
from routing import DispatchResult, dispatch_dedup  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# MockClient — captures every call and replays scripted results
# ──────────────────────────────────────────────────────────────────────

class MockClient:
    """Minimal stub of pymilvus's MilvusClient. Records every call into
    `self.calls` and dispatches results through three layers:

      `query_by_filter[substring]` — returns the matching list when the
        call's `filter` kwarg contains `substring`. Useful to script the
        F4 hitCount query (`article_hash != ""`) and the article-browse
        query (`manufacturerName in [...]`) independently of the offer
        resolve / probe.

      `query_by_collection[name]` — per-collection FIFO queue. Used to
        keep articles vs offers calls separate when both fire under
        `asyncio.gather`.

      `query_results` — global FIFO fallback for tests that don't need
        the disambiguation.

    Why three layers: routing.py runs offer resolve, article meta, and
    hitCount queries concurrently via `asyncio.gather`, and Python's
    threadpool can interleave `pop(0)` calls non-deterministically. The
    pattern + per-collection routing keeps tests stable."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.search_results: list[list] = []
        self.query_results: list[list[dict]] = []
        self.query_by_filter: dict[str, list[dict]] = {}
        self.query_by_collection: dict[str, list[list[dict]]] = {}

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        if not self.search_results:
            return [[]]
        return [self.search_results.pop(0)]

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        flt = kwargs.get("filter", "")
        for pattern, result in self.query_by_filter.items():
            if pattern in flt:
                return result
        col = kwargs.get("collection_name", "")
        if col in self.query_by_collection and self.query_by_collection[col]:
            return self.query_by_collection[col].pop(0)
        if not self.query_results:
            return []
        return self.query_results.pop(0)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _req(**overrides) -> SearchRequest:
    base = {
        "search_mode": SearchMode.HITS_ONLY,
        "selected_article_sources": SelectedArticleSources(),
        "currency": "EUR",
    }
    base.update(overrides)
    return SearchRequest(**base)


async def _embed_dummy(text: str) -> list[float]:
    return [0.0] * 128


def _dispatch(req: SearchRequest, **kwargs) -> DispatchResult:
    """Sync wrapper around the async dispatcher."""
    return asyncio.run(dispatch_dedup(req, **kwargs))


def _ann_hit(hash_: str, score: float) -> dict:
    """Mimic Milvus's per-hit shape: `{'distance': ..., 'entity': {'article_hash': ...}}`."""
    return {"distance": score, "entity": {"article_hash": hash_}}


def _offer(id_: str, hash_: str, *, vendor_id: str = "v1", prices: list | None = None) -> dict:
    return {"id": id_, "article_hash": hash_, "vendor_id": vendor_id, "prices": prices or []}


# ──────────────────────────────────────────────────────────────────────
# Path A — no per-offer filter
# ──────────────────────────────────────────────────────────────────────

def test_path_a_no_query_no_filter_returns_empty() -> None:
    """No query, no filters → no defensible browse → empty hits."""
    c = MockClient()
    req = _req()
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.hits == []
    assert res.recall_clipped is False
    assert res.debug["path"] == "A"


def test_path_a_with_query_runs_dense_and_bm25_then_resolves_offers() -> None:
    c = MockClient()
    c.search_results = [
        [_ann_hit("h1", 0.9), _ann_hit("h2", 0.5)],   # dense
        [_ann_hit("h1", 0.8), _ann_hit("h2", 0.4)],   # bm25
    ]
    # Offer resolve from offers collection.
    c.query_by_collection["offers"] = [
        [_offer("o1a", "h1"), _offer("o1b", "h1"), _offer("o2", "h2")],
    ]
    # hitCount query against articles ('article_hash != ""' sentinel
    # since article_expr is None).
    c.query_by_collection["articles"] = [
        [{"article_hash": "h1"}, {"article_hash": "h2"}],
    ]

    req = _req(query="bohrmaschine")
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )

    assert res.debug["path"] == "A"
    assert res.recall_clipped is False
    assert res.hit_count == 2
    assert res.hit_count_clipped is False
    assert sorted(h.id for h in res.hits) == ["o1a", "o2"]
    # 2 ANN searches + 1 offer query + 1 articles count query.
    methods = [m for m, _ in c.calls]
    assert methods.count("search") == 2
    assert methods.count("query") == 2


def test_path_a_with_article_filter_pushes_down() -> None:
    """`manufacturers_filter` is article-side — must reach the article
    ANN's `filter` kwarg."""
    c = MockClient()
    c.search_results = [
        [_ann_hit("h1", 0.9)],
        [_ann_hit("h1", 0.8)],
    ]
    c.query_by_collection["offers"] = [[_offer("o1", "h1")]]
    c.query_by_collection["articles"] = [[{"article_hash": "h1"}]]  # count query

    req = _req(query="x", manufacturers_filter=["Bosch"])
    _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )

    search_calls = [kw for m, kw in c.calls if m == "search"]
    assert all('manufacturerName in ["Bosch"]' in kw["filter"] for kw in search_calls)


def test_path_a_browse_no_query_with_article_filter() -> None:
    """No query but an article-only filter → query() against articles
    (browse mode), then resolve offers."""
    c = MockClient()
    c.query_by_collection["articles"] = [
        [{"article_hash": "h1"}, {"article_hash": "h2"}],   # browse
        [{"article_hash": "h1"}, {"article_hash": "h2"}],   # count
    ]
    c.query_by_collection["offers"] = [
        [_offer("o1", "h1"), _offer("o2", "h2")],
    ]
    req = _req(manufacturers_filter=["Bosch"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.debug["path"] == "A"
    assert res.hit_count == 2
    assert sorted(h.id for h in res.hits) == ["o1", "o2"]


# ──────────────────────────────────────────────────────────────────────
# Path B — per-offer filter active
# ──────────────────────────────────────────────────────────────────────

def test_path_b_with_query_runs_probe_then_constrained_article_rank() -> None:
    c = MockClient()
    # Probe: two offers, two distinct hashes.
    c.query_results = [
        [_offer("o1", "h1"), _offer("o2", "h2")],
    ]
    # Article ranking: dense + BM25.
    c.search_results = [
        [_ann_hit("h1", 0.9), _ann_hit("h2", 0.5)],
        [_ann_hit("h1", 0.8), _ann_hit("h2", 0.4)],
    ]
    req = _req(query="bohr", vendor_ids_filter=["v1"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )

    assert res.debug["path"] == "B"
    assert res.recall_clipped is False
    assert sorted(h.id for h in res.hits) == ["o1", "o2"]

    # Probe call: must use the offer expr.
    probe = c.calls[0]
    assert probe[0] == "query"
    assert probe[1]["collection_name"] == "offers"
    assert probe[1]["filter"] == 'vendor_id in ["v1"]'

    # Article-rank ANN: must include `article_hash IN`.
    ann = c.calls[1]
    assert ann[0] == "search"
    assert "article_hash in" in ann[1]["filter"]


def test_path_b_no_query_returns_probe_hashes_in_order() -> None:
    """No query in Path B: present probe hashes deterministically (no
    article ANN). Each hash → representative offer."""
    c = MockClient()
    c.query_results = [
        [_offer("o2", "h2"), _offer("o1", "h1")],
    ]
    req = _req(vendor_ids_filter=["v1"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.debug["path"] == "B"
    # Hashes presented in sorted order (deterministic), then materialise.
    assert sorted(h.id for h in res.hits) == ["o1", "o2"]


def test_path_b_no_query_with_article_filter_browses_articles() -> None:
    """No query + per-offer filter + article-side filter → probe offers,
    then `query(articles, hash IN probe AND article_expr)` to filter."""
    c = MockClient()
    c.query_by_collection["offers"] = [
        [_offer("o1", "h1"), _offer("o2", "h2")],   # probe
    ]
    c.query_by_collection["articles"] = [
        [{"article_hash": "h1"}],                    # article browse
        [{"article_hash": "h1"}],                    # count
    ]
    req = _req(vendor_ids_filter=["v1"], manufacturers_filter=["Bosch"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert [h.id for h in res.hits] == ["o1"]
    assert res.hit_count == 1


def test_path_b_overflow_falls_back_to_path_a_with_recall_clipped() -> None:
    c = MockClient()
    # Probe returns N+1 hashes → overflow.
    overflow = [_offer(f"o{i}", f"h{i}") for i in range(5)]
    c.query_by_collection["offers"] = [
        overflow,                                       # probe (overflow with limit=2)
        [_offer("o0", "h0")],                            # path A resolve
    ]
    c.search_results = [
        [_ann_hit("h0", 0.9)],
        [_ann_hit("h0", 0.8)],
    ]
    # No articles count needed under fallback (skip_count=True).

    req = _req(query="x", vendor_ids_filter=["v1"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
        path_b_hash_limit=2,
    )

    assert res.recall_clipped is True
    assert res.hit_count == 2  # path_b_hash_limit lower bound
    assert res.hit_count_clipped is True
    assert res.debug["path"] == "A_fallback"
    assert res.debug["probe_overflowed"] is True
    assert res.debug["distinct_hashes"] == 5
    assert [h.id for h in res.hits] == ["o0"]


# ──────────────────────────────────────────────────────────────────────
# articleIdsFilter — offer-side `id IN [...]`
# ──────────────────────────────────────────────────────────────────────

def test_article_ids_filter_routed_to_offer_side() -> None:
    """Even with no other offer filter, articleIdsFilter alone triggers
    Path B because it lands on the offer side."""
    c = MockClient()
    c.query_results = [
        [_offer("a:1", "h1"), _offer("b:2", "h2")],
    ]
    req = _req(article_ids_filter=["a:1", "b:2"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.debug["path"] == "B"
    probe = c.calls[0]
    assert 'id in ["a:1", "b:2"]' in probe[1]["filter"]
    assert sorted(h.id for h in res.hits) == ["a:1", "b:2"]


# ──────────────────────────────────────────────────────────────────────
# Per-vendor blocked_eclass_vendors — Python post-pass
# ──────────────────────────────────────────────────────────────────────

def test_per_vendor_blocked_eclass_drops_correlated_hits() -> None:
    """Vendor v-block is restricted; article h1 has eclass 1000 in the
    blocked set → offer (v-block, h1) drops, but offer (v-other, h1)
    passes (vendor not in restricted list)."""
    c = MockClient()
    c.search_results = [
        [_ann_hit("h1", 0.9)],
        [_ann_hit("h1", 0.8)],
    ]
    c.query_by_collection["offers"] = [
        [_offer("blocked", "h1", vendor_id="v-block"),
         _offer("ok", "h1", vendor_id="v-other")],
    ]
    # Article meta (eclass) and count are both against `articles`.
    # Disambiguate via filter: the meta query has `article_hash in [...]`
    # while the count query uses `article_hash != ""`.
    c.query_by_filter = {
        'article_hash != ""': [{"article_hash": "h1"}],   # count
    }
    c.query_by_collection["articles"] = [
        [{"article_hash": "h1", "eclass5_code": [1000]}],   # meta
    ]
    req = _req(query="x", blocked_eclass_vendors_filters=[
        BlockedEClassVendorsFilter(
            vendorIds=["v-block"],
            eClassVersion=EClassVersion.ECLASS_5_1,
            blockedEClassGroups=[BlockedEClassGroup(eClassGroupCode=1000, value=True)],
        ),
    ])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert [h.id for h in res.hits] == ["ok"]


# ──────────────────────────────────────────────────────────────────────
# Representative-offer + price post-pass
# ──────────────────────────────────────────────────────────────────────

def test_representative_picks_alphabetically_lowest_id() -> None:
    """No price filter → first sorted offer wins."""
    c = MockClient()
    c.search_results = [[_ann_hit("h1", 0.9)], [_ann_hit("h1", 0.8)]]
    c.query_by_collection["offers"] = [[
        _offer("zzz", "h1"),
        _offer("aaa", "h1"),
        _offer("mmm", "h1"),
    ]]
    c.query_by_collection["articles"] = [[{"article_hash": "h1"}]]   # count
    req = _req(query="x")
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert [h.id for h in res.hits] == ["aaa"]


def test_price_filter_skips_failing_offer_takes_next() -> None:
    """Lowest-id offer fails price filter → next one wins. Article only
    drops if every candidate fails."""
    c = MockClient()
    c.search_results = [[_ann_hit("h1", 0.9)], [_ann_hit("h1", 0.8)]]
    c.query_by_collection["offers"] = [[
        _offer("aaa", "h1", prices=[
            {"price": 10.0, "currency": "EUR", "priority": 1, "sourcePriceListId": "p1"},
        ]),
        _offer("bbb", "h1", prices=[
            {"price": 100.0, "currency": "EUR", "priority": 1, "sourcePriceListId": "p1"},
        ]),
    ]]
    c.query_by_collection["articles"] = [[{"article_hash": "h1"}]]   # count
    req = _req(
        query="x",
        price_filter=PriceFilter(min=5000, max=20000, currencyCode="EUR"),
        selected_article_sources=SelectedArticleSources(sourcePriceListIds=["p1"]),
    )
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert [h.id for h in res.hits] == ["bbb"]


def test_price_filter_drops_article_when_all_offers_fail() -> None:
    c = MockClient()
    c.search_results = [[_ann_hit("h1", 0.9)], [_ann_hit("h1", 0.8)]]
    c.query_by_collection["offers"] = [[
        _offer("aaa", "h1", prices=[
            {"price": 10.0, "currency": "EUR", "priority": 1, "sourcePriceListId": "p1"},
        ]),
    ]]
    c.query_by_collection["articles"] = [[{"article_hash": "h1"}]]   # count
    req = _req(
        query="x",
        price_filter=PriceFilter(min=50000, max=99999, currencyCode="EUR"),
        selected_article_sources=SelectedArticleSources(sourcePriceListIds=["p1"]),
    )
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.hits == []


# ──────────────────────────────────────────────────────────────────────
# Debug envelope sanity
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# F4 — sort, hitCount, relevance-pool bound
# ──────────────────────────────────────────────────────────────────────


def _sort_plan(field: str, direction: str = "asc"):
    """Build a SortPlan from string args without going through models'
    SortClause validator."""
    from sorting import SortPlan, SortField
    from models import SortDirection
    return SortPlan(
        SortField(field) if field == "relevance" else getattr(SortField, field.upper()),
        SortDirection(direction),
    )


def test_sort_articleid_asc_picks_lowest_then_orders_by_id() -> None:
    """Two hashes, each with multiple offers. articleId,asc selects the
    lowest id per hash, then the global order is by representative id."""
    from sorting import SortField, SortPlan
    from models import SortDirection
    c = MockClient()
    c.search_results = [
        [_ann_hit("h1", 0.5), _ann_hit("h2", 0.9)],
        [_ann_hit("h1", 0.4), _ann_hit("h2", 0.8)],
    ]
    c.query_by_collection["offers"] = [[
        _offer("zzz1", "h1"), _offer("aaa1", "h1"),
        _offer("zzz2", "h2"), _offer("bbb2", "h2"),
    ]]
    c.query_by_collection["articles"] = [[{"article_hash": "h1"}, {"article_hash": "h2"}]]
    req = _req(query="x")
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        sort_plan=SortPlan(SortField.ARTICLE_ID, SortDirection.ASC),
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    # Reps are aaa1 + bbb2; final order asc → aaa1, bbb2.
    assert [h.id for h in res.hits] == ["aaa1", "bbb2"]


def test_sort_articleid_desc_picks_highest() -> None:
    from sorting import SortField, SortPlan
    from models import SortDirection
    c = MockClient()
    c.search_results = [
        [_ann_hit("h1", 0.5), _ann_hit("h2", 0.9)],
        [_ann_hit("h1", 0.4), _ann_hit("h2", 0.8)],
    ]
    c.query_by_collection["offers"] = [[
        _offer("zzz1", "h1"), _offer("aaa1", "h1"),
        _offer("zzz2", "h2"), _offer("bbb2", "h2"),
    ]]
    c.query_by_collection["articles"] = [[{"article_hash": "h1"}, {"article_hash": "h2"}]]
    req = _req(query="x")
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        sort_plan=SortPlan(SortField.ARTICLE_ID, SortDirection.DESC),
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    # Reps are zzz1 + zzz2; final order desc → zzz2, zzz1.
    assert [h.id for h in res.hits] == ["zzz2", "zzz1"]


def test_sort_name_fetches_article_meta_and_orders() -> None:
    """sort=name needs article-level name fetched. Two articles with
    distinct names; result ordered by name asc."""
    from sorting import SortField, SortPlan
    from models import SortDirection
    c = MockClient()
    c.search_results = [
        [_ann_hit("h1", 0.5), _ann_hit("h2", 0.9)],
        [_ann_hit("h1", 0.4), _ann_hit("h2", 0.8)],
    ]
    c.query_by_collection["offers"] = [[_offer("o1", "h1"), _offer("o2", "h2")]]
    # Distinguish articles count vs articles meta by filter pattern.
    c.query_by_filter = {
        'article_hash != ""': [{"article_hash": "h1"}, {"article_hash": "h2"}],   # count
    }
    c.query_by_collection["articles"] = [
        [{"article_hash": "h1", "name": "Zebra"},
         {"article_hash": "h2", "name": "Antelope"}],   # meta
    ]
    req = _req(query="x")
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        sort_plan=SortPlan(SortField.NAME, SortDirection.ASC),
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    # Antelope (h2/o2) before Zebra (h1/o1).
    assert [h.id for h in res.hits] == ["o2", "o1"]


def test_sort_price_asc_picks_cheapest_per_hash_then_orders() -> None:
    from sorting import SortField, SortPlan
    from models import SortDirection
    c = MockClient()
    c.search_results = [
        [_ann_hit("h1", 0.5), _ann_hit("h2", 0.9)],
        [_ann_hit("h1", 0.4), _ann_hit("h2", 0.8)],
    ]
    # h1: cheap=10, expensive=100. h2: cheap=50.
    eur = lambda p: [{"price": p, "currency": "EUR", "priority": 1, "sourcePriceListId": "p1"}]
    c.query_by_collection["offers"] = [[
        _offer("h1-cheap", "h1", prices=eur(10.0)),
        _offer("h1-expensive", "h1", prices=eur(100.0)),
        _offer("h2-cheap", "h2", prices=eur(50.0)),
    ]]
    c.query_by_collection["articles"] = [[{"article_hash": "h1"}, {"article_hash": "h2"}]]
    req = _req(
        query="x",
        # No price filter — but sort=price still resolves prices via the
        # request scope: currency=EUR + sourcePriceListIds.
        selected_article_sources=SelectedArticleSources(sourcePriceListIds=["p1"]),
    )
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        sort_plan=SortPlan(SortField.PRICE, SortDirection.ASC),
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    # Reps: h1-cheap (10), h2-cheap (50). Final order asc → h1-cheap, h2-cheap.
    assert [h.id for h in res.hits] == ["h1-cheap", "h2-cheap"]


def test_relevance_pool_cap_truncates_for_non_relevance_with_query() -> None:
    """`relevance_pool_max=5` caps the post-RRF pool: even though dense
    + BM25 return 10 each, only 5 rank through to the sort+materialise
    stage. Visible by the offer count returned (≤ 5)."""
    from sorting import SortField, SortPlan
    from models import SortDirection
    c = MockClient()
    c.search_results = [
        [_ann_hit(f"h{i}", 1.0 - i * 0.1) for i in range(10)],
        [_ann_hit(f"h{i}", 1.0 - i * 0.1) for i in range(10)],
    ]
    offers = [_offer(f"o{i}", f"h{i}") for i in range(10)]
    c.query_by_collection["offers"] = [offers]
    c.query_by_collection["articles"] = [[{"article_hash": f"h{i}"} for i in range(10)]]

    req = _req(query="x")
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        sort_plan=SortPlan(SortField.ARTICLE_ID, SortDirection.ASC),
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
        relevance_pool_max=5,
        relevance_score_floor=0.0,
    )
    # Top 5 by RRF rank are h0..h4 → reps o0..o4. Sort=articleId,asc → o0..o4.
    assert [h.id for h in res.hits] == ["o0", "o1", "o2", "o3", "o4"]


def test_relevance_score_floor_culls_low_scoring_candidates() -> None:
    """RRF score floor drops candidates below `floor × top_score`. With
    a generous pool cap and a permissive RRF distribution, the floor is
    the only active filter."""
    from sorting import SortField, SortPlan
    from models import SortDirection
    c = MockClient()
    # Three hashes appear in both legs (consistent rank 1, 2, 3) — RRF
    # produces well-separated scores. h4..h9 appear in only one leg (rank
    # tail) with much lower fused scores.
    c.search_results = [
        [_ann_hit("h1", 0.9), _ann_hit("h2", 0.8), _ann_hit("h3", 0.7)]
        + [_ann_hit(f"h{i}", 1.0 / (i + 1)) for i in range(4, 10)],
        [_ann_hit("h1", 0.9), _ann_hit("h2", 0.8), _ann_hit("h3", 0.7)],
    ]
    offers = [_offer(f"o{i}", f"h{i}") for i in range(1, 10)]
    c.query_by_collection["offers"] = [offers]
    c.query_by_collection["articles"] = [
        [{"article_hash": f"h{i}"} for i in range(1, 10)]
    ]

    req = _req(query="x")
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        sort_plan=SortPlan(SortField.ARTICLE_ID, SortDirection.ASC),
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
        relevance_pool_max=100,           # generous: cap doesn't bite
        relevance_score_floor=0.5,        # 50% of top_score is a steep cut
    )
    # h1..h3 have RRF ≈ 2/61. h4..h9 have RRF ≈ 1/61, well below 0.5×top.
    # Only h1..h3 should survive the floor.
    surviving_ids = {h.id for h in res.hits}
    assert surviving_ids == {"o1", "o2", "o3"}
    assert (res.debug["relevance_bound_dropped"] or 0) >= 6


def test_relevance_pool_bound_skipped_for_browse_no_query() -> None:
    """Browse traffic (no query) doesn't apply the relevance-pool bound,
    even with non-relevance sort."""
    from sorting import SortField, SortPlan
    from models import SortDirection
    c = MockClient()
    # Browse + count (filter sentinel disambiguates).
    c.query_by_filter = {
        'article_hash != ""': [{"article_hash": f"h{i}"} for i in range(10)],   # count
    }
    c.query_by_collection["articles"] = [
        [{"article_hash": f"h{i}"} for i in range(10)],   # browse
    ]
    c.query_by_collection["offers"] = [[_offer(f"o{i}", f"h{i}") for i in range(10)]]
    req = _req(manufacturers_filter=["Bosch"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        sort_plan=SortPlan(SortField.ARTICLE_ID, SortDirection.ASC),
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
        relevance_pool_max=5,
    )
    # Bound did NOT apply — all 10 offers come back.
    assert len(res.hits) == 10
    assert res.debug["relevance_bound_dropped"] is None


def test_relevance_pool_bound_skipped_for_relevance_sort() -> None:
    """Default relevance sort + queryString → bound NOT applied."""
    from sorting import SortField, SortPlan
    from models import SortDirection
    c = MockClient()
    c.search_results = [
        [_ann_hit(f"h{i}", 1.0 - i * 0.1) for i in range(10)],
        [_ann_hit(f"h{i}", 1.0 - i * 0.1) for i in range(10)],
    ]
    c.query_by_collection["offers"] = [[_offer(f"o{i}", f"h{i}") for i in range(10)]]
    c.query_by_collection["articles"] = [[{"article_hash": f"h{i}"} for i in range(10)]]
    req = _req(query="x")
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        # No sort_plan → defaults to RELEVANCE/DESC inside dispatch_dedup.
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
        relevance_pool_max=5,
    )
    assert len(res.hits) == 10
    assert res.debug["relevance_bound_dropped"] is None


def test_pagination_returns_page_2_slice() -> None:
    """page=2 pageSize=3 → returns positions 4-6 of the sorted list."""
    from sorting import SortField, SortPlan
    from models import SortDirection
    c = MockClient()
    c.search_results = [
        [_ann_hit(f"h{i}", 1.0 - i * 0.1) for i in range(10)],
        [_ann_hit(f"h{i}", 1.0 - i * 0.1) for i in range(10)],
    ]
    c.query_by_collection["offers"] = [[_offer(f"o{i}", f"h{i}") for i in range(10)]]
    c.query_by_collection["articles"] = [[{"article_hash": f"h{i}"} for i in range(10)]]
    req = _req(query="x")
    res = _dispatch(
        req, page=2, page_size=3, overfetch_n=10,
        sort_plan=SortPlan(SortField.ARTICLE_ID, SortDirection.ASC),
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    # Sorted asc: o0, o1, o2, o3, o4, ... page 2 size 3 → o3, o4, o5.
    assert [h.id for h in res.hits] == ["o3", "o4", "o5"]


def test_hit_count_capped_when_count_overflows() -> None:
    """`hitcount_cap=2`: count returns 5 rows → hit_count=2, clipped=True."""
    c = MockClient()
    c.search_results = [
        [_ann_hit("h1", 0.9)],
        [_ann_hit("h1", 0.8)],
    ]
    c.query_by_collection["offers"] = [[_offer("o1", "h1")]]
    # Count returns more than the cap.
    c.query_by_collection["articles"] = [
        [{"article_hash": f"h{i}"} for i in range(5)],
    ]
    req = _req(query="x")
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
        hitcount_cap=2,
    )
    assert res.hit_count == 2
    assert res.hit_count_clipped is True


def test_hit_count_path_b_no_article_expr_uses_distinct_hash_count() -> None:
    """Path B with offer_expr only (no article_expr) → hit_count is
    just the distinct-hash count from the probe; no extra count query."""
    c = MockClient()
    c.query_by_collection["offers"] = [[
        _offer("o1", "h1"), _offer("o2", "h2"), _offer("o3", "h3"),
    ]]
    req = _req(vendor_ids_filter=["v1"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.hit_count == 3
    assert res.hit_count_clipped is False
    # Only the probe query against offers — no count query.
    queries = [kw for m, kw in c.calls if m == "query"]
    assert len(queries) == 1
    assert queries[0]["collection_name"] == "offers"


def test_debug_envelope_carries_path_and_exprs() -> None:
    c = MockClient()
    c.query_results = [[]]
    req = _req(vendor_ids_filter=["v1"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.debug["path"] == "B"
    assert res.debug["offer_expr"] == 'vendor_id in ["v1"]'
    assert res.debug["article_expr"] is None


# ──────────────────────────────────────────────────────────────────────
# F5 — summaries
# ──────────────────────────────────────────────────────────────────────

def _req_summaries(*kinds, **overrides) -> SearchRequest:
    from models import SummaryKind
    base = {
        "search_mode": SearchMode.BOTH,
        "selected_article_sources": SelectedArticleSources(),
        "currency": "EUR",
        "summaries": list(kinds),
    }
    base.update(overrides)
    return SearchRequest(**base)


def test_hits_only_mode_omits_summaries() -> None:
    """HITS_ONLY → result.summaries is None even when summaries
    requested. (The mode flag wins over the kinds list.)"""
    from models import SummaryKind
    c = MockClient()
    c.search_results = [[_ann_hit("h1", 0.9)], [_ann_hit("h1", 0.8)]]
    c.query_by_collection["offers"] = [[_offer("o1", "h1")]]
    c.query_by_collection["articles"] = [[{"article_hash": "h1"}]]
    req = _req(query="x", search_mode=SearchMode.HITS_ONLY,
               summaries=[SummaryKind.MANUFACTURERS])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.summaries is None


def test_both_mode_with_no_kinds_skips_summaries() -> None:
    """BOTH but empty summaries list → still no summary fetch (nothing
    requested)."""
    c = MockClient()
    c.search_results = [[_ann_hit("h1", 0.9)], [_ann_hit("h1", 0.8)]]
    c.query_by_collection["offers"] = [[_offer("o1", "h1")]]
    c.query_by_collection["articles"] = [[{"article_hash": "h1"}]]
    req = _req(query="x", search_mode=SearchMode.BOTH, summaries=[])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.summaries is None


def test_both_mode_path_a_fetches_articles_for_summaries() -> None:
    from models import SummaryKind
    c = MockClient()
    c.search_results = [[_ann_hit("h1", 0.9)], [_ann_hit("h1", 0.8)]]
    c.query_by_collection["offers"] = [[_offer("o1", "h1")]]
    # Disambiguate by filter pattern: count uses sentinel, summaries
    # query uses sentinel too — same pattern. Use per-collection FIFO
    # ordered as: count first, then summary fetch.
    c.query_by_collection["articles"] = [
        [{"article_hash": "h1"}],                                  # count
        [{"article_hash": "h1", "manufacturerName": "Bosch"}],     # summaries
    ]
    req = _req_summaries(SummaryKind.MANUFACTURERS, query="x")
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.summaries is not None
    assert len(res.summaries.manufacturer_summaries) == 1
    assert res.summaries.manufacturer_summaries[0].name == "Bosch"


def test_both_mode_path_b_fetches_offers_for_vendor_summary() -> None:
    """Path B with VENDORS summary refetches offers via offer_expr (the
    summary fetch may need fields the probe didn't request)."""
    from models import SummaryKind
    c = MockClient()
    c.query_by_collection["offers"] = [
        [_offer("o1", "h1", vendor_id="v1"), _offer("o2", "h2", vendor_id="v1")],   # probe
        [_offer("o1", "h1", vendor_id="v1"), _offer("o2", "h2", vendor_id="v1")],   # summary refetch
    ]
    req = _req_summaries(SummaryKind.VENDORS, vendor_ids_filter=["v1"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.summaries is not None
    by_v = {s.vendor_id: s.count for s in res.summaries.vendor_summaries}
    # 2 distinct articles for v1.
    assert by_v == {"v1": 2}


def test_summaries_only_mode_skips_rank_and_returns_empty_hits() -> None:
    """SUMMARIES_ONLY: no ANN/BM25, no representative pick, no sort/page.
    Just count + summary fetch. Empty hits, populated summaries."""
    from models import SummaryKind
    c = MockClient()
    # No search calls expected. Count + summary fetch only.
    c.query_by_collection["articles"] = [
        [{"article_hash": "h1"}, {"article_hash": "h2"}],   # count
        [
            {"article_hash": "h1", "manufacturerName": "Bosch"},
            {"article_hash": "h2", "manufacturerName": "Makita"},
        ],   # summary fetch
    ]
    req = _req_summaries(SummaryKind.MANUFACTURERS,
                         query="x", search_mode=SearchMode.SUMMARIES_ONLY,
                         manufacturers_filter=["Bosch", "Makita"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.hits == []
    assert res.hit_count == 2
    assert res.summaries is not None
    assert {s.name for s in res.summaries.manufacturer_summaries} == {"Bosch", "Makita"}
    # No ANN/BM25 ran.
    methods = [m for m, _ in c.calls]
    assert methods.count("search") == 0


def test_summaries_only_path_b_uses_probe_for_distinct_hashes() -> None:
    """SUMMARIES_ONLY in Path B: probe → distinct hashes → summaries."""
    from models import SummaryKind
    c = MockClient()
    c.query_by_collection["offers"] = [
        [_offer("o1", "h1", vendor_id="v1"), _offer("o2", "h2", vendor_id="v1")],   # probe
        [_offer("o1", "h1", vendor_id="v1"), _offer("o2", "h2", vendor_id="v1")],   # summary fetch
    ]
    req = _req_summaries(SummaryKind.VENDORS,
                         search_mode=SearchMode.SUMMARIES_ONLY,
                         vendor_ids_filter=["v1"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.hits == []
    # hit_count = distinct hashes from probe.
    assert res.hit_count == 2
    assert res.summaries is not None
    by_v = {s.vendor_id: s.count for s in res.summaries.vendor_summaries}
    assert by_v == {"v1": 2}


def test_summaries_only_probe_overflow_marks_clipped_and_skips_summary_compute() -> None:
    """Probe overflow in SUMMARIES_ONLY: hit count is the lower bound,
    recall_clipped + hit_count_clipped both true, summaries not
    computed (the truncated set would mislead aggregation counts)."""
    from models import SummaryKind
    c = MockClient()
    overflow = [_offer(f"o{i}", f"h{i}") for i in range(5)]
    c.query_by_collection["offers"] = [overflow]
    req = _req_summaries(SummaryKind.VENDORS,
                         search_mode=SearchMode.SUMMARIES_ONLY,
                         vendor_ids_filter=["v1"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
        path_b_hash_limit=2,
    )
    assert res.hits == []
    assert res.recall_clipped is True
    assert res.hit_count == 2
    assert res.hit_count_clipped is True
    assert res.summaries is None  # skipped to avoid misleading counts


def test_field_set_planner_avoids_unneeded_offer_fetch() -> None:
    """Article-only kinds (MANUFACTURERS) must not trigger an offers
    summary refetch in Path A. Verify by counting offers queries."""
    from models import SummaryKind
    c = MockClient()
    c.search_results = [[_ann_hit("h1", 0.9)], [_ann_hit("h1", 0.8)]]
    c.query_by_collection["offers"] = [[_offer("o1", "h1")]]   # only resolve
    c.query_by_collection["articles"] = [
        [{"article_hash": "h1"}],
        [{"article_hash": "h1", "manufacturerName": "Bosch"}],
    ]
    req = _req_summaries(SummaryKind.MANUFACTURERS, query="x")
    _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    offer_queries = [kw for m, kw in c.calls
                     if m == "query" and kw.get("collection_name") == "offers"]
    assert len(offer_queries) == 1   # only the resolve, no summary refetch
