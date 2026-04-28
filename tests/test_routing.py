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
    """Minimal stub of pymilvus's MilvusClient. Each call records the
    invoked method + kwargs into `self.calls`; the test pre-populates
    `self.search_results` / `self.query_results` with the rows to
    return per-call (popped FIFO)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.search_results: list[list] = []
        self.query_results: list[list[dict]] = []

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        if not self.search_results:
            return [[]]
        return [self.search_results.pop(0)]

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
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
    # Article ranking: dense + BM25 both return h1 (high) and h2 (low).
    c.search_results = [
        [_ann_hit("h1", 0.9), _ann_hit("h2", 0.5)],   # dense
        [_ann_hit("h1", 0.8), _ann_hit("h2", 0.4)],   # bm25
    ]
    # Offer resolve: two offers per hash.
    c.query_results = [
        [_offer("o1a", "h1"), _offer("o1b", "h1"), _offer("o2", "h2")],
    ]

    req = _req(query="bohrmaschine")
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )

    assert res.debug["path"] == "A"
    assert res.recall_clipped is False
    # Two articles materialised; representative is alphabetically lowest id.
    assert sorted(h.id for h in res.hits) == ["o1a", "o2"]
    # Dense + BM25 + one offer query.
    methods = [m for m, _ in c.calls]
    assert methods.count("search") == 2
    assert methods.count("query") == 1


def test_path_a_with_article_filter_pushes_down() -> None:
    """`manufacturers_filter` is article-side — must reach the article
    ANN's `filter` kwarg."""
    c = MockClient()
    c.search_results = [
        [_ann_hit("h1", 0.9)],
        [_ann_hit("h1", 0.8)],
    ]
    c.query_results = [[_offer("o1", "h1")]]

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
    # Browse query → article rows.
    # Resolve query → offers for those hashes.
    c.query_results = [
        [{"article_hash": "h1"}, {"article_hash": "h2"}],
        [_offer("o1", "h1"), _offer("o2", "h2")],
    ]
    req = _req(manufacturers_filter=["Bosch"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    assert res.debug["path"] == "A"
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
    # Probe + article browse.
    c.query_results = [
        [_offer("o1", "h1"), _offer("o2", "h2")],          # probe
        [{"article_hash": "h1"}],                            # article browse — only h1 passes mfr filter
    ]
    req = _req(vendor_ids_filter=["v1"], manufacturers_filter=["Bosch"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
    )
    # Only h1 survives the article filter; h2 dropped.
    assert [h.id for h in res.hits] == ["o1"]


def test_path_b_overflow_falls_back_to_path_a_with_recall_clipped() -> None:
    c = MockClient()
    # Probe returns N+1 hashes → overflow.
    overflow = [_offer(f"o{i}", f"h{i}") for i in range(5)]
    c.query_results = [
        overflow,                                            # probe (overflow with limit=2)
    ]
    # Path A's ANN + BM25 (called after fallback).
    c.search_results = [
        [_ann_hit("h0", 0.9)],
        [_ann_hit("h0", 0.8)],
    ]
    # Path A's offer resolve.
    c.query_results.append([_offer("o0", "h0")])

    req = _req(query="x", vendor_ids_filter=["v1"])
    res = _dispatch(
        req, page_size=10, overfetch_n=10,
        client=c, embed=_embed_dummy,
        articles_collection="articles", offers_collection="offers",
        path_b_hash_limit=2,
    )

    assert res.recall_clipped is True
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
    # Path A: per-vendor blocked_eclass is *not* split-emitted (no
    # offer_expr). So this is Path A: rank + offers resolve + article
    # meta fetch.
    c.search_results = [
        [_ann_hit("h1", 0.9)],
        [_ann_hit("h1", 0.8)],
    ]
    c.query_results = [
        [_offer("blocked", "h1", vendor_id="v-block"),
         _offer("ok", "h1", vendor_id="v-other")],   # offer resolve
        [{"article_hash": "h1", "eclass5_code": [1000]}],   # article meta
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
    # Representative pick: alphabetically lowest among non-blocked → "ok".
    # ("blocked" comes before "ok" alphabetically but is dropped first.)
    assert [h.id for h in res.hits] == ["ok"]


# ──────────────────────────────────────────────────────────────────────
# Representative-offer + price post-pass
# ──────────────────────────────────────────────────────────────────────

def test_representative_picks_alphabetically_lowest_id() -> None:
    """No price filter → first sorted offer wins."""
    c = MockClient()
    c.search_results = [[_ann_hit("h1", 0.9)], [_ann_hit("h1", 0.8)]]
    c.query_results = [[
        _offer("zzz", "h1"),
        _offer("aaa", "h1"),
        _offer("mmm", "h1"),
    ]]
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
    c.query_results = [[
        _offer("aaa", "h1", prices=[
            {"price": 10.0, "currency": "EUR", "priority": 1, "sourcePriceListId": "p1"},
        ]),  # 1000 minor < 5000 minor → fails
        _offer("bbb", "h1", prices=[
            {"price": 100.0, "currency": "EUR", "priority": 1, "sourcePriceListId": "p1"},
        ]),  # 10000 minor > 5000 minor → passes
    ]]
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
    c.query_results = [[
        _offer("aaa", "h1", prices=[
            {"price": 10.0, "currency": "EUR", "priority": 1, "sourcePriceListId": "p1"},
        ]),
    ]]
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
