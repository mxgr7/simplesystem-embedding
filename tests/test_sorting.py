"""F4 — pure-function tests for `search-api/sorting.py`.

Three concern groups:

  * `parse_plan` — first-key-only policy, default = relevance/desc.
  * `bound_relevance_pool` — cap + relative floor.
  * `pick_representative` + `sort_items` — sort-aware representative
    selection and the universal articleId-asc tiebreak.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_API_DIR = REPO_ROOT / "search-api"
sys.path.insert(0, str(SEARCH_API_DIR))

from models import SortClause, SortDirection  # noqa: E402
from sorting import (  # noqa: E402
    SortField,
    SortPlan,
    _Materialised,
    bound_relevance_pool,
    parse_plan,
    pick_representative,
    sort_items,
)


# ──────────────────────────────────────────────────────────────────────
# parse_plan
# ──────────────────────────────────────────────────────────────────────

def test_parse_plan_empty_yields_relevance_desc() -> None:
    plan = parse_plan([])
    assert plan.field is SortField.RELEVANCE
    assert plan.direction is SortDirection.DESC
    assert plan.is_relevance


def test_parse_plan_single_key() -> None:
    plan = parse_plan([SortClause(field="price", direction=SortDirection.ASC)])
    assert plan.field is SortField.PRICE
    assert plan.direction is SortDirection.ASC
    assert not plan.is_relevance


def test_parse_plan_multi_key_uses_only_first() -> None:
    """Multi-key sort: F4 deviation — only the first key is honoured."""
    plan = parse_plan([
        SortClause(field="name", direction=SortDirection.DESC),
        SortClause(field="price", direction=SortDirection.ASC),
    ])
    assert plan.field is SortField.NAME
    assert plan.direction is SortDirection.DESC


def test_parse_plan_articleId_alias() -> None:
    plan = parse_plan([SortClause(field="articleId", direction=SortDirection.ASC)])
    assert plan.field is SortField.ARTICLE_ID


# ──────────────────────────────────────────────────────────────────────
# bound_relevance_pool
# ──────────────────────────────────────────────────────────────────────

def test_bound_pool_caps_at_pool_max() -> None:
    ranked = [(f"h{i}", 1.0 - i * 0.01) for i in range(50)]
    capped = bound_relevance_pool(ranked, pool_max=10, score_floor=0.0)
    assert len(capped) == 10
    assert capped == ranked[:10]


def test_bound_pool_floor_drops_low_scores() -> None:
    """Top score 1.0, floor 0.5 → drop everything < 0.5."""
    ranked = [("h1", 1.0), ("h2", 0.8), ("h3", 0.5), ("h4", 0.4), ("h5", 0.1)]
    out = bound_relevance_pool(ranked, pool_max=100, score_floor=0.5)
    assert [h for h, _ in out] == ["h1", "h2", "h3"]


def test_bound_pool_floor_zero_keeps_all_within_cap() -> None:
    ranked = [("h1", 1.0), ("h2", 0.001)]
    out = bound_relevance_pool(ranked, pool_max=100, score_floor=0.0)
    assert out == ranked


def test_bound_pool_empty_returns_empty() -> None:
    assert bound_relevance_pool([], pool_max=10, score_floor=0.5) == []


def test_bound_pool_negative_top_skips_floor() -> None:
    """Negative top score (corner case) → can't apply a meaningful
    proportional floor. Return capped only."""
    ranked = [("h1", -0.1), ("h2", -0.5)]
    out = bound_relevance_pool(ranked, pool_max=100, score_floor=0.5)
    assert out == ranked


# ──────────────────────────────────────────────────────────────────────
# pick_representative
# ──────────────────────────────────────────────────────────────────────

def _o(id_: str, **extras) -> dict:
    return {"id": id_, **extras}


def _no_filter_resolver(_o: dict) -> Decimal | None:
    return None  # no price filter; resolver returns informational price


def _passthrough_resolver(prices_by_id: dict[str, Decimal | None]):
    """Resolver that returns the pre-computed price for a given offer
    id. Used to script price post-pass behaviour without going through
    the full prices module."""
    def f(o: dict) -> Decimal | None:
        return prices_by_id.get(str(o["id"]))
    return f


def test_pick_repr_relevance_picks_lowest_id() -> None:
    plan = SortPlan(SortField.RELEVANCE, SortDirection.DESC)
    chosen = pick_representative(
        [_o("zzz"), _o("aaa"), _o("mmm")], plan=plan,
        price_filter_active=False, price_resolver=_no_filter_resolver,
    )
    assert chosen is not None
    assert chosen[0]["id"] == "aaa"


def test_pick_repr_articleid_asc_picks_lowest() -> None:
    plan = SortPlan(SortField.ARTICLE_ID, SortDirection.ASC)
    chosen = pick_representative(
        [_o("zzz"), _o("aaa"), _o("mmm")], plan=plan,
        price_filter_active=False, price_resolver=_no_filter_resolver,
    )
    assert chosen[0]["id"] == "aaa"


def test_pick_repr_articleid_desc_picks_highest() -> None:
    plan = SortPlan(SortField.ARTICLE_ID, SortDirection.DESC)
    chosen = pick_representative(
        [_o("zzz"), _o("aaa"), _o("mmm")], plan=plan,
        price_filter_active=False, price_resolver=_no_filter_resolver,
    )
    assert chosen[0]["id"] == "zzz"


def test_pick_repr_price_asc_picks_cheapest() -> None:
    plan = SortPlan(SortField.PRICE, SortDirection.ASC)
    resolver = _passthrough_resolver({
        "aaa": Decimal("100"), "bbb": Decimal("50"), "ccc": Decimal("200"),
    })
    chosen = pick_representative(
        [_o("aaa"), _o("bbb"), _o("ccc")], plan=plan,
        price_filter_active=True, price_resolver=resolver,
    )
    assert chosen[0]["id"] == "bbb"
    assert chosen[1] == Decimal("50")


def test_pick_repr_price_desc_picks_most_expensive() -> None:
    plan = SortPlan(SortField.PRICE, SortDirection.DESC)
    resolver = _passthrough_resolver({
        "aaa": Decimal("100"), "bbb": Decimal("50"), "ccc": Decimal("200"),
    })
    chosen = pick_representative(
        [_o("aaa"), _o("bbb"), _o("ccc")], plan=plan,
        price_filter_active=True, price_resolver=resolver,
    )
    assert chosen[0]["id"] == "ccc"
    assert chosen[1] == Decimal("200")


def test_pick_repr_price_drops_offer_with_no_resolved_price() -> None:
    """sort=price excludes offers whose price doesn't resolve in scope —
    matches legacy ES behaviour (missing prices excluded, not sorted last)."""
    plan = SortPlan(SortField.PRICE, SortDirection.ASC)
    resolver = _passthrough_resolver({
        "aaa": None, "bbb": Decimal("100"), "ccc": None,
    })
    chosen = pick_representative(
        [_o("aaa"), _o("bbb"), _o("ccc")], plan=plan,
        price_filter_active=True, price_resolver=resolver,
    )
    assert chosen[0]["id"] == "bbb"


def test_pick_repr_price_returns_none_when_all_offers_lack_price() -> None:
    plan = SortPlan(SortField.PRICE, SortDirection.ASC)
    resolver = _passthrough_resolver({"aaa": None, "bbb": None})
    chosen = pick_representative(
        [_o("aaa"), _o("bbb")], plan=plan,
        price_filter_active=True, price_resolver=resolver,
    )
    assert chosen is None


def test_pick_repr_price_tiebreaks_on_id_asc() -> None:
    """Equal prices → lowest articleId wins regardless of sort direction."""
    plan = SortPlan(SortField.PRICE, SortDirection.DESC)
    resolver = _passthrough_resolver({
        "zzz": Decimal("100"), "aaa": Decimal("100"),
    })
    chosen = pick_representative(
        [_o("zzz"), _o("aaa")], plan=plan,
        price_filter_active=True, price_resolver=resolver,
    )
    assert chosen[0]["id"] == "aaa"


def test_pick_repr_price_filter_drops_failing_offer_then_picks_next() -> None:
    """price_filter_active=True + non-price sort → resolver returns None
    for failing offers, those drop, next candidate (lowest id) wins."""
    plan = SortPlan(SortField.RELEVANCE, SortDirection.DESC)
    resolver = _passthrough_resolver({
        "aaa": None,             # fails price filter
        "bbb": Decimal("10"),    # passes
        "ccc": Decimal("20"),    # passes
    })
    chosen = pick_representative(
        [_o("aaa"), _o("bbb"), _o("ccc")], plan=plan,
        price_filter_active=True, price_resolver=resolver,
    )
    assert chosen[0]["id"] == "bbb"


# ──────────────────────────────────────────────────────────────────────
# sort_items
# ──────────────────────────────────────────────────────────────────────

def _m(hash_: str, id_: str, *, score: float = 0.0,
       price: Decimal | None = None, name: str | None = None) -> _Materialised:
    return _Materialised(
        article_hash=hash_,
        relevance_score=score,
        representative_offer={"id": id_},
        resolved_price=price,
        article_name=name,
    )


def test_sort_items_relevance_descending_with_id_tiebreak() -> None:
    items = [
        _m("h1", "zzz", score=0.5),
        _m("h2", "aaa", score=0.5),     # tied with above; id asc tiebreak
        _m("h3", "ccc", score=0.9),
    ]
    out = sort_items(items, SortPlan(SortField.RELEVANCE, SortDirection.DESC))
    assert [m.representative_offer["id"] for m in out] == ["ccc", "aaa", "zzz"]


def test_sort_items_articleid_asc() -> None:
    items = [_m("h1", "zzz"), _m("h2", "aaa"), _m("h3", "mmm")]
    out = sort_items(items, SortPlan(SortField.ARTICLE_ID, SortDirection.ASC))
    assert [m.representative_offer["id"] for m in out] == ["aaa", "mmm", "zzz"]


def test_sort_items_articleid_desc() -> None:
    items = [_m("h1", "zzz"), _m("h2", "aaa"), _m("h3", "mmm")]
    out = sort_items(items, SortPlan(SortField.ARTICLE_ID, SortDirection.DESC))
    assert [m.representative_offer["id"] for m in out] == ["zzz", "mmm", "aaa"]


def test_sort_items_name_case_insensitive() -> None:
    items = [_m("h1", "i1", name="Bohrmaschine"),
             _m("h2", "i2", name="akku-bohrer"),
             _m("h3", "i3", name="Akkuschrauber")]
    out = sort_items(items, SortPlan(SortField.NAME, SortDirection.ASC))
    # Lowercase compare: "akku-bohrer" < "akkuschrauber" < "bohrmaschine"
    assert [m.representative_offer["id"] for m in out] == ["i2", "i3", "i1"]


def test_sort_items_name_with_id_tiebreak() -> None:
    items = [_m("h1", "zzz", name="Same"),
             _m("h2", "aaa", name="Same")]
    out = sort_items(items, SortPlan(SortField.NAME, SortDirection.ASC))
    assert [m.representative_offer["id"] for m in out] == ["aaa", "zzz"]


def test_sort_items_price_asc() -> None:
    items = [_m("h1", "i1", price=Decimal("100")),
             _m("h2", "i2", price=Decimal("50")),
             _m("h3", "i3", price=Decimal("200"))]
    out = sort_items(items, SortPlan(SortField.PRICE, SortDirection.ASC))
    assert [str(m.resolved_price) for m in out] == ["50", "100", "200"]


def test_sort_items_price_desc_with_id_tiebreak() -> None:
    items = [_m("h1", "zzz", price=Decimal("100")),
             _m("h2", "aaa", price=Decimal("100")),
             _m("h3", "i3", price=Decimal("50"))]
    out = sort_items(items, SortPlan(SortField.PRICE, SortDirection.DESC))
    # Same price → id asc tiebreak: aaa before zzz
    assert [m.representative_offer["id"] for m in out] == ["aaa", "zzz", "i3"]


def test_sort_items_empty_returns_empty() -> None:
    assert sort_items([], SortPlan(SortField.NAME, SortDirection.ASC)) == []
