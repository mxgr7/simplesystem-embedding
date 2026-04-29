"""F4 — sort + relevance-pool bounding helpers.

Pure-function helpers shared by `routing.py`. Three concerns:

  * `parse_plan` — first-key-only sort policy (multi-key requests apply
    only the first key; secondary ignored, per F4 §"Multi-key sort").
  * `bound_relevance_pool` — RELEVANCE_POOL_MAX cap + RELEVANCE_SCORE_FLOOR
    relative cull. Applied only when sort is non-relevance AND a query
    string is present (per F4 §"Relevance-pool bounding").
  * `pick_representative` + `sort_items` — sort-aware representative-offer
    selection (cheapest/most-expensive per article for sort=price; lowest
    /highest articleId per article for sort=articleId; alphabetical for
    relevance and name) and final article-level sort with the deterministic
    `articleId,asc` tiebreak.

Design notes:

  * The legacy contract supports `name` and `price` sorts (both
    asc/desc), plus `articleId,asc|desc` as a deviation. Multi-key sort
    is documented as a deviation: only the first key is honoured.
  * `name` and `price` sorts always tiebreak on `articleId,asc` so that
    equal names / prices resolve deterministically across re-runs.
  * Sort by `name` is case-insensitive (lowercase compare). Matches
    legacy ES behaviour where `name.keyword` was lowercased at index
    time.
  * `price` is resolved per representative offer via the F3 prices
    module — currency × sourcePriceListIds × priority. Articles whose
    representative has no resolved price drop from sort=price (matches
    legacy: missing prices are excluded from the price-sort pool, not
    placed at the end).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Callable

from models import SortClause, SortDirection


class SortField(str, Enum):
    """Sort dimension for a SearchResponse. RELEVANCE is the default
    (no `sort` param); the rest correspond to the legacy options."""
    RELEVANCE = "relevance"
    ARTICLE_ID = "articleId"
    NAME = "name"
    PRICE = "price"


@dataclass(slots=True, frozen=True)
class SortPlan:
    """First-key-only sort policy derived from `parse_sort_params`."""
    field: SortField
    direction: SortDirection

    @property
    def is_relevance(self) -> bool:
        return self.field is SortField.RELEVANCE

    @property
    def descending(self) -> bool:
        return self.direction is SortDirection.DESC


_FIELD_MAP = {
    "articleId": SortField.ARTICLE_ID,
    "name": SortField.NAME,
    "price": SortField.PRICE,
}


def parse_plan(sort_clauses: list[SortClause]) -> SortPlan:
    """Multi-key sort applies only the first key — F4 §"Multi-key sort"
    deviation. Empty sort → RELEVANCE descending (default)."""
    if not sort_clauses:
        return SortPlan(SortField.RELEVANCE, SortDirection.DESC)
    first = sort_clauses[0]
    field = _FIELD_MAP.get(first.field)
    if field is None:
        # parse_sort_params already validated against SORTABLE_FIELDS,
        # so an unmapped value here means SORTABLE_FIELDS and _FIELD_MAP
        # have drifted — fail loud.
        raise ValueError(f"unmapped sort field {first.field!r}")
    return SortPlan(field, first.direction)


# ──────────────────────────────────────────────────────────────────────
# Relevance-pool bounding
# ──────────────────────────────────────────────────────────────────────

def bound_relevance_pool(
    ranked: list[tuple[str, float]],
    *,
    pool_max: int,
    score_floor: float,
) -> list[tuple[str, float]]:
    """Cap to `pool_max`, then drop any entry whose score is below
    `score_floor × top_score`. Idempotent on already-bounded input.

    Why a *relative* floor: absolute thresholds need per-query tuning
    because RRF scores depend on candidate density. `floor × top_score`
    adapts per-query without manual tuning (F4 §"Relevance-pool bounding").
    """
    if not ranked:
        return ranked
    capped = ranked[:pool_max]
    top = capped[0][1]
    if top <= 0 or score_floor <= 0:
        # Negative/zero top score (e.g. all candidates failed BM25
        # rerank) → can't apply a meaningful floor. Return capped only.
        return capped
    threshold = score_floor * top
    return [r for r in capped if r[1] >= threshold]


# ──────────────────────────────────────────────────────────────────────
# Representative-offer selection per sort
# ──────────────────────────────────────────────────────────────────────

PriceResolver = Callable[[dict], Decimal | None]


@dataclass(slots=True)
class _Materialised:
    """A hash that survived sort-aware representative selection."""
    article_hash: str
    relevance_score: float
    representative_offer: dict
    resolved_price: Decimal | None
    article_name: str | None


def pick_representative(
    offers: list[dict],
    *,
    plan: SortPlan,
    price_filter_active: bool,
    price_resolver: PriceResolver,
) -> tuple[dict, Decimal | None] | None:
    """Pick the representative offer for an article hash according to `plan`.

    Returns `(offer, resolved_price)` or `None` if no offer survives the
    price post-pass. `price_filter_active` distinguishes "the request
    has a price filter that must pass" from "no filter — resolved_price
    is informational only".

    Per-sort rules:

      relevance / name: alphabetically lowest articleId among offers
        passing the price filter (deterministic, sort-agnostic).
      articleId,asc: lowest articleId. articleId,desc: highest.
      price,asc: cheapest resolved price (lowest articleId tiebreak).
      price,desc: most expensive resolved price.

    For sort=price, an offer with no resolved price drops from the
    representative pool — matches legacy ES behaviour (missing prices
    don't sort to the end; they're excluded).
    """
    if not offers:
        return None

    if plan.field is SortField.PRICE:
        # Each candidate must have a resolved price to participate.
        priced: list[tuple[dict, Decimal]] = []
        for o in offers:
            p = price_resolver(o)
            if p is None:
                continue
            priced.append((o, p))
        if not priced:
            return None
        # Sort by (price, id) to get the chosen one deterministically.
        # `desc` flips both — id tiebreak still asc on equal prices is
        # what we want for stability, but Python's reverse=True flips
        # both keys. Use explicit negation for the price key when desc.
        if plan.descending:
            priced.sort(key=lambda r: (-float(r[1]), str(r[0]["id"])))
        else:
            priced.sort(key=lambda r: (float(r[1]), str(r[0]["id"])))
        return priced[0][0], priced[0][1]

    # All other plans: filter by price post-pass first, then pick by id.
    survivors: list[tuple[dict, Decimal | None]] = []
    for o in offers:
        if price_filter_active:
            p = price_resolver(o)
            if p is None:
                continue
            survivors.append((o, p))
        else:
            survivors.append((o, None))

    if not survivors:
        return None

    if plan.field is SortField.ARTICLE_ID and plan.descending:
        survivors.sort(key=lambda r: str(r[0]["id"]), reverse=True)
    else:
        # relevance / name / articleId asc: alphabetically lowest.
        survivors.sort(key=lambda r: str(r[0]["id"]))

    return survivors[0]


# ──────────────────────────────────────────────────────────────────────
# Final sort
# ──────────────────────────────────────────────────────────────────────

def sort_items(items: list[_Materialised], plan: SortPlan) -> list[_Materialised]:
    """Final article-level sort with the deterministic `articleId,asc`
    tiebreak. Sort is stable, so a two-pass sort (tiebreak first,
    primary second) yields a primary-then-tiebreak ordering."""
    if not items:
        return items

    # Tiebreak pass — articleId asc — is universal.
    items = sorted(items, key=lambda m: str(m.representative_offer["id"]))

    if plan.is_relevance:
        # Relevance descending. Stable sort preserves the articleId asc
        # tiebreak within equal relevance scores.
        return sorted(items, key=lambda m: m.relevance_score, reverse=True)

    if plan.field is SortField.ARTICLE_ID:
        return sorted(items, key=lambda m: str(m.representative_offer["id"]),
                      reverse=plan.descending)

    if plan.field is SortField.NAME:
        return sorted(items, key=lambda m: (m.article_name or "").lower(),
                      reverse=plan.descending)

    if plan.field is SortField.PRICE:
        # Articles without a resolved price never reach this point —
        # `pick_representative(plan=PRICE)` drops them. So `resolved_price`
        # is non-None for every item here.
        return sorted(
            items,
            key=lambda m: m.resolved_price if m.resolved_price is not None else Decimal(0),
            reverse=plan.descending,
        )

    raise ValueError(f"unhandled sort plan: {plan!r}")  # defensive
