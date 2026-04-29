"""Price-resolution helpers shared by F3 (priceFilter post-pass), F4
(sort-by-price), and F5 (PRICES aggregation).

Single source of truth for: given a row's `prices` JSON array and a
request's `(currency, sourcePriceListIds)` scope, what scalar price
applies?

Legacy parity (per spec §3 + §4.3, simplified per F3 packet):

  * Filter to entries matching `currency` AND
    `sourcePriceListId ∈ scope_price_list_ids`.
  * Among matching entries, pick the one with the highest `priority`
    (PricingType ladder: OPEN=1, CLOSED=2, GROUP=3, DEDICATED=4).
  * Return the entry's `price` as Decimal, or None if no entry qualifies.

`priceFilter.min`/`max` are minor-unit integers (per spec §3); decoding to
a Decimal happens via `decode_minor_units(value, currency_code)` using
ISO-4217 default fraction digits. EUR → 2 (1500 → 15.00); JPY → 0
(1500 → 1500). Unknown currency codes raise — silent fallback to "2"
would mis-decode JPY.

The two currency roles per §3:
  - top-level `currency` drives matching (`resolve_price.currency`)
  - `priceFilter.currencyCode` drives bound decoding only
    (`decode_minor_units.currency_code`)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable, Mapping

# ISO-4217 default fraction digits for every currency the catalogue
# carries (see `CATALOG_CURRENCIES`). USD/JPY are out-of-catalogue but
# kept for legacy callers. To add a currency: extend here, the
# CATALOG_CURRENCIES tuple, and the matching constants in
# `scripts/create_{articles,offers}_collection.py` and
# `indexer/projection.py` (see `test_catalog_currencies_match_script`).
_FRACTION_DIGITS: Mapping[str, int] = {
    "EUR": 2,
    "USD": 2,
    "GBP": 2,
    "CHF": 2,
    "HUF": 2,
    "PLN": 2,
    "CZK": 2,
    "CNY": 2,
    "JPY": 0,
}

# Lowercased currencies that have a `{ccy}_price_min/max` envelope column
# on `offers_v{N}` (per F8). Mirror of `scripts/create_offers_collection.py`
# and `indexer/projection.py`. Filter translation consults this set
# before emitting a price-band clause: if the top-level currency isn't
# in the catalogue, the clause is skipped (no envelope column to compare
# against) and the post-pass alone enforces the price filter.
CATALOG_CURRENCIES: tuple[str, ...] = ("eur", "chf", "huf", "pln", "gbp", "czk", "cny")


def decode_minor_units(value: int, currency_code: str) -> Decimal:
    digits = _FRACTION_DIGITS.get(currency_code.upper())
    if digits is None:
        raise ValueError(
            f"unknown currency_code {currency_code!r}; extend prices._FRACTION_DIGITS"
        )
    if digits == 0:
        return Decimal(int(value))
    return Decimal(int(value)) / (Decimal(10) ** digits)


def resolve_price(
    prices: Iterable[Mapping] | None,
    *,
    currency: str,
    source_price_list_ids: Iterable[str],
) -> Decimal | None:
    if not prices:
        return None
    allowed = {str(x) for x in source_price_list_ids}
    if not allowed:
        # Legacy `PriceListFilterBuilder.terms = []` matches no entries.
        return None
    cur = currency.upper()
    best: Mapping | None = None
    for entry in prices:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("currency", "")).upper() != cur:
            continue
        if str(entry.get("sourcePriceListId", "")) not in allowed:
            continue
        if best is None or int(entry.get("priority", 0)) > int(best.get("priority", 0)):
            best = entry
    if best is None or best.get("price") is None:
        return None
    return Decimal(str(best["price"]))


def passes_price_filter(
    prices: Iterable[Mapping] | None,
    *,
    request_currency: str,
    source_price_list_ids: Iterable[str],
    bound_currency_code: str,
    min_minor: int | None,
    max_minor: int | None,
) -> bool:
    """True iff the row's resolved price exists and lies in [min, max]."""
    resolved = resolve_price(
        prices,
        currency=request_currency,
        source_price_list_ids=source_price_list_ids,
    )
    if resolved is None:
        return False
    if min_minor is not None and resolved < decode_minor_units(min_minor, bound_currency_code):
        return False
    if max_minor is not None and resolved > decode_minor_units(max_minor, bound_currency_code):
        return False
    return True
