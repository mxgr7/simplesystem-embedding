"""Unit tests for `search-api/prices.py` (F3.2)."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

SEARCH_API_DIR = Path(__file__).resolve().parent.parent / "search-api"
sys.path.insert(0, str(SEARCH_API_DIR))

from prices import decode_minor_units, passes_price_filter, resolve_price  # noqa: E402


# ---------- decode_minor_units -------------------------------------------

def test_decode_eur_two_digits() -> None:
    assert decode_minor_units(1500, "EUR") == Decimal("15.00")


def test_decode_jpy_zero_digits() -> None:
    assert decode_minor_units(1500, "JPY") == Decimal("1500")


def test_decode_unknown_currency_raises() -> None:
    with pytest.raises(ValueError):
        decode_minor_units(100, "XXX")


def test_decode_case_insensitive() -> None:
    assert decode_minor_units(99, "eur") == Decimal("0.99")


# ---------- resolve_price -------------------------------------------------

@pytest.fixture
def offer_prices() -> list[dict]:
    # Mixed currencies × price lists × priorities. Priority 4 = DEDICATED.
    return [
        {"price": 100.00, "currency": "EUR", "priority": 1, "sourcePriceListId": "list-A"},
        {"price": 95.00,  "currency": "EUR", "priority": 2, "sourcePriceListId": "list-A"},
        {"price": 90.00,  "currency": "EUR", "priority": 4, "sourcePriceListId": "list-B"},
        {"price": 110.00, "currency": "USD", "priority": 1, "sourcePriceListId": "list-A"},
    ]


def test_resolve_picks_highest_priority(offer_prices: list[dict]) -> None:
    # Both list-A and list-B in scope → DEDICATED@list-B (priority 4) wins.
    assert resolve_price(
        offer_prices, currency="EUR", source_price_list_ids=["list-A", "list-B"],
    ) == Decimal("90.00")


def test_resolve_filters_by_price_list(offer_prices: list[dict]) -> None:
    # Only list-A in scope → CLOSED@list-A (priority 2) wins.
    assert resolve_price(
        offer_prices, currency="EUR", source_price_list_ids=["list-A"],
    ) == Decimal("95.00")


def test_resolve_filters_by_currency(offer_prices: list[dict]) -> None:
    assert resolve_price(
        offer_prices, currency="USD", source_price_list_ids=["list-A"],
    ) == Decimal("110.00")


def test_resolve_returns_none_when_no_match(offer_prices: list[dict]) -> None:
    assert resolve_price(
        offer_prices, currency="GBP", source_price_list_ids=["list-A"],
    ) is None


def test_resolve_empty_scope_returns_none(offer_prices: list[dict]) -> None:
    # Legacy: empty terms list matches nothing.
    assert resolve_price(offer_prices, currency="EUR", source_price_list_ids=[]) is None


def test_resolve_empty_prices_returns_none() -> None:
    assert resolve_price([], currency="EUR", source_price_list_ids=["list-A"]) is None
    assert resolve_price(None, currency="EUR", source_price_list_ids=["list-A"]) is None


# ---------- passes_price_filter ------------------------------------------

def test_passes_within_range_eur() -> None:
    prices = [{"price": 15.00, "currency": "EUR", "priority": 1, "sourcePriceListId": "L"}]
    assert passes_price_filter(
        prices,
        request_currency="EUR",
        source_price_list_ids=["L"],
        bound_currency_code="EUR",
        min_minor=1000,   # 10.00
        max_minor=2000,   # 20.00
    )


def test_rejects_below_min() -> None:
    prices = [{"price": 5.00, "currency": "EUR", "priority": 1, "sourcePriceListId": "L"}]
    assert not passes_price_filter(
        prices,
        request_currency="EUR",
        source_price_list_ids=["L"],
        bound_currency_code="EUR",
        min_minor=1000,
        max_minor=None,
    )


def test_rejects_above_max() -> None:
    prices = [{"price": 25.00, "currency": "EUR", "priority": 1, "sourcePriceListId": "L"}]
    assert not passes_price_filter(
        prices,
        request_currency="EUR",
        source_price_list_ids=["L"],
        bound_currency_code="EUR",
        min_minor=None,
        max_minor=2000,
    )


def test_jpy_bounds_decode_with_zero_digits() -> None:
    # 1500 JPY (no decimal scaling)
    prices = [{"price": 1500, "currency": "JPY", "priority": 1, "sourcePriceListId": "L"}]
    assert passes_price_filter(
        prices,
        request_currency="JPY",
        source_price_list_ids=["L"],
        bound_currency_code="JPY",
        min_minor=1000,
        max_minor=2000,
    )


def test_currency_two_roles_split() -> None:
    """Top-level `currency` matches; bound `currencyCode` only decodes.

    Row priced 15.00 EUR. Request asks for EUR price between bounds
    decoded as JPY (so 1500 → 1500, not 15.00). 15.00 EUR is far below
    1500, so the filter rejects.
    """
    prices = [{"price": 15.00, "currency": "EUR", "priority": 1, "sourcePriceListId": "L"}]
    assert not passes_price_filter(
        prices,
        request_currency="EUR",
        source_price_list_ids=["L"],
        bound_currency_code="JPY",
        min_minor=1000,    # JPY 1000
        max_minor=2000,    # JPY 2000
    )


def test_no_resolved_price_fails() -> None:
    assert not passes_price_filter(
        prices=[{"price": 10.00, "currency": "USD", "priority": 1, "sourcePriceListId": "L"}],
        request_currency="EUR",
        source_price_list_ids=["L"],
        bound_currency_code="EUR",
        min_minor=None,
        max_minor=None,
    )
