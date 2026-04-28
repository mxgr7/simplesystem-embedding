"""Canonical MongoDB → Milvus projection (I1).

Pure function: a joined MongoDB record (offer + pricings collection rows
+ coreArticleMarkers collection rows, as produced by `dump_mongo_sample.js`)
maps to a Milvus row dict matching spec §7 / F1's collection schema —
minus `offer_embedding`, which the caller adds (deterministic stub for
tests, TEI for production).

Legacy parity sources:
  * `articleId` composite ........ `commons/.../domain/ArticleId.java`
  * `friendlyId` from vendorId ... `indexer/friendly_id.py` (port of
                                   `com.devskiller.friendly_id` 1.1.0)
  * Category path encoding ....... `commons/.../domain/CategoryPath.java`
  * Single-unit price ............ `indexer/.../CalculatingPrice.java`
                                   `indexer/.../Pricing.java#priceForLowestAmount`
  * PricingType priorities ....... OPEN=1 CLOSED=2 GROUP=3 DEDICATED=4
                                   per `commons/.../PricingType.java`
  * Features token format ........ `name=value`; legacy never serialises
                                   through `=` so per-spec we reject + log
                                   + drop entries whose value contains `=`

Two intentional simplifications vs. legacy:
  * `eclass{5,7}_code` / `s2class_code` take the FIRST element of the
    legacy `Set<Integer>`. Production data has at most one element in
    practice, but the data model permits multiple — if multi-eclass rows
    appear we'd extend to ARRAY columns.
  * Empty/missing fields fall back to schema defaults (empty arrays,
    empty strings, zero ints) rather than failing the row. Callers can
    inspect the returned `dropped_features` list if they need to surface
    bad input.
"""

from __future__ import annotations

import base64
import logging
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from indexer.friendly_id import to_friendly_id

log = logging.getLogger(__name__)


# Legacy `commons/.../PricingType.java`. Lower = higher priority at
# query-time resolution (see `PriceFilterProvider` + `prices.resolve_price`).
_PRICING_TYPE_PRIORITY = {
    "OPEN": 1,
    "CLOSED": 2,
    "GROUP": 3,
    "DEDICATED": 4,
}

# `commons/.../CategoryPath.java`.
_PATH_SEPARATOR = "¦"
_PATH_ESCAPE = "|"

_MAX_CATEGORY_DEPTH = 5  # Schema has category_l1..l5.


@dataclass
class ProjectionResult:
    """A single projected row plus any soft-failure diagnostics."""

    row: dict[str, Any]
    dropped_features: list[tuple[str, str]] = field(default_factory=list)


# ---------- helpers -------------------------------------------------------

def _decode_uuid(value: Any) -> uuid.UUID:
    """Accept the EJSON binary form `{"$binary": {"base64": ..., "subType": "04"}}`,
    a plain `uuid.UUID`, or a hyphenated string."""
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str):
        return uuid.UUID(value)
    if isinstance(value, dict) and "$binary" in value:
        b64 = value["$binary"]["base64"]
        return uuid.UUID(bytes=base64.b64decode(b64))
    raise ValueError(f"unrecognised UUID encoding: {value!r}")


def _b64url_no_pad(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).rstrip(b"=").decode("ascii")


def _encode_path(elements: list[str]) -> str:
    return _PATH_SEPARATOR.join(
        e.replace(_PATH_SEPARATOR, _PATH_ESCAPE) for e in elements
    )


def _to_decimal(value: Any, *, default: Decimal | None = None) -> Decimal | None:
    if value is None or value == "":
        return default
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _single_unit_price(pricing_details: dict[str, Any]) -> Decimal | None:
    """Mirror `CalculatingPrice.singleUnitPrice`: take the staggered entry
    with the lowest `minQuantity`, then divide by `priceQuantity` (treated
    as 1 if null/zero)."""
    prices = (pricing_details.get("prices") or {})
    staggered = prices.get("staggeredPrices") or []
    if not staggered:
        return None
    lowest = min(
        staggered,
        key=lambda s: _to_decimal(s.get("minQuantity"), default=Decimal("0")) or Decimal("0"),
    )
    base_price = _to_decimal(lowest.get("price"))
    if base_price is None:
        return None
    qty = _to_decimal(pricing_details.get("priceQuantity"), default=Decimal("1"))
    if qty is None or qty == 0:
        qty = Decimal("1")
    return base_price / qty


def _project_one_pricing(
    pricing_details: dict[str, Any], *, fallback_currency: str | None = None,
) -> dict[str, Any] | None:
    """Project a single legacy `PricingDetails` into the row's `prices`
    JSON entry. Returns None if the entry has no resolvable price."""
    if not pricing_details:
        return None
    price = _single_unit_price(pricing_details)
    if price is None:
        return None
    type_name = pricing_details.get("type") or "OPEN"
    priority = _PRICING_TYPE_PRIORITY.get(type_name, _PRICING_TYPE_PRIORITY["OPEN"])
    currency = (pricing_details.get("prices") or {}).get("currencyCode") or fallback_currency or ""
    source_id = pricing_details.get("sourcePriceListId")
    return {
        "price": float(price),
        "currency": currency,
        "priority": priority,
        "sourcePriceListId": str(_decode_uuid(source_id)) if source_id else "",
    }


def _project_features(
    raw_features: list[dict[str, Any]] | None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """`name=value` ARRAY tokens (§7). Reject + log + drop entries whose
    value contains `=` so the separator stays unambiguous."""
    tokens: list[str] = []
    dropped: list[tuple[str, str]] = []
    for f in raw_features or []:
        name = f.get("name") or ""
        for v in f.get("values") or []:
            v_str = str(v)
            if "=" in v_str:
                dropped.append((name, v_str))
                log.warning(
                    "dropping feature with '=' in value: name=%r value=%r", name, v_str,
                )
                continue
            tokens.append(f"{name}={v_str}")
    return tokens, dropped


def _project_categories(
    paths: list[dict[str, Any]] | None,
) -> dict[str, list[str]]:
    """Each CategoryPath emits one entry per depth (1..min(depth, 5)) into
    the matching `category_l{N}` array. De-duped per depth."""
    bins: list[list[str]] = [[] for _ in range(_MAX_CATEGORY_DEPTH)]
    for p in paths or []:
        elements = p.get("elements") or []
        for depth in range(1, min(len(elements), _MAX_CATEGORY_DEPTH) + 1):
            encoded = _encode_path(elements[:depth])
            if encoded not in bins[depth - 1]:
                bins[depth - 1].append(encoded)
    return {f"category_l{d}": bins[d - 1] for d in range(1, _MAX_CATEGORY_DEPTH + 1)}


def _project_eclass(eclass_groups: dict[str, Any] | None, key: str) -> int:
    if not eclass_groups:
        return 0
    codes = eclass_groups.get(key) or []
    return int(codes[0]) if codes else 0


def _project_markers(
    markers: list[dict[str, Any]] | None,
) -> tuple[list[str], list[str]]:
    enabled: list[str] = []
    disabled: list[str] = []
    for m in markers or []:
        src = m.get("coreArticleListSourceId")
        if src is None:
            continue
        src_id = str(_decode_uuid(src))
        if m.get("coreArticleMarker"):
            if src_id not in enabled:
                enabled.append(src_id)
        else:
            if src_id not in disabled:
                disabled.append(src_id)
    return enabled, disabled


# ---------- top-level ----------------------------------------------------

def project(record: dict[str, Any]) -> ProjectionResult:
    """Project a joined `(offer, pricings, markers)` record into a Milvus row.

    The output dict matches every §7 column except `offer_embedding`,
    which the caller fills in (stub vector for tests, TEI for prod)."""
    outer = record["offer"]
    inner = outer["offer"]              # nested Offer
    params = inner.get("offerParams") or {}

    vendor_uuid = _decode_uuid(outer["vendorId"])
    article_number = outer["articleNumber"]
    pk = f"{to_friendly_id(vendor_uuid)}:{_b64url_no_pad(article_number)}"

    # Prices: built-in `pricings.{open,closed}` + joined `pricings[]` rows.
    prices_out: list[dict[str, Any]] = []
    nested_pricings = inner.get("pricings") or {}
    for slot in ("open", "closed"):
        entry = _project_one_pricing(nested_pricings.get(slot))
        if entry is not None:
            prices_out.append(entry)
    for p in record.get("pricings") or []:
        entry = _project_one_pricing(p.get("pricingDetails") or {})
        if entry is not None:
            prices_out.append(entry)

    features_tokens, dropped = _project_features(params.get("features"))
    categories = _project_categories(params.get("categoryPaths"))
    enabled_sources, disabled_sources = _project_markers(record.get("markers"))
    eclass_groups = params.get("eclassGroups") or {}

    related = inner.get("relatedArticleNumbers") or {}

    catalog_version_id = outer.get("catalogVersionId")
    catalog_version_ids = [str(_decode_uuid(catalog_version_id))] if catalog_version_id else []

    row: dict[str, Any] = {
        "id": pk,
        "name": params.get("name") or "",
        "manufacturerName": params.get("manufacturerName") or "",
        "ean": params.get("ean") or "",
        "article_number": article_number,
        "vendor_id": str(vendor_uuid),
        "catalog_version_ids": catalog_version_ids,
        **categories,
        "prices": prices_out,
        "delivery_time_days_max": int(params.get("deliveryTime") or 0),
        "core_marker_enabled_sources": enabled_sources,
        "core_marker_disabled_sources": disabled_sources,
        "eclass5_code": _project_eclass(eclass_groups, "ECLASS_5_1"),
        "eclass7_code": _project_eclass(eclass_groups, "ECLASS_7_1"),
        "s2class_code": _project_eclass(eclass_groups, "S2CLASS"),
        "features": features_tokens,
        "relationship_accessory_for": list(related.get("accessoryFor") or []),
        "relationship_spare_part_for": list(related.get("sparePartFor") or []),
        "relationship_similar_to": list(related.get("similarTo") or []),
    }
    return ProjectionResult(row=row, dropped_features=dropped)
