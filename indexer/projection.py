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

One intentional simplification vs. legacy:
  * Empty/missing fields fall back to schema defaults (empty arrays,
    empty strings, zero ints) rather than failing the row. Callers can
    inspect the returned `dropped_features` list if they need to surface
    bad input.

`eclass{5,7}_code` / `s2class_code` are projected as `list[int]` —
every level of the legacy hierarchy is carried verbatim so a `terms`
query at any level matches via `array_contains[_any]` (matches the
ES `offers.eclass51Groups` / `eclass71Groups` / `s2classGroups`
keyword-array shape). Collapsing to a single int loses the parent
codes and silently breaks recall on parent-level filters.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable

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

# F9 article-hash. Bumping invalidates every existing hash and forces a
# full rebuild through the alias-swing playbook (I3) — there is no live
# migration path. Grep-able from operational tooling.
HASH_VERSION = "v1"

# Canonical-form separators for the hash input. Both are ASCII control
# chars (US, RS) that never appear in legitimate user-supplied text, so
# the canonical string needs no escaping.
_HASH_FIELD_SEP = "\x1f"
_HASH_ELEM_SEP = "\x1e"

# Catalogue currencies the per-article envelope spans (F8 + F9). Mirror
# of the same constant in `scripts/create_articles_collection.py`; kept
# in sync by `test_catalog_currencies_match_script` in
# `tests/test_projection.py`.
CATALOG_CURRENCIES = ("eur", "chf", "huf", "pln", "gbp", "czk", "cny")

# Envelope sentinel for "no price in this currency on this article".
# F9 doc specified NaN — Milvus 2.6 rejects NaN *and* ±Inf on FLOAT
# scalars (only finite values accepted, despite the misleading "is not
# a number or infinity" reject message). The asymmetric large-finite
# pair below preserves the natural range-filter semantics:
#
#   `{ccy}_price_min <= X` is false for `+MAX_PRICE_SENTINEL` (any plausible X)
#   `{ccy}_price_max >= Y` is false for `-MAX_PRICE_SENTINEL` (any plausible Y ≥ 0)
#   `ORDER BY {ccy}_price_min ASC` puts sentinel rows last (F4 browse safe)
#
# Value chosen well above any plausible price (~3.4e38, fp32 ceiling)
# so callers don't need a `> 0` / `< X` guard — the natural predicate
# does the right thing.
MAX_PRICE_SENTINEL = 3.4028234e38

# Article-level fields per the F9 topology block. Used by the row
# splitter below (offer rows = projected row minus these) and by the
# article aggregator (these fields are invariant across the hash group,
# OR are aggregated by it — see `customer_article_numbers` below).
#
# `customer_article_numbers` is the only entry here that's NOT invariant
# by hash construction: it's per-offer source data that the article
# aggregator UNIONs across the dedup group keyed by value. We carry it
# on the flat row so `aggregate_article` can read it uniformly via
# `r["customer_article_numbers"]` instead of receiving the raw Mongo
# joined rows again — same pattern as `prices` (per-offer source, lives
# on offers_v{N}, but read by `aggregate_article` to derive the
# article-side envelope columns). `to_offer_row` strips it because the
# offer collection has no column to receive it (article-level only per
# F9; `articles_v{N}` carries it as JSON).
_ARTICLE_LEVEL_KEYS: tuple[str, ...] = (
    "name", "manufacturerName",
    "category_l1", "category_l2", "category_l3", "category_l4", "category_l5",
    "eclass5_code", "eclass7_code", "s2class_code",
    "customer_article_numbers",
)


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


def _project_eclass(eclass_groups: dict[str, Any] | None, key: str) -> list[int]:
    if not eclass_groups:
        return []
    return [int(c) for c in eclass_groups.get(key) or []]


def _project_customer_numbers(
    joined_rows: list[dict[str, Any]] | None,
    *,
    catalog_value: str | None,
    catalog_version_id: uuid.UUID | None,
) -> list[dict[str, Any]]:
    """Per-offer customer-supplied SKU aliases in the inverted-by-value
    shape that mirrors the legacy ES `SearchArticleDocument.customerArticleNumbers`
    Nested field.

    Two sources fold into the same field per legacy
    `SearchArticleDocumentMapper.java:101-137`:

      - Joined `customerArticleNumbers` Mongo rows: each row carries
        `customerArticleNumber` (value) + `customerArticleNumbersListVersionId`
        (version_id from a customer's price-list / article-list upload).
      - The offer's own `offerParams.customerArticleNumber` (catalog-supplied),
        paired with the offer's `catalogVersionId` as version_id (per
        `OfferSourceId.asCustomerArticleNumberSourceId()` — a pure
        type-cast wrapping the same UUID).

    The inversion (one entry per distinct value, version_ids set) is
    load-bearing for entitlement filtering: ftsearch scopes a value
    match to only those version_ids the requesting customer is entitled
    to. A flat parallel-arrays encoding would silently match a value
    when only an unrelated value's version_id is allowed."""
    by_value: dict[str, set[str]] = {}
    for r in joined_rows or []:
        value = r.get("customerArticleNumber")
        if not value:
            continue
        version_raw = r.get("customerArticleNumbersListVersionId")
        if not version_raw:
            continue
        by_value.setdefault(value, set()).add(str(_decode_uuid(version_raw)))
    if catalog_value and catalog_version_id is not None:
        by_value.setdefault(catalog_value, set()).add(str(catalog_version_id))
    # Sort entries by value and version_ids within each entry — bulk-rerun
    # stability and easier diffing.
    return [
        {"value": v, "version_ids": sorted(by_value[v])}
        for v in sorted(by_value)
    ]


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
    catalog_version_uuid = _decode_uuid(catalog_version_id) if catalog_version_id else None
    catalog_version_ids = [str(catalog_version_uuid)] if catalog_version_uuid else []

    customer_numbers = _project_customer_numbers(
        record.get("customerArticleNumbers"),
        catalog_value=params.get("customerArticleNumber"),
        catalog_version_id=catalog_version_uuid,
    )

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
        "customer_article_numbers": customer_numbers,
    }
    return ProjectionResult(row=row, dropped_features=dropped)


# ---------- F9 two-stream emission ---------------------------------------
#
# `project()` above produces a single flat row matching the legacy
# pre-F9 single-collection schema. F9 splits that row across two
# collections — `articles_v{N}` (vector + BM25 + article-level scalars +
# per-currency envelope) and `offers_v{N}` (per-offer scalars + the
# `article_hash` join key). The helpers below convert flat rows into
# the two-stream shape:
#
#   compute_article_hash(row)     — sha256 of canonicalised embedded-field
#                                   tuple, truncated to 16 bytes hex.
#   to_offer_row(row, hash)       — strips article-level fields, attaches
#                                   `article_hash` + the placeholder
#                                   vector required by Milvus 2.6.
#   aggregate_article(rows)       — folds one or more flat rows that share
#                                   a hash into the article row, including
#                                   text_codes (BM25 corpus) and
#                                   `{ccy}_price_min/max` envelope columns.
#
# Article-side `offer_embedding` is *not* set here; the caller (test_loader
# for tests, the bulk orchestrator with TEI-cache-by-hash for production)
# attaches it. See F9 §"Hash function and embedded-field set" and
# §Topology.


def compute_article_hash(row: dict[str, Any]) -> str:
    """Hash the canonicalised embedded-field tuple for an article row.

    Inputs (per F9): name, manufacturerName, category_l1..l5,
    eclass5_code, eclass7_code, s2class_code. Array fields are sorted to
    canonicalise order — the projection's array order is not stable
    across re-runs, so the hash must be order-independent. Missing
    fields project to empty.

    Output: 32-char lowercase hex (sha256 truncated to 16 bytes).
    Collision probability at 10⁸ articles ≈ 10⁻²⁰. Halves the IN-clause
    wire cost vs full sha256 — relevant at PATH_B_HASH_LIMIT scale."""
    cat_blocks = []
    for d in range(1, _MAX_CATEGORY_DEPTH + 1):
        elems = sorted(row.get(f"category_l{d}") or [])
        cat_blocks.append(_HASH_ELEM_SEP.join(elems))
    eclass5 = _HASH_ELEM_SEP.join(str(c) for c in sorted(row.get("eclass5_code") or []))
    eclass7 = _HASH_ELEM_SEP.join(str(c) for c in sorted(row.get("eclass7_code") or []))
    s2class = _HASH_ELEM_SEP.join(str(c) for c in sorted(row.get("s2class_code") or []))
    canonical = _HASH_FIELD_SEP.join([
        row.get("name") or "",
        row.get("manufacturerName") or "",
        *cat_blocks,
        eclass5, eclass7, s2class,
    ])
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return digest[:16].hex()


def to_offer_row(
    row: dict[str, Any],
    *,
    article_hash: str,
    currencies: tuple[str, ...] = CATALOG_CURRENCIES,
) -> dict[str, Any]:
    """Project a flat row into the `offers_v{N}` shape: drop article-level
    fields, attach `article_hash` (join key) and `_placeholder_vector`
    (Milvus 2.6 requires every collection to declare at least one
    indexed vector field — see `scripts/create_offers_collection.py`).

    Also attaches F8 per-offer envelope columns derived from this row's
    `prices` JSON:

      - `price_list_ids` — sorted union of `prices[].sourcePriceListId`.
      - `currencies` — sorted union of `prices[].currency`. Restricted
        to known catalogue currencies (a currency outside `currencies`
        has no envelope column to land in, so its row in `prices` is
        kept on the JSON column for the post-pass to resolve).
      - `{ccy}_price_min/max` — min/max across this offer's prices in
        each known currency. `MAX_PRICE_SENTINEL` / `-MAX_PRICE_SENTINEL`
        when absent (range predicates naturally exclude — see
        `MAX_PRICE_SENTINEL` for the rejected-NaN-and-Inf deviation).
    """
    out = {k: v for k, v in row.items() if k not in _ARTICLE_LEVEL_KEYS}
    out["article_hash"] = article_hash
    out["_placeholder_vector"] = [0.0, 0.0]
    out.update(_offer_envelope(row.get("prices") or [], currencies=currencies))
    return out


def _offer_envelope(
    prices: list[dict[str, Any]],
    *,
    currencies: tuple[str, ...],
) -> dict[str, Any]:
    """Derive the F8 per-offer envelope columns from a single offer's
    `prices` list. `currencies` is the lowercased catalogue tuple."""
    price_list_ids = sorted({
        p["sourcePriceListId"] for p in prices
        if p.get("sourcePriceListId")
    })
    # `currencies` array column carries every currency the offer prices
    # in (lowercased to match the column tuple). We keep all observed
    # currencies, not just the catalogue subset — narrow filters on
    # rare currencies still need the array_contains pre-filter to drop
    # this row when its currency isn't requested. Range columns below
    # are still catalogue-restricted (one per declared `{ccy}_*` column).
    seen_currencies = sorted({
        (p.get("currency") or "").lower() for p in prices
        if p.get("currency")
    })

    envelope: dict[str, Any] = {
        "price_list_ids": price_list_ids,
        "currencies": seen_currencies,
    }
    by_ccy: dict[str, list[float]] = {c: [] for c in currencies}
    for p in prices:
        ccy = (p.get("currency") or "").lower()
        if ccy in by_ccy:
            by_ccy[ccy].append(float(p["price"]))
    for ccy, vals in by_ccy.items():
        envelope[f"{ccy}_price_min"] = min(vals) if vals else MAX_PRICE_SENTINEL
        envelope[f"{ccy}_price_max"] = max(vals) if vals else -MAX_PRICE_SENTINEL
    return envelope


def aggregate_article(
    rows: list[dict[str, Any]],
    *,
    currencies: tuple[str, ...] = CATALOG_CURRENCIES,
) -> dict[str, Any]:
    """Fold a hash group of flat rows into a single `articles_v{N}` row.

    Article-level fields (name, manufacturerName, categories, eclass)
    are invariant by hash construction — read off the first row.
    Aggregations across the group:

      - `text_codes` (BM25 input): name + manufacturerName + sorted
        distinct EANs across offers + sorted distinct article_numbers
        across offers. Sorting is for re-run stability.
      - `{ccy}_price_min/max`: min/max across every offer's `prices`
        entry whose lowercased currency matches a column in `currencies`.
        Sentinel for "no price in this currency": `+MAX_PRICE_SENTINEL`
        on `_min`, `-MAX_PRICE_SENTINEL` on `_max`. Range predicates
        naturally exclude (`_min <= X` is false; `_max >= Y` is false).
        Sort `ORDER BY {ccy}_price_min ASC` puts sentinel rows last (F4
        safe). See `MAX_PRICE_SENTINEL` for the rejected-NaN-and-Inf
        deviation from the F9 doc.

    Caller attaches `offer_embedding` (TEI in production, stub in tests)."""
    if not rows:
        raise ValueError("aggregate_article requires at least one row")
    rep = rows[0]
    article: dict[str, Any] = {
        "article_hash": compute_article_hash(rep),
        "name": rep.get("name") or "",
        "manufacturerName": rep.get("manufacturerName") or "",
    }
    for d in range(1, _MAX_CATEGORY_DEPTH + 1):
        article[f"category_l{d}"] = list(rep.get(f"category_l{d}") or [])
    for f in ("eclass5_code", "eclass7_code", "s2class_code"):
        article[f] = list(rep.get(f) or [])

    eans = sorted({r["ean"] for r in rows if r.get("ean")})
    nums = sorted({r["article_number"] for r in rows if r.get("article_number")})
    text_pieces = [article["name"], article["manufacturerName"], *eans, *nums]
    article["text_codes"] = " ".join(p for p in text_pieces if p)

    # UNION customer_article_numbers across the dedup group keyed by
    # value: when two offers in the same hash group have the same
    # customer-supplied SKU under different version_ids (e.g., catalog A
    # and price-list B both expose "BOLT-001" for the same physical
    # article), the article row's entry merges both version_ids.
    cans_union: dict[str, set[str]] = {}
    for r in rows:
        for entry in r.get("customer_article_numbers") or []:
            value = entry.get("value")
            if not value:
                continue
            cans_union.setdefault(value, set()).update(entry.get("version_ids") or [])
    article["customer_article_numbers"] = [
        {"value": v, "version_ids": sorted(cans_union[v])}
        for v in sorted(cans_union)
    ]

    by_ccy: dict[str, list[float]] = {c: [] for c in currencies}
    for r in rows:
        for p in r.get("prices") or []:
            ccy = (p.get("currency") or "").lower()
            if ccy in by_ccy:
                by_ccy[ccy].append(float(p["price"]))
    for ccy, vals in by_ccy.items():
        article[f"{ccy}_price_min"] = min(vals) if vals else MAX_PRICE_SENTINEL
        article[f"{ccy}_price_max"] = max(vals) if vals else -MAX_PRICE_SENTINEL

    return article


def group_by_hash(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group flat projected rows by their computed article_hash. Used by
    the bulk loader and by `aggregate_article` callers."""
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_hash.setdefault(compute_article_hash(r), []).append(r)
    return by_hash
