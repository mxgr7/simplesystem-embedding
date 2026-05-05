"""Data-integrity audit for the articles_v6 / offers_v6 Milvus collections.

Checks referential integrity, field validity, and cross-collection
consistency using sampled queries (bounded by ``limit=`` to avoid
overwhelming Milvus).

Findings are classified:
  CRITICAL  -- would cause runtime join failures, missing results, or crashes.
  WARN      -- data-quality issue that degrades search relevance / UX.

Skipped when Milvus is not reachable or either collection is missing.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import pytest
from pymilvus import MilvusClient

MILVUS_URI = "http://localhost:19530"
ARTICLES = "articles_v6"
OFFERS = "offers_v6"

# 32-char lower-case hex hash
_HASH_RE = re.compile(r"^[0-9a-f]{32}$")

# Sentinel used for articles/offers that lack a price in a given currency.
# See scripts/create_articles_collection.py  MAX_PRICE_SENTINEL.
_SENTINEL = 3.4028234663852886e38

CATALOG_CURRENCIES = ("eur", "chf", "huf", "pln", "gbp", "czk", "cny")

# Milvus caps (offset+limit) at 16384.  Stay under that.
SAMPLE_LIMIT = 10_000
_MILVUS_MAX_LIMIT = 16_384


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> MilvusClient:
    try:
        c = MilvusClient(uri=MILVUS_URI)
        if not c.has_collection(ARTICLES) or not c.has_collection(OFFERS):
            pytest.skip(f"One of {ARTICLES!r}/{OFFERS!r} missing")
        return c
    except Exception as exc:
        pytest.skip(f"Milvus not reachable: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query(client: MilvusClient, collection: str, **kwargs: Any) -> list[dict]:
    """Thin wrapper that sets a generous timeout for audit queries."""
    kwargs.setdefault("limit", SAMPLE_LIMIT)
    kwargs.setdefault("timeout", 30)
    return client.query(collection_name=collection, **kwargs)


# ---------------------------------------------------------------------------
# 1. Orphaned offers  [CRITICAL]
# ---------------------------------------------------------------------------

def test_no_orphaned_offers(client: MilvusClient) -> None:
    """Every offer's article_hash must exist in articles_v6.
    Orphans cause join failures at search time (article found via ANN
    but no offer data to return)."""
    offers = _query(client, OFFERS,
                    filter='id != ""',
                    output_fields=["id", "article_hash"],
                    limit=SAMPLE_LIMIT)
    offer_hashes = {o["article_hash"] for o in offers}

    # Resolve which of those hashes exist in articles
    if not offer_hashes:
        pytest.skip("offers collection is empty")

    # Query articles for each unique hash (batch via IN)
    batch = list(offer_hashes)[:500]  # cap batch size for expr length
    expr = "article_hash in [" + ",".join(f'"{h}"' for h in batch) + "]"
    existing = _query(client, ARTICLES,
                      filter=expr,
                      output_fields=["article_hash"],
                      limit=len(batch))
    existing_set = {a["article_hash"] for a in existing}

    orphan_hashes = set(batch) - existing_set
    orphan_offers = [o for o in offers if o["article_hash"] in orphan_hashes]

    assert not orphan_offers, (
        f"CRITICAL: {len(orphan_offers)} orphaned offers (article_hash not in {ARTICLES}). "
        f"Sample IDs: {[o['id'] for o in orphan_offers[:5]]}"
    )


# ---------------------------------------------------------------------------
# 2. Articles with zero offers  [WARN]
# ---------------------------------------------------------------------------

def test_articles_have_at_least_one_offer(client: MilvusClient) -> None:
    """Articles that exist without any matching offers show up in search
    results but have no price/vendor data to display."""
    articles = _query(client, ARTICLES,
                      filter='article_hash != ""',
                      output_fields=["article_hash"],
                      limit=SAMPLE_LIMIT)
    if not articles:
        pytest.skip("articles collection is empty")

    # Check in small batches to stay within Milvus's 16384 result cap.
    sample_hashes = [a["article_hash"] for a in articles[:500]]
    covered: set[str] = set()
    batch_size = 50
    for i in range(0, len(sample_hashes), batch_size):
        batch = sample_hashes[i : i + batch_size]
        expr = "article_hash in [" + ",".join(f'"{h}"' for h in batch) + "]"
        offers = _query(client, OFFERS,
                        filter=expr,
                        output_fields=["article_hash"],
                        limit=_MILVUS_MAX_LIMIT)
        covered.update(o["article_hash"] for o in offers)

    uncovered = [h for h in sample_hashes if h not in covered]

    if uncovered:
        pct = len(uncovered) / len(sample_hashes) * 100
        pytest.fail(
            f"WARN: {len(uncovered)}/{len(sample_hashes)} sampled articles "
            f"({pct:.1f}%) have zero offers. "
            f"Sample: {uncovered[:5]}"
        )


# ---------------------------------------------------------------------------
# 3. Broken article_hash format  [CRITICAL]
# ---------------------------------------------------------------------------

def test_article_hash_format_in_articles(client: MilvusClient) -> None:
    """article_hash must be a 32-char lowercase hex string."""
    rows = _query(client, ARTICLES,
                  filter='article_hash != ""',
                  output_fields=["article_hash"],
                  limit=SAMPLE_LIMIT)
    bad = [r["article_hash"] for r in rows if not _HASH_RE.match(r["article_hash"])]
    assert not bad, (
        f"CRITICAL: {len(bad)} articles with malformed article_hash. "
        f"Sample: {bad[:5]}"
    )


def test_article_hash_format_in_offers(client: MilvusClient) -> None:
    """article_hash on offers must also be 32-char hex."""
    rows = _query(client, OFFERS,
                  filter='id != ""',
                  output_fields=["article_hash"],
                  limit=SAMPLE_LIMIT)
    bad = [r["article_hash"] for r in rows if not _HASH_RE.match(r["article_hash"])]
    assert not bad, (
        f"CRITICAL: {len(bad)} offers with malformed article_hash. "
        f"Sample: {bad[:5]}"
    )


# ---------------------------------------------------------------------------
# 4. Empty/null critical fields on offers  [CRITICAL]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["vendor_id", "catalog_version_id", "article_hash"])
def test_offers_critical_fields_not_empty(client: MilvusClient, field: str) -> None:
    """Critical FK/grouping fields must not be empty strings."""
    rows = _query(client, OFFERS,
                  filter=f'{field} == ""',
                  output_fields=["id", field],
                  limit=100)
    assert not rows, (
        f"CRITICAL: {len(rows)} offers with empty {field}. "
        f"Sample IDs: {[r['id'] for r in rows[:5]]}"
    )


# ---------------------------------------------------------------------------
# 5. Price data consistency on offers  [CRITICAL]
# ---------------------------------------------------------------------------

def test_offers_prices_well_formed(client: MilvusClient) -> None:
    """Each offer's `prices` JSON must be a non-empty list of objects with
    at least {price, currency, sourcePriceListId}."""
    rows = _query(client, OFFERS,
                  filter='id != ""',
                  output_fields=["id", "prices"],
                  limit=SAMPLE_LIMIT)
    if not rows:
        pytest.skip("no offers")

    bad: list[str] = []
    required_keys = {"price", "currency", "sourcePriceListId"}

    for r in rows:
        prices = r["prices"]
        # prices is stored as JSON; pymilvus returns it as a Python object
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except (json.JSONDecodeError, TypeError):
                bad.append(f"{r['id']} (unparseable JSON)")
                continue

        if not isinstance(prices, list) or len(prices) == 0:
            bad.append(f"{r['id']} (empty or not a list)")
            continue

        for entry in prices:
            if not isinstance(entry, dict):
                bad.append(f"{r['id']} (entry not a dict)")
                break
            missing = required_keys - set(entry.keys())
            if missing:
                bad.append(f"{r['id']} (missing keys: {missing})")
                break

    assert not bad, (
        f"CRITICAL: {len(bad)} offers with malformed prices. "
        f"Sample: {bad[:10]}"
    )


# ---------------------------------------------------------------------------
# 6. Currency min/max consistency on articles  [WARN]
# ---------------------------------------------------------------------------

def test_article_price_envelope_not_inverted(client: MilvusClient) -> None:
    """For each currency, price_min must be <= price_max (excluding
    sentinel values used for 'no price in this currency')."""
    bad_total: list[tuple[str, str]] = []

    for ccy in CATALOG_CURRENCIES:
        col_min = f"{ccy}_price_min"
        col_max = f"{ccy}_price_max"
        # Exclude sentinel pairs (+MAX/-MAX) which are the intentional
        # 'no price' encoding -- only check real prices.
        rows = _query(
            client, ARTICLES,
            filter=f"{col_min} < {_SENTINEL} and {col_min} > {col_max}",
            output_fields=["article_hash", col_min, col_max],
            limit=100,
        )
        for r in rows:
            bad_total.append((r["article_hash"], ccy))

    assert not bad_total, (
        f"WARN: {len(bad_total)} articles with inverted price envelope "
        f"(min > max, excluding sentinels). "
        f"Sample: {bad_total[:10]}"
    )


# ---------------------------------------------------------------------------
# 7. Duplicate article_hash in articles  [CRITICAL]
# ---------------------------------------------------------------------------

def test_no_duplicate_article_hashes(client: MilvusClient) -> None:
    """article_hash is the PK so Milvus enforces uniqueness. This test
    verifies the invariant holds by sampling and checking for repeats."""
    rows = _query(client, ARTICLES,
                  filter='article_hash != ""',
                  output_fields=["article_hash"],
                  limit=SAMPLE_LIMIT)
    hashes = [r["article_hash"] for r in rows]
    dupes = len(hashes) - len(set(hashes))
    assert dupes == 0, (
        f"CRITICAL: {dupes} duplicate article_hash values found in sample "
        f"of {len(hashes)}"
    )


# ---------------------------------------------------------------------------
# 8. Category path depth consistency  [WARN]
# ---------------------------------------------------------------------------

def test_category_hierarchy_not_broken(client: MilvusClient) -> None:
    """If category_l2 is populated, category_l1 must also be populated.
    A broken hierarchy means the L1 facet filter won't match articles
    that do have deeper categories."""
    rows = _query(
        client, ARTICLES,
        filter='array_length(category_l2) > 0',
        output_fields=["article_hash", "category_l1", "category_l2"],
        limit=SAMPLE_LIMIT,
    )
    broken = [
        r["article_hash"] for r in rows
        if _is_empty_array(r.get("category_l1"))
    ]
    assert not broken, (
        f"WARN: {len(broken)} articles have category_l2 populated but "
        f"category_l1 empty (broken hierarchy). Sample: {broken[:5]}"
    )


def _is_empty_array(val: Any) -> bool:
    """Return True if val is an empty array, empty string, or the string '[]'."""
    if val is None:
        return True
    if isinstance(val, list):
        return len(val) == 0
    if isinstance(val, str):
        return val.strip() in ("", "[]")
    # pymilvus returns ARRAY fields as RepeatedScalarContainer
    try:
        return len(val) == 0
    except TypeError:
        return False


def _to_list(val: Any) -> list:
    """Coerce pymilvus ARRAY field values (RepeatedScalarContainer, str,
    or plain list) into a Python list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    # RepeatedScalarContainer or similar iterable
    try:
        return list(val)
    except TypeError:
        return []


# ---------------------------------------------------------------------------
# 9. Feature array size  [WARN]
# ---------------------------------------------------------------------------

def test_no_extremely_long_feature_arrays(client: MilvusClient) -> None:
    """Offers with very large feature arrays (>500 entries) can cause
    performance issues during filtering and serialisation."""
    # Milvus doesn't support array_length > N in filter for ARRAY<VARCHAR>,
    # so we sample and check client-side.
    rows = _query(client, OFFERS,
                  filter='array_length(features) > 0',
                  output_fields=["id", "features"],
                  limit=SAMPLE_LIMIT)
    oversized = []
    for r in rows:
        features = _to_list(r["features"])
        if len(features) > 500:
            oversized.append((r["id"], len(features)))

    assert not oversized, (
        f"WARN: {len(oversized)} offers with >500 features. "
        f"Sample: {oversized[:5]}"
    )


# ---------------------------------------------------------------------------
# 10. Relationship consistency  [WARN]
# ---------------------------------------------------------------------------

def test_relationship_accessory_for_targets_exist(client: MilvusClient) -> None:
    """Offers with relationship_accessory_for pointing to article_numbers
    that don't correspond to any offer in the system. Note: these are
    article_numbers (not article_hashes), so we check against offers.article_number."""
    rows = _query(client, OFFERS,
                  filter='array_length(relationship_accessory_for) > 0',
                  output_fields=["id", "relationship_accessory_for"],
                  limit=1000)
    if not rows:
        pytest.skip("no offers with relationship_accessory_for")

    # Collect a sample of referenced article_numbers.
    # pymilvus returns ARRAY fields as RepeatedScalarContainer (not
    # plain list), so use _to_list() for safe conversion.
    all_targets: set[str] = set()
    for r in rows:
        refs = _to_list(r["relationship_accessory_for"])
        all_targets.update(refs[:10])  # sample from each offer

    if not all_targets:
        pytest.skip("no relationship targets found")

    # Probe a batch of targets against offers.article_number
    batch = list(all_targets)[:200]
    expr = "article_number in [" + ",".join(f'"{t}"' for t in batch) + "]"
    found = _query(client, OFFERS,
                   filter=expr,
                   output_fields=["article_number"],
                   limit=min(len(batch), _MILVUS_MAX_LIMIT))
    found_set = {o["article_number"] for o in found}
    missing = set(batch) - found_set

    pct_missing = len(missing) / len(batch) * 100 if batch else 0
    # Relationship targets commonly reference articles from catalogs not
    # yet loaded, so a high miss rate is expected during incremental
    # import.  Report as a warning rather than hard-failing.
    if pct_missing > 10:
        import warnings
        warnings.warn(
            f"WARN: {len(missing)}/{len(batch)} ({pct_missing:.0f}%) "
            f"relationship_accessory_for targets not found in offers.article_number. "
            f"Sample missing: {list(missing)[:5]}",
            stacklevel=1,
        )
