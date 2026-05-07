"""A3 — ftsearch response → legacy envelope.

Pure function: takes the raw ftsearch JSON response + the legacy
request's `explain` flag, returns the legacy-shaped response dict.

The two contracts are mostly aligned by design (F2). The mapper's
real work:

  - **`articles[].score` → dropped**. Legacy returns `articleId` +
    optional `explanation`; the score is an ftsearch-internal signal
    callers don't expose.
  - **`articles[].explanation` injected**. Per §2.2 deviation: the
    string `"N/A"` when the legacy request had `explain=true`, else
    omitted (key not present). The schema lists it as nullable, so
    callers parse the absence as null/None.
  - **`metadata.recallClipped`, `hitCountClipped` → dropped**.
    These are ftsearch-side observability flags (F9 PathB-overflow,
    F4 hitcount-cap) the legacy contract doesn't carry. They surface
    on ftsearch's own /metrics for operators; the ACL drops them
    from the wire before the next-gen client sees the response.
  - **Summaries pass through unchanged**. F2 + F5 designed the wire
    shape to match legacy field-for-field.
"""

from __future__ import annotations

import base64
import string
import uuid
from typing import Any

_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _uuid_to_friendly(u: uuid.UUID) -> str:
    """Devskiller FriendlyId-compatible base62 encoding of a UUID."""
    n = int.from_bytes(u.bytes, "big", signed=False)
    digits: list[str] = []
    while n:
        n, rem = divmod(n, 62)
        digits.append(_BASE62[rem])
    return "".join(reversed(digits)).rjust(22, "0")


def _b64url_no_pad(value: str) -> str:
    return (
        base64.urlsafe_b64encode(value.encode("utf-8"))
        .rstrip(b"=")
        .decode("ascii")
    )


def _is_b64url_utf8_token(value: str) -> bool:
    """Best-effort detection for already legacy-encoded article numbers.

    Historical ftsearch offer ids used ``{uuid}:{b64url(articleNumber)}:{uuid}``,
    while the v7 collections currently used for local testing use
    ``{uuid}:{rawArticleNumber}:{uuid}``. The legacy backend always expects
    the second part to be b64url. Preserve already-encoded ids, but encode raw
    ids before forwarding them.
    """
    if not value or value.isdigit() or any(c.isspace() for c in value):
        return False
    if any(c not in string.ascii_letters + string.digits + "-_" for c in value):
        return False
    if len(value) % 4 == 1:
        return False
    try:
        padded = value + ("=" * ((4 - len(value) % 4) % 4))
        decoded = base64.urlsafe_b64decode(padded).decode("utf-8")
    except Exception:
        return False
    if not decoded or any(not c.isprintable() for c in decoded):
        return False
    return _b64url_no_pad(decoded) == value


def _to_legacy_article_id(ftsearch_id: str) -> str:
    """Convert ftsearch's 3-part articleId to legacy 2-part format.

    ftsearch v7: ``{rawUUID}:{rawArtNum}:{rawUUID}``
    older ids:   ``{rawUUID}:{b64url(artNum)}:{rawUUID}``
    legacy:      ``{friendlyId(vendorUUID)}:{b64url(artNum)}``
    """
    parts = ftsearch_id.split(":")
    if len(parts) == 3:
        try:
            vendor_uuid = uuid.UUID(parts[0])
        except ValueError:
            return ftsearch_id
        article_part = (
            parts[1] if _is_b64url_utf8_token(parts[1]) else _b64url_no_pad(parts[1])
        )
        return f"{_uuid_to_friendly(vendor_uuid)}:{article_part}"
    return ftsearch_id


def map_response(
    ftsearch_body: dict[str, Any],
    *,
    explain: bool,
) -> dict[str, Any]:
    """Build the legacy `{articles, summaries, metadata}` response from
    ftsearch's response.

    `explain` comes from the original `LegacySearchRequest.explain`
    field — needed because the response from ftsearch doesn't carry
    it (we drop it on the request side per A2)."""
    raw_articles = ftsearch_body.get("articles") or []
    articles_out: list[dict[str, Any]] = []
    seen_article_ids: set[str] = set()
    for raw in raw_articles:
        legacy_article_id = _to_legacy_article_id(raw["articleId"])
        # ftsearch v7 returns offer ids (`vendor:article:catalogVersion`).
        # The legacy backend consumes article ids (`vendor:article`) and
        # fails on duplicates when multiple catalog versions of the same
        # article appear in one page. Preserve first-hit order and drop
        # duplicate offers for the same legacy article.
        if legacy_article_id in seen_article_ids:
            continue
        seen_article_ids.add(legacy_article_id)
        legacy: dict[str, Any] = {"articleId": legacy_article_id}
        if explain:
            # §2.2 — stub. The legacy schema lets clients parse a
            # non-null string here without trying to deserialize a
            # scoring breakdown.
            legacy["explanation"] = "N/A"
        # `score` intentionally not forwarded.
        articles_out.append(legacy)

    _SUMMARIES_KEYS = {
        "vendorSummaries", "manufacturerSummaries", "featureSummaries",
        "pricesSummary", "categoriesSummary", "eClass5Categories",
        "eClass7Categories", "s2ClassCategories", "eClassesAggregations",
    }
    raw_summaries = ftsearch_body.get("summaries") or {}
    summaries_out = {k: v for k, v in raw_summaries.items() if k in _SUMMARIES_KEYS}

    # eClassesAggregations: pass through as {id, count} — matches legacy spec.
    if "eClassesAggregations" in summaries_out:
        summaries_out["eClassesAggregations"] = [
            {"id": item.get("id", ""), "count": item.get("count", 0)}
            for item in summaries_out["eClassesAggregations"]
        ]

    _METADATA_KEYS = {"page", "pageSize", "pageCount", "term", "hitCount"}
    raw_metadata = ftsearch_body.get("metadata") or {}
    metadata_out = {k: v for k, v in raw_metadata.items() if k in _METADATA_KEYS}
    # Legacy always returns pageCount >= 1 (even for empty results).
    if metadata_out.get("pageCount", 1) < 1:
        metadata_out["pageCount"] = 1

    return {
        "articles": articles_out,
        "summaries": summaries_out,
        "metadata": metadata_out,
    }


__all__ = ["map_response"]
