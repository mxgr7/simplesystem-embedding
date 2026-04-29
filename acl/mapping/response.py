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

from typing import Any


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
    for raw in raw_articles:
        legacy: dict[str, Any] = {"articleId": raw["articleId"]}
        if explain:
            # §2.2 — stub. The legacy schema lets clients parse a
            # non-null string here without trying to deserialize a
            # scoring breakdown.
            legacy["explanation"] = "N/A"
        # `score` intentionally not forwarded.
        articles_out.append(legacy)

    summaries_out = ftsearch_body.get("summaries") or {}

    raw_metadata = ftsearch_body.get("metadata") or {}
    # Drop ACL-extension fields that aren't in the legacy contract.
    metadata_out = {
        k: v for k, v in raw_metadata.items()
        if k not in ("recallClipped", "hitCountClipped")
    }

    return {
        "articles": articles_out,
        "summaries": summaries_out,
        "metadata": metadata_out,
    }


__all__ = ["map_response"]
