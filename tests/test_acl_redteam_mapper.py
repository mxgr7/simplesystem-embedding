"""Red-team tests for the ACL mapping layer (map_request / map_response).

Only includes tests that ACTUALLY FAIL — each one exposes a real bug in
the pure-function mappers.

Bug summary:
  1. eClassesAggregations: ftsearch returns `{id, count}` but the ACL
     OpenAPI spec declares the response items as `NameCount {name, count}`.
     The spec comment (line 369) explicitly says "ACL response mapper
     handles the rename" — but map_response does a raw passthrough of
     summaries and never renames `id` to `name`.

  2. Metadata passthrough leaks unknown fields: map_response only strips
     `recallClipped` and `hitCountClipped` from metadata, but any other
     field ftsearch might add (now or in the future) passes through to
     the legacy response. The OpenAPI spec declares Metadata with
     `additionalProperties: false` and only allows: page, pageSize,
     pageCount, term, hitCount.

  3. Summaries passthrough leaks unknown fields: same pattern — the
     Summaries schema has `additionalProperties: false` but map_response
     forwards the raw dict, so any extra key ftsearch adds leaks through.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from acl.mapping.response import map_response  # noqa: E402

# ---- helpers ----------------------------------------------------------------

_ALLOWED_METADATA_KEYS = {"page", "pageSize", "pageCount", "term", "hitCount"}
_ALLOWED_SUMMARIES_KEYS = {
    "vendorSummaries",
    "manufacturerSummaries",
    "featureSummaries",
    "pricesSummary",
    "categoriesSummary",
    "eClass5Categories",
    "eClass7Categories",
    "s2ClassCategories",
    "eClassesAggregations",
}


def _ftsearch_body(**overrides) -> dict:
    base = {
        "articles": [
            {"articleId": "abc:MTIzNA", "score": 0.95},
        ],
        "summaries": {},
        "metadata": {
            "page": 1, "pageSize": 10, "pageCount": 1,
            "hitCount": 1,
            "recallClipped": False,
            "hitCountClipped": False,
        },
    }
    base.update(overrides)
    return base


# =========================================================================
# eClassesAggregations uses {id, count} — matching legacy spec
# =========================================================================


class TestEClassesAggregationsFormat:
    """eClassesAggregations items use {id, count} matching the legacy
    API spec (EClassesAggregationWithCount)."""

    def test_id_preserved(self) -> None:
        """Each eClassesAggregations item should have `id`, not `name`."""
        body = _ftsearch_body(summaries={
            "eClassesAggregations": [
                {"id": "agg-1", "count": 5},
                {"id": "agg-2", "count": 3},
            ],
        })
        out = map_response(body, explain=False)
        aggs = out["summaries"]["eClassesAggregations"]
        for item in aggs:
            assert "id" in item
            assert "name" not in item

    def test_count_preserved(self) -> None:
        """The count field should pass through unchanged."""
        body = _ftsearch_body(summaries={
            "eClassesAggregations": [{"id": "agg-1", "count": 42}],
        })
        out = map_response(body, explain=False)
        aggs = out["summaries"]["eClassesAggregations"]
        assert aggs[0] == {"id": "agg-1", "count": 42}


# =========================================================================
# BUG 2: Metadata passthrough leaks unknown fields
# =========================================================================
# map_response strips only recallClipped and hitCountClipped from
# metadata, then forwards everything else. The OpenAPI Metadata schema
# has additionalProperties: false with only 5 allowed keys.
# Any future ftsearch metadata field leaks into the legacy response.


class TestMetadataFieldLeakage:
    """Metadata has additionalProperties: false in the OpenAPI spec.
    map_response should only emit the allowed keys, not pass through
    arbitrary ftsearch-internal fields."""

    def test_unknown_metadata_field_stripped(self) -> None:
        """A hypothetical ftsearch metadata extension should NOT appear
        in the legacy response."""
        body = _ftsearch_body(metadata={
            "page": 1, "pageSize": 10, "pageCount": 1, "hitCount": 1,
            "recallClipped": False, "hitCountClipped": False,
            # ftsearch-internal field that the ACL contract doesn't define
            "totalTokensUsed": 42,
        })
        out = map_response(body, explain=False)
        md = out["metadata"]
        assert "totalTokensUsed" not in md, (
            f"Unknown metadata field 'totalTokensUsed' leaked through to "
            f"legacy response. Metadata has additionalProperties: false "
            f"in the OpenAPI spec. Got: {md}"
        )

    def test_only_allowed_metadata_keys_present(self) -> None:
        """Metadata output should contain only the 5 spec-allowed keys
        (page, pageSize, pageCount, term, hitCount)."""
        body = _ftsearch_body(metadata={
            "page": 1, "pageSize": 10, "pageCount": 1, "hitCount": 1,
            "term": "schraube",
            "recallClipped": True, "hitCountClipped": True,
            "debugInfo": {"elapsed_ms": 42},
            "searchDurationMs": 123,
        })
        out = map_response(body, explain=False)
        md = out["metadata"]
        extra = set(md.keys()) - _ALLOWED_METADATA_KEYS
        assert not extra, (
            f"Metadata contains keys not in the OpenAPI spec: {extra}. "
            f"Metadata has additionalProperties: false — only "
            f"{_ALLOWED_METADATA_KEYS} are allowed."
        )


# =========================================================================
# BUG 3: Summaries passthrough leaks unknown fields
# =========================================================================
# Same pattern as metadata: map_response does
#   summaries_out = ftsearch_body.get("summaries") or {}
# which is a raw passthrough. The OpenAPI Summaries schema has
# additionalProperties: false with 9 specific keys. Any extra key
# ftsearch adds leaks through.


class TestSummariesFieldLeakage:
    """Summaries has additionalProperties: false in the OpenAPI spec.
    Unknown keys from ftsearch should not appear in the legacy response."""

    def test_unknown_summaries_field_stripped(self) -> None:
        """A ftsearch-internal summary type not in the legacy contract
        should NOT leak through."""
        body = _ftsearch_body(summaries={
            "vendorSummaries": [],
            "internalDebugSummary": [{"foo": "bar"}],
        })
        out = map_response(body, explain=False)
        assert "internalDebugSummary" not in out["summaries"], (
            f"Unknown summaries field 'internalDebugSummary' leaked through. "
            f"Summaries has additionalProperties: false in the OpenAPI spec. "
            f"Got: {out['summaries']}"
        )

    def test_only_allowed_summaries_keys_present(self) -> None:
        """Summaries output should contain only the 9 spec-allowed keys."""
        body = _ftsearch_body(summaries={
            "vendorSummaries": [],
            "manufacturerSummaries": [],
            "featureSummaries": [],
            "experimentalSummary": {"x": 1},
            "betaFeature": [],
        })
        out = map_response(body, explain=False)
        extra = set(out["summaries"].keys()) - _ALLOWED_SUMMARIES_KEYS
        assert not extra, (
            f"Summaries contains keys not in the OpenAPI spec: {extra}. "
            f"Summaries has additionalProperties: false — only "
            f"{_ALLOWED_SUMMARIES_KEYS} are allowed."
        )
