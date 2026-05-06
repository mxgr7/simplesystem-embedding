"""Red-team unit tests for acl/mapping/response.py — edge cases.

Exercises malformed, boundary, and unexpected inputs against map_response().
No network calls; pure function testing.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from acl.mapping.response import map_response  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_body(**overrides) -> dict:
    """Minimal valid ftsearch response body."""
    base: dict = {
        "articles": [],
        "summaries": {},
        "metadata": {
            "page": 1,
            "pageSize": 10,
            "pageCount": 1,
            "term": "bolt",
            "hitCount": 0,
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. articleId with fewer or more than 3 parts (passthrough)
# ---------------------------------------------------------------------------

class TestArticleIdPartCount:
    """articleId not matching the 3-part pattern should pass through unchanged."""

    def test_single_part_passthrough(self):
        body = _minimal_body(articles=[{"articleId": "only-one-part", "score": 1.0}])
        out = map_response(body, explain=False)
        assert out["articles"][0]["articleId"] == "only-one-part"

    def test_two_parts_passthrough(self):
        body = _minimal_body(articles=[{"articleId": "part1:part2", "score": 1.0}])
        out = map_response(body, explain=False)
        assert out["articles"][0]["articleId"] == "part1:part2"

    def test_four_parts_passthrough(self):
        body = _minimal_body(articles=[{"articleId": "a:b:c:d", "score": 1.0}])
        out = map_response(body, explain=False)
        assert out["articles"][0]["articleId"] == "a:b:c:d"

    def test_empty_string_passthrough(self):
        body = _minimal_body(articles=[{"articleId": "", "score": 0.0}])
        out = map_response(body, explain=False)
        assert out["articles"][0]["articleId"] == ""


# ---------------------------------------------------------------------------
# 2. articleId with malformed UUID in position 0
# ---------------------------------------------------------------------------

class TestArticleIdMalformedUUID:
    """3-part articleId where part[0] is not a valid UUID passes through unchanged."""

    def test_not_a_uuid_passes_through(self):
        body = _minimal_body(articles=[{"articleId": "not-a-uuid:MTIz:also-bad", "score": 0.5}])
        result = map_response(body, explain=False)
        assert result["articles"][0]["articleId"] == "not-a-uuid:MTIz:also-bad"

    def test_short_hex_passes_through(self):
        body = _minimal_body(articles=[{"articleId": "1234abcd:MTIz:5678efgh", "score": 0.5}])
        result = map_response(body, explain=False)
        assert result["articles"][0]["articleId"] == "1234abcd:MTIz:5678efgh"

    def test_uuid_with_extra_chars_passes_through(self):
        valid = str(uuid.uuid4())
        bad = valid + "ff"
        body = _minimal_body(articles=[{"articleId": f"{bad}:MTIz:{valid}", "score": 0.1}])
        result = map_response(body, explain=False)
        assert result["articles"][0]["articleId"] == f"{bad}:MTIz:{valid}"


# ---------------------------------------------------------------------------
# 3. Empty articles list
# ---------------------------------------------------------------------------

class TestEmptyArticles:
    def test_empty_list_returns_empty(self):
        out = map_response(_minimal_body(articles=[]), explain=False)
        assert out["articles"] == []

    def test_none_articles_treated_as_empty(self):
        body = _minimal_body()
        body["articles"] = None
        out = map_response(body, explain=False)
        assert out["articles"] == []

    def test_missing_articles_key(self):
        body = _minimal_body()
        del body["articles"]
        out = map_response(body, explain=False)
        assert out["articles"] == []


# ---------------------------------------------------------------------------
# 4. Missing summaries/metadata keys
# ---------------------------------------------------------------------------

class TestMissingSummariesMetadata:
    def test_missing_summaries_key(self):
        body = _minimal_body()
        del body["summaries"]
        out = map_response(body, explain=False)
        assert out["summaries"] == {}

    def test_none_summaries(self):
        body = _minimal_body()
        body["summaries"] = None
        out = map_response(body, explain=False)
        assert out["summaries"] == {}

    def test_missing_metadata_key(self):
        body = _minimal_body()
        del body["metadata"]
        out = map_response(body, explain=False)
        assert out["metadata"] == {}

    def test_none_metadata(self):
        body = _minimal_body()
        body["metadata"] = None
        out = map_response(body, explain=False)
        assert out["metadata"] == {}


# ---------------------------------------------------------------------------
# 5. eClassesAggregations output uses {id, count} per legacy spec
# ---------------------------------------------------------------------------

class TestEClassesAggregationsMissingId:
    def test_missing_id_falls_back_to_empty_string(self):
        body = _minimal_body()
        body["summaries"] = {
            "eClassesAggregations": [
                {"count": 5},  # no "id" and no "name"
            ]
        }
        out = map_response(body, explain=False)
        aggs = out["summaries"]["eClassesAggregations"]
        assert aggs == [{"id": "", "count": 5}]

    def test_missing_count_defaults_to_zero(self):
        body = _minimal_body()
        body["summaries"] = {
            "eClassesAggregations": [
                {"id": "27-11-01-01"},  # no "count"
            ]
        }
        out = map_response(body, explain=False)
        aggs = out["summaries"]["eClassesAggregations"]
        assert aggs == [{"id": "27-11-01-01", "count": 0}]

    def test_no_id_key_defaults_to_empty(self):
        """If input has no `id` key, output gets empty string."""
        body = _minimal_body()
        body["summaries"] = {
            "eClassesAggregations": [
                {"name": "fallback-name", "count": 3},
            ]
        }
        out = map_response(body, explain=False)
        aggs = out["summaries"]["eClassesAggregations"]
        assert aggs == [{"id": "", "count": 3}]

    def test_id_preserved_extra_keys_stripped(self):
        body = _minimal_body()
        body["summaries"] = {
            "eClassesAggregations": [
                {"id": "primary", "name": "secondary", "count": 1},
            ]
        }
        out = map_response(body, explain=False)
        aggs = out["summaries"]["eClassesAggregations"]
        assert aggs[0] == {"id": "primary", "count": 1}

    def test_empty_aggregations_list(self):
        body = _minimal_body()
        body["summaries"] = {"eClassesAggregations": []}
        out = map_response(body, explain=False)
        assert out["summaries"]["eClassesAggregations"] == []


# ---------------------------------------------------------------------------
# 6. Very large UUID values (near max 128-bit)
# ---------------------------------------------------------------------------

class TestLargeUUIDs:
    def test_max_uuid(self):
        """UUID with all bits set (ffffffff-ffff-ffff-ffff-ffffffffffff)."""
        max_uuid = "ffffffff-ffff-ffff-ffff-ffffffffffff"
        catalog_uuid = str(uuid.uuid4())
        article_id = f"{max_uuid}:MTIz:{catalog_uuid}"
        body = _minimal_body(articles=[{"articleId": article_id, "score": 0.9}])
        out = map_response(body, explain=False)
        result_id = out["articles"][0]["articleId"]
        # Should be base62-encoded max UUID + ":MTIz"
        assert result_id.endswith(":MTIz")
        friendly_part = result_id.split(":")[0]
        assert len(friendly_part) > 0
        # All chars should be valid base62
        base62_chars = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        assert all(c in base62_chars for c in friendly_part)

    def test_zero_uuid(self):
        """UUID with all bits zero (00000000-0000-0000-0000-000000000000)."""
        zero_uuid = "00000000-0000-0000-0000-000000000000"
        catalog_uuid = str(uuid.uuid4())
        article_id = f"{zero_uuid}:MTIz:{catalog_uuid}"
        body = _minimal_body(articles=[{"articleId": article_id, "score": 0.1}])
        out = map_response(body, explain=False)
        result_id = out["articles"][0]["articleId"]
        assert result_id == "0000000000000000000000:MTIz"

    def test_uuid_v4_typical(self):
        """Standard v4 UUID round-trips without error."""
        v4 = str(uuid.uuid4())
        catalog = str(uuid.uuid4())
        article_id = f"{v4}:YWJj:{catalog}"
        body = _minimal_body(articles=[{"articleId": article_id, "score": 0.5}])
        out = map_response(body, explain=False)
        result_id = out["articles"][0]["articleId"]
        parts = result_id.split(":")
        assert len(parts) == 2
        assert parts[1] == "YWJj"


# ---------------------------------------------------------------------------
# 7. pageCount = 0, negative pageCount
# ---------------------------------------------------------------------------

class TestPageCountClamping:
    def test_page_count_zero_clamped_to_one(self):
        body = _minimal_body()
        body["metadata"]["pageCount"] = 0
        out = map_response(body, explain=False)
        assert out["metadata"]["pageCount"] == 1

    def test_page_count_negative_clamped_to_one(self):
        body = _minimal_body()
        body["metadata"]["pageCount"] = -5
        out = map_response(body, explain=False)
        assert out["metadata"]["pageCount"] == 1

    def test_page_count_one_unchanged(self):
        body = _minimal_body()
        body["metadata"]["pageCount"] = 1
        out = map_response(body, explain=False)
        assert out["metadata"]["pageCount"] == 1

    def test_page_count_large_value_unchanged(self):
        body = _minimal_body()
        body["metadata"]["pageCount"] = 9999
        out = map_response(body, explain=False)
        assert out["metadata"]["pageCount"] == 9999

    def test_page_count_missing_no_clamping_needed(self):
        """If pageCount is absent, the mapper should not inject it."""
        body = _minimal_body()
        del body["metadata"]["pageCount"]
        out = map_response(body, explain=False)
        assert "pageCount" not in out["metadata"]


# ---------------------------------------------------------------------------
# 8. explain=true with empty articles
# ---------------------------------------------------------------------------

class TestExplainWithEmptyArticles:
    def test_explain_true_empty_articles(self):
        out = map_response(_minimal_body(articles=[]), explain=True)
        assert out["articles"] == []

    def test_explain_true_injects_explanation(self):
        v = str(uuid.uuid4())
        body = _minimal_body(articles=[{"articleId": f"{v}:MTIz:{v}", "score": 0.8}])
        out = map_response(body, explain=True)
        assert out["articles"][0]["explanation"] == "N/A"

    def test_explain_false_no_explanation_key(self):
        v = str(uuid.uuid4())
        body = _minimal_body(articles=[{"articleId": f"{v}:MTIz:{v}", "score": 0.8}])
        out = map_response(body, explain=False)
        assert "explanation" not in out["articles"][0]


# ---------------------------------------------------------------------------
# 9. Unexpected extra keys in summaries/metadata (should be stripped)
# ---------------------------------------------------------------------------

class TestExtraKeysStripped:
    def test_extra_summary_keys_stripped(self):
        body = _minimal_body()
        body["summaries"] = {
            "vendorSummaries": [{"vendorId": "x", "count": 1}],
            "bogusKey": "should-not-appear",
            "internalDebug": {"foo": "bar"},
        }
        out = map_response(body, explain=False)
        assert "bogusKey" not in out["summaries"]
        assert "internalDebug" not in out["summaries"]
        assert "vendorSummaries" in out["summaries"]

    def test_extra_metadata_keys_stripped(self):
        body = _minimal_body()
        body["metadata"]["recallClipped"] = True
        body["metadata"]["hitCountClipped"] = True
        body["metadata"]["internalTraceId"] = "abc-123"
        out = map_response(body, explain=False)
        assert "recallClipped" not in out["metadata"]
        assert "hitCountClipped" not in out["metadata"]
        assert "internalTraceId" not in out["metadata"]

    def test_only_allowed_metadata_keys_present(self):
        body = _minimal_body()
        body["metadata"]["extra1"] = 1
        body["metadata"]["extra2"] = "x"
        out = map_response(body, explain=False)
        allowed = {"page", "pageSize", "pageCount", "term", "hitCount"}
        assert set(out["metadata"].keys()).issubset(allowed)

    def test_only_allowed_summary_keys_present(self):
        allowed = {
            "vendorSummaries", "manufacturerSummaries", "featureSummaries",
            "pricesSummary", "categoriesSummary", "eClass5Categories",
            "eClass7Categories", "s2ClassCategories", "eClassesAggregations",
        }
        body = _minimal_body()
        body["summaries"] = {k: [] for k in allowed}
        body["summaries"]["sneakyExtra"] = "nope"
        out = map_response(body, explain=False)
        assert set(out["summaries"].keys()) == allowed


# ---------------------------------------------------------------------------
# 10. articleId already in 2-part format (passthrough)
# ---------------------------------------------------------------------------

class TestTwoPartPassthrough:
    def test_two_part_friendly_id_unchanged(self):
        """Already-legacy format should pass through without modification."""
        legacy_id = "5cXLhMlLbQKWREJb0kzd1w:MTIz"
        body = _minimal_body(articles=[{"articleId": legacy_id, "score": 0.5}])
        out = map_response(body, explain=False)
        assert out["articles"][0]["articleId"] == legacy_id

    def test_two_part_with_complex_b64(self):
        legacy_id = "abc123:YWJjLzEyMy80NTY="
        body = _minimal_body(articles=[{"articleId": legacy_id, "score": 0.5}])
        out = map_response(body, explain=False)
        assert out["articles"][0]["articleId"] == legacy_id
