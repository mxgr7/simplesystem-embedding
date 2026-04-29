"""Unit tests for `acl/mapping/response.py`.

A3 contract:
  - Drop `articles[].score`; inject `explanation = "N/A"` only when
    the original request had `explain=true` (per §2.2).
  - Pass through `summaries` unchanged (F2 alignment).
  - Drop ACL-extension `metadata.recallClipped` and `hitCountClipped`
    that aren't in the legacy contract (§3 metadata shape locked).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from acl.mapping.response import map_response  # noqa: E402


def _ftsearch_body(**overrides) -> dict:
    base = {
        "articles": [
            {"articleId": "abc:MTIzNA", "score": 0.95},
            {"articleId": "def:NTY3OA", "score": 0.7},
        ],
        "summaries": {
            "vendorSummaries": [{"vendorId": "v-uuid", "count": 2}],
            "manufacturerSummaries": [{"name": "Acme", "count": 2}],
        },
        "metadata": {
            "page": 1, "pageSize": 10, "pageCount": 1,
            "term": "schraube", "hitCount": 2,
            "recallClipped": False,
            "hitCountClipped": False,
        },
    }
    base.update(overrides)
    return base


def test_score_dropped_from_articles() -> None:
    out = map_response(_ftsearch_body(), explain=False)
    for art in out["articles"]:
        assert "score" not in art


def test_explanation_omitted_when_explain_false() -> None:
    out = map_response(_ftsearch_body(), explain=False)
    for art in out["articles"]:
        assert "explanation" not in art


def test_explanation_stub_when_explain_true() -> None:
    """§2.2 — `explain=true` returns the literal "N/A" instead of a
    real scoring breakdown."""
    out = map_response(_ftsearch_body(), explain=True)
    for art in out["articles"]:
        assert art["explanation"] == "N/A"


def test_articleId_preserved() -> None:
    """Wire format `{friendlyId}:{base64Url(articleNumber)}` matches
    legacy exactly — downstream code parses it."""
    out = map_response(_ftsearch_body(), explain=False)
    assert out["articles"][0]["articleId"] == "abc:MTIzNA"
    assert out["articles"][1]["articleId"] == "def:NTY3OA"


def test_summaries_pass_through_unchanged() -> None:
    out = map_response(_ftsearch_body(), explain=False)
    assert out["summaries"]["vendorSummaries"] == [{"vendorId": "v-uuid", "count": 2}]
    assert out["summaries"]["manufacturerSummaries"] == [{"name": "Acme", "count": 2}]


def test_metadata_drops_recall_and_hitcount_clipped() -> None:
    out = map_response(_ftsearch_body(), explain=False)
    md = out["metadata"]
    assert "recallClipped" not in md
    assert "hitCountClipped" not in md
    # Legacy fields preserved.
    assert md["hitCount"] == 2
    assert md["pageCount"] == 1


def test_empty_articles_summaries_only_mode() -> None:
    """SUMMARIES_ONLY mode: ftsearch returns empty articles[] with
    populated summaries. Mapper preserves that shape."""
    body = _ftsearch_body(articles=[])
    out = map_response(body, explain=False)
    assert out["articles"] == []
    assert "vendorSummaries" in out["summaries"]


def test_missing_summaries_yields_empty_dict() -> None:
    """ftsearch may omit summaries entirely (HITS_ONLY mode); map to
    `{}` rather than missing key so the legacy schema's `summaries`
    field is always present."""
    body = _ftsearch_body()
    body.pop("summaries")
    out = map_response(body, explain=False)
    assert out["summaries"] == {}


def test_mapper_is_pure() -> None:
    body = _ftsearch_body()
    a = map_response(body, explain=True)
    b = map_response(body, explain=True)
    assert a == b
