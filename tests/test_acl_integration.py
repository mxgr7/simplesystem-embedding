"""End-to-end ACL test — POST `/article-features/search` flows through
the request mapper and a stub ftsearch (mocked via `httpx.MockTransport`).

Validates the full A2 wiring:
  - The handler accepts a fully-populated legacy request body.
  - The mapper produces the expected ftsearch wire shape.
  - The httpx call hits the stub with the right URL + body + params.
  - The response forwards through to the caller.
  - Upstream errors (ftsearch returns 5xx, ftsearch unreachable)
    surface as the legacy error envelope on the way back out.

Real ftsearch is exercised by the search-api's own integration suite —
this test deliberately mocks it so the ACL behaviour is the only
moving part.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "acl"))
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from acl.clients.ftsearch import FtsearchClient  # noqa: E402
from acl.main import app  # noqa: E402


def _request_body() -> dict:
    return {
        "searchMode": "BOTH",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {
            "closedCatalogVersionIds": ["aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"],
        },
        "queryString": "schraube",
        "maxDeliveryTime": 5,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
        "currency": "EUR",
        "explain": True,
        "summaries": ["VENDORS", "MANUFACTURERS"],
    }


def _stub_ftsearch_response() -> dict:
    return {
        "articles": [
            {"articleId": "abcdefg:MTIzNA", "score": 0.9},
            {"articleId": "abcdefg:NTY3OA", "score": 0.8},
        ],
        "summaries": {},
        "metadata": {"page": 1, "pageSize": 10, "pageCount": 1, "term": "schraube", "hitCount": 2},
    }


@pytest.fixture
def stub_ftsearch_request_handler():
    """Captures the httpx Request the ACL sends to ftsearch so the
    test can assert on URL + body + headers."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        captured["url"] = str(req.url)
        captured["method"] = req.method
        captured["body"] = json.loads(req.content) if req.content else None
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json=_stub_ftsearch_response())
    return handler, captured


@pytest.fixture
def client_with_stub_ftsearch(stub_ftsearch_request_handler):
    handler, _captured = stub_ftsearch_request_handler
    # Replace the lifespan-built FtsearchClient with one wired to
    # a MockTransport so no real HTTP happens.
    with TestClient(app) as c:
        c.app.state.ftsearch._client = httpx.AsyncClient(  # type: ignore[attr-defined]
            transport=httpx.MockTransport(handler),
        )
        yield c


# ---------- happy path ---------------------------------------------------

def test_full_round_trip(client_with_stub_ftsearch, stub_ftsearch_request_handler) -> None:
    """Happy path: legacy request → mapper → ftsearch (mocked) →
    response mapper → legacy envelope on the way back."""
    client = client_with_stub_ftsearch
    _handler, captured = stub_ftsearch_request_handler

    r = client.post(
        "/article-features/search",
        params={"page": 1, "pageSize": 10},
        json=_request_body(),
    )
    assert r.status_code == 200, r.text

    # Response in legacy shape — A3 mapping applied.
    body = r.json()
    assert len(body["articles"]) == 2
    assert body["articles"][0]["articleId"] == "abcdefg:MTIzNA"
    # `explain=true` in the request → `explanation = "N/A"` per §2.2.
    assert body["articles"][0]["explanation"] == "N/A"
    # `score` from ftsearch is dropped on the way out (legacy doesn't carry).
    assert "score" not in body["articles"][0]
    assert body["metadata"]["hitCount"] == 2

    # ftsearch was called correctly.
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/articles/_search?page=1&pageSize=10")
    sent_body = captured["body"]
    # Renames + drops applied (request side, A2).
    assert sent_body["query"] == "schraube"
    assert "queryString" not in sent_body
    assert "searchArticlesBy" not in sent_body
    assert "explain" not in sent_body
    # Fields preserved.
    assert sent_body["currency"] == "EUR"
    assert sent_body["summaries"] == ["VENDORS", "MANUFACTURERS"]


def test_explain_false_omits_explanation_in_response(
    client_with_stub_ftsearch,
) -> None:
    client = client_with_stub_ftsearch
    body = _request_body()
    body["explain"] = False
    r = client.post(
        "/article-features/search",
        params={"page": 1, "pageSize": 10},
        json=body,
    )
    assert r.status_code == 200
    for art in r.json()["articles"]:
        assert "explanation" not in art


def test_pagination_and_sort_query_params_forwarded(
    client_with_stub_ftsearch, stub_ftsearch_request_handler,
) -> None:
    client = client_with_stub_ftsearch
    _handler, captured = stub_ftsearch_request_handler
    r = client.post(
        "/article-features/search",
        params=[("page", "3"), ("pageSize", "25"), ("sort", "name,asc"), ("sort", "articleId,desc")],
        json=_request_body(),
    )
    assert r.status_code == 200
    assert "page=3" in captured["url"]
    assert "pageSize=25" in captured["url"]
    # httpx url params is a list-friendly mapping; sort=... appears twice.
    assert captured["url"].count("sort=") == 2


# ---------- validation ---------------------------------------------------

def test_search_articles_by_other_than_standard_rejected(
    client_with_stub_ftsearch,
) -> None:
    """§2.1 — only `STANDARD` is allowed. ARTICLE_NUMBER (which the
    legacy enum had) must reject before any ftsearch call is made."""
    client = client_with_stub_ftsearch
    body = _request_body()
    body["searchArticlesBy"] = "ARTICLE_NUMBER"
    r = client.post(
        "/article-features/search", params={"page": 1, "pageSize": 10}, json=body,
    )
    # A4 reshaped Pydantic 422 into the legacy 400 envelope.
    assert r.status_code == 400


def test_unknown_field_rejected(client_with_stub_ftsearch) -> None:
    """`extra='forbid'` so a typo or dropped field surfaces clearly
    to the next-gen caller."""
    client = client_with_stub_ftsearch
    body = _request_body()
    body["unknownNewField"] = "junk"
    r = client.post(
        "/article-features/search", params={"page": 1, "pageSize": 10}, json=body,
    )
    # A4 reshaped Pydantic 422 into the legacy 400 envelope.
    assert r.status_code == 400


def test_pagesize_above_500_rejected(client_with_stub_ftsearch) -> None:
    client = client_with_stub_ftsearch
    r = client.post(
        "/article-features/search",
        params={"page": 1, "pageSize": 501},
        json=_request_body(),
    )
    # A4 reshaped Pydantic 422 into the legacy 400 envelope.
    assert r.status_code == 400


# ---------- upstream error handling --------------------------------------

def test_upstream_5xx_wrapped_in_legacy_envelope() -> None:
    """ftsearch returning 503 → ACL forwards the status with the
    legacy `{message, details, timestamp}` envelope."""
    def fail_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overload")

    with TestClient(app) as c:
        c.app.state.ftsearch._client = httpx.AsyncClient(  # type: ignore[attr-defined]
            transport=httpx.MockTransport(fail_handler),
        )
        r = c.post(
            "/article-features/search",
            params={"page": 1, "pageSize": 10},
            json=_request_body(),
        )
    assert r.status_code == 503
    body = r.json()
    assert set(body.keys()) == {"message", "details", "timestamp"}
    assert "non-2xx" in body["message"]


def test_network_failure_wrapped_in_legacy_envelope() -> None:
    """ftsearch unreachable → 503 with a legacy envelope, no Python
    traceback leaking to the caller."""
    def boom(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with TestClient(app) as c:
        c.app.state.ftsearch._client = httpx.AsyncClient(  # type: ignore[attr-defined]
            transport=httpx.MockTransport(boom),
        )
        r = c.post(
            "/article-features/search",
            params={"page": 1, "pageSize": 10},
            json=_request_body(),
        )
    assert r.status_code == 503
    body = r.json()
    assert "unreachable" in body["message"]
    assert "ConnectError" in body["details"][0]
