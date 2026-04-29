"""A4 — legacy error envelope on every error path.

Validates:
  - Pydantic validation failures map to the legacy
    `{message: "Validation failure", details: [...], timestamp: ...}`
    envelope with HTTP 400 (NOT FastAPI's default 422 / `{detail}`).
  - Each §2.1-dropped enum value (`searchArticlesBy: ARTICLE_NUMBER`,
    `EAN`, etc.) returns 400 with a field-level message.
  - Cross-field rule from §3: `priceFilter.currencyCode` required
    when `min` or `max` is set.
  - Generic 5xx → "Internal server error" envelope, no traceback,
    no internal hostnames in the response body.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "acl"))
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from acl.main import app  # noqa: E402


@pytest.fixture
def client():
    """`raise_server_exceptions=False` so the unhandled-exception
    handler under test ACTUALLY runs — by default TestClient re-raises
    Python exceptions instead of letting them flow through the
    ASGI exception-handler chain."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _legacy_error_shape(body: dict) -> bool:
    return set(body.keys()) == {"message", "details", "timestamp"}


def _valid_request() -> dict:
    return {
        "searchMode": "BOTH",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {"closedCatalogVersionIds": []},
        "maxDeliveryTime": 0,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
        "currency": "EUR",
        "explain": False,
    }


# ---- Pydantic validation → legacy envelope ------------------------------

def test_missing_required_field_returns_400_with_legacy_envelope(client) -> None:
    body = _valid_request()
    body.pop("currency")
    r = client.post("/article-features/search", json=body)
    assert r.status_code == 400, f"got {r.status_code}: {r.text}"
    assert _legacy_error_shape(r.json())
    assert r.json()["message"] == "Validation failure"
    assert any("currency" in d for d in r.json()["details"])


def test_unknown_field_returns_400(client) -> None:
    body = _valid_request()
    body["unknownField"] = "junk"
    r = client.post("/article-features/search", json=body)
    assert r.status_code == 400
    assert _legacy_error_shape(r.json())
    assert any("unknownField" in d for d in r.json()["details"])


def test_pagesize_above_500_rejected_with_legacy_envelope(client) -> None:
    r = client.post(
        "/article-features/search",
        params={"page": 1, "pageSize": 501},
        json=_valid_request(),
    )
    assert r.status_code == 400
    assert _legacy_error_shape(r.json())
    assert any("pagesize" in d.lower() for d in r.json()["details"])


def test_currency_pattern_rejected(client) -> None:
    """Lowercase `eur` doesn't match `^[A-Z]{3}$` — must reject."""
    body = _valid_request()
    body["currency"] = "eur"
    r = client.post("/article-features/search", json=body)
    assert r.status_code == 400
    assert _legacy_error_shape(r.json())


def test_invalid_json_body_returns_400(client) -> None:
    r = client.post(
        "/article-features/search",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400
    assert _legacy_error_shape(r.json())


# ---- §2.1 dropped-enum rejection ----------------------------------------
# Spec §2.1 collapsed `searchArticlesBy` to single-value `[STANDARD]`.
# Each legacy value below was a valid enum member — rejecting them now
# is the contract guard that catches stale next-gen clients.

@pytest.mark.parametrize("dropped_value", [
    "ALL_ATTRIBUTES",
    "ARTICLE_NUMBER",
    "CUSTOMER_ARTICLE_NUMBER",
    "VENDOR_ARTICLE_NUMBER",
    "EAN",
    "TEST_PROFILE_01",
    "TEST_PROFILE_20",
])
def test_dropped_search_articles_by_enum_rejected(
    client, dropped_value: str,
) -> None:
    body = _valid_request()
    body["searchArticlesBy"] = dropped_value
    r = client.post("/article-features/search", json=body)
    assert r.status_code == 400, f"{dropped_value} should reject; got {r.status_code}"
    assert _legacy_error_shape(r.json())
    # Field-level path included so callers see WHICH field rejected.
    assert any("searchArticlesBy" in d or "search_articles_by" in d
               for d in r.json()["details"])
    # Message references STANDARD or §2.1.
    msg = " ".join(r.json()["details"])
    assert "STANDARD" in msg or "§2.1" in msg


# ---- §3 cross-field rule ------------------------------------------------

def test_pricefilter_min_set_without_currency_code_rejected(client) -> None:
    body = _valid_request()
    body["priceFilter"] = {"min": 100}
    r = client.post("/article-features/search", json=body)
    assert r.status_code == 400
    assert _legacy_error_shape(r.json())
    msg = " ".join(r.json()["details"])
    assert "currencyCode" in msg or "currency_code" in msg


def test_pricefilter_max_set_without_currency_code_rejected(client) -> None:
    body = _valid_request()
    body["priceFilter"] = {"max": 1000}
    r = client.post("/article-features/search", json=body)
    assert r.status_code == 400


def test_pricefilter_with_only_currency_code_accepted(client) -> None:
    """Bound-decoding currency without bounds is fine — operator
    intent is to pass through but not filter."""
    body = _valid_request()
    body["priceFilter"] = {"currencyCode": "EUR"}
    # Even though ftsearch is unreachable here, the 4xx-vs-5xx is what
    # matters: validation passed (no 400), upstream failure → 503.
    r = client.post("/article-features/search", json=body)
    assert r.status_code != 400, f"got 400 — validation should pass; body: {r.text}"


# ---- 5xx envelope -------------------------------------------------------

def test_unhandled_exception_returns_500_with_no_traceback(client) -> None:
    """Programming bug → generic 500 envelope. The exception message
    + traceback live in the server log; the wire response stays
    bounded to the legacy envelope shape — no leakage of internal
    paths or hostnames."""
    # Inject a raise inside the FtsearchClient so we trip the
    # generic Exception handler.
    class _Boom:
        async def search(self, *args, **kwargs):
            raise RuntimeError("internal hostname=db-01.internal")
        async def aclose(self):
            pass
    original = client.app.state.ftsearch
    client.app.state.ftsearch = _Boom()
    try:
        r = client.post("/article-features/search", json=_valid_request())
    finally:
        client.app.state.ftsearch = original
    assert r.status_code == 500
    body = r.json()
    assert _legacy_error_shape(body)
    assert body["message"] == "Internal server error"
    # No leakage.
    assert "hostname=db-01.internal" not in (body["message"] + " ".join(body["details"]))
    assert "Traceback" not in (body["message"] + " ".join(body["details"]))
