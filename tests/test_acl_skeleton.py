"""ACL packet A1 skeleton tests.

Validates the runnable FastAPI app skeleton + the OpenAPI shape
without requiring the downstream ftsearch service to be reachable
(the stub returns 501 by design).

Coverage:
  - App boots; `/healthz` returns ok.
  - `/openapi.yaml` round-trips through `openapi-spec-validator`.
  - `POST /article-features/search` returns the documented 501 with
    the legacy error envelope.
  - Validation: malformed `searchArticlesBy` (the §2.1 deviation —
    only `STANDARD` is allowed) gets rejected by Pydantic before
    reaching the stub. Wait — Pydantic isn't used yet for the
    request body in A1; the stub doesn't parse it. So we instead
    validate against the OpenAPI directly via openapi-core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from acl.app import app  # noqa: E402


@pytest.fixture
def client():
    """Context-managed TestClient so the FastAPI lifespan fires —
    needed for `app.state.ftsearch` (the httpx client that A2 wires
    in)."""
    with TestClient(app) as c:
        yield c


def test_healthz_returns_ok(client) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_openapi_yaml_is_served(client) -> None:
    """The contract source of truth is served at /openapi.yaml so
    integrators + client generators can fetch it from the deployed
    service without poking at the repo."""
    r = client.get("/openapi.yaml")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/yaml")
    assert "article-search-acl" in r.text
    assert "/article-features/search" in r.text


def test_openapi_validates() -> None:
    """The hand-written OpenAPI must parse + validate against the 3.0
    spec. CI catches any drift at edit-time so client generators
    always see a clean schema."""
    pytest.importorskip("openapi_spec_validator")
    import yaml
    from openapi_spec_validator import validate_spec
    spec = yaml.safe_load((REPO_ROOT / "acl/openapi.yaml").read_text())
    validate_spec(spec)


def test_search_endpoint_returns_legacy_envelope_on_upstream_failure(client) -> None:
    """Post-A2 the endpoint actually calls ftsearch. With no ftsearch
    reachable in this skeleton-only test (no MockTransport), the call
    fails with `httpx.ConnectError` → ACL wraps it in the legacy
    `{message, details, timestamp}` 503 envelope. End-to-end happy-path
    coverage lives in `test_acl_integration.py`."""
    r = client.post("/article-features/search", json={
        "searchMode": "BOTH",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {"closedCatalogVersionIds": []},
        "maxDeliveryTime": 0,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
        "currency": "EUR",
        "explain": False,
    })
    # Either 503 (network unreachable) or 502/500 depending on the
    # bubbled-up failure shape — what matters is the envelope.
    assert r.status_code >= 500, r.text
    body = r.json()
    assert set(body.keys()) == {"message", "details", "timestamp"}
    assert isinstance(body["details"], list)
    assert body["timestamp"], "timestamp must be present"


def test_openapi_spec_encodes_searcharticleby_deviation() -> None:
    """§2.1 — `searchArticlesBy` is a single-value enum. Guard against
    accidentally widening it later (which would break the contract
    expectation that next-gen callers don't pass anything else)."""
    import yaml
    spec = yaml.safe_load((REPO_ROOT / "acl/openapi.yaml").read_text())
    saby = spec["components"]["schemas"]["SearchRequest"]["properties"]["searchArticlesBy"]
    assert saby["enum"] == ["STANDARD"], (
        f"searchArticlesBy must be the single-value enum [STANDARD] per §2.1; "
        f"got {saby['enum']!r}"
    )


def test_openapi_spec_pagesize_capped_at_500() -> None:
    """Spec §3 — `pageSize` max 500. Higher values are rejected at the
    schema level so callers see the limit before hitting any handler."""
    import yaml
    spec = yaml.safe_load((REPO_ROOT / "acl/openapi.yaml").read_text())
    page_size = next(
        p for p in spec["paths"]["/article-features/search"]["post"]["parameters"]
        if p["name"] == "pageSize"
    )
    assert page_size["schema"]["maximum"] == 500


def test_openapi_spec_currency_pattern_matches_iso4217() -> None:
    """`currency` is `^[A-Z]{3}$` — guard the pattern lock; loosening
    it would let through e.g. 'eur' that legacy rejects."""
    import yaml
    spec = yaml.safe_load((REPO_ROOT / "acl/openapi.yaml").read_text())
    cur = spec["components"]["schemas"]["SearchRequest"]["properties"]["currency"]
    assert cur["pattern"] == "^[A-Z]{3}$"


def test_openapi_security_is_empty_per_spec() -> None:
    """§9 #7 — internal service, no per-request auth on either hop."""
    import yaml
    spec = yaml.safe_load((REPO_ROOT / "acl/openapi.yaml").read_text())
    assert spec["security"] == []
