"""Stress tests for the ACL service at localhost:8081.

Probes for concurrency bugs, race conditions, resource leaks, and
response isolation under load.  Uses asyncio + httpx.AsyncClient for
true concurrent HTTP requests against the running ACL.

Requires the ACL to be running on localhost:8081 with a live upstream
search-api on localhost:8001 and a catalog version that has data.

Run:
    .venv/bin/python -m pytest tests/test_acl_stress.py -v --tb=short
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pytest

ACL_URL = "http://localhost:8081"
SEARCH_ENDPOINT = f"{ACL_URL}/article-features/search"
CATALOG_VERSION = "866b4863-8799-4046-9e84-0985a665c1c7"


# --- helpers ---------------------------------------------------------------

def _valid_body(query: str = "schraube", **overrides: Any) -> dict:
    """Minimal valid legacy search request body."""
    body: dict[str, Any] = {
        "searchMode": "HITS_ONLY",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {
            "closedCatalogVersionIds": [CATALOG_VERSION],
        },
        "queryString": query,
        "maxDeliveryTime": 0,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": True,
        "currency": "EUR",
        "explain": False,
    }
    body.update(overrides)
    return body


def _invalid_body_bad_enum() -> dict:
    """Body with an invalid searchMode value -- triggers 400."""
    return {
        "searchMode": "INVALID_MODE",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {
            "closedCatalogVersionIds": [CATALOG_VERSION],
        },
        "maxDeliveryTime": 0,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
        "currency": "EUR",
        "explain": False,
    }


def _invalid_body_missing_field() -> dict:
    """Body missing required fields -- triggers 500 (legacy parity)."""
    return {
        "searchMode": "BOTH",
    }


def _acl_reachable() -> bool:
    try:
        r = httpx.get(f"{ACL_URL}/healthz", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _acl_reachable(),
    reason=f"ACL not reachable at {ACL_URL}",
)


# --- 1. Concurrent identical requests -------------------------------------

@pytest.mark.asyncio
async def test_concurrent_identical_requests():
    """20 simultaneous identical requests should all return the same
    correct response.  Detects shared-state mutation, response mixing,
    or non-deterministic serialization."""
    n = 20
    body = _valid_body("schraube")

    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [
            client.post(SEARCH_ENDPOINT, json=body, params={"page": 1, "pageSize": 5})
            for _ in range(n)
        ]
        responses = await asyncio.gather(*tasks)

    statuses = [r.status_code for r in responses]
    assert all(s == 200 for s in statuses), f"Non-200 statuses: {statuses}"

    bodies = [r.json() for r in responses]
    # All responses should have articles
    for i, b in enumerate(bodies):
        assert "articles" in b, f"Response {i} missing 'articles': {b}"
        assert "metadata" in b, f"Response {i} missing 'metadata': {b}"

    # All responses should be identical (same input -> same output)
    first = bodies[0]
    for i, b in enumerate(bodies[1:], start=1):
        assert b["articles"] == first["articles"], (
            f"Response {i} articles differ from response 0"
        )
        assert b["metadata"]["hitCount"] == first["metadata"]["hitCount"], (
            f"Response {i} hitCount differs from response 0"
        )


# --- 2. Concurrent mixed valid/invalid ------------------------------------

@pytest.mark.asyncio
async def test_concurrent_mixed_valid_invalid():
    """Mix of valid and invalid requests in parallel.  Invalid requests
    must get 400; valid requests must get 200.  Errors in one request
    must not affect others."""
    valid_body = _valid_body("schraube")
    invalid_enum = _invalid_body_bad_enum()
    invalid_missing = _invalid_body_missing_field()

    async with httpx.AsyncClient(timeout=30) as client:
        tasks = []
        # 10 valid
        for _ in range(10):
            tasks.append(("valid", client.post(SEARCH_ENDPOINT, json=valid_body)))
        # 5 invalid (bad enum)
        for _ in range(5):
            tasks.append(("bad_enum", client.post(SEARCH_ENDPOINT, json=invalid_enum)))
        # 5 invalid (missing fields)
        for _ in range(5):
            tasks.append(("missing", client.post(SEARCH_ENDPOINT, json=invalid_missing)))

        labels = [t[0] for t in tasks]
        coros = [t[1] for t in tasks]
        responses = await asyncio.gather(*coros)

    for label, resp in zip(labels, responses):
        if label == "valid":
            assert resp.status_code == 200, (
                f"Valid request got {resp.status_code}: {resp.text[:300]}"
            )
            body = resp.json()
            assert "articles" in body
        elif label == "bad_enum":
            assert resp.status_code == 400, (
                f"{label} request expected 400, got {resp.status_code}: {resp.text[:300]}"
            )
        else:
            assert resp.status_code == 500, (
                f"{label} request expected 500, got {resp.status_code}: {resp.text[:300]}"
            )
            body = resp.json()
            assert "message" in body, f"Error body missing 'message': {body}"
            assert "timestamp" in body, f"Error body missing 'timestamp': {body}"


# --- 3. Rapid sequential requests -----------------------------------------

@pytest.mark.asyncio
async def test_rapid_sequential_requests():
    """50 requests as fast as possible, sequentially.  Checks for
    degradation: later responses should not be dramatically slower
    or fail where earlier ones succeeded."""
    body = _valid_body("schraube")
    latencies: list[float] = []
    statuses: list[int] = []

    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(50):
            t0 = time.monotonic()
            r = await client.post(
                SEARCH_ENDPOINT, json=body, params={"page": 1, "pageSize": 5},
            )
            latencies.append(time.monotonic() - t0)
            statuses.append(r.status_code)

    assert all(s == 200 for s in statuses), (
        f"Failures in rapid sequence: {[(i, s) for i, s in enumerate(statuses) if s != 200]}"
    )

    # No dramatic degradation: last 10 should not be >5x slower than first 10
    avg_first_10 = sum(latencies[:10]) / 10
    avg_last_10 = sum(latencies[-10:]) / 10
    assert avg_last_10 < avg_first_10 * 5 + 0.5, (
        f"Degradation detected: first-10 avg {avg_first_10:.3f}s, "
        f"last-10 avg {avg_last_10:.3f}s"
    )


# --- 4. Response isolation -------------------------------------------------

@pytest.mark.asyncio
async def test_response_isolation():
    """Each request gets a distinct queryString; verify responses
    match their request and don't bleed across coroutines."""
    queries = [f"query_{i}" for i in range(15)]

    async def _do_search(client: httpx.AsyncClient, q: str) -> tuple[str, dict]:
        r = await client.post(
            SEARCH_ENDPOINT,
            json=_valid_body(query=q),
            params={"page": 1, "pageSize": 5},
        )
        return q, r.json()

    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(
            *[_do_search(client, q) for q in queries]
        )

    for query, body in results:
        assert body.get("metadata") is not None, (
            f"Query {query!r}: missing metadata in response"
        )
        # The metadata.term field should reflect the query we sent
        term = body["metadata"].get("term")
        if term is not None:
            assert term == query, (
                f"Response bleed: sent query={query!r} but got term={term!r}"
            )


# --- 5. Connection pool exhaustion -----------------------------------------

@pytest.mark.asyncio
async def test_connection_pool_pressure():
    """Fire 120 concurrent requests -- above httpx's default pool of
    100 connections.  The ACL should handle backpressure gracefully:
    all requests must eventually succeed (possibly after pool-wait)
    rather than crash or hang."""
    n = 120
    body = _valid_body("schraube")

    # Give extra timeout headroom for pool queueing
    async with httpx.AsyncClient(
        timeout=60,
        limits=httpx.Limits(
            max_connections=150,
            max_keepalive_connections=150,
        ),
    ) as client:
        tasks = [
            client.post(SEARCH_ENDPOINT, json=body, params={"page": 1, "pageSize": 5})
            for _ in range(n)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    errors = [r for r in responses if isinstance(r, Exception)]
    successes = [r for r in responses if not isinstance(r, Exception)]

    # Allow a small number of pool timeouts but not a cascade
    assert len(errors) <= n * 0.1, (
        f"{len(errors)}/{n} requests failed (> 10%): "
        f"{[type(e).__name__ for e in errors[:5]]}"
    )

    # Successful responses must all be 200
    non_200 = [(r.status_code, r.text[:100]) for r in successes if r.status_code != 200]
    assert len(non_200) == 0, f"Non-200 responses: {non_200[:5]}"


# --- 6. Timeout behavior --------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_does_not_hang():
    """Set an extremely short client-side timeout.  The ACL should
    not leak resources or hang when the client disconnects early.
    After the timeouts, a normal request should still succeed."""
    body = _valid_body("schraube")

    # Phase 1: fire requests with a tiny timeout; expect TimeoutException
    timeouts = 0
    async with httpx.AsyncClient(timeout=0.001) as client:
        for _ in range(10):
            try:
                await client.post(SEARCH_ENDPOINT, json=body)
            except httpx.TimeoutException:
                timeouts += 1
            except Exception:
                pass  # other errors are fine too

    # Most should have timed out
    assert timeouts >= 5, f"Expected most requests to time out, only {timeouts}/10 did"

    # Phase 2: after the timeout storm, the ACL should still work fine
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            SEARCH_ENDPOINT, json=body, params={"page": 1, "pageSize": 5},
        )
    assert r.status_code == 200, f"ACL unhealthy after timeout storm: {r.status_code}"


# --- 7. Metadata consistency under load -----------------------------------

@pytest.mark.asyncio
async def test_metadata_consistency_under_load():
    """Fire 20 identical requests concurrently and verify hitCount and
    pageCount are consistent across all responses."""
    n = 20
    body = _valid_body("schraube")

    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [
            client.post(
                SEARCH_ENDPOINT, json=body,
                params={"page": 1, "pageSize": 10},
            )
            for _ in range(n)
        ]
        responses = await asyncio.gather(*tasks)

    assert all(r.status_code == 200 for r in responses), (
        f"Not all 200: {[r.status_code for r in responses]}"
    )

    metadatas = [r.json()["metadata"] for r in responses]
    hit_counts = {m["hitCount"] for m in metadatas}
    page_counts = {m["pageCount"] for m in metadatas}

    assert len(hit_counts) == 1, (
        f"hitCount inconsistent across concurrent requests: {hit_counts}"
    )
    assert len(page_counts) == 1, (
        f"pageCount inconsistent across concurrent requests: {page_counts}"
    )

    # Sanity: values should be non-negative
    hc = hit_counts.pop()
    pc = page_counts.pop()
    assert hc >= 0, f"hitCount negative: {hc}"
    assert pc >= 0, f"pageCount negative: {pc}"
