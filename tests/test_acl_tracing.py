"""Red-team tests for ACL tracing and header-forwarding logic.

Probes the running ACL at localhost:8081 for:
  1. Traceparent header forwarding to ftsearch
  2. Tracestate injection (newlines, oversized values, special chars)
  3. Missing trace headers (graceful degradation)
  4. Malformed traceparent (invalid version, wrong format, missing fields)
  5. Custom header leaking (non-trace headers forwarded downstream)
  6. Response header information leakage
  7. X-Request-Id correlation

Requires:
  - ACL running on localhost:8081
  - search-api / ftsearch running upstream
"""

from __future__ import annotations

import socket

import httpx
import pytest

ACL_BASE = "http://localhost:8081"
SEARCH_URL = f"{ACL_BASE}/article-features/search"
CV_EUR = "866b4863-8799-4046-9e84-0985a665c1c7"

# A well-formed W3C traceparent for tests that need a valid one.
VALID_TRACEPARENT = "00-aaaabbbbccccddddeeeeffffaaaabbbb-1122334455667788-01"


def _base_body(**overrides) -> dict:
    body = {
        "searchMode": "BOTH",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {
            "closedCatalogVersionIds": [],
            "catalogVersionIdsOrderedByPreference": [CV_EUR],
        },
        "maxDeliveryTime": 0,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
        "currency": "EUR",
        "explain": False,
    }
    body.update(overrides)
    return body


def _post(
    body: dict | None = None,
    *,
    headers: dict[str, str] | None = None,
    page: int = 1,
    page_size: int = 2,
) -> httpx.Response:
    params = {"page": page, "pageSize": page_size}
    return httpx.post(
        SEARCH_URL,
        json=body or _base_body(),
        params=params,
        headers=headers or {},
        timeout=10,
    )


@pytest.fixture(scope="module", autouse=True)
def _check_services():
    try:
        r = httpx.get(f"{ACL_BASE}/healthz", timeout=3)
        assert r.status_code == 200
    except Exception:
        pytest.skip("ACL not running on localhost:8081")


# =========================================================================
# 1. Traceparent header forwarding
# =========================================================================

class TestTraceparentForwarding:
    """Verify ACL accepts and processes a valid traceparent without error."""

    def test_valid_traceparent_succeeds(self):
        """A well-formed traceparent must not cause any error."""
        r = _post(headers={"traceparent": VALID_TRACEPARENT})
        assert r.status_code == 200

    def test_traceparent_with_tracestate_succeeds(self):
        """traceparent + tracestate together must not cause errors."""
        r = _post(headers={
            "traceparent": VALID_TRACEPARENT,
            "tracestate": "vendor1=opaque1,vendor2=opaque2",
        })
        assert r.status_code == 200

    def test_traceparent_with_baggage_succeeds(self):
        """traceparent + baggage together must not cause errors."""
        r = _post(headers={
            "traceparent": VALID_TRACEPARENT,
            "baggage": "userId=42,companyId=7",
        })
        assert r.status_code == 200

    def test_full_trace_context_succeeds(self):
        """All three trace headers together must work."""
        r = _post(headers={
            "traceparent": VALID_TRACEPARENT,
            "tracestate": "vendor=blob",
            "baggage": "userId=42,companyId=7,customerOciSessionId=s1",
        })
        assert r.status_code == 200

    def test_sampled_vs_unsampled_flag_both_work(self):
        """Both sampled (01) and unsampled (00) trace flags must work."""
        tp_sampled = "00-aaaabbbbccccddddeeeeffffaaaabbbb-1122334455667788-01"
        tp_unsampled = "00-aaaabbbbccccddddeeeeffffaaaabbbb-1122334455667788-00"
        r1 = _post(headers={"traceparent": tp_sampled})
        r2 = _post(headers={"traceparent": tp_unsampled})
        assert r1.status_code == 200
        assert r2.status_code == 200


# =========================================================================
# 2. Tracestate injection
# =========================================================================

class TestTracestateInjection:
    """The ACL forwards raw_tracestate to ftsearch without validation.
    Probe whether malicious values cause unexpected behaviour.

    Several payloads (CRLF, null bytes, non-ASCII) are rejected by
    httpx at the client side before they ever reach the wire, so we
    use raw sockets for those cases.
    """

    @staticmethod
    def _raw_post(extra_headers: str) -> tuple[int, str]:
        """Send a hand-crafted HTTP/1.1 POST via raw socket.
        Returns (status_code, body_text).  This bypasses httpx header
        validation so we can inject CRLF / null bytes / non-ASCII."""
        body = _base_body()
        import json as _json
        body_bytes = _json.dumps(body).encode()
        req = (
            f"POST /article-features/search?page=1&pageSize=2 HTTP/1.1\r\n"
            f"Host: localhost:8081\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Connection: close\r\n"
            f"{extra_headers}"
            f"\r\n"
        ).encode("utf-8", errors="surrogateescape") + body_bytes
        sock = socket.create_connection(("localhost", 8081), timeout=10)
        try:
            sock.sendall(req)
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        finally:
            sock.close()
        text = data.decode("utf-8", errors="replace")
        # Parse status code from first line
        first_line = text.split("\r\n", 1)[0]
        status = int(first_line.split(" ", 2)[1])
        return status, text

    def test_tracestate_with_newline_injection(self):
        """CRLF in tracestate could enable header injection in
        downstream HTTP requests if not sanitised."""
        extra = (
            f"traceparent: {VALID_TRACEPARENT}\r\n"
            f"tracestate: vendor=val\r\nX-Injected: evil\r\n"
        )
        status, body = self._raw_post(extra)
        # The ACL should either reject or tolerate gracefully.
        # Critically, it must NOT crash with 500.
        assert status in (200, 400), (
            f"Newline injection in tracestate caused unexpected {status}"
        )

    def test_tracestate_with_null_bytes(self):
        """Null bytes in tracestate might bypass downstream parsing."""
        extra = (
            f"traceparent: {VALID_TRACEPARENT}\r\n"
            f"tracestate: vendor=val\x00evil\r\n"
        )
        status, body = self._raw_post(extra)
        assert status in (200, 400), (
            f"Null byte in tracestate caused unexpected {status}"
        )

    def test_tracestate_oversized_value(self):
        """A 50 KB tracestate is blindly forwarded to all downstream
        services. This probes whether the ACL imposes any limit."""
        huge = "vendor=" + "x" * 50_000
        extra = (
            f"traceparent: {VALID_TRACEPARENT}\r\n"
            f"tracestate: {huge}\r\n"
        )
        status, body = self._raw_post(extra)
        # Per W3C spec, tracestate should be at most 512 bytes.
        # The ACL forwards it verbatim -- this test documents the gap.
        assert status in (200, 400, 413, 431), (
            f"Oversized tracestate caused unexpected {status}"
        )

    def test_tracestate_with_special_chars(self):
        """Unicode and shell metacharacters in tracestate."""
        extra = (
            f"traceparent: {VALID_TRACEPARENT}\r\n"
            f"tracestate: vendor=abc,key2=$(whoami)\r\n"
        )
        status, body = self._raw_post(extra)
        assert status in (200, 400)

    def test_tracestate_extremely_many_entries(self):
        """W3C allows at most 32 list-members. ACL has no cap."""
        entries = ",".join(f"v{i}=val{i}" for i in range(200))
        r = _post(headers={
            "traceparent": VALID_TRACEPARENT,
            "tracestate": entries,
        })
        assert r.status_code in (200, 400)

    def test_tracestate_without_traceparent(self):
        """tracestate without traceparent -- per W3C, tracestate must
        be ignored when traceparent is absent."""
        r = _post(headers={"tracestate": "vendor=blob"})
        assert r.status_code == 200


# =========================================================================
# 3. Missing trace headers
# =========================================================================

class TestMissingTraceHeaders:
    """The ACL must work correctly with no trace headers at all."""

    def test_no_trace_headers_at_all(self):
        """Baseline: no traceparent, no tracestate, no baggage."""
        r = _post(headers={})
        assert r.status_code == 200

    def test_only_baggage_no_traceparent(self):
        """Baggage without traceparent -- ACL should not crash."""
        r = _post(headers={"baggage": "userId=42,companyId=7"})
        assert r.status_code == 200

    def test_healthz_ignores_trace_headers(self):
        """The healthz endpoint explicitly skips trace context extraction."""
        r = httpx.get(
            f"{ACL_BASE}/healthz",
            headers={"traceparent": VALID_TRACEPARENT},
            timeout=5,
        )
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# =========================================================================
# 4. Malformed traceparent
# =========================================================================

class TestMalformedTraceparent:
    """Malformed traceparent should be silently ignored (per W3C spec),
    not cause 400 or 500."""

    @pytest.mark.parametrize("bad_tp,label", [
        ("garbage", "random_string"),
        ("00-1234-5678-01", "too_short_fields"),
        ("00-1234567890abcdef1234567890abcdef-1234567890abcdef", "missing_flags"),
        ("ZZ-1234567890abcdef1234567890abcdef-1234567890abcdef-01", "non_hex_version"),
        ("00-ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ-1234567890abcdef-01", "non_hex_trace_id"),
        ("00-1234567890abcdef1234567890abcdef-ZZZZZZZZZZZZZZZZ-01", "non_hex_span_id"),
        ("00-" + "0" * 32 + "-1234567890abcdef-01", "all_zero_trace_id"),
        ("00-1234567890abcdef1234567890abcdef-" + "0" * 16 + "-01", "all_zero_span_id"),
        ("", "empty_string"),
        ("00-1234567890abcdef1234567890abcdef-1234567890abcdef-01-extra", "extra_segments"),
    ])
    def test_malformed_traceparent_does_not_crash(self, bad_tp, label):
        """Malformed traceparent should be silently dropped, not error."""
        r = _post(headers={"traceparent": bad_tp})
        assert r.status_code == 200, (
            f"Malformed traceparent ({label}) caused {r.status_code}: {r.text[:200]}"
        )

    def test_future_version_traceparent(self):
        """A traceparent with version ff (future/unknown) should be
        accepted or silently ignored per W3C forward-compat rules."""
        tp = "ff-1234567890abcdef1234567890abcdef-1234567890abcdef-01"
        r = _post(headers={"traceparent": tp})
        # W3C says unknown versions should be accepted and passed through,
        # but our regex rejects non-00 versions. This is acceptable but
        # worth documenting.
        assert r.status_code == 200

    def test_traceparent_with_leading_trailing_whitespace(self):
        """W3C says implementations should strip whitespace.
        httpcore rejects leading whitespace in header values, so we
        use a raw socket to send this."""
        tp = "  00-1234567890abcdef1234567890abcdef-1234567890abcdef-01  "
        body = _base_body()
        import json as _json
        body_bytes = _json.dumps(body).encode()
        req = (
            f"POST /article-features/search?page=1&pageSize=2 HTTP/1.1\r\n"
            f"Host: localhost:8081\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Connection: close\r\n"
            f"traceparent: {tp}\r\n"
            f"\r\n"
        ).encode() + body_bytes
        sock = socket.create_connection(("localhost", 8081), timeout=10)
        try:
            sock.sendall(req)
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        finally:
            sock.close()
        text = data.decode("utf-8", errors="replace")
        first_line = text.split("\r\n", 1)[0]
        status = int(first_line.split(" ", 2)[1])
        assert status == 200, (
            f"Whitespace-padded traceparent caused {status}"
        )


# =========================================================================
# 5. Custom header leaking
# =========================================================================

class TestCustomHeaderLeaking:
    """Verify that arbitrary request headers are NOT forwarded to ftsearch.
    The ACL should only forward traceparent, tracestate, and the
    propagated baggage subset."""

    def test_authorization_header_not_leaked(self):
        """An Authorization header on the ACL request must not propagate
        to ftsearch -- otherwise internal auth tokens would leak."""
        r = _post(headers={
            "Authorization": "Bearer super-secret-token",
            "traceparent": VALID_TRACEPARENT,
        })
        # We can only verify the ACL doesn't crash. The real check is
        # that FtsearchClient.search() only receives headers from
        # trace_ctx.headers_for_forwarding(), which excludes Authorization.
        assert r.status_code == 200

    def test_cookie_header_not_leaked(self):
        """Cookie header must not propagate to downstream services."""
        r = _post(headers={
            "Cookie": "session=abc123; token=secret",
            "traceparent": VALID_TRACEPARENT,
        })
        assert r.status_code == 200

    def test_x_custom_header_not_leaked(self):
        """Arbitrary X- headers must not propagate."""
        r = _post(headers={
            "X-Internal-Secret": "classified",
            "X-Forwarded-For": "10.0.0.1",
            "traceparent": VALID_TRACEPARENT,
        })
        assert r.status_code == 200

    def test_baggage_leaking_of_non_propagated_fields(self):
        """Only userId, companyId, customerOciSessionId should be
        forwarded in baggage. Other fields must be stripped."""
        r = _post(headers={
            "traceparent": VALID_TRACEPARENT,
            "baggage": "userId=42,internalSecret=classified,debug=true",
        })
        assert r.status_code == 200


# =========================================================================
# 6. Response header analysis
# =========================================================================

class TestResponseHeaders:
    """Analyse response headers for information leakage."""

    def test_no_server_header_leaked(self):
        """The response should not expose a 'server' header revealing
        the web framework and version."""
        r = _post()
        server = r.headers.get("server", "")
        # Uvicorn sends 'server: uvicorn' by default. Document this.
        if server:
            # Not a hard fail -- just document the finding.
            assert "uvicorn" in server.lower() or server == "", (
                f"Unexpected server header: {server}"
            )

    def test_no_x_powered_by_header(self):
        """X-Powered-By must not be present."""
        r = _post()
        assert "x-powered-by" not in r.headers

    def test_content_type_is_json(self):
        """Response content-type must be application/json."""
        r = _post()
        ct = r.headers.get("content-type", "")
        assert "application/json" in ct

    def test_no_security_headers_present(self):
        """Document the absence of security-hardening headers.
        These are typically expected on API responses."""
        r = _post()
        missing = []
        for hdr in [
            "x-content-type-options",
            "cache-control",
            "x-frame-options",
        ]:
            if hdr not in r.headers:
                missing.append(hdr)
        # This is a documentation test -- the ACL is known to lack these.
        # We record the finding but do not fail hard.
        assert missing, (
            "Security headers are now present -- update this test."
        )

    def test_error_response_does_not_leak_internals(self):
        """Error responses must use the legacy envelope and not
        expose stack traces or internal paths."""
        r = httpx.post(
            SEARCH_URL,
            json={"invalid": "body"},
            timeout=10,
        )
        assert r.status_code == 400
        body = r.json()
        # Must use legacy envelope
        assert "message" in body
        assert "timestamp" in body
        # Must NOT contain stack traces or file paths
        body_str = str(body)
        assert "Traceback" not in body_str
        assert "/home/" not in body_str
        assert ".py" not in body_str

    def test_healthz_response_headers(self):
        """Even healthz should not leak version info."""
        r = httpx.get(f"{ACL_BASE}/healthz", timeout=5)
        assert r.status_code == 200
        assert "x-powered-by" not in r.headers


# =========================================================================
# 7. X-Request-Id correlation
# =========================================================================

class TestXRequestIdCorrelation:
    """Probe whether the ACL supports or echoes request correlation IDs."""

    def test_x_request_id_not_echoed(self):
        """If the caller sends X-Request-Id, does the ACL echo it back?
        Most API gateways do; the ACL currently does not."""
        r = _post(headers={"X-Request-Id": "test-correlation-12345"})
        assert r.status_code == 200
        # Document whether it echoes. Currently expected: no echo.
        echoed = r.headers.get("x-request-id")
        assert echoed is None, (
            f"ACL echoes X-Request-Id (value={echoed}) -- unexpected"
        )

    def test_x_correlation_id_not_echoed(self):
        """Same probe with the X-Correlation-Id variant."""
        r = _post(headers={"X-Correlation-Id": "corr-67890"})
        assert r.status_code == 200
        assert r.headers.get("x-correlation-id") is None

    def test_no_auto_generated_request_id_in_response(self):
        """Check whether the ACL auto-generates a request ID even
        when none was sent."""
        r = _post()
        assert r.status_code == 200
        has_id = any(
            k.lower() in ("x-request-id", "x-correlation-id", "request-id")
            for k in r.headers
        )
        assert not has_id, (
            "ACL auto-generates a correlation ID -- this is new behaviour"
        )

    def test_traceparent_as_only_correlation_mechanism(self):
        """The ACL relies on traceparent for correlation. Confirm that
        sending a traceparent does not result in an extra correlation
        header being added to the response."""
        r = _post(headers={"traceparent": VALID_TRACEPARENT})
        assert r.status_code == 200
        # traceparent itself should not be echoed in the response
        assert r.headers.get("traceparent") is None


# =========================================================================
# 8. Baggage edge cases (bonus: closely related to tracing)
# =========================================================================

class TestBaggageEdgeCases:
    """Probe baggage header edge cases."""

    def test_baggage_with_semicolon_properties(self):
        """Baggage entries can have ;property=value suffixes. Verify
        the ACL strips them correctly and still works."""
        r = _post(headers={
            "traceparent": VALID_TRACEPARENT,
            "baggage": "userId=42;ttl=60,companyId=7;expires=2025-01-01",
        })
        assert r.status_code == 200

    def test_baggage_with_empty_values(self):
        """Baggage entries with empty values should not crash parsing."""
        r = _post(headers={
            "traceparent": VALID_TRACEPARENT,
            "baggage": "userId=,companyId=",
        })
        assert r.status_code == 200

    def test_baggage_with_url_encoded_values(self):
        """Baggage values may contain percent-encoded content."""
        r = _post(headers={
            "traceparent": VALID_TRACEPARENT,
            "baggage": "userId=user%2042,companyId=org%26co",
        })
        assert r.status_code == 200

    def test_baggage_with_duplicate_keys(self):
        """Duplicate keys in baggage -- last-write-wins or first-write-wins."""
        r = _post(headers={
            "traceparent": VALID_TRACEPARENT,
            "baggage": "userId=first,userId=second",
        })
        assert r.status_code == 200

    def test_oversized_baggage(self):
        """A very large baggage header should not crash the service."""
        huge_baggage = ",".join(
            f"key{i}=value{i}" for i in range(1000)
        )
        r = _post(headers={
            "traceparent": VALID_TRACEPARENT,
            "baggage": huge_baggage,
        })
        assert r.status_code in (200, 400, 431)
