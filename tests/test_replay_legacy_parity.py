"""Unit tests for the shape-extraction + diff logic in
``scripts/replay_legacy_parity.py``.

The HTTP orchestration path is exercised via ``httpx.MockTransport``;
the reporting path renders a small fixture and is sanity-checked
against expected substrings rather than a full snapshot.
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_legacy_parity import (  # noqa: E402
    HARD,
    SOFT,
    NextgenStgAuth,
    build_argparser,
    diff_paths,
    load_nextgen_auth_state,
    load_requests,
    main,
    render_report,
    replay_one,
    shape,
)


# ---- shape() -------------------------------------------------------------

def test_shape_scalars_and_none():
    assert shape(None) == "NoneType"
    assert shape(True) == "bool"
    assert shape(7) == "int"
    assert shape(1.5) == "float"
    assert shape("hi") == "str"


def test_shape_dict_recurses():
    s = shape({"a": 1, "b": "x", "c": None})
    assert s == {"a": "int", "b": "str", "c": "NoneType"}


def test_shape_empty_list_uses_sentinel():
    assert shape([]) == {"__list_of__": "__empty_list__"}


def test_shape_homogeneous_list_collapses_to_one_element():
    s = shape([{"id": "a"}, {"id": "b"}, {"id": "c"}])
    assert s == {"__list_of__": {"id": "str"}}


def test_shape_heterogeneous_list_unions_keys():
    s = shape([{"id": "a"}, {"id": "b", "extra": 1}])
    assert s == {"__list_of__": {"id": "str", "extra": "int"}}


def test_shape_nested_array_in_dict():
    body = {"articles": [{"articleId": "x", "explanation": "N/A"}],
            "metadata": {"hitCount": 5}}
    assert shape(body) == {
        "articles": {"__list_of__": {"articleId": "str", "explanation": "str"}},
        "metadata": {"hitCount": "int"},
    }


# ---- diff_paths() --------------------------------------------------------

def test_diff_identical_shapes_returns_empty():
    a = shape({"x": 1, "y": [{"id": "z"}]})
    assert diff_paths(a, a) == []


def test_diff_flags_hard_type_mismatch():
    a = shape({"hitCount": 5})
    b = shape({"hitCount": 5.0})
    diffs = diff_paths(a, b)
    assert len(diffs) == 1
    path, sev, reason = diffs[0]
    assert path == "hitCount"
    assert sev == HARD
    assert "type mismatch" in reason


def test_diff_flags_soft_missing_key_in_acl():
    a = shape({"articles": [{"articleId": "x", "explanation": "N/A"}]})
    b = shape({"articles": [{"articleId": "x"}]})
    diffs = diff_paths(a, b)
    assert any(p == "articles[].explanation" and s == SOFT for p, s, _ in diffs)


def test_diff_flags_soft_extra_key_in_acl():
    a = shape({"articles": [{"articleId": "x"}]})
    b = shape({"articles": [{"articleId": "x", "score": 0.5}]})
    diffs = diff_paths(a, b)
    assert any(p == "articles[].score" and s == SOFT for p, s, _ in diffs)


def test_diff_treats_empty_list_as_compatible():
    """Local Milvus may return 0 hits while legacy returns N. The
    diff must not flag an empty array as a shape divergence."""
    legacy = shape({"articles": [{"articleId": "x"}], "summaries": {}})
    acl_empty_hits = shape({"articles": [], "summaries": {}})
    assert diff_paths(legacy, acl_empty_hits) == []


def test_diff_treats_empty_summaries_as_compatible():
    """Same logic applies to nested empty arrays inside summaries."""
    legacy = shape({"summaries": {"CATEGORIES": [{"id": "x", "count": 1}]}})
    acl = shape({"summaries": {"CATEGORIES": []}})
    diffs = diff_paths(legacy, acl)
    # The list-of element shape diff is suppressed because one side
    # is __empty_list__.
    assert diffs == []


def test_diff_walks_into_legacy_envelope():
    legacy = shape({
        "articles": [{"articleId": "x", "explanation": "N/A"}],
        "summaries": {"CATEGORIES": [{"id": "c1", "count": 3}]},
        "metadata": {"hitCount": 100, "totalPages": 10},
    })
    acl = shape({
        "articles": [{"articleId": "y", "explanation": "N/A"}],
        "summaries": {"CATEGORIES": [{"id": "c2", "count": 7}]},
        "metadata": {"hitCount": 5, "totalPages": 1},
    })
    # Same shape, different values → no diffs.
    assert diff_paths(legacy, acl) == []


def test_diff_finds_acl_dropping_field_legacy_returns():
    """The realistic ACL-bug scenario: A3 forgot to forward a field."""
    legacy = shape({
        "metadata": {"hitCount": 5, "queryDuration": 12, "warnings": ["x"]},
    })
    acl = shape({
        "metadata": {"hitCount": 5},
    })
    diffs = diff_paths(legacy, acl)
    paths = {p for p, _, _ in diffs}
    assert "metadata.queryDuration" in paths
    assert "metadata.warnings" in paths


# ---- load_requests() -----------------------------------------------------

def test_load_requests_json_list(tmp_path: Path):
    p = tmp_path / "reqs.json"
    p.write_text(json.dumps([{"a": 1}, {"a": 2}]))
    assert load_requests(p) == [{"a": 1}, {"a": 2}]


def test_load_requests_jsonl(tmp_path: Path):
    p = tmp_path / "reqs.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n\n')
    assert load_requests(p) == [{"a": 1}, {"a": 2}]


def test_load_requests_single_object(tmp_path: Path):
    p = tmp_path / "one.json"
    p.write_text(json.dumps({"a": 1}))
    assert load_requests(p) == [{"a": 1}]


def test_load_requests_rejects_top_level_scalar(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("42")
    with pytest.raises(SystemExit):
        load_requests(p)


# ---- next-gen staging auth -----------------------------------------------

def _fake_jwt(exp: int | None = None) -> str:
    def enc(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    payload = {"exp": exp or int(time.time()) + 3600}
    return f"{enc({'alg': 'none', 'typ': 'JWT'})}.{enc(payload)}.sig"


def test_nextgen_auth_missing_state_prompts_refresh_and_saves(tmp_path: Path):
    state_path = tmp_path / ".nextgen-auth"
    refresh_url = "https://auth.test/refresh"
    bearer = _fake_jwt()
    prompts: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert str(req.url) == refresh_url
        assert json.loads(req.content)["refreshToken"] == "initial-refresh"
        return httpx.Response(
            200,
            json={"token": bearer, "refreshToken": "rotated-refresh"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        auth = NextgenStgAuth(
            client,
            state_path=state_path,
            refresh_url=refresh_url,
            prompt_fn=lambda prompt: prompts.append(prompt) or "initial-refresh",
        )
        headers = auth.authenticated_headers({"Content-Type": "application/json"})

    assert headers["Authorization"] == f"Bearer {bearer}"
    assert prompts == ["Paste NEXTGEN refresh token (input hidden): "]
    assert load_nextgen_auth_state(state_path) == {
        "token": bearer,
        "refreshToken": "rotated-refresh",
    }
    assert state_path.stat().st_mode & 0o777 == 0o600


def test_nextgen_auth_refresh_failure_prompts_for_new_refresh_token(tmp_path: Path):
    state_path = tmp_path / ".nextgen-auth"
    state_path.write_text(json.dumps({"refreshToken": "stale-refresh"}))
    refresh_url = "https://auth.test/refresh"
    bearer = _fake_jwt()
    attempts: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        refresh_token = json.loads(req.content)["refreshToken"]
        attempts.append(refresh_token)
        if refresh_token == "stale-refresh":
            return httpx.Response(401, json={"message": "bad refresh"})
        return httpx.Response(
            200,
            json={"token": bearer, "refreshToken": "rotated-refresh"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        auth = NextgenStgAuth(
            client,
            state_path=state_path,
            refresh_url=refresh_url,
            prompt_fn=lambda _prompt: "fresh-refresh",
        )
        headers = auth.authenticated_headers({})

    assert headers["Authorization"] == f"Bearer {bearer}"
    assert attempts == ["stale-refresh", "fresh-refresh"]
    assert load_nextgen_auth_state(state_path)["refreshToken"] == "rotated-refresh"


# ---- replay_one() with mocked HTTP ---------------------------------------

LEGACY_RESPONSE = {
    "articles": [{"articleId": "v1:abc"}, {"articleId": "v2:def"}],
    "summaries": {"CATEGORIES": [{"id": "c1", "count": 5}]},
    "metadata": {"hitCount": 2},
}


def _make_client(legacy_body, legacy_status, acl_body, acl_status):
    def handler(req: httpx.Request) -> httpx.Response:
        if "legacy" in str(req.url):
            return httpx.Response(legacy_status, json=legacy_body)
        return httpx.Response(acl_status, json=acl_body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_replay_one_clean_when_shapes_match():
    client = _make_client(LEGACY_RESPONSE, 200, LEGACY_RESPONSE, 200)
    rec = replay_one(
        client, body={"q": "x"},
        legacy_url="http://legacy.test/search",
        acl_url="http://acl.test/search",
        legacy_headers={}, acl_headers={},
        params={"page": 1, "pageSize": 10},
    )
    assert rec["legacy_status"] == 200
    assert rec["acl_status"] == 200
    assert rec["diffs"] == []


def test_replay_one_flags_drop_in_acl():
    acl_body = {**LEGACY_RESPONSE,
                "metadata": {**LEGACY_RESPONSE["metadata"]}}
    # Legacy returns warnings; ACL forgets them.
    legacy_body = {**LEGACY_RESPONSE,
                   "metadata": {**LEGACY_RESPONSE["metadata"],
                                "warnings": ["x"]}}
    client = _make_client(legacy_body, 200, acl_body, 200)
    rec = replay_one(
        client, body={"q": "x"},
        legacy_url="http://legacy.test/search",
        acl_url="http://acl.test/search",
        legacy_headers={}, acl_headers={},
        params={},
    )
    assert any(p == "metadata.warnings" for p, _, _ in rec["diffs"])


def test_replay_one_refreshes_and_retries_legacy_401(tmp_path: Path):
    state_path = tmp_path / ".nextgen-auth"
    state_path.write_text(json.dumps({"refreshToken": "r1"}))
    refresh_url = "https://auth.test/refresh"
    refresh_calls = 0
    legacy_auth_headers: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal refresh_calls
        if str(req.url) == refresh_url:
            refresh_calls += 1
            return httpx.Response(
                200,
                json={"token": f"t{refresh_calls}",
                      "refreshToken": f"r{refresh_calls + 1}"},
            )
        if "legacy.test" in str(req.url):
            auth_header = req.headers.get("Authorization")
            legacy_auth_headers.append(auth_header)
            if auth_header == "Bearer t1":
                return httpx.Response(401, json={"message": "expired"})
            return httpx.Response(200, json=LEGACY_RESPONSE)
        return httpx.Response(200, json=LEGACY_RESPONSE)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        auth = NextgenStgAuth(
            client,
            state_path=state_path,
            refresh_url=refresh_url,
            prompt_fn=lambda _prompt: pytest.fail("prompt should not be needed"),
        )
        rec = replay_one(
            client, body={"q": "x"},
            legacy_url="http://legacy.test/search",
            acl_url="http://acl.test/search",
            legacy_headers={}, acl_headers={},
            params={},
            legacy_auth=auth,
        )

    assert rec["legacy_status"] == 200
    assert rec["acl_status"] == 200
    assert legacy_auth_headers == ["Bearer t1", "Bearer t2"]


def test_replay_one_records_legacy_error():
    def handler(req: httpx.Request) -> httpx.Response:
        if "legacy" in str(req.url):
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json=LEGACY_RESPONSE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rec = replay_one(
        client, body={"q": "x"},
        legacy_url="http://legacy.test/search",
        acl_url="http://acl.test/search",
        legacy_headers={}, acl_headers={},
        params={},
    )
    assert "legacy_error" in rec
    assert rec["acl_status"] == 200
    assert "diffs" not in rec  # no shape diff when one side failed


def test_replay_one_records_non_json_response():
    def handler(req: httpx.Request) -> httpx.Response:
        if "legacy" in str(req.url):
            return httpx.Response(502, text="Bad Gateway",
                                  headers={"content-type": "text/html"})
        return httpx.Response(200, json=LEGACY_RESPONSE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rec = replay_one(
        client, body={"q": "x"},
        legacy_url="http://legacy.test/search",
        acl_url="http://acl.test/search",
        legacy_headers={}, acl_headers={},
        params={},
    )
    assert rec["legacy_status"] == 502
    assert "diffs" not in rec


# ---- render_report() -----------------------------------------------------

def test_render_report_includes_summary_and_diff_table():
    args = build_argparser().parse_args([
        "--requests-file", "/dev/null",
        "--legacy-url", "http://legacy.test/search",
    ])
    results = [
        {"request": {"q": "hi"},
         "legacy_status": 200, "acl_status": 200,
         "diffs": [("metadata.queryDuration", SOFT, "present in legacy only: \"int\"")]},
        {"request": {"q": "world"},
         "legacy_status": 200, "acl_status": 200,
         "diffs": []},
    ]
    md = render_report(results, args)
    assert "Legacy vs ACL parity replay" in md
    assert "shape clean             | 1" in md
    assert "metadata.queryDuration" in md
    assert "Shape: OK" in md  # the second request's section


# ---- main() smoke (CLI wiring) -------------------------------------------

def test_main_runs_end_to_end(tmp_path: Path, monkeypatch):
    """Wire main() to a MockTransport and write a real report."""
    reqs = tmp_path / "reqs.jsonl"
    reqs.write_text(json.dumps({"q": "schraube"}) + "\n")
    out = tmp_path / "out" / "report.md"

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=LEGACY_RESPONSE)

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def fake_client(*a, **kw):
        return real_client(transport=transport)

    monkeypatch.setattr("scripts.replay_legacy_parity.httpx.Client", fake_client)

    rc = main([
        "--requests-file", str(reqs),
        "--legacy-url", "http://legacy.test/article-features/search",
        "--acl-url", "http://acl.test/article-features/search",
        "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()
    md = out.read_text()
    assert "Shape: OK" in md
