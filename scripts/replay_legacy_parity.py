"""Replay legacy search requests against the ACL and diff response shape.

For each request in `--requests-file`, POSTs to the legacy search API
and to the ACL, then compares the two responses *structurally* — same
top-level keys, same nested keys, same scalar types — while ignoring
hit contents (different corpora produce different `articleId`s,
different summary buckets, different `hitCount`s; that is expected).

Use cases:
  - Smoke-test the ACL response mapper (A3) against captured PostHog
    traffic before flipping production callers from legacy to ACL.
  - Catch shape regressions when the legacy API changes upstream.
  - Land the deferred A6 captured-traffic smoke (per
    `ARTICLE_SEARCH_REPLACEMENT_STATUS.md`).

Input (`--requests-file`):
  - JSONL: one legacy request body or replay record per line, OR
  - JSON list: a top-level array of request bodies / replay records, OR
  - JSON object: a single request body / replay record (treated as a 1-item list).

Replay records let fixtures carry per-request query params and provenance:
  {"name": "case-name", "source": "where it came from",
   "params": {"page": 2, "pageSize": 20, "sort": ["name,desc"]},
   "body": { ... legacy request body ... }}

Plain body-only files remain supported for captured traffic JSONL.

Output:
  - `reports/parity/replay.md` (overrideable via `--out`) — per-request
    detail + summary table + top divergent paths.
  - Stdout: one-line summary.

Pagination:
  Both endpoints get `?page=N&pageSize=M` query params by default.
  Replay-record `params` override those defaults, so checked-in fixtures
  can cover sort/page variants. Pass `--no-pagination` if your legacy
  endpoint puts pagination in the body (or rejects unknown query params).

Auth:
  `--legacy-auth` value goes verbatim into the `Authorization` header
  on legacy requests. For the staging next-gen frontend/legacy endpoint,
  pass `--nextgen-stg-auth`: the script refreshes a bearer token via
  `/authentication/refresh`, stores `{token, refreshToken}` in
  `.nextgen-auth`, and prompts for a fresh refresh token when that state
  is missing/empty or refresh fails. The ACL is assumed unauthenticated
  (typical for in-cluster service-to-service).
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import getpass
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import httpx


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NEXTGEN_AUTH_STATE = REPO_ROOT / ".nextgen-auth"
DEFAULT_NEXTGEN_REFRESH_URL = (
    "https://api-nextgen-stg.simplesystem.com/authentication/refresh"
)
NEXTGEN_REFRESH_MARGIN_SECONDS = 5


# ---------- next-gen staging auth -----------------------------------------

class NextgenAuthError(RuntimeError):
    """Raised when the next-gen refresh-token flow cannot produce a bearer."""


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _decode_jwt_payload(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode("ascii"))
        obj = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _token_expires_at(token: str | None) -> float | None:
    if not token:
        return None
    payload = _decode_jwt_payload(token)
    if not payload:
        return None
    exp = payload.get("exp")
    return float(exp) if isinstance(exp, (int, float)) else None


def _token_needs_refresh(
    token: str | None,
    *,
    margin_seconds: int = NEXTGEN_REFRESH_MARGIN_SECONDS,
) -> bool:
    if not token:
        return True
    exp = _token_expires_at(token)
    if exp is None:
        # Opaque token: keep using it until a 401 forces a refresh.
        return False
    return exp <= time.time() + margin_seconds


def load_nextgen_auth_state(path: Path) -> dict[str, str]:
    """Load `.nextgen-auth` without ever raising on missing/empty/corrupt data.

    Corrupt or wrong-shaped files are treated like a missing state file so the
    caller can prompt for a fresh refresh token.
    """
    try:
        txt = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    if not txt.strip():
        return {}
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}

    state: dict[str, str] = {}
    token = obj.get("token") or obj.get("accessToken")
    refresh = obj.get("refreshToken") or obj.get("refresh_token")
    if isinstance(token, str) and token.strip():
        state["token"] = token.strip()
    if isinstance(refresh, str) and refresh.strip():
        state["refreshToken"] = refresh.strip()
    return state


def save_nextgen_auth_state(path: Path, state: dict[str, str]) -> None:
    token = state.get("token")
    refresh = state.get("refreshToken")
    if not token or not refresh:
        raise NextgenAuthError("refusing to save incomplete next-gen auth state")

    payload = {
        "token": token,
        "refreshToken": refresh,
        "updatedAt": _now_utc_iso(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _extract_nextgen_auth_state(
    body: Any,
    *,
    fallback_refresh_token: str,
) -> dict[str, str]:
    if not isinstance(body, dict):
        raise NextgenAuthError("refresh response was not a JSON object")
    token = body.get("token") or body.get("accessToken")
    refresh = (
        body.get("refreshToken") or body.get("refresh_token") or fallback_refresh_token
    )
    if not isinstance(token, str) or not token.strip():
        raise NextgenAuthError("refresh response did not contain a bearer token")
    if not isinstance(refresh, str) or not refresh.strip():
        raise NextgenAuthError("refresh response did not contain a refresh token")
    return {"token": token.strip(), "refreshToken": refresh.strip()}


def refresh_nextgen_auth(
    client: httpx.Client,
    refresh_url: str,
    refresh_token: str,
) -> dict[str, str]:
    try:
        response = client.post(
            refresh_url,
            json={"refreshToken": refresh_token},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
    except httpx.HTTPError as e:
        raise NextgenAuthError(f"refresh request failed: {e!r}") from e

    if response.status_code >= 400:
        raise NextgenAuthError(f"refresh failed with HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError as e:
        raise NextgenAuthError("refresh response was not JSON") from e
    return _extract_nextgen_auth_state(body, fallback_refresh_token=refresh_token)


class NextgenStgAuth:
    """Small stateful auth helper for the staging frontend/legacy endpoint.

    State is read from and written to `.nextgen-auth` as JSON with 0600
    permissions. Refresh tokens are rotated when the backend returns a new one.
    """

    def __init__(
        self,
        client: httpx.Client,
        *,
        state_path: Path = DEFAULT_NEXTGEN_AUTH_STATE,
        refresh_url: str = DEFAULT_NEXTGEN_REFRESH_URL,
        prompt_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.client = client
        self.state_path = state_path
        self.refresh_url = refresh_url
        self.prompt_fn = prompt_fn
        self.state = load_nextgen_auth_state(state_path)

    def authenticated_headers(
        self,
        headers: dict[str, str],
        *,
        force_refresh: bool = False,
    ) -> dict[str, str]:
        out = dict(headers)
        out["Authorization"] = f"Bearer {self.token(force_refresh=force_refresh)}"
        return out

    def token(self, *, force_refresh: bool = False) -> str:
        token = self.state.get("token")
        if force_refresh or _token_needs_refresh(token):
            return self._refresh_with_prompt_fallback()
        return token

    def _refresh_with_prompt_fallback(self) -> str:
        while True:
            refresh_token = self.state.get("refreshToken")
            if not refresh_token:
                refresh_token = self._prompt_refresh_token()
                self.state["refreshToken"] = refresh_token
            try:
                self.state = refresh_nextgen_auth(
                    self.client, self.refresh_url, refresh_token
                )
                save_nextgen_auth_state(self.state_path, self.state)
                return self.state["token"]
            except NextgenAuthError as e:
                print(f"next-gen auth refresh failed: {e}", file=sys.stderr)
                self.state.pop("token", None)
                self.state["refreshToken"] = self._prompt_refresh_token()

    def _prompt_refresh_token(self) -> str:
        prompt = "Paste NEXTGEN refresh token (input hidden): "
        if self.prompt_fn is not None:
            raw = self.prompt_fn(prompt)
        else:
            if not sys.stdin.isatty():
                raise NextgenAuthError(
                    f"{self.state_path} is missing/empty or its refresh token "
                    "failed, and stdin is not interactive. Run this command in "
                    "a terminal once and paste a fresh refresh token."
                )
            raw = getpass.getpass(prompt)
        token = (raw or "").strip()
        if not token:
            raise NextgenAuthError("no refresh token provided")
        return token


# ---------- shape extraction ----------------------------------------------

def shape(value: Any) -> Any:
    """Reduce a JSON value to its structural sketch.

    Scalars become their python type name; objects become dicts of
    field-name → child-shape; arrays collapse to a single merged
    element-shape under the sentinel ``__list_of__`` (or
    ``__empty_list__`` when the array is empty).

    The merged element-shape is the structural union across all
    elements: heterogeneous arrays surface as combined shapes, not as
    just the first element's shape.
    """
    if value is None:
        return "NoneType"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        if not value:
            return {"__list_of__": "__empty_list__"}
        merged: Any = shape(value[0])
        for item in value[1:]:
            merged = _merge(merged, shape(item))
        return {"__list_of__": merged}
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    return type(value).__name__


def _merge(a: Any, b: Any) -> Any:
    """Structural union of two shape sketches.

    Keys present on only one side are kept; types that disagree fold
    into the literal ``"<type>|<type>"`` (sorted).
    """
    if a == b:
        return a
    if a == "__empty_list__":
        return b
    if b == "__empty_list__":
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        out: dict[str, Any] = {}
        for k in set(a) | set(b):
            if k in a and k in b:
                out[k] = _merge(a[k], b[k])
            else:
                out[k] = a.get(k, b.get(k))
        return out
    if isinstance(a, str) and isinstance(b, str):
        return "|".join(sorted({a, b}))
    return f"MIXED({a!r},{b!r})"


# ---------- shape diff ----------------------------------------------------

# Severity tags used in the diff report.
HARD = "HARD"   # structural — almost certainly a real ACL bug
SOFT = "SOFT"   # data-dependent — flagged for review, often expected


def diff_paths(
    legacy: Any, acl: Any, path: str = ""
) -> list[tuple[str, str, str]]:
    """Walk two shape sketches in lockstep.

    Returns a list of ``(path, severity, reason)`` tuples for points
    where the two sketches disagree. Empty arrays (``__empty_list__``)
    on either side are treated as compatible with any element shape on
    the other side — the local Milvus corpus may simply not match.
    """
    if legacy == acl:
        return []

    if legacy == "__empty_list__" or acl == "__empty_list__":
        return []

    if isinstance(legacy, dict) and isinstance(acl, dict):
        diffs: list[tuple[str, str, str]] = []
        # Recurse into list-of element shapes.
        if "__list_of__" in legacy and "__list_of__" in acl:
            return diff_paths(legacy["__list_of__"], acl["__list_of__"], f"{path}[]")
        for k in sorted(set(legacy) | set(acl)):
            sub = f"{path}.{k}" if path else k
            if k not in legacy:
                diffs.append((sub, SOFT, f"present in ACL only: {_repr(acl[k])}"))
            elif k not in acl:
                diffs.append((sub, SOFT, f"present in legacy only: {_repr(legacy[k])}"))
            else:
                diffs.extend(diff_paths(legacy[k], acl[k], sub))
        return diffs

    # Scalar / mixed-type mismatch.
    return [(
        path or "<root>",
        HARD,
        f"type mismatch — legacy={_repr(legacy)} acl={_repr(acl)}",
    )]


def _repr(shape_node: Any) -> str:
    s = json.dumps(shape_node, sort_keys=True, default=str)
    return s if len(s) <= 80 else s[:77] + "..."


# ---------- I/O -----------------------------------------------------------

def _load_json_items(path: Path) -> list[Any]:
    """Accept JSON-list, JSONL, or single JSON object."""
    txt = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in txt.splitlines() if line.strip()]
    obj = json.loads(txt)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise SystemExit(f"unsupported request file shape in {path}: {type(obj).__name__}")


def load_requests(path: Path) -> list[dict]:
    """Back-compatible body-only loader used by older callers/tests."""
    return [r["body"] for r in load_request_records(path)]


def load_request_records(path: Path) -> list[dict]:
    """Load replay inputs as normalized records.

    Each item may be either a raw legacy request body or a replay record with
    `{body, params?, name?, source?}`. The normalized shape is always:
    `{body: dict, params: dict, name?: str, source?: str}`.
    """
    return [_normalize_request_record(obj, path) for obj in _load_json_items(path)]


def _normalize_request_record(obj: Any, path: Path) -> dict:
    if not isinstance(obj, dict):
        raise SystemExit(
            f"unsupported request item in {path}: {type(obj).__name__}"
        )

    if "body" not in obj:
        return {"body": obj, "params": {}}

    body = obj.get("body")
    if not isinstance(body, dict):
        raise SystemExit(
            f"replay record in {path} has non-object body: {type(body).__name__}"
        )
    params = obj.get("params") or {}
    if not isinstance(params, dict):
        raise SystemExit(
            f"replay record in {path} has non-object params: {type(params).__name__}"
        )

    rec: dict[str, Any] = {"body": body, "params": params}
    for key in ("name", "source"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            rec[key] = value.strip()
    return rec


def post(client: httpx.Client, url: str, body: dict, params: dict, headers: dict) -> httpx.Response:
    return client.post(url, params=params, json=body, headers=headers)


def replay_one(
    client: httpx.Client,
    body: dict,
    legacy_url: str,
    acl_url: str,
    legacy_headers: dict,
    acl_headers: dict,
    params: dict,
    legacy_auth: NextgenStgAuth | None = None,
) -> dict:
    """POST one request to both endpoints, return a result record.

    The record carries status codes, error strings, response shapes,
    and the diff list. Any single-side failure leaves the diffs blank
    so the operator can deal with infra issues separately from shape
    findings.
    """
    rec: dict[str, Any] = {"request": body}
    legacy_body: Any = None
    acl_body: Any = None

    try:
        request_headers = (
            legacy_auth.authenticated_headers(legacy_headers)
            if legacy_auth is not None else legacy_headers
        )
        r = post(client, legacy_url, body, params, request_headers)
        if r.status_code == 401 and legacy_auth is not None:
            # Bearers on staging are short-lived. Refresh once and retry.
            request_headers = legacy_auth.authenticated_headers(
                legacy_headers, force_refresh=True
            )
            r = post(client, legacy_url, body, params, request_headers)
        rec["legacy_status"] = r.status_code
        if r.headers.get("content-type", "").startswith("application/json"):
            legacy_body = r.json()
    except (httpx.HTTPError, NextgenAuthError) as e:
        rec["legacy_error"] = repr(e)

    try:
        r = post(client, acl_url, body, params, acl_headers)
        rec["acl_status"] = r.status_code
        if r.headers.get("content-type", "").startswith("application/json"):
            acl_body = r.json()
    except httpx.HTTPError as e:
        rec["acl_error"] = repr(e)

    if legacy_body is not None and acl_body is not None:
        rec["diffs"] = diff_paths(shape(legacy_body), shape(acl_body))
    return rec


# ---------- report rendering ----------------------------------------------

def render_report(results: list[dict], args: argparse.Namespace) -> str:
    out = []
    out.append("# Legacy vs ACL parity replay\n")
    out.append(f"- legacy: `{args.legacy_url}`")
    out.append(f"- acl:    `{args.acl_url}`")
    out.append(f"- pagination: page={args.page} pageSize={args.page_size}"
               + ("" if not args.no_pagination else " (disabled)"))
    if getattr(args, "nextgen_stg_auth", False):
        out.append(f"- legacy auth: next-gen staging refresh-token flow "
                   f"(`{args.nextgen_auth_state}`)")
    elif getattr(args, "legacy_auth", None):
        out.append("- legacy auth: explicit Authorization header")
    if getattr(args, "legacy_impersonate_user", None):
        out.append(f"- legacy impersonation: `{args.legacy_impersonate_user}`")
    out.append(f"- requests: {len(results)}\n")

    n = len(results)
    n_clean = sum(1 for r in results if not r.get("diffs"))
    n_hard = sum(1 for r in results if any(s == HARD for _, s, _ in r.get("diffs", [])))
    n_soft = sum(1 for r in results if r.get("diffs") and not any(
        s == HARD for _, s, _ in r["diffs"]))
    legacy_4xx5xx = sum(1 for r in results if "legacy_error" in r or r.get("legacy_status", 0) >= 400)
    acl_4xx5xx = sum(1 for r in results if "acl_error" in r or r.get("acl_status", 0) >= 400)

    out.append("## Summary\n")
    out.append("| metric | count |")
    out.append("|---|---|")
    out.append(f"| total requests          | {n} |")
    out.append(f"| shape clean             | {n_clean} |")
    out.append(f"| shape diff (HARD only)  | {n_hard} |")
    out.append(f"| shape diff (SOFT only)  | {n_soft} |")
    out.append(f"| legacy 4xx/5xx or error | {legacy_4xx5xx} |")
    out.append(f"| acl 4xx/5xx or error    | {acl_4xx5xx} |")
    out.append("")

    # Aggregate diff paths.
    counts: Counter[tuple[str, str]] = Counter()
    for r in results:
        for path, sev, _ in r.get("diffs", []):
            counts[(path, sev)] += 1
    if counts:
        out.append("## Top divergent paths\n")
        out.append("| path | severity | requests |")
        out.append("|---|---|---|")
        for (path, sev), c in counts.most_common(30):
            out.append(f"| `{path}` | {sev} | {c} |")
        out.append("")

    # Per-request detail.
    out.append("## Per-request detail\n")
    for i, r in enumerate(results, 1):
        legacy_label = r.get("legacy_status", r.get("legacy_error", "—"))
        acl_label = r.get("acl_status", r.get("acl_error", "—"))
        title = f"#{i}"
        if r.get("name"):
            title += f" — {r['name']}"
        out.append(f"### {title} — legacy={legacy_label} acl={acl_label}\n")
        if r.get("source"):
            out.append(f"Source: `{r['source']}`")
        if r.get("params"):
            out.append("Query params:")
            out.append("```json")
            out.append(json.dumps(r["params"], indent=2, ensure_ascii=False))
            out.append("```")
        out.append("Request:")
        out.append("```json")
        out.append(json.dumps(r["request"], indent=2, ensure_ascii=False)[:2000])
        out.append("```")
        diffs = r.get("diffs")
        if not diffs:
            out.append("Shape: OK\n")
            continue
        out.append("Shape diffs:\n")
        for path, sev, reason in diffs:
            out.append(f"- `{path}` [{sev}] — {reason}")
        out.append("")
    return "\n".join(out)


# ---------- entry point ---------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--requests-file", type=Path, required=True,
                   help="Path to a JSON list, JSONL, or single-JSON-object file "
                        "of legacy request bodies or replay records.")
    p.add_argument("--legacy-url", required=True,
                   help="Full URL to legacy /article-features/search "
                        "(e.g. https://prod.example/article-features/search).")
    p.add_argument("--acl-url", default="http://localhost:8081/article-features/search",
                   help="Full URL to ACL endpoint. Default: %(default)s")
    p.add_argument("--legacy-auth", default=None,
                   help="Authorization header value sent to the legacy "
                        "endpoint, e.g. 'Bearer <token>'. Not sent to ACL.")
    p.add_argument("--nextgen-stg-auth", action="store_true",
                   help="Use the staging next-gen refresh-token flow for "
                        "legacy requests. Stores token state in "
                        "--nextgen-auth-state and refreshes/retries on 401.")
    p.add_argument("--nextgen-auth-state", type=Path,
                   default=DEFAULT_NEXTGEN_AUTH_STATE,
                   help="Hidden JSON auth state file for --nextgen-stg-auth. "
                        "Default: %(default)s")
    p.add_argument("--nextgen-refresh-url", default=DEFAULT_NEXTGEN_REFRESH_URL,
                   help="Refresh endpoint for --nextgen-stg-auth. "
                        "Default: %(default)s")
    p.add_argument("--legacy-impersonate-user", default=None,
                   help="Optional Impersonate-User header sent to legacy "
                        "requests, commonly required on staging.")
    p.add_argument("--page", type=int, default=1)
    p.add_argument("--page-size", type=int, default=10)
    p.add_argument("--no-pagination", action="store_true",
                   help="Skip page/pageSize query params on both endpoints.")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap the number of requests sent (0 = all).")
    p.add_argument("--out", type=Path,
                   default=Path("reports/parity/replay.md"),
                   help="Output report path. Default: %(default)s")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    request_records = load_request_records(args.requests_file)
    if args.limit > 0:
        request_records = request_records[: args.limit]
    if not request_records:
        print("no requests to replay", file=sys.stderr)
        return 1

    if args.legacy_auth and args.nextgen_stg_auth:
        print(
            "--legacy-auth and --nextgen-stg-auth are mutually exclusive",
            file=sys.stderr,
        )
        return 2

    legacy_headers = {"Content-Type": "application/json"}
    if args.legacy_auth:
        legacy_headers["Authorization"] = args.legacy_auth
    if args.legacy_impersonate_user:
        legacy_headers["Impersonate-User"] = args.legacy_impersonate_user
    acl_headers = {"Content-Type": "application/json"}
    params: dict[str, Any] = {} if args.no_pagination else {
        "page": args.page, "pageSize": args.page_size,
    }

    results: list[dict] = []
    with httpx.Client(timeout=args.timeout) as client:
        legacy_auth = NextgenStgAuth(
            client,
            state_path=args.nextgen_auth_state,
            refresh_url=args.nextgen_refresh_url,
        ) if args.nextgen_stg_auth else None
        for request_record in request_records:
            body = request_record["body"]
            request_params = dict(params)
            request_params.update(request_record.get("params") or {})
            rec = replay_one(
                client, body,
                legacy_url=args.legacy_url, acl_url=args.acl_url,
                legacy_headers=legacy_headers, acl_headers=acl_headers,
                params=request_params,
                legacy_auth=legacy_auth,
            )
            for key in ("name", "source"):
                if request_record.get(key):
                    rec[key] = request_record[key]
            if request_params:
                rec["params"] = request_params
            results.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(results, args), encoding="utf-8")

    n = len(results)
    n_clean = sum(1 for r in results if not r.get("diffs"))
    n_hard = sum(1 for r in results if any(s == HARD for _, s, _ in r.get("diffs", [])))
    print(f"replayed {n} requests; clean={n_clean} hard-diff={n_hard} "
          f"soft-diff={n - n_clean - n_hard}")
    print(f"  report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
