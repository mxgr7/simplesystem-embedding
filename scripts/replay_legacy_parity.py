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
  - JSONL: one legacy request body per line, OR
  - JSON list: a top-level array of legacy request bodies, OR
  - JSON object: a single legacy request body (treated as a 1-item list).

Output:
  - `reports/parity/replay.md` (overrideable via `--out`) — per-request
    detail + summary table + top divergent paths.
  - Stdout: one-line summary.

Pagination:
  Both endpoints get `?page=N&pageSize=M` query params by default.
  Pass `--no-pagination` if your legacy endpoint puts pagination in
  the body (or rejects unknown query params).

Auth:
  `--legacy-auth` value goes verbatim into the `Authorization` header
  on legacy requests. The ACL is assumed unauthenticated (typical for
  in-cluster service-to-service).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx


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

def load_requests(path: Path) -> list[dict]:
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
        r = post(client, legacy_url, body, params, legacy_headers)
        rec["legacy_status"] = r.status_code
        if r.headers.get("content-type", "").startswith("application/json"):
            legacy_body = r.json()
    except httpx.HTTPError as e:
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
        out.append(f"### #{i} — legacy={legacy_label} acl={acl_label}\n")
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
                        "of legacy request bodies.")
    p.add_argument("--legacy-url", required=True,
                   help="Full URL to legacy /article-features/search "
                        "(e.g. https://prod.example/article-features/search).")
    p.add_argument("--acl-url", default="http://localhost:8081/article-features/search",
                   help="Full URL to ACL endpoint. Default: %(default)s")
    p.add_argument("--legacy-auth", default=None,
                   help="Authorization header value sent to the legacy "
                        "endpoint, e.g. 'Bearer <token>'. Not sent to ACL.")
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

    requests = load_requests(args.requests_file)
    if args.limit > 0:
        requests = requests[: args.limit]
    if not requests:
        print("no requests to replay", file=sys.stderr)
        return 1

    legacy_headers = {"Content-Type": "application/json"}
    if args.legacy_auth:
        legacy_headers["Authorization"] = args.legacy_auth
    acl_headers = {"Content-Type": "application/json"}
    params: dict[str, Any] = {} if args.no_pagination else {
        "page": args.page, "pageSize": args.page_size,
    }

    results: list[dict] = []
    with httpx.Client(timeout=args.timeout) as client:
        for body in requests:
            results.append(replay_one(
                client, body,
                legacy_url=args.legacy_url, acl_url=args.acl_url,
                legacy_headers=legacy_headers, acl_headers=acl_headers,
                params=params,
            ))

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
