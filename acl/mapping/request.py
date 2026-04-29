"""A2 — legacy `LegacySearchRequest` → ftsearch SearchRequest body.

Pure function: takes the parsed legacy DTO + the inbound query-string
params (page, pageSize, sort), returns:

  - `body`:    the JSON-shaped dict to POST to ftsearch.
  - `params`:  the query-string dict to attach to that POST.

ftsearch (`search-api`) accepts the same field names as the legacy
contract for almost everything — the wire shapes were aligned by F2
on purpose. The only translations:

  - `queryString` → `query`. Legacy used the verbose name; ftsearch
    follows the lighter convention.
  - `searchArticlesBy` → dropped. §2.1 collapsed the enum to a single
    value; ftsearch doesn't carry it.
  - `explain` → dropped. §2.2 stubs the response in A3; ftsearch
    doesn't need to know.
  - `page`, `pageSize`, `sort` → moved from body (legacy treated them
    as body fields too) to query string (ftsearch's choice — F2).

Currency two-roles (§3): top-level `currency` and
`priceFilter.currencyCode` are forwarded to ftsearch independently.
The mapper does NOT collapse them or assert equality — that's a
matter of legacy semantics that ftsearch implements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from acl.models import LegacySearchRequest


@dataclass(frozen=True)
class FtsearchRequest:
    """Output of the mapper. Plain dict + dict so callers can hand
    them straight to httpx without further serialization."""
    body: dict[str, Any]
    params: dict[str, str | int | list[str]]


def map_request(
    req: LegacySearchRequest,
    *,
    page: int = 1,
    page_size: int = 10,
    sort: list[str] | None = None,
) -> FtsearchRequest:
    """Build the ftsearch HTTP request from a parsed legacy request.

    Pure: same input → same output, no I/O, no global state. Suitable
    for property-based testing of the wire-level translation."""
    body = req.model_dump(by_alias=True, exclude_none=True)

    # `queryString` → `query`. Both nullable; preserve the absence
    # rather than emit `null`.
    if "queryString" in body:
        body["query"] = body.pop("queryString")

    # §2.1 — `searchArticlesBy` is single-value; ftsearch doesn't
    # carry it. The legacy DTO already validated it's STANDARD.
    body.pop("searchArticlesBy", None)

    # §2.2 — `explain` never reaches ftsearch. A3 fills in the
    # `articles[].explanation` field on the way back.
    body.pop("explain", None)

    params: dict[str, str | int | list[str]] = {
        "page": page,
        "pageSize": page_size,
    }
    if sort:
        # FastAPI `Query(default_factory=list)` accepts repeated `?sort=`
        # values; httpx serialises a list value as `?sort=a&sort=b`.
        params["sort"] = list(sort)

    return FtsearchRequest(body=body, params=params)
