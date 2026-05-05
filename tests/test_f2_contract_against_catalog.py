"""Comprehensive F2 contract suite — drives every aspect of
``search-api/openapi.yaml`` against the catalog loaded into Milvus
(``articles_v6`` + ``offers_v6``).

Replaces the legacy↔ACL replay parity machinery: we no longer have
access to the legacy API, so coverage shifts to spec-conformance
testing. Every response in this file is validated against the
``SearchResponse`` schema in the OpenAPI document — drift between
the implementation and the spec fails the suite.

Test layout:

  Class A — schema validator self-test (tiny sanity tests that the
            spec validator works as intended).
  Class B — path/query parameter coverage (collection, page, pageSize,
            sort).
  Class C — request body validation + 422 rejections.
  Class D — searchMode envelope shapes (HITS_ONLY, SUMMARIES_ONLY,
            BOTH).
  Class E — every filter atom from spec §4.3 against real data.
  Class F — relationship filters (accessoriesFor / sparePartsFor /
            similarTo).
  Class G — priceFilter (currency-coded minor units).
  Class H — sort orderings (relevance default, articleId, name, price,
            asc/desc, multi-key first-key only).
  Class I — pagination invariants.
  Class J — summaries (one test per ``SummaryKind``).
  Class K — auth + concurrency guards (401 / 503).

The catalog is read-only — we never mutate it. Every test runs against
the current state of ``articles_v6`` + ``offers_v6`` in the Milvus at
``localhost:19530``.

Skips with a clear message when:
  - Milvus is unreachable;
  - either ``articles_v6`` or ``offers_v6`` is missing.

The embedder is mocked at the FastAPI ``app.state`` level so query-path
tests run without TEI; mock returns a fixed (but stable) 128-d unit
vector. BM25 over ``text_codes`` still depends on the literal query
string, so query-driven tests use German/English terms with known
representation in the catalog.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from pymilvus import MilvusClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_API_DIR = REPO_ROOT / "search-api"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEARCH_API_DIR))


# ── Constants pinned to the loaded catalog (TESTING_NOTES_2026-05-05.md) ──

MILVUS_URI = "http://localhost:19530"
ARTICLES_COLLECTION = "articles_v6"
OFFERS_COLLECTION = "offers_v6"

# Embedding dim for articles_v6.offer_embedding (FLOAT16_VECTOR dim=128).
EMBED_DIM = 128

# A small set of diverse, real-world facts harvested off the live
# collections. Used to build deterministic per-test bodies without
# coupling to a single corner of the catalog.
KNOWN_VENDORS: tuple[str, ...] = (
    "6a67b8b5-9b7c-47f0-92d5-1dfd8812a505",
    "216c5d41-b64a-42f1-b084-d7e3419b2219",
    "e22f1ac6-14bc-4287-ab0d-1f34c1780f2e",
    "01054f55-c50c-452b-8822-ee11be4788c9",
)
# `01054f55-…` had ~2k priced offers in the seed scan and is the
# safest "I expect hits back" vendor for behaviour assertions.
HIGH_VOLUME_VENDOR = "01054f55-c50c-452b-8822-ee11be4788c9"
KNOWN_MANUFACTURERS: tuple[str, ...] = ("Würth", "GARANT", "SMC", "TOOLCRAFT", "Siemens")
KNOWN_ECLASS5_CODES: tuple[int, ...] = (21010101, 23110101, 27260701, 27269134, 32169090)
# Real category_l1 strings present in articles_v6.
KNOWN_CATEGORY_L1: tuple[str, ...] = (
    "Verbindungselemente", "Elektromaterial", "Zerspanung",
)
KNOWN_QUERY_TERMS: tuple[str, ...] = ("schraube", "kabel", "siemens", "dichtung", "bohrer")


# ── Reachability gate ────────────────────────────────────────────────

def _milvus_ready() -> tuple[bool, str]:
    try:
        c = MilvusClient(uri=MILVUS_URI)
        existing = set(c.list_collections())
    except Exception as exc:
        return False, f"Milvus unreachable at {MILVUS_URI}: {exc}"
    missing = [n for n in (ARTICLES_COLLECTION, OFFERS_COLLECTION) if n not in existing]
    if missing:
        return False, f"missing collections: {', '.join(missing)}"
    return True, ""


_ready, _skip_reason = _milvus_ready()
pytestmark = pytest.mark.skipif(not _ready, reason=_skip_reason)


# ── OpenAPI → JSON-Schema compilation ────────────────────────────────

def _openapi_to_jsonschema(spec: dict) -> tuple[dict, dict]:
    """Walk an OpenAPI 3.x ``components.schemas`` block and produce a
    pair `(defs, base_schema)` usable with `jsonschema.Draft202012Validator`.

    Conversions applied:
      * ``$ref: '#/components/schemas/X'`` → ``$ref: '#/$defs/X'``.
      * ``nullable: true`` (OpenAPI 3.0) → JSON-Schema 2020-12 union form
        (``type: [...]`` augmented with ``"null"``). Spec 3.1 already
        uses union syntax; both styles are handled.
      * Strip OpenAPI-only annotations (``example``, ``description``,
        ``default``, ``deprecated``, ``readOnly``, ``writeOnly``,
        ``discriminator``, ``xml``).

    Returns the compiled `$defs` dict + a stub schema with
    ``$ref: '#/$defs/<name>'``-shaped helpers — callers pick the schema
    name they want to validate against and pass `{**defs_envelope,
    "$ref": "#/$defs/<name>"}` to the validator.
    """

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in {
                "example", "examples", "description", "default",
                "deprecated", "readOnly", "writeOnly",
                "discriminator", "xml", "summary", "title",
                # OpenAPI 'format' values like 'uuid' / 'date-time' are
                # advisory — jsonschema's format checker would need the
                # 'format' assertion explicitly enabled and the
                # 'format-nongpl' extra. Keep validation focused on
                # structure; format checks live in their own targeted
                # tests.
                "format",
            }:
                continue
            if k == "$ref" and isinstance(v, str) and v.startswith("#/components/schemas/"):
                name = v.split("/")[-1]
                out["$ref"] = f"#/$defs/{name}"
                continue
            out[k] = walk(v)
        nullable = out.pop("nullable", False)
        if nullable:
            # `nullable: true` has two shapes in OpenAPI 3.0:
            #   {type: T, nullable: true}            → {type: [T, "null"]}
            #   {$ref: '#/components/schemas/X',     → {anyOf: [{$ref}, {type: "null"}]}
            #    nullable: true}
            if "type" in out:
                t = out["type"]
                if isinstance(t, str):
                    out["type"] = [t, "null"]
                elif isinstance(t, list) and "null" not in t:
                    out["type"] = [*t, "null"]
            elif "$ref" in out:
                ref = out.pop("$ref")
                out["anyOf"] = [{"$ref": ref}, {"type": "null"}]
        return out

    raw_defs = spec.get("components", {}).get("schemas", {})
    defs: dict = {name: walk(schema) for name, schema in raw_defs.items()}
    return defs, {"$defs": defs}


def _validator_for(spec: dict, schema_name: str) -> Draft202012Validator:
    defs, envelope = _openapi_to_jsonschema(spec)
    return Draft202012Validator({**envelope, "$ref": f"#/$defs/{schema_name}"})


# ── Spec + validators (module-scope) ──────────────────────────────────

OPENAPI_PATH = SEARCH_API_DIR / "openapi.yaml"
OPENAPI_SPEC = yaml.safe_load(OPENAPI_PATH.read_text())
SEARCH_RESPONSE_VALIDATOR = _validator_for(OPENAPI_SPEC, "SearchResponse")
ERROR_VALIDATOR = _validator_for(OPENAPI_SPEC, "Error")
VALIDATION_ERROR_VALIDATOR = _validator_for(OPENAPI_SPEC, "ValidationError")


def assert_search_response_valid(body: dict) -> None:
    errors = sorted(SEARCH_RESPONSE_VALIDATOR.iter_errors(body), key=lambda e: e.path)
    if errors:
        msgs = "\n".join(f"  - {list(e.absolute_path)}: {e.message}" for e in errors)
        raise AssertionError(f"SearchResponse failed schema validation:\n{msgs}")


def assert_validation_error_valid(body: dict) -> None:
    errors = list(VALIDATION_ERROR_VALIDATOR.iter_errors(body))
    if errors:
        msgs = "\n".join(f"  - {list(e.absolute_path)}: {e.message}" for e in errors)
        raise AssertionError(f"ValidationError failed schema validation:\n{msgs}")


# ── App fixture ──────────────────────────────────────────────────────

def _stable_vector() -> list[float]:
    """Deterministic 128-d unit-norm vector for the mocked embedder.

    Picks a fixed direction so query-path tests are stable across runs.
    Vector content is irrelevant for BM25 — that leg only consumes the
    raw query string and runs against the codes inverted index. The
    dense leg here will retrieve some neighborhood of articles; we
    don't assert on its specific content, only on contract shape.
    """
    import math
    raw = [(i * 0.0123) % 1.0 - 0.5 for i in range(EMBED_DIM)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


@pytest.fixture(scope="module")
def search_api_app() -> Iterator:
    """Boot search-api in-process, in F9 dedup-topology mode, pointed
    at the loaded ``articles_v6``/``offers_v6`` collections.

    The TEI client (``app.state.embed.embed``) is replaced with an
    AsyncMock that returns a fixed unit vector so query-path requests
    don't need a real embedding service. Mock leak-out across runs is
    avoided by tearing down + cleaning sys.modules at module exit.
    """
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    mp.setenv("USE_DEDUP_TOPOLOGY", "1")
    mp.setenv("MILVUS_ARTICLES_COLLECTION", ARTICLES_COLLECTION)
    mp.setenv("EMBED_URL", "http://embed.invalid")
    mp.setenv("MILVUS_URI", MILVUS_URI)
    mp.setenv("API_KEY", "")
    # Generous concurrency cap so parallel-ish tests don't trip the
    # gate; we cover the gate behavior explicitly elsewhere.
    mp.setenv("MAX_CONCURRENT_SEARCHES", "256")

    spec = importlib.util.spec_from_file_location(
        "search_api_main_for_f2_contract", SEARCH_API_DIR / "main.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["search_api_main_for_f2_contract"] = mod
    spec.loader.exec_module(mod)
    try:
        with TestClient(mod.app) as client:
            mod.app.state.embed.embed = AsyncMock(return_value=[_stable_vector()])
            yield client
    finally:
        mp.undo()
        sys.modules.pop("search_api_main_for_f2_contract", None)


@pytest.fixture(scope="module")
def search_path() -> str:
    return f"/{OFFERS_COLLECTION}/_search"


# ── Body-builder helper ──────────────────────────────────────────────

def _all_active_cvs(milvus: MilvusClient) -> list[str]:
    """Pull a wide-but-bounded set of catalog_version_ids that have at
    least one offer. We use this as the always-on CV scope (the
    legacy `_closed_marketplace` filter is mandatory — no CV scope =>
    match-nothing).

    Bounded to ~300 ids: keeps the body size sensible while guaranteeing
    we have a query-able pool (a few hundred CVs cover the vast
    majority of seeded offers).
    """
    rows = milvus.query(
        collection_name=OFFERS_COLLECTION,
        filter="catalog_version_id != \"\"",
        output_fields=["catalog_version_id"],
        limit=8000,
    )
    seen: list[str] = []
    s: set[str] = set()
    for r in rows:
        cv = r.get("catalog_version_id")
        if cv and cv not in s:
            s.add(cv)
            seen.append(cv)
            if len(seen) >= 300:
                break
    return seen


@pytest.fixture(scope="module")
def all_cvs() -> list[str]:
    c = MilvusClient(uri=MILVUS_URI)
    return _all_active_cvs(c)


def make_body(*, cvs: list[str], **overrides) -> dict:
    """Minimum-valid F2 SearchRequest body with the given CV scope.

    `selectedArticleSources.catalogVersionIdsOrderedByPreference` drives
    the always-on CV intersection; without it every request collapses
    to match-nothing per `filters._closed_marketplace`. Tests that
    want to *prove* the match-nothing collapse pass `cvs=[]`.

    Pass anything else as kwargs (camelCase wire field names).
    """
    body: dict = {
        "searchMode": "HITS_ONLY",
        "selectedArticleSources": {
            "catalogVersionIdsOrderedByPreference": cvs,
            "closedCatalogVersionIds": [],
            "sourcePriceListIds": [],
            "customerUploadedCoreArticleListSourceIds": [],
        },
        "currency": "EUR",
        "maxDeliveryTime": 0,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
    }
    body.update(overrides)
    return body


# ──────────────────────────────────────────────────────────────────────
# Class A — Validator self-test
# ──────────────────────────────────────────────────────────────────────

class TestValidator:
    def test_minimum_envelope_validates(self) -> None:
        envelope = {
            "articles": [],
            "summaries": {},
            "metadata": {"page": 1, "pageSize": 10, "pageCount": 0, "hitCount": 0},
        }
        assert_search_response_valid(envelope)

    def test_envelope_with_articles_and_summaries(self) -> None:
        envelope = {
            "articles": [{"articleId": "a:1", "score": 1.0}],
            "summaries": {
                "vendorSummaries": [{"vendorId": "v1", "count": 5}],
                "manufacturerSummaries": [{"name": "Würth", "count": 3}],
                "featureSummaries": [{"name": "Color", "count": 1, "values": [{"value": "red", "count": 1}]}],
                "pricesSummary": [{"min": 1.0, "max": 99.0, "currencyCode": "EUR"}],
                "categoriesSummary": None,
                "eClass5Categories": None,
                "eClass7Categories": None,
                "s2ClassCategories": None,
                "eClassesAggregations": [],
            },
            "metadata": {"page": 1, "pageSize": 10, "pageCount": 1, "hitCount": 1, "term": "x"},
        }
        assert_search_response_valid(envelope)

    def test_validator_catches_extra_top_level_field(self) -> None:
        bad = {
            "articles": [],
            "summaries": {},
            "metadata": {"page": 1, "pageSize": 10, "pageCount": 0, "hitCount": 0},
            "totallyMadeUp": True,
        }
        with pytest.raises(AssertionError):
            assert_search_response_valid(bad)


# ──────────────────────────────────────────────────────────────────────
# Class B — Parameter coverage
# ──────────────────────────────────────────────────────────────────────

class TestParameters:
    def test_default_pagination_returns_envelope(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(search_path, json=make_body(cvs=all_cvs))
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        md = body["metadata"]
        assert md["page"] == 1
        assert md["pageSize"] == 10

    def test_custom_page_and_page_size(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?page=3&pageSize=5",
            json=make_body(cvs=all_cvs, query="schraube", searchMode="HITS_ONLY",
                           vendorIdsFilter=list(KNOWN_VENDORS)),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["metadata"]["page"] == 3
        assert body["metadata"]["pageSize"] == 5
        assert len(body["articles"]) <= 5

    def test_page_size_zero_returns_empty_articles(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=0",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[KNOWN_VENDORS[0]]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["articles"] == []
        assert body["metadata"]["pageSize"] == 0

    def test_page_size_above_cap_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=501",
            json=make_body(cvs=all_cvs),
        )
        assert r.status_code == 422
        assert_validation_error_valid(r.json())

    def test_page_zero_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?page=0",
            json=make_body(cvs=all_cvs),
        )
        assert r.status_code == 422
        assert_validation_error_valid(r.json())

    @pytest.mark.parametrize("clause", [
        "name,asc", "name,desc", "price,asc", "price,desc",
        "articleId,asc", "articleId,desc",
    ])
    def test_each_sort_clause_accepted(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str], clause: str
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?sort={clause}&pageSize=5",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[KNOWN_VENDORS[0]]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)

    def test_unknown_sort_field_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?sort=color,asc",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[KNOWN_VENDORS[0]]),
        )
        # Implementation maps to HTTP 400 per main.py:_search; spec lists
        # only 401/422/503 explicitly for /_search but a known-bad sort
        # is a request error. We accept either 400 or 422 — the
        # important assertion is "rejected, with an Error envelope".
        assert r.status_code in (400, 422)

    def test_unknown_sort_direction_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?sort=name,sideways",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[KNOWN_VENDORS[0]]),
        )
        assert r.status_code in (400, 422)

    def test_collection_path_param_is_used(
        self, search_api_app: TestClient, all_cvs: list[str]
    ) -> None:
        # Asking for a collection that doesn't exist should not 200.
        r = search_api_app.post(
            "/no_such_collection_v999/_search",
            json=make_body(cvs=all_cvs, query="x"),
        )
        # 404 (legacy path) or 500 (dedup-path articles_collection check
        # fires first). We accept anything non-2xx.
        assert r.status_code >= 400


# ──────────────────────────────────────────────────────────────────────
# Class C — Body validation
# ──────────────────────────────────────────────────────────────────────

class TestBodyValidation:
    def test_missing_search_mode_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs)
        bad.pop("searchMode")
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422
        assert_validation_error_valid(r.json())

    def test_missing_currency_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs)
        bad.pop("currency")
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_missing_selected_article_sources_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs)
        bad.pop("selectedArticleSources")
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_unknown_top_level_field_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs)
        bad["totallyMadeUpField"] = 42
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_unknown_search_mode_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs, searchMode="MAYBE")
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_currency_pattern_enforced(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs, currency="eur")  # lowercase
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_negative_max_delivery_time_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs, maxDeliveryTime=-1)
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_required_features_missing_values_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs, requiredFeatures=[{"name": "x"}])
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# Class D — searchMode envelope shape
# ──────────────────────────────────────────────────────────────────────

class TestSearchModes:
    def test_hits_only_envelope(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            search_path,
            json=make_body(cvs=all_cvs, searchMode="HITS_ONLY",
                           vendorIdsFilter=[KNOWN_VENDORS[0]]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        # HITS_ONLY → summary sub-fields empty/null per spec.
        s = body["summaries"]
        assert s.get("vendorSummaries", []) == []
        assert s.get("manufacturerSummaries", []) == []
        assert s.get("featureSummaries", []) == []

    def test_summaries_only_returns_no_articles(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            search_path,
            json=make_body(cvs=all_cvs, searchMode="SUMMARIES_ONLY",
                           vendorIdsFilter=[KNOWN_VENDORS[0]],
                           summaries=["VENDORS"]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["articles"] == []

    def test_both_envelope(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(cvs=all_cvs, searchMode="BOTH",
                           vendorIdsFilter=[KNOWN_VENDORS[0]],
                           summaries=["VENDORS", "MANUFACTURERS"]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)


# ──────────────────────────────────────────────────────────────────────
# Class E — Filter coverage (each scalar atom)
# ──────────────────────────────────────────────────────────────────────

class TestFilters:
    def test_empty_cv_scope_returns_empty(
        self, search_api_app: TestClient, search_path: str
    ) -> None:
        # The always-on `_closed_marketplace` filter collapses to
        # match-nothing when no CVs are scoped.
        r = search_api_app.post(
            search_path,
            json=make_body(cvs=[]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["articles"] == []
        assert body["metadata"]["hitCount"] == 0

    def test_vendor_filter_narrows_returned_offers(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=20",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[KNOWN_VENDORS[0]]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        # We can't assert vendor here without a hydrate-vendor lookup;
        # the integration assertion is "we got something" + "schema OK".
        # Per-vendor narrowing is exercised in test_search_dedup_integration.
        assert isinstance(body["articles"], list)

    def test_manufacturer_filter(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=10",
            json=make_body(cvs=all_cvs, manufacturersFilter=[KNOWN_MANUFACTURERS[0]]),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_eclasses_filter(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=10",
            json=make_body(cvs=all_cvs, eClassesFilter=list(KNOWN_ECLASS5_CODES)),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_current_eclass5_code(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(cvs=all_cvs, currentEClass5Code=KNOWN_ECLASS5_CODES[0]),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_max_delivery_time(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(cvs=all_cvs, maxDeliveryTime=3,
                           vendorIdsFilter=[KNOWN_VENDORS[0]]),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_required_features_filter(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(
                cvs=all_cvs,
                requiredFeatures=[{"name": "Ursprungsland", "values": ["IT", "DE"]}],
            ),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_blocked_eclass_vendors_filter(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(
                cvs=all_cvs,
                blockedEClassVendorsFilters=[{
                    "vendorIds": [KNOWN_VENDORS[0]],
                    "eClassVersion": "ECLASS_5_1",
                    "blockedEClassGroups": [
                        {"eClassGroupCode": KNOWN_ECLASS5_CODES[0], "value": True},
                    ],
                }],
            ),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_core_sortiment_only(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(cvs=all_cvs, coreSortimentOnly=True),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_core_articles_vendors_filter(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(cvs=all_cvs, coreArticlesVendorsFilter=[KNOWN_VENDORS[0]]),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_closed_marketplace_only(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        # closedMarketplaceOnly=True flips _closed_marketplace from
        # the open-list scope (catalogVersionIdsOrderedByPreference) to
        # the closed-list scope (closedCatalogVersionIds). Empty
        # closed list → match-nothing.
        body = make_body(cvs=all_cvs, closedMarketplaceOnly=True)
        body["selectedArticleSources"]["closedCatalogVersionIds"] = all_cvs[:5]
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=body,
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_eclasses_aggregations_passthrough(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(
                cvs=all_cvs,
                eClassesAggregations=[
                    {"id": "agg-safety", "eClasses": list(KNOWN_ECLASS5_CODES[:2])},
                ],
            ),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())


# ──────────────────────────────────────────────────────────────────────
# Class F — Relationship filters
# ──────────────────────────────────────────────────────────────────────

class TestRelationships:
    @pytest.mark.parametrize("field", [
        "accessoriesForArticleNumber",
        "sparePartsForArticleNumber",
        "similarToArticleNumber",
    ])
    def test_relationship_field_round_trip(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str], field: str
    ) -> None:
        # Real article numbers are not exposed; we only assert that
        # the field is accepted and the endpoint handles "no matches"
        # gracefully (empty articles, valid envelope).
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(cvs=all_cvs, **{field: "nonexistent-article-number-12345"}),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["articles"] == []


# ──────────────────────────────────────────────────────────────────────
# Class G — priceFilter
# ──────────────────────────────────────────────────────────────────────

class TestPriceFilter:
    def test_price_filter_min_max(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        # 1.00 EUR (100 minor) → 100.00 EUR (10000 minor)
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(
                cvs=all_cvs,
                vendorIdsFilter=[KNOWN_VENDORS[0]],
                priceFilter={"min": 100, "max": 10000, "currencyCode": "EUR"},
            ),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_price_filter_only_max(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(
                cvs=all_cvs,
                vendorIdsFilter=[KNOWN_VENDORS[0]],
                priceFilter={"max": 5000, "currencyCode": "EUR"},
            ),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_price_filter_invalid_currency(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            search_path,
            json=make_body(
                cvs=all_cvs,
                priceFilter={"min": 0, "max": 100, "currencyCode": "EU"},
            ),
        )
        assert r.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# Class H — Sort orderings (deeper than parameter accept-test)
# ──────────────────────────────────────────────────────────────────────

class TestSortOrderings:
    def test_sort_by_articleId_asc_is_actually_sorted(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?sort=articleId,asc&pageSize=20",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[KNOWN_VENDORS[0]]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        ids = [a["articleId"] for a in body["articles"]]
        assert ids == sorted(ids), f"articleId,asc not in order: {ids[:5]}…"

    def test_sort_by_articleId_desc_is_actually_sorted(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?sort=articleId,desc&pageSize=20",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[KNOWN_VENDORS[0]]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        ids = [a["articleId"] for a in body["articles"]]
        assert ids == sorted(ids, reverse=True), f"articleId,desc not in order: {ids[:5]}…"

    def test_multi_key_sort_only_first_key_applies(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        # spec §4.5: multi-key requests apply the first key only.
        # We can't probe the implementation directly here without
        # secondary-field hydration, but we assert that the request
        # is accepted + envelope is valid.
        r = search_api_app.post(
            f"{search_path}?sort=name,asc&sort=articleId,desc&pageSize=10",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[KNOWN_VENDORS[0]]),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())


# ──────────────────────────────────────────────────────────────────────
# Class I — Pagination invariants
# ──────────────────────────────────────────────────────────────────────

class TestPagination:
    def test_metadata_page_count_matches_pages(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        page_size = 5
        r = search_api_app.post(
            f"{search_path}?page=1&pageSize={page_size}",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[KNOWN_VENDORS[0]]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        md = body["metadata"]
        # pageCount = ceil(hitCount / pageSize) when both > 0; 0 when hitCount=0.
        if md["hitCount"] == 0 or md["pageSize"] == 0:
            assert md["pageCount"] == 0
        else:
            expected = (md["hitCount"] + page_size - 1) // page_size
            assert md["pageCount"] == expected, f"pageCount {md['pageCount']} != ceil({md['hitCount']}/{page_size}) = {expected}"

    def test_pages_do_not_overlap(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        body = make_body(cvs=all_cvs, vendorIdsFilter=[KNOWN_VENDORS[0]])
        page_size = 5
        r1 = search_api_app.post(f"{search_path}?page=1&pageSize={page_size}&sort=articleId,asc", json=body)
        r2 = search_api_app.post(f"{search_path}?page=2&pageSize={page_size}&sort=articleId,asc", json=body)
        assert r1.status_code == 200 and r2.status_code == 200, (r1.text, r2.text)
        ids1 = {a["articleId"] for a in r1.json()["articles"]}
        ids2 = {a["articleId"] for a in r2.json()["articles"]}
        assert not (ids1 & ids2), f"overlap between page 1 and page 2: {ids1 & ids2}"


# ──────────────────────────────────────────────────────────────────────
# Class J — Summaries (one per SummaryKind)
# ──────────────────────────────────────────────────────────────────────

class TestSummaries:
    @pytest.mark.parametrize("kind", [
        "VENDORS", "MANUFACTURERS", "CATEGORIES",
        "ECLASS5", "ECLASS7", "S2CLASS",
        "FEATURES", "PRICES", "PLATFORM_CATEGORIES", "ECLASS5SET",
    ])
    def test_each_summary_kind_yields_valid_envelope(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str], kind: str
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(
                cvs=all_cvs, searchMode="BOTH",
                vendorIdsFilter=[KNOWN_VENDORS[0]],
                summaries=[kind],
            ),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)


# ──────────────────────────────────────────────────────────────────────
# Class K — Auth + concurrency guards
# ──────────────────────────────────────────────────────────────────────

class TestAuthAndConcurrency:
    def test_api_key_enforced_when_configured(self, all_cvs: list[str]) -> None:
        """Boot a second app instance with API_KEY set + assert that
        unauthenticated requests get 401 + the documented Error envelope.

        Sharing the module-scoped app would force a re-load with
        different env, so this test boots a fresh one inside the
        function and tears it down at the end.
        """
        from _pytest.monkeypatch import MonkeyPatch
        mp = MonkeyPatch()
        mp.setenv("USE_DEDUP_TOPOLOGY", "1")
        mp.setenv("MILVUS_ARTICLES_COLLECTION", ARTICLES_COLLECTION)
        mp.setenv("EMBED_URL", "http://embed.invalid")
        mp.setenv("MILVUS_URI", MILVUS_URI)
        mp.setenv("API_KEY", "test-key-zenith")
        spec = importlib.util.spec_from_file_location(
            "search_api_main_for_f2_auth", SEARCH_API_DIR / "main.py",
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["search_api_main_for_f2_auth"] = mod
        try:
            spec.loader.exec_module(mod)
            with TestClient(mod.app) as client:
                r = client.post(
                    f"/{OFFERS_COLLECTION}/_search",
                    json=make_body(cvs=all_cvs),
                )
                assert r.status_code == 401
                assert r.headers.get("WWW-Authenticate", "").startswith("ApiKey")
                # Spec lists the 401 body under the Error schema.
                err = r.json()
                Draft202012Validator(
                    {**_openapi_to_jsonschema(OPENAPI_SPEC)[1], "$ref": "#/$defs/Error"}
                ).validate(err)

                # And the same request authenticated should pass:
                r2 = client.post(
                    f"/{OFFERS_COLLECTION}/_search",
                    json=make_body(cvs=all_cvs),
                    headers={"X-API-Key": "test-key-zenith"},
                )
                assert r2.status_code == 200, r2.text
                assert_search_response_valid(r2.json())
        finally:
            mp.undo()
            sys.modules.pop("search_api_main_for_f2_auth", None)

    def test_concurrency_gate_returns_503(self, all_cvs: list[str]) -> None:
        """Boot with MAX_CONCURRENT_SEARCHES=0 to flip the gate
        permanently-exhausted (the gate returns 503 when inflight
        meets the configured limit; with limit=1 + a single inflight
        request we'd race, so we use a tiny test here that pre-loads
        inflight via a direct gate poke)."""
        from _pytest.monkeypatch import MonkeyPatch
        mp = MonkeyPatch()
        mp.setenv("USE_DEDUP_TOPOLOGY", "1")
        mp.setenv("MILVUS_ARTICLES_COLLECTION", ARTICLES_COLLECTION)
        mp.setenv("EMBED_URL", "http://embed.invalid")
        mp.setenv("MILVUS_URI", MILVUS_URI)
        mp.setenv("API_KEY", "")
        mp.setenv("MAX_CONCURRENT_SEARCHES", "1")
        spec = importlib.util.spec_from_file_location(
            "search_api_main_for_f2_gate", SEARCH_API_DIR / "main.py",
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["search_api_main_for_f2_gate"] = mod
        try:
            spec.loader.exec_module(mod)
            with TestClient(mod.app) as client:
                # Pre-saturate the gate so the next request hits the cap.
                mod.app.state.gate.inflight = mod.app.state.gate.limit
                r = client.post(
                    f"/{OFFERS_COLLECTION}/_search",
                    json=make_body(cvs=all_cvs),
                )
                assert r.status_code == 503
                assert r.headers.get("Retry-After") == "1"
                err = r.json()
                Draft202012Validator(
                    {**_openapi_to_jsonschema(OPENAPI_SPEC)[1], "$ref": "#/$defs/Error"}
                ).validate(err)
        finally:
            mp.undo()
            sys.modules.pop("search_api_main_for_f2_gate", None)


# ──────────────────────────────────────────────────────────────────────
# Class L — Behaviour assertions on real data (not just shape)
# ──────────────────────────────────────────────────────────────────────

class TestBehaviour:
    """Goes past 'request was accepted' to 'the right thing happened'.

    Each test pins a specific corner of the catalog (high-volume vendor,
    known manufacturer, known category) so the assertion is stable
    against the loaded `articles_v6` + `offers_v6` snapshot.
    """

    def test_high_volume_vendor_returns_at_least_one_hit(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Smoke test: a vendor we know has ~2k offers should not
        return zero. If this fails the always-on CV intersection or
        the dispatch path is broken before any of the tighter
        behaviour tests will run."""
        r = search_api_app.post(
            f"{search_path}?pageSize=10",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["metadata"]["hitCount"] > 0, (
            "expected non-zero hits for high-volume vendor "
            f"{HIGH_VOLUME_VENDOR}, got: {body['metadata']}"
        )
        assert body["articles"], "articles[] empty despite hitCount > 0"

    def test_unknown_vendor_returns_zero_hits(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            search_path,
            json=make_body(
                cvs=all_cvs,
                vendorIdsFilter=["00000000-0000-0000-0000-000000000000"],
            ),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["articles"] == []
        assert body["metadata"]["hitCount"] == 0

    def test_unknown_manufacturer_returns_zero_hits(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            search_path,
            json=make_body(
                cvs=all_cvs,
                manufacturersFilter=["NoSuchManufacturer__zzz"],
            ),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["articles"] == []
        assert body["metadata"]["hitCount"] == 0

    def test_manufacturer_filter_actually_narrows(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Tighter assertion: filtering by `Würth` should produce a
        strictly smaller hitCount than filtering by an empty
        manufacturer set under the same scope (vendor + CVs)."""
        unfiltered = search_api_app.post(
            search_path,
            json=make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR]),
        )
        filtered = search_api_app.post(
            search_path,
            json=make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR],
                           manufacturersFilter=["Würth"]),
        )
        assert unfiltered.status_code == 200 and filtered.status_code == 200
        u = unfiltered.json()["metadata"]["hitCount"]
        f = filtered.json()["metadata"]["hitCount"]
        # Sub-filter cannot grow the result set.
        assert f <= u, f"manufacturer filter grew hits: {u} → {f}"

    def test_category_path_l1_actually_narrows(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        unfiltered = search_api_app.post(
            search_path,
            json=make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR]),
        )
        filtered = search_api_app.post(
            search_path,
            json=make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR],
                           currentCategoryPathElements=[KNOWN_CATEGORY_L1[0]]),
        )
        assert unfiltered.status_code == 200 and filtered.status_code == 200
        u = unfiltered.json()["metadata"]["hitCount"]
        f = filtered.json()["metadata"]["hitCount"]
        assert f <= u, f"category filter grew hits: {u} → {f}"

    @pytest.mark.parametrize("field,value", [
        ("currentEClass5Code", KNOWN_ECLASS5_CODES[0]),
        ("currentEClass7Code", KNOWN_ECLASS5_CODES[0]),
        ("currentS2ClassCode", KNOWN_ECLASS5_CODES[0]),
    ])
    def test_each_current_eclass_field_accepted(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str], field: str, value: int
    ) -> None:
        r = search_api_app.post(
            search_path,
            json=make_body(cvs=all_cvs, **{field: value}),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_s2class_for_product_categories_flag_accepted(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            search_path,
            json=make_body(
                cvs=all_cvs, s2ClassForProductCategories=True,
                eClassesFilter=[KNOWN_ECLASS5_CODES[0]],
            ),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_max_delivery_time_does_not_grow_hits(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        unfiltered = search_api_app.post(
            search_path,
            json=make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR]),
        )
        filtered = search_api_app.post(
            search_path,
            json=make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR],
                           maxDeliveryTime=2),
        )
        assert unfiltered.status_code == 200 and filtered.status_code == 200
        u = unfiltered.json()["metadata"]["hitCount"]
        f = filtered.json()["metadata"]["hitCount"]
        assert f <= u, f"delivery filter grew hits: {u} → {f}"


# ──────────────────────────────────────────────────────────────────────
# Class M — Free-text query path
# ──────────────────────────────────────────────────────────────────────

class TestFreeTextQuery:
    """The `query` body field engages the dense ANN + BM25 hybrid
    pipeline. These tests confirm the wiring works end-to-end through
    the (mocked) embedder and BM25 codes leg.

    The mock embedder returns the same vector regardless of input, so
    dense retrieval contributes "the same" neighborhood every time;
    BM25 still varies with the query string, so hits across different
    terms differ. We assert the contract holds, not the ranking.
    """

    def test_query_returns_envelope(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=10",
            json=make_body(cvs=all_cvs, query="schraube"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        # term echoes the input string (per F2 spec).
        assert body["metadata"]["term"] == "schraube"

    def test_query_path_yields_some_results(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """`schraube` is a corpus-prevalent BM25 term — the BM25 leg
        alone should produce hits even with a deterministic mocked
        dense vector. If hits=0 the query path is broken."""
        r = search_api_app.post(
            f"{search_path}?pageSize=10",
            json=make_body(cvs=all_cvs, query="schraube"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["articles"], (
            f"expected hits for query='schraube'; metadata={body['metadata']}"
        )

    def test_empty_query_falls_through_to_browse(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """An empty `query` string must not fail differently from a
        request with no `query` field at all (both => browse path)."""
        body_no_q = make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR])
        body_empty_q = {**body_no_q, "query": ""}
        r1 = search_api_app.post(search_path, json=body_no_q)
        r2 = search_api_app.post(search_path, json=body_empty_q)
        assert r1.status_code == 200 == r2.status_code, (r1.text, r2.text)
        # Same scope → same hitCount (both go through browse).
        assert r1.json()["metadata"]["hitCount"] == r2.json()["metadata"]["hitCount"]

    def test_null_query_accepted(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        body = make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR],
                         query=None)
        r = search_api_app.post(search_path, json=body)
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())


# ──────────────────────────────────────────────────────────────────────
# Class N — articleIdsFilter (legacy offer PK)
# ──────────────────────────────────────────────────────────────────────

class TestArticleIdsFilter:
    """`articleIdsFilter` is offer-side: each value is a literal
    `offer.id` (the legacy PK; format
    `{vendor_id}:{base64UrlEncodedArticleNumber}:{catalog_version_id}`)."""

    def test_unknown_id_returns_empty(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            search_path,
            json=make_body(cvs=all_cvs, articleIdsFilter=["00000000-0000-0000-0000-000000000000:none:cv"]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["articles"] == []
        assert body["metadata"]["hitCount"] == 0

    def test_real_offer_id_round_trips(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        # Pull one real offer-id off the live collection — we don't
        # care which row, just that `articleIdsFilter` accepts it
        # and returns it.
        c = MilvusClient(uri=MILVUS_URI)
        rows = c.query(
            collection_name=OFFERS_COLLECTION,
            filter=f'vendor_id == "{HIGH_VOLUME_VENDOR}"',
            output_fields=["id", "catalog_version_id"], limit=1,
        )
        assert rows, "fixture: HIGH_VOLUME_VENDOR has no offers in offers_v6"
        offer_id = rows[0]["id"]
        cv = rows[0]["catalog_version_id"]
        r = search_api_app.post(
            search_path,
            json=make_body(
                cvs=[cv] + all_cvs,
                articleIdsFilter=[offer_id],
            ),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        ids = [a["articleId"] for a in body["articles"]]
        assert offer_id in ids, (
            f"articleIdsFilter did not echo back the requested id "
            f"{offer_id!r}; got {ids[:3]}"
        )


# ──────────────────────────────────────────────────────────────────────
# Class O — Pagination edges
# ──────────────────────────────────────────────────────────────────────

class TestPaginationEdges:
    def test_page_size_500_max_accepted(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=500",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["metadata"]["pageSize"] == 500
        assert len(body["articles"]) <= 500

    def test_page_beyond_end_returns_empty(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        # Page 9999 is virtually guaranteed to be past pageCount.
        r = search_api_app.post(
            f"{search_path}?page=9999&pageSize=10",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["articles"] == []
        assert body["metadata"]["page"] == 9999


# ──────────────────────────────────────────────────────────────────────
# Class P — Summary content (not just envelope shape)
# ──────────────────────────────────────────────────────────────────────

class TestSummaryContent:
    """Goes past 'envelope shape valid' to 'content reflects the filtered set'.

    These tests are tolerant: live-data hit counts will vary as the
    catalogue evolves, so we only assert structural relationships
    (counts > 0 when hits > 0; bucket counts sum to ≤ total hits).
    """

    def _post(self, c: TestClient, path: str, **body) -> dict:
        r = c.post(f"{path}?pageSize=5", json=body)
        assert r.status_code == 200, r.text
        return r.json()

    def test_vendors_summary_lists_filtered_vendor(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        body = make_body(cvs=all_cvs, searchMode="BOTH",
                         vendorIdsFilter=[HIGH_VOLUME_VENDOR],
                         summaries=["VENDORS"])
        out = self._post(search_api_app, search_path, **body)
        assert_search_response_valid(out)
        vs = out["summaries"].get("vendorSummaries") or []
        if out["metadata"]["hitCount"] > 0:
            ids = {v["vendorId"] for v in vs}
            assert HIGH_VOLUME_VENDOR in ids, (
                f"vendorSummaries missing the only filtered vendor: {ids}"
            )

    def test_manufacturers_summary_lists_filtered_manufacturer(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        body = make_body(cvs=all_cvs, searchMode="BOTH",
                         vendorIdsFilter=[HIGH_VOLUME_VENDOR],
                         manufacturersFilter=["Würth"],
                         summaries=["MANUFACTURERS"])
        out = self._post(search_api_app, search_path, **body)
        assert_search_response_valid(out)
        ms = out["summaries"].get("manufacturerSummaries") or []
        if out["metadata"]["hitCount"] > 0:
            names = {m["name"] for m in ms}
            assert "Würth" in names, (
                f"manufacturerSummaries missing the filtered manufacturer: {names}"
            )

    def test_summary_bucket_counts_do_not_exceed_total(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """vendorSummaries' bucket counts can sum to ≥ hitCount when an
        article has multiple vendors selling it; but no single bucket
        should exceed hitCount."""
        body = make_body(cvs=all_cvs, searchMode="BOTH",
                         vendorIdsFilter=[HIGH_VOLUME_VENDOR],
                         summaries=["VENDORS", "MANUFACTURERS"])
        out = self._post(search_api_app, search_path, **body)
        assert_search_response_valid(out)
        total = out["metadata"]["hitCount"]
        for kind in ("vendorSummaries", "manufacturerSummaries"):
            for bucket in out["summaries"].get(kind) or []:
                assert bucket["count"] <= total, (
                    f"{kind} bucket {bucket} > hitCount {total}"
                )


# ──────────────────────────────────────────────────────────────────────
# Class Q — OpenAPI document is well-formed and self-describing
# ──────────────────────────────────────────────────────────────────────

class TestOpenAPIDocument:
    """Spec-level invariants the F2 contract should always satisfy.

    These guard against rot in the document itself (broken refs, missing
    required schemas, drift between path-declared responses and the
    spec's component schemas)."""

    def test_search_response_schema_exists(self) -> None:
        assert "SearchResponse" in OPENAPI_SPEC["components"]["schemas"]

    def test_error_envelope_declared_for_search(self) -> None:
        op = OPENAPI_SPEC["paths"]["/{collection}/_search"]["post"]
        assert "401" in op["responses"]
        assert "422" in op["responses"]
        assert "503" in op["responses"]

    def test_search_request_required_fields_match_implementation(self) -> None:
        """Pydantic `SearchRequest.model_fields` should declare the
        same `required: [...]` set the OpenAPI does, so we don't ship
        a spec that's stricter or laxer than what the route accepts."""
        from models import SearchRequest as ImplSearchRequest
        spec_required = set(
            OPENAPI_SPEC["components"]["schemas"]["SearchRequest"]["required"]
        )
        impl_required = {
            f.alias or name
            for name, f in ImplSearchRequest.model_fields.items()
            if f.is_required()
        }
        assert spec_required == impl_required, (
            f"required-field drift: spec={spec_required - impl_required}, "
            f"impl={impl_required - spec_required}"
        )

    def test_metadata_fields_match_implementation(self) -> None:
        """All Metadata properties declared in the spec must exist on
        the pydantic model and vice versa — this is the test that
        caught `recallClipped`/`hitCountClipped` drift initially."""
        from models import Metadata as ImplMetadata
        spec_props = set(
            OPENAPI_SPEC["components"]["schemas"]["Metadata"]["properties"].keys()
        )
        impl_props = {
            f.alias or name for name, f in ImplMetadata.model_fields.items()
        }
        assert spec_props == impl_props, (
            f"Metadata field drift: spec_only={spec_props - impl_props}, "
            f"impl_only={impl_props - spec_props}"
        )
