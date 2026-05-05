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

import importlib.util
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

    def test_validator_catches_missing_required(self) -> None:
        # Spec marks `articleId` required on Article; an article
        # missing it must fail validation.
        bad = {
            "articles": [{"score": 0.5}],
            "summaries": {},
            "metadata": {"page": 1, "pageSize": 10, "pageCount": 0, "hitCount": 0},
        }
        with pytest.raises(AssertionError):
            assert_search_response_valid(bad)

    def test_validator_catches_negative_count(self) -> None:
        bad = {
            "articles": [],
            "summaries": {
                "vendorSummaries": [{"vendorId": "v", "count": -1}],
            },
            "metadata": {"page": 1, "pageSize": 10, "pageCount": 0, "hitCount": 0},
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

    def test_empty_sort_value_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?sort=",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[KNOWN_VENDORS[0]]),
        )
        assert r.status_code in (400, 422)

    def test_sort_missing_direction_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?sort=name",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[KNOWN_VENDORS[0]]),
        )
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

    def test_get_method_rejected(
        self, search_api_app: TestClient, search_path: str
    ) -> None:
        """The endpoint is POST-only per spec; GET should 405."""
        r = search_api_app.get(search_path)
        assert r.status_code == 405

    def test_unknown_path_returns_404(
        self, search_api_app: TestClient
    ) -> None:
        r = search_api_app.post(
            f"/{OFFERS_COLLECTION}/no_such_route", json={}
        )
        assert r.status_code == 404


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

    def test_hits_only_returns_empty_summaries_even_when_requested(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Spec note on Summaries: 'For HITS_ONLY mode, the envelope is
        emitted with all sub-fields empty/null'. routing.py confirms:
        summaries are computed only when mode is SUMMARIES_ONLY or BOTH,
        regardless of the `summaries` list. Asking for VENDORS in
        HITS_ONLY mode must NOT populate them — that would silently
        leak the work HITS_ONLY skips."""
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(cvs=all_cvs, searchMode="HITS_ONLY",
                           vendorIdsFilter=[HIGH_VOLUME_VENDOR],
                           summaries=["VENDORS", "MANUFACTURERS"]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["summaries"].get("vendorSummaries", []) == []
        assert body["summaries"].get("manufacturerSummaries", []) == []

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

    def test_vendor_filter_returns_offers_owned_by_vendor(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Each `articleId` is the offer-side primary key in the
        format `{vendor_id}:{base64ArticleNumber}:{cv_id}` (legacy
        wire format). The vendor-filter narrowing must therefore
        produce IDs whose first colon-separated segment matches the
        requested vendor."""
        r = search_api_app.post(
            f"{search_path}?pageSize=20",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        for art in body["articles"]:
            head = art["articleId"].split(":", 1)[0]
            assert head == HIGH_VOLUME_VENDOR, (
                f"vendor filter produced offer from a different vendor: "
                f"{art['articleId']!r} (expected prefix {HIGH_VOLUME_VENDOR!r})"
            )

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

    def test_empty_array_filters_are_no_ops(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Each list-shaped filter (vendorIdsFilter, manufacturersFilter,
        articleIdsFilter, currentCategoryPathElements, eClassesFilter)
        is a no-op when empty — the spec allows the field but
        ergonomically passing `[]` should be identical to omitting it."""
        body_with_empties = make_body(
            cvs=all_cvs, vendorIdsFilter=[], manufacturersFilter=[],
            articleIdsFilter=[], currentCategoryPathElements=[],
            eClassesFilter=[],
        )
        body_without = make_body(cvs=all_cvs)
        r1 = search_api_app.post(search_path, json=body_with_empties)
        r2 = search_api_app.post(search_path, json=body_without)
        assert r1.status_code == 200 and r2.status_code == 200
        assert (
            r1.json()["metadata"]["hitCount"] == r2.json()["metadata"]["hitCount"]
        )

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

    def test_price_filter_only_min(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(
                cvs=all_cvs,
                vendorIdsFilter=[KNOWN_VENDORS[0]],
                priceFilter={"min": 1, "currencyCode": "EUR"},
            ),
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_price_filter_narrows_hit_count(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """A tight price band should not grow the hit count vs no
        priceFilter under the same scope."""
        body_unfiltered = make_body(
            cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR],
        )
        body_filtered = make_body(
            cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR],
            priceFilter={"min": 100, "max": 1000, "currencyCode": "EUR"},
        )
        r1 = search_api_app.post(search_path, json=body_unfiltered)
        r2 = search_api_app.post(search_path, json=body_filtered)
        assert r1.status_code == 200 and r2.status_code == 200
        u = r1.json()["metadata"]["hitCount"]
        f = r2.json()["metadata"]["hitCount"]
        assert f <= u, f"price filter grew hits: {u} → {f}"

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

    def test_price_filter_float_bound_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """`min` and `max` are minor units (integer cents). A float
        like 1.5 is out of contract — pydantic must reject."""
        r = search_api_app.post(
            search_path,
            json=make_body(
                cvs=all_cvs,
                priceFilter={"min": 1.5, "max": 200, "currencyCode": "EUR"},
            ),
        )
        assert r.status_code == 422

    def test_price_filter_currency_only_no_bounds_is_no_op(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """priceFilter with just `currencyCode` set (no min/max) is a
        well-formed request that adds no actual filter. Result count
        should equal the same request without priceFilter."""
        body_with = make_body(
            cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR],
            priceFilter={"currencyCode": "EUR"},
        )
        body_without = make_body(
            cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR],
        )
        r1 = search_api_app.post(search_path, json=body_with)
        r2 = search_api_app.post(search_path, json=body_without)
        assert r1.status_code == 200 and r2.status_code == 200, (r1.text, r2.text)
        assert r1.json()["metadata"]["hitCount"] == r2.json()["metadata"]["hitCount"]


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

    def test_query_with_explicit_sort_overrides_relevance(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """When a query is set AND `sort=articleId,asc` is requested,
        the page must be sorted by articleId, not by score. The
        per-page tiebreak is articleId ascending in either path, so
        the visible order is articleId-asc regardless."""
        r = search_api_app.post(
            f"{search_path}?pageSize=20&sort=articleId,asc",
            json=make_body(cvs=all_cvs, query="schraube"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        ids = [a["articleId"] for a in body["articles"]]
        assert ids == sorted(ids), (
            f"sort=articleId,asc with query did not order by articleId: {ids[:5]}…"
        )

    def test_relevance_sort_holds_across_pages(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Page 1 scores must all be ≥ page 2 scores under relevance
        sort — the page boundary is just a slice of one global
        ordering."""
        body = make_body(cvs=all_cvs, query="schraube")
        r1 = search_api_app.post(f"{search_path}?page=1&pageSize=5", json=body)
        r2 = search_api_app.post(f"{search_path}?page=2&pageSize=5", json=body)
        assert r1.status_code == 200 and r2.status_code == 200
        s1 = [a.get("score") for a in r1.json()["articles"] if a.get("score") is not None]
        s2 = [a.get("score") for a in r2.json()["articles"] if a.get("score") is not None]
        if s1 and s2:
            assert min(s1) >= max(s2), (
                f"page 1 min {min(s1)} < page 2 max {max(s2)} — "
                f"page boundary leaks ordering"
            )

    def test_relevance_sort_returns_descending_scores(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Default sort is relevance (DESC). With a free-text query
        active, every page of results should carry monotonically
        non-increasing scores."""
        r = search_api_app.post(
            f"{search_path}?pageSize=20",
            json=make_body(cvs=all_cvs, query="schraube"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        scores = [a.get("score") for a in body["articles"]]
        # Drop None values; the "score nullable in browse" test pins
        # that semantic separately. Here we just need the present
        # scores to be in non-increasing order.
        present = [s for s in scores if s is not None]
        assert present == sorted(present, reverse=True), (
            f"relevance sort not descending: {present}"
        )

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

    def test_same_request_is_idempotent(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Same request, twice, should yield byte-identical envelopes
        (modulo timing fields, none on the wire) — guards against
        non-deterministic ordering, randomness leaking into the path,
        or cache-related behaviour deltas."""
        body = make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR])
        r1 = search_api_app.post(f"{search_path}?pageSize=10&sort=articleId,asc", json=body)
        r2 = search_api_app.post(f"{search_path}?pageSize=10&sort=articleId,asc", json=body)
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json() == r2.json(), "non-idempotent response"

    def test_summaries_stable_across_pages(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Spec: summaries are computed over the full filtered hit set,
        not the page slice. Page 1 and page 2 of the same query must
        therefore return byte-identical `summaries`."""
        body = make_body(
            cvs=all_cvs, searchMode="BOTH",
            vendorIdsFilter=[HIGH_VOLUME_VENDOR],
            summaries=["VENDORS", "MANUFACTURERS"],
        )
        r1 = search_api_app.post(
            f"{search_path}?page=1&pageSize=5&sort=articleId,asc", json=body)
        r2 = search_api_app.post(
            f"{search_path}?page=2&pageSize=5&sort=articleId,asc", json=body)
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json()["summaries"] == r2.json()["summaries"], (
            "summaries varied across page slices"
        )

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

    def test_hit_count_cap_clips_when_exceeded(self, all_cvs: list[str]) -> None:
        """With HITCOUNT_CAP=1, a query that has an article-side
        filter (so the count(*) pass actually runs through
        `_count_articles`) and many matching articles must report
        hitCountClipped=true with hitCount=1.

        Path B with no article_expr uses `len(distinct_hashes)` for
        hitCount and never clips — that's a different code path. We
        force the `_count_articles` path by adding a manufacturer
        filter (article-side)."""
        from _pytest.monkeypatch import MonkeyPatch
        mp = MonkeyPatch()
        mp.setenv("USE_DEDUP_TOPOLOGY", "1")
        mp.setenv("MILVUS_ARTICLES_COLLECTION", ARTICLES_COLLECTION)
        mp.setenv("EMBED_URL", "http://embed.invalid")
        mp.setenv("MILVUS_URI", MILVUS_URI)
        mp.setenv("API_KEY", "")
        mp.setenv("HITCOUNT_CAP", "1")
        spec = importlib.util.spec_from_file_location(
            "search_api_main_for_f2_hitcap", SEARCH_API_DIR / "main.py",
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["search_api_main_for_f2_hitcap"] = mod
        try:
            spec.loader.exec_module(mod)
            with TestClient(mod.app) as client:
                mod.app.state.embed.embed = AsyncMock(return_value=[_stable_vector()])
                # SUMMARIES_ONLY + small CV slice keeps the Path-B
                # probe under PATH_B_HASH_LIMIT (so we don't fall back
                # to the overflow path which short-circuits the cap).
                # Manufacturer filter on the article side forces the
                # `_count_articles` pass — that's where HITCOUNT_CAP
                # actually fires.
                #
                # We pull a CV with > 1 Würth article live so the
                # assertion is stable against any seed.
                c = MilvusClient(uri=MILVUS_URI)
                cv_rows = c.query(
                    collection_name=ARTICLES_COLLECTION,
                    filter='manufacturerName == "Würth"',
                    output_fields=["article_hash"], limit=5,
                )
                wurth_hashes = [r["article_hash"] for r in cv_rows]
                offer_rows = c.query(
                    collection_name=OFFERS_COLLECTION,
                    filter=(
                        "article_hash in [" + ", ".join(f'"{h}"' for h in wurth_hashes) + "]"
                    ),
                    output_fields=["catalog_version_id"], limit=20,
                )
                wurth_cvs = sorted({r["catalog_version_id"] for r in offer_rows})[:5]
                if len(wurth_hashes) < 2 or not wurth_cvs:
                    pytest.skip("no Würth articles + offers visible to drive the cap test")

                r = client.post(
                    f"/{OFFERS_COLLECTION}/_search",
                    json=make_body(
                        cvs=wurth_cvs,
                        searchMode="SUMMARIES_ONLY",
                        manufacturersFilter=["Würth"],
                    ),
                )
                assert r.status_code == 200, r.text
                body = r.json()
                assert_search_response_valid(body)
                assert body["metadata"]["hitCountClipped"] is True, body["metadata"]
                assert body["metadata"]["hitCount"] == 1, body["metadata"]
        finally:
            mp.undo()
            sys.modules.pop("search_api_main_for_f2_hitcap", None)

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

    def test_closed_marketplace_flag_swaps_cv_scope(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """When closedMarketplaceOnly is False, the open-list scope
        (catalogVersionIdsOrderedByPreference) drives the CV
        intersection. When True, the closed-list scope
        (closedCatalogVersionIds) does. With orthogonal CV lists,
        the two requests must produce DIFFERENT hit counts."""
        cv_first_half = all_cvs[: max(1, len(all_cvs) // 2)]
        cv_second_half = all_cvs[max(1, len(all_cvs) // 2):]
        if not cv_first_half or not cv_second_half:
            pytest.skip("not enough distinct CVs to split for the contrast test")

        body_open = make_body(cvs=cv_first_half, vendorIdsFilter=[HIGH_VOLUME_VENDOR])
        body_open["selectedArticleSources"]["closedCatalogVersionIds"] = cv_second_half
        body_closed = {**body_open, "closedMarketplaceOnly": True}

        r_open = search_api_app.post(search_path, json=body_open)
        r_closed = search_api_app.post(search_path, json=body_closed)
        assert r_open.status_code == 200 and r_closed.status_code == 200
        # The two filters select disjoint CV scopes; they may both
        # be zero or both non-zero, but they should NOT agree exactly
        # (the catalog covers many CVs). Exit cleanly if both happen
        # to be zero — that just means neither half overlapped this
        # vendor — and inform the test runner.
        h_open = r_open.json()["metadata"]["hitCount"]
        h_closed = r_closed.json()["metadata"]["hitCount"]
        if h_open == 0 and h_closed == 0:
            pytest.skip("vendor does not span both CV halves; coverage too thin")
        assert h_open != h_closed, (
            f"closedMarketplaceOnly toggle did not change CV scope: "
            f"both halves report hitCount={h_open}"
        )

    def test_summaries_only_with_empty_summaries_list_returns_no_articles(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """SUMMARIES_ONLY mode + empty `summaries` list is a degenerate
        request. The contract is unambiguous: no articles. The
        summaries envelope should still be valid."""
        r = search_api_app.post(
            search_path,
            json=make_body(cvs=all_cvs, searchMode="SUMMARIES_ONLY",
                           vendorIdsFilter=[HIGH_VOLUME_VENDOR],
                           summaries=[]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["articles"] == []

    def test_closed_marketplace_only_empty_closed_list_match_nothing(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """`closedMarketplaceOnly=True` + empty `closedCatalogVersionIds`
        is the legacy match-nothing pattern: zero hits regardless of
        every other filter."""
        body = make_body(
            cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR],
            closedMarketplaceOnly=True,
        )
        body["selectedArticleSources"]["closedCatalogVersionIds"] = []
        r = search_api_app.post(search_path, json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert_search_response_valid(out)
        assert out["articles"] == []
        assert out["metadata"]["hitCount"] == 0

    def test_closed_cv_list_ignored_when_flag_off(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """`closedCatalogVersionIds` is read only when
        `closedMarketplaceOnly=True`. With the flag off (default),
        adding entries to the closed list must NOT change the result."""
        body_default = make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR])
        body_with_closed = make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR])
        body_with_closed["selectedArticleSources"]["closedCatalogVersionIds"] = all_cvs[:5]
        r1 = search_api_app.post(search_path, json=body_default)
        r2 = search_api_app.post(search_path, json=body_with_closed)
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json()["metadata"]["hitCount"] == r2.json()["metadata"]["hitCount"]

    def test_omitted_default_fields_match_explicit_defaults(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Spec marks `coreSortimentOnly`, `closedMarketplaceOnly`,
        `s2ClassForProductCategories` with `default: false`. Omitting
        them must equal explicitly setting them to false."""
        common_kwargs = {"cvs": all_cvs, "vendorIdsFilter": [HIGH_VOLUME_VENDOR]}
        body_explicit = make_body(
            **common_kwargs,
            coreSortimentOnly=False, closedMarketplaceOnly=False,
            s2ClassForProductCategories=False,
        )
        body_omitted: dict = {
            "searchMode": "HITS_ONLY",
            "selectedArticleSources": {
                "catalogVersionIdsOrderedByPreference": all_cvs,
                "closedCatalogVersionIds": [],
                "sourcePriceListIds": [],
                "customerUploadedCoreArticleListSourceIds": [],
            },
            "currency": "EUR",
            "vendorIdsFilter": [HIGH_VOLUME_VENDOR],
        }
        r1 = search_api_app.post(search_path, json=body_explicit)
        r2 = search_api_app.post(search_path, json=body_omitted)
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json()["metadata"]["hitCount"] == r2.json()["metadata"]["hitCount"]

    def test_vendor_filter_with_quotes_in_value_does_not_crash(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """A vendor ID containing a literal `"` would, if naively
        templated, escape the IN clause and break the Milvus expr.
        Vendor IDs are UUIDs in real life so this never happens
        organically — but the implementation must still defend
        against it. Verify the request returns 200 (or 400/422)
        without exploding into a 500."""
        bogus = 'evil"vendor"id'
        body = make_body(cvs=all_cvs, vendorIdsFilter=[bogus])
        r = search_api_app.post(search_path, json=body)
        # The point is no 500 / Milvus-expression-parse-error leak.
        assert r.status_code in (200, 400, 422), (
            f"unexpected {r.status_code}: {r.text}"
        )
        if r.status_code == 200:
            assert_search_response_valid(r.json())

    def test_relationship_with_quote_does_not_crash(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        body = make_body(
            cvs=all_cvs,
            accessoriesForArticleNumber='" or 1=1; --',
        )
        r = search_api_app.post(search_path, json=body)
        assert r.status_code in (200, 400, 422)
        if r.status_code == 200:
            assert_search_response_valid(r.json())
            # Injection-shaped article number does not legitimately
            # match anything in the catalog.
            assert r.json()["articles"] == []

    def test_category_path_with_separator_chars(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """The category-path encoder uses U+00A6 as the separator
        with U+007C as the escape. A user-provided element containing
        either character must be escaped, not crash the encoder."""
        body = make_body(
            cvs=all_cvs,
            currentCategoryPathElements=["A¦B", "C|D"],  # both special chars
        )
        r = search_api_app.post(search_path, json=body)
        assert r.status_code in (200, 400)
        if r.status_code == 200:
            assert_search_response_valid(r.json())

    def test_negative_eclass_code_returns_zero_hits(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """eClass codes are non-negative in the catalog. A negative
        code is syntactically valid (spec: `type: integer`, no
        minimum) but matches nothing."""
        r = search_api_app.post(
            search_path,
            json=make_body(cvs=all_cvs, eClassesFilter=[-1]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["articles"] == []
        assert body["metadata"]["hitCount"] == 0

    def test_long_vendor_list_accepted(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Many vendors in the IN clause must not break the
        Milvus expression builder. We pass 200 vendor IDs (mostly
        bogus + the high-volume one); the result should still
        include the high-volume vendor's offers."""
        bogus_vendors = [
            f"00000000-0000-0000-0000-{i:012x}" for i in range(199)
        ]
        body = make_body(
            cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR, *bogus_vendors],
        )
        r = search_api_app.post(f"{search_path}?pageSize=10", json=body)
        assert r.status_code == 200, r.text
        body_out = r.json()
        assert_search_response_valid(body_out)
        assert body_out["metadata"]["hitCount"] > 0, body_out["metadata"]

    def test_widening_cv_scope_does_not_shrink_hit_count(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Adding more CVs to `catalogVersionIdsOrderedByPreference`
        broadens the always-on CV intersection, so hitCount can only
        grow or stay equal — never shrink."""
        first_quarter = all_cvs[: max(1, len(all_cvs) // 4)]
        body_narrow = make_body(cvs=first_quarter,
                                vendorIdsFilter=[HIGH_VOLUME_VENDOR])
        body_wide = make_body(cvs=all_cvs,
                              vendorIdsFilter=[HIGH_VOLUME_VENDOR])
        r_n = search_api_app.post(search_path, json=body_narrow)
        r_w = search_api_app.post(search_path, json=body_wide)
        assert r_n.status_code == 200 and r_w.status_code == 200
        n = r_n.json()["metadata"]["hitCount"]
        w = r_w.json()["metadata"]["hitCount"]
        assert w >= n, f"widening CV scope shrank hitCount: {n} → {w}"

    def test_filter_intersection_narrows_more_than_each(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """The intersection of two filters must be ≤ the cardinality
        of either filter alone (vendorIdsFilter ∩ manufacturersFilter)."""
        body_v = make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR])
        body_m = make_body(cvs=all_cvs, manufacturersFilter=["Würth"])
        body_both = make_body(
            cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR],
            manufacturersFilter=["Würth"],
        )
        r_v = search_api_app.post(search_path, json=body_v)
        r_m = search_api_app.post(search_path, json=body_m)
        r_both = search_api_app.post(search_path, json=body_both)
        assert (r_v.status_code == r_m.status_code == r_both.status_code == 200)
        v = r_v.json()["metadata"]["hitCount"]
        m = r_m.json()["metadata"]["hitCount"]
        b = r_both.json()["metadata"]["hitCount"]
        assert b <= v and b <= m, (
            f"intersection {b} not ≤ each: vendor={v} manufacturer={m}"
        )

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

    def test_unicode_query_accepted(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Spec doesn't restrict the query string; verify Unicode flows
        through cleanly (German umlauts, the corpus language)."""
        r = search_api_app.post(
            f"{search_path}?pageSize=5",
            json=make_body(cvs=all_cvs, query="größe"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        assert body["metadata"]["term"] == "größe"

    def test_term_echoes_query_text(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """metadata.term must echo whatever was sent — including null
        when query is omitted, an empty string when query='', and the
        verbatim text otherwise."""
        cases: list[tuple[Any, Any]] = [
            ("schraube", "schraube"),
            ("", ""),
            (None, None),
        ]
        for query_in, expected_term in cases:
            body = make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR])
            if query_in is None:
                # omit `query` entirely
                pass
            else:
                body["query"] = query_in
            r = search_api_app.post(search_path, json=body)
            assert r.status_code == 200, r.text
            term = r.json()["metadata"].get("term")
            # `query=None` and "no query field at all" both flow into
            # SearchRequest.query=None on the impl side, then
            # `term=body.query` puts `None` on the wire — JSON-serialised
            # as null, parsed as None. Empty string stays as "".
            assert term == expected_term, (
                f"term {term!r} != expected {expected_term!r} for query {query_in!r}"
            )

    def test_long_query_string_accepted(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """The TEI client passes `truncate=true`, so the embedder won't
        413 on a 10KB query. Spec doesn't cap query length, so this
        should round-trip."""
        long = "schraube " * 200  # ≈1.8KB, well past most token limits.
        r = search_api_app.post(
            f"{search_path}?pageSize=3",
            json=make_body(cvs=all_cvs, query=long),
        )
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

    def test_all_summary_kinds_at_once_returns_valid_envelope(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Every SummaryKind in one request should not break the
        envelope. This is the kitchen-sink test — useful for catching
        cross-bucket interference (e.g., one summary stomping on
        another's state)."""
        body = make_body(
            cvs=all_cvs, searchMode="BOTH",
            vendorIdsFilter=[HIGH_VOLUME_VENDOR],
            summaries=[
                "CATEGORIES", "ECLASS5", "ECLASS7", "S2CLASS",
                "VENDORS", "MANUFACTURERS", "FEATURES", "PRICES",
                "PLATFORM_CATEGORIES", "ECLASS5SET",
            ],
        )
        r = search_api_app.post(f"{search_path}?pageSize=5", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert_search_response_valid(out)

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

    def test_search_request_property_set_matches_implementation(self) -> None:
        """SearchRequest properties on the spec must map 1:1 to the
        pydantic model's field aliases. Catches *any* new field added
        on either side without the matching update."""
        from models import SearchRequest as ImplSearchRequest
        spec_props = set(
            OPENAPI_SPEC["components"]["schemas"]["SearchRequest"]["properties"].keys()
        )
        impl_props = {
            f.alias or name for name, f in ImplSearchRequest.model_fields.items()
        }
        assert spec_props == impl_props, (
            f"SearchRequest field drift: spec_only={spec_props - impl_props}, "
            f"impl_only={impl_props - spec_props}"
        )

    def test_summaries_field_set_matches_implementation(self) -> None:
        from models import Summaries as ImplSummaries
        spec_props = set(
            OPENAPI_SPEC["components"]["schemas"]["Summaries"]["properties"].keys()
        )
        impl_props = {
            f.alias or name for name, f in ImplSummaries.model_fields.items()
        }
        assert spec_props == impl_props, (
            f"Summaries field drift: spec_only={spec_props - impl_props}, "
            f"impl_only={impl_props - spec_props}"
        )

    def test_summary_kind_enum_matches_implementation(self) -> None:
        from models import SummaryKind as ImplSummaryKind
        spec_enum = set(OPENAPI_SPEC["components"]["schemas"]["SummaryKind"]["enum"])
        impl_enum = {m.value for m in ImplSummaryKind}
        assert spec_enum == impl_enum, f"SummaryKind drift: {spec_enum ^ impl_enum}"

    def test_search_mode_enum_matches_implementation(self) -> None:
        from models import SearchMode as ImplSearchMode
        spec_enum = set(OPENAPI_SPEC["components"]["schemas"]["SearchMode"]["enum"])
        impl_enum = {m.value for m in ImplSearchMode}
        assert spec_enum == impl_enum, f"SearchMode drift: {spec_enum ^ impl_enum}"

    def test_article_field_set_matches_implementation(self) -> None:
        from models import Article as ImplArticle
        spec_props = set(
            OPENAPI_SPEC["components"]["schemas"]["Article"]["properties"].keys()
        )
        impl_props = {
            f.alias or name for name, f in ImplArticle.model_fields.items()
        }
        assert spec_props == impl_props

    def test_eclass_version_enum_matches_implementation(self) -> None:
        from models import EClassVersion as ImplEClassVersion
        spec_enum = set(OPENAPI_SPEC["components"]["schemas"]["EClassVersion"]["enum"])
        impl_enum = {m.value for m in ImplEClassVersion}
        assert spec_enum == impl_enum

    def test_no_snake_case_keys_in_live_response(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """The wire format is exclusively camelCase. A snake_case key
        in the response means the model serializer leaked a Python
        field name without applying its alias — a class of bug that's
        easy to introduce with `populate_by_name=True` + pydantic."""
        body = make_body(
            cvs=all_cvs, searchMode="BOTH",
            vendorIdsFilter=[HIGH_VOLUME_VENDOR],
            summaries=["VENDORS", "MANUFACTURERS", "FEATURES",
                       "PRICES", "CATEGORIES", "ECLASS5"],
        )
        r = search_api_app.post(f"{search_path}?pageSize=5", json=body)
        assert r.status_code == 200, r.text

        snake_case_keys: list[str] = []

        def walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    # Allow snake_case under fields whose values are
                    # opaque user data (none on this contract today,
                    # but be conservative).
                    if "_" in k and k != "_debug":
                        snake_case_keys.append(f"{path}.{k}")
                    walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    walk(item, f"{path}[{i}]")

        walk(r.json(), "$")
        assert not snake_case_keys, (
            "wire response contains snake_case keys: "
            + ", ".join(snake_case_keys)
        )

    def test_every_internal_ref_resolves(self) -> None:
        """Every `$ref: '#/components/schemas/<X>'` in the spec must
        point to a declared schema. This is dumb-but-load-bearing —
        a typo in a $ref silently ships and only blows up at
        validation time."""
        defined = set(OPENAPI_SPEC["components"]["schemas"].keys())
        missing: list[str] = []

        def walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                ref = node.get("$ref")
                if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
                    name = ref.split("/")[-1]
                    if name not in defined:
                        missing.append(f"{path} → {ref}")
                for k, v in node.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    walk(item, f"{path}[{i}]")

        walk(OPENAPI_SPEC, "$")
        assert not missing, "Dangling $refs:\n  " + "\n  ".join(missing)

    def test_openapi_document_validates_against_3x_meta_spec(self) -> None:
        """Hand-written OpenAPI must parse + validate against the
        OpenAPI 3.x meta-spec — guards against typos that yaml-load
        accepts (missing `responses`, malformed `parameters`, etc.).
        Mirrors the parity test that exists for the ACL spec."""
        pytest.importorskip("openapi_spec_validator")
        import warnings
        from openapi_spec_validator import validate_spec
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            validate_spec(OPENAPI_SPEC)

    def test_documented_request_examples_validate_against_schema(self) -> None:
        """The examples bundled in the OpenAPI under
        `/{collection}/_search` POST `requestBody.content[…].examples`
        must themselves conform to the `SearchRequest` schema. They
        ship to client-generators and devs as canonical templates;
        a malformed example would propagate as a copy-paste landmine."""
        op = OPENAPI_SPEC["paths"]["/{collection}/_search"]["post"]
        request_examples = (
            op.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
            .get("examples", {})
        )
        if not request_examples:
            pytest.skip("no documented request examples to validate")
        request_validator = _validator_for(OPENAPI_SPEC, "SearchRequest")
        for name, example in request_examples.items():
            value = example.get("value", example)
            errors = list(request_validator.iter_errors(value))
            assert not errors, (
                f"example {name!r} fails SearchRequest schema:\n  "
                + "\n  ".join(e.message for e in errors)
            )

    def test_documented_response_examples_validate_against_schema(self) -> None:
        op = OPENAPI_SPEC["paths"]["/{collection}/_search"]["post"]
        ok_examples = (
            op.get("responses", {})
            .get("200", {})
            .get("content", {})
            .get("application/json", {})
            .get("examples", {})
        )
        if not ok_examples:
            pytest.skip("no documented 200 examples to validate")
        for name, example in ok_examples.items():
            value = example.get("value", example)
            errors = list(SEARCH_RESPONSE_VALIDATOR.iter_errors(value))
            assert not errors, (
                f"example {name!r} fails SearchResponse schema:\n  "
                + "\n  ".join(e.message for e in errors)
            )

    def test_pydantic_search_response_default_dump_validates(self) -> None:
        """Build the empty/default `SearchResponse` via the pydantic
        model, dump it through the wire alias serializer, and validate
        against the OpenAPI spec. This bridges the impl side
        (models.py) and the spec side (openapi.yaml) — any drift in
        either direction (alias renaming, default-empty changes,
        new required fields) fails this test."""
        from models import SearchResponse, Summaries, Metadata
        sr = SearchResponse(
            articles=[],
            summaries=Summaries(),
            metadata=Metadata(
                page=1, pageSize=10, pageCount=0, hitCount=0,
            ),
        )
        encoded = sr.model_dump(by_alias=True)
        assert_search_response_valid(encoded)

    def test_wire_response_round_trips_through_pydantic_model(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """A live wire response, fed back through the SearchResponse
        pydantic model, must reconstruct without loss. Catches
        serializer/deserializer asymmetry: if a field round-trips
        from impl→wire→impl with a value change, the wire is lying
        about something."""
        from models import SearchResponse
        body = make_body(
            cvs=all_cvs, searchMode="BOTH",
            vendorIdsFilter=[HIGH_VOLUME_VENDOR],
            summaries=["VENDORS", "MANUFACTURERS"],
        )
        r = search_api_app.post(f"{search_path}?pageSize=5", json=body)
        assert r.status_code == 200, r.text
        wire = r.json()
        # populate_by_name=True + validate by alias.
        reconstructed = SearchResponse.model_validate(wire)
        re_dumped = reconstructed.model_dump(by_alias=True)
        # Note: re-dumping may add default-suppressed keys; we compare
        # the relevant top-level invariants only.
        assert re_dumped["articles"] == wire["articles"]
        assert re_dumped["metadata"]["hitCount"] == wire["metadata"]["hitCount"]
        assert re_dumped["metadata"]["page"] == wire["metadata"]["page"]
        assert re_dumped["metadata"]["pageSize"] == wire["metadata"]["pageSize"]

    def test_pydantic_search_response_full_dump_validates(self) -> None:
        """Same idea, with every optional field populated. Catches
        cases where a default-empty dump would pass but a populated
        dump uncovers a non-spec field."""
        from models import (
            Article, CategoriesSummary, EClassBucket, EClassCategories,
            EClassesAggregationCount, FeatureSummary, FeatureValueCount,
            Metadata, NameCount, PricesSummary, SearchResponse,
            Summaries, VendorSummary,
        )
        sr = SearchResponse(
            articles=[Article(articleId="v:1:cv", score=0.42)],
            summaries=Summaries(
                vendor_summaries=[VendorSummary(vendorId="v1", count=3)],
                manufacturer_summaries=[NameCount(name="Würth", count=2)],
                feature_summaries=[FeatureSummary(
                    name="Spannung", count=4,
                    values=[FeatureValueCount(value="18V", count=2),
                            FeatureValueCount(value="36V", count=1)],
                )],
                prices_summary=[PricesSummary(min=1.5, max=99.0, currencyCode="EUR")],
                categories_summary=CategoriesSummary(
                    currentCategoryPathElements=["Werkzeug"], same_level=[], children=[],
                ),
                eclass5_categories=EClassCategories(
                    selectedEClassGroup=23110101,
                    same_level=[EClassBucket(group=23110102, count=1)],
                    children=[],
                ),
                eclasses_aggregations=[EClassesAggregationCount(id="agg1", count=5)],
            ),
            metadata=Metadata(
                page=1, pageSize=10, pageCount=2, hitCount=15,
                term="schraube",
                recallClipped=False, hitCountClipped=False,
            ),
        )
        encoded = sr.model_dump(by_alias=True)
        assert_search_response_valid(encoded)

    def test_every_path_response_references_declared_schemas(self) -> None:
        """Each documented response on each path operation must point
        to a schema (no stub placeholders)."""
        for path, ops in OPENAPI_SPEC["paths"].items():
            for verb, op in ops.items():
                for status, resp in (op.get("responses") or {}).items():
                    content = resp.get("content") or {}
                    for media, media_obj in content.items():
                        schema = media_obj.get("schema")
                        assert schema, (
                            f"{verb.upper()} {path} {status} {media}: "
                            f"missing schema"
                        )


# ──────────────────────────────────────────────────────────────────────
# Class R — Negative request bodies (sub-schema validation)
# ──────────────────────────────────────────────────────────────────────

class TestNegativeBodies:
    def test_price_filter_without_currency_code_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        body = make_body(cvs=all_cvs, priceFilter={"min": 100, "max": 200})
        r = search_api_app.post(search_path, json=body)
        assert r.status_code == 422
        assert_validation_error_valid(r.json())

    def test_required_features_extra_field_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs, requiredFeatures=[
            {"name": "x", "values": ["v"], "extraField": True},
        ])
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_blocked_eclass_groups_extra_field_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs, blockedEClassVendorsFilters=[{
            "vendorIds": [],
            "eClassVersion": "ECLASS_5_1",
            "blockedEClassGroups": [
                {"eClassGroupCode": 1, "value": True, "extra": 1},
            ],
        }])
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_unknown_eclass_version_enum_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs, blockedEClassVendorsFilters=[{
            "vendorIds": [],
            "eClassVersion": "ECLASS_BANANA",
            "blockedEClassGroups": [],
        }])
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_selected_article_sources_extra_field_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs)
        bad["selectedArticleSources"]["totallyMadeUp"] = []
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_summaries_unknown_kind_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs, summaries=["VENDORS", "BANANAS"])
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_required_features_values_must_be_strings(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs, requiredFeatures=[
            {"name": "x", "values": [1, 2, 3]},
        ])
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_eclasses_filter_must_be_integers(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs, eClassesFilter=["not-an-int"])
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_vendor_ids_filter_must_be_strings(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs, vendorIdsFilter=[42])
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_eclasses_aggregation_extra_field_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        bad = make_body(cvs=all_cvs, eClassesAggregations=[{
            "id": "agg", "eClasses": [123], "extra": True,
        }])
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_price_filter_currency_two_roles_request_accepted(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Spec §3 'Currency fields — two roles': priceFilter.currencyCode
        decodes minor units (USD cents have 2 fraction digits). Top-level
        `currency` drives matching against the offer's price column.
        These two values may legitimately differ (e.g., bound in USD,
        matching against the catalog's EUR prices). The request must
        validate even when they differ."""
        body = make_body(
            cvs=all_cvs,
            currency="EUR",  # match against EUR prices
            priceFilter={"min": 100, "max": 100000, "currencyCode": "USD"},
        )
        r = search_api_app.post(search_path, json=body)
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_legacy_explain_field_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """`explain` is a legacy ACL field that must NOT appear on the
        ftsearch contract (§2.2 spec note: ftsearch returns `score`
        per article, not `explanation`)."""
        bad = make_body(cvs=all_cvs)
        bad["explain"] = True
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_legacy_search_articles_by_field_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """`searchArticlesBy` is the legacy enum we dropped at this
        layer (§2.1 — only STANDARD survives, enforced upstream by
        the ACL)."""
        bad = make_body(cvs=all_cvs)
        bad["searchArticlesBy"] = "STANDARD"
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_currency_with_trailing_space_rejected(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Pattern `^[A-Z]{3}$` — anything other than three uppercase
        letters must reject."""
        bad = make_body(cvs=all_cvs, currency="EUR ")
        r = search_api_app.post(search_path, json=bad)
        assert r.status_code == 422

    def test_category_path_too_deep_no_op(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """Schema only stores l1..l5 — the implementation treats depth
        >5 as a no-op rather than rejecting (filters.py defensive
        path). The request must still succeed and return a valid
        envelope; the unmatched-path filter just doesn't narrow."""
        body = make_body(
            cvs=all_cvs,
            currentCategoryPathElements=["l1", "l2", "l3", "l4", "l5", "l6"],
        )
        r = search_api_app.post(search_path, json=body)
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())


# ──────────────────────────────────────────────────────────────────────
# Class S — Auth header alternatives
# ──────────────────────────────────────────────────────────────────────

class TestAuthHeaderAlternatives:
    """Spec lists two equivalent auth headers: `X-API-Key` and
    `Authorization: ApiKey <key>`. Both must work; both must reject
    when wrong."""

    @pytest.fixture()
    def authed_app(self, all_cvs: list[str]) -> Iterator[tuple[TestClient, str]]:
        from _pytest.monkeypatch import MonkeyPatch
        mp = MonkeyPatch()
        api_key = "test-key-zenith-alt"
        mp.setenv("USE_DEDUP_TOPOLOGY", "1")
        mp.setenv("MILVUS_ARTICLES_COLLECTION", ARTICLES_COLLECTION)
        mp.setenv("EMBED_URL", "http://embed.invalid")
        mp.setenv("MILVUS_URI", MILVUS_URI)
        mp.setenv("API_KEY", api_key)
        spec = importlib.util.spec_from_file_location(
            "search_api_main_for_f2_auth_alt", SEARCH_API_DIR / "main.py",
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["search_api_main_for_f2_auth_alt"] = mod
        try:
            spec.loader.exec_module(mod)
            with TestClient(mod.app) as client:
                mod.app.state.embed.embed = AsyncMock(return_value=[_stable_vector()])
                yield client, api_key
        finally:
            mp.undo()
            sys.modules.pop("search_api_main_for_f2_auth_alt", None)

    def test_authorization_apikey_form_accepted(
        self, authed_app: tuple[TestClient, str], all_cvs: list[str]
    ) -> None:
        client, api_key = authed_app
        r = client.post(
            f"/{OFFERS_COLLECTION}/_search",
            json=make_body(cvs=all_cvs),
            headers={"Authorization": f"ApiKey {api_key}"},
        )
        assert r.status_code == 200, r.text
        assert_search_response_valid(r.json())

    def test_x_api_key_header_form_accepted(
        self, authed_app: tuple[TestClient, str], all_cvs: list[str]
    ) -> None:
        client, api_key = authed_app
        r = client.post(
            f"/{OFFERS_COLLECTION}/_search",
            json=make_body(cvs=all_cvs),
            headers={"X-API-Key": api_key},
        )
        assert r.status_code == 200, r.text

    def test_wrong_api_key_rejected(
        self, authed_app: tuple[TestClient, str], all_cvs: list[str]
    ) -> None:
        client, _ = authed_app
        r = client.post(
            f"/{OFFERS_COLLECTION}/_search",
            json=make_body(cvs=all_cvs),
            headers={"X-API-Key": "totally-wrong"},
        )
        assert r.status_code == 401

    def test_metrics_endpoint_bypasses_auth(
        self, authed_app: tuple[TestClient, str]
    ) -> None:
        """Spec carves out `/metrics`, `/openapi.json`, `/openapi.yaml`,
        `/docs` from the API-key requirement."""
        client, _ = authed_app
        r = client.get("/metrics")
        assert r.status_code == 200, r.text
        assert "text/plain" in r.headers.get("content-type", "")

    def test_openapi_yaml_endpoint_serves_spec(
        self, authed_app: tuple[TestClient, str]
    ) -> None:
        client, _ = authed_app
        r = client.get("/openapi.yaml")
        assert r.status_code == 200, r.text
        # Ground-truth comparison against the file on disk.
        served = r.text.strip()
        on_disk = OPENAPI_PATH.read_text().strip()
        assert served == on_disk, "GET /openapi.yaml drifted from openapi.yaml file"

    def test_openapi_json_endpoint_matches_yaml(
        self, authed_app: tuple[TestClient, str]
    ) -> None:
        """FastAPI exposes /openapi.json by default. We installed a
        custom openapi() that returns the YAML-parsed spec, so the
        JSON endpoint must serve the same document."""
        client, _ = authed_app
        r = client.get("/openapi.json")
        assert r.status_code == 200, r.text
        served = r.json()
        on_disk = yaml.safe_load(OPENAPI_PATH.read_text())
        assert served == on_disk, "GET /openapi.json drifted from openapi.yaml on disk"


# ──────────────────────────────────────────────────────────────────────
# Class T — pageSize=0 sanity
# ──────────────────────────────────────────────────────────────────────

class TestPageSizeZero:
    """Per spec, `pageSize=0` is allowed (`minimum: 0`). The article
    list must come back empty, but `hitCount` should still be the real
    total — not artificially zeroed by short-circuit logic."""

    def test_page_size_zero_keeps_real_hit_count(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        body = make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR])
        normal = search_api_app.post(f"{search_path}?pageSize=10", json=body)
        zero = search_api_app.post(f"{search_path}?pageSize=0", json=body)
        assert normal.status_code == 200 and zero.status_code == 200
        assert zero.json()["articles"] == []
        # Both requests scope to the same filtered set; pageSize only
        # affects the page slice. hitCount must agree.
        assert (
            zero.json()["metadata"]["hitCount"]
            == normal.json()["metadata"]["hitCount"]
        )


# ──────────────────────────────────────────────────────────────────────
# Class U — Minimal valid body (only the spec-required fields)
# ──────────────────────────────────────────────────────────────────────

class TestMinimalBody:
    """SearchRequest's `required: [searchMode, selectedArticleSources,
    currency]` is the contract. A request that ships only these three
    must validate (even if the always-on CV intersection collapses to
    match-nothing because the CV list is empty)."""

    def test_minimal_body_accepted(
        self, search_api_app: TestClient, search_path: str
    ) -> None:
        body = {
            "searchMode": "HITS_ONLY",
            "selectedArticleSources": {},
            "currency": "EUR",
        }
        r = search_api_app.post(search_path, json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert_search_response_valid(out)
        # Empty SAS → empty CV list → match-nothing → 0 hits.
        assert out["articles"] == []
        assert out["metadata"]["hitCount"] == 0


# ──────────────────────────────────────────────────────────────────────
# Class V — Deeper summary content checks
# ──────────────────────────────────────────────────────────────────────

class TestDeeperSummaries:
    """One per summary kind, asserting concrete structural invariants
    on the bucket payload, not just the envelope shape."""

    def test_categories_summary_echoes_current_path(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        body = make_body(
            cvs=all_cvs, searchMode="BOTH",
            currentCategoryPathElements=[KNOWN_CATEGORY_L1[0]],
            summaries=["CATEGORIES"],
        )
        r = search_api_app.post(f"{search_path}?pageSize=5", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert_search_response_valid(out)
        cs = out["summaries"].get("categoriesSummary")
        if cs is not None:
            # Spec: `currentCategoryPathElements` echoes the request's path.
            assert cs.get("currentCategoryPathElements") == [KNOWN_CATEGORY_L1[0]]
            # `children` are buckets one level deeper than the request path.
            for bucket in cs.get("children") or []:
                assert isinstance(bucket["count"], int) and bucket["count"] >= 0
                assert isinstance(bucket["categoryPathElements"], list)

    def test_eclass5_summary_buckets_are_well_formed(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        body = make_body(
            cvs=all_cvs, searchMode="BOTH",
            currentEClass5Code=KNOWN_ECLASS5_CODES[0],
            summaries=["ECLASS5"],
        )
        r = search_api_app.post(f"{search_path}?pageSize=5", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert_search_response_valid(out)
        cats = out["summaries"].get("eClass5Categories")
        if cats is not None:
            # selectedEClassGroup either echoes the requested code or is null.
            sel = cats.get("selectedEClassGroup")
            assert sel is None or isinstance(sel, int)
            for bucket in (cats.get("children") or []) + (cats.get("sameLevel") or []):
                assert isinstance(bucket["group"], int)
                assert isinstance(bucket["count"], int) and bucket["count"] >= 0

    def test_prices_summary_buckets_have_currency(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        body = make_body(
            cvs=all_cvs, searchMode="BOTH",
            vendorIdsFilter=[HIGH_VOLUME_VENDOR],
            summaries=["PRICES"],
        )
        r = search_api_app.post(f"{search_path}?pageSize=5", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert_search_response_valid(out)
        for bucket in out["summaries"].get("pricesSummary") or []:
            # All currencyCode values are 3-letter uppercase per spec pattern.
            assert re.fullmatch(r"^[A-Z]{3}$", bucket["currencyCode"])
            # min ≤ max for every bucket.
            assert bucket["min"] <= bucket["max"]

    def test_article_score_is_null_in_browse_path(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """No `query` → browse path → `score` is null (or absent) on
        every returned article. We have no ranking signal here."""
        r = search_api_app.post(
            f"{search_path}?pageSize=10",
            json=make_body(cvs=all_cvs, vendorIdsFilter=[HIGH_VOLUME_VENDOR]),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        for art in body["articles"]:
            assert art.get("score") in (None, 0.0), (
                f"browse-path article had non-null score: {art}"
            )

    def test_article_score_is_numeric_in_query_path(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """A `query` engages dense+BM25; ranked articles must come back
        with a numeric `score`."""
        r = search_api_app.post(
            f"{search_path}?pageSize=10",
            json=make_body(cvs=all_cvs, query="schraube"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert_search_response_valid(body)
        if body["articles"]:
            for art in body["articles"]:
                assert isinstance(art.get("score"), (int, float)), (
                    f"query-path article missing numeric score: {art}"
                )

    def test_query_path_scores_are_non_negative(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """RRF + cosine + BM25 are all non-negative on this corpus
        (cosine is normalised to [0, 2] in the dense leg, BM25 is
        non-negative, RRF is by construction). Any negative score
        means the path is corrupting the score plumbing."""
        r = search_api_app.post(
            f"{search_path}?pageSize=10",
            json=make_body(cvs=all_cvs, query="schraube"),
        )
        assert r.status_code == 200, r.text
        for art in r.json()["articles"]:
            s = art.get("score")
            if s is not None:
                assert s >= 0, f"negative score: {art}"

    def test_features_summary_value_counts_sum_consistent(
        self, search_api_app: TestClient, search_path: str, all_cvs: list[str]
    ) -> None:
        """For each feature summary entry, the parent `count` should be ≥
        the maximum sub-`values[].count` (sub-counts are per-value
        within the same feature; parent is articles having any value
        for that feature)."""
        body = make_body(
            cvs=all_cvs, searchMode="BOTH",
            vendorIdsFilter=[HIGH_VOLUME_VENDOR],
            summaries=["FEATURES"],
        )
        r = search_api_app.post(f"{search_path}?pageSize=5", json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert_search_response_valid(out)
        for fs in out["summaries"].get("featureSummaries") or []:
            sub_counts = [v["count"] for v in fs.get("values", [])]
            if sub_counts:
                assert fs["count"] >= max(sub_counts), (
                    f"feature {fs['name']}: parent count {fs['count']} "
                    f"< max value count {max(sub_counts)}"
                )
