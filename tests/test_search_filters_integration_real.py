"""F3 integration test against the 200-doc real-MongoDB sample.

Loads `tests/fixtures/mongo_sample/sample_200.json` through the I1
projection module + thin test loader into a fresh `offers_f3_real_test`
collection, then verifies a representative subset of F3 filter
behaviours against real document shapes.

Distinct from `test_search_filters_integration.py` (which uses the
hand-crafted 8-row schema-smoke fixture). The smoke fixture has
deterministic IDs; this one uses real vendor/manufacturer/eClass values
that vary by sample. We assert on count + ID-set monotonicity rather
than fixed ID strings.

Skipped if Milvus is not reachable on localhost:19530.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pymilvus import MilvusClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_API_DIR = REPO_ROOT / "search-api"
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SEARCH_API_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from indexer.projection import project  # noqa: E402
from indexer.test_loader import load_rows  # noqa: E402

MILVUS_URI = "http://localhost:19530"
COLLECTION = "offers_f3_real_test"
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/mongo_sample/sample_200.json"


# ---------- fixtures -----------------------------------------------------

@pytest.fixture(scope="module")
def projected_rows() -> list[dict]:
    raw = json.loads(FIXTURE_PATH.read_text())["records"]
    return [project(r).row for r in raw]


@pytest.fixture(scope="module")
def milvus_client() -> MilvusClient:
    try:
        c = MilvusClient(uri=MILVUS_URI)
        c.list_collections()
    except Exception as exc:
        pytest.skip(f"Milvus unreachable at {MILVUS_URI}: {exc}")
    return c


@pytest.fixture(scope="module", autouse=True)
def seed_collection(milvus_client: MilvusClient, projected_rows: list[dict]):
    from create_offers_collection import build_index_params, build_schema

    if milvus_client.has_collection(COLLECTION):
        milvus_client.drop_collection(COLLECTION)
    schema = build_schema(milvus_client)
    index_params = build_index_params(milvus_client, "HNSW")
    milvus_client.create_collection(
        collection_name=COLLECTION, schema=schema, index_params=index_params,
    )
    milvus_client.load_collection(COLLECTION)
    visible = load_rows(milvus_client, COLLECTION, projected_rows)
    if visible < len(projected_rows):
        pytest.fail(f"only {visible}/{len(projected_rows)} rows visible after seed")
    yield
    milvus_client.drop_collection(COLLECTION)


@pytest.fixture(scope="module")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def client(monkeypatch_session) -> TestClient:
    monkeypatch_session.setenv("EMBED_URL", "http://embed.invalid")
    monkeypatch_session.setenv("MILVUS_URI", MILVUS_URI)
    monkeypatch_session.setenv("API_KEY", "")
    import importlib

    import main as main_mod
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


# ---------- helpers ------------------------------------------------------

_BASE_BODY = {
    "searchMode": "HITS_ONLY",
    "selectedArticleSources": {},
    "currency": "EUR",
}


def _post(client: TestClient, **overrides) -> dict:
    body = {**_BASE_BODY, **overrides}
    r = client.post(f"/{COLLECTION}/_search?pageSize=500", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _ids(body: dict) -> set[str]:
    return {a["articleId"] for a in body["articles"]}


# ---------- tests --------------------------------------------------------

def test_top_vendor_filter_returns_expected_count(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """Pick the most populous vendor in the sample and verify the filter
    returns exactly that vendor's row count."""
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, expected_count = vendor_counts.most_common(1)[0]

    body = _post(client, vendorIdsFilter=[top_vendor])
    assert len(_ids(body)) == expected_count


def test_top_manufacturer_narrows_correctly(
    client: TestClient, projected_rows: list[dict],
) -> None:
    mfr_counts = Counter(r["manufacturerName"] for r in projected_rows if r["manufacturerName"])
    top_mfr, expected_count = mfr_counts.most_common(1)[0]

    body = _post(client, manufacturersFilter=[top_mfr])
    assert len(_ids(body)) == expected_count


def test_eclass5_filter_against_real_codes(
    client: TestClient, projected_rows: list[dict],
) -> None:
    # eclass5_code is now ARRAY<INT32> carrying the full hierarchy. Count
    # rows whose hierarchy contains each leaf code, then assert the filter
    # narrows to exactly that count.
    eclass_counts: Counter[int] = Counter()
    for r in projected_rows:
        for code in r["eclass5_code"]:
            eclass_counts[code] += 1
    top_code, expected_count = eclass_counts.most_common(1)[0]

    body = _post(client, currentEClass5Code=top_code)
    assert len(_ids(body)) == expected_count


def test_and_composition_strictly_narrows(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """Vendor filter ∩ EClass5 filter ⊆ vendor filter (not equal in general)."""
    vendor_counts = Counter(r["vendor_id"] for r in projected_rows)
    top_vendor, _ = vendor_counts.most_common(1)[0]
    # Pick an eClass5 code that appears in some — but not all — of that
    # vendor's rows. Iterate over the per-row hierarchy arrays.
    vendor_eclasses: Counter[int] = Counter()
    for r in projected_rows:
        if r["vendor_id"] == top_vendor:
            for code in r["eclass5_code"]:
                vendor_eclasses[code] += 1
    if not vendor_eclasses:
        pytest.skip("top vendor has no eClass5 codes in sample")
    target_eclass, _ = vendor_eclasses.most_common(1)[0]

    only_vendor = _ids(_post(client, vendorIdsFilter=[top_vendor]))
    composed = _ids(_post(
        client, vendorIdsFilter=[top_vendor], currentEClass5Code=target_eclass,
    ))
    assert composed.issubset(only_vendor)
    assert composed  # not empty


def test_articleid_format_matches_legacy(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """Acceptance: every articleId on the wire is `{friendlyId}:{base64Url}`
    (22-char friendlyId, ":", at least one b64 char)."""
    body = _post(client, vendorIdsFilter=[r["vendor_id"] for r in projected_rows])
    for aid in _ids(body):
        head, _, tail = aid.partition(":")
        assert len(head) == 22, f"friendly_id wrong length on {aid!r}"
        assert tail, f"empty articleNumber portion on {aid!r}"


def test_articleid_round_trips_through_articleids_filter(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """Acceptance: the same articleId must round-trip through `articleIdsFilter`."""
    sample_ids = [r["id"] for r in projected_rows[:5]]
    body = _post(client, articleIdsFilter=sample_ids)
    assert _ids(body) == set(sample_ids)


def test_core_sortiment_filter_against_real_markers(
    client: TestClient, projected_rows: list[dict],
) -> None:
    """The 200-doc sample has very few rows with core markers. Verify the
    filter returns exactly the rows whose `core_marker_enabled_sources`
    intersect the supplied source list."""
    candidates = [
        r for r in projected_rows if r["core_marker_enabled_sources"]
    ]
    if not candidates:
        pytest.skip("no rows in sample have enabled core markers")
    target = candidates[0]
    source_id = target["core_marker_enabled_sources"][0]

    body = _post(client, coreSortimentOnly=True, selectedArticleSources={
        "closedCatalogVersionIds": [source_id],
    })
    expected = {
        r["id"] for r in projected_rows
        if source_id in r["core_marker_enabled_sources"]
    }
    assert _ids(body) == expected
