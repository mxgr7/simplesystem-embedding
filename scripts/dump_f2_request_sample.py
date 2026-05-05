"""Capture one real F2 request + response pair against the loaded
``offers_v6`` collection and write each to disk for inspection.

Boots ``search-api`` the same way the F2 contract suite does (in-process
TestClient, dedup topology, mocked TEI), then issues a representative
BOTH-mode search with a vendor filter, a free-text query, sort, and a
couple of summary kinds. Writes:

  artifacts/f2_sample_request.json   — URL + headers + body
  artifacts/f2_sample_response.json  — status, headers, body
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_API_DIR = REPO_ROOT / "search-api"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEARCH_API_DIR))

from fastapi.testclient import TestClient  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402


MILVUS_URI = "http://localhost:19530"
ARTICLES_COLLECTION = "articles_v6"
OFFERS_COLLECTION = "offers_v6"
HIGH_VOLUME_VENDOR = "01054f55-c50c-452b-8822-ee11be4788c9"
EMBED_DIM = 128


def stable_vector() -> list[float]:
    import math
    raw = [(i * 0.0123) % 1.0 - 0.5 for i in range(EMBED_DIM)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def cv_scope(client: MilvusClient, limit: int = 200) -> list[str]:
    rows = client.query(
        collection_name=OFFERS_COLLECTION,
        filter='catalog_version_id != ""',
        output_fields=["catalog_version_id"],
        limit=2000,
    )
    seen: list[str] = []
    s: set[str] = set()
    for r in rows:
        cv = r.get("catalog_version_id")
        if cv and cv not in s:
            s.add(cv)
            seen.append(cv)
            if len(seen) >= limit:
                break
    return seen


def boot_app():
    os.environ["USE_DEDUP_TOPOLOGY"] = "1"
    os.environ["MILVUS_ARTICLES_COLLECTION"] = ARTICLES_COLLECTION
    os.environ["EMBED_URL"] = "http://embed.invalid"
    os.environ["MILVUS_URI"] = MILVUS_URI
    os.environ["API_KEY"] = ""
    spec = importlib.util.spec_from_file_location(
        "search_api_main_for_dump", SEARCH_API_DIR / "main.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["search_api_main_for_dump"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    milvus = MilvusClient(uri=MILVUS_URI)
    cvs = cv_scope(milvus)

    mod = boot_app()
    with TestClient(mod.app) as client:
        mod.app.state.embed.embed = AsyncMock(return_value=[stable_vector()])

        url_path = f"/{OFFERS_COLLECTION}/_search"
        params = {"page": 1, "pageSize": 5, "sort": "articleId,asc"}
        body = {
            "searchMode": "BOTH",
            "selectedArticleSources": {
                "catalogVersionIdsOrderedByPreference": cvs,
                "closedCatalogVersionIds": [],
                "sourcePriceListIds": [],
                "customerUploadedCoreArticleListSourceIds": [],
            },
            "query": "schraube",
            "vendorIdsFilter": [HIGH_VOLUME_VENDOR],
            "currency": "EUR",
            "maxDeliveryTime": 0,
            "coreSortimentOnly": False,
            "closedMarketplaceOnly": False,
            "summaries": ["VENDORS", "MANUFACTURERS", "PRICES"],
        }

        request_dump = {
            "method": "POST",
            "url": url_path,
            "queryParameters": params,
            "headers": {"Content-Type": "application/json"},
            "body": body,
        }

        response = client.post(
            url_path,
            params=params,
            json=body,
        )

        response_dump = {
            "status": response.status_code,
            "headers": dict(response.headers),
            "body": response.json(),
        }

    request_path = ARTIFACTS_DIR / "f2_sample_request.json"
    response_path = ARTIFACTS_DIR / "f2_sample_response.json"
    request_path.write_text(json.dumps(request_dump, indent=2, ensure_ascii=False) + "\n")
    response_path.write_text(json.dumps(response_dump, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {request_path}")
    print(f"wrote {response_path}")
    print(f"status={response.status_code} hits={len(response.json()['articles'])} "
          f"hitCount={response.json()['metadata']['hitCount']}")


if __name__ == "__main__":
    main()
