"""Create and resumably populate a fresh article index with SPLADE vectors.

Examples:
  uv run python scripts/reindex_articles_with_splade.py inspect
  uv run python scripts/reindex_articles_with_splade.py create
  uv run python scripts/reindex_articles_with_splade.py run --backend-urls http://gpu:8080
  uv run python scripts/reindex_articles_with_splade.py finalize
  uv run python scripts/reindex_articles_with_splade.py validate

The source index is only read. Full documents are indexed into a separately
created destination so explicit IDs make page replay safe after a lost PIT.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import httpx
from dotenv import load_dotenv


REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "splade-service"
sys.path.insert(0, str(SERVICE))

from codec import (  # noqa: E402
    pack_sparse,
    pack_sparse_batch,
    unpack_sparse,
    unpack_sparse_batch,
)
from constants import (  # noqa: E402
    MODEL_ID,
    MODEL_SHA256,
    TOP_K,
    VOCAB_SIZE,
    model_metadata,
)
from rendering import render_from_nul  # noqa: E402
from source_assembler import (  # noqa: E402
    S2CLASS_MAPPING_PATH,
    S2CLASS_MAPPING_SHA256,
    assemble_nul,
    load_s2_mapping,
    mapping_sha256,
)


DEFAULT_SRC = "prod-article-index-v1-semantic-bf16"
DEFAULT_DST = "prod-article-index-v1-semantic-splade-20260726"
DEFAULT_ES = "http://localhost:9200"
DEFAULT_PROJECTION = '{"includes":["*"],"exclude_vectors":false}'
TRANSFORM_VERSION = "article-splade-aggregation-bakeoff-v1"
EXPECTED_SOURCE_COUNT = 125_755_244
EXPECTED_OFFER_COUNT = 113_560_531
# The 8-hour bound is the actual constraint on a full run, and the minimum rate
# is derived from it so the two cannot disagree. The previous hard-coded 4,900
# assumed the GPU encoder was the limiter; measurement since then puts the limit
# on the destination host's disk (~137 MB/s EBS against a >1 TB copy), so a rate
# floor above what that disk can absorb would reject an otherwise healthy run.
# The runtime and cost ceilings in load_approval_report are unchanged.
MAX_FULL_RUNTIME_HOURS = 8
MIN_FULL_RATE = -(-EXPECTED_SOURCE_COUNT // (MAX_FULL_RUNTIME_HOURS * 3600))
BAKEOFF_REPORT_SHA256 = (
    "9baac998fad5da36b2350b92539f866b381fe919ad1f8289fdd15f1a925c72a7"
)
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
PORTABLE_SETTING_KEYS = {
    "analysis",
    "codec",
    "mapping",
    # Query-time limits are as load-bearing as analysis. The source raises
    # max_terms_count to 400,000 because the offline eval harnesses pin searches
    # to a fixed universe via a terms lookup against the `eval_universe` index
    # (see pipeline/id_arm_case_run.py), and those universes are 149,418 (gold)
    # and 166,874 (seg) ids -- both far over the 65,536 default. An index that
    # silently falls back to the default rejects those queries at search time,
    # which no count- or coverage-based validation can catch.
    "max_terms_count",
    "max_ngram_diff",
    "max_shingle_diff",
    "number_of_routing_shards",
    "number_of_shards",
    "routing_partition_size",
    "similarity",
    "sort",
}
# Settings deliberately NOT cloned, with the reason, so the next reader does not
# have to re-derive it:
#   number_of_replicas / refresh_interval / translog.*  - the loader owns these
#                                                         and finalize restores
#                                                         them from the source
#   merge.*            - retuned for the destination's disk
#   uuid / creation_date / provided_name / version / history.uuid - per-index
#   routing.allocation.*                              - cluster topology
#   store.preload / compound_format                   - static, performance-only;
#                                                       set them by hand on a
#                                                       closed index if wanted
UNPORTABLE_BY_DESIGN = {
    "number_of_replicas", "refresh_interval", "translog", "merge", "uuid",
    "creation_date", "provided_name", "version", "history", "routing",
    "store", "compound_format", "codec",
}


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def host_only(url):
    parsed = urlparse(url)
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc += f":{parsed.port}"
    return urlunparse((parsed.scheme, netloc, parsed.path.rstrip("/"), "", "", ""))


def ensure_separate_source(src, dst):
    if src == dst:
        raise ValueError("source and destination index must differ")


def parse_slice_ids(spec, total):
    """'' -> every slice; '0-7' / '0,2,4' -> that subset.

    The client is a single Python process and measurement showed it pinned at
    ~99% of one core while ES sat at 7% CPU and the GPU at 58% -- the GIL, not
    the cluster or the encoder, was the ceiling. Splitting the slice set across
    several processes is what lifts it. Each process keeps its own state file
    and its own PITs; slices are already independent, and `--slices` still
    defines the global slice count so the PIT slice partition is unchanged.
    """
    if not spec:
        return list(range(total))
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    ids = sorted(dict.fromkeys(ids))
    if not ids:
        raise ValueError("--slice-ids selected no slices")
    if ids[0] < 0 or ids[-1] >= total:
        raise ValueError(f"--slice-ids must be within 0..{total - 1}")
    return ids


def portable_settings(settings):
    """Copy only creation-safe settings that affect index semantics."""
    return {
        key: copy.deepcopy(value)
        for key, value in settings.items()
        if key in PORTABLE_SETTING_KEYS
    }


def deep_merge_settings(base, patch):
    """Merge patch into base in place: dicts recurse, everything else replaces.

    Lists replace wholesale on purpose -- an analyzer's filter chain is
    ordered, so element-wise merging could only produce a chain nobody wrote.
    """
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge_settings(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def assert_patch_applied(actual, patch, path=""):
    """Fail create if any leaf of the settings patch did not land verbatim.

    ES normalizes scalars to strings in _settings, so scalars compare via str().
    """
    for key, value in patch.items():
        here = f"{path}.{key}" if path else key
        if isinstance(value, dict):
            got = actual.get(key)
            if not isinstance(got, dict):
                raise ValueError(f"settings patch not applied at {here}")
            assert_patch_applied(got, value, here)
        elif isinstance(value, list):
            if actual.get(key) != value:
                raise ValueError(
                    f"settings patch not applied at {here}: "
                    f"{actual.get(key)!r} != {value!r}"
                )
        else:
            if str(actual.get(key)) != str(value):
                raise ValueError(
                    f"settings patch not applied at {here}: "
                    f"{actual.get(key)!r} != {value!r}"
                )


def dropped_settings(settings):
    """Source settings that are neither cloned nor deliberately excluded.

    An allowlist silently loses anything nobody thought of -- that is how
    `max_terms_count` (raised to 400,000 on the source for union-by-ID queries)
    failed to reach the 2026-07-26 destination. Nothing here blocks a run; it
    surfaces the residue so a human decides, instead of finding out at query
    time.
    """
    return sorted(
        key for key in settings
        if key not in PORTABLE_SETTING_KEYS and key not in UNPORTABLE_BY_DESIGN
    )


def target_mapping(source_mapping):
    """Copy the mapping, retaining the dense field and normalizing SPLADE fields."""
    mapping = copy.deepcopy(source_mapping)
    source = mapping.get("_source")
    if source is False or isinstance(source, dict) and source.get("enabled") is False:
        raise ValueError("destination requires _source to be enabled")
    if source is None or source is True:
        source = {}
        mapping["_source"] = source
    excludes = source.get("excludes") or []
    if isinstance(excludes, str):
        excludes = [excludes]
    source["excludes"] = list(dict.fromkeys(
        list(excludes) + ["embeddings.vector", "spladeVector"]
    ))
    properties = mapping.setdefault("properties", {})
    properties["spladeVector"] = {"type": "sparse_vector"}
    properties["spladeModelVersion"] = {"type": "keyword"}
    properties["spladeTransformVersion"] = {"type": "keyword"}
    properties["spladeAggregationVersion"] = {"type": "keyword"}
    properties["spladeEncodingVersion"] = {"type": "keyword"}
    return mapping


def dense_mapping(mapping):
    try:
        return mapping["properties"]["embeddings"]["properties"]["vector"]
    except (KeyError, TypeError):
        raise ValueError("mapping is missing embeddings.vector") from None


def verify_target_mapping(expected, actual):
    if actual != expected:
        raise ValueError("destination mapping did not round-trip exactly")
    if dense_mapping(actual) != dense_mapping(expected):
        raise ValueError("destination dense vector mapping changed during creation")
    properties = actual.get("properties", {})
    if properties.get("spladeVector") != {"type": "sparse_vector"}:
        raise ValueError("destination spladeVector is not sparse_vector")
    if properties.get("spladeModelVersion") != {"type": "keyword"}:
        raise ValueError("destination spladeModelVersion is not keyword")
    if properties.get("spladeTransformVersion") != {"type": "keyword"}:
        raise ValueError("destination spladeTransformVersion is not keyword")
    if properties.get("spladeAggregationVersion") != {"type": "keyword"}:
        raise ValueError("destination spladeAggregationVersion is not keyword")
    if properties.get("spladeEncodingVersion") != {"type": "keyword"}:
        raise ValueError("destination spladeEncodingVersion is not keyword")
    excludes = actual.get("_source", {}).get("excludes") or []
    if isinstance(excludes, str):
        excludes = [excludes]
    missing = {"embeddings.vector", "spladeVector"} - set(excludes)
    if missing:
        raise ValueError(f"destination _source exclusions missing {sorted(missing)}")


def validate_dense_source(source):
    embeddings = source.get("embeddings")
    if embeddings is None:
        return
    if not isinstance(embeddings, list):
        raise ValueError("embeddings must be a list")
    for position, embedding in enumerate(embeddings):
        if not isinstance(embedding, dict) or not isinstance(
            embedding.get("inputHash"), str
        ) or not embedding["inputHash"]:
            raise ValueError(f"embeddings[{position}] is missing inputHash")
        vector = embedding.get("vector")
        if not isinstance(vector, list) or len(vector) != 128:
            raise ValueError(f"embeddings[{position}].vector must contain 128 values")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in vector):
            raise ValueError(f"embeddings[{position}].vector has a non-finite value")


def validate_source_hit(hit):
    source = hit.get("_source")
    if not isinstance(source, dict) or not source:
        raise ValueError(f"article {hit.get('_id')} is missing full _source")
    if "articleId" not in source:
        raise ValueError(f"article {hit.get('_id')} is missing articleId")
    validate_dense_source(source)


def normalize_sparse_vector(vector):
    if not isinstance(vector, dict):
        raise ValueError("backend vector must be an object")
    if len(vector) > TOP_K:
        raise ValueError(f"backend vector has more than {TOP_K} entries")
    for token, weight in vector.items():
        try:
            token_id = int(token)
        except (TypeError, ValueError):
            raise ValueError(f"invalid sparse token {token!r}") from None
        if token_id < 0 or token_id >= VOCAB_SIZE:
            raise ValueError(f"invalid sparse token {token!r}")
        if str(token_id) != str(token) or isinstance(weight, bool) or not isinstance(
            weight, (int, float)
        ) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"invalid sparse entry {token!r}: {weight!r}")
    return unpack_sparse(pack_sparse(vector))


def verify_metadata(expected, actual):
    if expected != actual:
        differing = sorted(set(expected) | set(actual))
        differing = [key for key in differing if expected.get(key) != actual.get(key)]
        raise ValueError(f"run metadata mismatch: {', '.join(differing)}")


def prepare_slice_resume(item, pit_alive, latest_pit_id=None, identity_matches=True):
    """Return one copied slice checkpoint with only a safe cursor retained."""
    if not identity_matches:
        raise ValueError("pinned source identity changed")
    state = copy.deepcopy(item)
    if state.get("completed"):
        return state, False
    if state.get("pit_id") and pit_alive:
        state["pit_id"] = latest_pit_id or state["pit_id"]
        return state, True
    state["pit_id"] = None
    state["pit_closed"] = False
    state["search_after"] = None
    state["stats"] = empty_stats()
    return state, False


def select_capacity(candidates):
    """Choose fastest near-lowest-cost capacity under hard cost/runtime limits."""
    feasible = [candidate for candidate in candidates
                if candidate["total_cost"] <= 80
                and candidate["projected_runtime_hours"] <= 8]
    if not feasible:
        return None
    lowest = min(candidate["total_cost"] for candidate in feasible)
    near = [candidate for candidate in feasible
            if candidate["total_cost"] <= lowest * 1.15]
    return min(near, key=lambda candidate: (
        candidate["projected_runtime_hours"], candidate["total_cost"],
        candidate.get("name", ""),
    ))


def load_approval_report(path, args):
    if not path:
        raise ValueError("full run requires --approval-report")
    report = json.loads(Path(path).read_text())
    required = {
        "aggregation",
        "encode_batch_size",
        "end_to_end_docs_per_second",
        "projected_runtime_hours",
        "projected_total_cost",
        "document_encoding_version",
        "backend_contract",
        "retrieval_parity_passed",
        "bakeoff_report_sha256",
        "h11_selected_aggregation",
        "h11_v5_incremental_union_recall",
    }
    missing = required - set(report)
    if missing:
        raise ValueError(f"approval report is missing {sorted(missing)}")
    if report["aggregation"] != args.aggregation:
        raise ValueError("approval report aggregation does not match the run")
    if report["bakeoff_report_sha256"] != BAKEOFF_REPORT_SHA256:
        raise ValueError("approval report does not pin the final aggregation bakeoff")
    if report["h11_selected_aggregation"] != "v1":
        raise ValueError("H11 has not selected V1 for the full run")
    if not isinstance(report["h11_v5_incremental_union_recall"], (int, float)):
        raise ValueError("approval report H11 recall must be numeric")
    if report["encode_batch_size"] != args.encode_batch_size:
        raise ValueError("approval report encode batch size does not match the run")
    if report["end_to_end_docs_per_second"] < MIN_FULL_RATE:
        raise ValueError(f"approval report throughput is below {MIN_FULL_RATE:,} docs/s")
    if not 0 < report["projected_runtime_hours"] <= MAX_FULL_RUNTIME_HOURS:
        raise ValueError(
            f"approval report runtime must be in (0, {MAX_FULL_RUNTIME_HOURS}] hours"
        )
    if not 0 < report["projected_total_cost"] <= 80:
        raise ValueError("approval report total cost must be in (0, 80]")
    if not report["retrieval_parity_passed"]:
        raise ValueError("approval report requires retrieval parity to pass")
    return report


def empty_stats():
    return {
        "docs": 0,
        "offer_docs": 0,
        "no_offer_docs": 0,
        "encoded": 0,
        "indexed": 0,
        "bytes": 0,
        "retries": 0,
    }


def add_stats(destination, addition):
    for key in destination:
        destination[key] += addition.get(key, 0)


def aggregate_stats(state):
    total = empty_stats()
    for item in state.get("slices", {}).values():
        add_stats(total, item.get("stats", {}))
    return total


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


async def request_with_retries(client, method, path, attempts=8, retry_delay=0.25, **kwargs):
    retries = 0
    for attempt in range(attempts):
        try:
            response = await client.request(method, path, **kwargs)
        except httpx.TransportError:
            if attempt + 1 == attempts:
                raise
        else:
            if response.status_code not in TRANSIENT_STATUS:
                response.raise_for_status()
                return response, retries
            if attempt + 1 == attempts:
                response.raise_for_status()
        retries += 1
        await asyncio.sleep(retry_delay * min(2 ** attempt, 16))
    raise RuntimeError("request retry loop exhausted")


async def fetch_json(client, path):
    response, retries = await request_with_retries(client, "GET", path)
    return response.json(), retries


async def source_snapshot(client, index):
    settings_doc, _ = await fetch_json(client, f"/{index}/_settings")
    if len(settings_doc) != 1:
        raise ValueError(f"source must resolve to one concrete index, got {list(settings_doc)}")
    concrete = next(iter(settings_doc))
    mapping_doc, _ = await fetch_json(client, f"/{concrete}/_mapping")
    count_doc, _ = await fetch_json(client, f"/{concrete}/_count")
    try:
        stats_doc, _ = await fetch_json(
            client, f"/{concrete}/_stats/docs,seq_no?level=shards"
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 400 or "unrecognized metric: [seq_no]" not in (
            exc.response.text
        ):
            raise
        stats_doc, _ = await fetch_json(
            client, f"/{concrete}/_stats/docs?level=shards"
        )
    settings = settings_doc[concrete]["settings"]["index"]
    mapping = mapping_doc[concrete]["mappings"]
    primaries = stats_doc.get("_all", {}).get("primaries", {})
    primary_state = {
        "docs": primaries.get("docs"),
        "seq_no": primaries.get("seq_no"),
    }
    identity = {
        "concrete_index": concrete,
        "uuid": settings.get("uuid"),
        "docs_count": count_doc["count"],
        "primary_state": primary_state,
        "mapping_digest": canonical_digest(mapping),
    }
    return identity, settings, mapping


async def index_identity(client, index):
    settings_doc, _ = await fetch_json(client, f"/{index}/_settings")
    if len(settings_doc) != 1:
        raise ValueError(f"index must resolve to one concrete index, got {list(settings_doc)}")
    concrete = next(iter(settings_doc))
    settings = settings_doc[concrete]["settings"]["index"]
    return {"concrete_index": concrete, "uuid": settings.get("uuid")}


async def cluster_uuid(client):
    root, _ = await fetch_json(client, "/")
    return root.get("cluster_uuid")


async def open_pit(client, index, keep_alive):
    response, retries = await request_with_retries(
        client, "POST", f"/{index}/_pit?keep_alive={keep_alive}"
    )
    return response.json()["id"], retries


async def pit_is_alive(client, pit_id, keep_alive):
    response = await client.post("/_search", json={
        "size": 0,
        "pit": {"id": pit_id, "keep_alive": keep_alive},
    })
    if response.status_code in {400, 404}:
        return False, pit_id
    response.raise_for_status()
    return True, response.json().get("pit_id", pit_id)


async def close_pit(client, pit_id):
    if not pit_id:
        return
    try:
        await client.request("DELETE", "/_pit", json={"id": pit_id})
    except httpx.HTTPError:
        pass


def validate_backend_metadata(metadata, allow_cpu, require_optimized=False):
    expected = model_metadata()
    mismatch = {key: (expected[key], metadata.get(key)) for key in expected
                if metadata.get(key) != expected[key]}
    if mismatch:
        raise ValueError(f"backend model contract mismatch: {mismatch}")
    device = str(metadata.get("device", "")).lower()
    if "cuda" not in device and not allow_cpu:
        raise ValueError(f"backend device is not CUDA: {metadata.get('device')!r}")
    if require_optimized and not metadata.get("optimized_document_encoder"):
        implementation = metadata.get("encoder_implementation", "unknown")
        raise ValueError(
            f"full run requires an optimized document encoder, got {implementation}"
        )
    if require_optimized and "splade-u16-f16-batch-v1" not in metadata.get(
        "document_transports", []
    ):
        raise ValueError("full run requires the packed document transport")


async def validate_backends(
    urls, bearer_key, allow_cpu, require_optimized=False, transport=None
):
    headers = {"Authorization": f"Bearer {bearer_key}"} if bearer_key else {}
    clients = []
    expected_contract = None
    try:
        for url in urls:
            client = httpx.AsyncClient(
                base_url=url.rstrip("/"), headers=headers,
                timeout=httpx.Timeout(120, connect=10), transport=transport,
            )
            clients.append(client)
            response, _ = await request_with_retries(client, "GET", "/metadata")
            metadata = response.json()
            validate_backend_metadata(
                metadata, allow_cpu, require_optimized=require_optimized
            )
            contract = {
                key: metadata.get(key)
                for key in (
                    "encoder_implementation",
                    "document_compute_dtype",
                    "document_encoding_version",
                    "document_transports",
                    # Part of the *model* contract, not a tuning flag: a checkpoint
                    # trained with model.fold_vocab_mask=True served without the
                    # mask has arbitrary activations on 14,777 unregularised cased
                    # dims (query nnz 5 -> ~4,530 for soup2b50). The sha256 pins
                    # the exact kept-dimension set -- special ids, train-time
                    # stopword stoplist and folded-vocab mask together -- which two
                    # booleans could not. In the contract so a run cannot start
                    # against a backend that disagrees with the approval report.
                    # NOTE: existing approval reports predate these keys and will
                    # be rejected -- regenerate with make_approval_report.py,
                    # which copies the live contract.
                    "fold_vocab_mask",
                    "vocab_mask_sha256",
                )
            }
            if expected_contract is None:
                expected_contract = contract
            elif contract != expected_contract:
                raise ValueError("SPLADE backends have different document contracts")
            client.splade_metadata = metadata
            if "splade-u16-f16-batch-v1" in metadata.get("document_transports", []):
                canary, _ = await request_with_retries(
                    client, "POST", "/encode-packed",
                    json={"inputs": ["Article Name: backend canary"], "document": True},
                )
                if len(unpack_sparse_batch(canary.content)) != 1:
                    raise ValueError("packed backend canary cardinality mismatch")
        return clients, expected_contract
    except Exception:
        await asyncio.gather(*(client.aclose() for client in clients))
        raise


async def encode_backend(client, texts, retry_delay=0.25):
    transports = getattr(client, "splade_metadata", {}).get("document_transports", [])
    if "splade-u16-f16-batch-v1" in transports:
        response, retries = await request_with_retries(
            client, "POST", "/encode-packed",
            json={"inputs": texts, "document": True}, retry_delay=retry_delay,
        )
        vectors = unpack_sparse_batch(response.content)
        if len(vectors) != len(texts):
            raise ValueError("backend response cardinality does not match inputs")
        return vectors, retries
    response, retries = await request_with_retries(
        client, "POST", "/encode", json={"inputs": texts, "document": True},
        retry_delay=retry_delay,
    )
    vectors = response.json()
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValueError("backend response cardinality does not match inputs")
    return [normalize_sparse_vector(vector) for vector in vectors], retries


async def encode_with_failover(backends, preferred, texts):
    errors = []
    retries = 0
    for offset in range(len(backends)):
        backend = backends[(preferred + offset) % len(backends)]
        try:
            vectors, backend_retries = await encode_backend(backend, texts)
            return vectors, retries + backend_retries
        except Exception as exc:
            retries += 1
            errors.append(exc)
    raise RuntimeError(
        f"all {len(backends)} SPLADE backends failed: {errors[-1]}"
    ) from errors[-1]


def action_bytes(action):
    metadata, document = action
    return (json.dumps(metadata, separators=(",", ":"), ensure_ascii=False).encode()
            + b"\n"
            + json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode()
            + b"\n")


def bulk_body(actions):
    return b"".join(action_bytes(action) for action in actions)


def chunk_actions(actions, maximum_bytes):
    chunk = []
    size = 0
    for action in actions:
        encoded_size = len(action_bytes(action))
        if encoded_size > maximum_bytes:
            raise ValueError(
                f"single bulk action is {encoded_size} bytes, over limit {maximum_bytes}"
            )
        if chunk and size + encoded_size > maximum_bytes:
            yield chunk
            chunk = []
            size = 0
        chunk.append(action)
        size += encoded_size
    if chunk:
        yield chunk


def transient_bulk_item(item):
    status = item.get("status", 500)
    error = str(item.get("error", ""))
    return status in TRANSIENT_STATUS or status >= 500 or "rejected_execution" in error


async def send_bulk(client, actions, attempts=8, retry_delay=0.25):
    pending = list(actions)
    indexed = retries = sent_bytes = 0
    headers = {"Content-Type": "application/x-ndjson"}
    for attempt in range(attempts):
        body = bulk_body(pending)
        sent_bytes += len(body)
        response, transport_retries = await request_with_retries(
            client, "POST", "/_bulk", content=body, headers=headers,
            attempts=attempts, retry_delay=retry_delay,
        )
        retries += transport_retries
        data = response.json()
        items = data.get("items")
        if not isinstance(items, list) or len(items) != len(pending):
            raise ValueError("bulk response cardinality does not match request")
        rejected = []
        for action, result in zip(pending, items):
            operation = next(iter(result.values())) if isinstance(result, dict) else {}
            status = operation.get("status", 500)
            if 200 <= status < 300:
                indexed += 1
            elif transient_bulk_item(operation):
                rejected.append(action)
            else:
                doc_id = action[0]["index"]["_id"]
                raise RuntimeError(
                    f"permanent bulk error status={status} _id={doc_id}: "
                    f"{operation.get('error')}"
                )
        if not rejected:
            return indexed, retries, sent_bytes
        pending = rejected
        retries += 1
        if attempt + 1 == attempts:
            break
        await asyncio.sleep(retry_delay * min(2 ** attempt, 16))
    raise RuntimeError(f"bulk exhausted retries with {len(pending)} rejected items")


def hit_routing(hit):
    if hit.get("_routing") is not None:
        return hit["_routing"]
    value = hit.get("fields", {}).get("_routing")
    if isinstance(value, list) and value:
        return value[0]
    return value


async def search_page(client, pit_id, keep_alive, projection, page_size,
                      slice_id, slices, search_after):
    body = {
        "size": page_size,
        "track_total_hits": False,
        "pit": {"id": pit_id, "keep_alive": keep_alive},
        "_source": projection,
        "stored_fields": ["_routing"],
        "sort": [{"_shard_doc": "asc"}],
        "query": {"match_all": {}},
    }
    if slices > 1:
        body["slice"] = {"id": slice_id, "max": slices}
    if search_after is not None:
        body["search_after"] = search_after
    response, retries = await request_with_retries(client, "POST", "/_search", json=body)
    data = response.json()
    return data["hits"]["hits"], retries, data.get("pit_id", pit_id)


def prepare_page_documents(hits, s2_mapping, args):
    page_stats = empty_stats()
    documents = []
    rendered = []
    rendered_positions = []
    for hit in hits:
        validate_source_hit(hit)
        source = dict(hit["_source"])
        for field in (
            "spladeVector", "spladeModelVersion", "spladeTransformVersion",
            "spladeAggregationVersion", "spladeEncodingVersion",
        ):
            source.pop(field, None)
        if source.get("offers"):
            wire = assemble_nul(source, s2_mapping, args.aggregation)
            text = render_from_nul(wire) if wire is not None else ""
            if not text:
                raise ValueError(f"article {hit.get('_id')} with offers rendered empty")
            if not any(wire.split("\x00")):
                raise ValueError(f"article {hit.get('_id')} has no modeled offer signals")
            rendered_positions.append(len(documents))
            rendered.append(text)
            page_stats["offer_docs"] += 1
        else:
            page_stats["no_offer_docs"] += 1
        documents.append((hit, source))
    page_stats["docs"] = len(documents)
    return documents, rendered, rendered_positions, page_stats


async def process_page(hits, backends, preferred_backend, s2_mapping, args):
    documents, rendered, rendered_positions, page_stats = await asyncio.to_thread(
        prepare_page_documents, hits, s2_mapping, args
    )
    vectors = []
    for start in range(0, len(rendered), args.encode_batch_size):
        batch, retries = await encode_with_failover(
            backends, preferred_backend,
            rendered[start:start + args.encode_batch_size],
        )
        vectors.extend(batch)
        page_stats["retries"] += retries
    for position, vector in zip(rendered_positions, vectors):
        if not vector:
            raise ValueError(
                f"article {documents[position][0].get('_id')} with offers encoded empty"
            )
        documents[position][1]["spladeVector"] = vector
        documents[position][1]["spladeModelVersion"] = MODEL_ID
        documents[position][1]["spladeTransformVersion"] = TRANSFORM_VERSION
        documents[position][1]["spladeAggregationVersion"] = args.aggregation
        documents[position][1]["spladeEncodingVersion"] = args.encoding_version
    page_stats["encoded"] = len(vectors)
    actions = []
    for hit, source in documents:
        operation = {"_index": args.dst, "_id": hit["_id"]}
        routing = hit_routing(hit)
        if routing is not None:
            operation["routing"] = routing
        actions.append(({"index": operation}, source))
    return actions, page_stats


async def slice_worker(slice_id, args, source_client, destination_client, backends,
                       s2_mapping, state, state_lock, worker_sem):
    async with worker_sem:
        item = state["slices"][str(slice_id)]
        if item.get("completed"):
            return
        search_after = item.get("search_after")
        cap = 0
        if args.limit:
            cap = args.limit // args.slices + int(slice_id < args.limit % args.slices)
            if cap == 0:
                item["completed"] = True
                item["limited"] = True
                await close_pit(source_client, item.get("pit_id"))
                item["pit_closed"] = True
                async with state_lock:
                    atomic_write(args.state, state)
                return
        while True:
            already = item["stats"]["docs"]
            if cap and already >= cap:
                item["completed"] = True
                item["limited"] = True
                await close_pit(source_client, item.get("pit_id"))
                item["pit_closed"] = True
                async with state_lock:
                    atomic_write(args.state, state)
                return
            async with state_lock:
                pit_id = item["pit_id"]
            hits, retries, latest_pit_id = await search_page(
                source_client, pit_id, args.pit_keep_alive,
                args.projection, args.page_size, slice_id, args.slices, search_after,
            )
            if latest_pit_id != pit_id:
                async with state_lock:
                    item["pit_id"] = latest_pit_id
            if not hits:
                item["completed"] = True
                await close_pit(source_client, item.get("pit_id"))
                item["pit_closed"] = True
                async with state_lock:
                    atomic_write(args.state, state)
                return
            if cap:
                hits = hits[:max(0, cap - already)]
            actions, page_stats = await process_page(
                hits, backends, slice_id % len(backends), s2_mapping, args
            )
            page_stats["retries"] += retries
            for chunk in chunk_actions(actions, args.bulk_bytes):
                indexed, retries, sent_bytes = await send_bulk(destination_client, chunk)
                page_stats["indexed"] += indexed
                page_stats["retries"] += retries
                page_stats["bytes"] += sent_bytes
            if page_stats["indexed"] != len(hits):
                raise RuntimeError("page was not fully durable")
            search_after = hits[-1]["sort"]
            async with state_lock:
                add_stats(item["stats"], page_stats)
                item["search_after"] = search_after
                item["updated_at"] = int(time.time())
                atomic_write(args.state, state)


async def progress_loop(state, started):
    while True:
        await asyncio.sleep(5)
        stats = aggregate_stats(state)
        elapsed = max(time.time() - started, 0.001)
        print(
            f"docs={stats['docs']:,} offers={stats['offer_docs']:,} "
            f"no_offers={stats['no_offer_docs']:,} encoded={stats['encoded']:,} "
            f"indexed={stats['indexed']:,} bytes={stats['bytes'] / 1048576:.1f}MiB "
            f"retries={stats['retries']:,} rate={stats['docs'] / elapsed:.1f}/s",
            flush=True,
        )


def load_checkpoint(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def make_metadata(args, identity, destination_identity, s2_sha):
    return {
        "source_identity": identity,
        "source_index": args.src,
        "destination_index": args.dst,
        "destination_identity": destination_identity,
        "mapping_digest": identity["mapping_digest"],
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "s2_path": str(Path(args.s2_path).resolve()),
        "s2_sha256": s2_sha,
        "transform_version": TRANSFORM_VERSION,
        "aggregation": args.aggregation,
        "document_encoding_version": args.encoding_version,
        "backend_contract": args.backend_contract,
        "slice_count": args.slices,
        "source_es_url": host_only(args.src_url),
        "destination_es_url": host_only(args.dst_url),
        "limit": args.limit,
        "source_projection": args.projection,
        "measured_docs_per_second": args.measured_docs_per_second,
        "projected_runtime_hours": args.projected_runtime_hours,
        "projected_total_cost": args.projected_total_cost,
        "expected_source_count": EXPECTED_SOURCE_COUNT,
        "expected_offer_count": EXPECTED_OFFER_COUNT,
        "bakeoff_report_sha256": BAKEOFF_REPORT_SHA256,
    }


async def command_inspect(args):
    async with make_es_client(args.src_url) as source_client:
        identity, settings, mapping = await source_snapshot(source_client, args.src)
        pit_id, _ = await open_pit(
            source_client, identity["concrete_index"], "1m"
        )
        try:
            hits, _, _ = await search_page(
                source_client, pit_id, "1m", args.projection, 1, 0, 1, None
            )
            if hits:
                validate_dense_source(hits[0].get("_source", {}))
            projection_probe = {
                "documents": len(hits),
                "embeddings": len(hits[0].get("_source", {}).get("embeddings") or [])
                if hits else 0,
                "valid": True,
            }
        finally:
            await close_pit(source_client, pit_id)
        output = {
            "source_identity": identity,
            "portable_settings": portable_settings(settings),
            "dense_mapping": dense_mapping(mapping),
            "target_source": target_mapping(mapping).get("_source"),
            "source_projection": args.projection,
            "projection_probe": projection_probe,
        }
        print(json.dumps(output, indent=2, sort_keys=True))


async def command_create(args):
    ensure_separate_source(args.src, args.dst)
    async with make_es_client(args.src_url) as source_client, make_es_client(
        args.dst_url
    ) as destination_client:
        identity, source_settings, source_mapping = await source_snapshot(
            source_client, args.src
        )
        exists = await destination_client.head(f"/{args.dst}")
        if exists.status_code == 200:
            destination_identity = await index_identity(destination_client, args.dst)
            same_cluster = (
                await cluster_uuid(source_client) == await cluster_uuid(destination_client)
            )
            if (
                same_cluster
                and destination_identity["concrete_index"] == identity["concrete_index"]
            ):
                raise ValueError("destination resolves to the protected source index")
            if not args.delete_existing:
                raise RuntimeError(
                    f"destination {args.dst} already exists; use --delete-existing"
                )
            response, _ = await request_with_retries(
                destination_client, "DELETE", f"/{args.dst}"
            )
            response.raise_for_status()
        elif exists.status_code != 404:
            exists.raise_for_status()
        settings = portable_settings(source_settings)
        settings_patch = None
        settings_patch_sha = None
        if args.settings_patch:
            settings_patch = json.loads(Path(args.settings_patch).read_text())
            settings_patch_sha = canonical_digest(settings_patch)
            deep_merge_settings(settings, settings_patch)
        restore = {
            "number_of_replicas": settings.get("number_of_replicas", "1"),
            "refresh_interval": settings.get("refresh_interval", "1s"),
        }
        settings["number_of_replicas"] = "0"
        settings["refresh_interval"] = "-1"
        settings["translog"] = {
            "durability": "async",
            "sync_interval": "120s",
            "flush_threshold_size": "2gb",
        }
        # The destination host's disk, not the GPU, is the bottleneck for this
        # copy: its EBS volume measures ~137 MB/s while the run moves well over a
        # terabyte. Both of these settings trade CPU (32 mostly-idle cores) and
        # search latency for bytes written, and both also buy disk headroom.
        #
        # `codec` is a static setting -- it can only be chosen at creation time,
        # and portable_settings() would otherwise clone the source's (unset =
        # LZ4). best_compression is DEFLATE on stored fields, ~20-25% smaller.
        settings["codec"] = args.codec
        # A relaxed merge policy is the other half: tolerating more segments and
        # capping merged-segment size keeps large segments from being rewritten
        # over and over, which is where write amplification comes from. Merge
        # threads stay low on purpose -- with a slow volume, more concurrent
        # merges only contend for the same bandwidth.
        settings["merge"] = {
            "policy": {
                "segments_per_tier": str(args.segments_per_tier),
                "floor_segment": args.floor_segment,
                "max_merged_segment": args.max_merged_segment,
            },
            "scheduler": {"max_thread_count": str(args.merge_threads)},
        }
        mapping = target_mapping(source_mapping)
        response, _ = await request_with_retries(
            destination_client, "PUT", f"/{args.dst}",
            json={"settings": {"index": settings}, "mappings": mapping},
        )
        response.raise_for_status()
        actual_doc, _ = await fetch_json(destination_client, f"/{args.dst}/_mapping")
        actual = actual_doc[next(iter(actual_doc))]["mappings"]
        verify_target_mapping(mapping, actual)
        settings_doc, _ = await fetch_json(destination_client, f"/{args.dst}/_settings")
        actual_settings = settings_doc[next(iter(settings_doc))]["settings"]["index"]
        if str(actual_settings.get("number_of_replicas")) != "0":
            raise ValueError("destination did not retain replicas=0")
        if actual_settings.get("refresh_interval") != "-1":
            raise ValueError("destination did not retain refresh_interval=-1")
        # codec is static: if it did not stick there is no fixing it later, and
        # the run would silently cost ~20% more disk than it was planned against.
        if actual_settings.get("codec") != args.codec:
            raise ValueError(
                f"destination codec is {actual_settings.get('codec')!r}, "
                f"expected {args.codec!r}"
            )
        if settings_patch:
            assert_patch_applied(actual_settings, settings_patch)
        residue = dropped_settings(source_settings)
        print(json.dumps({
            "created": args.dst,
            "source_identity": identity,
            "mapping_digest": canonical_digest(mapping),
            "restore_settings": restore,
            "source_settings_not_carried": residue,
            "settings_patch": args.settings_patch or None,
            "settings_patch_sha256": settings_patch_sha,
        }, indent=2, sort_keys=True))
        if residue:
            print(
                f"\nNOTE: {len(residue)} source setting(s) were not carried to "
                f"{args.dst}: {', '.join(residue)}\n"
                "Review each one -- query-time limits in particular are invisible "
                "to count/coverage validation.",
                file=sys.stderr,
            )


async def command_run(args):
    ensure_separate_source(args.src, args.dst)
    if not args.limit and not args.confirm_full_run:
        raise ValueError("full run requires --confirm-full-run after explicit approval")
    if not args.limit and args.aggregation != "v1":
        raise ValueError("the final full-scale bakeoff approves only V1")
    approval_report = load_approval_report(args.approval_report, args) if not args.limit else None
    if approval_report:
        args.measured_docs_per_second = approval_report["end_to_end_docs_per_second"]
        args.projected_runtime_hours = approval_report["projected_runtime_hours"]
        args.projected_total_cost = approval_report["projected_total_cost"]
    backend_urls = [url.strip() for url in args.backend_urls.split(",") if url.strip()]
    if not backend_urls:
        raise ValueError("--backend-urls is required")
    owned_slice_ids = parse_slice_ids(args.slice_ids, args.slices)
    owns_all_slices = len(owned_slice_ids) == args.slices
    actual_s2_sha = mapping_sha256(args.s2_path)
    if args.s2_sha256 and actual_s2_sha != args.s2_sha256:
        raise ValueError(
            f"S2 mapping SHA256 mismatch: expected {args.s2_sha256}, got {actual_s2_sha}"
        )
    s2_mapping = load_s2_mapping(args.s2_path, actual_s2_sha)
    source_client = make_es_client(args.src_url)
    destination_client = make_es_client(args.dst_url)
    backends = []
    completed = False
    try:
        identity, _, source_mapping = await source_snapshot(source_client, args.src)
        destination_identity = await index_identity(destination_client, args.dst)
        if (
            await cluster_uuid(source_client) == await cluster_uuid(destination_client)
            and destination_identity["concrete_index"] == identity["concrete_index"]
        ):
            raise ValueError("destination resolves to the protected source index")
        if not args.limit and identity["docs_count"] != EXPECTED_SOURCE_COUNT:
            raise ValueError(
                f"full run source count changed: expected {EXPECTED_SOURCE_COUNT:,}, "
                f"got {identity['docs_count']:,}"
            )
        if not args.limit:
            offer_count = await count_query(
                source_client, identity["concrete_index"],
                {"nested": {
                    "path": "offers", "query": {"match_all": {}}, "score_mode": "none",
                }},
            )
            if offer_count != EXPECTED_OFFER_COUNT:
                raise ValueError(
                    f"full run offer count changed: expected {EXPECTED_OFFER_COUNT:,}, "
                    f"got {offer_count:,}"
                )
        target_doc, _ = await fetch_json(destination_client, f"/{args.dst}/_mapping")
        target = target_doc[next(iter(target_doc))]["mappings"]
        verify_target_mapping(target_mapping(source_mapping), target)
        backends, backend_contract = await validate_backends(
            backend_urls, args.backend_key, args.allow_cpu,
            require_optimized=not bool(args.limit),
        )
        args.backend_contract = backend_contract
        args.encoding_version = backend_contract["document_encoding_version"]
        if approval_report and approval_report["document_encoding_version"] != args.encoding_version:
            raise ValueError("approval report encoding version does not match the backend")
        if approval_report and approval_report["backend_contract"] != backend_contract:
            raise ValueError("approval report backend contract does not match the backend")
        metadata = make_metadata(args, identity, destination_identity, actual_s2_sha)
        state = load_checkpoint(args.state)
        if state:
            verify_metadata(metadata, state.get("metadata", {}))
            for slice_key, item in list(state["slices"].items()):
                alive = False
                latest_pit_id = item.get("pit_id")
                if latest_pit_id and not item.get("completed"):
                    alive, latest_pit_id = await pit_is_alive(
                        source_client, latest_pit_id, args.pit_keep_alive
                    )
                state["slices"][slice_key], _ = prepare_slice_resume(
                    item, alive, latest_pit_id, True
                )
        else:
            if owns_all_slices:
                destination_count = (await fetch_json(
                    destination_client, f"/{args.dst}/_count"
                ))[0]["count"]
                if destination_count:
                    raise ValueError(
                        f"new run requires an empty destination, found "
                        f"{destination_count} docs"
                    )
            state = {
                "metadata": metadata,
                "slices": {str(index): {
                    "pit_id": None,
                    "pit_closed": False,
                    "search_after": None,
                    "completed": False,
                    "limited": False,
                    "stats": empty_stats(),
                } for index in owned_slice_ids},
            }
        if not all(item.get("completed") for item in state["slices"].values()):
            for item in state["slices"].values():
                if not item.get("completed") and not item.get("pit_id"):
                    item["pit_id"], _ = await open_pit(
                        source_client, identity["concrete_index"], args.pit_keep_alive
                    )
                    item["pit_closed"] = False
            identity_after_pit, _, _ = await source_snapshot(source_client, args.src)
            if identity_after_pit != identity:
                await asyncio.gather(*(
                    close_pit(source_client, item.get("pit_id"))
                    for item in state["slices"].values()
                    if not item.get("completed")
                ))
                raise ValueError("source identity changed while opening PITs")
            atomic_write(args.state, state)
            probe_slice, probe_item = next(
                (int(key), item) for key, item in state["slices"].items()
                if not item.get("completed")
            )
            probe_hits, _, latest_pit_id = await search_page(
                source_client, probe_item["pit_id"], args.pit_keep_alive,
                args.projection, 1, probe_slice, args.slices, None,
            )
            probe_item["pit_id"] = latest_pit_id
            if probe_hits:
                validate_source_hit(probe_hits[0])
            started = time.time()
            reporter = asyncio.create_task(progress_loop(state, started))
            state_lock = asyncio.Lock()
            worker_sem = asyncio.Semaphore(args.workers)
            tasks = [asyncio.create_task(slice_worker(
                slice_id, args, source_client, destination_client,
                backends, s2_mapping, state,
                state_lock, worker_sem,
            )) for slice_id in owned_slice_ids]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            finally:
                reporter.cancel()
                await asyncio.gather(reporter, return_exceptions=True)
            elapsed = max(time.time() - started, 0.001)
        else:
            elapsed = 0.001
        completed = all(item.get("completed") for item in state["slices"].values())
        stats = aggregate_stats(state)
        if completed and not args.limit and owns_all_slices:
            if stats["docs"] != EXPECTED_SOURCE_COUNT:
                raise RuntimeError(
                    f"completed run scanned {stats['docs']:,}, expected {EXPECTED_SOURCE_COUNT:,}"
                )
            if stats["offer_docs"] != EXPECTED_OFFER_COUNT:
                raise RuntimeError(
                    f"completed run encoded {stats['offer_docs']:,} offer docs, "
                    f"expected {EXPECTED_OFFER_COUNT:,}"
                )
        state["completed"] = completed
        state["completed_at"] = int(time.time()) if completed else None
        atomic_write(args.state, state)
        print(
            f"complete={completed} docs={stats['docs']:,} offers={stats['offer_docs']:,} "
            f"no_offers={stats['no_offer_docs']:,} encoded={stats['encoded']:,} "
            f"indexed={stats['indexed']:,} bytes={stats['bytes'] / 1048576:.1f}MiB "
            f"retries={stats['retries']:,} rate={stats['docs'] / elapsed:.1f}/s"
        )
    finally:
        if backends:
            await asyncio.gather(*(backend.aclose() for backend in backends))
        await source_client.aclose()
        await destination_client.aclose()


async def command_finalize(args):
    ensure_separate_source(args.src, args.dst)
    state = load_checkpoint(args.state)
    if not state or not state.get("completed"):
        raise RuntimeError("a completed run checkpoint is required")
    if state["metadata"].get("destination_index") != args.dst:
        raise ValueError("checkpoint destination does not match --dst")
    async with make_es_client(args.src_url) as source_client, make_es_client(
        args.dst_url
    ) as destination_client:
        identity, settings, _ = await source_snapshot(source_client, args.src)
        if identity != state["metadata"]["source_identity"]:
            raise ValueError("pinned source identity changed")
        destination_identity = await index_identity(destination_client, args.dst)
        if destination_identity != state["metadata"].get("destination_identity"):
            raise ValueError("pinned destination identity changed")
        source_count = identity["docs_count"]
        destination_count = (await fetch_json(
            destination_client, f"/{args.dst}/_count"
        ))[0]["count"]
        limited = bool(state["metadata"].get("limit"))
        if not limited and destination_count != source_count:
            raise ValueError(
                f"count mismatch source={source_count} destination={destination_count}"
            )
        if limited and not args.allow_limited:
            raise ValueError("limited smoke run requires --allow-limited to finalize")
        replicas = args.replicas
        if replicas is None:
            replicas = settings.get("number_of_replicas", "1")
        refresh = args.refresh_interval
        if refresh is None:
            refresh = settings.get("refresh_interval", "1s")
        await request_with_retries(
            destination_client, "POST", f"/{args.dst}/_flush?wait_if_ongoing=true"
        )
        source_translog = settings.get("translog", {})
        await request_with_retries(
            destination_client, "PUT", f"/{args.dst}/_settings", json={"index": {
                "number_of_replicas": str(replicas),
                "refresh_interval": refresh,
                "translog.durability": source_translog.get("durability", "request"),
                "translog.sync_interval": source_translog.get("sync_interval"),
                "translog.flush_threshold_size": source_translog.get(
                    "flush_threshold_size"
                ),
            }},
        )
        await request_with_retries(destination_client, "POST", f"/{args.dst}/_refresh")
        await request_with_retries(
            destination_client, "GET",
            f"/_cluster/health/{args.dst}?wait_for_status=yellow&timeout=10m",
        )
        print(
            f"finalized {args.dst}: count={destination_count:,} replicas={replicas} "
            f"refresh={refresh} (no force merge)"
        )


async def count_query(client, index, query):
    response, _ = await request_with_retries(
        client, "POST", f"/{index}/_count", json={"query": query}
    )
    return response.json()["count"]


async def command_validate(args):
    ensure_separate_source(args.src, args.dst)
    async with make_es_client(args.src_url) as source_client, make_es_client(
        args.dst_url
    ) as destination_client:
        identity, _, source_mapping = await source_snapshot(source_client, args.src)
        state = load_checkpoint(args.state)
        if not state:
            raise ValueError("validation requires the completed run checkpoint")
        destination_identity = await index_identity(destination_client, args.dst)
        if destination_identity != state["metadata"].get("destination_identity"):
            raise ValueError("pinned destination identity changed")
        destination_count = (await fetch_json(
            destination_client, f"/{args.dst}/_count"
        ))[0]["count"]
        target_doc, _ = await fetch_json(destination_client, f"/{args.dst}/_mapping")
        target = target_doc[next(iter(target_doc))]["mappings"]
        verify_target_mapping(target_mapping(source_mapping), target)
        offer_count = await count_query(
            source_client, identity["concrete_index"],
            {"nested": {
                "path": "offers",
                "query": {"match_all": {}},
                "score_mode": "none",
            }},
        )
        model_count = await count_query(destination_client, args.dst, {
            "bool": {"filter": [
                {"exists": {"field": "spladeVector"}},
                {"term": {"spladeModelVersion": MODEL_ID}},
            ]}
        })
        aggregation_filters = [
            {"term": {"spladeModelVersion": MODEL_ID}},
            {"term": {"spladeTransformVersion": TRANSFORM_VERSION}},
            {"term": {"spladeAggregationVersion": state["metadata"]["aggregation"]}},
            {"term": {
                "spladeEncodingVersion": state["metadata"]["document_encoding_version"]
            }},
        ]
        aggregation_count = await count_query(destination_client, args.dst, {
            "bool": {"filter": aggregation_filters}
        })
        missing_offer_vectors = await count_query(destination_client, args.dst, {
            "bool": {
                "filter": [{"nested": {
                    "path": "offers", "query": {"match_all": {}}, "score_mode": "none",
                }}],
                "must_not": aggregation_filters,
            }
        })
        unexpected_no_offer_vectors = await count_query(destination_client, args.dst, {
            "bool": {
                "filter": [{"exists": {"field": "spladeVector"}}],
                "must_not": [{"nested": {
                    "path": "offers", "query": {"match_all": {}}, "score_mode": "none",
                }}],
            }
        })
        dense_query = {"nested": {
            "path": "embeddings",
            "query": {"exists": {"field": "embeddings.vector"}},
            "score_mode": "none",
        }}
        source_dense_count = await count_query(
            source_client, identity["concrete_index"], dense_query
        )
        destination_dense_count = await count_query(
            destination_client, args.dst, dense_query
        )
        result = {
            "source_count": identity["docs_count"],
            "destination_count": destination_count,
            "source_offer_docs": offer_count,
            "destination_model_docs": model_count,
            "destination_aggregation_docs": aggregation_count,
            "missing_offer_vectors": missing_offer_vectors,
            "unexpected_no_offer_vectors": unexpected_no_offer_vectors,
            "source_dense_docs": source_dense_count,
            "destination_dense_docs": destination_dense_count,
            "counts_match": identity["docs_count"] == destination_count,
            "model_coverage_matches": offer_count == model_count,
            "aggregation_coverage_matches": offer_count == aggregation_count,
            "dense_coverage_matches": source_dense_count == destination_dense_count,
            "mapping_valid": True,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["counts_match"] and not args.allow_limited:
            raise ValueError("source and destination counts differ")
        if not result["model_coverage_matches"] and not args.allow_limited:
            raise ValueError("SPLADE model coverage differs from source offer coverage")
        if (
            missing_offer_vectors
            or unexpected_no_offer_vectors
            or not result["aggregation_coverage_matches"]
            or not result["dense_coverage_matches"]
        ) and not args.allow_limited:
            raise ValueError("destination vector coverage validation failed")


def make_es_client(url):
    return httpx.AsyncClient(
        base_url=url.rstrip("/"), timeout=httpx.Timeout(300, connect=10),
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
    )


def add_common(parser):
    parser.add_argument("--es", default=os.environ.get("ELASTIC_URL", DEFAULT_ES),
                        help="single-cluster fallback URL")
    parser.add_argument("--src-es", default=os.environ.get("ELASTIC_PROD_URL", ""),
                        help="source ES URL (default ELASTIC_PROD_URL, then --es)")
    parser.add_argument("--dst-es", default=os.environ.get("ELASTIC_URL", ""),
                        help="destination ES URL (default ELASTIC_URL, then --es)")
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--dst", default=DEFAULT_DST)


def parse_args(argv=None):
    load_dotenv(REPO / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    add_common(common)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", parents=[common])
    create_parser = commands.add_parser("create", parents=[common])
    create_parser.add_argument("--delete-existing", action="store_true")
    # Bulk-load shape for a disk-bound destination; see command_create.
    create_parser.add_argument("--codec", default="best_compression",
                               choices=("best_compression", "default"))
    create_parser.add_argument("--segments-per-tier", type=int, default=32)
    create_parser.add_argument("--floor-segment", default="64mb")
    create_parser.add_argument("--max-merged-segment", default="2gb")
    create_parser.add_argument("--merge-threads", type=int, default=2)
    create_parser.add_argument(
        "--settings-patch", default="",
        help="JSON file deep-merged into the cloned source settings before "
             "creation (MXG-79/95: the settled analysis changes cannot come "
             "from the source index, which predates them). Leaves are "
             "verified against _settings after creation.",
    )
    run_parser = commands.add_parser("run", parents=[common])
    run_parser.add_argument("--backend-urls", default=os.environ.get("SPLADE_BACKEND_URLS", ""))
    run_parser.add_argument("--backend-key", default=os.environ.get("SPLADE_BACKEND_API_KEY", ""))
    run_parser.add_argument("--allow-cpu", action="store_true")
    run_parser.add_argument("--s2-path", default=S2CLASS_MAPPING_PATH)
    run_parser.add_argument("--s2-sha256", default=S2CLASS_MAPPING_SHA256)
    run_parser.add_argument("--state", default="reindex-splade-state.json")
    run_parser.add_argument("--slices", type=int, default=16)
    run_parser.add_argument(
        "--slice-ids", default="",
        help="subset of slices this process owns, e.g. '0-7' (default: all). "
             "Run several processes with disjoint subsets and separate --state "
             "files to get past the single-process GIL ceiling.",
    )
    run_parser.add_argument("--workers", type=int, default=16)
    run_parser.add_argument("--page-size", type=int, default=250)
    run_parser.add_argument("--encode-batch-size", type=int, default=256)
    run_parser.add_argument("--bulk-bytes", type=int, default=4 * 1024 * 1024)
    run_parser.add_argument("--pit-keep-alive", default="30m")
    run_parser.add_argument("--limit", type=int, default=0,
                            help="distributed smoke cap across all slices")
    run_parser.add_argument(
        "--aggregation", choices=("v1", "v1-v2-all4-cap3", "v3"),
        default="v1",
        help="article aggregation selected by the full-scale bakeoff",
    )
    run_parser.add_argument(
        "--confirm-full-run", action="store_true",
        help="required with --limit=0 after the user approves the full run",
    )
    run_parser.add_argument("--approval-report")
    finalize_parser = commands.add_parser("finalize", parents=[common])
    finalize_parser.add_argument("--state", default="reindex-splade-state.json")
    finalize_parser.add_argument("--replicas", type=int)
    finalize_parser.add_argument("--refresh-interval")
    finalize_parser.add_argument("--allow-limited", action="store_true")
    validate_parser = commands.add_parser("validate", parents=[common])
    validate_parser.add_argument("--state", default="reindex-splade-state.json")
    validate_parser.add_argument("--allow-limited", action="store_true")
    args = parser.parse_args(argv)
    args.src_url = args.src_es or args.es
    args.dst_url = args.dst_es or args.es
    args.measured_docs_per_second = 0
    args.projected_runtime_hours = 0
    args.projected_total_cost = 0
    args.projection = json.loads(DEFAULT_PROJECTION)
    if hasattr(args, "slices"):
        if args.slices < 1 or args.workers < 1:
            parser.error("--slices and --workers must be positive")
        if args.limit < 0:
            parser.error("--limit must be non-negative")
        if args.page_size < 1 or args.encode_batch_size < 1 or args.bulk_bytes < 1:
            parser.error("page size, encode batch size, and bulk bytes must be positive")
    return args


async def main_async(args):
    commands = {
        "inspect": command_inspect,
        "create": command_create,
        "run": command_run,
        "finalize": command_finalize,
        "validate": command_validate,
    }
    await commands[args.command](args)


def main():
    try:
        asyncio.run(main_async(parse_args()))
    except (ValueError, RuntimeError, httpx.HTTPError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
