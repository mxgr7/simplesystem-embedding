import asyncio
import copy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reindex_articles_with_splade",
    REPO / "scripts" / "reindex_articles_with_splade.py",
)
reindex = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reindex)


def test_mapping_clone_unions_exclusions_without_mutating_source():
    source = {
        "_source": {"excludes": ["private"]},
        "properties": {
            "embeddings": {"properties": {
                "vector": {"type": "dense_vector", "dims": 128, "index_options": {"type": "bbq_hnsw"}},
            }},
            "spladeVector": {"type": "keyword"},
        },
    }
    before = copy.deepcopy(source)
    target = reindex.target_mapping(source)
    assert source == before
    assert target["_source"]["excludes"] == [
        "private", "embeddings.vector", "spladeVector",
    ]
    assert target["properties"]["spladeVector"] == {"type": "sparse_vector"}
    assert target["properties"]["spladeModelVersion"] == {"type": "keyword"}
    assert target["properties"]["spladeAggregationVersion"] == {"type": "keyword"}
    assert target["properties"]["spladeEncodingVersion"] == {"type": "keyword"}
    assert reindex.dense_mapping(target) == reindex.dense_mapping(source)


def test_portable_settings_are_recursive_and_do_not_mutate_input():
    source = {
        "uuid": "x", "version": {"created": "1"}, "routing": {"allocation": {"tier": "hot"}},
        "analysis": {"analyzer": {"custom": {"type": "standard"}}},
        "number_of_shards": "4",
    }
    before = copy.deepcopy(source)
    result = reindex.portable_settings(source)
    assert source == before
    assert result == {
        "analysis": {"analyzer": {"custom": {"type": "standard"}}},
        "number_of_shards": "4",
    }


def test_source_equals_destination_is_refused():
    with pytest.raises(ValueError, match="must differ"):
        reindex.ensure_separate_source("articles", "articles")


def test_dense_source_validation_is_strict():
    good = {"embeddings": [{"inputHash": "abc", "vector": [0.0] * 128}]}
    reindex.validate_dense_source(good)
    bad_dimension = copy.deepcopy(good)
    bad_dimension["embeddings"][0]["vector"].pop()
    with pytest.raises(ValueError, match="128"):
        reindex.validate_dense_source(bad_dimension)
    bad_number = copy.deepcopy(good)
    bad_number["embeddings"][0]["vector"][3] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        reindex.validate_dense_source(bad_number)
    with pytest.raises(ValueError, match="inputHash"):
        reindex.validate_dense_source({"embeddings": [{"vector": [0.0] * 128}]})


def test_backend_codec_and_cardinality_validation():
    async def run_case():
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            assert json.loads(request.content) == {"inputs": ["one"], "document": True}
            return httpx.Response(200, json=[{"2": 1.2345, "1": 0.25}])

        async with httpx.AsyncClient(
            base_url="http://backend", transport=httpx.MockTransport(handler)
        ) as client:
            vectors, retries = await reindex.encode_backend(client, ["one"], retry_delay=0)
        assert calls == 1
        assert retries == 0
        assert vectors == [reindex.unpack_sparse(reindex.pack_sparse({"2": 1.2345, "1": 0.25}))]

        async def cardinality(request):
            return httpx.Response(200, json=[])

        async with httpx.AsyncClient(
            base_url="http://backend", transport=httpx.MockTransport(cardinality)
        ) as client:
            with pytest.raises(ValueError, match="cardinality"):
                await reindex.encode_backend(client, ["one"], retry_delay=0)

    asyncio.run(run_case())


def test_backend_codec_rejects_out_of_vocabulary_tokens():
    with pytest.raises(ValueError, match="token"):
        reindex.normalize_sparse_vector({str(reindex.VOCAB_SIZE): 1.0})


def test_backend_prefers_packed_document_transport():
    async def run_case():
        calls = []
        packed = reindex.pack_sparse_batch([
            {"2": 1.2345, "1": 0.25},
        ])

        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(200, content=packed)

        async with httpx.AsyncClient(
            base_url="http://backend", transport=httpx.MockTransport(handler)
        ) as client:
            client.splade_metadata = {
                "document_transports": ["splade-u16-f16-batch-v1"]
            }
            vectors, retries = await reindex.encode_backend(client, ["one"])
        assert retries == 0
        assert calls == ["/encode-packed"]
        assert vectors == [reindex.unpack_sparse(reindex.pack_sparse({
            "2": 1.2345, "1": 0.25,
        }))]

    asyncio.run(run_case())


def test_partial_bulk_retries_only_rejected_items():
    async def run_case():
        bodies = []

        def handler(request):
            bodies.append(request.content.decode())
            if len(bodies) == 1:
                return httpx.Response(200, json={"errors": True, "items": [
                    {"index": {"status": 201, "_id": "a"}},
                    {"index": {"status": 429, "_id": "b", "error": "rejected_execution"}},
                ]})
            return httpx.Response(200, json={"errors": False, "items": [
                {"index": {"status": 201, "_id": "b"}},
            ]})

        actions = [
            ({"index": {"_index": "dst", "_id": "a"}}, {"value": 1}),
            ({"index": {"_index": "dst", "_id": "b"}}, {"value": 2}),
        ]
        async with httpx.AsyncClient(
            base_url="http://es", transport=httpx.MockTransport(handler)
        ) as client:
            indexed, retries, sent = await reindex.send_bulk(
                client, actions, retry_delay=0
            )
        assert indexed == 2
        assert retries == 1
        assert sent == len(bodies[0].encode()) + len(bodies[1].encode())
        assert '"_id":"a"' in bodies[0]
        assert '"_id":"a"' not in bodies[1]
        assert '"_id":"b"' in bodies[1]

    asyncio.run(run_case())


def test_run_metadata_mismatch_is_refused():
    reindex.verify_metadata({"source": "a", "slices": 4}, {"source": "a", "slices": 4})
    with pytest.raises(ValueError, match="slices"):
        reindex.verify_metadata(
            {"source": "a", "slices": 4}, {"source": "a", "slices": 8}
        )


def test_expired_pit_resets_incomplete_slice_cursor():
    item = {
        "pit_id": "old",
        "completed": False,
        "search_after": [20],
        "stats": {"docs": 20},
    }
    reset, reused = reindex.prepare_slice_resume(item, False)
    assert not reused
    assert reset["pit_id"] is None
    assert reset["search_after"] is None
    assert reset["stats"] == reindex.empty_stats()
    assert item["search_after"] == [20]


def test_live_pit_reuses_exact_cursor_and_identity_change_refuses_resume():
    item = {
        "pit_id": "live",
        "completed": False,
        "search_after": [42],
        "stats": {},
    }
    resumed, reused = reindex.prepare_slice_resume(item, True, "latest")
    assert reused
    assert resumed["pit_id"] == "latest"
    assert resumed["search_after"] == [42]
    with pytest.raises(ValueError, match="identity"):
        reindex.prepare_slice_resume(item, True, identity_matches=False)


def test_capacity_policy_rejects_limits_and_chooses_fastest_within_cost_band():
    candidates = [
        {"name": "cheap", "total_cost": 60, "projected_runtime_hours": 7},
        {"name": "fast-near", "total_cost": 68, "projected_runtime_hours": 2},
        {"name": "too-costly", "total_cost": 81, "projected_runtime_hours": 1},
        {"name": "too-slow", "total_cost": 20, "projected_runtime_hours": 9},
        {"name": "outside-band", "total_cost": 70, "projected_runtime_hours": 1},
    ]
    assert reindex.select_capacity(candidates)["name"] == "fast-near"
    assert reindex.select_capacity([
        {"name": "bad", "total_cost": 81, "projected_runtime_hours": 9},
    ]) is None


def test_full_run_requires_explicit_confirmation():
    args = reindex.parse_args([
        "run", "--backend-urls", "http://gpu", "--aggregation", "v1",
    ])
    with pytest.raises(ValueError, match="confirm-full-run"):
        asyncio.run(reindex.command_run(args))


def test_aggregation_choice_is_pinned_in_run_metadata():
    args = reindex.parse_args([
        "run", "--backend-urls", "http://gpu", "--limit", "1",
        "--aggregation", "v1",
    ])
    identity = {"mapping_digest": "mapping"}
    args.encoding_version = "prod-soup-top256-fp32-fp16codec-v1"
    args.backend_contract = {"encoder_implementation": "test"}
    metadata = reindex.make_metadata(
        args, identity, {"concrete_index": "dst", "uuid": "uuid"}, "s2"
    )
    assert metadata["transform_version"] == (
        "article-splade-aggregation-bakeoff-v1"
    )
    assert metadata["aggregation"] == "v1"


def test_default_aggregation_is_full_scale_winner_v1():
    args = reindex.parse_args([
        "run", "--backend-urls", "http://gpu", "--limit", "1",
    ])
    assert args.aggregation == "v1"


def test_approval_report_pins_runtime_cost_batch_and_encoding(tmp_path):
    args = reindex.parse_args([
        "run", "--backend-urls", "http://gpu", "--confirm-full-run",
    ])
    report = {
        "aggregation": "v1",
        "encode_batch_size": 256,
        "end_to_end_docs_per_second": 5_000,
        "projected_runtime_hours": 7.5,
        "projected_total_cost": 40,
        "document_encoding_version": "prod-soup-top256-bf16-fp16codec-v1",
        "backend_contract": {"encoder_implementation": "test"},
        "retrieval_parity_passed": True,
        "bakeoff_report_sha256": reindex.BAKEOFF_REPORT_SHA256,
        "h11_selected_aggregation": "v1",
        "h11_v5_incremental_union_recall": 0.0,
    }
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(report))
    assert reindex.load_approval_report(path, args) == report
    report["end_to_end_docs_per_second"] = 4_899
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="throughput"):
        reindex.load_approval_report(path, args)


def test_invalid_numeric_run_arguments_are_rejected():
    with pytest.raises(SystemExit):
        reindex.parse_args([
            "run", "--backend-urls", "http://gpu", "--limit", "-1",
        ])
    with pytest.raises(SystemExit):
        reindex.parse_args([
            "run", "--backend-urls", "http://gpu", "--limit", "1",
            "--encode-batch-size", "0",
        ])
