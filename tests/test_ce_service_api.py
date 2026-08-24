"""`ce-service` HTTP contract, driven against a stub scorer.

No checkpoint, no GPU: what is under test here is the request/response contract
and the disposition of every candidate, not the model. The model's own numbers
are covered by `test_ce_splice_parity.py` (the splice) and, from the research
repo and against a running service, `pipeline/ce_service_agreement.py` (the
scores). The latter used to be cited here as `test_ce_score_agreement.py`, a
path in this repo that has never existed — MXG-219.

The contract this pins, in one sentence: **every input id comes back exactly
once, in `results` or in `skipped`, in request order**. A response that silently
returned 118 scores for 120 candidates would reorder a subset of the caller's
window and look perfectly healthy.

MXG-144.
"""
import asyncio
import base64
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ce-service"))

from ceserve import app as appmod  # noqa: E402
from ceserve.constants import (  # noqa: E402
    MODEL_SHA256,
    QUERY_CONTRACT,
    TOKENIZER_VERSION,
)
from ceserve.splice import assemble  # noqa: E402

OTHER_VERSION = "ce_dist_l12_v3-2026-07-18"  # the pre-MXG-108 stamp


def pack(ids):
    return base64.b64encode(np.asarray(ids, dtype="<i4").tobytes()).decode("ascii")


_DEFAULT = object()  # so `blob=None` can mean a literal JSON null


def candidate(cid, ids=(10, 11, 12), version=TOKENIZER_VERSION, blob=_DEFAULT,
              count=_DEFAULT):
    return {
        "id": cid,
        "tokens_b64": pack(ids) if blob is _DEFAULT else blob,
        "token_count": len(ids) if count is _DEFAULT else count,
        "tokenizer_version": version,
    }


class StubTokenizer:
    def __init__(self):
        # Every text the service asked to encode, so tests can pin what the
        # model would actually see (the folded query, no prefix).
        self.seen = []

    def no_padding(self):
        pass

    def no_truncation(self):
        pass

    def encode(self, text, add_special_tokens=False):
        self.seen.append(text)

        class Enc:
            ids = list(range(100, 100 + min(len(text.split()), 20)))

        return Enc()


class StubScorer:
    """Runs the REAL splice, returns deterministic probabilities.

    Real splice on purpose: `padded_width`, `truncated` and the query-token
    count are all derived from it, and a stub that faked them would let a
    regression through in exactly the arithmetic the caller uses to predict
    latency.
    """

    max_len = 192
    chunk = 256
    dtype_name = "fp32"
    serving_contract = "ce-stub-text_es-L192-fp32-splice_v1"

    def __init__(self):
        self.tokenizer = StubTokenizer()
        self.warmed_up = True
        self.degraded = False
        self.device = type("D", (), {"type": "cpu", "index": None})()
        self.calls = 0
        self.raises = None

    def warmup(self):
        return (8, 4)

    def score(self, q_ids, arts, max_len):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        ids, _, _ = assemble(q_ids, arts, max_len, None)
        n = len(arts)
        # Deterministic and distinguishable per row, and a valid distribution.
        probs = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            probs[i] = [0.7 - 0.01 * i, 0.2, 0.05 + 0.01 * i, 0.05]
        return probs, {"padded_width": int(ids.shape[1]), "chunks": 1}

    def metadata(self):
        return {
            "model_id": "stub", "model_sha256": MODEL_SHA256,
            "tokenizer_version": TOKENIZER_VERSION,
            "serving_contract": self.serving_contract,
            "query_contract": QUERY_CONTRACT,
            "device": "cpu", "dtype": "fp32", "max_len": self.max_len,
            "warmed_up": self.warmed_up, "degraded": self.degraded,
        }


@pytest.fixture
def scorer():
    return StubScorer()


@pytest.fixture
def client(scorer, monkeypatch):
    monkeypatch.setenv("CE_ALLOW_CPU", "1")
    monkeypatch.setenv("CE_DEVICE", "cpu")
    monkeypatch.setattr(appmod, "CrossEncoderScorer", lambda config: scorer)
    monkeypatch.setattr(appmod, "_exit_process", lambda: None)
    with TestClient(appmod.app) as test_client:
        yield test_client


@pytest.fixture
def keyed_client(scorer, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    monkeypatch.setattr(appmod, "CrossEncoderScorer", lambda config: scorer)
    monkeypatch.setattr(appmod, "_exit_process", lambda: None)
    with TestClient(appmod.app) as test_client:
        yield test_client


def post(client, **overrides):
    body = {"query": "bosch akkuschrauber", "candidates": [candidate("a")]}
    body.update(overrides)
    return client.post("/rerank", json=body)


# ------------------------------------------------------------ the happy path --

def test_scores_every_candidate_in_request_order(client):
    ids = [f"art-{i}" for i in range(5)]
    response = post(client, candidates=[candidate(i) for i in ids])
    assert response.status_code == 200
    body = response.json()
    assert [r["id"] for r in body["results"]] == ids, (
        "request order, not score order — the service does not own ranking policy"
    )
    assert body["n_input"] == 5 and body["n_scored"] == 5 and body["n_skipped"] == 0


def test_ce_score_is_the_gain_weighted_sum(client):
    body = post(client).json()
    row = body["results"][0]
    expected = (row["ce_p_e"] * 4 + row["ce_p_s"] * 2 + row["ce_p_c"] * 1) / 4
    assert row["ce_score"] == pytest.approx(expected, abs=1e-9)
    total = row["ce_p_e"] + row["ce_p_s"] + row["ce_p_c"] + row["ce_p_i"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_response_carries_the_serving_contract_not_just_the_model(client):
    """`model_sha256` cannot express fp16-vs-fp32 or L192-vs-L256, and the
    caller caches reranked windows for 300 s — a rolling model change means two
    contracts in flight."""
    body = post(client).json()
    assert body["serving_contract"]
    assert body["tokenizer_version"] == TOKENIZER_VERSION
    assert body["model_sha256"] == MODEL_SHA256
    assert body["query_contract"] == "fold-de-v1-no-prefix"


def test_padded_width_and_query_token_count_are_reported(client):
    """Latency is set by padded width, a per-window order statistic. Without it
    a latency regression has no explanation."""
    body = post(client, candidates=[candidate("a", ids=range(10)),
                                    candidate("b", ids=range(40))]).json()
    assert body["padded_width"] == 40 + body["query_token_count"] + 4
    # StubTokenizer emits one id per whitespace token, and the encoded text is
    # fold_de("bosch akkuschrauber") -> "bosch akkuschrauber" -> 2. Under the
    # retired prefixed contract this was 3 ("[P_product_noun] ..."); this is
    # what pins that NO prefix is applied (fold-de-v1-no-prefix, MXG-177).
    assert body["query_token_count"] == 2


def test_truncated_flag_marks_articles_over_the_budget(client):
    long_ids = list(range(10, 10 + 400))
    body = post(client, max_len=64,
                candidates=[candidate("short", ids=(1, 2, 3)),
                            candidate("long", ids=long_ids)]).json()
    by_id = {r["id"]: r for r in body["results"]}
    assert by_id["short"]["truncated"] is False
    assert by_id["long"]["truncated"] is True
    assert by_id["long"]["article_token_count"] == 400, (
        "article_token_count is what was STORED, not what survived the trim"
    )


# ---------------------------------------------------------- the version guard --

def test_version_mismatch_is_skipped_never_scored(client):
    body = post(client, candidates=[candidate("ok"),
                                    candidate("stale", version=OTHER_VERSION)]).json()
    assert [r["id"] for r in body["results"]] == ["ok"]
    assert body["skipped"] == [{
        "id": "stale",
        "reason": "tokenizer_version_mismatch",
        "detail": f"got {OTHER_VERSION!r}, expected {TOKENIZER_VERSION!r}",
    }]


def test_all_candidates_skipped_is_a_200_not_an_error(client, scorer):
    """`n_scored: 0` is a correct, informative answer — the index is not
    backfilled for this window — and the caller's fallback handles it. A 5xx
    would fire the wrong alert and hide the real cause."""
    body = post(client, candidates=[candidate(f"c{i}", version=OTHER_VERSION)
                                    for i in range(4)])
    assert body.status_code == 200
    payload = body.json()
    assert payload["n_scored"] == 0 and payload["n_skipped"] == 4
    assert payload["results"] == []
    assert scorer.calls == 0, "nothing to score means no forward"


def test_absent_tokens_are_skipped_as_no_tokens(client):
    """Offer-less articles have the field UNSET (CeTokenEnricher), and some
    documents in the live index carry a matching `ceTokenizerVersion` with NO
    ids — an earlier in-place pass stamped them so it would stop revisiting
    them. Tokens are therefore checked before the version."""
    body = post(client, candidates=[
        candidate("none", blob=None, count=0),
        candidate("blank", blob="", count=0),
        candidate("stamped-but-empty", blob=None, count=0,
                  version=TOKENIZER_VERSION),
    ]).json()
    assert body["n_scored"] == 0
    assert {s["reason"] for s in body["skipped"]} == {"no_tokens"}


def test_a_single_corrupt_blob_does_not_fail_the_window(client):
    candidates = [candidate(f"ok{i}") for i in range(9)]
    candidates.append(candidate("bad", blob="!!!not base64!!!", count=3))
    body = post(client, candidates=candidates)
    assert body.status_code == 200
    payload = body.json()
    assert payload["n_scored"] == 9 and payload["n_skipped"] == 1
    assert payload["skipped"][0]["reason"] == "decode_failed"


def test_token_count_disagreement_is_caught(client):
    """`ceTokenCount` is redundant with `len(tokens_b64)`, and the redundancy is
    the point: a disagreement means a truncated or rewritten `_source`."""
    body = post(client, candidates=[candidate("ok"), candidate("ok2"),
                                    candidate("a", ids=(1, 2, 3), count=99)]).json()
    assert body["n_scored"] == 2
    assert body["skipped"][0]["reason"] == "decode_failed"
    assert "ceTokenCount" in body["skipped"][0]["detail"]


def test_wholesale_decode_failure_escalates_to_400(client):
    """One bad `_source` is noise; most of a window failing is a wire-format
    break, and answering 200 would make it look like a sparsely indexed corpus."""
    candidates = [candidate(f"bad{i}", blob="@@@@", count=1) for i in range(8)]
    candidates.append(candidate("ok"))
    response = post(client, candidates=candidates)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"
    assert "CE_MAX_DECODE_FAILURE_RATIO" in response.json()["detail"]


def test_every_input_id_appears_exactly_once(client):
    """The completeness contract the caller enforces client-side."""
    candidates = ([candidate(f"ok{i}") for i in range(6)]
                  + [candidate("stale", version=OTHER_VERSION)]
                  + [candidate("empty", blob="", count=0)])
    body = post(client, candidates=candidates).json()
    returned = [r["id"] for r in body["results"]] + [s["id"] for s in body["skipped"]]
    assert sorted(returned) == sorted(c["id"] for c in candidates)
    assert len(returned) == len(set(returned)) == body["n_input"]


# ------------------------------------------------- the query contract (MXG-177) --

@pytest.mark.parametrize("value", ["P_product_noun", None, "anything"])
def test_segment_key_is_rejected_with_400(client, value):
    """PRESENCE of the key is the violation, null included: a caller that still
    sends it was built against the retired prefixed contract, and its scores
    would be silently wrong."""
    response = post(client, segment=value)
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "invalid_request"
    assert "segment" in payload["detail"]
    assert QUERY_CONTRACT in payload["detail"]


def test_segment_rejection_logs_error_without_query_text(client, caplog):
    import logging

    secret_query = "hochdruckreiniger kaercher K7"
    blob = pack([10, 11, 12])
    with caplog.at_level(logging.ERROR, logger="ceserve.app"):
        response = client.post("/rerank", json={
            "query": secret_query, "segment": None,
            "candidates": [candidate("a")],
        })
    assert response.status_code == 400
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1, "exactly one contract-violation error record"
    logged = errors[0].getMessage()
    assert "segment" in logged
    assert secret_query not in logged and blob not in logged
    assert secret_query not in response.json()["detail"]


def test_query_is_folded_before_encoding_with_whitespace_preserved(client, scorer):
    post(client, query="Kühlschrank  GROßE")
    assert scorer.tokenizer.seen[-1] == "kuehlschrank  grosse", (
        "fold_de alone — umlauts expand, case folds, and the double space "
        "SURVIVES (no normalize_text collapse; that is the SPLADE chain)"
    )


def test_empty_folded_query_declines_without_inference(client, scorer, caplog):
    import logging

    with caplog.at_level(logging.ERROR, logger="ceserve.app"):
        # Combining acute + diaeresis: nonblank under str.strip(), folds to "".
        response = post(client, query="́̈",
                        candidates=[candidate("a"), candidate("b")])
    assert response.status_code == 200
    body = response.json()
    assert body["declined_reason"] == "empty_folded_query"
    assert body["results"] == [] and body["skipped"] == []
    assert body["n_input"] == 2 and body["n_scored"] == 0 and body["n_skipped"] == 0
    assert body["query_contract"] == QUERY_CONTRACT
    assert scorer.calls == 0, "a decline must not run inference"
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
        "a decline is a degenerate input, not an incident"
    )


def test_whitespace_producing_fold_also_declines(client, scorer):
    # U+00B4 ACUTE ACCENT: NFKD -> space + combining acute -> folds to " ".
    # Whitespace-only folded text tokenizes to zero ids, so it declines too.
    body = post(client, query="´").json()
    assert body["declined_reason"] == "empty_folded_query"
    assert scorer.calls == 0


def test_scored_responses_do_not_carry_declined_reason(client):
    assert "declined_reason" not in post(client).json()


# ------------------------------------------------------------------ max_len --

def test_max_len_override_is_honoured_and_reported(client):
    assert post(client, max_len=128).json()["max_len"] == 128


@pytest.mark.parametrize("bad", [0, 7, 193, 512])
def test_max_len_outside_the_dial_is_a_400(client, bad):
    response = post(client, max_len=bad)
    assert response.status_code == 400 and "max_len" in response.json()["detail"]


# ------------------------------------------------------ request-shape errors --

@pytest.mark.parametrize("body,needle", [
    ({"query": "", "candidates": [candidate("a")]}, "query"),
    ({"candidates": [candidate("a")]}, "query"),
    ({"query": "q", "candidates": []}, "candidates"),
    ({"query": "q"}, "candidates"),
    ({"query": "q", "candidates": [{"tokens_b64": pack([1])}]}, "id"),
])
def test_malformed_requests_are_400_with_our_error_shape(client, body, needle):
    response = client.post("/rerank", json=body)
    assert response.status_code == 400, (
        "the error contract must be ours, not FastAPI's 422 — the models are "
        "documentation, not validation"
    )
    payload = response.json()
    assert payload["error"] == "invalid_request" and needle in payload["detail"]


def test_duplicate_ids_are_refused(client):
    response = post(client, candidates=[candidate("dup"), candidate("dup")])
    assert response.status_code == 400 and "duplicate" in response.json()["detail"]


def test_too_many_candidates_is_413_not_a_silent_truncation(client):
    """`BedrockRerankClient` did `subList(0, MAX_SOURCES)`. Truncating reorders
    a subset of the window and looks healthy; refusing is honest."""
    response = post(client, candidates=[candidate(f"c{i}") for i in range(257)])
    assert response.status_code == 413
    assert response.json()["error"] == "too_many_candidates"
    assert "MAX_INPUTS_PER_REQUEST" in response.json()["detail"]


def test_non_json_body_is_400(client):
    response = client.post("/rerank", content=b"{not json",
                           headers={"content-type": "application/json"})
    assert response.status_code == 400


# ------------------------------------------------- capacity and degradation --

def test_at_capacity_returns_429(client, monkeypatch):
    client.app.state.inflight = 99
    response = post(client)
    assert response.status_code == 429
    assert response.json()["error"] == "at_capacity"


def test_timeout_keeps_gpu_capacity_until_the_worker_stops(client, scorer):
    started = threading.Event()
    release = threading.Event()
    score = scorer.score

    def blocked_score(*args):
        started.set()
        assert release.wait(1)
        return score(*args)

    scorer.score = blocked_score
    client.app.state.config.max_inflight = 1
    client.app.state.config.request_budget_s = 0.01

    response = post(client)
    assert response.status_code == 504
    assert started.is_set()
    assert client.app.state.inflight == 1, (
        "the response timed out, but its GPU worker still owns the slot"
    )

    refused = post(client)
    assert refused.status_code == 429
    assert scorer.calls == 0, "no second forward may start over the orphan"

    release.set()
    deadline = time.monotonic() + 1
    while client.app.state.inflight and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.app.state.inflight == 0
    assert scorer.calls == 1


def test_failure_after_timeout_still_requests_a_restart(client, scorer):
    started = threading.Event()
    release = threading.Event()

    def late_failure(*args):
        started.set()
        assert release.wait(1)
        raise RuntimeError("late CUDA failure")

    scorer.score = late_failure
    client.app.state.config.request_budget_s = 0.01

    response = post(client)
    assert response.status_code == 504
    assert started.is_set()
    assert not client.app.state.restart_requested.is_set()

    release.set()
    deadline = time.monotonic() + 1
    while (
        not client.app.state.restart_requested.is_set()
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert client.app.state.restart_requested.is_set()
    assert scorer.degraded
    assert client.app.state.inflight == 0


def test_inference_failure_degrades_the_service(client, scorer):
    scorer.raises = RuntimeError("CUDA error: an illegal memory access")
    response = post(client)
    assert response.status_code == 500
    assert response.json()["error"] == "inference_failed"
    assert scorer.degraded is True, (
        "a poisoned CUDA context must stop the process answering — one that "
        "keeps going serves garbage"
    )
    assert client.app.state.restart_requested.is_set()
    assert client.get("/readyz").status_code == 503
    assert post(client).status_code == 503
    assert post(client).json()["error"] == "model_not_ready"


def test_inflight_is_released_after_a_failure(client, scorer):
    scorer.raises = RuntimeError("boom")
    post(client)
    assert client.app.state.inflight == 0


def test_restart_watchdog_exits_after_degradation(monkeypatch):
    called = []
    app = SimpleNamespace(
        state=SimpleNamespace(restart_requested=asyncio.Event())
    )
    monkeypatch.setattr(appmod, "RESTART_GRACE_S", 0)
    monkeypatch.setattr(appmod, "_exit_process", lambda: called.append(True))

    async def run():
        task = asyncio.create_task(appmod._restart_when_requested(app))
        app.state.restart_requested.set()
        await task

    asyncio.run(run())
    assert called == [True]


def test_request_outcomes_separate_scoring_declines_and_errors(client):
    counters = {
        outcome: appmod.RERANKS.labels(outcome)
        for outcome in ("scored", "ce_declined", "ce_error")
    }
    before = {name: counter._value.get() for name, counter in counters.items()}

    assert post(client).status_code == 200
    assert post(client, query="́̈").status_code == 200
    assert post(client, query="").status_code == 400

    assert counters["scored"]._value.get() == before["scored"] + 1
    assert counters["ce_declined"]._value.get() == before["ce_declined"] + 1
    assert counters["ce_error"]._value.get() == before["ce_error"] + 1


# ------------------------------------------------------------ the endpoints --

def test_healthz_is_liveness_only(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_readyz_reports_the_warmup(client, scorer):
    body = client.get("/readyz")
    assert body.status_code == 200 and body.json()["ready"] is True
    scorer.warmed_up = False
    body = client.get("/readyz")
    assert body.status_code == 503 and body.json()["checks"]["warmup"] is False


def test_metadata_reports_the_pins_and_the_knobs(client):
    body = client.get("/metadata").json()
    for field in ("model_sha256", "tokenizer_version", "serving_contract",
                  "query_contract", "max_len",
                  "max_inputs_per_request", "max_inflight", "request_budget_s"):
        assert field in body, field
    assert body["query_contract"] == "fold-de-v1-no-prefix"
    assert "default_segment" not in body and "segments" not in body, (
        "the segment vocabulary left with the prefixed contract"
    )


def test_metrics_exposes_the_runtime_collector(client):
    post(client)
    text = client.get("/metrics").text
    for name in ("ce_service_ready", "ce_service_requests_total",
                 "ce_service_rerank_total", "ce_service_candidates_total",
                 "ce_service_padded_width",
                 "ce_service_candidates_per_request"):
        assert name in text, name


# ------------------------------------------------------------------- auth --

def test_public_paths_need_no_key(keyed_client):
    for path in ("/healthz", "/readyz", "/metadata", "/metrics"):
        assert keyed_client.get(path).status_code in (200, 503), path


def test_rerank_requires_the_bearer_when_a_key_is_set(keyed_client):
    assert post(keyed_client).status_code == 401
    body = {"query": "q", "candidates": [candidate("a")]}
    ok = keyed_client.post("/rerank", json=body,
                           headers={"authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    bad = keyed_client.post("/rerank", json=body,
                            headers={"authorization": "Bearer wrong"})
    assert bad.status_code == 401


def test_no_key_configured_means_open(client):
    assert post(client).status_code == 200
