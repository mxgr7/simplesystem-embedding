"""Every way `ce-service` can be misconfigured must fail at boot, loudly.

The bar these tests hold is not "an exception is raised" — it is that the
message NAMES THE ENV VAR that pins the value, because the person reading it is
looking at a container that will not start on a box they did not build. Each
test therefore asserts on the text, not just the type.

The corresponding failure modes, if these did not fire, are all silent: a
tokenizer swapped underneath a version string, an emulated-bf16 backend at 10x
the cost, a 3-class head broadcasting against a 4-element gain vector, a budget
no legal request can finish inside. None of them raise at request time.

MXG-144.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ce-service"))

from ceserve import scorer as S  # noqa: E402
from ceserve.constants import BACKBONE, VOCAB_SIZE  # noqa: E402


def _config_dict(**overrides):
    payload = {
        "architectures": [BACKBONE],
        "vocab_size": VOCAB_SIZE,
        "num_labels": 4,
        "id2label": {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2", "3": "LABEL_3"},
        "pad_token_id": 1,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "max_position_embeddings": 514,
    }
    payload.update(overrides)
    return payload


def _model_dir(tmp_path, nested=False, weights=b"weights", tokenizer=b"tokenizer",
               config=None):
    root = tmp_path / "release-2026-08-14"
    target = root / "run_name" if nested else root
    target.mkdir(parents=True)
    (target / "config.json").write_text(json.dumps(config or _config_dict()))
    (target / "model.safetensors").write_bytes(weights)
    (target / "tokenizer.json").write_bytes(tokenizer)
    (target / "tokenizer_config.json").write_text("{}")
    return root, target


# --------------------------------------------- 1. the nested-directory trap --

def test_nested_release_directory_names_the_inner_path(tmp_path):
    """`ship_mxg84.sh` nests the run name inside the dated release directory.

    Pointing at the outer one otherwise raises a tokenizer backend error that
    names nothing — the trap `pipeline/ce_serve_skew.py` records in a comment.
    """
    root, inner = _model_dir(tmp_path, nested=True)
    with pytest.raises(RuntimeError) as exc:
        S.resolve_model_dir(str(root))
    message = str(exc.value)
    assert str(inner) in message, "the message must hand back the path to use"
    assert "CE_MODEL_DIR" in message
    assert "ship_mxg84.sh" in message


def test_flat_release_directory_resolves_to_itself(tmp_path):
    _, target = _model_dir(tmp_path, nested=False)
    assert S.resolve_model_dir(str(target)) == str(target)


def test_missing_directory_and_missing_files(tmp_path):
    with pytest.raises(RuntimeError, match="not a directory"):
        S.resolve_model_dir(str(tmp_path / "nope"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError) as exc:
        S.resolve_model_dir(str(empty))
    assert "config.json" in str(exc.value)


# ------------------------------------------------------ 2, 3, 4. the pins --

def test_weights_digest_mismatch(tmp_path, monkeypatch):
    _, target = _model_dir(tmp_path, weights=b"not the checkpoint")
    monkeypatch.setattr(S, "MODEL_SHA256", "0" * 64)
    monkeypatch.setattr(
        S, "TOKENIZER_SHA256", hashlib.sha256(b"tokenizer").hexdigest())
    with pytest.raises(RuntimeError) as exc:
        S.assert_digests(str(target))
    assert "checkpoint SHA mismatch" in str(exc.value)
    assert "CE_MODEL_SHA256" in str(exc.value)


def test_tokenizer_digest_mismatch_explains_the_consequence(tmp_path, monkeypatch):
    """A tokenizer swapped underneath a version string is the one failure mode
    no downstream check would catch — 115,485,224 stored blobs become ids in a
    foreign vocabulary and every score stays plausible."""
    _, target = _model_dir(tmp_path, weights=b"w", tokenizer=b"other vocabulary")
    monkeypatch.setattr(S, "MODEL_SHA256", hashlib.sha256(b"w").hexdigest())
    monkeypatch.setattr(S, "TOKENIZER_SHA256", "0" * 64)
    with pytest.raises(RuntimeError) as exc:
        S.assert_digests(str(target))
    message = str(exc.value)
    assert "tokenizer SHA mismatch" in message
    assert "CE_TOKENIZER_SHA256" in message
    assert "indexer" in message and "different vocabulary" in message


def test_blank_tokenizer_version_is_refused(tmp_path, monkeypatch):
    """Blank matches no candidate's `ceTokenizerVersion`, so every candidate
    would be skipped while the service reported healthy."""
    _, target = _model_dir(tmp_path, weights=b"w", tokenizer=b"t")
    monkeypatch.setattr(S, "MODEL_SHA256", hashlib.sha256(b"w").hexdigest())
    monkeypatch.setattr(S, "TOKENIZER_SHA256", hashlib.sha256(b"t").hexdigest())
    monkeypatch.setattr(S, "TOKENIZER_VERSION", "   ")
    with pytest.raises(RuntimeError) as exc:
        S.assert_digests(str(target))
    assert "CE_TOKENIZER_VERSION" in str(exc.value)


def test_digests_pass_on_the_real_checkpoint_constants(tmp_path, monkeypatch):
    _, target = _model_dir(tmp_path, weights=b"w", tokenizer=b"t")
    monkeypatch.setattr(S, "MODEL_SHA256", hashlib.sha256(b"w").hexdigest())
    monkeypatch.setattr(S, "TOKENIZER_SHA256", hashlib.sha256(b"t").hexdigest())
    S.assert_digests(str(target))  # does not raise


# --------------------------------------------------- 5, 6, 7, 11. the config --

def test_config_accepts_the_shipped_checkpoint():
    S.assert_config(_config_dict(), 192)


@pytest.mark.parametrize(
    "overrides,needle",
    [
        ({"architectures": ["BertForSequenceClassification"]}, "architectures"),
        ({"vocab_size": 250_000}, "vocab_size"),
        ({"num_labels": 2, "id2label": {"0": "a", "1": "b"}}, "labels"),
        ({"pad_token_id": 3}, "pad_token_id"),
        ({"bos_token_id": 101}, "bos_token_id"),
        ({"eos_token_id": 102}, "eos_token_id"),
    ],
)
def test_config_rejects_a_checkpoint_the_splice_is_not_portable_to(overrides, needle):
    with pytest.raises(RuntimeError) as exc:
        S.assert_config(_config_dict(**overrides), 192)
    assert needle in str(exc.value)


def test_special_ids_failure_explains_why_it_matters():
    with pytest.raises(RuntimeError) as exc:
        S.assert_config(_config_dict(pad_token_id=3), 192)
    assert "semantically wrong" in str(exc.value), (
        "a checkpoint with different specials splices a syntactically valid "
        "sequence — the message has to say so"
    )


@pytest.mark.parametrize("max_len", [4, 7, 513, 1024])
def test_max_len_outside_the_model_window_is_refused(max_len):
    with pytest.raises(RuntimeError) as exc:
        S.assert_config(_config_dict(), max_len)
    assert "CE_MAX_LEN" in str(exc.value)


def test_max_len_512_is_the_ceiling_not_514():
    """XLM-R offsets positions past `padding_idx`, so the usable width is
    `max_position_embeddings - 2`. The indexer runs 512 on purpose."""
    S.assert_config(_config_dict(), 512)
    with pytest.raises(RuntimeError):
        S.assert_config(_config_dict(), 513)


# ----------------------------------------------------- 8, 9, 10. the device --

def test_cpu_is_refused_unless_explicitly_allowed():
    with pytest.raises(RuntimeError) as exc:
        S.resolve_device("cpu", allow_cpu=False)
    message = str(exc.value)
    assert "CE_ALLOW_CPU" in message
    assert "150 ms" in message and "healthcheck" in message
    assert S.resolve_device("cpu", allow_cpu=True).type == "cpu"


def test_cuda_without_cuda_is_refused(monkeypatch):
    monkeypatch.setattr(S.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CE_DEVICE=cuda"):
        S.resolve_device("cuda", allow_cpu=True)


def test_pre_turing_capability_is_refused(monkeypatch):
    monkeypatch.setattr(S.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(S.torch.cuda, "get_device_capability", lambda *a: (6, 1))
    with pytest.raises(RuntimeError, match="compute capability"):
        S.resolve_device("cuda", allow_cpu=True)


def test_bf16_on_a_card_without_native_bf16_is_refused(monkeypatch):
    """The whole point of `_native_bf16`. `is_bf16_supported()` answers True on
    sm_75 and the process would start silently on emulated bf16 at ~10x cost."""
    monkeypatch.setattr(S, "_native_bf16", lambda: False)
    monkeypatch.setattr(S.torch.cuda, "get_device_capability", lambda *a: (7, 5))
    with pytest.raises(RuntimeError) as exc:
        S.resolve_dtype("bf16", S.torch.device("cuda"))
    message = str(exc.value)
    assert "emulated bf16" in message and "CE_DTYPE" in message


def test_bf16_on_a_native_card_is_allowed(monkeypatch):
    monkeypatch.setattr(S, "_native_bf16", lambda: True)
    name, dtype = S.resolve_dtype("bf16", S.torch.device("cuda"))
    assert name == "bf16" and dtype is S.torch.bfloat16


def test_unknown_dtype_is_refused():
    with pytest.raises(RuntimeError, match="CE_DTYPE"):
        S.resolve_dtype("int4", S.torch.device("cpu"))


def test_cpu_forces_fp32_regardless_of_request():
    assert S.resolve_dtype("fp16", S.torch.device("cpu")) == ("fp32", S.torch.float32)


# ------------------------------------------------------------ 12. the budget --

def test_budget_and_max_inputs_must_be_consistent():
    """`embedding-service` shipped MAX_INPUTS=256 against a budget nothing could
    finish inside, and every legal request timed out."""
    with pytest.raises(RuntimeError) as exc:
        S.assert_budget_is_consistent(1000, 0.5)
    message = str(exc.value)
    assert "MAX_INPUTS_PER_REQUEST" in message and "REQUEST_BUDGET_S" in message
    assert "embedding-service" in message


def test_the_shipped_defaults_are_consistent():
    # 0.654 * 256 + 0.3 = 167.7 ms, inside a 500 ms response budget.
    S.assert_budget_is_consistent(256, 0.5)


def test_the_budget_check_uses_the_measured_slope():
    assert (S.MS_PER_CANDIDATE, S.MS_FIXED) == (0.654, 0.3), (
        "report/pipeline_v2/ce_latency_t4_v1.md, k-sweep least squares at k>=60"
    )


# ---------------------------------------------------- 13. the golden fixture --

def test_golden_fixture_passes_as_shipped():
    cases, rows = S.assert_golden_fixture()
    assert cases >= 10 and rows >= 25


def test_a_broken_splice_fails_the_boot(monkeypatch):
    """The check that makes a wrong splice a startup failure on the serving box
    rather than a quietly wrong SERP."""
    real = S.assemble

    def sabotaged(q_ids, arts, max_len, pad_to):
        ids, mask, max_seq = real(q_ids, arts, max_len, pad_to)
        ids = ids.copy()
        ids[0, 0] = 9  # one token, in the BOS slot
        return ids, mask, max_seq

    monkeypatch.setattr(S, "assemble", sabotaged)
    with pytest.raises(RuntimeError) as exc:
        S.assert_golden_fixture()
    assert "GOLDEN SPLICE MISMATCH" in str(exc.value)
    assert "plausible scores" in str(exc.value)


def test_a_stale_fixture_is_refused(tmp_path, monkeypatch):
    payload = json.loads(S.SPLICE_FIXTURE.read_text())
    payload["spliceVersion"] = "splice_v0"
    path = tmp_path / "splice_fixture.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="spliceVersion"):
        S.assert_golden_fixture(path)


def test_a_fixture_from_another_tokenizer_is_refused(tmp_path):
    payload = json.loads(S.SPLICE_FIXTURE.read_text())
    payload["tokenizerSha256"] = "0" * 64
    path = tmp_path / "splice_fixture.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="different tokenizer"):
        S.assert_golden_fixture(path)


# ------------------------------------------------ 14. the query contract ----

def test_query_contract_check_passes_as_shipped():
    S.assert_query_contract()


def test_query_contract_check_covers_the_tokenizer_path():
    class Recorder:
        def encode(self, text, add_special_tokens=False):
            class Enc:
                ids = [hash(w) % 1000 for w in text.split()]

            return Enc()

    S.assert_query_contract(Recorder())


def test_a_broken_fold_fails_the_boot(monkeypatch):
    """A merge that collapses whitespace (the SPLADE normalize_text chain) or
    resurrects the prefix must be a startup failure, not a plausible score."""
    monkeypatch.setattr(S, "fold_de", lambda text: " ".join(str(text).split()))
    with pytest.raises(RuntimeError) as exc:
        S.assert_query_contract()
    assert "fold-de-v1-no-prefix" in str(exc.value)


def test_a_resurrected_prefix_fails_the_boot(monkeypatch):
    real = S.encode_query

    def prefixed(tokenizer, query):
        return real(tokenizer, f"[P_product_noun] {query}")

    class Recorder:
        def encode(self, text, add_special_tokens=False):
            class Enc:
                ids = [hash(w) % 1000 for w in text.split()]

            return Enc()

    monkeypatch.setattr(S, "encode_query", prefixed)
    with pytest.raises(RuntimeError) as exc:
        S.assert_query_contract(Recorder())
    assert "fold-de-v1-no-prefix" in str(exc.value)


# ------------------------------------------------------------- ce_score ----

def test_ce_score_is_the_offline_formula():
    """`train_ce.py`: sum(p * [4,2,1,0]) / 4. One implementation, so the
    agreement check cannot be comparing two different quantities."""
    import numpy as np
    assert S.ce_score(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert S.ce_score(np.array([0.0, 0.0, 0.0, 1.0])) == pytest.approx(0.0)
    assert S.ce_score(np.array([0.0, 1.0, 0.0, 0.0])) == pytest.approx(0.5)
    assert S.ce_score(np.array([0.25] * 4)) == pytest.approx(7 / 16)
    batch = S.ce_score(np.array([[1.0, 0, 0, 0], [0, 0, 0, 1.0]]))
    assert batch.tolist() == pytest.approx([1.0, 0.0])
