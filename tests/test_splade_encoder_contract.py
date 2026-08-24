"""MXG-111: the encoder contract, and the two silent failures it exists to stop.

`model_metadata()` pins which checkpoint a backend serves. It does not pin how that
checkpoint is executed, so an H100 on bf16 and a T4 on fp16 agree on every key in it
while producing measurably different vectors (measured on the MXG-111 sample:
top-256 jaccard 0.9762 between emulated-bf16 and fp32-exact on one card). Both
failures below are silent without these checks -- no exception, no log line, just
vectors that disagree with the ones already in the index.

MXG-203 added the third one, a floor below the first two: a backend can disagree
with *itself*. `/encode` and `/encode-packed` are one encoder under one
`document_encoding_version`, so the pool's checks are only worth what the
agreement between the two transports is worth.

The transport tests below use a tiny checkpoint stub on CPU. The real-checkpoint
probe in `test_splade_checkpoint_parity.py` self-skips unless
`SPLADE_PARITY_CHECKPOINT` names a checkpoint. The CUDA backstop is the box-side
`/workspace/pipeline/mxg111_dtype_parity.py compare --mode pair --tag <candidate>
--against prodt4` agreement gate. A green stub run does not claim bit-identical
CUDA output.
"""
import asyncio
import importlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from conftest import load_flat_service

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "splade-service"
splade = load_flat_service(
    "splade_service", SERVICE, "backend_pool", "constants", "backend", "codec"
)
backend_pool = splade.backend_pool
constants = splade.constants


BF16 = {
    "document_compute_dtype": "bf16",
    "document_encoding_version": "prod-soup-top256-bf16-fp16codec-wcast-chead-v2",
    "fold_vocab_mask": True,
    "vocab_mask_sha256": "9e022b943bfe90b1",
}
FP16 = dict(BF16,
            document_compute_dtype="fp16",
            document_encoding_version="prod-soup-top256-fp16-fp16codec-wcast-chead-v2")


class FakeBackend:
    """A backend that has already passed verify(), carrying only its metadata."""

    def __init__(self, backend_id, metadata, url="http://backend"):
        self.id = backend_id
        self.url = url
        self.metadata = dict(metadata)


def _pool(*backends):
    pool = backend_pool.BackendPool()
    pool.backends = {b.id: b for b in backends}
    return pool


def test_matching_encoder_contract_is_accepted():
    pool = _pool(FakeBackend("b1", FP16))
    pool._check_encoder_contract(FakeBackend("b2", FP16, "http://second"))


def test_burst_backend_on_a_different_dtype_is_rejected():
    """The concrete MXG-111 path: an H100 registered for a re-encode burst behind
    the frontend that already fronts the fp16 T4. Same checkpoint, same sha, same
    vocab mask -- and vectors that do not match, written into a shared cache."""
    pool = _pool(FakeBackend("b1", FP16))

    with pytest.raises(ValueError) as excinfo:
        pool._check_encoder_contract(FakeBackend("b2", BF16, "http://h100-burst"))

    message = str(excinfo.value)
    assert "http://h100-burst" in message
    assert "document_compute_dtype" in message


def test_a_differing_vocab_mask_is_rejected_even_on_the_same_dtype():
    """Serving a fold-masked checkpoint without the mask takes query nnz from 5 to
    ~4,530. The mask digest is part of the contract for that reason."""
    pool = _pool(FakeBackend("b1", FP16))
    unmasked = dict(FP16, fold_vocab_mask=False, vocab_mask_sha256="0000000000000000")

    with pytest.raises(ValueError, match="fold_vocab_mask"):
        pool._check_encoder_contract(FakeBackend("b2", unmasked, "http://unmasked"))


def test_declared_encoding_version_must_match_the_backend(monkeypatch):
    """`SPLADE_ENCODING_VERSION` namespaces the cache keyspace, so a backend that
    disagrees with it would file its vectors under a name that does not describe
    them -- indistinguishable, afterwards, from the ones that do."""
    monkeypatch.setattr(backend_pool, "ENCODING_VERSION",
                        BF16["document_encoding_version"])
    pool = backend_pool.BackendPool()

    with pytest.raises(ValueError, match="SPLADE_ENCODING_VERSION"):
        pool._check_encoder_contract(FakeBackend("b1", FP16))


def test_first_backend_is_accepted_against_an_empty_pool():
    pool = backend_pool.BackendPool()
    pool._check_encoder_contract(FakeBackend("b1", FP16))


def test_cache_prefix_is_namespaced_by_encoding_version_when_declared():
    """Without this, an fp16 and a bf16 encode of the same input collide on one key.

    Reloaded rather than asserted on the imported module because CACHE_PREFIX is
    computed at import time from the environment.
    """
    import os

    previous = os.environ.get("SPLADE_ENCODING_VERSION")
    try:
        os.environ["SPLADE_ENCODING_VERSION"] = FP16["document_encoding_version"]
        reloaded = splade.reload("constants")
        assert reloaded.CACHE_PREFIX == (
            f"splade:{reloaded.MODEL_ID}:{FP16['document_encoding_version']}:")

        del os.environ["SPLADE_ENCODING_VERSION"]
        reloaded = splade.reload("constants")
        # Unset degrades to the historical model-only prefix, which is what the
        # existing keyspace holds -- declaring the version must be opt-in or the
        # first deploy silently orphans every cached vector.
        assert reloaded.CACHE_PREFIX == f"splade:{reloaded.MODEL_ID}:"
    finally:
        if previous is None:
            os.environ.pop("SPLADE_ENCODING_VERSION", None)
        else:
            os.environ["SPLADE_ENCODING_VERSION"] = previous
        splade.reload("constants")


def test_native_bf16_predicate_rejects_turing():
    """The guard must ask about NATIVE bf16. `is_bf16_supported()` defaults to
    `including_emulation=True` and answers True on sm_75, so the check it was
    written for never fired and the backend ran on emulated bf16 at 2.48 TFLOP/s
    against fp16's 25.50 -- measured on the T4 this issue is about."""
    backend = splade.backend
    torch = importlib.import_module("torch")

    class FakeCuda:
        def __init__(self, capability):
            self.capability = capability
            self.asked_with_emulation = None

        def is_bf16_supported(self, including_emulation=True):
            self.asked_with_emulation = including_emulation
            return including_emulation or self.capability[0] >= 8

        def get_device_capability(self, device=None):
            return self.capability

    turing = FakeCuda((7, 5))
    ampere = FakeCuda((8, 0))
    original = torch.cuda
    try:
        torch.cuda = turing
        assert backend._native_bf16() is False
        assert turing.asked_with_emulation is False
        torch.cuda = ampere
        assert backend._native_bf16() is True
    finally:
        torch.cuda = original


# --- MXG-203: the two document transports, and the one arithmetic they share ---
#
# `SpladeEncoder` has two document code paths. `encode()` serves `/encode`, which
# is what `BackendPool` posts to, which is what the frontend's `/embed` uses,
# which is what the indexer uses; `encode_packed()` serves `/encode-packed`,
# which is what the bulk reindex uses. They write into one cache keyspace, and
# that keyspace is namespaced by `document_encoding_version` -- a string that
# names one compute dtype. Only `_encode_batch` used to consult it: `encode()`
# ran in whatever dtype the *weights* happened to be in, so any profile that set
# `DOCUMENT_DTYPE` without `DOCUMENT_WEIGHTS_CAST` (`compose.gpu.yaml` is exactly
# that) served fp32 through the transport that declared bf16. Measured on the
# stub below before the fix: 234 of 256 weights differed on the first document.
#
# These tests build a real (tiny) BertForMaskedLM through the real constructor
# rather than a mock, because the thing under test is the arithmetic.

STUB_VOCAB_SIZE = 1024
STUB_SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
STUB_CONFIG = {
    "architectures": ["BertForMaskedLM"],
    "model_type": "bert",
    "vocab_size": STUB_VOCAB_SIZE,
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
}
# Deliberately ragged: `encode` pads each chunk to its longest row while
# `encode_packed` sorts by length first, so equal-length inputs would hide any
# disagreement that comes from batching rather than from dtype.
STUB_TEXTS = ["w1 w2 w3", "w4 w5 w6 w7 w8 w9 w10 w11", "w12"]


def _stub_tokenizer(tmp_path):
    """The service's own fallback tokenizer, over a 1,024-token vocabulary.

    `load_tokenizer`'s WordPiece branch verbatim, so the test needs no network
    and still exercises a real fast tokenizer -- offsets, specials and padding
    included.
    """
    from tokenizers.implementations import BertWordPieceTokenizer
    from transformers import PreTrainedTokenizerFast

    vocab = tmp_path / "vocab.txt"
    tokens = list(STUB_SPECIALS) + [
        f"w{index}" for index in range(STUB_VOCAB_SIZE - len(STUB_SPECIALS))
    ]
    vocab.write_text("\n".join(tokens) + "\n")
    inner = BertWordPieceTokenizer(
        str(vocab), lowercase=False, strip_accents=False
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=inner._tokenizer,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        model_max_length=512,
    )


def _stub_encoder(tmp_path, dtype="bf16"):
    """A real `SpladeEncoder` over a 1-layer BERT, configured for `dtype`.

    Only what needs the network or the production checkpoint is replaced -- the
    digest check, `torch.load`, the backbone config, and the tokenizer download
    -- so the encoder is otherwise built by the constructor under test, vocab
    mask and encoding version included.

    The dtype is set on the instance afterwards because `__init__` pins any
    non-CUDA device to fp32, and the divergence this file is about only exists
    off fp32. CPU autocast is real for bf16, which is what makes that legal here.
    """
    import transformers

    torch = importlib.import_module("torch")
    torch.manual_seed(0)
    source_model = transformers.BertForMaskedLM(
        transformers.BertConfig.from_dict(STUB_CONFIG)
    )
    checkpoint = {
        "hyper_parameters": {
            "model": {
                "model_name": constants.MODEL_NAME,
                "fold_vocab_mask": False,
                "stopword_mask_ids": [],
            }
        },
        "state_dict": {
            f"encoder.{name}": value
            for name, value in source_model.state_dict().items()
        },
    }
    path = tmp_path / "stub.ckpt"
    path.write_bytes(b"stub")

    backend = splade.backend
    with patch.object(
        backend, "checkpoint_sha256", return_value=constants.MODEL_SHA256
    ), patch.object(
        backend.torch, "load", return_value=checkpoint
    ), patch.object(
        transformers.PretrainedConfig,
        "get_config_dict",
        return_value=(dict(STUB_CONFIG), {}),
    ), patch.object(
        backend, "load_tokenizer", return_value=_stub_tokenizer(tmp_path)
    ):
        encoder = backend.SpladeEncoder(str(path), "cpu", str(tmp_path), 8)

    encoder.document_dtype_name = dtype
    encoder.document_dtype = {
        "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None
    }[dtype]
    return encoder


def _record_forward_dtype(encoder):
    """The dtype the forward actually ran in, as the model itself reports it."""
    seen = []
    encoder.model.register_forward_hook(
        lambda module, args, output: seen.append(output.logits.dtype)
    )
    return seen


def _as_packed(vector):
    """A JSON-transport vector rounded through the packed transport's codec.

    The two transports declare different wire precisions -- `json-map-v1` carries
    the value, `splade-u16-f16-batch-v1` carries float16 -- and the encoding
    version says so (`-fp16codec`). Rounding the JSON side is what makes the
    comparison a question about the *encoder* rather than about the codec.
    """
    rounded = {}
    for token, weight in vector.items():
        value = float(np.float16(weight))
        if value > 0:
            rounded[token] = value
    return rounded


def test_both_document_transports_run_the_configured_dtype(tmp_path):
    """`/encode` used to ignore `DOCUMENT_DTYPE` entirely."""
    encoder = _stub_encoder(tmp_path, "bf16")
    torch = importlib.import_module("torch")
    seen = _record_forward_dtype(encoder)

    encoder.encode(STUB_TEXTS)
    json_dtypes = list(seen)
    seen.clear()
    encoder.encode_packed(STUB_TEXTS)
    packed_dtypes = list(seen)

    assert json_dtypes and packed_dtypes
    assert set(json_dtypes) == {torch.bfloat16}, (
        f"/encode ran in {set(json_dtypes)}, not the configured bf16 -- the "
        f"transport the indexer uses does not honour DOCUMENT_DTYPE"
    )
    assert set(packed_dtypes) == {torch.bfloat16}


def test_query_vectors_are_encoded_in_the_document_dtype(tmp_path):
    """A query vector and a document vector meet in a dot product.

    `encode(document=False)` is the query path, and it is the same forward, so
    there is no configuration in which the two sides should disagree on dtype.
    """
    encoder = _stub_encoder(tmp_path, "bf16")
    torch = importlib.import_module("torch")
    seen = _record_forward_dtype(encoder)

    encoder.encode(["w1 w2 w3"], document=False)

    assert set(seen) == {torch.bfloat16}


def test_json_and_packed_transports_produce_the_same_vectors(tmp_path):
    """Same encoder, same inputs, one `document_encoding_version` -- so one vector.

    Before the fix this failed on 234 of the first document's 256 weights: an
    fp32 forward through `/encode` against a bf16 one through `/encode-packed`,
    both filed under the bf16 name.
    """
    encoder = _stub_encoder(tmp_path, "bf16")

    from_json = encoder.encode(STUB_TEXTS, document=True)
    from_packed = splade.codec.unpack_sparse_batch(encoder.encode_packed(STUB_TEXTS))

    assert len(from_json) == len(from_packed) == len(STUB_TEXTS)
    for text, json_vector, packed_vector in zip(STUB_TEXTS, from_json, from_packed):
        assert _as_packed(json_vector) == packed_vector, (
            f"transports disagree on {text!r}"
        )


def test_fp32_configuration_autocasts_neither_transport(tmp_path):
    """The other direction: sharing one context must not force a cast on fp32.

    `_autocast()` returns a null context for fp32 and for the weights-cast case
    (where the weights already carry the dtype), and both transports have to see
    the same null.
    """
    encoder = _stub_encoder(tmp_path, "fp32")
    torch = importlib.import_module("torch")
    seen = _record_forward_dtype(encoder)

    encoder.encode(STUB_TEXTS)
    encoder.encode_packed(STUB_TEXTS)

    assert set(seen) == {torch.float32}


def test_model_metadata_claims_nothing_about_execution():
    """MXG-203: `precision: float32` was a literal in a dict of model identity.

    It was also false -- the live T4 casts the weights to fp16 -- and it sat in
    the one dict every backend is compared against field for field, which is the
    place a value that cannot be checked does the most harm. How a checkpoint is
    executed belongs to the backend that executes it.
    """
    assert "precision" not in constants.model_metadata()


def test_max_first_activation_is_exact(tmp_path):
    """The rewrite that made the two transports comparable must change no number.

    `encode()` used to run `log1p(relu(.))` over the whole [B,L,V] head and take
    the max afterwards; it now masks the logits, takes the max, and applies the
    activation to the [B,V] result -- `_encode_batch`'s shape, which is the only
    reason the transports can be compared at all. `log1p(relu(.))` is monotone
    non-decreasing, so the two agree elementwise; this asserts that in float32,
    which is what the real-checkpoint parity test (skipped without a checkpoint)
    would otherwise be the only thing standing behind.
    """
    encoder = _stub_encoder(tmp_path, "fp32")
    torch = importlib.import_module("torch")

    tokens = encoder.tokenizer(
        STUB_TEXTS, padding=True, truncation=True,
        max_length=constants.MAX_OFFER_LENGTH, return_tensors="pt",
    )
    with torch.inference_mode():
        # The pre-MXG-203 body, verbatim.
        logits = encoder.model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        ).logits
        activations = torch.log1p(torch.relu(logits))
        activations *= tokens["attention_mask"].unsqueeze(-1).to(activations.dtype)
        reference = activations.amax(dim=1)
        reference.masked_fill_(~encoder.mask, 0)

    for row, vector in zip(reference, encoder.encode(STUB_TEXTS)):
        values, ids = torch.topk(row, constants.TOP_K, dim=0, sorted=True)
        expected = {
            str(int(token_id)): float(weight)
            for token_id, weight in zip(ids, values) if weight > 0
        }
        assert vector == expected
