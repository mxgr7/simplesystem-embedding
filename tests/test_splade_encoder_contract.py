"""MXG-111: the encoder contract, and the two silent failures it exists to stop.

`model_metadata()` pins which checkpoint a backend serves. It does not pin how that
checkpoint is executed, so an H100 on bf16 and a T4 on fp16 agree on every key in it
while producing measurably different vectors (measured on the MXG-111 sample:
top-256 jaccard 0.9762 between emulated-bf16 and fp32-exact on one card). Both
failures below are silent without these checks -- no exception, no log line, just
vectors that disagree with the ones already in the index.
"""
import asyncio
import importlib
from pathlib import Path

import pytest

from conftest import load_flat_service

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "splade-service"
splade = load_flat_service(
    "splade_service", SERVICE, "backend_pool", "constants", "backend"
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
