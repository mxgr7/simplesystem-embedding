import numpy as np

from indexer.sparse_codec import merge_max, pack_sparse, unpack_sparse


def test_roundtrip_keys_and_values():
    vec = {5: 1.5, 31101: 0.25, 100: 3.0}
    out = unpack_sparse(pack_sparse(vec))
    assert set(out) == {"5", "31101", "100"}
    # fp16 round-trip: exact for these values
    assert out["5"] == 1.5 and out["100"] == 3.0 and out["31101"] == 0.25


def test_empty():
    assert unpack_sparse(pack_sparse({})) == {}


def test_large_random_roundtrip():
    rng = np.random.default_rng(0)
    ids = rng.choice(31102, size=800, replace=False)
    vec = {int(i): float(np.float16(rng.random() * 5)) for i in ids}
    out = unpack_sparse(pack_sparse(vec))
    assert len(out) == 800
    for i, w in vec.items():
        assert out[str(i)] == w


def test_merge_max():
    a = {"1": 0.5, "2": 2.0}
    b = {"2": 1.0, "3": 3.0}
    assert merge_max([a, b]) == {"1": 0.5, "2": 2.0, "3": 3.0}
    assert merge_max([a]) is a  # single-vector fast path
