"""Query/article token splicing — the whole correctness surface of this service.

No torch import, on purpose: everything here is exercised by
`tests/test_ce_splice_parity.py` against a committed golden fixture, with no GPU
and no 470 MB checkpoint. A wrong splice produces plausible scores rather than an
error, so this is the one file that gets a boot-time self-check as well as a CI
one (see `scorer.assert_golden_fixture`).

The contract, stated identically in `CeTokenizer`'s KDoc in /next-gen and in
`pipeline/bench_ce_t4.py`:

    [BOS] queryIds [EOS] [EOS] articleIds[:budget] [EOS]   (pad to width with PAD)
    budget = maxLen - queryIds.size - HEAD_EXTRA

The stored article ids carry NO special tokens — the specials belong to the
splice. The article is always the trimmed side.

MXG-144.
"""
import base64

import numpy as np

from ceserve.constants import BOS, EOS, HEAD_EXTRA, MIN_MAX_LEN, PAD, VOCAB_SIZE

class TokenDecodeError(ValueError):
    """A stored `ceTokenIds` blob that cannot be trusted.

    Raised per candidate, never per request: one corrupt `_source` must not fail
    a 120-candidate window.
    """


def decode_token_ids(blob_b64, expected_count=None, vocab_size=VOCAB_SIZE):
    """Decode the ES `binary` field `ceTokenIds` into article-side token ids.

    Wire format, written by `CeTokenizer.pack` in /next-gen:
    base64 on the wire, decoding to little-endian int32 per id. XLM-R's
    vocabulary is 250,002 entries, so ids do not fit in u16.
    """
    if not isinstance(blob_b64, str):
        raise TokenDecodeError(
            f"ceTokenIds is missing or not a string (got {type(blob_b64).__name__})"
        )
    # "" is NOT missing: an article that renders to empty text packs to zero
    # bytes, and `gen_ce_splice_fixture`'s `empty_article` case pins that the
    # splice handles it (`<s> q </s></s> </s>`). Absence is a different thing —
    # `CeTokenEnricher` leaves the fields UNSET for offer-less articles, so the
    # caller sends null and gets the branch above.
    try:
        raw = base64.b64decode(blob_b64, validate=True)
    except Exception as exc:  # binascii.Error and friends
        raise TokenDecodeError(f"ceTokenIds is not valid base64: {exc}") from None
    if len(raw) % 4:
        raise TokenDecodeError(
            f"ceTokenIds length {len(raw)} is not a multiple of 4 "
            "(little-endian int32 per id)"
        )
    # "<i4", never np.int32: the Kotlin writes ByteBuffer.LITTLE_ENDIAN, and a
    # native-order read is a silent byte-swap on a big-endian host — which
    # produces ids that are still in range and still score.
    ids = np.frombuffer(raw, dtype="<i4")
    if expected_count is not None and int(expected_count) != ids.shape[0]:
        # ceTokenCount is redundant with len(raw)//4, and that redundancy is the
        # point: a disagreement means a truncated or rewritten _source, which is
        # otherwise invisible.
        raise TokenDecodeError(
            f"ceTokenCount {int(expected_count)} != {ids.shape[0]} decoded ids"
        )
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= vocab_size):
        raise TokenDecodeError(
            f"token id outside vocabulary [0, {vocab_size}): "
            f"min={int(ids.min())} max={int(ids.max())}"
        )
    return ids.astype(np.int64, copy=False)


def encode_query(tokenizer, query):
    """Query-side ids, WITHOUT special tokens.

    The model contract is `fold_de(raw_query)` with NO segment prefix — train
    cell D, `train_ce.build_query(row, "none", "fold_de")`, query contract
    `fold-de-v1-no-prefix` (MXG-177). The CALLER passes the already-folded text
    (`app._rerank` folds exactly once, so the decline check and the encoded text
    cannot drift); this function encodes it verbatim.

    The tokenizer must already have had `no_padding()` / `no_truncation()` called
    — the Rust backend is stateful and shared, and a stale padding config makes
    the model attend pad tokens with no error anywhere. `bench_ce_t4._reset_backend`
    and `CeTokenizer.load()` both guard the same trap.
    """
    return np.asarray(
        tokenizer.encode(query, add_special_tokens=False).ids,
        dtype=np.int64,
    )


def assemble(q_ids, arts, max_len, pad_to):
    """Splice cached article ids with the query and pad — the whole per-request
    CPU side. `pad_to=None` means pad to the batch's longest (what a real server
    does); an int means a fixed width.

    Truncation policy is only_second (trim the article). HF's default for pairs
    is longest_first; the two agree whenever the query is the shorter side,
    which the parity gate verifies exhaustively rather than assumes.

    ⚠️ COPIED VERBATIM from `pipeline/bench_ce_t4.py::assemble` (the research
    repo, /workspace). That function's `parity_gate` validated it at 3,000/3,000
    ids identical to HF's own pair encoding and max |logit delta| 0.00e+00, and
    every latency number this service is sized against was measured through it.
    Do not clean it up, do not sort `arts` by length, do not vectorise the loop:
    the frozen reference in `golden/splice_fixture.json` is the contract, and a
    "harmless" rewrite that changes one padded width changes the measured cost.
    """
    eff = max_len if pad_to is None else min(max_len, pad_to)
    # A pathologically long query at a short width would leave no room for the
    # article; clamp it so the batch stays well-formed (HF would trim the query
    # here too, since longest_first trims whichever side is longer).
    q_ids = q_ids[:max(1, eff - HEAD_EXTRA - 1)]
    nq = int(q_ids.shape[0])
    budget = max(1, eff - nq - HEAD_EXTRA)
    n = len(arts)
    lens = np.fromiter((min(a.shape[0], budget) for a in arts), dtype=np.int64,
                       count=n)
    seq = lens + nq + HEAD_EXTRA
    width = int(seq.max()) if pad_to is None else pad_to
    ids = np.full((n, width), PAD, dtype=np.int64)
    head = np.empty(nq + 3, dtype=np.int64)
    head[0] = BOS
    head[1:nq + 1] = q_ids
    head[nq + 1] = EOS
    head[nq + 2] = EOS
    h = head.shape[0]
    ids[:, :h] = head
    for i, (a, l) in enumerate(zip(arts, lens)):
        li = int(l)
        ids[i, h:h + li] = a[:li]
        ids[i, h + li] = EOS
    mask = (np.arange(width)[None, :] < seq[:, None]).astype(np.int64)
    return ids, mask, int(seq.max())


def clamp_max_len(requested, configured_max_len):
    """`max_len` is a free serve-time dial (quality is flat 128->256, and 128
    costs ~0.26pt for ~2.2x less latency), so it is on the wire — an A/B can
    trade window depth against width with no redeploy. Clamped upward so a
    caller cannot ask for 512 and blow the 150 ms budget."""
    if requested is None:
        return int(configured_max_len)
    value = int(requested)
    if value < MIN_MAX_LEN or value > int(configured_max_len):
        raise ValueError(
            f"max_len {value} outside [{MIN_MAX_LEN}, {int(configured_max_len)}] "
            "(CE_MAX_LEN)"
        )
    return value
