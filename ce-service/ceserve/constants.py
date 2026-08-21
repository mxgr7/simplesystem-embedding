"""The cross-encoder serving contract, in one file.

Everything here is env-overridable for the same reason `splade-service/constants.py`
is: a second checkpoint has to be servable through the identical contract. Both
sides must declare the SAME values -- the service checks the files on disk against
the digests, and the query service checks the service's /metadata against its own
expectation -- so overriding one side alone fails loudly instead of silently
scoring one model's pairs under another model's name.

MXG-144.
"""
import os

import numpy as np

# ---------------------------------------------------------------- the model --

# The MXG-177 ship decision (2026-08-20): the cell-D negation-tuned student —
# distilled from the fold_de/no-prefix teacher and finished with the MXG-66
# negation overlay (the bare student fails the controlled negation gate, the
# overlay passes 0.912/0.960). Max explicitly overrode the teacher screen's
# predeclared no-change verdict; the override is recorded in
# `pipeline/out/mxg177_stage2/run_manifest.json::approved_contract`. Same
# checkpoint `pipeline/ce_serve_skew.py::CE_MODEL` pins for the offline stack.
MODEL_ID = os.environ.get("CE_MODEL_ID", "d_mxg177_d_mxg66_s66-2026-08-20")
MODEL_SHA256 = os.environ.get(
    "CE_MODEL_SHA256",
    "85a2cfdc088d6f779abf4a5ecc36c9fbafe13539df78a5710c4aeb96750c5036",
)

# ⚠️ TOKENIZER_SHA256 and TOKENIZER_VERSION pin two DIFFERENT things and both
# matter. The digest pins the vocabulary file; the version string pins the
# article *rendering* the stored ids were produced from. The `-fkr108` suffix is
# why they are separate: at MXG-108 the vocabulary did not change, the rendering
# did (bool-flag entries retained), and every stored blob had to be recomputed.
# `CeProperties`' KDoc in /next-gen states the same split.
#
# This digest is the one the indexer asserts (`s2ng-article-search.ce.tokenizer-sha256`).
# Verified 2026-08-18: `d_mxg84_new108t_mxg66_s68/tokenizer.json` is byte-identical
# to `ce_dist_l12_v3-2026-07-18/tokenizer.json`, so the 115,485,224 stored token
# blobs are valid for this student and no re-tokenization pass is needed.
# Re-verified 2026-08-21 for MXG-177: `d_mxg177_d_mxg66_s66/tokenizer.json`
# carries the same digest, so the stored blobs stay valid across this cutover too.
TOKENIZER_SHA256 = os.environ.get(
    "CE_TOKENIZER_SHA256",
    "3a56def25aa40facc030ea8b0b87f3688e4b3c39eb8b45d5702b3a1300fe2a20",
)
TOKENIZER_VERSION = os.environ.get(
    "CE_TOKENIZER_VERSION", "ce_dist_l12_v3-2026-07-18-fkr108"
)

BACKBONE = "XLMRobertaForSequenceClassification"
VOCAB_SIZE = 250_002
NUM_LABELS = 4
LABELS = ("E", "S", "C", "I")

# `train_ce.py::GAINS`. ce_score = sum(softmax(logits) * GAINS) / 4 -- the number
# the offline stack ranks by, so serving has to produce the same one or the
# agreement check in tests/test_ce_score_agreement.py is comparing two different
# quantities and cannot fail usefully.
GAINS = np.array([4.0, 2.0, 1.0, 0.0], dtype=np.float64)

# The rendering profile the article ids were produced from: everything the ES
# `_source` carries. `build_cheap_features.text_es`.
PROFILE = "text_es"

# --------------------------------------------------------------- the splice --

# XLM-R specials. Pair encoding is <s> A </s></s> B </s>.
BOS, EOS, PAD = 0, 2, 1
HEAD_EXTRA = 4  # bos + eos + eos + eos around a (query, article) pair

# Below 6 the clamp chain in `assemble` (`q_ids[:max(1, eff - HEAD_EXTRA - 1)]`
# then `budget = max(1, eff - nq - HEAD_EXTRA)`) can produce a row wider than
# `max_len`. 8 is the smallest width for which the clamps are provably safe, and
# it is also the floor applied to a per-request `max_len` override.
MIN_MAX_LEN = 8

# Bumped only when `splice.assemble` changes. It rides in `serving_contract`,
# so a splice change invalidates any cached window that names the old contract.
SPLICE_VERSION = "splice_v1"

# How the query side of the pair is built from the wire: the service applies
# `fold_de` to the untouched raw query and adds NO segment prefix (train cell D,
# `train_ce.build_query(row, "none", "fold_de")`). Deliberately NOT
# env-overridable, unlike everything else in this file: the identifier names
# what the CODE does, and an env override would let a deployment claim a
# contract the code cannot honor. A caller that needs a different query
# contract needs a different build. MXG-177.
QUERY_CONTRACT = "fold-de-v1-no-prefix"

PROTOCOL_VERSION = 1


def serving_contract(dtype, max_len):
    """A single string naming HOW a score was produced, not just by which model.

    `model_sha256` cannot express this: the same checkpoint served at fp16/L192
    and at fp32/L256 produces different numbers, and MXG-111 found exactly that
    hole in the SPLADE keyspace (a bf16 burst backend writing over an fp16
    backend's vectors under identical cache keys). Anything that caches or
    compares CE scores must key on this, not on the model id.
    """
    return f"ce-{MODEL_ID}-{PROFILE}-L{int(max_len)}-{dtype}-{SPLICE_VERSION}"


def model_metadata(dtype="fp16", max_len=192):
    """What this service believes it serves. Shared by /metadata and the
    startup assertions so the two cannot drift."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_version": TOKENIZER_VERSION,
        "backbone": BACKBONE,
        "num_labels": NUM_LABELS,
        "labels": list(LABELS),
        "gains": GAINS.tolist(),
        "score_formula": "sum(softmax(logits) * gains) / 4",
        "profile": PROFILE,
        "max_len": int(max_len),
        "min_max_len": MIN_MAX_LEN,
        "vocab_size": VOCAB_SIZE,
        "bos": BOS,
        "eos": EOS,
        "pad": PAD,
        "head_extra": HEAD_EXTRA,
        "splice_version": SPLICE_VERSION,
        "query_contract": QUERY_CONTRACT,
        "serving_contract": serving_contract(dtype, max_len),
    }
