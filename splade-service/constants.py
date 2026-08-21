import os

# Overridable so a second checkpoint can be served through the identical
# contract -- e.g. the query-L1 candidate (`soup2b50`) for the retrieval-latency
# A/B. Both sides must declare the SAME pair: the backend checks the file against
# MODEL_SHA256 and the reindex client checks the backend's /metadata against its
# own model_metadata(), so overriding one side alone fails loudly instead of
# silently indexing one model's vectors under another model's name.
MODEL_ID = os.environ.get("SPLADE_MODEL_ID", "prod-soup-folde-top256-v1")
MODEL_SHA256 = os.environ.get(
    "SPLADE_MODEL_SHA256",
    "5a7890a01deeabfc9c8a970d7abfffe9017799ee5a8431030190ef905b172437",
)
MODEL_NAME = "deepset/gbert-base"
PROTOCOL_VERSION = 1
VOCAB_SIZE = 31_102
TOP_K = 256
MAX_OFFER_LENGTH = 256
# Cached vectors are keyed by MODEL_ID *and* encoding version. The model id alone
# is not enough: the encoding version carries the compute dtype and the fast-path
# flags, and two backends can serve the same checkpoint under different ones. A
# bf16 burst backend registered alongside an fp16 backend passes every check in
# `model_metadata()` -- it is the same model -- and then writes its vectors over
# the other's under an identical key, with nothing downstream able to tell which
# encoder produced a given cached entry.
#
# Left empty this degrades to the old model-only prefix, which is correct for a
# single-encoder deployment and is what the pre-MXG-111 keyspace already holds.
# Set it to the backend's `document_encoding_version`; `BackendPool` asserts every
# backend agrees with it, so a mismatch is a startup error rather than a silent
# poisoning.
ENCODING_VERSION = os.environ.get("SPLADE_ENCODING_VERSION", "")
CACHE_PREFIX = (
    f"splade:{MODEL_ID}:{ENCODING_VERSION}:" if ENCODING_VERSION
    else f"splade:{MODEL_ID}:"
)

# Extends the dense service's first eight fields with the six kitchen-sink
# fields used by the production SPLADE checkpoint.
FIELD_ORDER = (
    "name",
    "manufacturer_name",
    "description",
    "category_paths",
    "ean",
    "article_number",
    "manufacturer_article_number",
    "manufacturer_article_type",
    "customer_artnos_text",
    "vendor_text",
    "category_leaf_text",
    "s2class_text",
    "keywords_text",
    "features_text",
)


def model_metadata():
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "model_name": MODEL_NAME,
        "vocab_size": VOCAB_SIZE,
        "top_k": TOP_K,
        "max_length": MAX_OFFER_LENGTH,
        "precision": "float32",
    }
