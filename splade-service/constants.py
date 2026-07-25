MODEL_ID = "prod-soup-folde-top256-v1"
MODEL_SHA256 = "5a7890a01deeabfc9c8a970d7abfffe9017799ee5a8431030190ef905b172437"
MODEL_NAME = "deepset/gbert-base"
PROTOCOL_VERSION = 1
VOCAB_SIZE = 31_102
TOP_K = 256
MAX_OFFER_LENGTH = 256
CACHE_PREFIX = f"splade:{MODEL_ID}:"

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
