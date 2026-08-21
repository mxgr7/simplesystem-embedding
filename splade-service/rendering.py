from constants import FIELD_ORDER
from fold_de import fold_de
from text import build_template, normalize_text


N_FIELDS = len(FIELD_ORDER)
DESCRIPTION_INDEX = FIELD_ORDER.index("description")

_TEMPLATE = build_template("""Article Name: {{ name }}
{% if ean %} EAN: {{ ean }}{% endif %}
{% if article_number %} Article Number: {{ article_number }}{% endif %}
{% if manufacturer_article_number %} Article Number (Manufacturer): {{ manufacturer_article_number }}{% endif %}
{% if customer_artnos_text %} Customer Article Numbers: {{ customer_artnos_text }}{% endif %}
{% if manufacturer_name %} Brand: {{ manufacturer_name }}{% endif %}
{% if vendor_text %} Vendor: {{ vendor_text }}{% endif %}
{% if manufacturer_article_type %} Article Type: {{ manufacturer_article_type }}{% endif %}
{% if category_leaf_text %} Category: {{ category_leaf_text }}{% endif %}
{% if s2class_text %} Classification: {{ s2class_text }}{% endif %}
{% if keywords_text %} Keywords: {{ keywords_text }}{% endif %}
{% if features_text %} Features: {{ features_text }}{% endif %}
{% if description %} Description: {{ description }}{% endif %}""")


def split_fields(value):
    fields = value.split("\x00")
    if len(fields) != N_FIELDS:
        raise ValueError(
            f"expected exactly {N_FIELDS} NUL-separated fields, got {len(fields)}"
        )
    return fields


def canonical_input(value):
    fields = split_fields(value)
    # This model is locked description-free. Canonicalising the ignored field
    # avoids duplicate cache entries when callers accidentally send a value.
    fields[DESCRIPTION_INDEX] = ""
    return "\x00".join(normalize_text(field) for field in fields)


def render_from_nul(value):
    fields = split_fields(canonical_input(value))
    # Training folded source fields before applying the template. Keeping the
    # static labels cased is byte-identical to the training input distribution.
    row = {
        name: fold_de(normalize_text(field))
        for name, field in zip(FIELD_ORDER, fields)
    }
    return normalize_text(_TEMPLATE.render(**row))
