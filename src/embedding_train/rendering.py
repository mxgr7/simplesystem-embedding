import math

from embedding_train.text import (
    build_template,
    clean_html_text,
    flatten_category_paths,
    normalize_text,
)


DEFAULT_COLUMN_MAPPING = {
    "query_id": "query_id",
    "offer_id": "offer_id_b64",
    "query_term": "query_term",
    "name": "name",
    "manufacturer_name": "manufacturer_name",
    "manufacturer_article_number": "manufacturer_article_number",
    "manufacturer_article_type": "manufacturer_article_type",
    "article_number": "article_number",
    "ean": "ean",
    "category_paths": "category_paths",
    "description": "description",
    "label": "label",
}


def resolve_column_mapping(data_cfg):
    raw = data_cfg.get("column_mapping", None) if hasattr(data_cfg, "get") else None
    mapping = dict(DEFAULT_COLUMN_MAPPING)
    if raw is None:
        return mapping
    items = raw.items() if hasattr(raw, "items") else dict(raw).items()
    for canonical, source in items:
        source_name = str(source).strip()
        if source_name:
            mapping[str(canonical)] = source_name
    return mapping


class RowTextRenderer:
    def __init__(self, data_cfg):
        self.data_cfg = data_cfg
        self.column_mapping = resolve_column_mapping(data_cfg)
        self.query_template = build_template(data_cfg.query_template)
        self.offer_template = build_template(data_cfg.offer_template)
        self.column_rename = dict(data_cfg.get("column_rename", None) or {})

    def build_training_record(self, row):
        context = self.build_context(row)
        query_text = self.render_query_text(row, context=context)
        offer_text = self.render_offer_text(row, context=context)

        if not query_text or not offer_text:
            return None

        return {
            "query_id": normalize_text(context.get("query_id")),
            "offer_id": normalize_text(context.get("offer_id")),
            "query_text": query_text,
            "offer_text": offer_text,
            "label": (
                1.0 if context.get("label") == self.data_cfg.positive_label else 0.0
            ),
            "raw_label": normalize_text(context.get("label")),
        }

    def render_query_text(self, row, context=None):
        if context is None:
            context = self.build_context(row)

        return normalize_text(self.query_template.render(**context))

    def render_offer_text(self, row, context=None):
        if context is None:
            context = self.build_context(row)

        return normalize_text(self.offer_template.render(**context))

    def build_context(self, row):
        context = {}

        for key, value in row.items():
            context[self.column_rename.get(key, key)] = self._safe_value(value)

        for canonical, source in self.column_mapping.items():
            if source in row:
                context[canonical] = self._safe_value(row.get(source))
            elif canonical not in context:
                context[canonical] = ""

        context["query_term"] = normalize_text(context.get("query_term"))
        context["name"] = normalize_text(context.get("name"))
        context["manufacturer_name"] = normalize_text(context.get("manufacturer_name"))
        context["article_number"] = normalize_text(context.get("article_number"))
        context["category_text"] = flatten_category_paths(context.get("category_paths"))

        description = context.get("description")
        if self.data_cfg.clean_html:
            context["clean_description"] = clean_html_text(description)
        else:
            context["clean_description"] = normalize_text(description)

        return context

    @staticmethod
    def _safe_value(value):
        if value is None:
            return ""

        if isinstance(value, float) and math.isnan(value):
            return ""

        return value
