"""Offer-only row renderer.

Trimmed copy of `src/embedding_train/rendering.py` — drops the query
template, label handling, and `build_training_record` (we only render
offers at serve time). `RowTextRenderer.render_offer_text` takes a row
dict and returns the rendered template string, with the same
column-rename + column-mapping + preprocessing pipeline the training
code uses.
"""

from __future__ import annotations

import math

from text import (
    build_template,
    clean_html_text,
    flatten_category_paths,
    normalize_text,
)


DEFAULT_COLUMN_MAPPING = {
    "name": "name",
    "manufacturer_name": "manufacturer_name",
    "manufacturer_article_number": "manufacturer_article_number",
    "manufacturer_article_type": "manufacturer_article_type",
    "article_number": "article_number",
    "ean": "ean",
    "category_paths": "category_paths",
    "description": "description",
}


def _resolve_column_mapping(data_cfg):
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
        self.column_mapping = _resolve_column_mapping(data_cfg)
        self.offer_template = build_template(data_cfg.offer_template)
        self.column_rename = dict(data_cfg.get("column_rename", None) or {})

    def render_offer_text(self, row, context=None):
        if context is None:
            context = self._build_context(row)
        return normalize_text(self.offer_template.render(**context))

    def _build_context(self, row):
        context = {}

        for key, value in row.items():
            context[self.column_rename.get(key, key)] = self._safe_value(value)

        for canonical, source in self.column_mapping.items():
            if source in row:
                context[canonical] = self._safe_value(row.get(source))
            elif canonical not in context:
                context[canonical] = ""

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
