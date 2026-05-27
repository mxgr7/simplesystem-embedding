"""NUL-separated 8-field input → rendered offer text.

The wire contract is: each input string is exactly 8 NUL-separated field
values in this fixed order:

    name \x00 manufacturerName \x00 description \x00 categoryPaths
         \x00 ean \x00 article_number \x00 manufacturerArticleNumber
         \x00 manufacturerArticleType

`render_from_nul(text)` splits, builds a row dict with snake_case keys
(matching what `row_renderer.RowTextRenderer` expects), and renders the
offer template. Used only on cache misses — hits skip rendering entirely.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import yaml

from row_renderer import RowTextRenderer


# Positional field order on the wire. Matches the user-stated contract.
# Snake-case keys here line up with the template's variable names so we
# don't need a column_rename map at runtime.
_FIELD_ORDER = (
    "name",
    "manufacturer_name",
    "description",
    "category_paths",
    "ean",
    "article_number",
    "manufacturer_article_number",
    "manufacturer_article_type",
)
N_FIELDS = len(_FIELD_ORDER)

_TEMPLATE_YAML = Path(__file__).resolve().parent / "template.yaml"


class _CfgWrapper:
    """Minimal attribute + .get adapter so RowTextRenderer accepts a plain
    dict instead of OmegaConf. Mirrors the indexer-side helper."""

    def __init__(self, d: dict[str, Any]) -> None:
        self._d = d

    def __getattr__(self, name: str) -> Any:
        if name in self._d:
            return self._d[name]
        raise AttributeError(name)

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)


_renderer: RowTextRenderer | None = None
_renderer_lock = Lock()


def _get_renderer() -> RowTextRenderer:
    global _renderer
    if _renderer is None:
        with _renderer_lock:
            if _renderer is None:
                with open(_TEMPLATE_YAML) as f:
                    cfg = yaml.safe_load(f)
                _renderer = RowTextRenderer(_CfgWrapper(cfg))
    return _renderer


def split_fields(text: str) -> list[str]:
    """Return the 8 fields parsed from a NUL-separated input string.

    Raises `ValueError` if the count is not exactly 8 — the handler maps
    this to a 400 at the API edge."""
    fields = text.split("\x00")
    if len(fields) != N_FIELDS:
        raise ValueError(
            f"expected exactly {N_FIELDS} NUL-separated fields, got {len(fields)}"
        )
    return fields


def fields_to_row(fields: list[str]) -> dict[str, str]:
    return dict(zip(_FIELD_ORDER, fields))


def render_from_nul(text: str) -> str:
    """One-call helper: split + render. Use this in the handler on misses."""
    return _get_renderer().render_offer_text(fields_to_row(split_fields(text)))
