"""Article-level embedding text rendering for the F9 bulk indexer.

The F9 spec pins the per-article embedded-field set (`name`,
`manufacturerName`, `category_l1..l5`, `eclass5_code`, `eclass7_code`,
`s2class_code`) — same fields that feed `compute_article_hash`. The
indexer must serialise those into a deterministic text string before
sending to TEI; two articles with identical fields must produce
identical text (and therefore identical embeddings, hash-keyed in
Redis).

The format below mirrors the structure of the production training
template (`configs/data/default.yaml#offer_template`) — `passage:`
prefix + labelled segments — but restricted to the article-level field
set per F9. Per-offer fields the production template includes (`ean`,
`article_number`, `manufacturerArticleNumber`, etc.) are intentionally
omitted; carrying them here would make the embedding offer-scoped
rather than article-scoped, breaking the F9 dedup guarantee.

Bumping the format breaks every cached embedding — pair the change with
a `HASH_VERSION` bump (`indexer/projection.py`) so the cache key prefix
shifts and Redis returns clean misses.
"""

from __future__ import annotations

from typing import Any

# Path separator used in the projected `category_l*` array entries
# (legacy `commons/.../CategoryPath.java`). Each entry is already a full
# encoded path like `Tools¦Hand¦Hammers`; multiple entries per level
# come from offers tagged in multiple sibling categories.
_PATH_SEPARATOR = "¦"

# Within-text separator between distinct paths at the same level / between
# eclass codes. Single space matches the production template's segment
# separator (the template uses Jinja whitespace + `{% if %}` guards).
_INTRA_FIELD_SEP = " "


def _join_categories(article: dict[str, Any]) -> str:
    """Concat every distinct path across `category_l1..l5` into a single
    space-separated string. Each entry is already path-encoded
    (`a¦b¦c`); we don't re-encode. Levels are emitted in depth order so
    that a model trained on "root → leaf" signal sees the more general
    paths first."""
    pieces: list[str] = []
    for d in range(1, 6):
        for entry in article.get(f"category_l{d}") or []:
            if entry:
                pieces.append(entry)
    return _INTRA_FIELD_SEP.join(pieces)


def _join_eclass(values: list[int] | None) -> str:
    if not values:
        return ""
    return _INTRA_FIELD_SEP.join(str(v) for v in sorted(values))


def article_to_text(article: dict[str, Any]) -> str:
    """Render an `articles_v{N}` row into the deterministic embedding
    text. Empty fields are skipped (no empty `Brand:` segment when
    `manufacturerName` is missing) so the text length scales with the
    actual signal carried per article."""
    name = (article.get("name") or "").strip()
    mfg = (article.get("manufacturerName") or "").strip()
    cat = _join_categories(article)
    ec5 = _join_eclass(article.get("eclass5_code"))
    ec7 = _join_eclass(article.get("eclass7_code"))
    s2  = _join_eclass(article.get("s2class_code"))

    # Match the production template's `passage:` prefix so the model
    # sees the same instruction-style cue at inference time as during
    # training. Subsequent segments use the same `Label: value` shape.
    parts = ["passage: Article Name: " + name] if name else ["passage:"]
    if mfg:
        parts.append(f"Brand: {mfg}")
    if cat:
        parts.append(f"Category: {cat}")
    if ec5:
        parts.append(f"eClass5: {ec5}")
    if ec7:
        parts.append(f"eClass7: {ec7}")
    if s2:
        parts.append(f"S2Class: {s2}")
    return _INTRA_FIELD_SEP.join(parts)


__all__ = ["article_to_text"]
