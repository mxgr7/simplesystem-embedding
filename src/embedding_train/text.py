import html
import re

from jinja2 import Environment


_BR_RE = re.compile(r"(?i)<br\\s*/?>")
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def build_template(template_string):
    environment = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
    return environment.from_string(template_string)


def normalize_text(value):
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    value = value.replace("\x00", " ")
    value = value.replace("\xa0", " ")
    value = value.replace("\r\n", "\n")
    value = value.strip()
    value = _SPACE_RE.sub(" ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = _MULTI_NL_RE.sub("\n\n", value)
    return value.strip()


def clean_html_text(value):
    value = normalize_text(value)
    if not value:
        return ""

    value = _BR_RE.sub("\n", value)
    value = _TAG_RE.sub(" ", value)
    value = html.unescape(value)
    return normalize_text(value)


def flatten_category_paths(value):
    collected = []

    def visit(item):
        if item is None:
            return

        if hasattr(item, "tolist") and not isinstance(item, str):
            visit(item.tolist())
            return

        if isinstance(item, dict):
            if "elements" in item:
                visit(item["elements"])
                return

            for nested in item.values():
                visit(nested)
            return

        if isinstance(item, (list, tuple)):
            if item and all(isinstance(part, str) for part in item):
                path = " > ".join(
                    normalize_text(part) for part in item if normalize_text(part)
                )
                if path:
                    collected.append(path)
                return

            for nested in item:
                visit(nested)
            return

        text = normalize_text(item)
        if text:
            collected.append(text)

    visit(value)

    unique_paths = []
    seen = set()
    for path in collected:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    return " | ".join(unique_paths)
