import re

from jinja2 import Environment, StrictUndefined


_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_SPACE_NL_RE = re.compile(r" *\n *")


def build_template(template_string):
    environment = Environment(
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    return environment.from_string(template_string)


def normalize_text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.replace("\x00", " ").replace("\xa0", " ")
    value = value.replace("\r\n", "\n").strip()
    value = _SPACE_RE.sub(" ", value)
    value = _SPACE_NL_RE.sub("\n", value)
    value = _MULTI_NL_RE.sub("\n\n", value)
    return value.strip()
