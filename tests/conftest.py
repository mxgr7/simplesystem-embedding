"""Pytest session fixtures.

The ACL service was renamed `acl/main.py` → `acl/app.py` so it no
longer competes for the bare `main` module slot in `sys.modules`.
Search-api's `search-api/main.py` is now the only `main` module the
test sweep imports, and ACL tests use the absolute `from acl.app
import app` form. No cross-file eviction is needed.

`embed_client` is a second colliding name, and this one still bites:
`search-api/` and `embedding-service/` both define one, the flat-import
convention gives them the same slot, and alphabetical collection order
means `tests/test_embed_client.py` (search-api's) claims it first. Every
later embedding-service test that reaches `main`'s `from embed_client
import ...` then dies with `ImportError: cannot import name 'TEIPool'`,
which is invisible when you run one file at a time and loud only in the
full sweep. Embedding-service tests therefore load their module by path
with `load_service_module` instead of trusting `sys.path` order.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_service_module(name: str, path: Path):
    """Import `path` as `name`, overwriting any same-named module.

    Registered in `sys.modules` before execution so that flat imports
    inside the loaded module (and later `from <name> import ...` in the
    service's own modules) resolve to this one."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
