"""Pytest session-level fixtures + sys.modules hygiene.

Resolves a long-standing test-suite issue: both `acl/main.py` and
`search-api/main.py` are named `main`, and Python's `sys.modules`
caches the first one loaded. Tests that do `import main as mod`
followed by `importlib.reload(mod)` then reload the WRONG module
when the cached entry is from the other service.

This conftest evicts `main` from `sys.modules` between test files
so each file's first `import main` gets a fresh resolution against
the current `sys.path`. Without this, the full pytest sweep fails
on test_search_dedup_integration (~26 tests) because the prior
test file's import cached the ACL main.

The cleaner long-term fix is to rename one of the `main.py` files
(e.g. to `app.py`) — but that's a wider refactor. Until then,
this hook keeps the cross-file ordering safe.
"""

from __future__ import annotations

import sys

import pytest


# Only `main` is the ambiguous one that needs eviction — both
# `acl/main.py` and `search-api/main.py` share the bare module name.
# Other modules (models, tracing, metrics, …) ALSO have name collisions
# but tests use absolute imports (`from acl.tracing import ...`,
# `from milvus_helpers import ...` after the appropriate `sys.path.insert`)
# and don't reload them. Evicting those breaks Pydantic class identity
# (test fixtures hold references to the cached module's classes).
_AMBIGUOUS_MODULE_NAMES = ("main",)


@pytest.fixture(autouse=True)
def _evict_ambiguous_modules_between_tests():
    """Auto-fixture: evict `main` and friends from `sys.modules` after
    each test so the next test's `import main` re-resolves against the
    current `sys.path` instead of returning the cached wrong-service
    module. Cheap (a few dict pops); only matters for the test files
    that import these names — others don't trigger eviction at all."""
    yield
    for name in _AMBIGUOUS_MODULE_NAMES:
        sys.modules.pop(name, None)


def pytest_collection_modifyitems(config, items):
    """Evict ambiguous modules before pytest starts running tests, so
    the FIRST test in each file resolves `import main` against the
    correct path. Without this, top-level `from main import app` in
    one test file (executed at collection time) caches that file's
    `main` for the entire pytest session — autouse fixtures fire too
    late."""
    for name in _AMBIGUOUS_MODULE_NAMES:
        sys.modules.pop(name, None)
