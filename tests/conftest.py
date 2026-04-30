"""Pytest session fixtures.

The ACL service was renamed `acl/main.py` → `acl/app.py` so it no
longer competes for the bare `main` module slot in `sys.modules`.
Search-api's `search-api/main.py` is now the only `main` module the
test sweep imports, and ACL tests use the absolute `from acl.app
import app` form. No cross-file eviction is needed.
"""
