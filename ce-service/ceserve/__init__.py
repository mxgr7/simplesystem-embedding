"""Cross-encoder fine-ranker serving runtime (MXG-144).

A PACKAGE, not the flat `uvicorn main:app` layout `splade-service` and
`embedding-service` use. Those two collide on the bare `constants` / `config` /
`main` module slots the moment more than one of them is imported in a single
process -- `tests/conftest.py` records the ACL service being renamed
`main.py` -> `app.py` for exactly that reason. A namespaced package costs one
`ceserve.` prefix and makes the collision impossible, in the test session and
in any future co-located tooling.

Entry point: `uvicorn ceserve.app:app`.
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = PACKAGE_DIR / "golden"
SPLICE_FIXTURE = GOLDEN_DIR / "splice_fixture.json"
