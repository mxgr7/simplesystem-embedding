"""Helpers shared by tests for the repository's flat-module services.

``embedding-service`` and ``splade-service`` are deployments rather than Python
packages.  Their production modules therefore use bare imports such as
``import cache`` and ``from constants import ...``.  Loading both deployments in
one pytest process must not let whichever suite collected first own those names.
"""

import importlib
import sys
from pathlib import Path


_FLAT_SERVICES = {}


class FlatServiceModules:
    """A cached, isolated module closure for one flat-module service."""

    def __init__(self, alias, service_dir):
        self.alias = alias
        self.service_dir = service_dir.resolve()
        self._modules = {}
        self._local_names = {
            path.stem
            for path in self.service_dir.rglob("*.py")
            if path.name != "__init__.py"
        }

    def __getattr__(self, name):
        try:
            return self._modules[name]
        except KeyError:
            raise AttributeError(name) from None

    def _belongs_to_service(self, module):
        filename = getattr(module, "__file__", None)
        if not filename:
            return False
        try:
            Path(filename).resolve().relative_to(self.service_dir)
        except (OSError, ValueError):
            return False
        return True

    def _enter(self):
        previous_modules = sys.modules.copy()
        previous_path = sys.path[:]

        # Remove every possible bare local name, not just the requested entry
        # points.  This makes transitive imports bind to this service's closure.
        for name in self._local_names:
            sys.modules.pop(name, None)
        sys.modules.update(self._modules)
        sys.path.insert(0, str(self.service_dir))
        return previous_modules, previous_path

    def _leave(self, previous_modules, previous_path):
        service_names = set()
        for name, module in tuple(sys.modules.items()):
            if self._belongs_to_service(module):
                self._modules[name] = module
                service_names.add(name)

        # Restore every local-name slot that service imports could have changed.
        # Leave unrelated third-party imports cached: removing native extension
        # modules from sys.modules can unload process-global state underneath
        # libraries such as torch.
        for name in self._local_names | service_names:
            if name in previous_modules:
                sys.modules[name] = previous_modules[name]
            else:
                sys.modules.pop(name, None)
        sys.path[:] = previous_path

    def load(self, *names):
        previous_modules, previous_path = self._enter()
        try:
            for name in names:
                importlib.import_module(name)
        finally:
            self._leave(previous_modules, previous_path)
        return self

    def reload(self, name):
        """Reload one captured module while its flat dependency closure is active."""
        if name not in self._modules:
            self.load(name)
        previous_modules, previous_path = self._enter()
        try:
            module = importlib.reload(self._modules[name])
        finally:
            self._leave(previous_modules, previous_path)
        return module


def load_flat_service(alias, service_dir, *module_names):
    """Load and cache a flat service closure under a stable test-only alias.

    Modules execute with their production bare-import convention.  The complete
    imported closure is captured, while the caller's ``sys.modules`` and
    ``sys.path`` are restored exactly before this function returns.
    """
    service_dir = Path(service_dir).resolve()
    service = _FLAT_SERVICES.get(alias)
    if service is None:
        service = FlatServiceModules(alias, service_dir)
        _FLAT_SERVICES[alias] = service
    elif service.service_dir != service_dir:
        raise ValueError(f"flat service alias {alias!r} already names {service.service_dir}")
    return service.load(*module_names)
