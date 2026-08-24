"""Offline environment and core-dependency health checks.

Importing this module performs no I/O, network access, environment lookups,
or global mutation. All checks run only when the public functions are called.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version
from typing import Final

CORE_DEPENDENCIES: Final[tuple[str, ...]] = ("pydantic", "langgraph")
_MISSING: Final[str] = "(missing)"

__all__ = [
    "CORE_DEPENDENCIES",
    "core_dependencies_ok",
    "installed_core_dependencies",
    "runtime_python",
    "summary",
]


def installed_core_dependencies() -> dict[str, str]:
    """Map each core runtime dependency name to its installed version.

    Dependencies that cannot be resolved from installed metadata map to the
    ``_MISSING`` sentinel string.
    """
    resolved: dict[str, str] = {}
    for name in CORE_DEPENDENCIES:
        try:
            resolved[name] = _metadata_version(name)
        except PackageNotFoundError:
            resolved[name] = _MISSING
    return resolved


def runtime_python() -> str:
    """Return the ``major.minor`` runtime Python version (e.g. ``3.12``)."""
    version_info = sys.version_info
    return f"{version_info.major}.{version_info.minor}"


def _importable(name: str) -> bool:
    """Return whether ``name`` can be imported without raising."""
    try:
        __import__(name)
    except Exception:
        return False
    return True


def core_dependencies_ok(result: Mapping[str, str] | None = None) -> bool:
    """Report whether every core dependency is installed and importable.

    ``result`` may be supplied as a resolved dependency mapping to avoid a
    redundant lookup and to ease unit testing of failure paths.
    """
    resolved = installed_core_dependencies() if result is None else result
    return all(value != _MISSING and _importable(name) for name, value in resolved.items())


def summary() -> dict[str, object]:
    """Return a small, JSON-serialisable health snapshot."""
    installed = installed_core_dependencies()
    return {
        "python": runtime_python(),
        "core_dependencies": installed,
        "core_dependencies_ok": core_dependencies_ok(installed),
    }
