"""Package health tests for the package baseline baseline.

These tests are pure unit tests and never require network, GPUs, external
services, credentials, or real data.
"""

from __future__ import annotations

import importlib
import sys

import pytest

import vision_research_ops
from vision_research_ops import health

pytestmark = pytest.mark.unit


def test_package_imports_cleanly() -> None:
    """Importing the top-level package must not perform side effects."""
    module = importlib.import_module("vision_research_ops")
    assert module is vision_research_ops


def test_version_is_populated() -> None:
    assert isinstance(vision_research_ops.__version__, str)
    assert vision_research_ops.__version__ != ""


def test_public_surface_is_consistent() -> None:
    """The declared public surface must actually exist."""
    expected = {"__version__", "health"}
    assert set(vision_research_ops.__all__) == expected
    for name in expected:
        assert hasattr(vision_research_ops, name)


def test_health_exposes_expected_functions() -> None:
    """The health module exposes its documented callable surface."""
    expected = {
        "CORE_DEPENDENCIES",
        "installed_core_dependencies",
        "core_dependencies_ok",
        "runtime_python",
        "summary",
    }
    assert set(health.__all__) == expected
    for name in expected - {"CORE_DEPENDENCIES"}:
        callable_ = getattr(health, name)
        assert callable(callable_)


def test_core_dependencies_are_installed() -> None:
    """The documented core runtime dependencies resolve to real versions."""
    expected = {"pydantic", "langgraph"}
    assert expected.issubset(set(health.CORE_DEPENDENCIES))
    installed = health.installed_core_dependencies()
    assert set(health.CORE_DEPENDENCIES).issubset(installed.keys())
    assert all(value != "(missing)" for value in installed.values())


def test_core_dependencies_ok_reports_true() -> None:
    assert health.core_dependencies_ok() is True


def test_core_dependencies_ok_detects_missing_package() -> None:
    """A missing core dependency must fail the check (fail closed)."""
    installed = dict(health.installed_core_dependencies())
    first_dependency = next(iter(installed))
    installed[first_dependency] = "(missing)"
    assert health.core_dependencies_ok(installed) is False


def test_runtime_python_matches_running_interpreter() -> None:
    expected = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert health.runtime_python() == expected
