"""Shared pytest configuration for the SEM Research Agent test suite.

Markers are declared authoritatively in ``pyproject.toml`` under
``[tool.pytest.ini_options]``; re-registering them here guarantees they are
also recognised when pytest is invoked from a different rootdir. pytest 9
rejects unknown markers by default.
"""

from __future__ import annotations

import pytest

_MARKERS = {
    "unit": "Pure unit tests; no external services (default).",
    "contract": "Port/adapter contract tests shared across implementations.",
    "graph": "LangGraph workflow graph tests with in-memory checkpointer.",
    "integration": "Controlled integration tests using local fakes and fixture servers.",
    "e2e": "Offline end-to-end acceptance tests.",
    "security": "Security and sandbox policy tests.",
    "external": "Require live external API or public network access; skipped by default.",
    "gpu": "Require a CUDA-capable GPU; skipped by default.",
    "slurm": "Require a Slurm cluster; skipped by default.",
}


def pytest_configure(config: pytest.Config) -> None:
    """Register and document all known test markers."""
    for name, description in _MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")
