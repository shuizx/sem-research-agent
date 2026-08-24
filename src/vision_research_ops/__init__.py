"""SEM Research Agent package.

A LangGraph-based workflow system for SEM research, code adaptation, and
deterministic evaluation (see ``docs/`` for the design).

This module exposes only a minimal public surface. Importing it must not
perform I/O, network access, environment reads, or any other side effect.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

from vision_research_ops import health

__all__ = ["__version__", "health"]


def _package_version() -> str:
    """Return the installed package version, falling back to a constant."""
    try:
        return _metadata_version("sem-research-agent")
    except PackageNotFoundError:
        return "0.1.0"


__version__ = _package_version()
