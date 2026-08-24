"""Deterministic dependencies for repository insight workflow graph and ToolNode tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vision_research_ops.application.repository_insight_runtime import (
    RepositoryInsightDependencies,
    RepositoryInsightState,
    create_repository_insight_state,
)
from vision_research_ops.repository_insight.fixture_repository import (
    FixtureGitHubInsightTransport,
)
from vision_research_ops.repository_insight.runtime import (
    build_repository_insight_dependencies,
)

FIXED_NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


@dataclass(slots=True)
class RepositoryInsightHarness:
    dependencies: RepositoryInsightDependencies
    transport: FixtureGitHubInsightTransport
    workspace: Path


@pytest.fixture
def make_repository_insight_harness(
    tmp_path: Path,
) -> Callable[..., RepositoryInsightHarness]:
    def factory(*, root: Path | None = None) -> RepositoryInsightHarness:
        workspace = root or (tmp_path / "conversation")
        transport = FixtureGitHubInsightTransport()
        dependencies = build_repository_insight_dependencies(
            workspace=workspace,
            mode="fixture",
            settings=None,
            fixture_transport=transport,
        )
        dependencies.clock = lambda: FIXED_NOW
        return RepositoryInsightHarness(
            dependencies=dependencies,
            transport=transport,
            workspace=workspace,
        )

    return factory


@pytest.fixture
def repository_insight_state() -> RepositoryInsightState:
    return create_repository_insight_state(
        {
            "schema_version": "1",
            "workflow_id": "workflow-repository-insight-1",
            "thread_id": "thread-repository-insight-1",
            "repository_url": "https://github.com/example/sem-classifier",
        }
    )
