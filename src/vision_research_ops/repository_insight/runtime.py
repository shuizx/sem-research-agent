"""Build live or deterministic dependencies for the repository-insight workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from vision_research_ops.adapters.llm import build_dashscope_repository_insight_planner
from vision_research_ops.adapters.repositories import (
    BoundedZipSourceReader,
    GitHubRepositoryProvider,
    ZipStaticRepositoryAnalyzer,
)
from vision_research_ops.application.repository_insight_runtime import (
    RepositoryInsightDependencies,
)
from vision_research_ops.application.runtime import InMemoryApprovalRecorder
from vision_research_ops.application.services.repository_insight_store import (
    LocalRepositoryInsightStore,
)
from vision_research_ops.application.services.target_profile import (
    SafeTargetProfile,
    load_target_profile,
)
from vision_research_ops.settings import Settings

from .fixture_planner import FixtureRepositoryInsightPlanner
from .fixture_repository import FixtureGitHubInsightTransport

RepositoryInsightMode = Literal["fixture", "live"]


def build_repository_insight_dependencies(
    *,
    workspace: Path,
    mode: RepositoryInsightMode,
    settings: Settings | None,
    fixture_transport: FixtureGitHubInsightTransport | None = None,
    before_model_call: Callable[[], None] | None = None,
    target_profile: SafeTargetProfile | None = None,
) -> RepositoryInsightDependencies:
    """Compose existing GitHub/static adapters with one four-tool code-reading planner."""
    if settings is not None and settings.dataset_profile_path is not None:
        target_profile = load_target_profile(settings.dataset_profile_path)
    if mode == "live":
        if settings is None:
            raise ValueError("live repository insight requires injected Settings")
        provider = GitHubRepositoryProvider(artifact_root=workspace)
        planner = build_dashscope_repository_insight_planner(
            settings, before_model_call=before_model_call, target_profile=target_profile
        )
    else:
        provider = GitHubRepositoryProvider(
            artifact_root=workspace,
            transport=fixture_transport or FixtureGitHubInsightTransport(),
        )
        planner = FixtureRepositoryInsightPlanner(target_profile=target_profile)
    reader = BoundedZipSourceReader(artifact_root=workspace)
    return RepositoryInsightDependencies(
        repository_provider=provider,
        static_analyzer=ZipStaticRepositoryAnalyzer(artifact_root=workspace),
        source_reader=reader,
        planner=planner,
        store=LocalRepositoryInsightStore(workspace),
        approval_recorder=InMemoryApprovalRecorder(),
    )


__all__ = ["RepositoryInsightMode", "build_repository_insight_dependencies"]
