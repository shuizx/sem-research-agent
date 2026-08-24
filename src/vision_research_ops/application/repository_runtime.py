"""Runtime-only dependencies for the repository workflow Repository Agent graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from vision_research_ops.ports import (
    RepositoryPolicy,
    RepositoryProvider,
    StaticRepositoryAnalyzer,
)

from .runtime import ApprovalRecorder
from .services.paper_store import LocalResearchStore
from .services.repository_store import LocalRepositoryStore


def repository_utc_now() -> datetime:
    """Return an aware UTC timestamp for injected repository dependencies."""
    return datetime.now(UTC)


def _default_policy() -> RepositoryPolicy:
    return RepositoryPolicy(
        schema_version="1",
        policy_id="pipeline-pytorch-classification",
        policy_version="1",
    )


@dataclass(slots=True)
class RepositoryDependencies:
    """Injected research result, repository tools, local store, and human recorder."""

    research_store: LocalResearchStore
    research_workflow_id: str
    selected_paper_id: str
    repository_provider: RepositoryProvider
    static_analyzer: StaticRepositoryAnalyzer
    store: LocalRepositoryStore
    approval_recorder: ApprovalRecorder
    actor_id: str = "pipeline-user"
    policy: RepositoryPolicy = field(default_factory=_default_policy)
    clock: Callable[[], datetime] = repository_utc_now

    def __post_init__(self) -> None:
        for name in ("research_workflow_id", "selected_paper_id", "actor_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        if not callable(self.clock):
            raise TypeError("repository clock must be callable")


__all__ = ["RepositoryDependencies", "repository_utc_now"]
