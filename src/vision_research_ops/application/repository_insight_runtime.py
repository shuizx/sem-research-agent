"""Runtime dependencies and small state for the repository insight workflow outer LangGraph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol, TypedDict

from pydantic import BaseModel, ConfigDict, field_validator

from vision_research_ops.adapters.repositories import BoundedZipSourceReader
from vision_research_ops.domain import ArtifactRef
from vision_research_ops.ports import (
    OperationContext,
    RepositoryMetadata,
    RepositoryPolicy,
    RepositoryProvider,
    RepositoryResolution,
    StaticRepositoryAnalyzer,
)

from .runtime import ApprovalRecorder
from .services.repository_insight_models import (
    RepositoryInsightPlannerOutput,
    RepositorySourceIndex,
    RepositoryStructureSummary,
)
from .services.repository_insight_store import LocalRepositoryInsightStore


class RepositoryInsightPlanner(Protocol):
    """Application-local code-reading planner surface."""

    async def analyze(
        self,
        *,
        resolution: RepositoryResolution,
        metadata: RepositoryMetadata,
        snapshot: ArtifactRef,
        source_index: RepositorySourceIndex,
        structure: RepositoryStructureSummary,
        source_reader: BoundedZipSourceReader,
        ctx: OperationContext,
    ) -> RepositoryInsightPlannerOutput:
        """Return strict advice through the internal four-tool LangGraph."""


def repository_insight_utc_now() -> datetime:
    return datetime.now(UTC)


def _default_policy() -> RepositoryPolicy:
    return RepositoryPolicy(
        schema_version="1",
        policy_id="pipeline-pytorch-classification",
        policy_version="1",
    )


@dataclass(slots=True)
class RepositoryInsightDependencies:
    """Injected read-only GitHub, source, LLM and local output boundaries."""

    repository_provider: RepositoryProvider
    static_analyzer: StaticRepositoryAnalyzer
    source_reader: BoundedZipSourceReader
    planner: RepositoryInsightPlanner
    store: LocalRepositoryInsightStore
    approval_recorder: ApprovalRecorder
    actor_id: str = "pipeline-user"
    policy: RepositoryPolicy = field(default_factory=_default_policy)
    clock: Callable[[], datetime] = repository_insight_utc_now

    def __post_init__(self) -> None:
        if not self.actor_id.strip():
            raise ValueError("repository insight actor_id must be non-blank")
        if not callable(self.clock):
            raise TypeError("repository insight clock must be callable")


class RepositoryInsightInput(BaseModel):
    """Strict initial input for one direct or paper-context repository insight."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    schema_version: Literal["1"] = "1"
    workflow_id: str
    thread_id: str
    repository_url: str

    @field_validator("workflow_id", "thread_id", "repository_url")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("repository insight initial values must be non-blank")
        return value


class RepositoryInsightState(TypedDict, total=False):
    """Checkpoint-safe outer state containing only small JSON values and relative refs."""

    schema_version: Literal["1"]
    workflow_id: str
    thread_id: str
    repository_url: str
    gate_id: str
    status: Literal[
        "PENDING",
        "WAITING_FOR_HUMAN",
        "RUNNING",
        "COMPLETED",
        "REJECTED",
        "FAILED",
    ]
    route: Literal["GATE", "ANALYZE", "REJECTED", "COMPLETED", "FAILED"]
    result_ref: str | None
    failure_code: str | None


def create_repository_insight_state(
    value: RepositoryInsightInput | dict[str, object],
) -> RepositoryInsightState:
    """Validate and project an initial small state."""
    initial = (
        value
        if isinstance(value, RepositoryInsightInput)
        else RepositoryInsightInput.model_validate(value)
    )
    return {
        "schema_version": "1",
        "workflow_id": initial.workflow_id,
        "thread_id": initial.thread_id,
        "repository_url": initial.repository_url,
        "gate_id": f"gate-public-repository-snapshot-{initial.workflow_id}-r1",
        "status": "PENDING",
        "route": "GATE",
        "result_ref": None,
        "failure_code": None,
    }


__all__ = [
    "RepositoryInsightDependencies",
    "RepositoryInsightInput",
    "RepositoryInsightPlanner",
    "RepositoryInsightState",
    "create_repository_insight_state",
    "repository_insight_utc_now",
]
