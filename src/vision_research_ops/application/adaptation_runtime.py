"""Runtime-only dependencies for the adaptation workflow Adaptation Agent graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from vision_research_ops.domain import DatasetProfile
from vision_research_ops.ports import OperationContext, StructuredGenerationResult

from .runtime import ApprovalRecorder
from .services.adaptation_models import (
    AdaptationInputFacts,
    AdaptationPlannerTrace,
    AdaptationPlanProposal,
    CompiledAdaptationPlan,
    PatchArtifactRecord,
    SmokeResultRecord,
)
from .services.adaptation_store import LocalAdaptationStore
from .services.repository_store import LocalRepositoryStore


def adaptation_utc_now() -> datetime:
    """Return an aware UTC timestamp for injected adaptation dependencies."""
    return datetime.now(UTC)


@runtime_checkable
class BoundedPatchTool(Protocol):
    """Apply only a validated compiled plan to the controlled fixture."""

    async def apply(
        self,
        plan: CompiledAdaptationPlan,
        *,
        ctx: OperationContext,
    ) -> PatchArtifactRecord:
        """Return immutable patch evidence without exposing a host path."""


@runtime_checkable
class BoundedSmokeTool(Protocol):
    """Run the fixed fixture validation stages for one exact patch."""

    async def run(
        self,
        patch: PatchArtifactRecord,
        *,
        ctx: OperationContext,
    ) -> SmokeResultRecord:
        """Return actual structured evidence for the bounded stage sequence."""


@dataclass(frozen=True, slots=True)
class AdaptationPlannerOutput:
    """Validated proposal provenance plus its hash-only tool trace."""

    generation: StructuredGenerationResult[AdaptationPlanProposal]
    trace: AdaptationPlannerTrace


@runtime_checkable
class AdaptationPlanner(Protocol):
    """Run one bounded, read-only tool-calling planning loop."""

    async def plan(
        self,
        facts: AdaptationInputFacts,
        *,
        ctx: OperationContext,
    ) -> AdaptationPlannerOutput:
        """Return a strict proposal only after required tool validation."""


@dataclass(slots=True)
class AdaptationDependencies:
    """Injected repository result, dataset, LLM, tools, local store, and human recorder."""

    repository_store: LocalRepositoryStore
    repository_workflow_id: str
    dataset_profile: DatasetProfile
    planner: AdaptationPlanner
    patch_tool: BoundedPatchTool
    smoke_tool: BoundedSmokeTool
    store: LocalAdaptationStore
    approval_recorder: ApprovalRecorder
    actor_id: str = "pipeline-user"
    clock: Callable[[], datetime] = adaptation_utc_now

    def __post_init__(self) -> None:
        for name in ("repository_workflow_id", "actor_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        if not callable(self.clock):
            raise TypeError("adaptation clock must be callable")


__all__ = [
    "AdaptationDependencies",
    "AdaptationPlanner",
    "AdaptationPlannerOutput",
    "BoundedPatchTool",
    "BoundedSmokeTool",
    "adaptation_utc_now",
]
