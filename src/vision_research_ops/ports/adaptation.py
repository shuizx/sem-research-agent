"""Patch-workspace and bounded validation-runner port interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vision_research_ops.domain import ArtifactRef, RepositorySnapshot, ValidationResult

from .common import (
    OperationContext,
    PatchDocument,
    PatchResult,
    ValidationStageSpec,
    WorkspaceRef,
)


@runtime_checkable
class PatchWorkspace(Protocol):
    """Create and mutate opaque patch workspaces through structured patch documents."""

    async def create(
        self,
        repository: RepositorySnapshot,
        operation_id: str,
        *,
        ctx: OperationContext,
    ) -> WorkspaceRef:
        """Create an isolated workspace for a fixed repository snapshot."""

    async def apply_patch(
        self,
        workspace: WorkspaceRef,
        patch: PatchDocument,
        *,
        ctx: OperationContext,
    ) -> PatchResult:
        """Apply a policy-approved structured patch to one opaque workspace."""

    async def export_patch(self, workspace: WorkspaceRef, *, ctx: OperationContext) -> ArtifactRef:
        """Export the workspace patch as an immutable artifact."""

    async def destroy(self, workspace: WorkspaceRef, *, ctx: OperationContext) -> None:
        """Destroy a completed or failed isolated workspace."""


@runtime_checkable
class ValidationRunner(Protocol):
    """Run one bounded, policy-generated validation stage at a time."""

    async def run_stage(
        self,
        spec: ValidationStageSpec,
        *,
        ctx: OperationContext,
    ) -> ValidationResult:
        """Return the structured result for exactly one validation stage."""


__all__ = ["PatchWorkspace", "ValidationRunner"]
