"""Bounded experiment-executor port interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .common import (
    CancellationResult,
    ExternalRunStatus,
    FrozenRunSpec,
    OperationContext,
    SubmissionResult,
)


@runtime_checkable
class ExperimentExecutor(Protocol):
    """Submit, poll, and cancel only structured, pre-approved experiment runs."""

    executor_name: str

    async def submit(self, run: FrozenRunSpec, *, ctx: OperationContext) -> SubmissionResult:
        """Submit an idempotent frozen run without interpreting a shell command string."""

    async def get_status(
        self,
        external_job_id: str,
        *,
        ctx: OperationContext,
    ) -> ExternalRunStatus:
        """Perform one bounded status query for an external job."""

    async def cancel(
        self,
        external_job_id: str,
        *,
        ctx: OperationContext,
    ) -> CancellationResult:
        """Request bounded cancellation of an external job."""


__all__ = ["ExperimentExecutor"]
