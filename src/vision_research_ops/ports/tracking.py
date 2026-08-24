"""Provider-neutral experiment tracking port interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vision_research_ops.domain import ArtifactRef, RunStatus

from .common import MetricPoint, OperationContext, RunManifest, TrackerRunRef


@runtime_checkable
class ExperimentTracker(Protocol):
    """Record structured run evidence without importing a concrete tracking SDK."""

    async def create_run(self, manifest: RunManifest, *, ctx: OperationContext) -> TrackerRunRef:
        """Create or safely replay a tracker run for one immutable run manifest."""

    async def log_metrics(
        self,
        run: TrackerRunRef,
        metrics: list[MetricPoint],
        *,
        ctx: OperationContext,
    ) -> None:
        """Append finite structured metrics for a tracker run."""

    async def log_artifact_refs(
        self,
        run: TrackerRunRef,
        artifacts: list[ArtifactRef],
        *,
        ctx: OperationContext,
    ) -> None:
        """Attach immutable artifact references without uploading arbitrary source data."""

    async def finalize(
        self,
        run: TrackerRunRef,
        status: RunStatus,
        *,
        ctx: OperationContext,
    ) -> None:
        """Finalize a tracker run with the normalized domain run status."""


__all__ = ["ExperimentTracker"]
