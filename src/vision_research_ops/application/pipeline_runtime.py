"""Runtime-only protocols for the integrated top-level Pipeline StateGraph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from vision_research_ops.domain import Approval

from .services.pipeline_models import (
    LocalPipelineSummaryStore,
    PipelineFailure,
    PipelineGateRecord,
    PipelineStageName,
    PipelineStageRecord,
    PipelineState,
    PipelineSummary,
)


@dataclass(frozen=True, slots=True)
class PipelineStageOutcome:
    """Compact result returned after driving one existing child graph."""

    record: PipelineStageRecord
    gates: tuple[PipelineGateRecord, ...] = ()
    conclusion: str | None = None


@runtime_checkable
class DecisionProvider(Protocol):
    """Provide one typed decision for each real child interrupt."""

    @property
    def scripted_fixture_decisions(self) -> bool:
        """Return whether decisions came from an explicit fixture-only script."""

    def decide(
        self,
        *,
        workflow_id: str,
        stage: PipelineStageName,
        payload: dict[str, object],
        occurrence: int,
        decided_at: datetime,
    ) -> Approval:
        """Return an Approval bound to the exact interrupt payload."""


@runtime_checkable
class PipelineStageDriver(Protocol):
    """Narrow driver for real research-to-evaluation graphs and their fixture dependencies."""

    async def run_stage(
        self,
        *,
        pipeline_workflow_id: str,
        stage: PipelineStageName,
    ) -> PipelineStageOutcome:
        """Invoke, interrupt, resume, and inspect one existing child graph."""

    def build_summary(self, state: PipelineState) -> PipelineSummary:
        """Build a strict summary by re-reading canonical child artifacts."""


@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    """Injected stage driver, write-once store, clock, and compact event sink."""

    driver: PipelineStageDriver
    summary_store: LocalPipelineSummaryStore
    event_sink: Callable[[str], None]
    clock: Callable[[], datetime]

    def __post_init__(self) -> None:
        if not callable(self.event_sink):
            raise TypeError("pipeline event_sink must be callable")
        if not callable(self.clock):
            raise TypeError("pipeline clock must be callable")


def stopped_failure(stage: PipelineStageName) -> PipelineFailure:
    """Return the stable top-level reason used for a human rejection."""
    return PipelineFailure(
        code="PIPELINE_HUMAN_REJECTED",
        message="A human Gate rejected the current stage; downstream stages were not run.",
        stage=stage,
    )


__all__ = [
    "DecisionProvider",
    "PipelineDependencies",
    "PipelineStageDriver",
    "PipelineStageOutcome",
    "stopped_failure",
]
