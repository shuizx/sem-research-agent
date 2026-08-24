"""Runtime-only dependencies for the fixture-only vertical-slice workflow graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from vision_research_ops.domain import Approval, ApprovalDecision, GateKind
from vision_research_ops.ports import ExperimentExecutor, FrozenRunSpec


def utc_now() -> datetime:
    """Return an aware UTC instant for injected runtime dependencies."""
    return datetime.now(UTC)


@runtime_checkable
class ApprovalRecorder(Protocol):
    """Small single-process audit boundary for fixture human approvals."""

    def record(self, approval: Approval) -> None:
        """Record an approval once by ID, rejecting conflicting re-use."""

    def find_exact(
        self,
        *,
        gate_kind: GateKind,
        subject_type: str,
        subject_id: str,
        subject_revision: int,
        decision: ApprovalDecision,
    ) -> Approval | None:
        """Return an approval bound to the exact reviewed subject revision."""


@dataclass(slots=True)
class InMemoryApprovalRecorder:
    """Deterministic in-memory approval recorder for one graph-process lifetime."""

    _approvals: dict[str, Approval] = field(default_factory=dict, init=False, repr=False)

    def record(self, approval: Approval) -> None:
        """Idempotently retain an approval while rejecting payload conflicts."""
        existing = self._approvals.get(approval.approval_id)
        if existing is None:
            self._approvals[approval.approval_id] = approval
            return
        if existing.model_dump(mode="json") != approval.model_dump(mode="json"):
            raise ValueError("approval_id cannot be reused with a different approval payload")

    def find_exact(
        self,
        *,
        gate_kind: GateKind,
        subject_type: str,
        subject_id: str,
        subject_revision: int,
        decision: ApprovalDecision,
    ) -> Approval | None:
        """Find a decision for one exact gate subject and immutable revision."""
        for approval in self._approvals.values():
            if (
                approval.gate_kind is gate_kind
                and approval.subject_type == subject_type
                and approval.subject_id == subject_id
                and approval.subject_revision == subject_revision
                and approval.decision is decision
            ):
                return approval
        return None

    def get(self, approval_id: str) -> Approval | None:
        """Return one recorded approval for graph-test assertions."""
        return self._approvals.get(approval_id)

    @property
    def approvals(self) -> tuple[Approval, ...]:
        """Expose a stable read-only snapshot for audit-oriented tests."""
        return tuple(self._approvals.values())


@dataclass(frozen=True, slots=True)
class WorkflowDependencies:
    """Explicit runtime-only fixture dependencies injected through LangGraph context."""

    executor: ExperimentExecutor
    run_spec: FrozenRunSpec
    fixture_paper_candidate_ids: tuple[str, ...]
    fixture_repository_snapshot_id: str
    fixture_repository_id: str
    fixture_plan_id: str
    fixture_experiment_id: str
    fixture_report_id: str
    approval_recorder: ApprovalRecorder
    clock: Callable[[], datetime] = utc_now

    def __post_init__(self) -> None:
        """Reject incomplete fixture wiring before a graph is compiled."""
        identifiers = {
            "fixture_repository_snapshot_id": self.fixture_repository_snapshot_id,
            "fixture_repository_id": self.fixture_repository_id,
            "fixture_plan_id": self.fixture_plan_id,
            "fixture_experiment_id": self.fixture_experiment_id,
            "fixture_report_id": self.fixture_report_id,
        }
        for name, value in identifiers.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank fixture identifier")
        if not self.fixture_paper_candidate_ids:
            raise ValueError("fixture_paper_candidate_ids must not be empty")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.fixture_paper_candidate_ids
        ):
            raise ValueError("fixture_paper_candidate_ids must contain non-blank identifiers")
        if not callable(self.clock):
            raise TypeError("clock must be callable")


__all__ = [
    "ApprovalRecorder",
    "InMemoryApprovalRecorder",
    "WorkflowDependencies",
    "utc_now",
]
