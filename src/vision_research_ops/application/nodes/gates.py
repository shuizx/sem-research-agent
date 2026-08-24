"""Human-gate interrupt node for the fixture-only vertical slice."""

from __future__ import annotations

import json

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from vision_research_ops.domain import (
    Approval,
    ApprovalDecision,
    GateKind,
    WorkflowPhase,
    WorkflowStatus,
)

from ..runtime import WorkflowDependencies
from ..state import WorkflowState
from .fixtures import (
    RUN_SUBMISSION_SUBJECT_TYPE,
    _required_text,
    fixture_gate_id,
    fixture_plan_revision,
)


def _gate_request(state: WorkflowState) -> dict[str, object]:
    """Build a compact JSON-safe request for one fixture run-submission gate."""
    plan_id = _required_text(state, "active_plan_id")
    revision = fixture_plan_revision(state)
    expected_gate_id = fixture_gate_id(plan_id, revision)
    if state.get("pending_gate_id") != expected_gate_id:
        raise ValueError("pending gate does not match the active fixture plan revision")
    return {
        "schema_version": "1",
        "gate_id": expected_gate_id,
        "gate_kind": GateKind.RUN_SUBMISSION.value,
        "subject_type": RUN_SUBMISSION_SUBJECT_TYPE,
        "subject_id": plan_id,
        "subject_revision": revision,
        "summary": "Fixture-only adaptation plan; approval controls fake run submission.",
    }


def _revalidate_approval(resume_value: object) -> Approval:
    """Force resume data through the existing strict Approval Pydantic model."""
    if isinstance(resume_value, Approval):
        resume_value = resume_value.model_dump(mode="json")
    return Approval.model_validate_json(json.dumps(resume_value, allow_nan=False))


def _validate_gate_subject(approval: Approval, state: WorkflowState) -> None:
    """Fail closed unless approval targets the current fixture plan revision exactly."""
    plan_id = _required_text(state, "active_plan_id")
    revision = fixture_plan_revision(state)
    if approval.gate_kind is not GateKind.RUN_SUBMISSION:
        raise ValueError("approval gate_kind is not RUN_SUBMISSION")
    if approval.subject_type != RUN_SUBMISSION_SUBJECT_TYPE:
        raise ValueError("approval subject_type is not adaptation_plan")
    if approval.subject_id != plan_id:
        raise ValueError("approval subject_id does not match the active fixture plan")
    if approval.subject_revision != revision:
        raise ValueError("approval subject_revision does not match the active fixture plan")


async def human_gate(
    state: WorkflowState,
    runtime: Runtime[WorkflowDependencies],
) -> WorkflowState:
    """Interrupt, validate a resumed Approval, and return a deterministic route."""
    resume_value = interrupt(_gate_request(state))
    approval = _revalidate_approval(resume_value)
    _validate_gate_subject(approval, state)
    runtime.context.approval_recorder.record(approval)

    if approval.decision is ApprovalDecision.APPROVE:
        return {
            "pending_gate_id": None,
            "route": ApprovalDecision.APPROVE.value,
            "status": WorkflowStatus.RUNNING,
            "phase": WorkflowPhase.RUN_SUBMISSION,
        }
    if approval.decision is ApprovalDecision.REJECT:
        return {
            "pending_gate_id": None,
            "route": ApprovalDecision.REJECT.value,
            "status": WorkflowStatus.REJECTED,
            "phase": WorkflowPhase.REJECTED,
        }
    if approval.decision is ApprovalDecision.EDIT:
        return {
            "pending_gate_id": None,
            "route": ApprovalDecision.EDIT.value,
            "status": WorkflowStatus.RUNNING,
            "phase": WorkflowPhase.ADAPTATION_PLANNING,
            "retry_counts": {"edit": 1},
        }
    raise ValueError("approval decision is not supported by the fixture gate")


__all__ = ["human_gate"]
