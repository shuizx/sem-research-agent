"""Fixture-only nodes for the first executable LangGraph vertical slice."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from vision_research_ops.domain import (
    ApprovalDecision,
    GateKind,
    WorkflowPhase,
    WorkflowStatus,
    approval_authorizes,
)
from vision_research_ops.ports import OperationContext, PortError, make_failure

from ..runtime import WorkflowDependencies
from ..state import InitialWorkflowInput, WorkflowState

RUN_SUBMISSION_SUBJECT_TYPE = "adaptation_plan"


def _required_text(state: WorkflowState, field: str) -> str:
    """Read one required small string reference without silently repairing state."""
    value = state.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"workflow state field {field} must be a non-blank string")
    return value


def fixture_plan_revision(state: WorkflowState) -> int:
    """Derive the current fixture plan revision from accumulated EDIT retries."""
    retries = state.get("retry_counts", {})
    edit_count = retries.get("edit", 0)
    if isinstance(edit_count, bool) or not isinstance(edit_count, int) or edit_count < 0:
        raise ValueError("retry_counts.edit must be a non-negative integer")
    return edit_count + 1


def fixture_plan_id(base_plan_id: str, revision: int) -> str:
    """Generate a stable fixture plan ID without storing a hidden revision field."""
    if revision < 1:
        raise ValueError("fixture plan revision must be positive")
    return base_plan_id if revision == 1 else f"{base_plan_id}-r{revision}"


def fixture_gate_id(plan_id: str, revision: int) -> str:
    """Generate a deterministic run-submission gate ID for one plan revision."""
    return f"gate-run-submission-{plan_id}-r{revision}"


async def validate_request(state: WorkflowState, config: RunnableConfig) -> WorkflowState:
    """Validate initial references and transition the fixture workflow to retrieval."""
    initial_values: dict[str, object] = {
        "schema_version": state.get("schema_version"),
        "workflow_id": state.get("workflow_id"),
        "thread_id": state.get("thread_id"),
        "request_id": state.get("request_id"),
        "dataset_profile_id": state.get("dataset_profile_id"),
    }
    initial = InitialWorkflowInput.model_validate(initial_values)
    configurable = config.get("configurable", {})
    if not isinstance(configurable, Mapping) or configurable.get("thread_id") != initial.thread_id:
        raise ValueError("workflow config thread_id must match the validated state thread_id")
    return {
        "status": WorkflowStatus.RUNNING,
        "phase": WorkflowPhase.PAPER_RETRIEVAL,
    }


async def retrieve_research_fixture(
    state: WorkflowState,
    runtime: Runtime[WorkflowDependencies],
) -> WorkflowState:
    """Load deterministic paper references without contacting a paper provider."""
    del state
    paper_ids = list(runtime.context.fixture_paper_candidate_ids)
    return {
        "paper_candidate_ids": paper_ids,
        "selected_paper_ids": [paper_ids[0]],
        "phase": WorkflowPhase.ADAPTATION_PLANNING,
    }


async def propose_adaptation_fixture(
    state: WorkflowState,
    runtime: Runtime[WorkflowDependencies],
) -> WorkflowState:
    """Create a stable fixture plan revision without LLM reasoning or a code patch."""
    dependencies = runtime.context
    revision = fixture_plan_revision(state)
    plan_id = fixture_plan_id(dependencies.fixture_plan_id, revision)
    return {
        "repository_snapshot_ids": [dependencies.fixture_repository_snapshot_id],
        "active_repository_id": dependencies.fixture_repository_id,
        "active_plan_id": plan_id,
        "experiment_id": dependencies.fixture_experiment_id,
        "phase": WorkflowPhase.AWAITING_RUN_APPROVAL,
        "status": WorkflowStatus.WAITING_FOR_HUMAN,
        "pending_gate_id": fixture_gate_id(plan_id, revision),
        "route": None,
    }


def _submit_failure(
    *,
    code: str,
    category: str,
    message: str,
) -> WorkflowState:
    """Return a terminal, de-sensitized fail-closed update without a port call."""
    return {
        "pending_gate_id": None,
        "status": WorkflowStatus.FAILED,
        "phase": WorkflowPhase.FAILED,
        "route": "FAILED",
        "last_error": make_failure(
            code=code,
            category=category,
            message=message,
            retryable=False,
            ctx=None,
        ),
    }


async def submit_training_fixture(
    state: WorkflowState,
    runtime: Runtime[WorkflowDependencies],
) -> WorkflowState:
    """Submit the injected fake executor only after exact, recorded approval."""
    dependencies = runtime.context
    try:
        plan_id = _required_text(state, "active_plan_id")
        revision = fixture_plan_revision(state)
        experiment_id = _required_text(state, "experiment_id")
    except ValueError:
        return _submit_failure(
            code="RUN_SUBMISSION_STATE_INVALID",
            category="STATE",
            message="The workflow state is incomplete for fixture run submission.",
        )

    if state.get("route") != ApprovalDecision.APPROVE.value:
        return _submit_failure(
            code="RUN_SUBMISSION_APPROVAL_REQUIRED",
            category="AUTHORIZATION",
            message="A matching run-submission approval is required before submission.",
        )
    if state.get("pending_gate_id") is not None:
        return _submit_failure(
            code="RUN_SUBMISSION_GATE_NOT_CLEARED",
            category="AUTHORIZATION",
            message="The run-submission gate must be cleared by a recorded approval.",
        )

    approval = dependencies.approval_recorder.find_exact(
        gate_kind=GateKind.RUN_SUBMISSION,
        subject_type=RUN_SUBMISSION_SUBJECT_TYPE,
        subject_id=plan_id,
        subject_revision=revision,
        decision=ApprovalDecision.APPROVE,
    )
    if approval is None or not approval_authorizes(
        approval,
        RUN_SUBMISSION_SUBJECT_TYPE,
        plan_id,
        revision,
    ):
        return _submit_failure(
            code="RUN_SUBMISSION_APPROVAL_REQUIRED",
            category="AUTHORIZATION",
            message="A matching run-submission approval is required before submission.",
        )
    if dependencies.run_spec.experiment_id != experiment_id:
        return _submit_failure(
            code="RUN_SPEC_EXPERIMENT_MISMATCH",
            category="STATE",
            message="The frozen fixture run does not match the active experiment reference.",
        )

    workflow_id = _required_text(state, "workflow_id")
    context = OperationContext(
        schema_version="1",
        correlation_id=f"corr-{workflow_id}-{dependencies.run_spec.run_id}",
        workflow_id=workflow_id,
        actor_id=approval.actor_id,
        idempotency_key=dependencies.run_spec.idempotency_key,
        sensitivity="INTERNAL",
    )
    try:
        await dependencies.executor.submit(dependencies.run_spec, ctx=context)
    except PortError as error:
        return {
            "pending_gate_id": None,
            "status": WorkflowStatus.FAILED,
            "phase": WorkflowPhase.FAILED,
            "route": "FAILED",
            "last_error": error.failure,
        }
    return {
        "run_ids": [dependencies.run_spec.run_id],
        "phase": WorkflowPhase.EVALUATION,
        "status": WorkflowStatus.RUNNING,
        "route": "SUBMITTED",
        "last_error": None,
    }


async def analyze_result_fixture(
    state: WorkflowState,
    runtime: Runtime[WorkflowDependencies],
) -> WorkflowState:
    """Record a fixture report reference after a successful fake submission only."""
    if state.get("route") != "SUBMITTED" or not state.get("run_ids"):
        raise ValueError("fixture analysis requires a successful fixture submission")
    return {
        "report_id": runtime.context.fixture_report_id,
        "phase": WorkflowPhase.COMPLETED,
        "status": WorkflowStatus.SUCCEEDED,
        "route": "COMPLETED",
        "pending_gate_id": None,
        "last_error": None,
    }


def route_after_human_gate(state: WorkflowState) -> Literal["APPROVE", "EDIT", "REJECT"]:
    """Route a validated human decision without interpreting free text."""
    route = state.get("route")
    if route in {"APPROVE", "EDIT", "REJECT"}:
        return cast(Literal["APPROVE", "EDIT", "REJECT"], route)
    raise ValueError("human gate did not produce a supported approval route")


def route_after_submit(state: WorkflowState) -> Literal["SUBMITTED", "FAILED"]:
    """Allow analysis only after a successful fixture executor submission."""
    route = state.get("route")
    if route in {"SUBMITTED", "FAILED"}:
        return cast(Literal["SUBMITTED", "FAILED"], route)
    raise ValueError("fixture submission did not produce a supported route")


__all__ = [
    "RUN_SUBMISSION_SUBJECT_TYPE",
    "analyze_result_fixture",
    "fixture_gate_id",
    "fixture_plan_id",
    "fixture_plan_revision",
    "propose_adaptation_fixture",
    "retrieve_research_fixture",
    "route_after_human_gate",
    "route_after_submit",
    "submit_training_fixture",
    "validate_request",
]
