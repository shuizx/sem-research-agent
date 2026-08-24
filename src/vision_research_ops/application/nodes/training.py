"""LangGraph nodes for exact training approval and two controlled local runs."""

from __future__ import annotations

import json
import re
from typing import Literal, cast

from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import ValidationError

from vision_research_ops.domain import (
    Approval,
    ApprovalDecision,
    GateKind,
    StructuredFailure,
    WorkflowPhase,
    WorkflowStatus,
)
from vision_research_ops.ports import OperationContext, make_failure

from ..services.training_freeze import (
    apply_training_edits,
    approval_edits,
    freeze_training_spec,
    submission_id,
    training_gate_id,
    validate_training_input,
)
from ..services.training_models import (
    FrozenTrainingSpec,
    TrainingInput,
    TrainingReviewRecord,
    TrainingWorkflowRecord,
)
from ..state import InitialWorkflowInput, WorkflowState
from ..training_runtime import TrainingDependencies, TrainingToolError

TRAINING_SUBJECT_TYPE = "training_submission"
_CANONICAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _required_text(state: WorkflowState, field: str) -> str:
    value = state.get(field)
    if not isinstance(value, str) or _CANONICAL_ID_RE.fullmatch(value) is None:
        raise ValueError(f"workflow state field {field} must be a canonical identifier")
    return value


def _ctx(
    state: WorkflowState,
    dependencies: TrainingDependencies,
    *,
    operation: str,
) -> OperationContext:
    workflow_id = _required_text(state, "workflow_id")
    return OperationContext(
        schema_version="1",
        correlation_id=f"corr-{workflow_id}-{operation}",
        workflow_id=workflow_id,
        actor_id=dependencies.actor_id,
        idempotency_key=f"{workflow_id}:{operation}",
        sensitivity="INTERNAL",
    )


def _failure(*, code: str, message: str) -> StructuredFailure:
    return make_failure(
        code=code,
        category="TRAINING",
        message=message,
        retryable=False,
        ctx=None,
    )


def _failure_state(failure: StructuredFailure) -> WorkflowState:
    return {
        "phase": WorkflowPhase.FAILED,
        "status": WorkflowStatus.FAILED,
        "pending_gate_id": None,
        "route": "FAILED",
        "last_error": failure,
    }


def _updated(record: TrainingWorkflowRecord, **updates: object) -> TrainingWorkflowRecord:
    return TrainingWorkflowRecord.model_validate({**record.model_dump(mode="json"), **updates})


def _terminal_failure(
    dependencies: TrainingDependencies,
    record: TrainingWorkflowRecord,
    failure: StructuredFailure,
) -> WorkflowState:
    failed = _updated(
        record,
        status="FAILED",
        gate_id=None,
        pending_edit=None,
        failure=failure.model_dump(mode="json"),
        updated_at=dependencies.clock(),
    )
    dependencies.store.write_workflow(failed)
    return _failure_state(failure)


def _request(dependencies: TrainingDependencies) -> TrainingInput:
    raw = dependencies.training_input
    payload = raw.model_dump(mode="json") if isinstance(raw, TrainingInput) else dict(raw)
    return TrainingInput.model_validate(payload)


async def validate_input(
    state: WorkflowState,
    runtime: Runtime[TrainingDependencies],
) -> WorkflowState:
    """Accept only exact adaptation-approved fixture evidence before a Gate can exist."""
    dependencies = runtime.context
    try:
        initial = InitialWorkflowInput.model_validate(
            {
                "schema_version": state.get("schema_version"),
                "workflow_id": state.get("workflow_id"),
                "thread_id": state.get("thread_id"),
                "request_id": state.get("request_id"),
                "dataset_profile_id": state.get("dataset_profile_id"),
            }
        )
        for value in (initial.workflow_id, initial.thread_id, initial.request_id):
            if _CANONICAL_ID_RE.fullmatch(value) is None:
                raise ValueError("training workflow identifiers must be canonical")
        request, _ = validate_training_input(
            dependencies.training_input,
            reader=dependencies.adaptation_reader,
            project_root=dependencies.project_root,
        )
        if initial.dataset_profile_id != request.dataset_id:
            raise ValueError("state dataset ID does not match the frozen training input")
        now = dependencies.clock()
        record = TrainingWorkflowRecord(
            workflow_id=initial.workflow_id,
            request_id=initial.request_id,
            adaptation_workflow_id=request.adaptation_workflow_id,
            status="INPUT_VALIDATED",
            created_at=now,
            updated_at=now,
        )
        record_ref = dependencies.store.write_workflow(record)
    except (OSError, TypeError, ValueError, ValidationError):
        return _failure_state(
            _failure(
                code="TRAINING_INPUT_INVALID",
                message=(
                    "Training requires exact accepted adaptation fixture, commit, patch, data, "
                    "and split evidence."
                ),
            )
        )
    return {
        "report_id": record_ref,
        "phase": WorkflowPhase.EXPERIMENT_FREEZE,
        "status": WorkflowStatus.RUNNING,
        "route": "FREEZE",
        "last_error": None,
    }


def _awaiting_record(
    dependencies: TrainingDependencies,
    record: TrainingWorkflowRecord,
    spec: FrozenTrainingSpec,
) -> TrainingWorkflowRecord:
    return _updated(
        record,
        status="AWAITING_APPROVAL",
        revision=spec.revision,
        current_spec_ref=dependencies.store.spec_ref(spec.workflow_id, spec.revision),
        current_spec_hash=spec.spec_hash,
        submission_id=submission_id(spec),
        gate_id=training_gate_id(spec),
        approval_id=None,
        pending_edit=None,
        failure=None,
        updated_at=dependencies.clock(),
    )


async def freeze_submission(
    state: WorkflowState,
    runtime: Runtime[TrainingDependencies],
) -> WorkflowState:
    """Freeze the same dataset/split/preprocess/seed/budget for both runs."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    record = dependencies.store.load_workflow(workflow_id)
    if record.status != "INPUT_VALIDATED":
        raise ValueError("initial freeze requires validated training input")
    try:
        request, evidence = validate_training_input(
            dependencies.training_input,
            reader=dependencies.adaptation_reader,
            project_root=dependencies.project_root,
        )
        spec = freeze_training_spec(
            workflow_id=workflow_id,
            request=request,
            evidence=evidence,
            revision=1,
        )
        spec_ref = dependencies.store.write_spec(spec)
        awaiting = _awaiting_record(dependencies, record, spec)
        record_ref = dependencies.store.write_workflow(awaiting)
    except (OSError, TypeError, ValueError, ValidationError):
        return _terminal_failure(
            dependencies,
            record,
            _failure(
                code="TRAINING_FREEZE_FAILED",
                message="The fair baseline/candidate training submission could not be frozen.",
            ),
        )
    return {
        "report_id": record_ref,
        "active_plan_id": awaiting.submission_id,
        "experiment_id": awaiting.submission_id,
        "pending_gate_id": awaiting.gate_id,
        "phase": WorkflowPhase.AWAITING_RUN_APPROVAL,
        "status": WorkflowStatus.WAITING_FOR_HUMAN,
        "route": "GATE",
        "last_error": None,
        "validation_result_ids": [spec_ref],
    }


def _revalidate_approval(value: object) -> Approval:
    if isinstance(value, Approval):
        value = value.model_dump(mode="json")
    return Approval.model_validate_json(json.dumps(value, allow_nan=False))


def _review(
    record: TrainingWorkflowRecord, spec: FrozenTrainingSpec, approval: Approval
) -> TrainingReviewRecord:
    if record.gate_id is None or record.submission_id is None:
        raise ValueError("training review requires complete Gate evidence")
    return TrainingReviewRecord(
        approval_id=approval.approval_id,
        decision=approval.decision.value,
        gate_id=record.gate_id,
        subject_id=record.submission_id,
        subject_revision=spec.revision,
        spec_hash=spec.spec_hash,
        actor_id=approval.actor_id,
        decided_at=approval.decided_at,
    )


async def training_gate(
    state: WorkflowState,
    runtime: Runtime[TrainingDependencies],
) -> WorkflowState:
    """Interrupt on one exact frozen spec and route approve/edit/reject."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    record = dependencies.store.load_workflow(workflow_id)
    if (
        record.status != "AWAITING_APPROVAL"
        or record.current_spec_ref is None
        or record.current_spec_hash is None
        or record.submission_id is None
        or record.gate_id is None
        or record.revision is None
        or state.get("pending_gate_id") != record.gate_id
    ):
        raise ValueError("training Gate requires exact persisted frozen evidence")
    spec = dependencies.store.load_spec(record.current_spec_ref)
    if (
        spec.spec_hash != record.current_spec_hash
        or spec.revision != record.revision
        or submission_id(spec) != record.submission_id
        or training_gate_id(spec) != record.gate_id
    ):
        raise ValueError("training Gate does not match the current frozen spec")
    resume_value = interrupt(
        {
            "schema_version": "1",
            "gate_id": record.gate_id,
            "gate_kind": GateKind.RUN_SUBMISSION.value,
            "subject_type": TRAINING_SUBJECT_TYPE,
            "subject_id": record.submission_id,
            "subject_revision": spec.revision,
            "frozen_spec_hash": spec.spec_hash,
            "base_commit_sha": spec.base_commit_sha,
            "patch_revision": spec.patch_revision,
            "patch_hash": spec.patch_hash,
            "dataset_id": spec.baseline.dataset_id,
            "dataset_version": spec.baseline.dataset_version,
            "split_ref": spec.baseline.split_ref,
            "split_hash": spec.baseline.split_hash,
            "preprocess_ref": spec.baseline.preprocess_ref,
            "preprocess_hash": spec.baseline.preprocess_hash,
            "seed": spec.baseline.seed,
            "budget": spec.baseline.budget.model_dump(mode="json"),
            "baseline_method": spec.baseline.method,
            "candidate_method": spec.candidate.method,
            "capability": spec.baseline.capability,
            "real_pytorch_training": False,
            "editable_fields": [
                "/budget/max_epochs",
                "/budget/max_steps",
                "/budget/max_walltime_seconds",
                "/seed",
            ],
        }
    )
    approval = _revalidate_approval(resume_value)
    if (
        approval.gate_kind is not GateKind.RUN_SUBMISSION
        or approval.subject_type != TRAINING_SUBJECT_TYPE
        or approval.subject_id != record.submission_id
        or approval.subject_revision != spec.revision
        or approval.subject_id != submission_id(spec)
    ):
        raise ValueError("approval does not target the current training spec revision and hash")
    dependencies.approval_recorder.record(approval)
    review = _review(record, spec, approval)
    reviews = [*record.reviews, review]
    if approval.decision is ApprovalDecision.REJECT:
        rejected = _updated(
            record,
            status="REJECTED",
            reviews=reviews,
            approval_id=approval.approval_id,
            gate_id=None,
            updated_at=dependencies.clock(),
        )
        record_ref = dependencies.store.write_workflow(rejected)
        return {
            "report_id": record_ref,
            "pending_gate_id": None,
            "phase": WorkflowPhase.REJECTED,
            "status": WorkflowStatus.REJECTED,
            "route": "REJECTED",
            "last_error": None,
        }
    if approval.decision is ApprovalDecision.APPROVE:
        approved = _updated(
            record,
            status="RUNNING",
            reviews=reviews,
            approval_id=approval.approval_id,
            gate_id=None,
            updated_at=dependencies.clock(),
        )
        record_ref = dependencies.store.write_workflow(approved)
        return {
            "report_id": record_ref,
            "pending_gate_id": None,
            "phase": WorkflowPhase.RUN_SUBMISSION,
            "status": WorkflowStatus.RUNNING,
            "route": "APPROVED",
            "last_error": None,
        }
    pending = approval_edits(approval)
    edited = _updated(
        record,
        status="EDIT_REQUESTED",
        reviews=reviews,
        approval_id=approval.approval_id,
        gate_id=None,
        pending_edit=pending.model_dump(mode="json"),
        updated_at=dependencies.clock(),
    )
    record_ref = dependencies.store.write_workflow(edited)
    return {
        "report_id": record_ref,
        "pending_gate_id": None,
        "phase": WorkflowPhase.EXPERIMENT_FREEZE,
        "status": WorkflowStatus.RUNNING,
        "route": "EDIT",
        "last_error": None,
    }


async def revise_submission(
    state: WorkflowState,
    runtime: Runtime[TrainingDependencies],
) -> WorkflowState:
    """Regenerate both run specs after one sanitized human edit and require a new Gate."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    record = dependencies.store.load_workflow(workflow_id)
    if (
        record.status != "EDIT_REQUESTED"
        or record.pending_edit is None
        or record.current_spec_ref is None
        or record.revision is None
    ):
        raise ValueError("training refreeze requires one exact pending human edit")
    try:
        current = dependencies.store.load_spec(record.current_spec_ref)
        request, evidence = validate_training_input(
            dependencies.training_input,
            reader=dependencies.adaptation_reader,
            project_root=dependencies.project_root,
        )
        effective = TrainingInput.model_validate(
            {
                **request.model_dump(mode="json"),
                "budget": current.baseline.budget.model_dump(mode="json"),
                "seed": current.baseline.seed,
            }
        )
        budget, seed = apply_training_edits(effective, record.pending_edit)
        revised = freeze_training_spec(
            workflow_id=workflow_id,
            request=request,
            evidence=evidence,
            revision=record.revision + 1,
            budget=budget,
            seed=seed,
        )
        spec_ref = dependencies.store.write_spec(revised)
        awaiting = _awaiting_record(dependencies, record, revised)
        record_ref = dependencies.store.write_workflow(awaiting)
    except (OSError, TypeError, ValueError, ValidationError):
        return _terminal_failure(
            dependencies,
            record,
            _failure(
                code="TRAINING_EDIT_INVALID",
                message=(
                    "The requested training edit could not satisfy the bounded fair-pair policy."
                ),
            ),
        )
    return {
        "report_id": record_ref,
        "active_plan_id": awaiting.submission_id,
        "experiment_id": awaiting.submission_id,
        "pending_gate_id": awaiting.gate_id,
        "phase": WorkflowPhase.AWAITING_RUN_APPROVAL,
        "status": WorkflowStatus.WAITING_FOR_HUMAN,
        "route": "GATE",
        "last_error": None,
        "retry_counts": {"training_edit": 1},
        "validation_result_ids": [spec_ref],
    }


def _cancelled(
    dependencies: TrainingDependencies,
    record: TrainingWorkflowRecord,
    *,
    point: Literal["BEFORE_BASELINE", "BETWEEN_RUNS"],
) -> WorkflowState:
    cancelled = _updated(
        record,
        status="CANCELLED",
        cancellation_point=point,
        updated_at=dependencies.clock(),
    )
    dependencies.store.write_workflow(cancelled)
    return {
        "phase": (
            WorkflowPhase.RUN_SUBMISSION
            if point == "BEFORE_BASELINE"
            else WorkflowPhase.RUN_MONITORING
        ),
        "status": WorkflowStatus.CANCELLED,
        "pending_gate_id": None,
        "route": "CANCELLED",
        "last_error": _failure(
            code="TRAINING_RUN_CANCELLED",
            message="Local training was cancelled at a normal submission boundary.",
        ),
    }


def _approved_spec(
    dependencies: TrainingDependencies,
    workflow_id: str,
) -> tuple[TrainingWorkflowRecord, FrozenTrainingSpec]:
    record = dependencies.store.load_workflow(workflow_id)
    if (
        record.status != "RUNNING"
        or record.current_spec_ref is None
        or record.current_spec_hash is None
        or record.approval_id is None
    ):
        raise ValueError("training execution requires a persisted exact approval")
    spec = dependencies.store.load_spec(record.current_spec_ref)
    review = next(
        (
            item
            for item in reversed(record.reviews)
            if item.approval_id == record.approval_id
            and item.decision == "APPROVE"
            and item.subject_id == submission_id(spec)
            and item.subject_revision == spec.revision
            and item.spec_hash == spec.spec_hash
        ),
        None,
    )
    if review is None or record.current_spec_hash != spec.spec_hash:
        raise ValueError("training execution approval does not match the frozen spec")
    return record, spec


async def run_baseline(
    state: WorkflowState,
    runtime: Runtime[TrainingDependencies],
) -> WorkflowState:
    """Check cancellation, then execute the approved baseline exactly once."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    record, spec = _approved_spec(dependencies, workflow_id)
    if dependencies.cancellation.is_cancelled():
        return _cancelled(dependencies, record, point="BEFORE_BASELINE")
    try:
        result = await dependencies.trainer.run(
            spec,
            spec.baseline,
            ctx=_ctx(state, dependencies, operation=f"training-{spec.baseline.run_id}"),
        )
        updated = _updated(
            record,
            baseline_result=result.model_dump(mode="json"),
            updated_at=dependencies.clock(),
        )
        record_ref = dependencies.store.write_workflow(updated)
    except TrainingToolError as error:
        return _terminal_failure(
            dependencies,
            record,
            _failure(
                code=error.code,
                message="The controlled baseline training run failed explicitly.",
            ),
        )
    return {
        "report_id": record_ref,
        "run_ids": [result.run_id],
        "phase": WorkflowPhase.RUN_MONITORING,
        "status": WorkflowStatus.RUNNING,
        "route": "CANDIDATE",
        "last_error": None,
    }


async def run_candidate(
    state: WorkflowState,
    runtime: Runtime[TrainingDependencies],
) -> WorkflowState:
    """Check between-run cancellation, then execute the candidate exactly once."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    record, spec = _approved_spec(dependencies, workflow_id)
    if record.baseline_result is None:
        raise ValueError("candidate training requires a completed baseline result")
    if dependencies.cancellation.is_cancelled():
        return _cancelled(dependencies, record, point="BETWEEN_RUNS")
    try:
        result = await dependencies.trainer.run(
            spec,
            spec.candidate,
            ctx=_ctx(state, dependencies, operation=f"training-{spec.candidate.run_id}"),
        )
        updated = _updated(
            record,
            candidate_result=result.model_dump(mode="json"),
            updated_at=dependencies.clock(),
        )
        record_ref = dependencies.store.write_workflow(updated)
    except TrainingToolError as error:
        return _terminal_failure(
            dependencies,
            record,
            _failure(
                code=error.code,
                message="The controlled candidate training run failed explicitly.",
            ),
        )
    return {
        "report_id": record_ref,
        "run_ids": [result.run_id],
        "phase": WorkflowPhase.RUN_MONITORING,
        "status": WorkflowStatus.RUNNING,
        "route": "COLLECT",
        "last_error": None,
    }


async def collect_outputs(
    state: WorkflowState,
    runtime: Runtime[TrainingDependencies],
) -> WorkflowState:
    """Revalidate both structured output sets and complete without evaluation evaluation."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    record, spec = _approved_spec(dependencies, workflow_id)
    if record.baseline_result is None or record.candidate_result is None:
        raise ValueError("output collection requires both completed run results")
    try:
        for run in (spec.baseline, spec.candidate):
            manifest = dependencies.store.load_manifest(run.run_id)
            metrics = dependencies.store.load_metrics(run.run_id)
            predictions = dependencies.store.load_predictions(run.run_id)
            if (
                manifest.spec_hash != spec.spec_hash
                or metrics.spec_hash != spec.spec_hash
                or predictions.spec_hash != spec.spec_hash
            ):
                raise ValueError("collected output spec hashes do not match")
        completed = _updated(
            record,
            status="SUCCEEDED",
            updated_at=dependencies.clock(),
        )
        record_ref = dependencies.store.write_workflow(completed)
    except (OSError, TypeError, ValueError, ValidationError):
        return _terminal_failure(
            dependencies,
            record,
            _failure(
                code="TRAINING_ARTIFACT_INVALID",
                message="Completed training outputs failed final schema collection.",
            ),
        )
    return {
        "report_id": record_ref,
        "phase": WorkflowPhase.COMPLETED,
        "status": WorkflowStatus.SUCCEEDED,
        "route": "COMPLETED",
        "last_error": None,
    }


def route_after_validation(state: WorkflowState) -> Literal["FREEZE", "FAILED"]:
    route = state.get("route")
    if route in {"FREEZE", "FAILED"}:
        return cast(Literal["FREEZE", "FAILED"], route)
    raise ValueError("training validation did not produce a supported route")


def route_after_freeze(state: WorkflowState) -> Literal["GATE", "FAILED"]:
    route = state.get("route")
    if route in {"GATE", "FAILED"}:
        return cast(Literal["GATE", "FAILED"], route)
    raise ValueError("training freeze did not produce a supported route")


def route_after_gate(state: WorkflowState) -> Literal["APPROVED", "EDIT", "REJECTED"]:
    route = state.get("route")
    if route in {"APPROVED", "EDIT", "REJECTED"}:
        return cast(Literal["APPROVED", "EDIT", "REJECTED"], route)
    raise ValueError("training Gate did not produce a supported route")


def route_after_revision(state: WorkflowState) -> Literal["GATE", "FAILED"]:
    return route_after_freeze(state)


def route_after_baseline(state: WorkflowState) -> Literal["CANDIDATE", "CANCELLED", "FAILED"]:
    route = state.get("route")
    if route in {"CANDIDATE", "CANCELLED", "FAILED"}:
        return cast(Literal["CANDIDATE", "CANCELLED", "FAILED"], route)
    raise ValueError("baseline training did not produce a supported route")


def route_after_candidate(state: WorkflowState) -> Literal["COLLECT", "CANCELLED", "FAILED"]:
    route = state.get("route")
    if route in {"COLLECT", "CANCELLED", "FAILED"}:
        return cast(Literal["COLLECT", "CANCELLED", "FAILED"], route)
    raise ValueError("candidate training did not produce a supported route")


def route_after_collect(state: WorkflowState) -> Literal["COMPLETED", "FAILED"]:
    route = state.get("route")
    if route in {"COMPLETED", "FAILED"}:
        return cast(Literal["COMPLETED", "FAILED"], route)
    raise ValueError("training collection did not produce a supported route")


__all__ = [
    "TRAINING_SUBJECT_TYPE",
    "collect_outputs",
    "freeze_submission",
    "revise_submission",
    "route_after_baseline",
    "route_after_candidate",
    "route_after_collect",
    "route_after_freeze",
    "route_after_gate",
    "route_after_revision",
    "route_after_validation",
    "run_baseline",
    "run_candidate",
    "training_gate",
    "validate_input",
]
