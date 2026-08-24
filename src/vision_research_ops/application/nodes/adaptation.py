"""LangGraph nodes for structured planning, bounded patching, smoke, and Gate."""

from __future__ import annotations

import json
from contextlib import suppress
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
from vision_research_ops.ports import OperationContext, PortError, make_failure

from ..adaptation_runtime import AdaptationDependencies
from ..services.adaptation_models import (
    AdaptationInputFacts,
    AdaptationResult,
    AttemptEvidence,
    PatchReviewRecord,
)
from ..services.adaptation_patch import PatchPolicyError
from ..services.adaptation_planning import (
    apply_human_plan_edits,
    compile_adaptation_plan,
    public_plan_summary,
    validate_adaptation_inputs,
)
from ..services.adaptation_repair import repair_failed_fixture_plan
from ..state import InitialWorkflowInput, WorkflowState

PATCH_SUBJECT_TYPE = "adaptation_patch"


def _required_text(state: WorkflowState, field: str) -> str:
    value = state.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"workflow state field {field} must be a non-blank string")
    return value


def _ctx(
    state: WorkflowState,
    dependencies: AdaptationDependencies,
    *,
    operation: str,
    sensitivity: Literal["PUBLIC", "INTERNAL", "RESTRICTED"] = "INTERNAL",
) -> OperationContext:
    workflow_id = _required_text(state, "workflow_id")
    return OperationContext(
        schema_version="1",
        correlation_id=f"corr-{workflow_id}-{operation}",
        workflow_id=workflow_id,
        actor_id=dependencies.actor_id,
        idempotency_key=f"{workflow_id}:{operation}",
        sensitivity=sensitivity,
    )


def _failure(*, code: str, message: str, retryable: bool = False) -> StructuredFailure:
    return make_failure(
        code=code,
        category="ADAPTATION",
        message=message,
        retryable=retryable,
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


def patch_subject_id(workflow_id: str, patch_hash: str) -> str:
    """Bind the Gate subject identity to the exact patch content hash."""
    digest = patch_hash.removeprefix("sha256:")
    if len(digest) != 64:
        raise ValueError("patch hash must be a canonical SHA-256 content hash")
    return f"adaptation-patch-{workflow_id}-{digest[:20]}"


def patch_gate_id(workflow_id: str, revision: int, patch_hash: str) -> str:
    """Return a Gate ID that visibly binds revision and patch hash."""
    subject = patch_subject_id(workflow_id, patch_hash)
    return f"gate-patch-acceptance-{subject}-r{revision}"


def _load_facts(dependencies: AdaptationDependencies) -> AdaptationInputFacts:
    repository = dependencies.repository_store.load_result(dependencies.repository_workflow_id)
    return validate_adaptation_inputs(repository, dependencies.dataset_profile)


def _terminal_failure(
    dependencies: AdaptationDependencies,
    result: AdaptationResult,
    failure: StructuredFailure,
) -> WorkflowState:
    dependencies.store.write_result(
        result.model_copy(
            update={
                "status": "FAILED",
                "failure": failure,
                "gate_id": None,
                "gate_revision": None,
                "gate_subject_id": None,
                "gate_patch_hash": None,
                "updated_at": dependencies.clock(),
            }
        )
    )
    return _failure_state(failure)


async def load_and_compare_inputs(
    state: WorkflowState,
    runtime: Runtime[AdaptationDependencies],
) -> WorkflowState:
    """Validate exact repository and synthetic dataset preconditions before any LLM or tool."""
    dependencies = runtime.context
    initial: InitialWorkflowInput | None = None
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
        if initial.dataset_profile_id != dependencies.dataset_profile.dataset_id:
            raise ValueError("state dataset_profile_id does not match the injected profile")
        facts = _load_facts(dependencies)
        now = dependencies.clock()
        result = AdaptationResult(
            workflow_id=initial.workflow_id,
            request_id=initial.request_id,
            repository_workflow_id=dependencies.repository_workflow_id,
            status="INPUT_VALIDATED",
            repository_id=facts.repository_id,
            repository_url=facts.repository_url,
            base_commit_sha=facts.base_commit_sha,
            dataset_id=facts.dataset_id,
            dataset_version=facts.dataset_version,
            dataset_content_hash=facts.dataset_content_hash,
            dataset_kind=facts.dataset_kind,
            repository_kind=facts.repository_kind,
            created_at=now,
            updated_at=now,
        )
        result_ref = dependencies.store.write_result(result)
    except (OSError, TypeError, ValueError, ValidationError):
        failure = _failure(
            code="ADAPTATION_INPUT_INVALID",
            message=(
                "Adaptation requires a completed supported repository fixture and an authorized "
                "de-identified synthetic DatasetProfile."
            ),
        )
        if (
            initial is not None
            and dependencies.dataset_profile.authorization.get("source_kind") == "SYNTHETIC"
        ):
            now = dependencies.clock()
            failed_result = AdaptationResult(
                workflow_id=initial.workflow_id,
                request_id=initial.request_id,
                repository_workflow_id=dependencies.repository_workflow_id,
                status="FAILED",
                dataset_id=dependencies.dataset_profile.dataset_id,
                dataset_version=dependencies.dataset_profile.version,
                dataset_content_hash=dependencies.dataset_profile.content_hash,
                dataset_kind="SYNTHETIC_SEM_FIXTURE",
                failure=failure,
                created_at=now,
                updated_at=now,
            )
            with suppress(OSError, ValueError):
                dependencies.store.write_result(failed_result)
        return _failure_state(failure)
    return {
        "report_id": result_ref,
        "active_repository_id": facts.repository_id,
        "phase": WorkflowPhase.ADAPTATION_PLANNING,
        "status": WorkflowStatus.RUNNING,
        "route": "PLAN",
        "last_error": None,
    }


async def plan_adaptation(
    state: WorkflowState,
    runtime: Runtime[AdaptationDependencies],
) -> WorkflowState:
    """Obtain one strict LLM proposal and compile it against deterministic facts."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    result = dependencies.store.load_result(workflow_id)
    if result.status != "INPUT_VALIDATED":
        raise ValueError("adaptation planning requires validated persisted inputs")
    ctx = _ctx(state, dependencies, operation="adaptation-plan")
    try:
        facts = _load_facts(dependencies)
        planner_output = await dependencies.planner.plan(facts, ctx=ctx)
        plan = compile_adaptation_plan(
            workflow_id=workflow_id,
            facts=facts,
            generation_result=planner_output.generation,
            now=dependencies.clock(),
        )
        planner_trace_ref = dependencies.store.write_planner_trace(planner_output.trace)
        plan_ref = dependencies.store.write_plan(plan)
        result_ref = dependencies.store.write_result(
            result.model_copy(
                update={
                    "status": "PLANNED",
                    "generation": plan.generation,
                    "gaps": plan.proposal.gaps,
                    "changes": plan.proposal.changes,
                    "plan_id": plan.plan_id,
                    "plan_revision": plan.revision,
                    "plan_ref": plan_ref,
                    "planner_trace_ref": planner_trace_ref,
                    "updated_at": dependencies.clock(),
                }
            )
        )
    except PortError as error:
        return _terminal_failure(dependencies, result, error.failure)
    except (OSError, TypeError, ValueError, ValidationError):
        failure = _failure(
            code="ADAPTATION_PLAN_INVALID",
            message="The schema-bound adaptation proposal failed deterministic validation.",
        )
        return _terminal_failure(dependencies, result, failure)
    return {
        "report_id": result_ref,
        "active_plan_id": plan.plan_id,
        "phase": WorkflowPhase.PATCH_GENERATION,
        "status": WorkflowStatus.RUNNING,
        "route": "PATCH",
        "last_error": None,
    }


def _upsert_attempt(
    attempts: list[AttemptEvidence],
    update: AttemptEvidence,
) -> list[AttemptEvidence]:
    return [item for item in attempts if item.attempt_id != update.attempt_id] + [update]


async def generate_bounded_patch(
    state: WorkflowState,
    runtime: Runtime[AdaptationDependencies],
) -> WorkflowState:
    """Apply the deterministic allowlisted template through the injected patch tool."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    result = dependencies.store.load_result(workflow_id)
    if result.status != "PLANNED" or result.plan_ref is None:
        raise ValueError("patch generation requires a persisted compiled plan")
    try:
        plan = dependencies.store.load_plan(result.plan_ref)
        patch = await dependencies.patch_tool.apply(
            plan,
            ctx=_ctx(state, dependencies, operation=f"adaptation-patch-r{plan.revision}"),
        )
        attempt = AttemptEvidence(
            attempt_id=patch.attempt_id,
            plan_revision=patch.plan_revision,
            patch_hash=patch.patch_hash,
            patch_ref=patch.patch_ref,
            patch_manifest_ref=patch.manifest_ref,
        )
        result_ref = dependencies.store.write_result(
            result.model_copy(
                update={
                    "status": "PATCHED",
                    "attempts": _upsert_attempt(result.attempts, attempt),
                    "gate_id": None,
                    "gate_revision": None,
                    "gate_subject_id": None,
                    "gate_patch_hash": None,
                    "accepted_patch_hash": None,
                    "approval_id": None,
                    "updated_at": dependencies.clock(),
                }
            )
        )
    except PortError as error:
        return _terminal_failure(dependencies, result, error.failure)
    except (OSError, PatchPolicyError, TypeError, ValueError, ValidationError):
        failure = _failure(
            code="ADAPTATION_PATCH_POLICY_FAILED",
            message="The deterministic patch could not satisfy the fixed fixture policy.",
        )
        return _terminal_failure(dependencies, result, failure)
    return {
        "report_id": result_ref,
        "active_attempt_id": patch.attempt_id,
        "phase": WorkflowPhase.PATCH_VALIDATION,
        "status": WorkflowStatus.RUNNING,
        "route": "SMOKE",
        "last_error": None,
    }


async def run_bounded_smoke(
    state: WorkflowState,
    runtime: Runtime[AdaptationDependencies],
) -> WorkflowState:
    """Run actual controlled fixture stages and route to Gate, one repair, or failure."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    attempt_id = _required_text(state, "active_attempt_id")
    result = dependencies.store.load_result(workflow_id)
    if result.status != "PATCHED":
        raise ValueError("smoke requires a persisted patch attempt")
    attempt = next((item for item in result.attempts if item.attempt_id == attempt_id), None)
    if attempt is None:
        raise ValueError("active patch attempt is absent from adaptation evidence")
    try:
        patch = dependencies.store.load_patch_record(attempt.patch_manifest_ref)
        smoke = await dependencies.smoke_tool.run(
            patch,
            ctx=_ctx(state, dependencies, operation=f"adaptation-smoke-r{patch.plan_revision}"),
        )
        updated_attempt = attempt.model_copy(
            update={"smoke_ref": smoke.result_ref, "smoke_status": smoke.status}
        )
        attempts = _upsert_attempt(result.attempts, updated_attempt)
        if smoke.status == "PASSED":
            gate_id = patch_gate_id(workflow_id, patch.plan_revision, patch.patch_hash)
            subject_id = patch_subject_id(workflow_id, patch.patch_hash)
            result_ref = dependencies.store.write_result(
                result.model_copy(
                    update={
                        "status": "AWAITING_APPROVAL",
                        "attempts": attempts,
                        "gate_id": gate_id,
                        "gate_revision": patch.plan_revision,
                        "gate_subject_id": subject_id,
                        "gate_patch_hash": patch.patch_hash,
                        "updated_at": dependencies.clock(),
                    }
                )
            )
            return {
                "report_id": result_ref,
                "pending_gate_id": gate_id,
                "phase": WorkflowPhase.PATCH_VALIDATION,
                "status": WorkflowStatus.WAITING_FOR_HUMAN,
                "route": "GATE",
                "last_error": None,
                "validation_result_ids": [f"smoke-{attempt_id}"],
            }

        if result.repair_count == 0 and smoke.retryable:
            observed_failure = _failure(
                code="ADAPTATION_SMOKE_FAILED_REPAIRABLE",
                message="The controlled fixture smoke failed and may use its single repair.",
                retryable=True,
            )
            result_ref = dependencies.store.write_result(
                result.model_copy(
                    update={
                        "status": "REPAIRING",
                        "attempts": attempts,
                        "updated_at": dependencies.clock(),
                    }
                )
            )
            return {
                "report_id": result_ref,
                "phase": WorkflowPhase.PATCH_VALIDATION,
                "status": WorkflowStatus.RUNNING,
                "route": "REPAIR",
                "last_error": observed_failure,
                "validation_result_ids": [f"smoke-{attempt_id}"],
            }
        terminal = _failure(
            code="ADAPTATION_SMOKE_FAILED_AFTER_REPAIR",
            message="The controlled fixture smoke failed after the single repair opportunity.",
        )
        terminal_result = result.model_copy(update={"attempts": attempts})
        return _terminal_failure(dependencies, terminal_result, terminal)
    except PortError as error:
        return _terminal_failure(dependencies, result, error.failure)
    except (OSError, TypeError, ValueError, ValidationError):
        failure = _failure(
            code="ADAPTATION_SMOKE_TOOL_FAILED",
            message="The bounded fixture smoke tool failed explicitly.",
        )
        return _terminal_failure(dependencies, result, failure)


async def repair_patch_once(
    state: WorkflowState,
    runtime: Runtime[AdaptationDependencies],
) -> WorkflowState:
    """Apply the sole deterministic repair and require a new patch hash and smoke."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    result = dependencies.store.load_result(workflow_id)
    if result.status != "REPAIRING" or result.repair_count != 0 or result.plan_ref is None:
        raise ValueError("repair is not authorized for the current adaptation state")
    failed_attempt = next(
        (
            item
            for item in reversed(result.attempts)
            if item.smoke_status == "FAILED" and item.smoke_ref is not None
        ),
        None,
    )
    if failed_attempt is None or failed_attempt.smoke_ref is None:
        raise ValueError("repair requires an exact failed smoke artifact")
    try:
        plan = dependencies.store.load_plan(result.plan_ref)
        smoke = dependencies.store.load_smoke_result(failed_attempt.smoke_ref)
        repaired = repair_failed_fixture_plan(plan, smoke, now=dependencies.clock())
        plan_ref = dependencies.store.write_plan(repaired)
        result_ref = dependencies.store.write_result(
            result.model_copy(
                update={
                    "status": "PLANNED",
                    "plan_revision": repaired.revision,
                    "plan_ref": plan_ref,
                    "repair_count": 1,
                    "updated_at": dependencies.clock(),
                }
            )
        )
    except (OSError, TypeError, ValueError, ValidationError):
        failure = _failure(
            code="ADAPTATION_REPAIR_FAILED",
            message="The single deterministic fixture repair could not be compiled.",
        )
        return _terminal_failure(dependencies, result, failure)
    return {
        "report_id": result_ref,
        "active_plan_id": repaired.plan_id,
        "active_attempt_id": None,
        "retry_counts": {"adaptation_repair": 1},
        "phase": WorkflowPhase.PATCH_GENERATION,
        "status": WorkflowStatus.RUNNING,
        "route": "PATCH",
        "last_error": None,
    }


def _revalidate_approval(value: object) -> Approval:
    if isinstance(value, Approval):
        value = value.model_dump(mode="json")
    return Approval.model_validate_json(json.dumps(value, allow_nan=False))


def _review_record(result: AdaptationResult, approval: Approval) -> PatchReviewRecord:
    if (
        result.gate_id is None
        or result.gate_subject_id is None
        or result.gate_revision is None
        or result.gate_patch_hash is None
    ):
        raise ValueError("patch review requires complete Gate binding evidence")
    return PatchReviewRecord(
        approval_id=approval.approval_id,
        decision=approval.decision.value,
        gate_id=result.gate_id,
        subject_id=result.gate_subject_id,
        subject_revision=result.gate_revision,
        patch_hash=result.gate_patch_hash,
        actor_id=approval.actor_id,
        decided_at=approval.decided_at,
    )


async def patch_acceptance_gate(
    state: WorkflowState,
    runtime: Runtime[AdaptationDependencies],
) -> WorkflowState:
    """Interrupt on an exact smoke-passed patch and handle approve/edit/reject."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    result = dependencies.store.load_result(workflow_id)
    if (
        result.status != "AWAITING_APPROVAL"
        or result.gate_id is None
        or result.gate_revision is None
        or result.gate_subject_id is None
        or result.gate_patch_hash is None
        or result.plan_ref is None
    ):
        raise ValueError("patch Gate requires exact persisted smoke-passed evidence")
    if state.get("pending_gate_id") != result.gate_id:
        raise ValueError("patch Gate does not match persisted adaptation evidence")
    current_attempt = next(
        (item for item in reversed(result.attempts) if item.plan_revision == result.gate_revision),
        None,
    )
    if (
        current_attempt is None
        or current_attempt.patch_hash != result.gate_patch_hash
        or current_attempt.smoke_status != "PASSED"
        or current_attempt.smoke_ref is None
    ):
        raise ValueError("patch Gate cannot proceed without exact passed smoke evidence")
    plan = dependencies.store.load_plan(result.plan_ref)
    smoke = dependencies.store.load_smoke_result(current_attempt.smoke_ref)
    if smoke.patch_hash != result.gate_patch_hash or smoke.status != "PASSED":
        raise ValueError("patch Gate smoke evidence does not match the current patch")

    resume_value = interrupt(
        {
            "schema_version": "1",
            "gate_id": result.gate_id,
            "gate_kind": GateKind.PATCH_ACCEPTANCE.value,
            "subject_type": PATCH_SUBJECT_TYPE,
            "subject_id": result.gate_subject_id,
            "subject_revision": result.gate_revision,
            "patch_hash": result.gate_patch_hash,
            "base_commit_sha": result.base_commit_sha,
            "repository_id": result.repository_id,
            "dataset_version": result.dataset_version,
            "plan": dict(public_plan_summary(plan)),
            "patch_ref": current_attempt.patch_ref,
            "smoke_ref": current_attempt.smoke_ref,
            "smoke_capability_boundary": smoke.capability_boundary,
            "real_pytorch_training": smoke.real_pytorch_training,
            "editable_fields": [
                "/channels",
                "/num_classes",
                "/label_mapping",
                "/group_split_key",
                "/metrics",
                "/metrics_output_file",
            ],
        }
    )
    approval = _revalidate_approval(resume_value)
    if approval.gate_kind is not GateKind.PATCH_ACCEPTANCE:
        raise ValueError("approval gate_kind is not PATCH_ACCEPTANCE")
    if approval.subject_type != PATCH_SUBJECT_TYPE:
        raise ValueError("approval subject_type is not adaptation_patch")
    if (
        approval.subject_id != result.gate_subject_id
        or approval.subject_revision != result.gate_revision
        or patch_subject_id(workflow_id, result.gate_patch_hash) != approval.subject_id
    ):
        raise ValueError("approval does not target the current patch revision and hash")

    if approval.decision is ApprovalDecision.REJECT:
        dependencies.approval_recorder.record(approval)
        review = _review_record(result, approval)
        result_ref = dependencies.store.write_result(
            result.model_copy(
                update={
                    "status": "REJECTED",
                    "reviews": [*result.reviews, review],
                    "approval_id": approval.approval_id,
                    "updated_at": dependencies.clock(),
                }
            )
        )
        return {
            "report_id": result_ref,
            "pending_gate_id": None,
            "phase": WorkflowPhase.REJECTED,
            "status": WorkflowStatus.REJECTED,
            "route": "REJECTED",
            "last_error": None,
        }
    if approval.decision is ApprovalDecision.APPROVE:
        dependencies.approval_recorder.record(approval)
        review = _review_record(result, approval)
        result_ref = dependencies.store.write_result(
            result.model_copy(
                update={
                    "status": "ACCEPTED",
                    "reviews": [*result.reviews, review],
                    "accepted_patch_hash": result.gate_patch_hash,
                    "approval_id": approval.approval_id,
                    "updated_at": dependencies.clock(),
                }
            )
        )
        return {
            "report_id": result_ref,
            "pending_gate_id": None,
            "phase": WorkflowPhase.COMPLETED,
            "status": WorkflowStatus.SUCCEEDED,
            "route": "ACCEPTED",
            "last_error": None,
        }

    facts = _load_facts(dependencies)
    edited = apply_human_plan_edits(plan, approval, facts, now=dependencies.clock())
    plan_ref = dependencies.store.write_plan(edited)
    dependencies.approval_recorder.record(approval)
    review = _review_record(result, approval)
    result_ref = dependencies.store.write_result(
        result.model_copy(
            update={
                "status": "PLANNED",
                "gaps": edited.proposal.gaps,
                "changes": edited.proposal.changes,
                "plan_revision": edited.revision,
                "plan_ref": plan_ref,
                "reviews": [*result.reviews, review],
                "gate_id": None,
                "gate_revision": None,
                "gate_subject_id": None,
                "gate_patch_hash": None,
                "accepted_patch_hash": None,
                "approval_id": None,
                "updated_at": dependencies.clock(),
            }
        )
    )
    return {
        "report_id": result_ref,
        "pending_gate_id": None,
        "active_plan_id": edited.plan_id,
        "active_attempt_id": None,
        "phase": WorkflowPhase.PATCH_GENERATION,
        "status": WorkflowStatus.RUNNING,
        "route": "EDIT",
        "last_error": None,
    }


def route_after_input_compare(state: WorkflowState) -> Literal["PLAN", "FAILED"]:
    """Route valid input to LLM planning and stop ordinary invalid input."""
    route = state.get("route")
    if route in {"PLAN", "FAILED"}:
        return cast(Literal["PLAN", "FAILED"], route)
    raise ValueError("input comparison did not produce a supported route")


def route_after_plan(state: WorkflowState) -> Literal["PATCH", "FAILED"]:
    """Route a compiled plan to patching or terminate explicit LLM failure."""
    route = state.get("route")
    if route in {"PATCH", "FAILED"}:
        return cast(Literal["PATCH", "FAILED"], route)
    raise ValueError("adaptation planning did not produce a supported route")


def route_after_patch(state: WorkflowState) -> Literal["SMOKE", "FAILED"]:
    """Route a bounded patch to smoke or stop policy failure."""
    route = state.get("route")
    if route in {"SMOKE", "FAILED"}:
        return cast(Literal["SMOKE", "FAILED"], route)
    raise ValueError("patch generation did not produce a supported route")


def route_after_smoke(state: WorkflowState) -> Literal["GATE", "REPAIR", "FAILED"]:
    """Route actual smoke evidence through the single repair bound."""
    route = state.get("route")
    if route in {"GATE", "REPAIR", "FAILED"}:
        return cast(Literal["GATE", "REPAIR", "FAILED"], route)
    raise ValueError("bounded smoke did not produce a supported route")


def route_after_patch_gate(state: WorkflowState) -> Literal["ACCEPTED", "EDIT", "REJECTED"]:
    """Route the exact human patch decision without invoking training training."""
    route = state.get("route")
    if route in {"ACCEPTED", "EDIT", "REJECTED"}:
        return cast(Literal["ACCEPTED", "EDIT", "REJECTED"], route)
    raise ValueError("patch Gate did not produce a supported route")


__all__ = [
    "PATCH_SUBJECT_TYPE",
    "generate_bounded_patch",
    "load_and_compare_inputs",
    "patch_acceptance_gate",
    "patch_gate_id",
    "patch_subject_id",
    "plan_adaptation",
    "repair_patch_once",
    "route_after_input_compare",
    "route_after_patch",
    "route_after_patch_gate",
    "route_after_plan",
    "route_after_smoke",
    "run_bounded_smoke",
]
