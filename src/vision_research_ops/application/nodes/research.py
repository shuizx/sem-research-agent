"""LangGraph nodes for paper retrieval, structured analysis, and candidate selection."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Literal, cast

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from vision_research_ops.domain import (
    Approval,
    ApprovalDecision,
    GateKind,
    PatchOperationType,
    ResearchRequest,
    WorkflowPhase,
    WorkflowStatus,
)
from vision_research_ops.ports import OperationContext, PortError, make_failure

from ..research_runtime import ResearchDependencies
from ..services.paper_analysis import (
    rank_assessments,
    recommended_paper_ids,
    score_assessments,
    unscored_assessment,
)
from ..services.paper_models import ResearchResult, ResearchWatermark
from ..services.paper_retrieval import (
    collect_provider_records,
    compute_retrieval_window,
    normalize_and_deduplicate,
    query_for_window,
    within_window,
)
from ..state import InitialWorkflowInput, WorkflowState

CANDIDATE_SUBJECT_TYPE = "paper_candidate_slate"


def candidate_slate_id(workflow_id: str) -> str:
    """Return a stable subject ID for the first research candidate slate."""
    return f"candidate-slate-{workflow_id}"


def candidate_gate_id(workflow_id: str) -> str:
    """Return a stable gate ID for candidate-slate revision one."""
    return f"gate-candidate-selection-{workflow_id}-r1"


def _required_text(state: WorkflowState, field: str) -> str:
    value = state.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"workflow state field {field} must be a non-blank string")
    return value


def _ctx(
    state: WorkflowState,
    request: ResearchRequest,
    *,
    operation: str,
) -> OperationContext:
    workflow_id = _required_text(state, "workflow_id")
    return OperationContext(
        schema_version="1",
        correlation_id=f"corr-{workflow_id}-{operation}",
        workflow_id=workflow_id,
        actor_id=request.requested_by,
        idempotency_key=None,
        sensitivity="PUBLIC",
    )


def _failure(error: PortError | None, *, code: str, message: str) -> WorkflowState:
    return {
        "phase": WorkflowPhase.FAILED,
        "status": WorkflowStatus.FAILED,
        "route": "FAILED",
        "pending_gate_id": None,
        "last_error": (
            error.failure
            if error is not None
            else make_failure(
                code=code,
                category="RESEARCH",
                message=message,
                retryable=False,
                ctx=None,
            )
        ),
    }


async def validate_research_request(
    state: WorkflowState,
    runtime: Runtime[ResearchDependencies],
) -> WorkflowState:
    """Validate the initial state against the injected immutable research request."""
    initial = InitialWorkflowInput.model_validate(
        {
            "schema_version": state.get("schema_version"),
            "workflow_id": state.get("workflow_id"),
            "thread_id": state.get("thread_id"),
            "request_id": state.get("request_id"),
            "dataset_profile_id": state.get("dataset_profile_id"),
        }
    )
    request = runtime.context.request
    if request.request_id != initial.request_id:
        raise ValueError("injected ResearchRequest does not match state request_id")
    if request.dataset_id != initial.dataset_profile_id:
        raise ValueError("injected ResearchRequest dataset_id does not match state profile ID")
    return {
        "phase": WorkflowPhase.PAPER_RETRIEVAL,
        "status": WorkflowStatus.RUNNING,
        "route": None,
        "last_error": None,
    }


async def retrieve_papers(
    state: WorkflowState,
    runtime: Runtime[ResearchDependencies],
) -> WorkflowState:
    """Retrieve bounded provider pages using the persisted UTC watermark window."""
    dependencies = runtime.context
    try:
        watermark = dependencies.store.load_watermark()
        now = dependencies.clock()
        window = compute_retrieval_window(
            now=now,
            last_successful_run_at=(
                None if watermark is None else watermark.last_successful_run_at
            ),
            overlap=timedelta(minutes=dependencies.overlap_minutes),
            initial_lookback=timedelta(hours=dependencies.initial_lookback_hours),
        )
        query_spec = query_for_window(dependencies.request.query_spec, window)
        raw_records, pages_used = await collect_provider_records(
            dependencies.paper_provider,
            query_id=f"query-{dependencies.request.request_id}-r{dependencies.request.revision}",
            query_spec=query_spec,
            max_pages=dependencies.request.budget.max_provider_pages,
            max_records=dependencies.request.budget.max_provider_records,
            page_size=dependencies.page_size,
            ctx=_ctx(state, dependencies.request, operation="paper-retrieval"),
        )
    except PortError as error:
        return _failure(
            error,
            code="RESEARCH_PROVIDER_REQUEST_FAILED",
            message="The paper provider request failed.",
        )
    except (OSError, TypeError, ValueError):
        return _failure(
            None,
            code="RESEARCH_RETRIEVAL_INPUT_INVALID",
            message="The research retrieval window or provider data is invalid.",
        )
    dependencies.session.raw_records = raw_records
    dependencies.session.retrieval_window = window
    dependencies.session.query_spec = query_spec
    dependencies.session.watermark_before = (
        None if watermark is None else watermark.last_successful_run_at
    )
    return {
        "budget_used": {
            "provider_pages": float(pages_used),
            "provider_records": float(len(raw_records)),
        },
        "route": "RETRIEVED",
    }


async def deduplicate_papers(
    state: WorkflowState,
    runtime: Runtime[ResearchDependencies],
) -> WorkflowState:
    """Normalize and merge arXiv/DOI/title duplicates, then enforce the exact window."""
    del state
    dependencies = runtime.context
    window = dependencies.session.retrieval_window
    if window is None:
        return _failure(
            None,
            code="RESEARCH_SESSION_WINDOW_MISSING",
            message="The runtime research session is missing its retrieval window.",
        )
    try:
        papers = [
            paper
            for paper in normalize_and_deduplicate(dependencies.session.raw_records)
            if within_window(paper, window)
        ]
    except (TypeError, ValueError):
        return _failure(
            None,
            code="RESEARCH_PAPER_NORMALIZATION_FAILED",
            message="A provider paper record failed deterministic normalization.",
        )
    dependencies.session.papers = papers
    return {
        "paper_candidate_ids": [paper.paper_id for paper in papers],
        "phase": WorkflowPhase.CANDIDATE_RANKING,
        "route": "DEDUPLICATED",
    }


async def hard_filter_candidates(
    state: WorkflowState,
    runtime: Runtime[ResearchDependencies],
) -> WorkflowState:
    """Apply classification/Python/PyTorch eligibility before any LLM call."""
    del state
    dependencies = runtime.context
    dependencies.session.assessments = [
        unscored_assessment(paper, request_id=dependencies.request.request_id)
        for paper in dependencies.session.papers
    ]
    return {"route": "FILTERED"}


async def analyze_applicability(
    state: WorkflowState,
    runtime: Runtime[ResearchDependencies],
) -> WorkflowState:
    """Call the structured LLM only for eligible papers and persist gate evidence."""
    dependencies = runtime.context
    try:
        assessments = await score_assessments(
            dependencies.session.assessments,
            problem=dependencies.problem_profile,
            request_id=dependencies.request.request_id,
            llm=dependencies.structured_llm,
            max_llm_calls=dependencies.request.budget.max_llm_calls,
            ctx=_ctx(state, dependencies.request, operation="paper-applicability"),
        )
    except PortError as error:
        return _failure(
            error,
            code="RESEARCH_LLM_REQUEST_FAILED",
            message="Structured paper applicability analysis failed.",
        )

    ranked = rank_assessments(assessments)
    dependencies.session.assessments = ranked
    recommended = recommended_paper_ids(
        ranked,
        limit=dependencies.request.candidate_limit,
    )
    window = dependencies.session.retrieval_window
    query_spec = dependencies.session.query_spec
    if window is None or query_spec is None:
        return _failure(
            None,
            code="RESEARCH_SESSION_RESULT_MISSING",
            message="The runtime research session is incomplete before result persistence.",
        )
    workflow_id = _required_text(state, "workflow_id")
    now = dependencies.clock()
    gate_id = candidate_gate_id(workflow_id) if recommended else None
    result = ResearchResult(
        workflow_id=workflow_id,
        request_id=dependencies.request.request_id,
        problem_profile=dependencies.problem_profile,
        retrieval_window=window,
        watermark_before=dependencies.session.watermark_before,
        query_spec=query_spec,
        assessments=ranked,
        recommended_paper_ids=recommended,
        selected_paper_ids=[],
        status="AWAITING_SELECTION" if recommended else "NO_CANDIDATES",
        gate_id=gate_id,
        gate_revision=1 if recommended else None,
        created_at=now,
        updated_at=now,
    )
    try:
        result_ref = dependencies.store.write_result(result)
        if not recommended:
            dependencies.store.write_watermark(
                ResearchWatermark(last_successful_run_at=window.end_at)
            )
    except OSError:
        return _failure(
            None,
            code="RESEARCH_OUTPUT_WRITE_FAILED",
            message="The local research result could not be written.",
        )
    llm_calls = sum(item.applicability is not None for item in ranked)
    if not recommended:
        return {
            "report_id": result_ref,
            "phase": WorkflowPhase.COMPLETED,
            "status": WorkflowStatus.SUCCEEDED,
            "pending_gate_id": None,
            "route": "DONE",
            "budget_used": {"llm_calls": float(llm_calls)},
        }
    return {
        "report_id": result_ref,
        "phase": WorkflowPhase.AWAITING_CANDIDATE_SELECTION,
        "status": WorkflowStatus.WAITING_FOR_HUMAN,
        "pending_gate_id": gate_id,
        "route": "GATE",
        "budget_used": {"llm_calls": float(llm_calls)},
    }


def _candidate_summary(result: ResearchResult) -> list[dict[str, object]]:
    by_id = {item.paper.paper_id: item for item in result.assessments}
    summaries: list[dict[str, object]] = []
    for paper_id in result.recommended_paper_ids:
        assessment = by_id[paper_id]
        decision = assessment.applicability
        if decision is None:
            raise ValueError("recommended paper is missing an applicability decision")
        summaries.append(
            {
                "paper_id": paper_id,
                "title": assessment.paper.title[:200],
                "arxiv_id": assessment.paper.arxiv_id,
                "relevance_score": decision.relevance_score,
                "recommendation": decision.recommendation,
                "evidence": [item.statement[:240] for item in decision.evidence[:2]],
                "risks": [risk[:240] for risk in decision.risks[:2]],
                "code_urls": list(assessment.paper.code_urls[:2]),
            }
        )
    return summaries


def _revalidate_approval(value: object) -> Approval:
    if isinstance(value, Approval):
        value = value.model_dump(mode="json")
    return Approval.model_validate_json(json.dumps(value, allow_nan=False))


def _edited_selection(approval: Approval, allowed: list[str]) -> list[str]:
    if len(approval.edits) != 1:
        raise ValueError("candidate EDIT requires exactly one structured operation")
    operation = approval.edits[0]
    if operation.op is not PatchOperationType.REPLACE or operation.path != "/selected_paper_ids":
        raise ValueError("candidate EDIT must replace /selected_paper_ids")
    value = operation.value
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError("selected_paper_ids must be a list of non-blank strings")
    selected = list(dict.fromkeys(cast(list[str], value)))
    if not selected or any(item not in allowed for item in selected):
        raise ValueError("edited selection must be a non-empty subset of recommended papers")
    return selected


async def candidate_selection_gate(
    state: WorkflowState,
    runtime: Runtime[ResearchDependencies],
) -> WorkflowState:
    """Interrupt and bind approve/edit/reject to the exact candidate slate revision."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    result = dependencies.store.load_result(workflow_id)
    gate_id = candidate_gate_id(workflow_id)
    subject_id = candidate_slate_id(workflow_id)
    if state.get("pending_gate_id") != gate_id or result.gate_id != gate_id:
        raise ValueError("candidate gate does not match the persisted research result")
    resume_value = interrupt(
        {
            "schema_version": "1",
            "gate_id": gate_id,
            "gate_kind": GateKind.CANDIDATE_SELECTION.value,
            "subject_type": CANDIDATE_SUBJECT_TYPE,
            "subject_id": subject_id,
            "subject_revision": 1,
            "recommended_papers": _candidate_summary(result),
        }
    )
    approval = _revalidate_approval(resume_value)
    if approval.gate_kind is not GateKind.CANDIDATE_SELECTION:
        raise ValueError("approval gate_kind is not CANDIDATE_SELECTION")
    if approval.subject_type != CANDIDATE_SUBJECT_TYPE:
        raise ValueError("approval subject_type is not paper_candidate_slate")
    if approval.subject_id != subject_id or approval.subject_revision != 1:
        raise ValueError("approval does not target the current candidate slate revision")
    dependencies.approval_recorder.record(approval)

    if approval.decision is ApprovalDecision.APPROVE:
        selected = list(result.recommended_paper_ids)
    elif approval.decision is ApprovalDecision.EDIT:
        selected = _edited_selection(approval, result.recommended_paper_ids)
    elif approval.decision is ApprovalDecision.REJECT:
        selected = []
    else:
        raise ValueError("unsupported candidate approval decision")
    return {
        "selected_paper_ids": selected,
        "pending_gate_id": None,
        "status": WorkflowStatus.RUNNING,
        "route": approval.decision.value,
    }


async def finalize_research_selection(
    state: WorkflowState,
    runtime: Runtime[ResearchDependencies],
) -> WorkflowState:
    """Persist human selection and advance the watermark after a completed gate."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    result = dependencies.store.load_result(workflow_id)
    route = state.get("route")
    if route not in {"APPROVE", "EDIT", "REJECT"}:
        raise ValueError("candidate gate did not produce a supported route")
    selected = [] if route == "REJECT" else list(state.get("selected_paper_ids", []))
    selected_set = set(selected)
    updated_assessments = [
        item.model_copy(
            update={
                "selected": item.paper.paper_id in selected_set,
                "candidate": item.candidate.model_copy(
                    update={"selected": item.paper.paper_id in selected_set}
                ),
            }
        )
        for item in result.assessments
    ]
    updated = result.model_copy(
        update={
            "assessments": updated_assessments,
            "selected_paper_ids": selected,
            "status": "REJECTED" if route == "REJECT" else "COMPLETED",
            "updated_at": dependencies.clock(),
        }
    )
    try:
        result_ref = dependencies.store.write_result(updated)
        dependencies.store.write_watermark(
            ResearchWatermark(last_successful_run_at=result.retrieval_window.end_at)
        )
    except OSError:
        return _failure(
            None,
            code="RESEARCH_OUTPUT_WRITE_FAILED",
            message="The final local research result could not be written.",
        )
    if route == "REJECT":
        return {
            "report_id": result_ref,
            "phase": WorkflowPhase.REJECTED,
            "status": WorkflowStatus.REJECTED,
            "route": "REJECTED",
            "last_error": None,
        }
    return {
        "report_id": result_ref,
        "phase": WorkflowPhase.COMPLETED,
        "status": WorkflowStatus.SUCCEEDED,
        "route": "COMPLETED",
        "last_error": None,
    }


def route_after_analysis(state: WorkflowState) -> Literal["GATE", "DONE", "FAILED"]:
    """Route persisted results to a gate, success end, or explicit failure end."""
    route = state.get("route")
    if route in {"GATE", "DONE", "FAILED"}:
        return cast(Literal["GATE", "DONE", "FAILED"], route)
    raise ValueError("applicability analysis did not produce a supported route")


def route_after_retrieval(state: WorkflowState) -> Literal["CONTINUE", "FAILED"]:
    """Stop on provider/window failure; otherwise continue to normalization."""
    route = state.get("route")
    if route == "FAILED":
        return "FAILED"
    if route == "RETRIEVED":
        return "CONTINUE"
    raise ValueError("paper retrieval did not produce a supported route")


def route_after_deduplication(state: WorkflowState) -> Literal["CONTINUE", "FAILED"]:
    """Stop on normalization failure; otherwise continue to the hard filter."""
    route = state.get("route")
    if route == "FAILED":
        return "FAILED"
    if route == "DEDUPLICATED":
        return "CONTINUE"
    raise ValueError("paper deduplication did not produce a supported route")


__all__ = [
    "CANDIDATE_SUBJECT_TYPE",
    "analyze_applicability",
    "candidate_gate_id",
    "candidate_selection_gate",
    "candidate_slate_id",
    "deduplicate_papers",
    "finalize_research_selection",
    "hard_filter_candidates",
    "retrieve_papers",
    "route_after_analysis",
    "route_after_deduplication",
    "route_after_retrieval",
    "validate_research_request",
]
