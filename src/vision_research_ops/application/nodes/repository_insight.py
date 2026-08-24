"""Outer LangGraph nodes for gated public-repository source insight."""

from __future__ import annotations

import json
from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from vision_research_ops.application.services.repository_models import (
    normalize_github_repository_url,
)
from vision_research_ops.domain import Approval, ApprovalDecision, GateKind
from vision_research_ops.ports import OperationContext, PortError

from ..repository_insight_runtime import RepositoryInsightDependencies, RepositoryInsightState
from ..services.repository_insight_models import RepositoryInsightResult, structure_summary

REPOSITORY_INSIGHT_SUBJECT_TYPE = "public_repository_source_snapshot"


def _required(state: RepositoryInsightState, field: str) -> str:
    value = state.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"repository insight state field {field} must be non-blank")
    return value


def _ctx(
    state: RepositoryInsightState,
    dependencies: RepositoryInsightDependencies,
    operation: str,
) -> OperationContext:
    workflow_id = _required(state, "workflow_id")
    return OperationContext(
        schema_version="1",
        correlation_id=f"corr-{workflow_id}-{operation}",
        workflow_id=workflow_id,
        actor_id=dependencies.actor_id,
        idempotency_key=f"{workflow_id}:{operation}",
        sensitivity="PUBLIC",
    )


async def validate_repository_insight_input(
    state: RepositoryInsightState,
    runtime: Runtime[RepositoryInsightDependencies],
) -> RepositoryInsightState:
    """Canonicalize the public GitHub URL without network, LLM, or artifact access."""
    del runtime
    try:
        canonical = normalize_github_repository_url(_required(state, "repository_url"))
    except ValueError:
        return {
            "status": "FAILED",
            "route": "FAILED",
            "failure_code": "GITHUB_REPOSITORY_URL_INVALID",
            "result_ref": None,
        }
    return {
        "repository_url": canonical.canonical_url,
        "status": "WAITING_FOR_HUMAN",
        "route": "GATE",
        "failure_code": None,
    }


def _revalidate_approval(value: object) -> Approval:
    if isinstance(value, Approval):
        value = value.model_dump(mode="json")
    return Approval.model_validate_json(json.dumps(value, allow_nan=False))


async def repository_snapshot_gate(
    state: RepositoryInsightState,
    runtime: Runtime[RepositoryInsightDependencies],
) -> RepositoryInsightState:
    """Interrupt before every GitHub, snapshot, source-reader, and code-LLM side effect."""
    workflow_id = _required(state, "workflow_id")
    gate_id = _required(state, "gate_id")
    repository_url = _required(state, "repository_url")
    subject_id = f"public-repository-{workflow_id}"
    resume = interrupt(
        {
            "schema_version": "1",
            "gate_id": gate_id,
            "gate_kind": GateKind.REPOSITORY_INGEST.value,
            "subject_type": REPOSITORY_INSIGHT_SUBJECT_TYPE,
            "subject_id": subject_id,
            "subject_revision": 1,
            "repository_url": repository_url,
            "requested_action": "DOWNLOAD_AND_READ_FIXED_COMMIT_SOURCE_SNAPSHOT",
            "notice": (
                "Approve to resolve one full commit SHA and download a bounded read-only ZIP "
                "source snapshot. This is not git clone and no repository code will run."
            ),
        }
    )
    approval = _revalidate_approval(resume)
    if (
        approval.gate_kind is not GateKind.REPOSITORY_INGEST
        or approval.subject_type != REPOSITORY_INSIGHT_SUBJECT_TYPE
        or approval.subject_id != subject_id
        or approval.subject_revision != 1
    ):
        raise ValueError("repository snapshot approval does not target the active gate")
    runtime.context.approval_recorder.record(approval)
    if approval.decision is ApprovalDecision.REJECT:
        return {
            "status": "REJECTED",
            "route": "REJECTED",
            "result_ref": None,
            "failure_code": None,
        }
    if approval.decision is not ApprovalDecision.APPROVE or approval.edits:
        raise ValueError("repository snapshot gate accepts only exact APPROVE or REJECT")
    return {
        "status": "RUNNING",
        "route": "ANALYZE",
        "failure_code": None,
    }


async def analyze_approved_repository_snapshot(
    state: RepositoryInsightState,
    runtime: Runtime[RepositoryInsightDependencies],
) -> RepositoryInsightState:
    """Pin, snapshot, inspect and ask the four-tool LLM only after exact approval."""
    dependencies = runtime.context
    workflow_id = _required(state, "workflow_id")
    repository_url = _required(state, "repository_url")
    ctx = _ctx(state, dependencies, "public-repository-insight")
    try:
        resolution = await dependencies.repository_provider.resolve(
            repository_url,
            None,
            ctx=ctx,
        )
        metadata = await dependencies.repository_provider.fetch_metadata(resolution, ctx=ctx)
        snapshot = await dependencies.repository_provider.snapshot(resolution, ctx=ctx)
        analysis = await dependencies.static_analyzer.analyze(
            snapshot,
            dependencies.policy,
            ctx=ctx,
        )
        structure = structure_summary(analysis)
        source_index = dependencies.source_reader.index(snapshot)
        planner_output = await dependencies.planner.analyze(
            resolution=resolution,
            metadata=metadata,
            snapshot=snapshot,
            source_index=source_index,
            structure=structure,
            source_reader=dependencies.source_reader,
            ctx=ctx,
        )
        result = RepositoryInsightResult(
            workflow_id=workflow_id,
            repository_url=resolution.canonical_url,
            resolution=resolution,
            metadata=metadata,
            snapshot=snapshot,
            source_index_count=len(source_index.files),
            structure=structure,
            advice=planner_output.advice,
            generation=planner_output.generation,
            read_files=planner_output.trace.read_files,
            advice_ref=dependencies.store.advice_ref(workflow_id),
            report_ref=dependencies.store.report_ref(workflow_id),
            trace_ref=dependencies.store.trace_ref(workflow_id),
            result_ref=dependencies.store.result_ref(workflow_id),
        )
        dependencies.store.write_completed(result, planner_output.trace)
    except PortError as error:
        return {
            "status": "FAILED",
            "route": "FAILED",
            "result_ref": None,
            "failure_code": error.failure.code,
        }
    except (OSError, TypeError, ValueError):
        return {
            "status": "FAILED",
            "route": "FAILED",
            "result_ref": None,
            "failure_code": "REPOSITORY_INSIGHT_PROCESSING_FAILED",
        }
    return {
        "status": "COMPLETED",
        "route": "COMPLETED",
        "result_ref": result.result_ref,
        "failure_code": None,
    }


def route_after_input(
    state: RepositoryInsightState,
) -> Literal["GATE", "FAILED"]:
    route = state.get("route")
    if route in {"GATE", "FAILED"}:
        return route
    raise ValueError("repository insight input did not produce a valid route")


def route_after_snapshot_gate(
    state: RepositoryInsightState,
) -> Literal["ANALYZE", "REJECTED"]:
    route = state.get("route")
    if route in {"ANALYZE", "REJECTED"}:
        return route
    raise ValueError("repository insight gate did not produce a valid route")


__all__ = [
    "REPOSITORY_INSIGHT_SUBJECT_TYPE",
    "analyze_approved_repository_snapshot",
    "repository_snapshot_gate",
    "route_after_input",
    "route_after_snapshot_gate",
    "validate_repository_insight_input",
]
