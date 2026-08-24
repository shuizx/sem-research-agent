"""LangGraph nodes for repository evidence, approval, and static profiling."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Literal, cast

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from vision_research_ops.domain import (
    Approval,
    ApprovalDecision,
    CodeLinkConfidence,
    CodeLinkEvidence,
    GateKind,
    LicenseStatus,
    PatchOperationType,
    ProvenanceRef,
    Reason,
    RepositorySnapshot,
    RiskFinding,
    SeverityLevel,
    StructuredFailure,
    WorkflowPhase,
    WorkflowStatus,
)
from vision_research_ops.ports import OperationContext, PortError, make_failure

from ..repository_runtime import RepositoryDependencies
from ..services.repository_models import (
    RepositoryProfile,
    RepositoryResult,
    normalize_github_repository_url,
)
from ..state import InitialWorkflowInput, WorkflowState

REPOSITORY_SUBJECT_TYPE = "repository_ingest_candidate"
_ALLOWED_LICENSES = {"Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MIT"}
_DENIED_LICENSES = {"Proprietary", "UNLICENSED"}


def repository_subject_id(workflow_id: str) -> str:
    """Return the stable human-gate subject for a repository workflow."""
    return f"repository-candidate-{workflow_id}"


def repository_gate_id(workflow_id: str) -> str:
    """Return the stable first revision Gate ID."""
    return f"gate-repository-ingest-{workflow_id}-r1"


def _required_text(state: WorkflowState, field: str) -> str:
    value = state.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"workflow state field {field} must be a non-blank string")
    return value


def _ctx(
    state: WorkflowState,
    dependencies: RepositoryDependencies,
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
        sensitivity="PUBLIC",
    )


def _failure_state(*, failure: StructuredFailure) -> WorkflowState:
    return {
        "phase": WorkflowPhase.FAILED,
        "status": WorkflowStatus.FAILED,
        "pending_gate_id": None,
        "route": "FAILED",
        "last_error": failure,
    }


def _generic_failure(*, code: str, message: str) -> StructuredFailure:
    return make_failure(
        code=code,
        category="REPOSITORY",
        message=message,
        retryable=False,
        ctx=None,
    )


def _record_failed_result(
    dependencies: RepositoryDependencies,
    result: RepositoryResult,
    failure: StructuredFailure,
) -> None:
    dependencies.store.write_result(
        result.model_copy(
            update={
                "status": "FAILED",
                "failure": failure,
                "updated_at": dependencies.clock(),
            }
        )
    )


async def prepare_repository_candidate(
    state: WorkflowState,
    runtime: Runtime[RepositoryDependencies],
) -> WorkflowState:
    """Load the selected research paper and persist one canonical GitHub candidate."""
    dependencies = runtime.context
    initial = InitialWorkflowInput.model_validate(
        {
            "schema_version": state.get("schema_version"),
            "workflow_id": state.get("workflow_id"),
            "thread_id": state.get("thread_id"),
            "request_id": state.get("request_id"),
            "dataset_profile_id": state.get("dataset_profile_id"),
        }
    )
    try:
        research = dependencies.research_store.load_result(dependencies.research_workflow_id)
        if research.status != "COMPLETED":
            raise ValueError("repository ingestion requires a completed research result")
        if dependencies.selected_paper_id not in research.selected_paper_ids:
            raise ValueError("repository paper must have been selected at the research human gate")
        assessment = next(
            item
            for item in research.assessments
            if item.paper.paper_id == dependencies.selected_paper_id
        )
        candidates = []
        for value in assessment.paper.code_urls:
            try:
                locator = normalize_github_repository_url(value)
            except ValueError:
                continue
            if locator.canonical_url not in candidates:
                candidates.append(locator.canonical_url)
        if not candidates:
            raise ValueError("selected paper has no allowed public GitHub repository link")
        requested_url = candidates[0]
        now = dependencies.clock()
        evidence = CodeLinkEvidence(
            schema_version="1",
            evidence_id=f"code-link-{assessment.paper.paper_id}",
            paper_id=assessment.paper.paper_id,
            repository_url=requested_url,
            evidence_type="paper_link",
            confidence=CodeLinkConfidence.OFFICIAL_HIGH,
            rationale_codes=["PAPER_METADATA_PUBLIC_CODE_LINK"],
            provenance=assessment.paper.provenance[0],
            verified_at=now,
        )
        gate_id = repository_gate_id(initial.workflow_id)
        result = RepositoryResult(
            workflow_id=initial.workflow_id,
            request_id=initial.request_id,
            research_workflow_id=dependencies.research_workflow_id,
            paper_id=assessment.paper.paper_id,
            requested_repository_url=requested_url,
            code_link_evidence=evidence,
            status="AWAITING_APPROVAL",
            gate_id=gate_id,
            gate_revision=1,
            created_at=now,
            updated_at=now,
        )
        result_ref = dependencies.store.write_result(result)
    except (OSError, StopIteration, ValueError):
        failure = _generic_failure(
            code="REPOSITORY_CANDIDATE_INVALID",
            message="The selected paper did not provide a valid repository candidate.",
        )
        return _failure_state(failure=failure)
    return {
        "phase": WorkflowPhase.AWAITING_INGEST_APPROVAL,
        "status": WorkflowStatus.WAITING_FOR_HUMAN,
        "selected_paper_ids": [dependencies.selected_paper_id],
        "report_id": result_ref,
        "pending_gate_id": gate_id,
        "route": "GATE",
        "last_error": None,
    }


def _revalidate_approval(value: object) -> Approval:
    if isinstance(value, Approval):
        value = value.model_dump(mode="json")
    return Approval.model_validate_json(json.dumps(value, allow_nan=False))


def _edited_repository_url(approval: Approval) -> str:
    if len(approval.edits) != 1:
        raise ValueError("repository EDIT requires exactly one structured operation")
    operation = approval.edits[0]
    if operation.op is not PatchOperationType.REPLACE or operation.path != "/repository_url":
        raise ValueError("repository EDIT must replace /repository_url")
    if not isinstance(operation.value, str):
        raise ValueError("edited repository URL must be a string")
    return normalize_github_repository_url(operation.value).canonical_url


async def repository_ingest_gate(
    state: WorkflowState,
    runtime: Runtime[RepositoryDependencies],
) -> WorkflowState:
    """Require an exact human decision before any repository network or archive access."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    result = dependencies.store.load_result(workflow_id)
    if state.get("pending_gate_id") != result.gate_id:
        raise ValueError("repository Gate does not match the persisted candidate")
    resume_value = interrupt(
        {
            "schema_version": "1",
            "gate_id": result.gate_id,
            "gate_kind": GateKind.REPOSITORY_INGEST.value,
            "subject_type": REPOSITORY_SUBJECT_TYPE,
            "subject_id": repository_subject_id(workflow_id),
            "subject_revision": result.gate_revision,
            "paper_id": result.paper_id,
            "repository_url": result.requested_repository_url,
            "evidence_type": result.code_link_evidence.evidence_type,
            "confidence": result.code_link_evidence.confidence.value,
        }
    )
    approval = _revalidate_approval(resume_value)
    if approval.gate_kind is not GateKind.REPOSITORY_INGEST:
        raise ValueError("approval gate_kind is not REPOSITORY_INGEST")
    if approval.subject_type != REPOSITORY_SUBJECT_TYPE:
        raise ValueError("approval subject_type is not repository_ingest_candidate")
    if (
        approval.subject_id != repository_subject_id(workflow_id)
        or approval.subject_revision != result.gate_revision
    ):
        raise ValueError("approval does not target the current repository candidate revision")
    dependencies.approval_recorder.record(approval)

    if approval.decision is ApprovalDecision.REJECT:
        updated = result.model_copy(
            update={"status": "REJECTED", "updated_at": dependencies.clock()}
        )
        result_ref = dependencies.store.write_result(updated)
        return {
            "report_id": result_ref,
            "pending_gate_id": None,
            "phase": WorkflowPhase.REJECTED,
            "status": WorkflowStatus.REJECTED,
            "route": "REJECTED",
        }
    evidence = result.code_link_evidence
    if approval.decision is ApprovalDecision.APPROVE:
        approved_url = result.requested_repository_url
    else:
        approved_url = _edited_repository_url(approval)
        evidence = CodeLinkEvidence(
            schema_version="1",
            evidence_id=f"code-link-human-edit-{workflow_id}",
            paper_id=result.paper_id,
            repository_url=approved_url,
            evidence_type="search",
            confidence=CodeLinkConfidence.PROBABLE_MEDIUM,
            rationale_codes=["HUMAN_CONFIRMED_REPOSITORY_EDIT"],
            provenance=ProvenanceRef(
                schema_version="1",
                source_type="user",
                source_id=approval.approval_id,
                source_url=approved_url,
                retrieved_at=approval.decided_at,
            ),
            verified_at=approval.decided_at,
        )
    updated = result.model_copy(
        update={
            "approved_repository_url": approved_url,
            "code_link_evidence": evidence,
            "status": "INGESTING",
            "updated_at": dependencies.clock(),
        }
    )
    result_ref = dependencies.store.write_result(updated)
    return {
        "report_id": result_ref,
        "pending_gate_id": None,
        "phase": WorkflowPhase.REPOSITORY_RESOLUTION,
        "status": WorkflowStatus.RUNNING,
        "route": "INGEST",
    }


async def resolve_repository(
    state: WorkflowState,
    runtime: Runtime[RepositoryDependencies],
) -> WorkflowState:
    """Pin the approved URL, read metadata, and obtain a non-executed zip snapshot."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    result = dependencies.store.load_result(workflow_id)
    if result.status != "INGESTING" or result.approved_repository_url is None:
        raise ValueError("repository resolution requires an approved persisted candidate")
    ctx = _ctx(state, dependencies, operation="repository-resolve")
    try:
        resolution = await dependencies.repository_provider.resolve(
            result.approved_repository_url,
            None,
            ctx=ctx,
        )
        metadata = await dependencies.repository_provider.fetch_metadata(resolution, ctx=ctx)
        archive = await dependencies.repository_provider.snapshot(resolution, ctx=ctx)
        result_ref = dependencies.store.write_result(
            result.model_copy(
                update={
                    "resolution": resolution,
                    "metadata": metadata,
                    "archive": archive,
                    "updated_at": dependencies.clock(),
                }
            )
        )
    except PortError as error:
        _record_failed_result(dependencies, result, error.failure)
        return _failure_state(failure=error.failure)
    except OSError:
        failure = _generic_failure(
            code="REPOSITORY_OUTPUT_WRITE_FAILED",
            message="Repository evidence could not be written locally.",
        )
        return _failure_state(failure=failure)
    return {
        "report_id": result_ref,
        "phase": WorkflowPhase.REPOSITORY_ANALYSIS,
        "status": WorkflowStatus.RUNNING,
        "route": "RESOLVED",
        "last_error": None,
    }


def _license_status(spdx: str | None) -> LicenseStatus:
    if spdx is None or spdx == "NOASSERTION":
        return LicenseStatus.UNKNOWN
    if spdx in _ALLOWED_LICENSES:
        return LicenseStatus.ALLOWLISTED
    if spdx in _DENIED_LICENSES:
        return LicenseStatus.DENIED
    return LicenseStatus.REVIEW_REQUIRED


def _structure_type(framework_evidence: list[str], *, supported: bool) -> str:
    if not supported:
        return "UNSUPPORTED"
    if "STRUCTURE:TORCHVISION_TIMM" in framework_evidence:
        return "TORCHVISION_TIMM"
    return "PLAIN_PYTORCH"


async def analyze_repository(
    state: WorkflowState,
    runtime: Runtime[RepositoryDependencies],
) -> WorkflowState:
    """Create a deterministic, license-aware profile without importing repository code."""
    dependencies = runtime.context
    workflow_id = _required_text(state, "workflow_id")
    result = dependencies.store.load_result(workflow_id)
    if result.resolution is None or result.metadata is None or result.archive is None:
        raise ValueError("repository analysis requires persisted resolution evidence")
    ctx = _ctx(state, dependencies, operation="repository-analyze")
    try:
        analysis = await dependencies.static_analyzer.analyze(
            result.archive,
            dependencies.policy,
            ctx=ctx,
        )
        spdx = result.metadata.license_spdx or analysis.license_spdx
        license_status = _license_status(spdx)
        findings = list(analysis.dangerous_patterns)
        license_mismatch = (
            result.metadata.license_spdx is not None
            and analysis.license_spdx is not None
            and result.metadata.license_spdx != analysis.license_spdx
        )
        if license_mismatch:
            findings.append(
                RiskFinding(
                    schema_version="1",
                    finding_id="risk-license-mismatch",
                    rule_id="LICENSE_METADATA_MISMATCH",
                    category="LICENSE_METADATA_MISMATCH",
                    severity=SeverityLevel.MEDIUM,
                    description="GitHub metadata and the repository license text disagree.",
                    location_ref="LICENSE",
                )
            )
        supported = (
            analysis.supported
            and license_status is LicenseStatus.ALLOWLISTED
            and not license_mismatch
        )
        reasons = list(analysis.support_reasons)
        if license_status is not LicenseStatus.ALLOWLISTED:
            reasons.append(
                Reason(
                    schema_version="1",
                    code="REPOSITORY_LICENSE_NOT_ALLOWLISTED",
                    message="The repository license is unknown, denied, or needs separate review.",
                )
            )
        repository_id = (
            f"repo-{result.resolution.owner}-{result.resolution.name}-"
            f"{result.resolution.commit_sha[:12]}"
        )
        provenance = ProvenanceRef(
            schema_version="1",
            source_type="api",
            source_id=(
                f"github:{result.resolution.owner}/{result.resolution.name}"
                f"@{result.resolution.commit_sha}"
            ),
            source_url=result.resolution.canonical_url,
            retrieved_at=dependencies.clock(),
            content_hash=result.archive.sha256,
            evidence_artifact_id=result.archive.artifact_id,
        )
        snapshot = RepositorySnapshot(
            schema_version="1",
            repository_id=repository_id,
            canonical_url=result.resolution.canonical_url,
            provider=result.resolution.provider,
            owner=result.resolution.owner,
            name=result.resolution.name,
            commit_sha=result.resolution.commit_sha,
            archive_artifact_id=result.archive.artifact_id,
            license_spdx=spdx,
            license_status=license_status,
            framework=(
                "PyTorch"
                if any(item.startswith("PYTORCH_IMPORT:") for item in analysis.framework_evidence)
                else None
            ),
            languages=result.metadata.languages,
            default_branch=result.metadata.default_branch,
            risk_findings=findings,
            provenance=[provenance],
            analyzed_at=dependencies.clock(),
        )
        dependency_set = set(analysis.dependency_files)
        configuration_files = sorted(
            item.path
            for item in analysis.file_tree_summary
            if PurePosixPath(item.path).suffix.casefold() in {".json", ".toml", ".yaml", ".yml"}
            and item.path not in dependency_set
        )
        model_head_evidence = sorted(
            item.split(":", maxsplit=1)[1]
            for item in analysis.framework_evidence
            if item.startswith("MODEL_HEAD:")
        )
        profile = RepositoryProfile(
            profile_id=f"profile-{repository_id}",
            paper_id=result.paper_id,
            repository_snapshot=snapshot,
            code_link_evidence=result.code_link_evidence,
            structure_type=cast(
                Literal["PLAIN_PYTORCH", "TORCHVISION_TIMM", "UNSUPPORTED"],
                _structure_type(analysis.framework_evidence, supported=supported),
            ),
            entrypoint_candidates=analysis.entrypoint_candidates,
            data_loader_candidates=analysis.data_loader_candidates,
            configuration_files=configuration_files,
            dependency_files=analysis.dependency_files,
            model_head_evidence=model_head_evidence,
            framework_evidence=analysis.framework_evidence,
            file_tree_summary=analysis.file_tree_summary,
            risk_findings=findings,
            supported=supported,
            support_reasons=reasons,
        )
        final_status = "COMPLETED" if supported else "UNSUPPORTED"
        completed = result.model_copy(
            update={
                "analysis": analysis,
                "profile": profile,
                "status": final_status,
                "updated_at": dependencies.clock(),
            }
        )
        result_ref = dependencies.store.write_result(completed)
    except PortError as error:
        _record_failed_result(dependencies, result, error.failure)
        return _failure_state(failure=error.failure)
    except OSError:
        failure = _generic_failure(
            code="REPOSITORY_OUTPUT_WRITE_FAILED",
            message="The repository profile could not be written locally.",
        )
        return _failure_state(failure=failure)
    return {
        "report_id": result_ref,
        "repository_snapshot_ids": [snapshot.repository_id],
        "active_repository_id": snapshot.repository_id if supported else None,
        "phase": WorkflowPhase.COMPLETED,
        "status": WorkflowStatus.SUCCEEDED,
        "route": "COMPLETED" if supported else "UNSUPPORTED",
        "last_error": None,
    }


def route_after_repository_gate(state: WorkflowState) -> Literal["INGEST", "REJECTED"]:
    """Route the human decision to static ingestion or a terminal rejection."""
    route = state.get("route")
    if route in {"INGEST", "REJECTED"}:
        return cast(Literal["INGEST", "REJECTED"], route)
    raise ValueError("repository Gate did not produce a supported route")


def route_after_candidate_preparation(state: WorkflowState) -> Literal["GATE", "FAILED"]:
    """Stop invalid research inputs before entering the human Gate."""
    route = state.get("route")
    if route in {"GATE", "FAILED"}:
        return cast(Literal["GATE", "FAILED"], route)
    raise ValueError("repository candidate preparation did not produce a supported route")


def route_after_repository_resolution(state: WorkflowState) -> Literal["ANALYZE", "FAILED"]:
    """Stop provider failures; otherwise continue to deterministic analysis."""
    route = state.get("route")
    if route == "RESOLVED":
        return "ANALYZE"
    if route == "FAILED":
        return "FAILED"
    raise ValueError("repository resolution did not produce a supported route")


__all__ = [
    "REPOSITORY_SUBJECT_TYPE",
    "analyze_repository",
    "prepare_repository_candidate",
    "repository_gate_id",
    "repository_ingest_gate",
    "repository_subject_id",
    "resolve_repository",
    "route_after_candidate_preparation",
    "route_after_repository_gate",
    "route_after_repository_resolution",
]
