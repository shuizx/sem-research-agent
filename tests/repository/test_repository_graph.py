"""Fixed repository workflow LangGraph Gate, resume, and local profile tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from vision_research_ops.application.services.repository_models import RepositoryResult
from vision_research_ops.application.state import WorkflowState, workflow_state_as_jsonable
from vision_research_ops.application.workflows import build_repository_graph, workflow_config
from vision_research_ops.domain import (
    Approval,
    ApprovalDecision,
    GateKind,
    PatchOperation,
    PatchOperationType,
    WorkflowPhase,
    WorkflowStatus,
)

from .conftest import FIXED_NOW, FixtureGitHubTransport, RepositoryHarness, repository_archive


def _gate_payload(result: Mapping[str, object]) -> dict[str, object]:
    interrupts = result.get("__interrupt__")
    assert isinstance(interrupts, list | tuple)
    assert len(interrupts) == 1
    value = interrupts[0].value
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _approval(
    payload: Mapping[str, object],
    *,
    decision: ApprovalDecision,
    approval_id: str,
    edited_url: str | None = None,
) -> Approval:
    edits: list[PatchOperation] = []
    if decision is ApprovalDecision.EDIT:
        assert edited_url is not None
        edits = [
            PatchOperation(
                schema_version="1",
                op=PatchOperationType.REPLACE,
                path="/repository_url",
                value=edited_url,
                reason="The reviewer supplied the official GitHub repository URL.",
            )
        ]
    return Approval(
        schema_version="1",
        approval_id=approval_id,
        gate_kind=GateKind.REPOSITORY_INGEST,
        subject_type=cast(str, payload["subject_type"]),
        subject_id=cast(str, payload["subject_id"]),
        subject_revision=cast(int, payload["subject_revision"]),
        decision=decision,
        edits=edits,
        reason="Pipeline reviewer decision for repository ingestion.",
        actor_id="pipeline-reviewer",
        decided_at=FIXED_NOW,
        idempotency_key=f"idempotency-{approval_id}",
    )


async def _pause(
    harness: RepositoryHarness,
    initial_state: WorkflowState,
    *,
    saver: InMemorySaver | None = None,
):
    graph = build_repository_graph(checkpointer=saver)
    paused = await graph.ainvoke(
        initial_state,
        config=workflow_config(initial_state["thread_id"]),
        context=harness.dependencies,
    )
    return graph, cast(dict[str, object], paused)


@pytest.mark.asyncio
@pytest.mark.graph
async def test_repository_graph_interrupts_before_any_external_ingest(
    make_repository_harness,
    repository_initial_state: WorkflowState,
) -> None:
    """Paper-code evidence reaches a real Gate before GitHub or archive access."""
    harness = make_repository_harness()
    _, paused = await _pause(harness, repository_initial_state)
    payload = _gate_payload(paused)

    assert payload["gate_kind"] == "REPOSITORY_INGEST"
    assert payload["confidence"] == "OFFICIAL_HIGH"
    assert payload["repository_url"] == "https://github.com/example/sem-classifier"
    assert paused["phase"] == WorkflowPhase.AWAITING_INGEST_APPROVAL
    assert paused["status"] == WorkflowStatus.WAITING_FOR_HUMAN
    assert harness.transport.call_count == 0
    pending = harness.store.load_result("workflow-repository-1")
    assert pending.status == "AWAITING_APPROVAL"
    assert pending.code_link_evidence.evidence_type == "paper_link"


@pytest.mark.asyncio
@pytest.mark.graph
async def test_approve_produces_pinned_explainable_local_profile(
    make_repository_harness,
    repository_initial_state: WorkflowState,
) -> None:
    """APPROVE completes the static happy path and persists all required evidence."""
    harness = make_repository_harness()
    graph, paused = await _pause(harness, repository_initial_state)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-repository-approve",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(repository_initial_state["thread_id"]),
        context=harness.dependencies,
    )

    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert final["phase"] == WorkflowPhase.COMPLETED
    assert final["route"] == "COMPLETED"
    assert final["active_repository_id"].startswith("repo-example-sem-classifier-")
    assert final["report_id"] == ("repositories/workflow-repository-1/repository-profile.json")
    result = harness.store.load_result("workflow-repository-1")
    assert result.status == "COMPLETED"
    assert result.resolution is not None
    assert result.resolution.commit_sha == "a" * 40
    assert result.profile is not None
    assert result.profile.supported is True
    assert result.profile.structure_type == "PLAIN_PYTORCH"
    assert result.profile.repository_snapshot.license_spdx == "MIT"
    assert result.profile.entrypoint_candidates == ["train.py"]
    assert result.profile.configuration_files == ["config.yaml"]
    assert result.profile.model_head_evidence == ["model.py", "train.py"]
    assert harness.transport.call_count == 4
    assert harness.recorder.get("approval-repository-approve") == approval
    workflow_state_as_jsonable(cast(Mapping[str, object], final))

    raw = json.loads(harness.store.result_path("workflow-repository-1").read_text(encoding="utf-8"))
    assert raw["resolution"]["commit_sha"] == "a" * 40
    assert raw["profile"]["code_link_evidence"]["confidence"] == "OFFICIAL_HIGH"
    assert raw["archive"]["uri"].startswith("snapshots/")
    assert "C:\\" not in json.dumps(raw)


@pytest.mark.asyncio
@pytest.mark.graph
async def test_reject_ends_without_provider_or_snapshot_calls(
    make_repository_harness,
    repository_initial_state: WorkflowState,
) -> None:
    """REJECT is terminal and preserves the zero-ingest Gate boundary."""
    harness = make_repository_harness()
    graph, paused = await _pause(harness, repository_initial_state)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.REJECT,
        approval_id="approval-repository-reject",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(repository_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.REJECTED
    assert final["phase"] == WorkflowPhase.REJECTED
    assert harness.transport.call_count == 0
    assert harness.store.load_result("workflow-repository-1").status == "REJECTED"


@pytest.mark.asyncio
@pytest.mark.graph
async def test_edit_accepts_only_a_human_supplied_canonical_github_target(
    make_repository_harness,
    repository_initial_state: WorkflowState,
) -> None:
    """EDIT replaces the candidate URL structurally and then follows the same static path."""
    harness = make_repository_harness()
    graph, paused = await _pause(harness, repository_initial_state)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.EDIT,
        approval_id="approval-repository-edit",
        edited_url="https://github.com/Alternate/Vision-Model.git",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(repository_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    result = harness.store.load_result("workflow-repository-1")
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert result.approved_repository_url == "https://github.com/alternate/vision-model"
    assert result.resolution is not None
    assert result.resolution.owner == "alternate"
    assert result.resolution.name == "vision-model"
    assert result.profile is not None
    assert result.profile.code_link_evidence.repository_url == result.resolution.canonical_url
    assert result.profile.code_link_evidence.confidence.value == "PROBABLE_MEDIUM"


@pytest.mark.asyncio
@pytest.mark.graph
async def test_new_graph_resumes_at_gate_without_preapproval_provider_reads(
    make_repository_harness,
    repository_initial_state: WorkflowState,
    tmp_path: Path,
) -> None:
    """A new graph and runtime resume the same checkpoint and ingest only after approval."""
    root = tmp_path / "shared"
    first = make_repository_harness(root=root)
    saver = InMemorySaver()
    _, paused = await _pause(first, repository_initial_state, saver=saver)
    assert first.transport.call_count == 0
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-repository-resume",
    )
    resumed = make_repository_harness(root=root)
    graph = build_repository_graph(checkpointer=saver)
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(repository_initial_state["thread_id"]),
        context=resumed.dependencies,
    )
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert first.transport.call_count == 0
    assert resumed.transport.call_count == 4


@pytest.mark.asyncio
@pytest.mark.graph
async def test_non_allowlisted_license_finishes_as_explainable_unsupported(
    make_repository_harness,
    repository_initial_state: WorkflowState,
) -> None:
    """A GPL metadata result is visible but never becomes an active allowlisted profile."""
    harness = make_repository_harness(license_spdx="GPL-3.0-only")
    graph, paused = await _pause(harness, repository_initial_state)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-repository-gpl",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(repository_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    result = harness.store.load_result("workflow-repository-1")
    assert final["route"] == "UNSUPPORTED"
    assert final["active_repository_id"] is None
    assert result.status == "UNSUPPORTED"
    assert result.profile is not None
    assert result.profile.supported is False
    assert result.profile.repository_snapshot.license_status.value == "REVIEW_REQUIRED"
    assert "LICENSE_METADATA_MISMATCH" in {item.rule_id for item in result.profile.risk_findings}


@pytest.mark.asyncio
@pytest.mark.graph
async def test_unsafe_source_pattern_is_profiled_but_not_allowlisted(
    make_repository_harness,
    repository_initial_state: WorkflowState,
) -> None:
    """A static os.system finding terminates normally as an unsupported candidate."""
    harness = make_repository_harness(dangerous_source=True)
    graph, paused = await _pause(harness, repository_initial_state)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-repository-unsafe",
    )
    await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(repository_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    result = harness.store.load_result("workflow-repository-1")
    assert result.status == "UNSUPPORTED"
    assert result.profile is not None
    assert [item.rule_id for item in result.profile.risk_findings] == ["OS_SYSTEM"]


@pytest.mark.asyncio
@pytest.mark.graph
async def test_invalid_p1_code_link_fails_before_gate_or_provider(
    make_repository_harness,
    repository_initial_state: WorkflowState,
) -> None:
    """An untrusted non-GitHub URL cannot become an interrupt or provider request."""
    harness = make_repository_harness(code_urls=["https://evil.example/repository"])
    graph = build_repository_graph()
    final = await graph.ainvoke(
        repository_initial_state,
        config=workflow_config(repository_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.FAILED
    assert final["last_error"].code == "REPOSITORY_CANDIDATE_INVALID"
    assert "__interrupt__" not in final
    assert harness.transport.call_count == 0


@pytest.mark.asyncio
@pytest.mark.graph
async def test_invalid_provider_sha_is_an_explicit_failed_result(
    make_repository_harness,
    repository_initial_state: WorkflowState,
) -> None:
    """Provider schema failure cannot silently produce an abbreviated repository pin."""
    transport = FixtureGitHubTransport(repository_archive(), commit_sha="abc123")
    harness = make_repository_harness(transport=transport)
    graph, paused = await _pause(harness, repository_initial_state)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-repository-invalid-sha",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(repository_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.FAILED
    assert final["last_error"].code == "GITHUB_COMMIT_SHA_INVALID"
    result = RepositoryResult.model_validate_json(
        harness.store.result_path("workflow-repository-1").read_text(encoding="utf-8")
    )
    assert result.status == "FAILED"
    assert result.failure is not None
    assert result.failure.code == "GITHUB_COMMIT_SHA_INVALID"
