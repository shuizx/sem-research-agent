"""Fixed LangGraph, repair, Gate, resume, and artifact acceptance tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from vision_research_ops.application.state import WorkflowState, workflow_state_as_jsonable
from vision_research_ops.application.workflows import (
    build_adaptation_graph,
    workflow_config,
)
from vision_research_ops.domain import (
    Approval,
    ApprovalDecision,
    GateKind,
    PatchOperation,
    PatchOperationType,
    WorkflowStatus,
)

from .conftest import FIXED_NOW, AdaptationHarness


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
    metrics_output_file: str | None = None,
) -> Approval:
    edits: list[PatchOperation] = []
    if decision is ApprovalDecision.EDIT:
        assert metrics_output_file is not None
        edits = [
            PatchOperation(
                schema_version="1",
                op=PatchOperationType.REPLACE,
                path="/metrics_output_file",
                value=metrics_output_file,
                reason="Reviewer selected another allowlisted relative JSON output.",
            )
        ]
    return Approval(
        schema_version="1",
        approval_id=approval_id,
        gate_kind=GateKind.PATCH_ACCEPTANCE,
        subject_type=cast(str, payload["subject_type"]),
        subject_id=cast(str, payload["subject_id"]),
        subject_revision=cast(int, payload["subject_revision"]),
        decision=decision,
        edits=edits,
        reason="Pipeline reviewer decision for the exact smoke-passed patch.",
        actor_id="pipeline-reviewer",
        decided_at=FIXED_NOW,
        idempotency_key=f"idempotency-{approval_id}",
    )


async def _pause(
    harness: AdaptationHarness,
    initial_state: WorkflowState,
    *,
    saver: InMemorySaver | None = None,
):
    graph = build_adaptation_graph(checkpointer=saver)
    paused = await graph.ainvoke(
        initial_state,
        config=workflow_config(initial_state["thread_id"]),
        context=harness.dependencies,
    )
    return graph, cast(dict[str, object], paused)


@pytest.mark.asyncio
@pytest.mark.graph
async def test_happy_path_smokes_before_exact_gate_and_accepts_no_training(
    make_adaptation_harness,
    adaptation_initial_state: WorkflowState,
) -> None:
    """A real four-stage smoke precedes the interrupt and exact approval is terminal."""
    harness: AdaptationHarness = make_adaptation_harness()
    graph, paused = await _pause(harness, adaptation_initial_state)
    payload = _gate_payload(paused)

    assert paused["status"] == WorkflowStatus.WAITING_FOR_HUMAN
    assert payload["gate_kind"] == "PATCH_ACCEPTANCE"
    assert payload["subject_revision"] == 1
    assert payload["patch_hash"].startswith("sha256:")
    assert payload["smoke_capability_boundary"] == "FIXTURE_CONTRACT_PROBE_NO_TORCH"
    assert payload["real_pytorch_training"] is False
    assert harness.llm.call_count == 1
    assert harness.patch_tool.call_count == 1
    assert harness.smoke_tool.call_count == 1
    pending = harness.store.load_result("workflow-adaptation-1")
    assert pending.status == "AWAITING_APPROVAL"
    assert pending.accepted_patch_hash is None
    assert pending.gate_patch_hash == payload["patch_hash"]
    assert pending.planner_trace_ref is not None
    trace = harness.store.load_planner_trace(pending.planner_trace_ref)
    assert [event.tool_name for event in trace.events] == [
        "inspect_repository_profile",
        "inspect_dataset_contract",
        "compare_repository_dataset",
        "validate_adaptation_plan",
    ]

    approval = _approval(
        payload,
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-adaptation-happy",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(adaptation_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    result = harness.store.load_result("workflow-adaptation-1")
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert final["route"] == "ACCEPTED"
    assert final["run_ids"] == []
    assert result.status == "ACCEPTED"
    assert result.accepted_patch_hash == payload["patch_hash"]
    assert result.approval_id == approval.approval_id
    assert [review.decision for review in result.reviews] == ["APPROVE"]
    assert harness.llm.call_count == 1
    assert harness.patch_tool.call_count == 1
    assert harness.smoke_tool.call_count == 1
    workflow_state_as_jsonable(cast(Mapping[str, object], final))

    adaptation_raw = harness.store.resolve_ref(
        harness.store.result_ref("workflow-adaptation-1")
    ).read_text(encoding="utf-8")
    plan_raw = harness.store.resolve_ref(cast(str, result.plan_ref)).read_text(encoding="utf-8")
    smoke_raw = harness.store.resolve_ref(cast(str, result.attempts[-1].smoke_ref)).read_text(
        encoding="utf-8"
    )
    evidence = adaptation_raw + plan_raw + smoke_raw
    assert "location_ref" not in evidence
    assert "fixture-dataset-handle" not in evidence
    assert "C:\\" not in evidence
    assert "DASHSCOPE_API_KEY" not in evidence
    assert '"provider_id": "offline-fixture"' in plan_raw
    assert '"area": "INPUT_CHANNELS"' in plan_raw
    assert '"real_pytorch_training": false' in smoke_raw


@pytest.mark.asyncio
@pytest.mark.graph
async def test_invalid_p2_input_fails_before_llm_patch_or_smoke(
    make_adaptation_harness,
    adaptation_initial_state: WorkflowState,
) -> None:
    """An unsupported repository result cannot reach any proposal or execution boundary."""
    harness: AdaptationHarness = make_adaptation_harness(supported_repository=False)
    graph = build_adaptation_graph()
    final = await graph.ainvoke(
        adaptation_initial_state,
        config=workflow_config(adaptation_initial_state["thread_id"]),
        context=harness.dependencies,
    )

    assert final["status"] == WorkflowStatus.FAILED
    assert final["last_error"].code == "ADAPTATION_INPUT_INVALID"
    assert "__interrupt__" not in final
    assert harness.llm.call_count == 0
    assert harness.patch_tool.call_count == 0
    assert harness.smoke_tool.call_count == 0
    failed = harness.store.load_result("workflow-adaptation-1")
    assert failed.status == "FAILED"
    assert failed.failure is not None
    assert failed.failure.code == "ADAPTATION_INPUT_INVALID"


@pytest.mark.asyncio
@pytest.mark.graph
async def test_malformed_p2_commit_fails_structurally_before_side_effects(
    make_adaptation_harness,
    adaptation_initial_state: WorkflowState,
) -> None:
    """A corrupted abbreviated SHA in persisted repository JSON is rejected on strict load."""
    harness: AdaptationHarness = make_adaptation_harness()
    path = harness.repository_store.result_path("workflow-repository-p3-fixture")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["resolution"]["commit_sha"] = "abc123"
    path.write_text(json.dumps(raw), encoding="utf-8")

    graph = build_adaptation_graph()
    final = await graph.ainvoke(
        adaptation_initial_state,
        config=workflow_config(adaptation_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["last_error"].code == "ADAPTATION_INPUT_INVALID"
    assert harness.llm.call_count == 0
    assert harness.patch_tool.call_count == 0
    assert harness.smoke_tool.call_count == 0


@pytest.mark.asyncio
@pytest.mark.graph
async def test_complete_64_character_p2_commit_fails_before_side_effects(
    make_adaptation_harness,
    adaptation_initial_state: WorkflowState,
) -> None:
    """adaptation rejects a repository-consistent SHA-256 object ID before planning or execution."""
    harness: AdaptationHarness = make_adaptation_harness(repository_commit_sha="b" * 64)
    graph = build_adaptation_graph()
    final = await graph.ainvoke(
        adaptation_initial_state,
        config=workflow_config(adaptation_initial_state["thread_id"]),
        context=harness.dependencies,
    )

    assert final["status"] == WorkflowStatus.FAILED
    assert final["last_error"].code == "ADAPTATION_INPUT_INVALID"
    assert "__interrupt__" not in final
    assert harness.llm.call_count == 0
    assert harness.patch_tool.call_count == 0
    assert harness.smoke_tool.call_count == 0


@pytest.mark.asyncio
@pytest.mark.graph
async def test_reject_is_terminal_and_never_claims_patch_acceptance(
    make_adaptation_harness,
    adaptation_initial_state: WorkflowState,
) -> None:
    """REJECT preserves smoke evidence but cannot become an accepted patch or run."""
    harness: AdaptationHarness = make_adaptation_harness()
    graph, paused = await _pause(harness, adaptation_initial_state)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.REJECT,
        approval_id="approval-adaptation-reject",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(adaptation_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    result = harness.store.load_result("workflow-adaptation-1")
    assert final["status"] == WorkflowStatus.REJECTED
    assert final["run_ids"] == []
    assert result.status == "REJECTED"
    assert result.accepted_patch_hash is None


@pytest.mark.asyncio
@pytest.mark.graph
async def test_edit_changes_revision_and_hash_reruns_only_patch_and_smoke(
    make_adaptation_harness,
    adaptation_initial_state: WorkflowState,
) -> None:
    """EDIT invalidates the old subject, recompiles, re-smokes, and interrupts again."""
    harness: AdaptationHarness = make_adaptation_harness()
    graph, first_pause = await _pause(harness, adaptation_initial_state)
    first_payload = _gate_payload(first_pause)
    edit = _approval(
        first_payload,
        decision=ApprovalDecision.EDIT,
        approval_id="approval-adaptation-edit",
        metrics_output_file="reviewed/metrics.json",
    )
    second_pause = await graph.ainvoke(
        Command(resume=edit.model_dump(mode="json")),
        config=workflow_config(adaptation_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    second_payload = _gate_payload(cast(Mapping[str, object], second_pause))

    assert second_payload["subject_revision"] == 2
    assert second_payload["patch_hash"] != first_payload["patch_hash"]
    assert second_payload["subject_id"] != first_payload["subject_id"]
    assert second_payload["plan"]["metrics_output_file"] == "reviewed/metrics.json"
    assert harness.llm.call_count == 1
    assert harness.patch_tool.call_count == 2
    assert harness.smoke_tool.call_count == 2

    current_approval = _approval(
        second_payload,
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-adaptation-after-edit",
    )
    final = await graph.ainvoke(
        Command(resume=current_approval.model_dump(mode="json")),
        config=workflow_config(adaptation_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    result = harness.store.load_result("workflow-adaptation-1")
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert result.accepted_patch_hash == second_payload["patch_hash"]
    assert [review.decision for review in result.reviews] == ["EDIT", "APPROVE"]


@pytest.mark.asyncio
@pytest.mark.graph
async def test_old_approval_cannot_accept_an_edited_patch_hash(
    make_adaptation_harness,
    adaptation_initial_state: WorkflowState,
) -> None:
    """A first-revision approval is rejected at the second interrupt after EDIT."""
    harness: AdaptationHarness = make_adaptation_harness()
    graph, first_pause = await _pause(harness, adaptation_initial_state)
    first_payload = _gate_payload(first_pause)
    edit = _approval(
        first_payload,
        decision=ApprovalDecision.EDIT,
        approval_id="approval-adaptation-edit-before-replay",
        metrics_output_file="reviewed/metrics.json",
    )
    second_pause = await graph.ainvoke(
        Command(resume=edit.model_dump(mode="json")),
        config=workflow_config(adaptation_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    second_payload = _gate_payload(cast(Mapping[str, object], second_pause))
    assert second_payload["patch_hash"] != first_payload["patch_hash"]

    stale_approval = _approval(
        first_payload,
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-adaptation-stale",
    )
    with pytest.raises(ValueError, match="current patch revision and hash"):
        await graph.ainvoke(
            Command(resume=stale_approval.model_dump(mode="json")),
            config=workflow_config(adaptation_initial_state["thread_id"]),
            context=harness.dependencies,
        )
    assert harness.patch_tool.call_count == 2
    assert harness.smoke_tool.call_count == 2
    assert harness.store.load_result("workflow-adaptation-1").status == "AWAITING_APPROVAL"


@pytest.mark.asyncio
@pytest.mark.graph
async def test_first_actual_smoke_failure_repairs_once_then_reaches_gate(
    make_adaptation_harness,
    adaptation_initial_state: WorkflowState,
) -> None:
    """The controlled revision probe fails r1, repair changes the hash, and r2 passes."""
    harness: AdaptationHarness = make_adaptation_harness(minimum_repair_revision=1)
    _, paused = await _pause(harness, adaptation_initial_state)
    payload = _gate_payload(paused)
    result = harness.store.load_result("workflow-adaptation-1")

    assert payload["subject_revision"] == 2
    assert paused["retry_counts"] == {"adaptation_repair": 1}
    assert result.repair_count == 1
    assert len(result.attempts) == 2
    assert result.attempts[0].smoke_status == "FAILED"
    assert result.attempts[1].smoke_status == "PASSED"
    assert result.attempts[0].patch_hash != result.attempts[1].patch_hash
    assert harness.llm.call_count == 1
    assert harness.patch_tool.call_count == 2
    assert harness.smoke_tool.call_count == 2
    plan = harness.store.load_plan(cast(str, result.plan_ref))
    assert plan.origin == "DETERMINISTIC_REPAIR"
    assert plan.repair_revision == 1
    assert [item.reason_code for item in plan.repair_history] == ["REPAIR_AFTER_STATIC_POLICY"]


@pytest.mark.asyncio
@pytest.mark.graph
async def test_second_actual_smoke_failure_is_terminal_without_loop_or_gate(
    make_adaptation_harness,
    adaptation_initial_state: WorkflowState,
) -> None:
    """A fixture requiring revision two still fails after the only repair and stops."""
    harness: AdaptationHarness = make_adaptation_harness(minimum_repair_revision=2)
    graph = build_adaptation_graph()
    final = await graph.ainvoke(
        adaptation_initial_state,
        config=workflow_config(adaptation_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    result = harness.store.load_result("workflow-adaptation-1")

    assert final["status"] == WorkflowStatus.FAILED
    assert final["last_error"].code == "ADAPTATION_SMOKE_FAILED_AFTER_REPAIR"
    assert final["retry_counts"] == {"adaptation_repair": 1}
    assert "__interrupt__" not in final
    assert result.status == "FAILED"
    assert result.repair_count == 1
    assert [item.smoke_status for item in result.attempts] == ["FAILED", "FAILED"]
    assert harness.llm.call_count == 1
    assert harness.patch_tool.call_count == 2
    assert harness.smoke_tool.call_count == 2


@pytest.mark.parametrize(
    ("mode", "failure_code"),
    [
        ("provider_failure", "FIXTURE_LLM_PROVIDER_FAILED"),
        ("schema_failure", "LLM_SCHEMA_VALIDATION_FAILED"),
    ],
)
@pytest.mark.asyncio
@pytest.mark.graph
async def test_llm_failure_is_explicit_and_never_falls_back_to_patch(
    make_adaptation_harness,
    adaptation_initial_state: WorkflowState,
    mode: str,
    failure_code: str,
) -> None:
    """Provider and schema failures stop with their structured failure evidence."""
    harness: AdaptationHarness = make_adaptation_harness(llm_mode=mode)
    graph = build_adaptation_graph()
    final = await graph.ainvoke(
        adaptation_initial_state,
        config=workflow_config(adaptation_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    result = harness.store.load_result("workflow-adaptation-1")
    assert final["last_error"].code == failure_code
    assert result.status == "FAILED"
    assert result.failure is not None
    assert result.failure.code == failure_code
    assert harness.llm.call_count == 1
    assert harness.patch_tool.call_count == 0
    assert harness.smoke_tool.call_count == 0


@pytest.mark.asyncio
@pytest.mark.graph
async def test_new_graph_runtime_resumes_gate_without_replanning_or_resmoking(
    make_adaptation_harness,
    adaptation_initial_state: WorkflowState,
    tmp_path: Path,
) -> None:
    """The shared checkpointer resumes at interrupt with zero repeated side effects."""
    root = tmp_path / "shared-runtime"
    first: AdaptationHarness = make_adaptation_harness(root=root)
    saver = InMemorySaver()
    _, paused = await _pause(first, adaptation_initial_state, saver=saver)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-adaptation-resume",
    )

    resumed: AdaptationHarness = make_adaptation_harness(root=root)
    graph = build_adaptation_graph(checkpointer=saver)
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(adaptation_initial_state["thread_id"]),
        context=resumed.dependencies,
    )
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert first.llm.call_count == 1
    assert first.patch_tool.call_count == 1
    assert first.smoke_tool.call_count == 1
    assert resumed.llm.call_count == 0
    assert resumed.patch_tool.call_count == 0
    assert resumed.smoke_tool.call_count == 0
    assert resumed.recorder.get("approval-adaptation-resume") == approval
