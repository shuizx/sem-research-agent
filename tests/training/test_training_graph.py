"""Fixed LangGraph, Gate, cancellation, resume, and artifact tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from vision_research_ops.application.state import WorkflowState, workflow_state_as_jsonable
from vision_research_ops.application.workflows.core import workflow_config
from vision_research_ops.application.workflows.training import build_training_graph
from vision_research_ops.domain import (
    Approval,
    ApprovalDecision,
    GateKind,
    PatchOperation,
    PatchOperationType,
    WorkflowStatus,
)

from .conftest import FIXED_NOW, TrainingHarness


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
    edit_path: str = "/budget/max_epochs",
    edit_value: int = 3,
) -> Approval:
    edits: list[PatchOperation] = []
    if decision is ApprovalDecision.EDIT:
        edits = [
            PatchOperation(
                schema_version="1",
                op=PatchOperationType.REPLACE,
                path=edit_path,
                value=edit_value,
                reason="Reviewer selected another bounded fixture value.",
            )
        ]
    return Approval(
        schema_version="1",
        approval_id=approval_id,
        gate_kind=GateKind.RUN_SUBMISSION,
        subject_type=cast(str, payload["subject_type"]),
        subject_id=cast(str, payload["subject_id"]),
        subject_revision=cast(int, payload["subject_revision"]),
        decision=decision,
        edits=edits,
        reason="Pipeline reviewer decision for the exact frozen training spec.",
        actor_id="pipeline-reviewer",
        decided_at=FIXED_NOW,
        idempotency_key=f"idempotency-{approval_id}",
    )


async def _pause(
    harness: TrainingHarness,
    initial_state: WorkflowState,
    *,
    saver: InMemorySaver | None = None,
):
    graph = build_training_graph(checkpointer=saver)
    paused = await graph.ainvoke(
        initial_state,
        config=workflow_config(initial_state["thread_id"]),
        context=harness.dependencies,
    )
    return graph, cast(dict[str, object], paused)


@pytest.mark.asyncio
@pytest.mark.graph
async def test_happy_path_requires_gate_and_runs_real_fair_fixture_pair(
    make_training_harness,
    training_initial_state: WorkflowState,
    tmp_path: Path,
) -> None:
    """Approval releases two actual optimizer runs and four strict artifacts each."""
    harness: TrainingHarness = make_training_harness(root=tmp_path / "happy")
    graph, paused = await _pause(harness, training_initial_state)
    payload = _gate_payload(paused)
    record = harness.store.load_workflow("workflow-training-1")
    assert paused["status"] == WorkflowStatus.WAITING_FOR_HUMAN
    assert payload["gate_kind"] == "RUN_SUBMISSION"
    assert payload["frozen_spec_hash"] == record.current_spec_hash
    assert payload["capability"] == "SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"
    assert payload["real_pytorch_training"] is False
    assert harness.trainer.call_count == 0
    assert record.current_spec_ref is not None
    spec = harness.store.load_spec(record.current_spec_ref)
    assert spec.baseline.split_hash == spec.candidate.split_hash
    assert spec.baseline.preprocess_hash == spec.candidate.preprocess_hash
    assert spec.baseline.seed == spec.candidate.seed
    assert spec.baseline.budget == spec.candidate.budget
    assert spec.baseline.method_config_ref != spec.candidate.method_config_ref

    approval = _approval(
        payload,
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-training-happy",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(training_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    completed = harness.store.load_workflow("workflow-training-1")
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert completed.status == "SUCCEEDED"
    assert harness.trainer.call_count == 2
    assert len(final["run_ids"]) == 2
    for result in (completed.baseline_result, completed.candidate_result):
        assert result is not None
        manifest = harness.store.load_manifest(result.run_id)
        metrics = harness.store.load_metrics(result.run_id)
        predictions = harness.store.load_predictions(result.run_id)
        assert manifest.capability == "SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"
        assert manifest.real_pytorch_training is False
        assert manifest.shell_used is False
        assert manifest.network_used is False
        assert len(metrics.step_losses) > 1
        assert len(metrics.epoch_losses) >= 1
        assert len(predictions.items) == 6
        for name in ("manifest.json", "train.log", "metrics.json", "predictions.json"):
            assert (harness.project_root / "var" / "runs" / result.run_id / name).is_file()
    state_json = json.dumps(workflow_state_as_jsonable(final), allow_nan=False)
    assert str(harness.project_root) not in state_json


@pytest.mark.asyncio
@pytest.mark.graph
async def test_reject_keeps_executor_at_zero(
    make_training_harness,
    training_initial_state: WorkflowState,
    tmp_path: Path,
) -> None:
    """Gate rejection is terminal and cannot invoke either trainer run."""
    harness: TrainingHarness = make_training_harness(root=tmp_path / "reject")
    graph, paused = await _pause(harness, training_initial_state)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.REJECT,
        approval_id="approval-training-reject",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(training_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.REJECTED
    assert final["run_ids"] == []
    assert harness.trainer.call_count == 0
    assert harness.store.load_workflow("workflow-training-1").status == "REJECTED"


@pytest.mark.asyncio
@pytest.mark.graph
async def test_edit_changes_spec_hash_and_requires_a_second_gate(
    make_training_harness,
    training_initial_state: WorkflowState,
    tmp_path: Path,
) -> None:
    """A bounded edit refreezes both runs and only the new subject may execute."""
    harness: TrainingHarness = make_training_harness(root=tmp_path / "edit")
    graph, first_pause = await _pause(harness, training_initial_state)
    first_payload = _gate_payload(first_pause)
    edit = _approval(
        first_payload,
        decision=ApprovalDecision.EDIT,
        approval_id="approval-training-edit",
        edit_value=3,
    )
    second_pause = await graph.ainvoke(
        Command(resume=edit.model_dump(mode="json")),
        config=workflow_config(training_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    second_payload = _gate_payload(cast(Mapping[str, object], second_pause))
    assert second_payload["subject_revision"] == 2
    assert second_payload["subject_id"] != first_payload["subject_id"]
    assert second_payload["frozen_spec_hash"] != first_payload["frozen_spec_hash"]
    assert cast(dict[str, object], second_payload["budget"])["max_epochs"] == 3
    assert harness.trainer.call_count == 0

    current = _approval(
        second_payload,
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-training-after-edit",
    )
    final = await graph.ainvoke(
        Command(resume=current.model_dump(mode="json")),
        config=workflow_config(training_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    record = harness.store.load_workflow("workflow-training-1")
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert [review.decision for review in record.reviews] == ["EDIT", "APPROVE"]
    assert harness.trainer.call_count == 2


@pytest.mark.asyncio
@pytest.mark.graph
async def test_old_approval_cannot_release_an_edited_spec(
    make_training_harness,
    training_initial_state: WorkflowState,
    tmp_path: Path,
) -> None:
    """Revision-one approval is invalid after edit changes the hash-bound subject."""
    harness: TrainingHarness = make_training_harness(root=tmp_path / "stale")
    graph, first_pause = await _pause(harness, training_initial_state)
    first_payload = _gate_payload(first_pause)
    edit = _approval(
        first_payload,
        decision=ApprovalDecision.EDIT,
        approval_id="approval-training-edit-stale",
    )
    await graph.ainvoke(
        Command(resume=edit.model_dump(mode="json")),
        config=workflow_config(training_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    stale = _approval(
        first_payload,
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-training-stale",
    )
    with pytest.raises(ValueError, match="current training spec"):
        await graph.ainvoke(
            Command(resume=stale.model_dump(mode="json")),
            config=workflow_config(training_initial_state["thread_id"]),
            context=harness.dependencies,
        )
    assert harness.trainer.call_count == 0
    assert harness.store.load_workflow("workflow-training-1").status == "AWAITING_APPROVAL"


@pytest.mark.asyncio
@pytest.mark.graph
@pytest.mark.parametrize(
    "input_updates",
    [
        {"base_commit_sha": "a" * 64},
        {"patch_hash": f"sha256:{'e' * 64}"},
        {"patch_revision": 2},
        {"split_hash": f"sha256:{'f' * 64}"},
    ],
)
async def test_invalid_exact_input_stops_before_gate_and_execution(
    make_training_harness,
    training_initial_state: WorkflowState,
    tmp_path: Path,
    input_updates: dict[str, object],
) -> None:
    """Malformed or mismatched commit/patch/split evidence cannot reach the Gate."""
    name = next(iter(input_updates))
    harness: TrainingHarness = make_training_harness(
        root=tmp_path / f"invalid-{name}",
        input_updates=input_updates,
    )
    graph = build_training_graph()
    final = await graph.ainvoke(
        training_initial_state,
        config=workflow_config(training_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.FAILED
    assert final["last_error"].code == "TRAINING_INPUT_INVALID"
    assert "__interrupt__" not in final
    assert harness.trainer.call_count == 0


@pytest.mark.asyncio
@pytest.mark.graph
@pytest.mark.parametrize(
    ("cancellation_values", "expected_calls", "point"),
    [
        ((True,), 0, "BEFORE_BASELINE"),
        ((False, True), 1, "BETWEEN_RUNS"),
    ],
)
async def test_cancellation_is_explicit_at_both_submission_boundaries(
    make_training_harness,
    training_initial_state: WorkflowState,
    tmp_path: Path,
    cancellation_values: tuple[bool, ...],
    expected_calls: int,
    point: str,
) -> None:
    """Ordinary cancellation never becomes fake success or an unbounded retry."""
    harness: TrainingHarness = make_training_harness(
        root=tmp_path / f"cancel-{point}",
        cancellation_values=cancellation_values,
    )
    graph, paused = await _pause(harness, training_initial_state)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id=f"approval-cancel-{point}",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(training_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    record = harness.store.load_workflow("workflow-training-1")
    assert final["status"] == WorkflowStatus.CANCELLED
    assert final["last_error"].code == "TRAINING_RUN_CANCELLED"
    assert record.status == "CANCELLED"
    assert record.cancellation_point == point
    assert harness.trainer.call_count == expected_calls


@pytest.mark.asyncio
@pytest.mark.graph
async def test_new_runtime_resumes_once_and_completed_resume_does_not_resubmit(
    make_training_harness,
    training_initial_state: WorkflowState,
    tmp_path: Path,
) -> None:
    """A shared checkpointer resumes at Gate and completed state has zero new submits."""
    root = tmp_path / "resume"
    first: TrainingHarness = make_training_harness(root=root)
    saver = InMemorySaver()
    _, paused = await _pause(first, training_initial_state, saver=saver)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-training-resume",
    )
    resumed: TrainingHarness = make_training_harness(root=root)
    graph = build_training_graph(checkpointer=saver)
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(training_initial_state["thread_id"]),
        context=resumed.dependencies,
    )
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert first.trainer.call_count == 0
    assert resumed.trainer.call_count == 2

    repeated: TrainingHarness = make_training_harness(root=root)
    completed = await build_training_graph(checkpointer=saver).ainvoke(
        None,
        config=workflow_config(training_initial_state["thread_id"]),
        context=repeated.dependencies,
    )
    assert completed["status"] == WorkflowStatus.SUCCEEDED
    assert repeated.trainer.call_count == 0
