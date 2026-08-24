"""Fixed offline acceptance matrix for the vertical-slice workflow LangGraph vertical slice."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import ValidationError

from tests.fakes import FailureStep, ScriptedExperimentExecutor
from vision_research_ops.application.nodes.fixtures import submit_training_fixture
from vision_research_ops.application.state import (
    InitialWorkflowInput,
    WorkflowState,
    add_budget_used,
    add_retry_counts,
    merge_stable_ids,
    workflow_state_as_jsonable,
)
from vision_research_ops.application.workflows import build_vertical_slice_graph, workflow_config
from vision_research_ops.domain import (
    Approval,
    ApprovalDecision,
    GateKind,
    PatchOperation,
    PatchOperationType,
    StructuredFailure,
    WorkflowPhase,
    WorkflowStatus,
)

from .conftest import FIXTURE_NOW, SHA256, GraphHarness


def _gate_payload(result: Mapping[str, object]) -> dict[str, object]:
    """Extract the one real LangGraph interrupt payload from an invocation result."""
    interrupts = result.get("__interrupt__")
    assert isinstance(interrupts, list | tuple)
    assert len(interrupts) == 1
    payload = interrupts[0].value
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _required_str(payload: Mapping[str, object], field: str) -> str:
    """Read a fixture gate string with a precise test assertion."""
    value = payload[field]
    assert isinstance(value, str)
    return value


def _required_int(payload: Mapping[str, object], field: str) -> int:
    """Read a fixture gate integer with a precise test assertion."""
    value = payload[field]
    assert isinstance(value, int)
    return value


def _approval_for_gate(
    payload: Mapping[str, object],
    *,
    decision: ApprovalDecision,
    approval_id: str,
) -> Approval:
    """Create a valid domain Approval bound to the exact surfaced gate subject."""
    edits: list[PatchOperation] = []
    if decision is ApprovalDecision.EDIT:
        edits = [
            PatchOperation(
                schema_version="1",
                op=PatchOperationType.REPLACE,
                path="/fixture_review_note",
                value="create a revised fixture plan",
                reason="Reviewer requested a fixture revision.",
            )
        ]
    return Approval(
        schema_version="1",
        approval_id=approval_id,
        gate_kind=GateKind.RUN_SUBMISSION,
        subject_type=_required_str(payload, "subject_type"),
        subject_id=_required_str(payload, "subject_id"),
        subject_revision=_required_int(payload, "subject_revision"),
        decision=decision,
        edits=edits,
        reason="Fixture review decision is recorded for the vertical slice.",
        actor_id="reviewer_fixture_1",
        decided_at=FIXTURE_NOW,
        idempotency_key=f"idempotency-{approval_id}",
    )


async def _interrupt(
    harness: GraphHarness,
    initial_state: WorkflowState,
    *,
    checkpointer: InMemorySaver | None = None,
):
    """Invoke a newly compiled graph through its first human interrupt."""
    graph = build_vertical_slice_graph(checkpointer=checkpointer)
    result = await graph.ainvoke(
        initial_state,
        config=workflow_config(initial_state["thread_id"]),
        context=harness.dependencies,
    )
    return graph, cast(dict[str, object], result)


@pytest.mark.graph
def test_graph_compiles_and_state_reducers_remain_small() -> None:
    """Compile the real StateGraph and prove the three normative reducers."""
    graph = build_vertical_slice_graph()
    assert graph.get_graph().nodes
    assert merge_stable_ids(["paper_a", "paper_b"], ["paper_b", "paper_c"]) == [
        "paper_a",
        "paper_b",
        "paper_c",
    ]
    assert add_retry_counts({"edit": 1}, {"edit": 1, "repair": 2}) == {
        "edit": 2,
        "repair": 2,
    }
    assert add_budget_used({"provider": 1.25}, {"provider": 0.75}) == {"provider": 2.0}
    with pytest.raises(ValueError):
        add_retry_counts({"edit": 1}, {"edit": -1})
    with pytest.raises(ValueError):
        add_budget_used({"provider": 1.0}, {"provider": -0.25})
    with pytest.raises(ValidationError):
        InitialWorkflowInput(
            workflow_id="",
            thread_id="thread_fixture_1",
            request_id="request_fixture_1",
            dataset_profile_id="dataset_fixture_1",
        )


@pytest.mark.asyncio
@pytest.mark.graph
async def test_initial_invocation_interrupts_before_executor_submission(
    make_harness,
    initial_state: WorkflowState,
) -> None:
    """The initial path pauses at the real human gate without any side effect."""
    harness = make_harness()
    _, paused = await _interrupt(harness, initial_state)
    payload = _gate_payload(paused)
    assert payload == {
        "schema_version": "1",
        "gate_id": "gate-run-submission-plan_fixture_1-r1",
        "gate_kind": "RUN_SUBMISSION",
        "subject_type": "adaptation_plan",
        "subject_id": "plan_fixture_1",
        "subject_revision": 1,
        "summary": "Fixture-only adaptation plan; approval controls fake run submission.",
    }
    assert harness.executor.submit_effect_count == 0


@pytest.mark.asyncio
@pytest.mark.graph
async def test_approve_resumes_once_and_completes(
    make_harness,
    initial_state: WorkflowState,
) -> None:
    """An exact recorded APPROVE reaches the fake executor once and completes."""
    harness = make_harness()
    graph, paused = await _interrupt(harness, initial_state)
    approval = _approval_for_gate(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval_approve_1",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert final["phase"] == WorkflowPhase.COMPLETED
    assert final["report_id"] == "report_fixture_1"
    assert final["run_ids"] == ["run_fixture_1"]
    assert harness.executor.submit_effect_count == 1
    assert harness.recorder.get("approval_approve_1") == approval


@pytest.mark.asyncio
@pytest.mark.graph
async def test_reject_resumes_without_submission_and_records_decision(
    make_harness,
    initial_state: WorkflowState,
) -> None:
    """A REJECT terminates at the gate and leaves the executor untouched."""
    harness = make_harness()
    graph, paused = await _interrupt(harness, initial_state)
    approval = _approval_for_gate(
        _gate_payload(paused),
        decision=ApprovalDecision.REJECT,
        approval_id="approval_reject_1",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.REJECTED
    assert final["phase"] == WorkflowPhase.REJECTED
    assert final["pending_gate_id"] is None
    assert harness.executor.submit_effect_count == 0
    assert harness.recorder.get("approval_reject_1") == approval


@pytest.mark.asyncio
@pytest.mark.graph
async def test_edit_reinterrupts_with_a_new_plan_then_approve_completes(
    make_harness,
    initial_state: WorkflowState,
) -> None:
    """EDIT records structured edits, derives revision two, and requires a new approval."""
    harness = make_harness()
    graph, first_pause = await _interrupt(harness, initial_state)
    first_payload = _gate_payload(first_pause)
    edit = _approval_for_gate(
        first_payload,
        decision=ApprovalDecision.EDIT,
        approval_id="approval_edit_1",
    )
    second_pause = await graph.ainvoke(
        Command(resume=edit.model_dump(mode="json")),
        config=workflow_config(initial_state["thread_id"]),
        context=harness.dependencies,
    )
    second_payload = _gate_payload(cast(dict[str, object], second_pause))
    assert second_payload["subject_id"] == "plan_fixture_1-r2"
    assert second_payload["subject_revision"] == 2
    assert second_pause["active_plan_id"] == "plan_fixture_1-r2"
    assert second_pause["retry_counts"] == {"edit": 1}
    assert harness.executor.submit_effect_count == 0

    approval = _approval_for_gate(
        second_payload,
        decision=ApprovalDecision.APPROVE,
        approval_id="approval_approve_2",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert harness.executor.submit_effect_count == 1


@pytest.mark.asyncio
@pytest.mark.graph
async def test_new_graph_instance_resumes_an_interrupted_thread(
    make_harness,
    initial_state: WorkflowState,
) -> None:
    """A separate compiled graph uses the same InMemorySaver to resume exactly once."""
    harness = make_harness()
    saver = InMemorySaver()
    _, paused = await _interrupt(harness, initial_state, checkpointer=saver)
    approval = _approval_for_gate(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval_restore_1",
    )
    resumed_graph = build_vertical_slice_graph(checkpointer=saver)
    final = await resumed_graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert final["report_id"] == "report_fixture_1"
    assert harness.executor.submit_effect_count == 1


@pytest.mark.asyncio
@pytest.mark.graph
@pytest.mark.parametrize(
    ("field", "replacement"),
    [("subject_id", "plan_fixture_stale"), ("subject_revision", 99)],
)
async def test_stale_approval_subject_or_revision_is_rejected_before_submission(
    make_harness,
    initial_state: WorkflowState,
    field: str,
    replacement: str | int,
) -> None:
    """Wrong subject IDs and revisions are rejected by the resumed gate itself."""
    harness = make_harness()
    graph, paused = await _interrupt(harness, initial_state)
    valid = _approval_for_gate(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id=f"approval_stale_{field}",
    )
    stale = valid.model_copy(update={field: replacement})
    with pytest.raises(ValueError):
        await graph.ainvoke(
            Command(resume=stale.model_dump(mode="json")),
            config=workflow_config(initial_state["thread_id"]),
            context=harness.dependencies,
        )
    assert harness.executor.submit_effect_count == 0
    assert harness.recorder.approvals == ()


@pytest.mark.asyncio
@pytest.mark.graph
async def test_direct_submit_without_recorded_approval_fails_closed(
    make_harness,
    initial_state: WorkflowState,
) -> None:
    """Manually reaching submit cannot bypass the recorder's exact approval check."""
    harness = make_harness()
    state = dict(initial_state)
    state.update(
        {
            "active_plan_id": "plan_fixture_1",
            "experiment_id": "exp_fixture_1",
            "route": "APPROVE",
            "pending_gate_id": None,
        }
    )
    failed = await submit_training_fixture(
        cast(WorkflowState, state),
        Runtime(context=harness.dependencies),
    )
    assert failed["status"] == WorkflowStatus.FAILED
    assert failed["phase"] == WorkflowPhase.FAILED
    assert failed["last_error"] is not None
    assert failed["last_error"].code == "RUN_SUBMISSION_APPROVAL_REQUIRED"
    assert harness.executor.submit_effect_count == 0


@pytest.mark.asyncio
@pytest.mark.graph
async def test_executor_port_error_ends_without_analysis(
    make_harness,
    initial_state: WorkflowState,
) -> None:
    """A structured deterministic port contract executor failure becomes terminal workflow state."""
    failure = StructuredFailure(
        schema_version="1",
        code="RUN_EXECUTOR_FAILURE",
        category="EXECUTOR",
        message="The deterministic fake executor declined the fixture run.",
        message_hash=SHA256,
        retryable=False,
    )
    executor = ScriptedExperimentExecutor(script={"executor.submit": [FailureStep(failure)]})
    harness = make_harness(executor=executor)
    graph, paused = await _interrupt(harness, initial_state)
    approval = _approval_for_gate(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval_port_error_1",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.FAILED
    assert final["phase"] == WorkflowPhase.FAILED
    assert final["route"] == "FAILED"
    assert final["report_id"] is None
    assert final["last_error"].code == "RUN_EXECUTOR_FAILURE"


@pytest.mark.asyncio
@pytest.mark.graph
async def test_final_state_is_small_and_json_safe(
    make_harness,
    initial_state: WorkflowState,
) -> None:
    """The final checkpoint projection contains only sanctioned small state fields."""
    harness = make_harness()
    graph, paused = await _interrupt(harness, initial_state)
    approval = _approval_for_gate(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval_json_safe_1",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(initial_state["thread_id"]),
        context=harness.dependencies,
    )
    jsonable = workflow_state_as_jsonable(cast(Mapping[str, object], final))
    encoded = json.dumps(jsonable, allow_nan=False)
    assert set(jsonable).issubset(WorkflowState.__annotations__)
    assert "run_spec" not in jsonable
    assert "executor" not in jsonable
    assert "fixture/workspace" not in encoded
    assert "C:\\" not in encoded
