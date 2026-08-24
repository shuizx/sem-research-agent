"""Bounded timeout, exit, and output-schema failure semantics for training."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from langgraph.types import Command

import vision_research_ops.adapters.execution.local_training as local_training_module
from vision_research_ops.application.state import WorkflowState
from vision_research_ops.application.training_runtime import TrainingToolError
from vision_research_ops.application.workflows.core import workflow_config
from vision_research_ops.domain import ApprovalDecision, WorkflowStatus
from vision_research_ops.ports import OperationContext

from .conftest import TrainingHarness
from .test_training_graph import _approval, _gate_payload, _pause


def _timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
    raise subprocess.TimeoutExpired(cmd=["python-current"], timeout=1)


def _nonzero(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["python-current"], returncode=7, stdout="", stderr="")


def _missing_artifacts(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["python-current"], returncode=0, stdout="", stderr="")


@pytest.mark.asyncio
@pytest.mark.graph
@pytest.mark.parametrize(
    ("fake_run", "failure_code"),
    [
        (_timeout, "TRAINING_RUN_TIMEOUT"),
        (_nonzero, "TRAINING_NONZERO_EXIT"),
        (_missing_artifacts, "TRAINING_ARTIFACT_INVALID"),
    ],
)
async def test_executor_failures_terminate_without_retry_or_fake_success(
    make_training_harness,
    training_initial_state: WorkflowState,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_run: Callable[..., subprocess.CompletedProcess[str]],
    failure_code: str,
) -> None:
    """Timeout, nonzero exit, and absent schema outputs are distinct terminal failures."""
    harness: TrainingHarness = make_training_harness(root=tmp_path / failure_code.casefold())
    graph, paused = await _pause(harness, training_initial_state)
    monkeypatch.setattr(local_training_module.subprocess, "run", fake_run)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id=f"approval-{failure_code.casefold()}",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(training_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    record = harness.store.load_workflow("workflow-training-1")
    assert final["status"] == WorkflowStatus.FAILED
    assert final["last_error"].code == failure_code
    assert record.status == "FAILED"
    assert record.failure is not None
    assert record.failure.code == failure_code
    assert harness.trainer.call_count == 1


@pytest.mark.asyncio
@pytest.mark.graph
async def test_completed_artifacts_are_idempotently_reused_without_new_process(
    make_training_harness,
    training_initial_state: WorkflowState,
    tmp_path: Path,
) -> None:
    """The adapter validates an existing exact run rather than submitting it again."""
    root = tmp_path / "adapter-reuse"
    harness: TrainingHarness = make_training_harness(root=root)
    graph, paused = await _pause(harness, training_initial_state)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-adapter-reuse",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(training_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.SUCCEEDED
    record = harness.store.load_workflow("workflow-training-1")
    assert record.current_spec_ref is not None
    spec = harness.store.load_spec(record.current_spec_ref)

    reused_harness: TrainingHarness = make_training_harness(root=root)
    reused = await reused_harness.trainer.run(
        spec,
        spec.baseline,
        ctx=OperationContext(
            schema_version="1",
            correlation_id="corr-training-reuse",
            workflow_id="workflow-training-1",
            actor_id="pipeline-user",
            idempotency_key="workflow-training-1:reuse-baseline",
            sensitivity="INTERNAL",
        ),
    )
    assert reused.reused_existing is True
    assert reused_harness.trainer.call_count == 0


@pytest.mark.asyncio
@pytest.mark.graph
async def test_malformed_jsonl_log_is_rejected_on_reuse_without_new_process(
    make_training_harness,
    training_initial_state: WorkflowState,
    tmp_path: Path,
) -> None:
    """A completed run with reviewer-style non-JSON log text fails closed on reuse."""
    root = tmp_path / "malformed-log-reuse"
    harness: TrainingHarness = make_training_harness(root=root)
    graph, paused = await _pause(harness, training_initial_state)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-malformed-log-reuse",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(training_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.SUCCEEDED
    record = harness.store.load_workflow("workflow-training-1")
    assert record.current_spec_ref is not None
    spec = harness.store.load_spec(record.current_spec_ref)
    harness.store.resolve_ref(harness.store.log_ref(spec.baseline.run_id)).write_text(
        "not-json\nstill-not-json\nreviewer-non-json\n",
        encoding="utf-8",
    )

    reused_harness: TrainingHarness = make_training_harness(root=root)
    with pytest.raises(TrainingToolError) as error:
        await reused_harness.trainer.run(
            spec,
            spec.baseline,
            ctx=OperationContext(
                schema_version="1",
                correlation_id="corr-training-malformed-log",
                workflow_id="workflow-training-1",
                actor_id="pipeline-user",
                idempotency_key="workflow-training-1:malformed-log-reuse",
                sensitivity="INTERNAL",
            ),
        )
    assert error.value.code == "TRAINING_ARTIFACT_INVALID"
    assert reused_harness.trainer.call_count == 0
