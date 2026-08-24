"""checkpoint recovery, exact reuse, hash conflict, and no-overwrite tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from vision_research_ops.application.evaluation_runtime import EvaluationDependencies
from vision_research_ops.application.services.evaluation_models import (
    EvaluationInitialInput,
    create_evaluation_state,
)
from vision_research_ops.application.workflows.core import workflow_config
from vision_research_ops.application.workflows.evaluation import build_evaluation_graph

from .conftest import EVALUATION_WORKFLOW_ID, EvaluationHarness


async def _run(
    harness: EvaluationHarness,
    *,
    saver: InMemorySaver | None = None,
    state: dict[str, object] | None = None,
) -> dict[str, object]:
    selected_state = harness.state if state is None else state
    result = await build_evaluation_graph(checkpointer=saver).ainvoke(
        selected_state,
        config=workflow_config(cast(str, selected_state["thread_id"])),
        context=harness.dependencies,
    )
    return cast(dict[str, object], result)


def _new_state(thread_id: str) -> dict[str, object]:
    return cast(
        dict[str, object],
        create_evaluation_state(
            EvaluationInitialInput(
                workflow_id=EVALUATION_WORKFLOW_ID,
                thread_id=thread_id,
                request_id=f"request-{thread_id}",
                training_workflow_id="workflow-training-evaluation-fixture",
            )
        ),
    )


@pytest.mark.asyncio
@pytest.mark.graph
async def test_new_graph_and_runtime_recover_completed_state_from_shared_checkpointer(
    make_evaluation_harness: Callable[..., EvaluationHarness],
    tmp_path: Path,
) -> None:
    """completed checkpoint recovery runs no persistence node a second time."""
    harness = make_evaluation_harness(root=tmp_path / "checkpoint-recovery")
    saver = InMemorySaver()
    first = await _run(harness, saver=saver)
    assert first["status"] == "COMPLETED"
    result = harness.evaluation_store.load_evaluation(EVALUATION_WORKFLOW_ID)
    evaluation_path = harness.evaluation_store.resolve_ref(result.evaluation_ref)
    report_path = harness.evaluation_store.resolve_ref(result.report_ref)
    before = (
        evaluation_path.read_bytes(),
        report_path.read_bytes(),
        evaluation_path.stat().st_mtime_ns,
        report_path.stat().st_mtime_ns,
    )

    dependencies = EvaluationDependencies(
        training_reader=harness.training_store,
        project_root=harness.project_root,
        store=harness.evaluation_store,
    )
    recovered = await build_evaluation_graph(checkpointer=saver).ainvoke(
        None,
        config=workflow_config(cast(str, harness.state["thread_id"])),
        context=dependencies,
    )
    assert recovered["status"] == "COMPLETED"
    after = (
        evaluation_path.read_bytes(),
        report_path.read_bytes(),
        evaluation_path.stat().st_mtime_ns,
        report_path.stat().st_mtime_ns,
    )
    assert after == before


@pytest.mark.asyncio
@pytest.mark.graph
async def test_exact_completed_evaluation_is_hash_verified_and_reused_without_overwrite(
    make_evaluation_harness: Callable[..., EvaluationHarness],
    tmp_path: Path,
) -> None:
    """a new thread recomputes exact bytes, then write-once store reuses both files."""
    harness = make_evaluation_harness(root=tmp_path / "exact-reuse")
    first = await _run(harness)
    assert first["status"] == "COMPLETED"
    result = harness.evaluation_store.load_evaluation(EVALUATION_WORKFLOW_ID)
    evaluation_path = harness.evaluation_store.resolve_ref(result.evaluation_ref)
    report_path = harness.evaluation_store.resolve_ref(result.report_ref)
    before = (
        evaluation_path.read_bytes(),
        report_path.read_bytes(),
        evaluation_path.stat().st_mtime_ns,
        report_path.stat().st_mtime_ns,
    )

    repeated = await _run(
        harness,
        state=_new_state("thread-evaluation-exact-reuse"),
    )
    assert repeated["status"] == "COMPLETED"
    assert repeated["evaluation_id"] == first["evaluation_id"]
    after = (
        evaluation_path.read_bytes(),
        report_path.read_bytes(),
        evaluation_path.stat().st_mtime_ns,
        report_path.stat().st_mtime_ns,
    )
    assert after == before


@pytest.mark.asyncio
@pytest.mark.graph
async def test_existing_report_hash_conflict_fails_explicitly_without_overwrite(
    make_evaluation_harness: Callable[..., EvaluationHarness],
    tmp_path: Path,
) -> None:
    """arbitrary narrative changes are never retained as a successful report."""
    harness = make_evaluation_harness(root=tmp_path / "report-conflict")
    await _run(harness)
    result = harness.evaluation_store.load_evaluation(EVALUATION_WORKFLOW_ID)
    report_path = harness.evaluation_store.resolve_ref(result.report_ref)
    corrupted = report_path.read_bytes() + b"\nmanual conclusion override\n"
    report_path.write_bytes(corrupted)

    repeated = await _run(
        harness,
        state=_new_state("thread-evaluation-report-conflict"),
    )
    assert repeated["status"] == "FAILED"
    failure = cast(dict[str, str], repeated["last_error"])
    assert failure["code"] == "EVALUATION_ARTIFACT_CONFLICT"
    assert report_path.read_bytes() == corrupted


@pytest.mark.asyncio
@pytest.mark.graph
async def test_changed_input_hash_gets_new_identity_then_explicit_output_conflict(
    make_evaluation_harness: Callable[..., EvaluationHarness],
    tmp_path: Path,
) -> None:
    """changed valid predictions cannot overwrite a prior workflow evaluation."""
    harness = make_evaluation_harness(root=tmp_path / "input-hash-conflict")
    first = await _run(harness)
    result = harness.evaluation_store.load_evaluation(EVALUATION_WORKFLOW_ID)
    evaluation_path = harness.evaluation_store.resolve_ref(result.evaluation_ref)
    original = evaluation_path.read_bytes()

    predictions_path = harness.training_store.resolve_ref(
        f"runs/{harness.spec.candidate.run_id}/predictions.json"
    )
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    items = cast(list[dict[str, object]], payload["items"])
    predicted_label = cast(int, items[0]["predicted_label"])
    scores = [0.05, 0.05, 0.05, 0.05]
    scores[predicted_label] = 0.85
    items[0]["scores"] = scores
    predictions_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    repeated = await _run(
        harness,
        state=_new_state("thread-evaluation-input-conflict"),
    )
    assert repeated["status"] == "FAILED"
    assert repeated["evaluation_id"] != first["evaluation_id"]
    failure = cast(dict[str, str], repeated["last_error"])
    assert failure["code"] == "EVALUATION_ARTIFACT_CONFLICT"
    assert evaluation_path.read_bytes() == original
