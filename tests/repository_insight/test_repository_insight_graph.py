"""ordinary outer graph acceptance for repository insight workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from vision_research_ops.application.repository_insight_runtime import (
    RepositoryInsightState,
    create_repository_insight_state,
)
from vision_research_ops.application.workflows import (
    build_repository_insight_graph,
    workflow_config,
)
from vision_research_ops.domain import Approval, ApprovalDecision, GateKind
from vision_research_ops.repository_insight.fixture_repository import FIXTURE_COMMIT_SHA

from .conftest import FIXED_NOW, RepositoryInsightHarness


def _gate_payload(result: Mapping[str, object]) -> dict[str, object]:
    interrupts = result.get("__interrupt__")
    assert isinstance(interrupts, list | tuple)
    assert len(interrupts) == 1
    value = interrupts[0].value
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _approval(
    payload: Mapping[str, object],
    decision: ApprovalDecision,
) -> Approval:
    return Approval(
        schema_version="1",
        approval_id=f"approval-repository-insight-{decision.value.casefold()}",
        gate_kind=GateKind.REPOSITORY_INGEST,
        subject_type=cast(str, payload["subject_type"]),
        subject_id=cast(str, payload["subject_id"]),
        subject_revision=cast(int, payload["subject_revision"]),
        decision=decision,
        edits=[],
        reason="User decision for a fixed public source snapshot.",
        actor_id="pipeline-user",
        decided_at=FIXED_NOW,
        idempotency_key=f"idempotency-repository-insight-{decision.value.casefold()}",
    )


async def _pause(
    harness: RepositoryInsightHarness,
    state: RepositoryInsightState,
    *,
    saver: InMemorySaver | None = None,
) -> tuple[object, dict[str, object]]:
    graph = build_repository_insight_graph(checkpointer=saver)
    paused = await graph.ainvoke(
        state,
        config=workflow_config(cast(str, state["thread_id"])),
        context=harness.dependencies,
    )
    return graph, cast(dict[str, object], paused)


@pytest.mark.asyncio
@pytest.mark.graph
async def test_gate_precedes_all_github_snapshot_source_and_llm_side_effects(
    make_repository_insight_harness,
    repository_insight_state: RepositoryInsightState,
) -> None:
    """the real outer graph interrupts before every downstream boundary."""
    harness = make_repository_insight_harness()
    _graph, paused = await _pause(harness, repository_insight_state)
    payload = _gate_payload(paused)

    assert payload["repository_url"] == "https://github.com/example/sem-classifier"
    assert payload["requested_action"] == "DOWNLOAD_AND_READ_FIXED_COMMIT_SOURCE_SNAPSHOT"
    assert "not git clone" in cast(str, payload["notice"])
    assert harness.transport.call_count == 0
    assert not harness.workspace.exists()


@pytest.mark.asyncio
@pytest.mark.graph
async def test_reject_has_zero_downstream_side_effects(
    make_repository_insight_harness,
    repository_insight_state: RepositoryInsightState,
) -> None:
    """reject ends without GitHub, snapshot, source, LLM, or artifact work."""
    harness = make_repository_insight_harness()
    graph, paused = await _pause(harness, repository_insight_state)
    final = await graph.ainvoke(
        Command(resume=_approval(_gate_payload(paused), ApprovalDecision.REJECT)),
        config=workflow_config(cast(str, repository_insight_state["thread_id"])),
        context=harness.dependencies,
    )
    assert final["status"] == "REJECTED"
    assert final["result_ref"] is None
    assert harness.transport.call_count == 0
    assert not harness.workspace.exists()


@pytest.mark.asyncio
@pytest.mark.graph
async def test_approve_pins_snapshot_runs_four_tools_and_writes_relative_evidence(
    make_repository_insight_harness,
    repository_insight_state: RepositoryInsightState,
) -> None:
    """approved happy path is real, bounded, honest, and locally evidenced."""
    harness = make_repository_insight_harness()
    graph, paused = await _pause(harness, repository_insight_state)
    final = await graph.ainvoke(
        Command(resume=_approval(_gate_payload(paused), ApprovalDecision.APPROVE)),
        config=workflow_config(cast(str, repository_insight_state["thread_id"])),
        context=harness.dependencies,
    )

    assert final["status"] == "COMPLETED"
    assert harness.transport.call_count == 4
    result = harness.dependencies.store.load_result("workflow-repository-insight-1")
    assert result.resolution.commit_sha == FIXTURE_COMMIT_SHA
    assert result.metadata.license_spdx == "MIT"
    assert result.snapshot.size_bytes <= 25 * 1024 * 1024
    assert result.snapshot.uri.startswith("snapshots/")
    assert result.source_snapshot_only is True
    assert result.git_clone_performed is False
    assert result.patch_generated is False
    assert result.smoke_test_run is False
    assert result.training_run is False
    assert result.company_data_used is False
    assert result.read_files
    assert sum(item.returned_bytes for item in result.read_files) <= 48 * 1024
    assert all(item.returned_bytes <= 8 * 1024 for item in result.read_files)
    assert not any(Path(ref).is_absolute() for ref in (result.result_ref, result.report_ref))
    planner = harness.dependencies.planner
    assert planner.last_graph_nodes == {
        "__start__",
        "code_reader_model",
        "tools",
        "__end__",
    }
    assert planner.last_tool_names == (
        "inspect_repository_summary",
        "inspect_target_profile",
        "read_repository_file",
        "submit_adaptation_advice",
    )
    trace_path = harness.workspace / result.trace_ref
    trace_text = trace_path.read_text(encoding="utf-8")
    assert "for images, labels" not in trace_text
    raw = json.loads((harness.workspace / result.result_ref).read_text(encoding="utf-8"))
    rendered = json.dumps(raw)
    assert "C:\\" not in rendered
    assert raw["resolution"]["commit_sha"] == FIXTURE_COMMIT_SHA


@pytest.mark.asyncio
@pytest.mark.graph
async def test_new_outer_graph_resumes_same_gate_before_fetching(
    make_repository_insight_harness,
    repository_insight_state: RepositoryInsightState,
    tmp_path: Path,
) -> None:
    """the in-memory checkpoint can resume with a new graph/runtime after approval."""
    root = tmp_path / "shared"
    saver = InMemorySaver()
    first = make_repository_insight_harness(root=root)
    _graph, paused = await _pause(first, repository_insight_state, saver=saver)
    assert first.transport.call_count == 0
    resumed = make_repository_insight_harness(root=root)
    graph = build_repository_insight_graph(checkpointer=saver)
    final = await graph.ainvoke(
        Command(resume=_approval(_gate_payload(paused), ApprovalDecision.APPROVE)),
        config=workflow_config(cast(str, repository_insight_state["thread_id"])),
        context=resumed.dependencies,
    )
    assert final["status"] == "COMPLETED"
    assert first.transport.call_count == 0
    assert resumed.transport.call_count == 4


@pytest.mark.asyncio
@pytest.mark.graph
async def test_invalid_direct_url_fails_before_gate_provider_llm_or_artifact(
    make_repository_insight_harness,
) -> None:
    """invalid targets never reach an interrupt or any downstream boundary."""
    harness = make_repository_insight_harness()
    state = create_repository_insight_state(
        {
            "schema_version": "1",
            "workflow_id": "workflow-repository-insight-invalid",
            "thread_id": "thread-repository-insight-invalid",
            "repository_url": "https://github.com/example/repo/tree/main",
        }
    )
    graph = build_repository_insight_graph()
    final = await graph.ainvoke(
        state,
        config=workflow_config(cast(str, state["thread_id"])),
        context=harness.dependencies,
    )
    assert final["failure_code"] == "GITHUB_REPOSITORY_URL_INVALID"
    assert "__interrupt__" not in final
    assert harness.transport.call_count == 0
    assert not harness.workspace.exists()
