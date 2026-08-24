"""integrated offline Pipeline Sample acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from vision_research_ops.application.services.evaluation_models import EvaluationResult
from vision_research_ops.application.services.pipeline_models import (
    LocalPipelineSummaryStore,
    PipelineScenario,
    PipelineSummary,
    canonical_json_bytes,
)
from vision_research_ops.application.workflows.pipeline import build_pipeline_graph
from vision_research_ops.cli.pipeline import (
    PipelineCliOptions,
    build_parser,
    parse_options,
    run,
)
from vision_research_ops.pipeline.decisions import DecisionName

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _options(
    workspace: Path,
    workflow_id: str,
    *,
    scenario: str = "happy",
    auto_approve: bool = False,
    decisions: tuple[DecisionName, ...] = (),
) -> PipelineCliOptions:
    return PipelineCliOptions(
        mode="fixture",
        adaptation_planner_mode="scripted",
        workspace=workspace,
        workflow_id=workflow_id,
        scenario=cast(PipelineScenario, scenario),
        auto_approve_sample=auto_approve,
        decisions=decisions,
    )


def _summary(workspace: Path, workflow_id: str) -> PipelineSummary:
    return LocalPipelineSummaryStore(workspace / "var").load_summary(workflow_id)


def _assert_canonical_summary(workspace: Path, summary: PipelineSummary) -> bytes:
    path = workspace / "var" / Path(*summary.summary_ref.split("/"))
    payload = path.read_bytes()
    assert payload == canonical_json_bytes(summary.model_dump(mode="json"))
    assert str(workspace.resolve()) not in payload.decode("utf-8")
    return payload


@pytest.mark.e2e
def test_help_and_top_level_graph_make_fixture_scope_and_failure_routes_visible(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """the CLI is bounded and the orchestrator is a real conditional graph."""
    with pytest.raises(SystemExit) as exited:
        build_parser().parse_args(["sample", "--help"])
    assert exited.value.code == 0
    help_text = capsys.readouterr().out
    assert "--workspace" in help_text
    assert "--scenario {happy,smoke-failure}" in help_text
    assert "--adaptation-planner {scripted,dashscope}" in help_text
    assert "--auto-approve-sample" in help_text
    assert "--decisions" in help_text

    parsed = parse_options(
        [
            "sample",
            "--workspace",
            "sample-workspace",
            "--decisions",
            "edit,reject",
        ]
    )
    assert parsed.decisions == ("edit", "reject")
    assert parsed.scripted_fixture_decisions is True

    graph = build_pipeline_graph().get_graph()
    assert set(graph.nodes) == {
        "__start__",
        "research",
        "repository",
        "adaptation",
        "training",
        "evaluation",
        "summarize",
        "__end__",
    }
    edges = {(edge.source, edge.target, edge.data) for edge in graph.edges}
    assert ("__start__", "research", None) in edges
    assert ("research", "repository", "CONTINUE") in edges
    assert ("repository", "adaptation", "CONTINUE") in edges
    assert ("adaptation", "training", "CONTINUE") in edges
    assert ("training", "evaluation", "CONTINUE") in edges
    assert ("evaluation", "summarize", "CONTINUE") in edges
    assert ("summarize", "__end__", None) in edges
    for stage in ("research", "repository", "adaptation", "training"):
        assert (stage, "summarize", "STOP") in edges


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_fixture_happy_path_runs_research_to_evaluation_and_exact_rerun_is_stable(
    tmp_path: Path,
) -> None:
    """one offline command produces a complete honest chain."""
    workspace = tmp_path / "happy"
    workflow_id = "pipeline-e2e-happy"
    options = _options(workspace, workflow_id, auto_approve=True)
    output: list[str] = []

    assert await run(options, output_fn=output.append, source_root=PROJECT_ROOT) == 0
    summary = _summary(workspace, workflow_id)
    summary_path = workspace / "var" / Path(*summary.summary_ref.split("/"))
    payload = _assert_canonical_summary(workspace, summary)
    mtime = summary_path.stat().st_mtime_ns

    assert summary.status == "SUCCEEDED"
    assert summary.conclusion == "IMPROVED"
    assert summary.fixture_labeled is True
    assert summary.scripted_fixture_decisions is True
    assert summary.decision_config.mode == "auto_approve_sample"
    assert summary.decision_config.decisions == []
    assert summary.synthetic_or_public_data_only is True
    assert summary.real_pytorch_training is False
    assert summary.real_company_evaluation is False
    assert [stage.status for stage in summary.stages] == ["SUCCEEDED"] * 5
    assert [stage.stage for stage in summary.stages] == [
        "research",
        "repository",
        "adaptation",
        "training",
        "evaluation",
    ]
    assert [stage.workflow_id for stage in summary.stages] == [
        f"{workflow_id}-research",
        f"{workflow_id}-repository",
        f"{workflow_id}-adaptation",
        f"{workflow_id}-training",
        f"{workflow_id}-evaluation",
    ]
    assert [gate.gate_kind for gate in summary.gates] == [
        "CANDIDATE_SELECTION",
        "REPOSITORY_INGEST",
        "PATCH_ACCEPTANCE",
        "RUN_SUBMISSION",
    ]
    assert all(gate.decision == "APPROVE" for gate in summary.gates)
    assert all(gate.scripted_fixture_decision is True for gate in summary.gates)
    assert summary.resume_count == 4

    assert summary.paper is not None
    assert summary.paper.paper_id == "paper-arxiv-2608.01234"
    assert summary.repository is not None
    assert summary.repository.fixed_commit_sha == "a" * 40
    assert summary.adaptation is not None
    assert summary.adaptation.smoke_capability == "FIXTURE_CONTRACT_PROBE_NO_TORCH"
    assert summary.adaptation_planner_mode == "scripted"
    assert summary.adaptation.planner_kind == "SCRIPTED_TOOL_CALLING"
    assert summary.adaptation.planner_tools == [
        "inspect_repository_profile",
        "inspect_dataset_contract",
        "compare_repository_dataset",
        "validate_adaptation_plan",
    ]
    trace_path = workspace / "var" / Path(*summary.adaptation.planner_trace_ref.split("/"))
    trace_text = trace_path.read_text(encoding="utf-8")
    assert "arguments_hash" in trace_text
    assert "output_hash" in trace_text
    assert "label_names" not in trace_text
    assert summary.training is not None
    assert summary.training.capability == "SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"
    assert summary.evaluation is not None
    assert summary.evaluation.capability == "DETERMINISTIC_SINGLE_PAIR_FIXTURE_EVALUATION"

    evaluation_path = workspace / "var" / Path(*summary.evaluation.evaluation_ref.split("/"))
    report_path = workspace / "var" / Path(*summary.evaluation.report_ref.split("/"))
    evaluation = EvaluationResult.model_validate_json(evaluation_path.read_bytes())
    assert evaluation.conclusion == summary.conclusion
    assert evaluation.baseline_metrics is not None
    assert evaluation.candidate_metrics is not None
    assert evaluation.deltas is not None
    assert summary.evaluation.metrics.baseline_macro_f1 == evaluation.baseline_metrics.macro_f1
    assert summary.evaluation.metrics.candidate_macro_f1 == evaluation.candidate_metrics.macro_f1
    assert summary.evaluation.metrics.macro_f1_delta == evaluation.deltas.macro_f1
    assert summary.evaluation.metrics.balanced_accuracy_delta == (
        evaluation.deltas.balanced_accuracy
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Conclusion: **IMPROVED**" in report
    assert "Real PyTorch training: `false`" in report
    assert "Real company evaluation: `false`" in report
    assert {child.name for child in workspace.iterdir()} == {"var"}

    rerun_output: list[str] = []
    assert await run(options, output_fn=rerun_output.append, source_root=PROJECT_ROOT) == 0
    assert summary_path.read_bytes() == payload
    assert summary_path.stat().st_mtime_ns == mtime
    rerun_event = json.loads(rerun_output[-1])
    assert rerun_event["reused_existing"] is True

    conflict_output: list[str] = []
    conflicting = _options(
        workspace,
        workflow_id,
        scenario="smoke-failure",
        auto_approve=True,
    )
    assert (
        await run(
            conflicting,
            output_fn=conflict_output.append,
            source_root=PROJECT_ROOT,
        )
        == 2
    )
    assert json.loads(conflict_output[-1])["error"] == "PIPELINE_SUMMARY_CONFLICT"
    assert summary_path.read_bytes() == payload

    explicit_output: list[str] = []
    explicit_approvals = _options(
        workspace,
        workflow_id,
        decisions=("approve", "approve", "approve", "approve"),
    )
    assert (
        await run(
            explicit_approvals,
            output_fn=explicit_output.append,
            source_root=PROJECT_ROOT,
        )
        == 2
    )
    assert json.loads(explicit_output[-1])["error"] == "PIPELINE_SUMMARY_CONFLICT"
    assert summary_path.read_bytes() == payload


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_dashscope_planner_mode_without_key_fails_explicitly_before_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tool-calling planner live opt-in never silently falls back to scripted planning."""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    options = PipelineCliOptions(
        mode="fixture",
        adaptation_planner_mode="dashscope",
        workspace=tmp_path / "missing-key",
        workflow_id="pipeline-missing-dashscope-key",
        scenario="happy",
        auto_approve_sample=True,
        decisions=(),
    )
    output: list[str] = []
    assert await run(options, output_fn=output.append, source_root=PROJECT_ROOT) == 2
    assert json.loads(output[-1]) == {
        "error": "DASHSCOPE_API_KEY_REQUIRED",
        "status": "FAILED",
    }
    assert not (options.workspace / "var" / "sample").exists()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_explicit_decision_identity_blocks_changed_script_without_rewrite(
    tmp_path: Path,
) -> None:
    """regression: complete ordered decisions are part of exact replay identity."""
    workspace = tmp_path / "decision-identity"
    workflow_id = "pipeline-e2e-decision-identity"
    approvals = _options(
        workspace,
        workflow_id,
        decisions=("approve", "approve", "approve", "approve"),
    )

    assert await run(approvals, output_fn=lambda _event: None, source_root=PROJECT_ROOT) == 0
    summary = _summary(workspace, workflow_id)
    summary_path = workspace / "var" / Path(*summary.summary_ref.split("/"))
    payload = summary_path.read_bytes()
    mtime = summary_path.stat().st_mtime_ns
    assert summary.schema_version == "3"
    assert summary.decision_config.mode == "explicit_ordered"
    assert summary.decision_config.decisions == ["approve"] * 4
    assert summary.decision_config.fingerprint.startswith("sha256:")

    replay_output: list[str] = []
    assert (
        await run(
            approvals,
            output_fn=replay_output.append,
            source_root=PROJECT_ROOT,
        )
        == 0
    )
    assert json.loads(replay_output[-1])["reused_existing"] is True
    assert summary_path.read_bytes() == payload
    assert summary_path.stat().st_mtime_ns == mtime

    changed_output: list[str] = []
    changed = _options(workspace, workflow_id, decisions=("reject",))
    assert (
        await run(
            changed,
            output_fn=changed_output.append,
            source_root=PROJECT_ROOT,
        )
        == 2
    )
    assert json.loads(changed_output[-1]) == {
        "error": "PIPELINE_SUMMARY_CONFLICT",
        "status": "FAILED",
    }
    assert summary_path.read_bytes() == payload
    assert summary_path.stat().st_mtime_ns == mtime


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_smoke_failure_stops_before_training_and_writes_failure_summary(
    tmp_path: Path,
) -> None:
    """A repeated smoke failure cannot create training or evaluation artifacts."""
    workspace = tmp_path / "smoke-failure"
    workflow_id = "pipeline-e2e-smoke-failure"
    options = _options(
        workspace,
        workflow_id,
        scenario="smoke-failure",
        auto_approve=True,
    )

    assert await run(options, output_fn=lambda _event: None, source_root=PROJECT_ROOT) == 2
    summary = _summary(workspace, workflow_id)
    _assert_canonical_summary(workspace, summary)

    assert summary.status == "FAILED"
    assert summary.failure_reason is not None
    assert summary.failure_reason.code == "ADAPTATION_SMOKE_FAILED_AFTER_REPAIR"
    assert summary.failure_reason.stage == "adaptation"
    assert [stage.status for stage in summary.stages] == [
        "SUCCEEDED",
        "SUCCEEDED",
        "FAILED",
        "NOT_RUN",
        "NOT_RUN",
    ]
    assert [gate.gate_kind for gate in summary.gates] == [
        "CANDIDATE_SELECTION",
        "REPOSITORY_INGEST",
    ]
    assert summary.resume_count == 2
    assert summary.training is None
    assert summary.evaluation is None
    adaptation_refs = summary.stages[2].artifact_refs
    assert any(ref.endswith("/r1/result.json") for ref in adaptation_refs)
    assert any(ref.endswith("/r2/result.json") for ref in adaptation_refs)
    assert not (workspace / "var" / "fixture-project" / "var" / "training").exists()
    assert not (workspace / "var" / "fixture-project" / "var" / "reports").exists()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_scripted_patch_reject_and_interactive_candidate_reject_are_distinct(
    tmp_path: Path,
) -> None:
    """reject is terminal and scripted provenance never impersonates a human."""
    scripted_workspace = tmp_path / "scripted-reject"
    scripted_id = "pipeline-e2e-scripted-reject"
    scripted = _options(
        scripted_workspace,
        scripted_id,
        decisions=("approve", "approve", "reject"),
    )
    assert (
        await run(
            scripted,
            output_fn=lambda _event: None,
            source_root=PROJECT_ROOT,
        )
        == 2
    )
    scripted_summary = _summary(scripted_workspace, scripted_id)
    assert scripted_summary.status == "STOPPED"
    assert scripted_summary.scripted_fixture_decisions is True
    assert [gate.decision for gate in scripted_summary.gates] == [
        "APPROVE",
        "APPROVE",
        "REJECT",
    ]
    assert scripted_summary.gates[-1].gate_kind == "PATCH_ACCEPTANCE"
    assert all(gate.scripted_fixture_decision is True for gate in scripted_summary.gates)
    assert scripted_summary.stages[3].status == "NOT_RUN"
    assert scripted_summary.stages[4].status == "NOT_RUN"
    assert scripted_summary.training is None
    assert scripted_summary.evaluation is None
    assert not (scripted_workspace / "var" / "fixture-project" / "var" / "training").exists()

    interactive_workspace = tmp_path / "interactive-reject"
    interactive_id = "pipeline-e2e-interactive-reject"
    answers = iter(["reject"])
    interactive_output: list[str] = []
    assert (
        await run(
            _options(interactive_workspace, interactive_id),
            input_fn=lambda _prompt: next(answers),
            output_fn=interactive_output.append,
            source_root=PROJECT_ROOT,
        )
        == 2
    )
    interactive_summary = _summary(interactive_workspace, interactive_id)
    assert interactive_summary.status == "STOPPED"
    assert interactive_summary.scripted_fixture_decisions is False
    assert len(interactive_summary.gates) == 1
    assert interactive_summary.gates[0].gate_kind == "CANDIDATE_SELECTION"
    assert interactive_summary.gates[0].decision == "REJECT"
    assert interactive_summary.gates[0].scripted_fixture_decision is False
    assert [stage.status for stage in interactive_summary.stages[1:]] == ["NOT_RUN"] * 4
    assert any("gate_pending" in event for event in interactive_output)
