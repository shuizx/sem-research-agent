"""Fixed research workflow LangGraph, structured LLM, gate, resume, and JSON tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.fakes import FailureStep, ScriptedPaperProvider
from vision_research_ops.application.services.paper_models import ResearchResult
from vision_research_ops.application.state import WorkflowState, workflow_state_as_jsonable
from vision_research_ops.application.workflows import build_research_graph, workflow_config
from vision_research_ops.domain import (
    Approval,
    ApprovalDecision,
    GateKind,
    PatchOperation,
    PatchOperationType,
    WorkflowPhase,
    WorkflowStatus,
)
from vision_research_ops.ports import PaperSearchPage, make_failure

from .conftest import (
    FIXED_NOW,
    ResearchHarness,
    excluded_paper,
    make_request,
    raw_paper,
)


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
    selected_ids: list[str] | None = None,
) -> Approval:
    edits: list[PatchOperation] = []
    if decision is ApprovalDecision.EDIT:
        edits = [
            PatchOperation(
                schema_version="1",
                op=PatchOperationType.REPLACE,
                path="/selected_paper_ids",
                value=selected_ids or [],
                reason="The reviewer selected a smaller candidate subset.",
            )
        ]
    return Approval(
        schema_version="1",
        approval_id=approval_id,
        gate_kind=GateKind.CANDIDATE_SELECTION,
        subject_type=cast(str, payload["subject_type"]),
        subject_id=cast(str, payload["subject_id"]),
        subject_revision=cast(int, payload["subject_revision"]),
        decision=decision,
        edits=edits,
        reason="Pipeline reviewer decision for the research candidate slate.",
        actor_id="pipeline-reviewer",
        decided_at=FIXED_NOW,
        idempotency_key=f"idempotency-{approval_id}",
    )


async def _pause(
    harness: ResearchHarness,
    initial_state: WorkflowState,
    *,
    saver: InMemorySaver | None = None,
):
    graph = build_research_graph(checkpointer=saver)
    result = await graph.ainvoke(
        initial_state,
        config=workflow_config(initial_state["thread_id"]),
        context=harness.dependencies,
    )
    return graph, cast(dict[str, object], result)


@pytest.mark.asyncio
@pytest.mark.graph
async def test_fixture_path_interrupts_after_filter_and_structured_llm(
    make_research_harness,
    research_initial_state: WorkflowState,
) -> None:
    """One eligible paper reaches the real candidate gate; excluded papers skip LLM."""
    harness = make_research_harness()
    _, paused = await _pause(harness, research_initial_state)
    payload = _gate_payload(paused)
    assert payload["gate_kind"] == "CANDIDATE_SELECTION"
    assert payload["subject_revision"] == 1
    recommended = cast(list[dict[str, object]], payload["recommended_papers"])
    assert [item["paper_id"] for item in recommended] == ["paper-arxiv-2608.01234"]
    assert harness.provider.call_count("paper.search") == 1
    assert harness.llm.call_count("llm.generate") == 1
    assert paused["status"] == WorkflowStatus.WAITING_FOR_HUMAN
    assert paused["phase"] == WorkflowPhase.AWAITING_CANDIDATE_SELECTION
    assert harness.store.watermark_path.exists() is False


@pytest.mark.asyncio
@pytest.mark.graph
async def test_provider_failure_terminates_without_llm_gate_or_watermark(
    make_research_harness,
    research_initial_state: WorkflowState,
) -> None:
    """A typed provider failure remains explicit and does not advance daily progress."""
    harness = make_research_harness()
    harness.dependencies.paper_provider = ScriptedPaperProvider(
        provider_name="arxiv",
        script={
            "paper.search": [
                FailureStep(
                    make_failure(
                        code="ARXIV_FIXTURE_PROVIDER_UNAVAILABLE",
                        category="PROVIDER",
                        message="The scripted arXiv provider is unavailable.",
                        retryable=True,
                        ctx=None,
                    )
                )
            ]
        },
    )
    graph = build_research_graph()
    final = await graph.ainvoke(
        research_initial_state,
        config=workflow_config(research_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.FAILED
    assert final["phase"] == WorkflowPhase.FAILED
    assert final["last_error"].code == "ARXIV_FIXTURE_PROVIDER_UNAVAILABLE"
    assert "__interrupt__" not in final
    assert harness.llm.call_count("llm.generate") == 0
    assert harness.store.watermark_path.exists() is False


@pytest.mark.asyncio
@pytest.mark.graph
async def test_approve_persists_selected_result_and_watermark(
    make_research_harness,
    research_initial_state: WorkflowState,
) -> None:
    """APPROVE completes with a canonical evidence-rich JSON result."""
    harness = make_research_harness()
    graph, paused = await _pause(harness, research_initial_state)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-research-approve",
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(research_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert final["phase"] == WorkflowPhase.COMPLETED
    assert final["selected_paper_ids"] == ["paper-arxiv-2608.01234"]
    assert final["report_id"] == "research/workflow-research-1/papers.json"

    result = harness.store.load_result("workflow-research-1")
    assert result.status == "COMPLETED"
    assert result.selected_paper_ids == ["paper-arxiv-2608.01234"]
    selected = next(item for item in result.assessments if item.selected)
    assert selected.applicability is not None
    assert selected.generation is not None
    assert selected.generation.model_id == "qwen-plus-fixture"
    assert selected.generation.prompt_template_id == "paper_applicability_sem_classification"
    assert selected.generation.prompt_version == "1.2.0"
    assert selected.generation.prompt_hash.startswith("sha256:")
    assert selected.generation.output_hash.startswith("sha256:")
    assert selected.applicability.evidence
    assert selected.applicability.risks
    assert selected.applicability.summary
    assert selected.paper.provenance[0].source_url == "https://arxiv.org/abs/2608.01234v1"
    assert harness.store.load_watermark().last_successful_run_at == FIXED_NOW
    assert harness.recorder.get("approval-research-approve") == approval
    workflow_state_as_jsonable(cast(Mapping[str, object], final))


@pytest.mark.asyncio
@pytest.mark.graph
@pytest.mark.parametrize("decision", [ApprovalDecision.REJECT, ApprovalDecision.EDIT])
async def test_reject_and_edit_routes(
    make_research_harness,
    research_initial_state: WorkflowState,
    decision: ApprovalDecision,
) -> None:
    """REJECT ends without selection; EDIT accepts an explicit recommended subset."""
    second = raw_paper(
        record_id="2608.07777v1",
        arxiv_id="2608.07777v1",
        doi="10.1000/sem.2026.2",
        title="PyTorch Classification of Microscopy Defects",
        published_at="2026-08-11T09:30:00Z",
        code_urls=["https://github.com/example/microscopy-classifier"],
    )
    page = PaperSearchPage(
        schema_version="1",
        provider_name="arxiv",
        records=[raw_paper(), second],
        next_cursor=None,
        provider_request_id="arxiv-two-eligible",
        retrieved_at=FIXED_NOW,
    )
    harness = make_research_harness(pages={None: page})
    graph, paused = await _pause(harness, research_initial_state)
    payload = _gate_payload(paused)
    chosen = ["paper-arxiv-2608.07777"] if decision is ApprovalDecision.EDIT else None
    approval = _approval(
        payload,
        decision=decision,
        approval_id=f"approval-research-{decision.value.casefold()}",
        selected_ids=chosen,
    )
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(research_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    result = harness.store.load_result("workflow-research-1")
    if decision is ApprovalDecision.REJECT:
        assert final["status"] == WorkflowStatus.REJECTED
        assert result.status == "REJECTED"
        assert result.selected_paper_ids == []
    else:
        assert final["status"] == WorkflowStatus.SUCCEEDED
        assert result.status == "COMPLETED"
        assert result.selected_paper_ids == chosen


@pytest.mark.asyncio
@pytest.mark.graph
async def test_new_graph_resumes_without_retrieval_or_llm_replay(
    make_research_harness,
    research_initial_state: WorkflowState,
    tmp_path: Path,
) -> None:
    """A new graph instance resumes the interrupted thread using persisted papers.json."""
    root = tmp_path / "shared-research"
    first = make_research_harness(root=root)
    saver = InMemorySaver()
    _, paused = await _pause(first, research_initial_state, saver=saver)
    approval = _approval(
        _gate_payload(paused),
        decision=ApprovalDecision.APPROVE,
        approval_id="approval-research-resume",
    )
    resumed = make_research_harness(root=root)
    graph = build_research_graph(checkpointer=saver)
    final = await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(research_initial_state["thread_id"]),
        context=resumed.dependencies,
    )
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert resumed.provider.call_count("paper.search") == 0
    assert resumed.llm.call_count("llm.generate") == 0
    assert resumed.store.load_result("workflow-research-1").selected_paper_ids == [
        "paper-arxiv-2608.01234"
    ]


@pytest.mark.asyncio
@pytest.mark.graph
async def test_invalid_structured_llm_output_fails_explicitly_without_gate(
    make_research_harness,
    research_initial_state: WorkflowState,
) -> None:
    """Schema failure becomes terminal StructuredFailure and never writes a watermark."""
    harness = make_research_harness(llm_outputs={"paper_applicability": {}})
    graph = build_research_graph()
    final = await graph.ainvoke(
        research_initial_state,
        config=workflow_config(research_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.FAILED
    assert final["phase"] == WorkflowPhase.FAILED
    assert final["last_error"].code == "LLM_SCHEMA_VALIDATION_FAILED"
    assert "__interrupt__" not in final
    assert harness.store.watermark_path.exists() is False


@pytest.mark.asyncio
@pytest.mark.graph
async def test_code_missing_paper_reaches_daily_llm_and_candidate_gate(
    make_research_harness,
    research_initial_state: WorkflowState,
) -> None:
    """a code-less but otherwise matching paper remains a daily candidate."""
    no_code = raw_paper(
        code_urls=[],
        abstract="Image classification for wafer SEM defects using Python and PyTorch.",
        comment="No public code link appears in this metadata.",
    )
    page = PaperSearchPage(
        schema_version="1",
        provider_name="arxiv",
        records=[no_code],
        next_cursor=None,
        provider_request_id="arxiv-no-code",
        retrieved_at=FIXED_NOW,
    )
    harness = make_research_harness(pages={None: page})
    _, paused = await _pause(harness, research_initial_state)

    assert harness.llm.call_count("llm.generate") == 1
    payload = _gate_payload(paused)
    assert [item["paper_id"] for item in payload["recommended_papers"]] == [
        "paper-arxiv-2608.01234"
    ]
    result = harness.store.load_result("workflow-research-1")
    assessment = result.assessments[0]
    assert assessment.hard_filter.code_available is False
    assert assessment.applicability is not None
    assert "RESEARCH_HARD_MISSING_CODE" not in {
        reason.code for reason in assessment.hard_filter.reasons
    }


@pytest.mark.asyncio
@pytest.mark.graph
async def test_no_eligible_candidates_completes_without_llm_or_gate(
    make_research_harness,
    research_initial_state: WorkflowState,
) -> None:
    """A normal empty-candidate day is a successful persisted result, not a fake failure."""
    page = PaperSearchPage(
        schema_version="1",
        provider_name="arxiv",
        records=[excluded_paper()],
        next_cursor=None,
        provider_request_id="arxiv-excluded-only",
        retrieved_at=FIXED_NOW,
    )
    harness = make_research_harness(
        pages={None: page},
        request=make_request(max_llm_calls=0),
    )
    graph = build_research_graph()
    final = await graph.ainvoke(
        research_initial_state,
        config=workflow_config(research_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    assert final["status"] == WorkflowStatus.SUCCEEDED
    assert final["phase"] == WorkflowPhase.COMPLETED
    assert final["route"] == "DONE"
    assert "__interrupt__" not in final
    assert harness.llm.call_count("llm.generate") == 0
    result = ResearchResult.model_validate_json(
        harness.store.result_path("workflow-research-1").read_text(encoding="utf-8")
    )
    assert result.status == "NO_CANDIDATES"
    assert harness.store.load_watermark() is not None
