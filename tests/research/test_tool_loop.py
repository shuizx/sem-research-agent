"""Ordinary Research observation, grounding, budgets and real Gate regressions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from vision_research_ops.application.services.paper_planner import LangGraphResearchPlanner
from vision_research_ops.application.services.supervisor_models import (
    ModelCallBudget,
    SupervisorError,
)
from vision_research_ops.application.workflows import workflow_config
from vision_research_ops.domain import ApprovalDecision
from vision_research_ops.ports import OperationContext, PaperSearchPage

from .conftest import FIXED_NOW, raw_paper, valid_decision
from .test_research_graph import _approval, _gate_payload, _pause


def call(name: str, args: dict[str, object]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])


def assessment() -> dict[str, object]:
    result = valid_decision()
    result["summary"] = "SEM defect classification based on microscopy morphology."
    result["evidence"] = [
        {"dimension": "TASK", "source_field": "abstract", "statement": "classification"}
    ]
    return result


class Agent:
    """Scripted transport choosing refinement from actual tool observations."""

    def __init__(self, *, bad_id: bool = False, bad_citation: bool = False) -> None:
        self.calls = 0
        self.bad_id = bad_id
        self.bad_citation = bad_citation
        self.goals: list[str] = []

    async def ainvoke(self, value: object) -> AIMessage:
        messages = cast(list[object], value)
        self.calls += 1
        self.goals.append(str(messages[1]))
        last = messages[-1]
        if not isinstance(last, ToolMessage):
            return call("search_papers", {"keywords": ["SEM morphology classification"]})
        if last.name == "search_papers":
            import json

            observation = json.loads(str(last.content))
            if not observation["papers"]:
                return call("search_papers", {"keywords": ["microscopy classification"]})
            return call(
                "inspect_paper",
                {"paper_id": "unknown" if self.bad_id else observation["papers"][0]["paper_id"]},
            )
        if last.name == "inspect_paper":
            decision = assessment()
            if self.bad_citation:
                decision["evidence"] = [
                    {
                        "dimension": "TASK",
                        "source_field": "abstract",
                        "statement": "invented source claim",
                    }
                ]
            return call(
                "submit_assessment", {"paper_id": "paper-arxiv-2608.01234", "assessment": decision}
            )
        return call("finish_research", {})


def attach(harness, agent: Agent, budget: ModelCallBudget | None = None):
    def bind(tools: Sequence[BaseTool]) -> Agent:
        assert {t.name for t in tools} == {
            "search_papers",
            "inspect_paper",
            "submit_assessment",
            "finish_research",
        }
        return agent

    planner = LangGraphResearchPlanner(
        bind, model_id="fake-live", before_model_call=None if budget is None else budget.consume
    )
    harness.dependencies.planner = planner
    harness.dependencies.goal = "Find morphology-based SEM classification papers"
    return planner


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", list(ApprovalDecision))
async def test_refinement_metadata_uncertainty_gate_and_no_replay(
    make_research_harness, research_initial_state, decision
):
    harness = make_research_harness()
    original = harness.dependencies.paper_provider

    class Provider:
        provider_name = "arxiv"
        calls = 0

        def __init__(self):
            self.keywords: list[list[str]] = []

        async def search(self, query, *, cursor, ctx):
            self.calls += 1
            self.keywords.append(query.query_spec.keywords)
            if self.calls == 1:
                return PaperSearchPage(
                    schema_version="1",
                    provider_name="arxiv",
                    records=[],
                    next_cursor=None,
                    provider_request_id="empty-first",
                    retrieved_at=FIXED_NOW,
                )
            page = await original.search(query, cursor=cursor, ctx=ctx)
            paper = raw_paper(
                abstract="Morphology classification of SEM defects.",
                title="SEM classification",
                comment="No implementation described.",
                code_urls=[],
            )
            return page.model_copy(update={"records": [paper, paper]})

    provider = Provider()
    harness.dependencies.paper_provider = provider
    agent = Agent()
    budget = ModelCallBudget()
    planner = attach(harness, agent, budget)
    graph, paused = await _pause(harness, research_initial_state)
    payload = _gate_payload(paused)
    item = harness.store.load_result(research_initial_state["workflow_id"]).assessments[0]
    assert item.applicability.applicable
    assert item.hard_filter.eligible
    assert item.hard_filter.mode == "EVIDENCE_ASSESSED"
    assert item.candidate.exclusion_reasons == []
    assert any("metadata unconfirmed" in x for x in item.applicability.risks)
    assert item.paper.code_urls == []
    assert len(harness.dependencies.session.papers) == 1
    assert provider.keywords == [["SEM morphology classification"], ["microscopy classification"]]
    assert "Find morphology-based" in agent.goals[0]
    assert planner.last_tools == [
        "search_papers",
        "search_papers",
        "inspect_paper",
        "submit_assessment",
        "finish_research",
    ]
    calls_before = agent.calls
    approval = _approval(
        payload, decision=decision, approval_id="loop-approval", selected_ids=[item.paper.paper_id]
    )
    await graph.ainvoke(
        Command(resume=approval.model_dump(mode="json")),
        config=workflow_config(research_initial_state["thread_id"]),
        context=harness.dependencies,
    )
    result = harness.store.load_result(research_initial_state["workflow_id"])
    assert result.status == ("REJECTED" if decision is ApprovalDecision.REJECT else "COMPLETED")
    assert agent.calls == calls_before == budget.used
    assert provider.calls == 2
    assert harness.llm.call_count("llm.generate") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id,bad_citation", [(True, False), (False, True)])
async def test_unretrieved_id_or_citation_fails(make_research_harness, bad_id, bad_citation):
    harness = make_research_harness()
    planner = attach(harness, Agent(bad_id=bad_id, bad_citation=bad_citation))
    from vision_research_ops.ports import LLMError

    with pytest.raises(LLMError, match="RESEARCH_TOOL_LOOP_FAILED"):
        await planner.analyze(harness.dependencies, context())
    assert not harness.store.watermark_path.exists()


def context() -> OperationContext:
    return OperationContext(
        schema_version="1",
        correlation_id="corr-test",
        workflow_id="test",
        actor_id="test",
        sensitivity="PUBLIC",
    )


@pytest.mark.asyncio
async def test_shared_budget_exhausts_before_research_provider_model(make_research_harness):
    harness = make_research_harness()
    agent = Agent()
    budget = ModelCallBudget(limit=1)
    budget.consume()  # preceding supervisor call
    planner = attach(harness, agent, budget)
    with pytest.raises(SupervisorError, match="BUDGET_EXHAUSTED"):
        await planner.analyze(harness.dependencies, context())
    assert agent.calls == 0


@pytest.mark.asyncio
async def test_schema_correction_charges_each_attempt_before_transport():
    from langchain_core.language_models.chat_models import BaseChatModel

    from vision_research_ops.adapters.llm.dashscope import DashScopeStructuredLLM

    from .test_dashscope_schema_recovery import (
        _request,
        _ScriptedStructuredRunnable,
        _StubChatModel,
    )

    runnable = _ScriptedStructuredRunnable([{"parsed": {}, "parsing_error": None}])
    budget = ModelCallBudget(limit=1)
    llm = DashScopeStructuredLLM(
        cast(BaseChatModel, _StubChatModel(runnable)),
        model_id="fake",
        before_model_call=budget.consume,
    )
    with pytest.raises(SupervisorError, match="BUDGET_EXHAUSTED"):
        await llm.generate(_request(), ctx=context())
    assert budget.used == len(runnable.calls) == 1


@pytest.mark.asyncio
async def test_explicit_paper_analysis_keeps_soft_metadata_notes(make_research_harness):
    from vision_research_ops.application.services.paper_analysis import (
        score_assessments,
        unscored_assessment,
    )
    from vision_research_ops.application.services.paper_retrieval import normalize_raw_paper

    harness = make_research_harness()
    paper = normalize_raw_paper(
        raw_paper(
            title="SEM classification",
            abstract="SEM classification.",
            comment="No implementation described.",
            code_urls=[],
        )
    )
    scored = await score_assessments(
        [unscored_assessment(paper, request_id="direct")],
        problem=harness.dependencies.problem_profile,
        request_id="direct",
        llm=harness.llm,
        max_llm_calls=1,
        ctx=context(),
        include_ineligible=True,
    )
    assert scored[0].candidate.exclusion_reasons == []
    assert any("metadata unconfirmed" in risk for risk in scored[0].applicability.risks)
    assert any("No public code" in risk for risk in scored[0].applicability.risks)


class Steps:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def ainvoke(self, value):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["search_limit", "schema", "provider", "code_url"])
async def test_ordinary_loop_limits_and_failures(make_research_harness, case):
    from vision_research_ops.ports import LLMError

    harness = make_research_harness()
    responses = [call("search_papers", {"keywords": ["classification"]})]
    if case == "search_limit":
        responses *= 3
    elif case == "schema":
        responses.append(call("inspect_paper", {"paper_id": "paper-arxiv-2608.01234", "extra": 1}))
    elif case == "provider":
        responses.append(RuntimeError("provider unavailable"))
    else:
        decision = assessment()
        decision["rationale"] = "Use https://github.com/invented/unsupported"
        responses.extend(
            [
                call("inspect_paper", {"paper_id": "paper-arxiv-2608.01234"}),
                call(
                    "submit_assessment",
                    {"paper_id": "paper-arxiv-2608.01234", "assessment": decision},
                ),
            ]
        )
    planner = LangGraphResearchPlanner(lambda _: Steps(responses), model_id="fake")
    with pytest.raises(LLMError, match="RESEARCH_TOOL_LOOP_FAILED"):
        await planner.analyze(harness.dependencies, context())
    assert harness.provider.call_count("paper.search") <= 2


@pytest.mark.asyncio
async def test_five_candidate_limit(make_research_harness):
    from vision_research_ops.ports import LLMError

    harness = make_research_harness(
        pages={
            None: PaperSearchPage(
                schema_version="1",
                provider_name="arxiv",
                provider_request_id="six",
                retrieved_at=FIXED_NOW,
                records=[
                    raw_paper(
                        record_id=f"2608.0000{i}",
                        arxiv_id=f"2608.0000{i}",
                        doi=None,
                        title=f"classification {i}",
                    )
                    for i in range(6)
                ],
            )
        }
    )
    responses = [call("search_papers", {"keywords": ["classification"]})]
    responses.extend(
        call("inspect_paper", {"paper_id": f"paper-arxiv-2608.0000{i}"}) for i in range(6)
    )
    planner = LangGraphResearchPlanner(lambda _: Steps(responses), model_id="fake")
    with pytest.raises(LLMError):
        await planner.analyze(harness.dependencies, context())
    assert planner.last_tools.count("inspect_paper") == 5


@pytest.mark.asyncio
async def test_explicit_no_results_advances_watermark(
    make_research_harness, research_initial_state
):
    harness = make_research_harness(
        pages={
            None: PaperSearchPage(
                schema_version="1",
                provider_name="arxiv",
                provider_request_id="empty",
                retrieved_at=FIXED_NOW,
                records=[],
            )
        }
    )
    harness.dependencies.planner = LangGraphResearchPlanner(
        lambda _: Steps(
            [call("search_papers", {"keywords": ["classification"]}), call("finish_research", {})]
        ),
        model_id="fake",
    )
    _, result = await _pause(harness, research_initial_state)
    assert not result.get("__interrupt__")
    assert (
        harness.store.load_result(research_initial_state["workflow_id"]).status == "NO_CANDIDATES"
    )
    assert harness.store.load_watermark().last_successful_run_at == FIXED_NOW


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [ApprovalDecision.APPROVE, ApprovalDecision.REJECT])
async def test_new_loop_compound_approved_evidence_only(make_research_harness, answer):
    from vision_research_ops.adapters.llm.supervisor import ScriptedSupervisorPolicy
    from vision_research_ops.application.agents.workers import research_worker
    from vision_research_ops.application.services.supervisor_models import WorkerResult
    from vision_research_ops.application.workflows.supervisor import build_supervisor_graph

    harness = make_research_harness()
    agent = Agent()
    attach(harness, agent)
    run = research_worker(harness.dependencies, "new-loop-compound")
    targets = []

    async def worker(task):
        if task.worker == "research":
            return await run()
        targets.append(task.target)
        return WorkerResult(worker="code", status="success")

    graph = build_supervisor_graph(
        policy=ScriptedSupervisorPolicy(), worker=worker, budget=ModelCallBudget()
    )
    config = workflow_config("new-loop-compound")
    paused = await graph.ainvoke(
        {
            "goal": "检索最新论文, 然后分析代码",
            "observations": [],
            "proposal": {},
            "delegations": 0,
            "failure_code": None,
        },
        config=config,
    )
    calls = agent.calls
    payload = _gate_payload(paused)
    approval = _approval(payload, decision=answer, approval_id="compound-approval")
    final = await graph.ainvoke(Command(resume=approval.model_dump(mode="json")), config=config)
    assert final["failure_code"] is None
    assert targets == (
        ["https://github.com/example/sem-classifier"] if answer is ApprovalDecision.APPROVE else []
    )
    assert agent.calls == calls
    assert harness.provider.call_count("paper.search") == 1
