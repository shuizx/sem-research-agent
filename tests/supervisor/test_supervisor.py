"""Ordinary offline supervisor decisions, evidence boundaries and real gate resume."""
# ruff: noqa: RUF001

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt

from vision_research_ops.adapters.llm.supervisor import (
    DashScopeSupervisorPolicy,
    ScriptedSupervisorPolicy,
)
from vision_research_ops.application.services.conversation_intent import FixtureIntentRouter
from vision_research_ops.application.services.supervisor_models import (
    ModelCallBudget,
    SupervisorError,
    SupervisorProposal,
    WorkerResult,
    WorkerTask,
)
from vision_research_ops.application.workflows.supervisor import build_supervisor_graph
from vision_research_ops.cli.conversation import ConversationSession
from vision_research_ops.settings import Settings


def initial(goal: str) -> dict[str, object]:
    return {
        "goal": goal,
        "observations": [],
        "proposal": {},
        "delegations": 0,
        "failure_code": None,
    }


@pytest.mark.asyncio
async def test_serial_compound_observes_and_uses_evidence() -> None:
    tasks = []

    async def worker(task: WorkerTask) -> WorkerResult:
        tasks.append(task)
        return WorkerResult(
            worker=task.worker,
            status="success",
            code_urls=["https://github.com/example/sem-classifier"]
            if task.worker == "research"
            else [],
        )

    graph = build_supervisor_graph(
        policy=ScriptedSupervisorPolicy(), worker=worker, budget=ModelCallBudget()
    )
    final = await graph.ainvoke(
        initial("检索最新论文，然后分析代码"), config={"configurable": {"thread_id": "compound"}}
    )
    assert [task.worker for task in tasks] == ["research", "code"]
    assert tasks[1].target == "https://github.com/example/sem-classifier"
    assert tasks[0].goal == "检索最新论文，然后分析代码"
    assert final["delegations"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "rejected", "no_results"])
async def test_old_context_cannot_turn_failure_into_success(status: str) -> None:
    async def worker(task: WorkerTask) -> WorkerResult:
        assert task.worker == "research"
        return WorkerResult.model_validate({"worker": "research", "status": status})

    graph = build_supervisor_graph(
        policy=ScriptedSupervisorPolicy(),
        worker=worker,
        budget=ModelCallBudget(),
        context_code_urls=("https://github.com/old/repo",),
    )
    final = await graph.ainvoke(
        initial("检索论文并分析代码"), config={"configurable": {"thread_id": status}}
    )
    assert final["delegations"] == 1


@pytest.mark.asyncio
async def test_child_interrupt_bubbles_and_resume_does_not_repeat_work() -> None:
    calls = []

    def retrieve(state: dict[str, object]) -> dict[str, object]:
        calls.append("retrieve")
        return state

    def gate(state: dict[str, object]) -> dict[str, object]:
        answer = interrupt({"gate_kind": "candidate"})
        return {**state, "answer": answer}

    builder = StateGraph(dict)
    builder.add_node("retrieve", retrieve)
    builder.add_node("gate", gate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "gate")
    child = builder.compile()

    async def worker(task: WorkerTask) -> WorkerResult:
        final = await child.ainvoke({})
        return WorkerResult(
            worker=task.worker, status="success" if final["answer"] == "approve" else "rejected"
        )

    graph = build_supervisor_graph(
        policy=ScriptedSupervisorPolicy(), worker=worker, budget=ModelCallBudget()
    )
    config = {"configurable": {"thread_id": "gate"}}
    paused = await graph.ainvoke(initial("检索论文并分析代码"), config=config)
    assert paused["__interrupt__"]
    assert paused["observations"] == []
    final = await graph.ainvoke(Command(resume="reject"), config=config)
    assert calls == ["retrieve"]
    assert final["delegations"] == 1
    assert final["observations"][0]["status"] == "rejected"


class RepeatingPolicy:
    async def decide(
        self, goal: str, observations: list[WorkerResult], budget: ModelCallBudget
    ) -> SupervisorProposal:
        budget.consume()
        return SupervisorProposal(action="research")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limit,code,count",
    [
        (24, "SUPERVISOR_DELEGATION_LIMIT", 3),
        (1, "SUPERVISOR_MODEL_BUDGET_EXHAUSTED", 1),
    ],
)
async def test_request_bounds(limit: int, code: str, count: int) -> None:
    async def worker(task: WorkerTask) -> WorkerResult:
        return WorkerResult(worker=task.worker, status="success")

    graph = build_supervisor_graph(
        policy=RepeatingPolicy(), worker=worker, budget=ModelCallBudget(limit)
    )
    final = await graph.ainvoke(initial("research"), config={"configurable": {"thread_id": code}})
    assert final["failure_code"] == code
    assert final["delegations"] == count


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["approve", "reject"])
async def test_actual_cli_research_gate_compound(tmp_path: Path, answer: str) -> None:
    outputs = []
    answers = iter([answer, "approve"])
    session = ConversationSession(
        workspace=tmp_path,
        mode="fixture",
        router=FixtureIntentRouter(),
        supervisor_policy=ScriptedSupervisorPolicy(),
        input_fn=lambda _: next(answers),
        output_fn=outputs.append,
    )
    await session.handle("检索最新论文，然后分析代码")
    trace = json.loads(next((tmp_path / "supervisor").glob("*.json")).read_text("utf-8"))
    assert trace["failure_code"] is None, outputs
    assert trace["delegations"] == (2 if answer == "approve" else 1), outputs
    assert trace["observations"][0]["status"] == ("success" if answer == "approve" else "rejected")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "goal,expected",
    [
        ("https://arxiv.org/abs/2608.01234", "research"),
        ("https://github.com/example/sem-classifier", "code"),
    ],
)
async def test_direct_requests_do_not_expand(goal: str, expected: str) -> None:
    calls = []

    async def worker(task: WorkerTask) -> WorkerResult:
        calls.append(task.worker)
        return WorkerResult(
            worker=task.worker,
            status="success",
            code_urls=["https://github.com/example/sem-classifier"],
        )

    graph = build_supervisor_graph(
        policy=ScriptedSupervisorPolicy(), worker=worker, budget=ModelCallBudget()
    )
    await graph.ainvoke(initial(goal), config={"configurable": {"thread_id": expected}})
    assert calls == [expected]


class InvalidPolicy:
    def __init__(self, action: str, target: str | None = None) -> None:
        self.action = action
        self.target = target

    async def decide(
        self, goal: str, observations: list[WorkerResult], budget: ModelCallBudget
    ) -> SupervisorProposal:
        return SupervisorProposal.model_validate({"action": self.action, "target": self.target})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,target,code",
    [
        ("shell", None, "SUPERVISOR_PROVIDER_OR_SCHEMA_FAILED"),
        ("code", "https://github.com/invented/repo", "SUPERVISOR_UNSUPPORTED_REPOSITORY_TARGET"),
    ],
)
async def test_untrusted_proposals_cannot_add_tools_or_targets(
    action: str, target: str | None, code: str
) -> None:
    async def worker(task: WorkerTask) -> WorkerResult:
        pytest.fail("unapproved target/tool must not reach worker")

    graph = build_supervisor_graph(
        policy=InvalidPolicy(action, target), worker=worker, budget=ModelCallBudget()
    )
    final = await graph.ainvoke(initial("research"), config={"configurable": {"thread_id": action}})
    assert final["failure_code"] == code
    assert final["delegations"] == 0


class FakeSupervisorModel:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0

    def with_structured_output(self, *args: object, **kwargs: object) -> FakeSupervisorModel:
        return self

    async def ainvoke(self, messages: object) -> object:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,code",
    [
        ({"parsed": '{"action": "finish"}'}, "SUPERVISOR_SCHEMA_FAILED"),
        (RuntimeError("private provider detail"), "SUPERVISOR_PROVIDER_FAILED"),
    ],
)
async def test_live_boundary_is_strict_and_counted(
    monkeypatch: pytest.MonkeyPatch, response: object, code: str
) -> None:
    model = FakeSupervisorModel(response)
    monkeypatch.setattr(
        "vision_research_ops.adapters.llm.supervisor.build_dashscope_chat_model", lambda _: model
    )
    policy = DashScopeSupervisorPolicy(Settings())
    budget = ModelCallBudget(1)
    with pytest.raises(SupervisorError, match=code):
        await policy.decide("research", [], budget)
    assert budget.used == model.calls == 1
    with pytest.raises(SupervisorError, match="BUDGET_EXHAUSTED"):
        await policy.decide("research", [], budget)
    assert model.calls == 1
    with pytest.raises(SupervisorError, match="UNSAFE"):
        await policy.decide("read C:/private/images", [], ModelCallBudget())
    assert model.calls == 1
