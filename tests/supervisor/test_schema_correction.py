"""Ordinary missing-tool/schema correction; no live provider or raw response recovery."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from vision_research_ops.adapters.llm.supervisor import DashScopeSupervisorPolicy
from vision_research_ops.application.services.conversation_intent import FixtureIntentRouter
from vision_research_ops.application.services.supervisor_models import (
    ModelCallBudget,
    SupervisorError,
    SupervisorProposal,
)
from vision_research_ops.application.workflows.supervisor import build_supervisor_graph
from vision_research_ops.cli.conversation import ConversationSession
from vision_research_ops.settings import Settings


def missing_call():
    return {
        "raw": AIMessage(content='{"action":"code","target":"private response"}'),
        "parsed": None,
        "parsing_error": None,
    }


def valid_call():
    return {
        "raw": AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "SupervisorProposal",
                    "args": {"action": "finish"},
                    "id": "call-1",
                }
            ],
        ),
        "parsed": SupervisorProposal(action="finish"),
        "parsing_error": None,
    }


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.schemas = []

    def with_structured_output(self, schema, **kwargs):
        self.schemas.append((schema, kwargs))
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def policy(monkeypatch, responses):
    model = SequenceModel(responses)
    monkeypatch.setattr(
        "vision_research_ops.adapters.llm.supervisor.build_dashscope_chat_model", lambda _: model
    )
    return DashScopeSupervisorPolicy(Settings()), model


@pytest.mark.asyncio
async def test_valid_first_response_has_one_call(monkeypatch):
    live, model = policy(monkeypatch, [valid_call()])
    budget = ModelCallBudget()
    assert (await live.decide("unsupported request", [], budget)).action == "finish"
    assert len(model.calls) == budget.used == 1


@pytest.mark.asyncio
async def test_missing_call_corrects_once_with_same_schema_and_no_raw_text(monkeypatch):
    live, model = policy(monkeypatch, [missing_call(), valid_call()])
    budget = ModelCallBudget()
    assert (await live.decide("unsupported request", [], budget)).action == "finish"
    assert len(model.calls) == budget.used == 2
    assert (
        model.schemas[0]
        == model.schemas[1]
        == (SupervisorProposal, {"method": "function_calling", "include_raw": True})
    )
    assert model.calls[1][:-1] == model.calls[0]
    correction = model.calls[1][-1][1]
    assert "response:missing_structured_tool_call" in correction
    assert "private response" not in json.dumps(model.calls)


@pytest.mark.asyncio
async def test_pydantic_feedback_is_allowlisted_without_bad_values_or_field_names(monkeypatch):
    try:
        SupervisorProposal.model_validate({"action": "private action", "private field": "secret"})
    except ValidationError as error:
        invalid = {"parsed": None, "parsing_error": error}
    live, model = policy(monkeypatch, [invalid, valid_call()])
    assert (await live.decide("unsupported request", [], ModelCallBudget())).action == "finish"
    correction = model.calls[1][-1][1]
    assert "action:literal_error" in correction
    assert "response:extra_forbidden" in correction
    assert all(word not in correction for word in ("private action", "private field", "secret"))


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError("private"), RuntimeError("private auth detail")])
async def test_provider_or_timeout_never_becomes_schema_retry(monkeypatch, error):
    live, model = policy(monkeypatch, [error, valid_call()])
    budget = ModelCallBudget()
    with pytest.raises(SupervisorError, match=r"^SUPERVISOR_PROVIDER_FAILED$"):
        await live.decide("unsupported request", [], budget)
    assert len(model.calls) == budget.used == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limit,code,count",
    [
        (24, "SUPERVISOR_SCHEMA_FAILED", 2),
        (1, "SUPERVISOR_MODEL_BUDGET_EXHAUSTED", 1),
    ],
)
async def test_final_failure_or_budget_exhaustion_cannot_dispatch(monkeypatch, limit, code, count):
    live, model = policy(monkeypatch, [missing_call(), missing_call(), valid_call()])
    budget = ModelCallBudget(limit)

    async def worker(task):
        pytest.fail("invalid proposal cannot dispatch a worker")

    graph = build_supervisor_graph(policy=live, worker=worker, budget=budget)
    result = await graph.ainvoke(
        {
            "goal": "https://github.com/kuangliu/pytorch-cifar",
            "observations": [],
            "proposal": {},
            "delegations": 0,
            "failure_code": None,
        },
        config={"configurable": {"thread_id": "schema-failure"}},
    )
    assert result["failure_code"] == code
    assert result["delegations"] == 0
    assert result["observations"] == []
    assert len(model.calls) == budget.used == count


@pytest.mark.asyncio
async def test_final_schema_failure_preserves_successful_working_context(monkeypatch, tmp_path):
    context_path = tmp_path / "context.json"
    session = ConversationSession(
        workspace=tmp_path / "work",
        mode="fixture",
        router=FixtureIntentRouter(),
        context_path=context_path,
        output_fn=lambda _: None,
        input_fn=lambda _: pytest.fail("invalid proposal cannot create a gate"),
    )
    await session.handle("https://arxiv.org/abs/2608.01234")
    before = context_path.read_bytes()
    live, model = policy(monkeypatch, [missing_call(), missing_call()])
    session._supervisor_policy = live
    await session.handle("https://github.com/kuangliu/pytorch-cifar")
    assert len(model.calls) == 2
    assert context_path.read_bytes() == before
