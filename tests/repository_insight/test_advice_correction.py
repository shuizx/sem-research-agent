"""Ordinary final-advice correction through the real bounded Code ToolNode."""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from vision_research_ops.application.services.conversation_intent import FixtureIntentRouter
from vision_research_ops.application.services.repository_insight_models import (
    RepositoryAdaptationAdvice,
    RepositoryInsightGeneration,
    structure_summary,
)
from vision_research_ops.application.services.repository_insight_planner import (
    MAX_MODEL_TURNS,
    MAX_READ_FILES,
    MAX_TOOL_CALLS,
    MAX_TOTAL_READ_BYTES,
    REPOSITORY_INSIGHT_TOOLS,
    LangGraphRepositoryInsightPlanner,
)
from vision_research_ops.application.services.supervisor_models import (
    ModelCallBudget,
    SupervisorError,
)
from vision_research_ops.cli.conversation import ConversationSession
from vision_research_ops.ports import LLMError, OperationContext
from vision_research_ops.prompts.repository_insight import PROMPT_VERSION, SYSTEM_PROMPT

from .conftest import RepositoryInsightHarness


def _advice() -> dict:
    return {
        "repository_summary": "A public image classification training entry was read.",
        "adaptation_fit": "MEDIUM",
        "code_evidence": [{"path": "train.py", "observation": "Uses a classification loss."}],
        "suggestions": [
            {
                "area": "INPUT_CHANNELS",
                "target_paths": ["train.py"],
                "recommendation": "Check the grayscale input shape.",
                "rationale": "The target has one channel.",
            }
        ],
        "risks": ["Only a bounded source subset was read."],
        "items_to_verify": ["Verify input tensor shapes."],
        "limitations": ["No source code was executed.", "No training or metric claims are made."],
    }


def _oversized() -> dict:
    return _advice() | {
        "risks": [f"Untrusted risk marker {i}." for i in range(7)],
        "items_to_verify": [f"Untrusted verification marker {i}." for i in range(7)],
    }


def _call(name: str, **args: object) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])


def _steps(*submissions: dict | Exception, six_reads: bool = False) -> list[AIMessage | Exception]:
    paths = ["train.py"]
    if six_reads:
        paths += ["model.py", "data.py", "README.md", "requirements.txt", "config.yaml"]
    return [
        _call("inspect_repository_summary"),
        _call("inspect_target_profile"),
        *[_call("read_repository_file", path=path) for path in paths],
        *[
            value
            if isinstance(value, Exception)
            else _call("submit_adaptation_advice", advice=value)
            for value in submissions
        ],
    ]


class ScriptedAgent:
    """Deliver schema-bound tool calls and retain only test-local observations."""

    def __init__(self, steps: list[AIMessage | Exception], budget: ModelCallBudget) -> None:
        self.steps = steps
        self.budget = budget
        self.calls = 0
        self.observations: list[ToolMessage] = []

    def bind(self, tools: Sequence[BaseTool]) -> ScriptedAgent:
        assert tuple(tool.name for tool in tools) == REPOSITORY_INSIGHT_TOOLS
        return self

    async def ainvoke(self, value: object) -> AIMessage:
        assert isinstance(value, list)
        # Even the correction call must be reserved before reaching this boundary.
        assert self.budget.used > self.calls
        self.observations = [message for message in value if isinstance(message, ToolMessage)]
        step = self.steps[self.calls]
        self.calls += 1
        if isinstance(step, Exception):
            raise step
        return step


def _planner(agent: ScriptedAgent) -> LangGraphRepositoryInsightPlanner:
    return LangGraphRepositoryInsightPlanner(
        agent_factory=agent.bind,
        planner_kind="SCRIPTED_TOOL_CALLING",
        provider_id="offline-test",
        model_id="scripted-correction",
        before_model_call=agent.budget.consume,
    )


async def _analyze(harness: RepositoryInsightHarness, agent: ScriptedAgent):
    dependencies = harness.dependencies
    ctx = OperationContext(
        schema_version="1",
        correlation_id="corr-correction",
        workflow_id="workflow-correction",
        actor_id="pipeline-user",
        idempotency_key="correction",
        sensitivity="PUBLIC",
    )
    resolution = await dependencies.repository_provider.resolve(
        "https://github.com/example/sem-classifier", None, ctx=ctx
    )
    metadata = await dependencies.repository_provider.fetch_metadata(resolution, ctx=ctx)
    snapshot = await dependencies.repository_provider.snapshot(resolution, ctx=ctx)
    analysis = await dependencies.static_analyzer.analyze(snapshot, dependencies.policy, ctx=ctx)
    return await _planner(agent).analyze(
        resolution=resolution,
        metadata=metadata,
        snapshot=snapshot,
        source_index=dependencies.source_reader.index(snapshot),
        structure=structure_summary(analysis),
        source_reader=dependencies.source_reader,
        ctx=ctx,
    )


@pytest.mark.asyncio
async def test_first_valid_finishes_without_an_extra_model_turn(make_repository_insight_harness):
    """Strict first-valid output keeps the four actual tools, prompt provenance and no retry."""
    agent = ScriptedAgent(_steps(_advice()), ModelCallBudget())
    result = await _analyze(make_repository_insight_harness(), agent)
    assert agent.calls == agent.budget.used == 4
    assert tuple(event.tool_name for event in result.trace.events) == REPOSITORY_INSIGHT_TOOLS
    assert result.advice == RepositoryAdaptationAdvice.model_validate(_advice())
    assert result.generation.prompt_version == PROMPT_VERSION == "1.0.1"
    legacy = result.generation.model_dump() | {"prompt_version": "1.0.0"}
    assert RepositoryInsightGeneration.model_validate(legacy).prompt_version == "1.0.0"
    for field in ("code_evidence", "suggestions", "risks", "items_to_verify", "limitations"):
        assert field in SYSTEM_PROMPT
    assert "1 to 6" in SYSTEM_PROMPT and "2 to 6" in SYSTEM_PROMPT and "1 to 3" in SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_observed_two_list_failure_corrects_at_exact_ten_call_budget(
    make_repository_insight_harness,
):
    """Six source reads survive correction without another read, fetch or completion call."""
    harness = make_repository_insight_harness()
    agent = ScriptedAgent(_steps(_oversized(), _advice(), six_reads=True), ModelCallBudget())
    result = await _analyze(harness, agent)
    assert (MAX_MODEL_TURNS, MAX_TOOL_CALLS, MAX_READ_FILES, MAX_TOTAL_READ_BYTES) == (
        10,
        10,
        6,
        49152,
    )
    assert agent.calls == agent.budget.used == len(result.trace.events) == 10
    assert len(result.trace.read_files) == 6
    assert sum(read.returned_bytes for read in result.trace.read_files) <= 48 * 1024
    assert harness.transport.call_count == 4
    assert result.advice.risks == _advice()["risks"]
    feedback = agent.observations[-1]
    assert feedback.status == "error"
    assert json.loads(feedback.content) == {
        "errors": [
            {"field": "advice.risks", "type": "too_long", "max_length": 6},
            {"field": "advice.items_to_verify", "type": "too_long", "max_length": 6},
        ]
    }
    assert "Untrusted" not in feedback.content
    assert "content" not in feedback.content


@pytest.mark.asyncio
async def test_repeated_invalid_advice_is_not_corrected_twice(make_repository_insight_harness):
    """The second oversized submission fails explicitly instead of being truncated or saved."""
    harness = make_repository_insight_harness()
    agent = ScriptedAgent(_steps(_oversized(), _oversized()), ModelCallBudget())
    with pytest.raises(LLMError) as failure:
        await _analyze(harness, agent)
    assert failure.value.failure.code == "REPOSITORY_INSIGHT_TOOL_FAILED"
    assert agent.calls == agent.budget.used == 5
    assert not list(harness.workspace.glob("repository-insight/**/*"))


@pytest.mark.asyncio
async def test_correction_respects_exhausted_shared_budget(make_repository_insight_harness):
    """The shared request's 25th invocation is rejected before model access."""
    budget = ModelCallBudget()
    for _ in range(20):
        budget.consume()
    agent = ScriptedAgent(_steps(_oversized(), _advice()), budget)
    with pytest.raises(SupervisorError, match="SUPERVISOR_MODEL_BUDGET_EXHAUSTED"):
        await _analyze(make_repository_insight_harness(), agent)
    assert budget.used == budget.limit == 24
    assert agent.calls == 4


@pytest.mark.asyncio
async def test_correction_respects_exhausted_local_turn_budget(make_repository_insight_harness):
    """A final invalid submission on turn ten cannot obtain an eleventh turn."""
    steps = _steps(_oversized(), _advice(), six_reads=True)
    steps.insert(2, _call("inspect_target_profile"))
    agent = ScriptedAgent(steps, ModelCallBudget())
    with pytest.raises(LLMError) as failure:
        await _analyze(make_repository_insight_harness(), agent)
    assert failure.value.failure.code == "REPOSITORY_INSIGHT_STEP_LIMIT"
    assert agent.calls == agent.budget.used == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["mixed_type", "command", "unread_path", "path_validation"])
async def test_non_length_failures_are_not_corrected(case, make_repository_insight_harness):
    """Type, safety and actual-read/canonical-path checks remain fail-closed."""
    advice = deepcopy(_advice())
    if case == "mixed_type":
        advice = _oversized() | {"code_executed": True}
    elif case == "command":
        advice["risks"] = ["Run pip install before review."]
    elif case == "unread_path":
        advice["code_evidence"][0]["path"] = "model.py"
    else:
        advice["code_evidence"][0]["path"] = "../train.py"
    agent = ScriptedAgent(_steps(advice, _advice()), ModelCallBudget())
    with pytest.raises(LLMError) as failure:
        await _analyze(make_repository_insight_harness(), agent)
    assert failure.value.failure.code == "REPOSITORY_INSIGHT_TOOL_FAILED"
    assert agent.calls == 4


@pytest.mark.asyncio
async def test_corrected_advice_still_requires_actual_read_paths(make_repository_insight_harness):
    """Length recovery does not bypass the final submit tool's grounding validation."""
    advice = deepcopy(_advice())
    advice["suggestions"][0]["target_paths"] = ["model.py"]
    agent = ScriptedAgent(_steps(_oversized(), advice), ModelCallBudget())
    with pytest.raises(LLMError) as failure:
        await _analyze(make_repository_insight_harness(), agent)
    assert failure.value.failure.code == "REPOSITORY_INSIGHT_TOOL_FAILED"
    assert agent.calls == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [TimeoutError("private provider marker"), RuntimeError("auth marker")]
)
async def test_provider_failures_are_explicit_without_retry(error, make_repository_insight_harness):
    """An error on the correction call cannot trigger another model invocation."""
    agent = ScriptedAgent(_steps(_oversized(), error, _advice()), ModelCallBudget())
    with pytest.raises(LLMError) as failure:
        await _analyze(make_repository_insight_harness(), agent)
    assert failure.value.failure.code == "REPOSITORY_INSIGHT_PROVIDER_FAILED"
    assert "marker" not in failure.value.failure.message
    assert agent.calls == agent.budget.used == 5


@pytest.mark.asyncio
async def test_unknown_tool_is_not_a_correction(make_repository_insight_harness):
    """Correction never opens the tool allowlist."""
    agent = ScriptedAgent([_call("run_shell", command="unused")], ModelCallBudget())
    with pytest.raises(LLMError) as failure:
        await _analyze(make_repository_insight_harness(), agent)
    assert failure.value.failure.code == "REPOSITORY_INSIGHT_TOOL_FORBIDDEN"
    assert agent.calls == 1


@pytest.mark.asyncio
async def test_invalid_after_correction_preserves_context_and_writes_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_repository_insight_harness
):
    """The real conversation/Gate path cannot persist rejected advice as a success."""
    harness = make_repository_insight_harness()
    agent = ScriptedAgent(_steps(_oversized(), _oversized()), ModelCallBudget())
    harness.dependencies.planner = _planner(agent)
    monkeypatch.setattr(
        "vision_research_ops.cli.conversation.build_repository_insight_dependencies",
        lambda **_kwargs: harness.dependencies,
    )
    output: list[str] = []
    approvals: list[str] = []

    def approve(prompt: str) -> str:
        approvals.append(prompt)
        return "approve"

    context_path = tmp_path / "sessions" / "context.json"
    session = ConversationSession(
        workspace=harness.workspace,
        context_path=context_path,
        mode="fixture",
        router=FixtureIntentRouter(),
        input_fn=approve,
        output_fn=output.append,
    )
    await session.handle("2608.01234")
    previous = context_path.read_bytes()
    await session.handle("https://github.com/example/sem-classifier")
    assert context_path.read_bytes() == previous
    assert "REPOSITORY_INSIGHT_TOOL_FAILED" in output[-1]
    assert agent.calls == 5
    assert len(approvals) == 1
    assert harness.transport.call_count == 4
    assert not list(harness.workspace.glob("repository-insight/**/*"))
