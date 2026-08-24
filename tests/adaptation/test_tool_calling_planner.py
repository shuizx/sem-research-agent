"""tool-calling planner fixed checks for the bounded real ToolNode planner."""

from __future__ import annotations

from typing import cast

import pytest
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import BaseTool

from vision_research_ops.adapters.llm import build_dashscope_adaptation_planner
from vision_research_ops.application.services.adaptation_planning import (
    validate_adaptation_inputs,
)
from vision_research_ops.application.services.adaptation_tool_planner import (
    REQUIRED_PLANNER_TOOLS,
    LangGraphAdaptationPlanner,
)
from vision_research_ops.ports import OperationContext, PortError
from vision_research_ops.settings import Settings

from .conftest import AdaptationHarness


def _operation_context() -> OperationContext:
    return OperationContext(
        schema_version="1",
        correlation_id="corr-tool-planner-test",
        workflow_id="workflow-tool-planner-test",
        actor_id="pipeline-user",
        idempotency_key="tool-planner-test",
        sensitivity="INTERNAL",
    )


def test_live_builder_requires_injected_dashscope_key() -> None:
    """optional live mode fails explicitly and never reads an implicit fallback."""
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        build_dashscope_adaptation_planner(Settings.from_env({}))


@pytest.mark.asyncio
async def test_scripted_planner_uses_real_tool_node_and_hash_only_trace(
    make_adaptation_harness,
) -> None:
    """all four read-only tools execute before strict generation."""
    harness: AdaptationHarness = make_adaptation_harness()
    facts = validate_adaptation_inputs(
        harness.repository_store.load_result("workflow-repository-p3-fixture"),
        harness.dependencies.dataset_profile,
    )
    output = await harness.llm.plan(facts, ctx=_operation_context())
    assert harness.llm.graph_nodes == {"__start__", "planner_model", "tools", "__end__"}
    assert harness.llm.tool_call_names == REQUIRED_PLANNER_TOOLS
    assert [event.tool_name for event in output.trace.events] == list(REQUIRED_PLANNER_TOOLS)
    assert output.trace.planner_kind == "SCRIPTED_TOOL_CALLING"
    serialized = output.trace.model_dump_json()
    assert "proposal" not in serialized
    assert "label_names" not in serialized
    assert "C:\\" not in serialized
    assert "api_key" not in serialized.casefold()
    assert output.generation.value.num_classes == len(facts.label_names)


class _StopsWithoutTools:
    async def ainvoke(self, _value: object) -> AIMessage:
        return AIMessage(content="ready")


@pytest.mark.asyncio
async def test_planner_cannot_finish_without_required_tool_observations(
    make_adaptation_harness,
) -> None:
    """a model response cannot skip the four required read-only observations."""
    harness: AdaptationHarness = make_adaptation_harness()
    facts = validate_adaptation_inputs(
        harness.repository_store.load_result("workflow-repository-p3-fixture"),
        harness.dependencies.dataset_profile,
    )
    planner = LangGraphAdaptationPlanner(
        agent_factory=lambda _tools: _StopsWithoutTools(),
        planner_kind="SCRIPTED_TOOL_CALLING",
        provider_id="offline-test",
        model_id="stops-without-tools",
    )
    with pytest.raises(PortError) as raised:
        await planner.plan(facts, ctx=_operation_context())
    assert raised.value.failure.code == "ADAPTATION_PLANNER_REQUIRED_TOOLS_MISSING"


class _ForbiddenToolAgent:
    async def ainvoke(self, _value: object) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_shell",
                    "args": {"command": "python train.py"},
                    "id": "forbidden-1",
                    "type": "tool_call",
                }
            ],
        )


@pytest.mark.asyncio
async def test_planner_rejects_non_allowlisted_execution_tool(
    make_adaptation_harness,
) -> None:
    """LLM cannot manufacture shell, Git, patch, training, or write tools."""
    harness: AdaptationHarness = make_adaptation_harness()
    facts = validate_adaptation_inputs(
        harness.repository_store.load_result("workflow-repository-p3-fixture"),
        harness.dependencies.dataset_profile,
    )
    observed_tools: list[BaseTool] = []

    def factory(tools: list[BaseTool]) -> _ForbiddenToolAgent:
        observed_tools.extend(tools)
        return _ForbiddenToolAgent()

    planner = LangGraphAdaptationPlanner(
        agent_factory=factory,
        planner_kind="SCRIPTED_TOOL_CALLING",
        provider_id="offline-test",
        model_id="forbidden-tool-agent",
    )
    with pytest.raises(PortError) as raised:
        await planner.plan(facts, ctx=_operation_context())
    assert raised.value.failure.code == "ADAPTATION_PLANNER_TOOL_FORBIDDEN"
    assert tuple(tool.name for tool in observed_tools) == REQUIRED_PLANNER_TOOLS


class _InvalidValidationArgumentsAgent:
    async def ainvoke(self, value: object) -> AIMessage:
        if not isinstance(value, list):
            raise TypeError("test agent requires a message list")
        messages = cast(list[AnyMessage], value)
        observed = [
            message.name
            for message in messages
            if isinstance(message, ToolMessage) and message.name is not None
        ]
        call_index = len(observed)
        name = REQUIRED_PLANNER_TOOLS[min(call_index, 3)]
        args: dict[str, object] = {}
        if name == "validate_adaptation_plan":
            args = {"proposal": {"schema_version": "1"}}
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"invalid-validation-{call_index + 1}",
                    "type": "tool_call",
                }
            ],
        )


@pytest.mark.asyncio
async def test_invalid_tool_arguments_become_structured_planner_failure(
    make_adaptation_harness,
) -> None:
    """ToolNode schema errors cannot escape as framework exceptions."""
    harness: AdaptationHarness = make_adaptation_harness()
    facts = validate_adaptation_inputs(
        harness.repository_store.load_result("workflow-repository-p3-fixture"),
        harness.dependencies.dataset_profile,
    )
    planner = LangGraphAdaptationPlanner(
        agent_factory=lambda _tools: _InvalidValidationArgumentsAgent(),
        planner_kind="SCRIPTED_TOOL_CALLING",
        provider_id="offline-test",
        model_id="invalid-validation-agent",
    )
    with pytest.raises(PortError) as raised:
        await planner.plan(facts, ctx=_operation_context())
    assert raised.value.failure.code == "ADAPTATION_PLANNER_TOOL_FAILED"
