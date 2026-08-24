"""Scripted real-tool-call planner used by offline adaptation and pipeline demos."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal, cast

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from vision_research_ops.application.adaptation_runtime import AdaptationPlannerOutput
from vision_research_ops.application.services.adaptation_models import AdaptationInputFacts
from vision_research_ops.application.services.adaptation_planning import adaptation_prompt_facts
from vision_research_ops.application.services.adaptation_tool_planner import (
    REQUIRED_PLANNER_TOOLS,
    LangGraphAdaptationPlanner,
)
from vision_research_ops.ports import (
    LLMError,
    OperationContext,
    StructuredOutputValidationError,
    make_failure,
)

from .fixture_llm import fixture_adaptation_proposal


class _ScriptedToolAgent:
    """Emit actual AIMessage tool calls in the required offline demonstration sequence."""

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        names = tuple(tool.name for tool in tools)
        if frozenset(names) != frozenset(REQUIRED_PLANNER_TOOLS):
            raise ValueError("scripted adaptation planner received a different tool allowlist")

    @staticmethod
    def _facts(messages: Sequence[AnyMessage]) -> dict[str, object]:
        for message in messages:
            if isinstance(message, HumanMessage) and isinstance(message.content, str):
                raw = json.loads(message.content)
                if isinstance(raw, dict):
                    return cast(dict[str, object], raw)
        raise ValueError("scripted adaptation planner is missing sanitized facts")

    async def ainvoke(self, value: object) -> AIMessage:
        """Choose the next allowlisted tool from prior ToolMessage observations."""
        if not isinstance(value, list):
            raise TypeError("scripted tool agent requires a message list")
        messages = cast(list[AnyMessage], value)
        observed = [
            message.name
            for message in messages
            if isinstance(message, ToolMessage) and message.name is not None
        ]
        if len(observed) >= len(REQUIRED_PLANNER_TOOLS):
            return AIMessage(
                content="Required read-only observations validated; planning is ready."
            )
        name = REQUIRED_PLANNER_TOOLS[len(observed)]
        args: dict[str, object] = {}
        if name == "validate_adaptation_plan":
            proposal = fixture_adaptation_proposal(self._facts(messages))
            args = {"proposal": proposal.model_dump(mode="json")}
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"fixture-tool-call-{len(observed) + 1}",
                    "type": "tool_call",
                }
            ],
        )


class FixtureToolCallingAdaptationPlanner:
    """Offline facade around the real LangGraph/ToolNode planning loop."""

    def __init__(
        self,
        *,
        mode: Literal["success", "provider_failure", "schema_failure"] = "success",
    ) -> None:
        self._mode = mode
        self._call_count = 0
        self._planner = LangGraphAdaptationPlanner(
            agent_factory=_ScriptedToolAgent,
            planner_kind="SCRIPTED_TOOL_CALLING",
            provider_id="offline-fixture",
            model_id="scripted-tool-calling-adaptation-v1",
        )

    @property
    def call_count(self) -> int:
        """Return planner invocations, preserving the adaptation test probe."""
        return self._call_count

    @property
    def tool_call_names(self) -> tuple[str, ...]:
        """Return the actual completed ToolNode call sequence."""
        return tuple(self._planner.last_tool_names)

    @property
    def graph_nodes(self) -> frozenset[str]:
        """Expose only node names for a stable topology assertion."""
        return self._planner.last_graph_nodes

    async def plan(
        self,
        facts: AdaptationInputFacts,
        *,
        ctx: OperationContext,
    ) -> AdaptationPlannerOutput:
        """Run the scripted model through the same real tools used by live mode."""
        self._call_count += 1
        if self._mode == "provider_failure":
            raise LLMError(
                make_failure(
                    code="FIXTURE_LLM_PROVIDER_FAILED",
                    category="LLM_PROVIDER",
                    message="The scripted adaptation provider failed explicitly.",
                    retryable=False,
                    ctx=ctx,
                )
            )
        if self._mode == "schema_failure":
            raise StructuredOutputValidationError("adaptation_plan", ctx)
        expected = adaptation_prompt_facts(facts)
        if not expected:
            raise ValueError("scripted adaptation facts cannot be empty")
        return await self._planner.plan(facts, ctx=ctx)


__all__ = ["FixtureToolCallingAdaptationPlanner"]
