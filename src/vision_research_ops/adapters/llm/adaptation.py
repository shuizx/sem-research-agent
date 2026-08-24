"""Optional DashScope tool-calling adaptation planner."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from langchain_core.tools import BaseTool

from vision_research_ops.application.services.adaptation_tool_planner import (
    LangGraphAdaptationPlanner,
    ToolCallingAgent,
)
from vision_research_ops.settings import Settings

from .dashscope import build_dashscope_chat_model


def build_dashscope_adaptation_planner(settings: Settings) -> LangGraphAdaptationPlanner:
    """Build a live planner using ChatOpenAI.bind_tools and strict final output."""
    client = build_dashscope_chat_model(settings)

    def bind(tools: Sequence[BaseTool]) -> ToolCallingAgent:
        return cast(
            ToolCallingAgent,
            client.bind_tools(
                tools,
                strict=True,
                parallel_tool_calls=False,
            ),
        )

    return LangGraphAdaptationPlanner(
        agent_factory=bind,
        planner_kind="DASHSCOPE_TOOL_CALLING",
        provider_id="dashscope-openai-compatible",
        model_id=settings.llm_model,
    )


__all__ = ["build_dashscope_adaptation_planner"]
