"""Optional DashScope ToolNode planner for approved public source snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from langchain_core.tools import BaseTool

from vision_research_ops.application.services.repository_insight_planner import (
    LangGraphRepositoryInsightPlanner,
    RepositoryInsightToolAgent,
)
from vision_research_ops.settings import Settings

from .dashscope import build_dashscope_chat_model


def build_dashscope_repository_insight_planner(
    settings: Settings,
) -> LangGraphRepositoryInsightPlanner:
    """Bind exactly the four repository-insight tools to live DashScope ChatOpenAI."""
    client = build_dashscope_chat_model(settings)

    def bind(tools: Sequence[BaseTool]) -> RepositoryInsightToolAgent:
        return cast(
            RepositoryInsightToolAgent,
            client.bind_tools(
                tools,
                strict=True,
                parallel_tool_calls=False,
            ),
        )

    return LangGraphRepositoryInsightPlanner(
        agent_factory=bind,
        planner_kind="DASHSCOPE_TOOL_CALLING",
        provider_id="dashscope-openai-compatible",
        model_id=settings.llm_model,
    )


__all__ = ["build_dashscope_repository_insight_planner"]
