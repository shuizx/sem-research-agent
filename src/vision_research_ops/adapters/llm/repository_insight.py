"""Optional DashScope ToolNode planner for approved public source snapshots."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

from langchain_core.tools import BaseTool

from vision_research_ops.application.services.repository_insight_planner import (
    LangGraphRepositoryInsightPlanner,
    RepositoryInsightToolAgent,
)
from vision_research_ops.application.services.target_profile import SafeTargetProfile
from vision_research_ops.settings import Settings

from .dashscope import build_dashscope_chat_model


def build_dashscope_repository_insight_planner(
    settings: Settings,
    *,
    before_model_call: Callable[[], None] | None = None,
    target_profile: SafeTargetProfile | None = None,
) -> LangGraphRepositoryInsightPlanner:
    """Bind exactly the four repository-insight tools to the configured ChatOpenAI transport."""
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
        provider_id=settings.llm_provider_id,
        model_id=settings.llm_model,
        before_model_call=before_model_call,
        target_profile=target_profile,
    )


__all__ = ["build_dashscope_repository_insight_planner"]
