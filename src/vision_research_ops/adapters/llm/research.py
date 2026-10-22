"""Compose the Research ToolNode planner using the existing DashScope transport."""

import json
from collections.abc import Callable, Sequence
from typing import cast

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from vision_research_ops.application.services.paper_planner import LangGraphResearchPlanner
from vision_research_ops.application.services.repository_insight_planner import (
    RepositoryInsightToolAgent,
)
from vision_research_ops.settings import Settings

from .dashscope import build_dashscope_chat_model


class ScriptedResearchToolAgent:
    """Explicit sample-only transport observing the same Research tools as live mode."""

    def __init__(self) -> None:
        self.pending: list[str] = []
        self.current: str | None = None
        self.searches = 0

    async def ainvoke(self, value: object) -> AIMessage:
        """Choose one deterministic sample call from the latest actual observation."""
        messages = cast(list[object], value)
        last = messages[-1]

        def call(name: str, args: dict[str, object]) -> AIMessage:
            return AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": f"sample-{len(messages)}"}],
            )

        if not isinstance(last, ToolMessage):
            self.searches += 1
            return call("search_papers", {"keywords": ["SEM defect classification"]})
        observation = json.loads(str(last.content))
        if last.name == "search_papers":
            self.pending = [item["paper_id"] for item in observation["papers"][:5]]
            if not self.pending and self.searches < 2:
                self.searches += 1
                return call("search_papers", {"keywords": ["microscopy classification"]})
        if last.name == "inspect_paper":
            paper = observation["paper"]
            applicable = "classification" in (paper["title"] + paper["abstract"]).casefold()
            return call(
                "submit_assessment",
                {
                    "paper_id": self.current,
                    "assessment": {
                        "summary": paper["title"],
                        "applicable": applicable,
                        "recommendation": "HIGH" if applicable else "REJECT",
                        "relevance_score": 0.9 if applicable else 0.1,
                        "confidence": 0.8,
                        "task_match": 0.9 if applicable else 0.1,
                        "modality_match": 0.7,
                        "data_match": 0.7,
                        "code_match": 0.5,
                        "compute_fit": 0.7,
                        "evidence": [
                            {
                                "dimension": "TASK",
                                "source_field": "abstract",
                                "statement": paper["abstract"][:300],
                            }
                        ],
                        "risks": ["Scripted sample reasoning, not a live model judgment."],
                        "rationale": (
                            "Sample classification evidence warrants human review."
                            if applicable
                            else "Sample evidence concerns a different task, not classification."
                        ),
                    },
                },
            )
        if self.pending:
            self.current = self.pending.pop(0)
            return call("inspect_paper", {"paper_id": self.current})
        return call("finish_research", {})


def build_scripted_research_planner() -> LangGraphResearchPlanner:
    """Build the offline model transport without bypassing the real ToolNode loop."""
    return LangGraphResearchPlanner(
        lambda _: ScriptedResearchToolAgent(), model_id="scripted-research-tool-v1"
    )


def build_dashscope_research_planner(
    settings: Settings, *, before_model_call: Callable[[], None] | None = None
) -> LangGraphResearchPlanner:
    """Bind only the fixed Research tools; no provider or settings duplication."""
    client = build_dashscope_chat_model(settings)

    def bind(tools: Sequence[BaseTool]) -> RepositoryInsightToolAgent:
        return cast(
            RepositoryInsightToolAgent,
            client.bind_tools(tools, strict=True, parallel_tool_calls=False),
        )

    return LangGraphResearchPlanner(
        bind,
        model_id=settings.llm_model,
        provider_id=settings.llm_provider_id,
        before_model_call=before_model_call,
    )
