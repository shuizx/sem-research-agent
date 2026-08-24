"""Scripted AIMessage tool calls for offline repository-insight demonstrations."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import BaseTool

from vision_research_ops.application.services.repository_insight_models import (
    RepositoryAdaptationAdvice,
    RepositoryAdaptationSuggestion,
    RepositoryCodeEvidence,
)
from vision_research_ops.application.services.repository_insight_planner import (
    REPOSITORY_INSIGHT_TOOLS,
    LangGraphRepositoryInsightPlanner,
)


def _tool_payload(message: ToolMessage) -> dict[str, object]:
    if not isinstance(message.content, str):
        raise ValueError("scripted repository insight requires JSON tool messages")
    value = json.loads(message.content)
    if not isinstance(value, dict):
        raise ValueError("scripted repository insight tool output must be an object")
    return cast(dict[str, object], value)


class _ScriptedRepositoryInsightAgent:
    """Choose the four real tools in a deterministic offline sequence."""

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        names = tuple(tool.name for tool in tools)
        if names != REPOSITORY_INSIGHT_TOOLS:
            raise ValueError("fixture repository insight received a different tool allowlist")

    async def ainvoke(self, value: object) -> AIMessage:
        """Return actual AIMessage tool calls consumed by the production ToolNode."""
        if not isinstance(value, list):
            raise TypeError("fixture repository insight requires a message list")
        messages = cast(list[AnyMessage], value)
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        observed = [message.name for message in tool_messages]
        if not observed:
            name = "inspect_repository_summary"
            args: dict[str, object] = {}
        elif observed == ["inspect_repository_summary"]:
            name = "inspect_target_profile"
            args = {}
        elif observed == ["inspect_repository_summary", "inspect_target_profile"]:
            summary = _tool_payload(tool_messages[0])
            paths = summary.get("available_source_paths")
            if not isinstance(paths, list) or not paths or not isinstance(paths[0], str):
                raise ValueError("fixture repository insight requires one indexed source path")
            name = "read_repository_file"
            args = {"path": paths[0]}
        elif observed == [
            "inspect_repository_summary",
            "inspect_target_profile",
            "read_repository_file",
        ]:
            read = _tool_payload(tool_messages[-1])
            path = read.get("path")
            if not isinstance(path, str):
                raise ValueError("fixture repository insight requires a read source path")
            advice = RepositoryAdaptationAdvice(
                repository_summary=(
                    "The fixed public snapshot exposes a small image-classification code layout."
                ),
                adaptation_fit="MEDIUM",
                code_evidence=[
                    RepositoryCodeEvidence(
                        path=path,
                        observation=(
                            "This file provides a concrete integration point that must be "
                            "reviewed against the one-channel SEM data contract."
                        ),
                    )
                ],
                suggestions=[
                    RepositoryAdaptationSuggestion(
                        area="DATA_LOADING",
                        target_paths=[path],
                        recommendation=(
                            "Align the observed input pipeline with one-channel SEM images and "
                            "a public or user-authorized label mapping."
                        ),
                        rationale=(
                            "The target has a narrower modality and leakage-aware split contract."
                        ),
                    )
                ],
                risks=["Only a bounded subset of the fixed public snapshot was inspected."],
                items_to_verify=[
                    "Verify dataset transforms, class indexing, and group-aware split behavior."
                ],
                limitations=[
                    "No repository code was patched, imported, installed, or executed.",
                    (
                        "No training was run, so compatibility and metric improvement are not "
                        "guaranteed."
                    ),
                ],
            )
            name = "submit_adaptation_advice"
            args = {"advice": advice.model_dump(mode="json")}
        else:
            return AIMessage(content="Strict repository adaptation advice has been submitted.")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"fixture-repository-insight-{len(tool_messages) + 1}",
                    "type": "tool_call",
                }
            ],
        )


class FixtureRepositoryInsightPlanner(LangGraphRepositoryInsightPlanner):
    """Production planner with deterministic model choices for offline tests/debugging."""

    def __init__(self) -> None:
        super().__init__(
            agent_factory=_ScriptedRepositoryInsightAgent,
            planner_kind="SCRIPTED_TOOL_CALLING",
            provider_id="offline-fixture",
            model_id="scripted-repository-insight-v1",
        )


__all__ = ["FixtureRepositoryInsightPlanner"]
