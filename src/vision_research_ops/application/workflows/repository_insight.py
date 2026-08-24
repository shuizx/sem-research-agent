"""Compiled outer LangGraph for approved public GitHub code insight."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..nodes.repository_insight import (
    analyze_approved_repository_snapshot,
    repository_snapshot_gate,
    route_after_input,
    route_after_snapshot_gate,
    validate_repository_insight_input,
)
from ..repository_insight_runtime import RepositoryInsightDependencies, RepositoryInsightState


def build_repository_insight_graph(
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[
    RepositoryInsightState,
    RepositoryInsightDependencies,
    RepositoryInsightState,
    RepositoryInsightState,
]:
    """Compile validate -> interrupt -> approved snapshot/code insight."""
    builder = StateGraph(RepositoryInsightState, context_schema=RepositoryInsightDependencies)
    builder.add_node("validate_repository_url", validate_repository_insight_input)
    builder.add_node("repository_snapshot_gate", repository_snapshot_gate)
    builder.add_node("analyze_approved_snapshot", analyze_approved_repository_snapshot)
    builder.add_edge(START, "validate_repository_url")
    builder.add_conditional_edges(
        "validate_repository_url",
        route_after_input,
        {"GATE": "repository_snapshot_gate", "FAILED": END},
    )
    builder.add_conditional_edges(
        "repository_snapshot_gate",
        route_after_snapshot_gate,
        {"ANALYZE": "analyze_approved_snapshot", "REJECTED": END},
    )
    builder.add_edge("analyze_approved_snapshot", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


__all__ = ["build_repository_insight_graph"]
