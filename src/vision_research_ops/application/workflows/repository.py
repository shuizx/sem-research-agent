"""Compiled LangGraph Repository Agent with an ingest approval boundary."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..nodes.repository import (
    analyze_repository,
    prepare_repository_candidate,
    repository_ingest_gate,
    resolve_repository,
    route_after_candidate_preparation,
    route_after_repository_gate,
    route_after_repository_resolution,
)
from ..repository_runtime import RepositoryDependencies
from ..state import WorkflowState

_CHECKPOINT_TYPE_ALLOWLIST = (
    ("vision_research_ops.domain.enums", "WorkflowPhase"),
    ("vision_research_ops.domain.enums", "WorkflowStatus"),
)


def _pipeline_memory_saver() -> InMemorySaver:
    return InMemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPE_ALLOWLIST)
    )


def build_repository_graph(
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[WorkflowState, RepositoryDependencies, WorkflowState, WorkflowState]:
    """Compile evidence -> interrupt -> pin -> static profile for one repository."""
    saver = checkpointer if checkpointer is not None else _pipeline_memory_saver()
    builder = StateGraph(WorkflowState, context_schema=RepositoryDependencies)
    builder.add_node("prepare_repository_candidate", prepare_repository_candidate)
    builder.add_node("repository_ingest_gate", repository_ingest_gate)
    builder.add_node("resolve_repository", resolve_repository)
    builder.add_node("analyze_repository", analyze_repository)

    builder.add_edge(START, "prepare_repository_candidate")
    builder.add_conditional_edges(
        "prepare_repository_candidate",
        route_after_candidate_preparation,
        {"GATE": "repository_ingest_gate", "FAILED": END},
    )
    builder.add_conditional_edges(
        "repository_ingest_gate",
        route_after_repository_gate,
        {"INGEST": "resolve_repository", "REJECTED": END},
    )
    builder.add_conditional_edges(
        "resolve_repository",
        route_after_repository_resolution,
        {"ANALYZE": "analyze_repository", "FAILED": END},
    )
    builder.add_edge("analyze_repository", END)
    return builder.compile(checkpointer=saver)


__all__ = ["build_repository_graph"]
