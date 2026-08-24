"""Compiled LangGraph Research Agent with structured LLM and a human gate."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..nodes.research import (
    analyze_applicability,
    candidate_selection_gate,
    deduplicate_papers,
    finalize_research_selection,
    hard_filter_candidates,
    retrieve_papers,
    route_after_analysis,
    route_after_deduplication,
    route_after_retrieval,
    validate_research_request,
)
from ..research_runtime import ResearchDependencies
from ..state import WorkflowState

_CHECKPOINT_TYPE_ALLOWLIST = (
    ("vision_research_ops.domain.enums", "WorkflowPhase"),
    ("vision_research_ops.domain.enums", "WorkflowStatus"),
)


def _pipeline_memory_saver() -> InMemorySaver:
    """Create a strict serializer with only state-enum reconstruction enabled."""
    return InMemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPE_ALLOWLIST)
    )


def build_research_graph(
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[WorkflowState, ResearchDependencies, WorkflowState, WorkflowState]:
    """Compile retrieve -> dedupe -> filter -> LLM -> interrupt -> persist."""
    saver = checkpointer if checkpointer is not None else _pipeline_memory_saver()
    builder = StateGraph(WorkflowState, context_schema=ResearchDependencies)
    builder.add_node("validate_research_request", validate_research_request)
    builder.add_node("retrieve_papers", retrieve_papers)
    builder.add_node("deduplicate_papers", deduplicate_papers)
    builder.add_node("hard_filter_candidates", hard_filter_candidates)
    builder.add_node("analyze_applicability", analyze_applicability)
    builder.add_node("candidate_selection_gate", candidate_selection_gate)
    builder.add_node("finalize_research_selection", finalize_research_selection)

    builder.add_edge(START, "validate_research_request")
    builder.add_edge("validate_research_request", "retrieve_papers")
    builder.add_conditional_edges(
        "retrieve_papers",
        route_after_retrieval,
        {
            "CONTINUE": "deduplicate_papers",
            "FAILED": END,
        },
    )
    builder.add_conditional_edges(
        "deduplicate_papers",
        route_after_deduplication,
        {
            "CONTINUE": "hard_filter_candidates",
            "FAILED": END,
        },
    )
    builder.add_edge("hard_filter_candidates", "analyze_applicability")
    builder.add_conditional_edges(
        "analyze_applicability",
        route_after_analysis,
        {
            "GATE": "candidate_selection_gate",
            "DONE": END,
            "FAILED": END,
        },
    )
    builder.add_edge("candidate_selection_gate", "finalize_research_selection")
    builder.add_edge("finalize_research_selection", END)
    return builder.compile(checkpointer=saver)


__all__ = ["build_research_graph"]
