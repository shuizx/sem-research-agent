"""Real LangGraph assembly for the vertical-slice workflow human-gated vertical slice."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..nodes.fixtures import (
    analyze_result_fixture,
    propose_adaptation_fixture,
    retrieve_research_fixture,
    route_after_human_gate,
    route_after_submit,
    submit_training_fixture,
    validate_request,
)
from ..nodes.gates import human_gate
from ..runtime import WorkflowDependencies
from ..state import WorkflowState


def workflow_config(thread_id: str) -> RunnableConfig:
    """Return the minimal checkpointer config for one validated workflow thread."""
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("thread_id must be a non-blank string")
    return {"configurable": {"thread_id": thread_id}}


def build_vertical_slice_graph(
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[WorkflowState, WorkflowDependencies, WorkflowState, WorkflowState]:
    """Compile the fixture graph; callers pass dependencies as ``ainvoke`` context."""
    saver = checkpointer if checkpointer is not None else InMemorySaver()
    builder = StateGraph(WorkflowState, context_schema=WorkflowDependencies)
    builder.add_node("validate_request", validate_request)
    builder.add_node("retrieve_research_fixture", retrieve_research_fixture)
    builder.add_node("propose_adaptation_fixture", propose_adaptation_fixture)
    builder.add_node("human_gate", human_gate)
    builder.add_node("submit_training_fixture", submit_training_fixture)
    builder.add_node("analyze_result_fixture", analyze_result_fixture)

    builder.add_edge(START, "validate_request")
    builder.add_edge("validate_request", "retrieve_research_fixture")
    builder.add_edge("retrieve_research_fixture", "propose_adaptation_fixture")
    builder.add_edge("propose_adaptation_fixture", "human_gate")
    builder.add_conditional_edges(
        "human_gate",
        route_after_human_gate,
        {
            "APPROVE": "submit_training_fixture",
            "EDIT": "propose_adaptation_fixture",
            "REJECT": END,
        },
    )
    builder.add_conditional_edges(
        "submit_training_fixture",
        route_after_submit,
        {
            "SUBMITTED": "analyze_result_fixture",
            "FAILED": END,
        },
    )
    builder.add_edge("analyze_result_fixture", END)
    return builder.compile(checkpointer=saver)


__all__ = ["build_vertical_slice_graph", "workflow_config"]
