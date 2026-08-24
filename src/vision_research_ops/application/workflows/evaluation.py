"""Compiled LangGraph evaluation Evaluation Agent with deterministic valid/invalid routing."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..evaluation_runtime import EvaluationDependencies
from ..nodes.evaluation import (
    compute_metrics_node,
    decide_conclusion,
    invalid_result,
    load_training_outputs,
    persist_evaluation,
    render_report,
    route_after_decision,
    route_after_load,
    route_after_metrics,
    route_after_persistence,
    route_after_validation,
    validate_comparability,
)
from ..services.evaluation_models import EvaluationState


def build_evaluation_graph(
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[EvaluationState, EvaluationDependencies, EvaluationState, EvaluationState]:
    """Compile load -> compare -> metrics/invalid -> persist -> template report."""
    saver = checkpointer if checkpointer is not None else InMemorySaver()
    builder = StateGraph(EvaluationState, context_schema=EvaluationDependencies)
    builder.add_node("load_training_outputs", load_training_outputs)
    builder.add_node("validate_comparability", validate_comparability)
    builder.add_node("invalid_result", invalid_result)
    builder.add_node("compute_metrics", compute_metrics_node)
    builder.add_node("decide_conclusion", decide_conclusion)
    builder.add_node("persist_evaluation", persist_evaluation)
    builder.add_node("render_report", render_report)

    builder.add_edge(START, "load_training_outputs")
    builder.add_conditional_edges(
        "load_training_outputs",
        route_after_load,
        {
            "VALIDATE": "validate_comparability",
            "INVALID": "invalid_result",
            "FAILED": END,
        },
    )
    builder.add_conditional_edges(
        "validate_comparability",
        route_after_validation,
        {
            "COMPUTE_METRICS": "compute_metrics",
            "INVALID": "invalid_result",
            "FAILED": END,
        },
    )
    builder.add_conditional_edges(
        "compute_metrics",
        route_after_metrics,
        {"DECIDE": "decide_conclusion", "FAILED": END},
    )
    builder.add_conditional_edges(
        "decide_conclusion",
        route_after_decision,
        {"PERSIST": "persist_evaluation", "FAILED": END},
    )
    builder.add_conditional_edges(
        "persist_evaluation",
        route_after_persistence,
        {"REPORT": "render_report", "FAILED": END},
    )
    builder.add_conditional_edges(
        "invalid_result",
        route_after_persistence,
        {"REPORT": "render_report", "FAILED": END},
    )
    builder.add_edge("render_report", END)
    return builder.compile(checkpointer=saver)


__all__ = ["build_evaluation_graph"]
