"""Compiled LangGraph training Training Agent with an exact human submission Gate."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..nodes.training import (
    collect_outputs,
    freeze_submission,
    revise_submission,
    route_after_baseline,
    route_after_candidate,
    route_after_collect,
    route_after_freeze,
    route_after_gate,
    route_after_revision,
    route_after_validation,
    run_baseline,
    run_candidate,
    training_gate,
    validate_input,
)
from ..state import WorkflowState
from ..training_runtime import TrainingDependencies

_CHECKPOINT_TYPE_ALLOWLIST = (
    ("vision_research_ops.domain.enums", "WorkflowPhase"),
    ("vision_research_ops.domain.enums", "WorkflowStatus"),
)


def _pipeline_memory_saver() -> InMemorySaver:
    return InMemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPE_ALLOWLIST)
    )


def build_training_graph(
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[WorkflowState, TrainingDependencies, WorkflowState, WorkflowState]:
    """Compile validate -> freeze -> Gate -> baseline -> candidate -> collect."""
    saver = checkpointer if checkpointer is not None else _pipeline_memory_saver()
    builder = StateGraph(WorkflowState, context_schema=TrainingDependencies)
    builder.add_node("validate_input", validate_input)
    builder.add_node("freeze_submission", freeze_submission)
    builder.add_node("training_gate", training_gate)
    builder.add_node("revise_submission", revise_submission)
    builder.add_node("run_baseline", run_baseline)
    builder.add_node("run_candidate", run_candidate)
    builder.add_node("collect_outputs", collect_outputs)

    builder.add_edge(START, "validate_input")
    builder.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {"FREEZE": "freeze_submission", "FAILED": END},
    )
    builder.add_conditional_edges(
        "freeze_submission",
        route_after_freeze,
        {"GATE": "training_gate", "FAILED": END},
    )
    builder.add_conditional_edges(
        "training_gate",
        route_after_gate,
        {"APPROVED": "run_baseline", "EDIT": "revise_submission", "REJECTED": END},
    )
    builder.add_conditional_edges(
        "revise_submission",
        route_after_revision,
        {"GATE": "training_gate", "FAILED": END},
    )
    builder.add_conditional_edges(
        "run_baseline",
        route_after_baseline,
        {"CANDIDATE": "run_candidate", "CANCELLED": END, "FAILED": END},
    )
    builder.add_conditional_edges(
        "run_candidate",
        route_after_candidate,
        {"COLLECT": "collect_outputs", "CANCELLED": END, "FAILED": END},
    )
    builder.add_conditional_edges(
        "collect_outputs",
        route_after_collect,
        {"COMPLETED": END, "FAILED": END},
    )
    return builder.compile(checkpointer=saver)


__all__ = ["build_training_graph"]
