"""Compiled LangGraph adaptation Adaptation Agent with smoke repair and patch Gate."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..adaptation_runtime import AdaptationDependencies
from ..nodes.adaptation import (
    generate_bounded_patch,
    load_and_compare_inputs,
    patch_acceptance_gate,
    plan_adaptation,
    repair_patch_once,
    route_after_input_compare,
    route_after_patch,
    route_after_patch_gate,
    route_after_plan,
    route_after_smoke,
    run_bounded_smoke,
)
from ..state import WorkflowState

_CHECKPOINT_TYPE_ALLOWLIST = (
    ("vision_research_ops.domain.enums", "WorkflowPhase"),
    ("vision_research_ops.domain.enums", "WorkflowStatus"),
)


def _pipeline_memory_saver() -> InMemorySaver:
    return InMemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPE_ALLOWLIST)
    )


def build_adaptation_graph(
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[WorkflowState, AdaptationDependencies, WorkflowState, WorkflowState]:
    """Compile compare -> plan -> patch -> smoke -> repair once -> exact patch Gate."""
    saver = checkpointer if checkpointer is not None else _pipeline_memory_saver()
    builder = StateGraph(WorkflowState, context_schema=AdaptationDependencies)
    builder.add_node("load_and_compare_inputs", load_and_compare_inputs)
    builder.add_node("plan_adaptation", plan_adaptation)
    builder.add_node("generate_bounded_patch", generate_bounded_patch)
    builder.add_node("run_bounded_smoke", run_bounded_smoke)
    builder.add_node("repair_patch_once", repair_patch_once)
    builder.add_node("patch_acceptance_gate", patch_acceptance_gate)

    builder.add_edge(START, "load_and_compare_inputs")
    builder.add_conditional_edges(
        "load_and_compare_inputs",
        route_after_input_compare,
        {"PLAN": "plan_adaptation", "FAILED": END},
    )
    builder.add_conditional_edges(
        "plan_adaptation",
        route_after_plan,
        {"PATCH": "generate_bounded_patch", "FAILED": END},
    )
    builder.add_conditional_edges(
        "generate_bounded_patch",
        route_after_patch,
        {"SMOKE": "run_bounded_smoke", "FAILED": END},
    )
    builder.add_conditional_edges(
        "run_bounded_smoke",
        route_after_smoke,
        {"GATE": "patch_acceptance_gate", "REPAIR": "repair_patch_once", "FAILED": END},
    )
    builder.add_edge("repair_patch_once", "generate_bounded_patch")
    builder.add_conditional_edges(
        "patch_acceptance_gate",
        route_after_patch_gate,
        {"ACCEPTED": END, "EDIT": "generate_bounded_patch", "REJECTED": END},
    )
    return builder.compile(checkpointer=saver)


__all__ = ["build_adaptation_graph"]
