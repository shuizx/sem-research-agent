"""Top-level StateGraph that serially composes research-to-evaluation workflows."""

from __future__ import annotations

from typing import Literal, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from ..pipeline_runtime import PipelineDependencies
from ..services.pipeline_models import (
    PipelineStageName,
    PipelineState,
    pipeline_state_as_jsonable,
)


async def _run_child_stage(
    state: PipelineState,
    runtime: Runtime[PipelineDependencies],
    *,
    stage: PipelineStageName,
) -> PipelineState:
    workflow_id = state.get("workflow_id")
    if not isinstance(workflow_id, str) or not workflow_id:
        raise ValueError("pipeline state lacks a workflow_id")
    outcome = await runtime.context.driver.run_stage(
        pipeline_workflow_id=workflow_id,
        stage=stage,
    )
    stage_records = [*state.get("stage_records", []), outcome.record.model_dump(mode="json")]
    gate_records = [
        *state.get("gate_records", []),
        *(item.model_dump(mode="json") for item in outcome.gates),
    ]
    artifact_refs = dict(state.get("artifact_refs", {}))
    for index, ref in enumerate(outcome.record.artifact_refs, start=1):
        artifact_refs[f"{stage}.{index}"] = ref
    if outcome.record.status == "SUCCEEDED":
        status = "SUCCEEDED" if stage == "evaluation" else "RUNNING"
        route: Literal["CONTINUE", "STOP"] = "CONTINUE"
        failure: dict[str, object] | None = None
    else:
        status = "STOPPED" if outcome.record.status == "STOPPED" else "FAILED"
        route = "STOP"
        failure = (
            cast(dict[str, object], outcome.record.failure.model_dump(mode="json"))
            if outcome.record.failure is not None
            else {
                "code": f"PIPELINE_{stage.upper()}_FAILED",
                "message": f"The {stage} child Agent stopped without a result.",
                "stage": stage,
            }
        )
    update: PipelineState = {
        "phase": stage.upper(),
        "status": cast(Literal["RUNNING", "SUCCEEDED", "FAILED", "STOPPED"], status),
        "route": route,
        "stage_records": stage_records,
        "gate_records": gate_records,
        "resume_count": state.get("resume_count", 0) + outcome.record.resume_count,
        "artifact_refs": artifact_refs,
        "failure_reason": failure,
    }
    if outcome.conclusion is not None:
        update["conclusion"] = outcome.conclusion
    pipeline_state_as_jsonable({**state, **update})
    return update


async def research_stage(
    state: PipelineState,
    runtime: Runtime[PipelineDependencies],
) -> PipelineState:
    """Drive the existing research Research Agent."""
    return await _run_child_stage(state, runtime, stage="research")


async def repository_stage(
    state: PipelineState,
    runtime: Runtime[PipelineDependencies],
) -> PipelineState:
    """Drive the existing repository Repository Agent."""
    return await _run_child_stage(state, runtime, stage="repository")


async def adaptation_stage(
    state: PipelineState,
    runtime: Runtime[PipelineDependencies],
) -> PipelineState:
    """Drive the existing adaptation Adaptation Agent."""
    return await _run_child_stage(state, runtime, stage="adaptation")


async def training_stage(
    state: PipelineState,
    runtime: Runtime[PipelineDependencies],
) -> PipelineState:
    """Drive the existing training Training Agent after its exact Gate."""
    return await _run_child_stage(state, runtime, stage="training")


async def evaluation_stage(
    state: PipelineState,
    runtime: Runtime[PipelineDependencies],
) -> PipelineState:
    """Drive the existing deterministic evaluation Evaluation Agent."""
    return await _run_child_stage(state, runtime, stage="evaluation")


async def summarize_stage(
    state: PipelineState,
    runtime: Runtime[PipelineDependencies],
) -> PipelineState:
    """Persist one canonical summary after either success or an explicit stop."""
    summary = runtime.context.driver.build_summary(state)
    reused = runtime.context.summary_store.write_summary(summary)
    runtime.context.event_sink(
        (
            '{"event":"summary_reused","summary_ref":"%s"}'
            if reused
            else '{"event":"summary_written","summary_ref":"%s"}'
        )
        % summary.summary_ref
    )
    return {
        "phase": "SUMMARIZE",
        "status": summary.status,
        "route": "DONE",
        "summary_ref": summary.summary_ref,
        "conclusion": summary.conclusion,
    }


def route_after_stage(state: PipelineState) -> Literal["CONTINUE", "STOP"]:
    """Continue only after a successful child; every other result goes to summary."""
    route = state.get("route")
    if route in {"CONTINUE", "STOP"}:
        return route
    raise ValueError("pipeline stage did not produce a supported route")


def build_pipeline_graph(
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[
    PipelineState,
    PipelineDependencies,
    PipelineState,
    PipelineState,
]:
    """Compile research -> repository -> adaptation -> training -> evaluation -> summary."""
    saver = checkpointer if checkpointer is not None else InMemorySaver()
    builder = StateGraph(PipelineState, context_schema=PipelineDependencies)
    builder.add_node("research", research_stage)
    builder.add_node("repository", repository_stage)
    builder.add_node("adaptation", adaptation_stage)
    builder.add_node("training", training_stage)
    builder.add_node("evaluation", evaluation_stage)
    builder.add_node("summarize", summarize_stage)

    builder.add_edge(START, "research")
    builder.add_conditional_edges(
        "research",
        route_after_stage,
        {"CONTINUE": "repository", "STOP": "summarize"},
    )
    builder.add_conditional_edges(
        "repository",
        route_after_stage,
        {"CONTINUE": "adaptation", "STOP": "summarize"},
    )
    builder.add_conditional_edges(
        "adaptation",
        route_after_stage,
        {"CONTINUE": "training", "STOP": "summarize"},
    )
    builder.add_conditional_edges(
        "training",
        route_after_stage,
        {"CONTINUE": "evaluation", "STOP": "summarize"},
    )
    builder.add_conditional_edges(
        "evaluation",
        route_after_stage,
        {"CONTINUE": "summarize", "STOP": "summarize"},
    )
    builder.add_edge("summarize", END)
    return builder.compile(checkpointer=saver)


__all__ = ["build_pipeline_graph"]
