"""LangGraph nodes for deterministic validation, metrics, conclusion, and reporting."""

from __future__ import annotations

from typing import Literal, cast

from langgraph.runtime import Runtime

from ..evaluation_runtime import EvaluationDependencies
from ..services.evaluation_engine import EvaluationEngineError, compute_evaluation
from ..services.evaluation_models import EvaluationFailure, EvaluationState
from ..services.evaluation_store import EvaluationStoreError


def _failed(code: str, message: str) -> EvaluationState:
    failure = EvaluationFailure(code=code, message=message)
    return {
        "status": "FAILED",
        "route": "FAILED",
        "last_error": {
            "code": failure.code,
            "message": failure.message,
        },
    }


def _identity_is_stable(state: EvaluationState, evaluation_id: str) -> bool:
    existing = state.get("evaluation_id")
    return existing is None or existing == evaluation_id


async def load_training_outputs(
    state: EvaluationState,
    runtime: Runtime[EvaluationDependencies],
) -> EvaluationState:
    """Read and schema-check exact training workflow, spec, and structured artifacts."""
    try:
        computation = compute_evaluation(state, runtime.context)
    except EvaluationEngineError as error:
        return _failed(error.code, "The deterministic evaluation configuration is invalid.")
    return {
        "status": "RUNNING",
        "route": "VALIDATE" if computation.outputs_loaded else "INVALID",
        "evaluation_id": computation.result.evaluation_id,
        "last_error": None,
    }


async def validate_comparability(
    state: EvaluationState,
    runtime: Runtime[EvaluationDependencies],
) -> EvaluationState:
    """Require exact data, split, preprocessing, seed, budget, samples, and truth."""
    try:
        computation = compute_evaluation(state, runtime.context)
    except EvaluationEngineError as error:
        return _failed(error.code, "The deterministic comparability check could not run.")
    if not _identity_is_stable(state, computation.result.evaluation_id):
        return _failed(
            "EVALUATION_INPUT_CHANGED_DURING_RUN",
            "Evaluation inputs changed after the graph started.",
        )
    return {
        "status": "RUNNING",
        "route": ("INVALID" if computation.result.conclusion == "INVALID" else "COMPUTE_METRICS"),
        "evaluation_id": computation.result.evaluation_id,
        "last_error": None,
    }


async def invalid_result(
    state: EvaluationState,
    runtime: Runtime[EvaluationDependencies],
) -> EvaluationState:
    """Persist a complete INVALID result rather than mislabeling it as regression."""
    try:
        computation = compute_evaluation(state, runtime.context)
        result = computation.result
        if result.conclusion != "INVALID" or not _identity_is_stable(state, result.evaluation_id):
            raise EvaluationEngineError("EVALUATION_INPUT_CHANGED_DURING_RUN")
        persisted = runtime.context.store.write_evaluation(result)
    except EvaluationEngineError as error:
        return _failed(error.code, "Invalid evaluation inputs changed during execution.")
    except EvaluationStoreError as error:
        return _failed(error.code, "The canonical INVALID evaluation could not be persisted.")
    return {
        "status": "RUNNING",
        "route": "REPORT",
        "conclusion": "INVALID",
        "evaluation_id": result.evaluation_id,
        "evaluation_ref": persisted.ref,
        "report_ref": result.report_ref,
        "last_error": None,
    }


async def compute_metrics_node(
    state: EvaluationState,
    runtime: Runtime[EvaluationDependencies],
) -> EvaluationState:
    """Verify that deterministic fixed-label metrics can be reproduced from predictions."""
    try:
        computation = compute_evaluation(state, runtime.context)
    except EvaluationEngineError as error:
        return _failed(error.code, "Deterministic classification metrics could not be computed.")
    result = computation.result
    if (
        result.conclusion == "INVALID"
        or result.baseline_metrics is None
        or result.candidate_metrics is None
        or result.deltas is None
        or not _identity_is_stable(state, result.evaluation_id)
    ):
        return _failed(
            "EVALUATION_INPUT_CHANGED_DURING_RUN",
            "Evaluation inputs no longer match the validated pair.",
        )
    return {
        "status": "RUNNING",
        "route": "DECIDE",
        "evaluation_id": result.evaluation_id,
        "last_error": None,
    }


async def decide_conclusion(
    state: EvaluationState,
    runtime: Runtime[EvaluationDependencies],
) -> EvaluationState:
    """Reproduce the pre-registered four-way deterministic conclusion."""
    try:
        result = compute_evaluation(state, runtime.context).result
    except EvaluationEngineError as error:
        return _failed(error.code, "The fixed conclusion policy could not be applied.")
    if result.conclusion == "INVALID" or not _identity_is_stable(state, result.evaluation_id):
        return _failed(
            "EVALUATION_INPUT_CHANGED_DURING_RUN",
            "Evaluation inputs changed before conclusion.",
        )
    return {
        "status": "RUNNING",
        "route": "PERSIST",
        "conclusion": result.conclusion,
        "evaluation_id": result.evaluation_id,
        "last_error": None,
    }


async def persist_evaluation(
    state: EvaluationState,
    runtime: Runtime[EvaluationDependencies],
) -> EvaluationState:
    """Write canonical evaluation.json once or verify exact existing bytes and hash."""
    try:
        result = compute_evaluation(state, runtime.context).result
        if (
            result.conclusion == "INVALID"
            or state.get("conclusion") != result.conclusion
            or not _identity_is_stable(state, result.evaluation_id)
        ):
            raise EvaluationEngineError("EVALUATION_INPUT_CHANGED_DURING_RUN")
        persisted = runtime.context.store.write_evaluation(result)
    except EvaluationEngineError as error:
        return _failed(error.code, "Evaluation inputs changed before persistence.")
    except EvaluationStoreError as error:
        return _failed(error.code, "The canonical evaluation could not be persisted.")
    return {
        "status": "RUNNING",
        "route": "REPORT",
        "evaluation_id": result.evaluation_id,
        "evaluation_ref": persisted.ref,
        "report_ref": result.report_ref,
        "last_error": None,
    }


async def render_report(
    state: EvaluationState,
    runtime: Runtime[EvaluationDependencies],
) -> EvaluationState:
    """Load validated evaluation.json and render the only permitted Markdown template."""
    workflow_id = state.get("workflow_id")
    if not isinstance(workflow_id, str):
        return _failed("EVALUATION_REQUEST_INVALID", "Evaluation workflow ID is missing.")
    try:
        result = runtime.context.store.load_evaluation(workflow_id)
        if (
            state.get("evaluation_id") != result.evaluation_id
            or state.get("conclusion") != result.conclusion
            or state.get("evaluation_ref") != result.evaluation_ref
            or state.get("report_ref") != result.report_ref
        ):
            raise EvaluationStoreError("EVALUATION_ARTIFACT_CONFLICT")
        persisted = runtime.context.store.write_report(result)
    except EvaluationStoreError as error:
        return _failed(error.code, "The deterministic Markdown report could not be rendered.")
    return {
        "status": "COMPLETED",
        "route": "COMPLETED",
        "conclusion": result.conclusion,
        "evaluation_id": result.evaluation_id,
        "evaluation_ref": result.evaluation_ref,
        "report_ref": persisted.ref,
        "last_error": None,
    }


def route_after_load(state: EvaluationState) -> Literal["VALIDATE", "INVALID", "FAILED"]:
    route = state.get("route")
    if route in {"VALIDATE", "INVALID", "FAILED"}:
        return cast(Literal["VALIDATE", "INVALID", "FAILED"], route)
    raise ValueError("evaluation loading did not produce a supported route")


def route_after_validation(
    state: EvaluationState,
) -> Literal["COMPUTE_METRICS", "INVALID", "FAILED"]:
    route = state.get("route")
    if route in {"COMPUTE_METRICS", "INVALID", "FAILED"}:
        return cast(Literal["COMPUTE_METRICS", "INVALID", "FAILED"], route)
    raise ValueError("evaluation comparability did not produce a supported route")


def route_after_metrics(state: EvaluationState) -> Literal["DECIDE", "FAILED"]:
    route = state.get("route")
    if route in {"DECIDE", "FAILED"}:
        return cast(Literal["DECIDE", "FAILED"], route)
    raise ValueError("evaluation metrics did not produce a supported route")


def route_after_decision(state: EvaluationState) -> Literal["PERSIST", "FAILED"]:
    route = state.get("route")
    if route in {"PERSIST", "FAILED"}:
        return cast(Literal["PERSIST", "FAILED"], route)
    raise ValueError("evaluation decision did not produce a supported route")


def route_after_persistence(state: EvaluationState) -> Literal["REPORT", "FAILED"]:
    route = state.get("route")
    if route in {"REPORT", "FAILED"}:
        return cast(Literal["REPORT", "FAILED"], route)
    raise ValueError("evaluation persistence did not produce a supported route")


__all__ = [
    "compute_metrics_node",
    "decide_conclusion",
    "invalid_result",
    "load_training_outputs",
    "persist_evaluation",
    "render_report",
    "route_after_decision",
    "route_after_load",
    "route_after_metrics",
    "route_after_persistence",
    "route_after_validation",
    "validate_comparability",
]
