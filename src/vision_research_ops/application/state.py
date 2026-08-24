"""Small, checkpoint-safe LangGraph state for the vertical-slice workflow vertical slice."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from math import isfinite
from typing import Annotated, Literal, TypedDict, cast

from pydantic import BaseModel, ConfigDict, field_validator

from vision_research_ops.domain import StructuredFailure, WorkflowPhase, WorkflowStatus


def merge_stable_ids(current: list[str] | None, update: list[str] | None) -> list[str]:
    """Append IDs once while preserving their first-observed order."""
    merged: list[str] = []
    for value in (current or []) + (update or []):
        if value not in merged:
            merged.append(value)
    return merged


def add_retry_counts(
    current: dict[str, int] | None,
    update: dict[str, int] | None,
) -> dict[str, int]:
    """Accumulate retry deltas by stable counter name."""
    merged = dict(current or {})
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in merged.values()
    ):
        raise ValueError("retry counts must be non-negative integers")
    for key, value in (update or {}).items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("retry count updates must be non-negative integers")
        merged[key] = merged.get(key, 0) + value
    return merged


def add_budget_used(
    current: dict[str, float] | None,
    update: dict[str, float] | None,
) -> dict[str, float]:
    """Accumulate budget-use deltas by stable category name."""
    merged = dict(current or {})
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or value < 0
        for value in merged.values()
    ):
        raise ValueError("budget use must be finite and non-negative")
    for key, value in (update or {}).items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(value)
            or value < 0
        ):
            raise ValueError("budget-use updates must be finite and non-negative")
        merged[key] = merged.get(key, 0.0) + value
    return merged


class WorkflowState(TypedDict, total=False):
    """Normative small-reference workflow state from docs/03 §7."""

    schema_version: Literal["1"]
    workflow_id: str
    thread_id: str
    request_id: str
    phase: WorkflowPhase
    status: WorkflowStatus

    dataset_profile_id: str
    paper_candidate_ids: Annotated[list[str], merge_stable_ids]
    selected_paper_ids: Annotated[list[str], merge_stable_ids]
    repository_snapshot_ids: Annotated[list[str], merge_stable_ids]
    active_repository_id: str | None
    active_plan_id: str | None
    active_attempt_id: str | None
    validation_result_ids: Annotated[list[str], merge_stable_ids]
    experiment_id: str | None
    run_ids: Annotated[list[str], merge_stable_ids]
    report_id: str | None

    pending_gate_id: str | None
    retry_counts: Annotated[dict[str, int], add_retry_counts]
    budget_used: Annotated[dict[str, float], add_budget_used]
    last_error: StructuredFailure | None
    next_poll_at: datetime | None
    route: str | None


class InitialWorkflowInput(BaseModel):
    """Validated minimum input for a new vertical-slice workflow workflow thread."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )

    schema_version: Literal["1"] = "1"
    workflow_id: str
    thread_id: str
    request_id: str
    dataset_profile_id: str

    @field_validator("workflow_id", "thread_id", "request_id", "dataset_profile_id")
    @classmethod
    def _require_nonblank_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("workflow identifiers must not be blank")
        return value


def create_initial_state(
    input_data: InitialWorkflowInput | Mapping[str, object],
) -> WorkflowState:
    """Create the smallest valid initial state after strict Pydantic validation."""
    initial = (
        input_data
        if isinstance(input_data, InitialWorkflowInput)
        else InitialWorkflowInput.model_validate(input_data)
    )
    state: WorkflowState = {
        "schema_version": initial.schema_version,
        "workflow_id": initial.workflow_id,
        "thread_id": initial.thread_id,
        "request_id": initial.request_id,
        "dataset_profile_id": initial.dataset_profile_id,
        "phase": WorkflowPhase.REQUEST_VALIDATION,
        "status": WorkflowStatus.PENDING,
        "paper_candidate_ids": [],
        "selected_paper_ids": [],
        "repository_snapshot_ids": [],
        "active_repository_id": None,
        "active_plan_id": None,
        "active_attempt_id": None,
        "validation_result_ids": [],
        "experiment_id": None,
        "run_ids": [],
        "report_id": None,
        "pending_gate_id": None,
        "retry_counts": {},
        "budget_used": {},
        "last_error": None,
        "next_poll_at": None,
        "route": None,
    }
    return state


def _to_jsonable(value: object) -> object:
    """Convert sanctioned state values to ordinary JSON-safe Python values."""
    if isinstance(value, BaseModel):
        dumped = cast(dict[str, object], value.model_dump(mode="json"))
        return _to_jsonable(dumped)
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("workflow state mappings must have string keys")
            converted[key] = _to_jsonable(item)
        return converted
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Enum):
        return value.value
    return value


def workflow_state_as_jsonable(state: Mapping[str, object]) -> dict[str, object]:
    """Return a JSON-safe projection and reject non-finite state values."""
    converted = _to_jsonable(state)
    if not isinstance(converted, dict):
        raise TypeError("workflow state must serialize to a JSON object")
    json.dumps(converted, allow_nan=False)
    return converted


__all__ = [
    "InitialWorkflowInput",
    "WorkflowState",
    "add_budget_used",
    "add_retry_counts",
    "create_initial_state",
    "merge_stable_ids",
    "workflow_state_as_jsonable",
]
