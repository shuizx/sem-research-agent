"""Interactive and explicitly scripted typed decisions for integrated child Gates."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal, cast

from vision_research_ops.application.pipeline_runtime import DecisionProvider
from vision_research_ops.application.services.pipeline_models import PipelineStageName
from vision_research_ops.domain import (
    Approval,
    ApprovalDecision,
    GateKind,
    JsonValue,
    PatchOperation,
    PatchOperationType,
)

type DecisionName = Literal["approve", "edit", "reject"]


class DecisionProviderError(ValueError):
    """Sanitized configuration/input failure at a real human Gate."""


def parse_decision_script(value: str) -> tuple[DecisionName, ...]:
    """Parse a comma-separated ordered fixture decision script."""
    tokens = tuple(item.strip().casefold() for item in value.split(",") if item.strip())
    if not tokens or any(item not in {"approve", "edit", "reject"} for item in tokens):
        raise DecisionProviderError(
            "--decisions must be a comma-separated sequence of approve, edit, or reject"
        )
    return cast(tuple[DecisionName, ...], tokens)


def _required(payload: dict[str, object], name: str, expected: type[object]) -> object:
    value = payload.get(name)
    if not isinstance(value, expected):
        raise DecisionProviderError(f"Gate payload field {name} is invalid")
    return value


def _candidate_edit(payload: dict[str, object], selected_ids: list[str]) -> PatchOperation:
    recommended = payload.get("recommended_papers")
    if not isinstance(recommended, list):
        raise DecisionProviderError("candidate Gate is missing recommended papers")
    allowed = {
        item.get("paper_id")
        for item in recommended
        if isinstance(item, dict) and isinstance(item.get("paper_id"), str)
    }
    if not selected_ids or any(item not in allowed for item in selected_ids):
        raise DecisionProviderError("candidate edit must select listed paper IDs")
    return PatchOperation(
        schema_version="1",
        op=PatchOperationType.REPLACE,
        path="/selected_paper_ids",
        value=cast(list[JsonValue], selected_ids),
        reason="The pipeline reviewer selected an explicit candidate subset.",
    )


def _repository_edit(repository_url: str) -> PatchOperation:
    return PatchOperation(
        schema_version="1",
        op=PatchOperationType.REPLACE,
        path="/repository_url",
        value=repository_url,
        reason="The pipeline reviewer supplied the public repository URL.",
    )


def _patch_edit(path: str, value: JsonValue) -> PatchOperation:
    allowed = {
        "/channels",
        "/num_classes",
        "/label_mapping",
        "/group_split_key",
        "/metrics",
        "/metrics_output_file",
    }
    if path not in allowed:
        raise DecisionProviderError("patch edit path is outside the bounded plan fields")
    return PatchOperation(
        schema_version="1",
        op=PatchOperationType.REPLACE,
        path=path,
        value=value,
        reason="The pipeline reviewer edited one bounded adaptation-plan field.",
    )


def _training_edit(path: str, value: int) -> PatchOperation:
    allowed = {
        "/budget/max_epochs",
        "/budget/max_steps",
        "/budget/max_walltime_seconds",
        "/seed",
    }
    if path not in allowed:
        raise DecisionProviderError("training edit path is outside the frozen fixture fields")
    return PatchOperation(
        schema_version="1",
        op=PatchOperationType.REPLACE,
        path=path,
        value=value,
        reason="The pipeline reviewer edited one bounded integer training field.",
    )


def _scripted_edit(payload: dict[str, object], gate_kind: GateKind) -> list[PatchOperation]:
    if gate_kind is GateKind.CANDIDATE_SELECTION:
        recommended = payload.get("recommended_papers")
        if (
            not isinstance(recommended, list)
            or not recommended
            or not isinstance(recommended[0], dict)
        ):
            raise DecisionProviderError("scripted candidate edit lacks a recommended paper")
        paper_id = recommended[0].get("paper_id")
        if not isinstance(paper_id, str):
            raise DecisionProviderError("scripted candidate paper ID is invalid")
        return [_candidate_edit(payload, [paper_id])]
    if gate_kind is GateKind.REPOSITORY_INGEST:
        repository_url = _required(payload, "repository_url", str)
        return [_repository_edit(cast(str, repository_url))]
    if gate_kind is GateKind.PATCH_ACCEPTANCE:
        plan = payload.get("plan")
        if not isinstance(plan, dict):
            raise DecisionProviderError("scripted patch edit lacks a public plan")
        output_file = plan.get("metrics_output_file")
        if not isinstance(output_file, str):
            raise DecisionProviderError("scripted patch edit lacks metrics_output_file")
        return [_patch_edit("/metrics_output_file", output_file)]
    if gate_kind is GateKind.RUN_SUBMISSION:
        seed = _required(payload, "seed", int)
        if isinstance(seed, bool):
            raise DecisionProviderError("scripted training seed is invalid")
        return [_training_edit("/seed", cast(int, seed) + 1)]
    raise DecisionProviderError("unsupported scripted Gate kind")


def _approval(
    *,
    workflow_id: str,
    payload: dict[str, object],
    occurrence: int,
    decision: DecisionName,
    edits: list[PatchOperation],
    decided_at: datetime,
) -> Approval:
    gate_kind = GateKind(cast(str, _required(payload, "gate_kind", str)))
    subject_type = cast(str, _required(payload, "subject_type", str))
    subject_id = cast(str, _required(payload, "subject_id", str))
    subject_revision = _required(payload, "subject_revision", int)
    if isinstance(subject_revision, bool):
        raise DecisionProviderError("Gate subject revision is invalid")
    typed_decision = ApprovalDecision(decision.upper())
    return Approval(
        schema_version="1",
        approval_id=f"approval-{workflow_id}-{gate_kind.value.casefold()}-{occurrence}",
        gate_kind=gate_kind,
        subject_type=subject_type,
        subject_id=subject_id,
        subject_revision=cast(int, subject_revision),
        decision=typed_decision,
        edits=edits,
        reason="Bounded pipeline fixture decision for a real LangGraph interrupt.",
        actor_id="pipeline-reviewer",
        decided_at=decided_at,
        idempotency_key=f"decision-{workflow_id}-{gate_kind.value.casefold()}-{occurrence}",
    )


class ScriptedDecisionProvider(DecisionProvider):
    """Fixture-only ordered decisions; auto mode emits APPROVE for every real Gate."""

    def __init__(
        self,
        *,
        decisions: Sequence[DecisionName] = (),
        auto_approve: bool = False,
    ) -> None:
        if auto_approve and decisions:
            raise DecisionProviderError("auto approval and an explicit decision script conflict")
        if not auto_approve and not decisions:
            raise DecisionProviderError("scripted decisions require a sequence or auto approval")
        self._decisions = tuple(decisions)
        self._auto_approve = auto_approve
        self._cursor = 0

    @property
    def scripted_fixture_decisions(self) -> bool:
        return True

    @property
    def consumed_count(self) -> int:
        """Return the number of real interrupts served by the script."""
        return self._cursor

    def decide(
        self,
        *,
        workflow_id: str,
        stage: PipelineStageName,
        payload: dict[str, object],
        occurrence: int,
        decided_at: datetime,
    ) -> Approval:
        del stage
        if self._auto_approve:
            decision: DecisionName = "approve"
        else:
            if self._cursor >= len(self._decisions):
                raise DecisionProviderError("fixture decision script ended before the next Gate")
            decision = self._decisions[self._cursor]
        self._cursor += 1
        gate_kind = GateKind(cast(str, _required(payload, "gate_kind", str)))
        edits = _scripted_edit(payload, gate_kind) if decision == "edit" else []
        return _approval(
            workflow_id=workflow_id,
            payload=payload,
            occurrence=occurrence,
            decision=decision,
            edits=edits,
            decided_at=decided_at,
        )


class InteractiveDecisionProvider(DecisionProvider):
    """Read approve/edit/reject locally and build only gate-specific typed edits."""

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._input = input_fn
        self._output = output_fn

    @property
    def scripted_fixture_decisions(self) -> bool:
        return False

    def _decision(self) -> DecisionName:
        entered = self._input("审批决定 [approve/edit/reject]: ").strip().casefold()
        if entered not in {"approve", "edit", "reject"}:
            raise DecisionProviderError("decision must be approve, edit, or reject")
        return cast(DecisionName, entered)

    def _interactive_edits(
        self,
        payload: dict[str, object],
        gate_kind: GateKind,
    ) -> list[PatchOperation]:
        if gate_kind is GateKind.CANDIDATE_SELECTION:
            raw = self._input("保留的 paper_id (英文逗号分隔): ")
            selected = list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
            return [_candidate_edit(payload, selected)]
        if gate_kind is GateKind.REPOSITORY_INGEST:
            return [_repository_edit(self._input("公开 GitHub repository URL: ").strip())]
        if gate_kind is GateKind.PATCH_ACCEPTANCE:
            path = self._input("plan JSON pointer (例如 /metrics_output_file): ").strip()
            raw_value = self._input("新的 JSON 值: ")
            try:
                value = cast(JsonValue, json.loads(raw_value))
            except json.JSONDecodeError as error:
                raise DecisionProviderError("patch edit value must be valid JSON") from error
            return [_patch_edit(path, value)]
        if gate_kind is GateKind.RUN_SUBMISSION:
            path = self._input("training JSON pointer (例如 /seed): ").strip()
            raw_value = self._input("新的整数值: ").strip()
            try:
                value = int(raw_value)
            except ValueError as error:
                raise DecisionProviderError("training edit value must be an integer") from error
            return [_training_edit(path, value)]
        raise DecisionProviderError("unsupported interactive Gate kind")

    def decide(
        self,
        *,
        workflow_id: str,
        stage: PipelineStageName,
        payload: dict[str, object],
        occurrence: int,
        decided_at: datetime,
    ) -> Approval:
        gate_kind = GateKind(cast(str, _required(payload, "gate_kind", str)))
        self._output(
            json.dumps(
                {
                    "event": "human_decision_required",
                    "stage": stage,
                    "gate_kind": gate_kind.value,
                    "subject_id": payload.get("subject_id"),
                    "subject_revision": payload.get("subject_revision"),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        decision = self._decision()
        edits = self._interactive_edits(payload, gate_kind) if decision == "edit" else []
        return _approval(
            workflow_id=workflow_id,
            payload=payload,
            occurrence=occurrence,
            decision=decision,
            edits=edits,
            decided_at=decided_at,
        )


__all__ = [
    "DecisionName",
    "DecisionProviderError",
    "InteractiveDecisionProvider",
    "ScriptedDecisionProvider",
    "parse_decision_script",
]
