"""Local JSON evidence store for the single-user adaptation pipeline workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .adaptation_models import (
    AdaptationPlannerTrace,
    AdaptationResult,
    CompiledAdaptationPlan,
    PatchArtifactRecord,
    SmokeResultRecord,
)

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TModel = TypeVar("TModel", bound=BaseModel)


def _safe_component(value: str, *, name: str) -> str:
    if _SAFE_COMPONENT_RE.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"{name} must be a safe local path component")
    return value


def _safe_relative_ref(value: str) -> str:
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or value.startswith("/")
        or ":" in value
        or "%" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("artifact reference must be a canonical POSIX relative path")
    return value


class LocalAdaptationStore:
    """Atomically persist validated adaptation evidence under one ignored ``var`` root."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        """Return the configured local var root for explicit test wiring."""
        return self._root

    def resolve_ref(self, relative_ref: str) -> Path:
        """Resolve a validated artifact ref and prove it remains below the var root."""
        safe = _safe_relative_ref(relative_ref)
        root = self._root.resolve()
        path = (root / Path(*safe.split("/"))).resolve()
        if not path.is_relative_to(root):
            raise ValueError("artifact reference escaped the configured var root")
        return path

    @staticmethod
    def result_ref(workflow_id: str) -> str:
        component = _safe_component(workflow_id, name="workflow_id")
        return f"adaptations/{component}/adaptation.json"

    @staticmethod
    def plan_ref(workflow_id: str, revision: int) -> str:
        component = _safe_component(workflow_id, name="workflow_id")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("plan revision must be a positive integer")
        return f"adaptations/{component}/plan-r{revision}.json"

    @staticmethod
    def planner_trace_ref(workflow_id: str) -> str:
        """Return the canonical hash-only tool trace reference."""
        component = _safe_component(workflow_id, name="workflow_id")
        return f"adaptations/{component}/planner-trace.json"

    @staticmethod
    def patch_manifest_ref(workflow_id: str, revision: int) -> str:
        component = _safe_component(workflow_id, name="workflow_id")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("patch revision must be a positive integer")
        return f"patches/{component}/r{revision}/manifest.json"

    @staticmethod
    def smoke_result_ref(workflow_id: str, revision: int) -> str:
        component = _safe_component(workflow_id, name="workflow_id")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("smoke revision must be a positive integer")
        return f"smoke/{component}/r{revision}/result.json"

    def _write_model(self, ref: str, value: BaseModel) -> str:
        path = self.resolve_ref(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(f"{encoded}\n", encoding="utf-8")
        temporary.replace(path)
        return ref

    def _load_model(self, ref: str, model: type[TModel]) -> TModel:
        return model.model_validate_json(self.resolve_ref(ref).read_text(encoding="utf-8"))

    def write_result(self, result: AdaptationResult) -> str:
        """Write the canonical adaptation evidence index."""
        return self._write_model(self.result_ref(result.workflow_id), result)

    def load_result(self, workflow_id: str) -> AdaptationResult:
        """Load one validated adaptation evidence index."""
        return self._load_model(self.result_ref(workflow_id), AdaptationResult)

    def write_plan(self, plan: CompiledAdaptationPlan) -> str:
        """Write one immutable plan revision."""
        return self._write_model(self.plan_ref(plan.workflow_id, plan.revision), plan)

    def load_plan(self, ref: str) -> CompiledAdaptationPlan:
        """Load one referenced plan revision."""
        return self._load_model(ref, CompiledAdaptationPlan)

    def write_planner_trace(self, trace: AdaptationPlannerTrace) -> str:
        """Write the de-sensitized LangGraph tool trace beside the plan."""
        return self._write_model(self.planner_trace_ref(trace.workflow_id), trace)

    def load_planner_trace(self, ref: str) -> AdaptationPlannerTrace:
        """Load one validated planner trace."""
        return self._load_model(ref, AdaptationPlannerTrace)

    def write_patch_record(self, record: PatchArtifactRecord) -> str:
        """Write the structured manifest adjacent to the exported diff."""
        ref = self.patch_manifest_ref(record.workflow_id, record.plan_revision)
        if record.manifest_ref != ref:
            raise ValueError("patch record manifest_ref is not canonical")
        return self._write_model(ref, record)

    def load_patch_record(self, ref: str) -> PatchArtifactRecord:
        """Load one referenced patch manifest."""
        return self._load_model(ref, PatchArtifactRecord)

    def write_smoke_result(self, result: SmokeResultRecord) -> str:
        """Write one bounded smoke result."""
        ref = self.smoke_result_ref(result.workflow_id, result.plan_revision)
        if result.result_ref != ref:
            raise ValueError("smoke result_ref is not canonical")
        return self._write_model(ref, result)

    def load_smoke_result(self, ref: str) -> SmokeResultRecord:
        """Load one referenced bounded smoke result."""
        return self._load_model(ref, SmokeResultRecord)


__all__ = ["LocalAdaptationStore"]
