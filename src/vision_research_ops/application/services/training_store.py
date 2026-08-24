"""Local JSON evidence store for the single-user training training workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .training_models import (
    FrozenTrainingSpec,
    TrainingMetrics,
    TrainingPredictions,
    TrainingRunManifest,
    TrainingWorkflowRecord,
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


class LocalTrainingStore:
    """Atomically persist validated training evidence below one ignored ``var`` root."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        """Return the configured local var root for explicit runtime wiring."""
        return self._root

    def resolve_ref(self, relative_ref: str) -> Path:
        """Resolve a canonical ref and prove containment below the var root."""
        safe = _safe_relative_ref(relative_ref)
        root = self._root.resolve()
        path = (root / Path(*safe.split("/"))).resolve()
        if not path.is_relative_to(root):
            raise ValueError("artifact reference escaped the configured var root")
        return path

    @staticmethod
    def workflow_ref(workflow_id: str) -> str:
        return f"training/{_safe_component(workflow_id, name='workflow_id')}/training.json"

    @staticmethod
    def spec_ref(workflow_id: str, revision: int) -> str:
        component = _safe_component(workflow_id, name="workflow_id")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("training revision must be a positive integer")
        return f"training/{component}/spec-r{revision}.json"

    @staticmethod
    def run_prefix(run_id: str) -> str:
        return f"runs/{_safe_component(run_id, name='run_id')}"

    @classmethod
    def manifest_ref(cls, run_id: str) -> str:
        return f"{cls.run_prefix(run_id)}/manifest.json"

    @classmethod
    def log_ref(cls, run_id: str) -> str:
        return f"{cls.run_prefix(run_id)}/train.log"

    @classmethod
    def metrics_ref(cls, run_id: str) -> str:
        return f"{cls.run_prefix(run_id)}/metrics.json"

    @classmethod
    def predictions_ref(cls, run_id: str) -> str:
        return f"{cls.run_prefix(run_id)}/predictions.json"

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

    def write_workflow(self, record: TrainingWorkflowRecord) -> str:
        """Write the canonical training workflow evidence index."""
        return self._write_model(self.workflow_ref(record.workflow_id), record)

    def load_workflow(self, workflow_id: str) -> TrainingWorkflowRecord:
        """Load one validated training workflow index."""
        return self._load_model(self.workflow_ref(workflow_id), TrainingWorkflowRecord)

    def write_spec(self, spec: FrozenTrainingSpec) -> str:
        """Write one immutable frozen submission revision."""
        return self._write_model(self.spec_ref(spec.workflow_id, spec.revision), spec)

    def load_spec(self, ref: str) -> FrozenTrainingSpec:
        """Load one referenced frozen submission revision."""
        return self._load_model(ref, FrozenTrainingSpec)

    def load_manifest(self, run_id: str) -> TrainingRunManifest:
        """Load one strict run manifest written by the trusted fixture."""
        return self._load_model(self.manifest_ref(run_id), TrainingRunManifest)

    def load_metrics(self, run_id: str) -> TrainingMetrics:
        """Load one strict metrics artifact."""
        return self._load_model(self.metrics_ref(run_id), TrainingMetrics)

    def load_predictions(self, run_id: str) -> TrainingPredictions:
        """Load one strict predictions artifact."""
        return self._load_model(self.predictions_ref(run_id), TrainingPredictions)


__all__ = ["LocalTrainingStore"]
