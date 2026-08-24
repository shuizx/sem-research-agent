"""Runtime-only boundaries for the evaluation deterministic Evaluation Agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .services.evaluation_models import EVALUATION_POLICY_REF
from .services.evaluation_store import LocalEvaluationStore


@runtime_checkable
class TrainingArtifactReader(Protocol):
    """Resolve only validated training-local relative refs below its trusted var root."""

    def resolve_ref(self, relative_ref: str) -> Path:
        """Resolve a canonical training artifact reference below the configured var root."""


@dataclass(frozen=True, slots=True)
class EvaluationDependencies:
    """Injected training reader, trusted fixture root, and write-once evaluation store."""

    training_reader: TrainingArtifactReader
    project_root: Path
    store: LocalEvaluationStore
    policy_ref: str = EVALUATION_POLICY_REF

    def __post_init__(self) -> None:
        if not isinstance(self.project_root, Path):
            raise TypeError("project_root must be a trusted Path")
        if self.policy_ref != EVALUATION_POLICY_REF:
            raise ValueError("evaluation supports only the pre-registered fixture policy")
        if self.store.root.resolve() != (self.project_root / "var").resolve():
            raise ValueError("evaluation outputs must remain below the trusted project var root")


__all__ = ["EvaluationDependencies", "TrainingArtifactReader"]
