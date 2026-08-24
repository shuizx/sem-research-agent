"""Runtime-only dependencies for the training workflow Training Agent graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from vision_research_ops.ports import OperationContext

from .runtime import ApprovalRecorder
from .services.training_freeze import AcceptedAdaptationReader
from .services.training_models import (
    FrozenRunSpec,
    FrozenTrainingSpec,
    TrainingInput,
    TrainingRunResult,
)
from .services.training_store import LocalTrainingStore


def training_utc_now() -> datetime:
    """Return an aware UTC timestamp for injected training dependencies."""
    return datetime.now(UTC)


class TrainingToolError(Exception):
    """Sanitized explicit failure returned by the controlled training boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@runtime_checkable
class ControlledTrainingTool(Protocol):
    """Execute one exact frozen run through a trusted local entrypoint."""

    async def run(
        self,
        spec: FrozenTrainingSpec,
        run: FrozenRunSpec,
        *,
        ctx: OperationContext,
    ) -> TrainingRunResult:
        """Return validated relative artifact references or fail explicitly."""


@runtime_checkable
class CancellationSignal(Protocol):
    """Small synchronous signal checked before each irreversible run submission."""

    def is_cancelled(self) -> bool:
        """Return whether ordinary user cancellation has been requested."""


class NeverCancelled:
    """Default single-process cancellation signal for the normal fixture path."""

    def is_cancelled(self) -> bool:
        return False


@dataclass(slots=True)
class TrainingDependencies:
    """Injected accepted adaptation reader, fixed input, local trainer/store, and Gate audit."""

    adaptation_reader: AcceptedAdaptationReader
    training_input: TrainingInput | Mapping[str, object]
    project_root: Path
    store: LocalTrainingStore
    trainer: ControlledTrainingTool
    approval_recorder: ApprovalRecorder
    cancellation: CancellationSignal
    actor_id: str = "pipeline-user"
    clock: Callable[[], datetime] = training_utc_now

    def __post_init__(self) -> None:
        if not isinstance(self.project_root, Path):
            raise TypeError("project_root must be a trusted Path")
        if not isinstance(self.actor_id, str) or not self.actor_id.strip():
            raise ValueError("actor_id must be a non-blank string")
        if not callable(self.clock):
            raise TypeError("training clock must be callable")


__all__ = [
    "CancellationSignal",
    "ControlledTrainingTool",
    "NeverCancelled",
    "TrainingDependencies",
    "TrainingToolError",
    "training_utc_now",
]
