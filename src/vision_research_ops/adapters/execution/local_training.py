"""Fixed local subprocess adapter for the training stdlib training fixture."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from math import isclose, isfinite
from pathlib import Path

from pydantic import ValidationError

from vision_research_ops.application.services.training_models import (
    TRAINING_CAPABILITY,
    TRAINING_ENTRYPOINT_REF,
    FrozenRunSpec,
    FrozenTrainingSpec,
    TrainingMetrics,
    TrainingRunResult,
    content_hash,
)
from vision_research_ops.application.services.training_store import LocalTrainingStore
from vision_research_ops.application.training_runtime import TrainingToolError
from vision_research_ops.ports import OperationContext

_ENTRYPOINT_HASH = "sha256:398fde426893130684430ddba27813ed9383b80e66788c88f0cba54b5957b50b"
_SENSITIVE_MARKERS = (
    "authorization:",
    "dashscope_api_key",
    "api_key=",
    "bearer ",
    "private key",
)
_LOG_EVENT_VOCABULARY = frozenset({"run_started", "epoch_completed", "run_completed"})


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _validate_training_log(
    log_text: str,
    *,
    run: FrozenRunSpec,
    metrics: TrainingMetrics,
) -> None:
    """Validate the fixture's small start/epoch/completion JSONL contract."""
    lines = [line for line in log_text.splitlines() if line.strip()]
    if len(lines) < 3 or len(lines) > run.budget.max_epochs + 2:
        raise ValueError("training log does not have a bounded event count")
    events: list[dict[str, object]] = []
    for line in lines:
        parsed = json.loads(line, parse_constant=_reject_nonfinite_json)
        if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
            raise ValueError("each training log line must be one JSON object")
        event_name = parsed.get("event")
        if event_name not in _LOG_EVENT_VOCABULARY:
            raise ValueError("training log event is outside the fixture vocabulary")
        events.append(parsed)

    started = events[0]
    if (
        set(started) != {"event", "role", "run_id"}
        or started["event"] != "run_started"
        or started["role"] != run.role
        or started["run_id"] != run.run_id
    ):
        raise ValueError("training log must start with the exact run identity")
    completed = events[-1]
    if (
        set(completed) != {"capability", "event", "real_pytorch_training"}
        or completed["event"] != "run_completed"
        or completed["capability"] != TRAINING_CAPABILITY
        or completed["real_pytorch_training"] is not False
    ):
        raise ValueError("training log must end with the exact completion event")

    epoch_events = events[1:-1]
    if len(epoch_events) != len(metrics.epoch_losses):
        raise ValueError("training log epoch count conflicts with metrics")
    for event, epoch_loss in zip(epoch_events, metrics.epoch_losses, strict=True):
        epoch = event.get("epoch")
        steps = event.get("steps")
        mean_loss = event.get("mean_loss")
        if (
            set(event) != {"epoch", "event", "mean_loss", "steps"}
            or event.get("event") != "epoch_completed"
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch != epoch_loss.epoch
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps != epoch_loss.steps
            or isinstance(mean_loss, bool)
            or not isinstance(mean_loss, int | float)
            or not isfinite(mean_loss)
            or mean_loss < 0.0
            or not isclose(mean_loss, epoch_loss.mean_loss, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError("training log epoch event conflicts with structured metrics")


class LocalTrainingExecutor:
    """Run only the hash-allowlisted fixture with fixed argv and strict outputs."""

    def __init__(self, *, project_root: Path, store: LocalTrainingStore) -> None:
        self._project_root = project_root.resolve()
        self._store = store
        if self._store.root.resolve() != (self._project_root / "var").resolve():
            raise ValueError("local training store must be the trusted project var directory")
        self.call_count = 0
        self.commands: list[tuple[str, ...]] = []

    @staticmethod
    def _minimal_environment() -> dict[str, str]:
        environment = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        for key in ("SystemRoot", "WINDIR", "TEMP", "TMP"):
            value = os.environ.get(key)
            if value:
                environment[key] = value
        return environment

    def _validate_entrypoint(self) -> None:
        path = self._project_root / Path(*TRAINING_ENTRYPOINT_REF.split("/"))
        if not path.is_file() or path.is_symlink():
            raise TrainingToolError("TRAINING_ENTRYPOINT_POLICY_FAILED")
        try:
            normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
        except (OSError, UnicodeError) as error:
            raise TrainingToolError("TRAINING_ENTRYPOINT_POLICY_FAILED") from error
        if content_hash(normalized) != _ENTRYPOINT_HASH:
            raise TrainingToolError("TRAINING_ENTRYPOINT_POLICY_FAILED")

    def _validate_artifacts(
        self,
        spec: FrozenTrainingSpec,
        run: FrozenRunSpec,
        *,
        reused_existing: bool,
    ) -> TrainingRunResult:
        try:
            manifest = self._store.load_manifest(run.run_id)
            metrics = self._store.load_metrics(run.run_id)
            predictions = self._store.load_predictions(run.run_id)
            log_path = self._store.resolve_ref(self._store.log_ref(run.run_id))
            if not log_path.is_file() or log_path.is_symlink():
                raise ValueError("training log is absent or unsafe")
            log_text = log_path.read_text(encoding="utf-8")
            if not log_text or len(log_text.encode("utf-8")) > 65_536:
                raise ValueError("training log is empty or exceeds the fixture bound")
            _validate_training_log(log_text, run=run, metrics=metrics)
            run_fields = (
                "run_id",
                "role",
                "method",
                "base_commit_sha",
                "candidate_patch_revision",
                "candidate_patch_hash",
                "dataset_id",
                "dataset_version",
                "dataset_content_hash",
                "dataset_ref",
                "dataset_ref_hash",
                "split_ref",
                "split_hash",
                "preprocess_ref",
                "preprocess_hash",
                "method_config_ref",
                "method_config_hash",
                "seed",
                "budget",
                "command",
            )
            for field in run_fields:
                if getattr(manifest, field) != getattr(run, field):
                    raise ValueError("training manifest conflicts with the frozen run")
            if manifest.spec_hash != spec.spec_hash or manifest.spec_ref != run.command.spec_ref:
                raise ValueError("training manifest conflicts with the frozen spec")
            if (
                metrics.run_id != run.run_id
                or metrics.role != run.role
                or metrics.spec_hash != spec.spec_hash
                or metrics.seed != run.seed
                or metrics.budget != run.budget
                or metrics.prediction_count != len(predictions.items)
                or predictions.run_id != run.run_id
                or predictions.role != run.role
                or predictions.spec_hash != spec.spec_hash
                or predictions.split_ref != run.split_ref
                or predictions.split_hash != run.split_hash
            ):
                raise ValueError("training outputs conflict with the frozen run")
            raw_artifacts = "\n".join(
                self._store.resolve_ref(ref).read_text(encoding="utf-8")
                for ref in (
                    self._store.manifest_ref(run.run_id),
                    self._store.log_ref(run.run_id),
                    self._store.metrics_ref(run.run_id),
                    self._store.predictions_ref(run.run_id),
                )
            )
            lowered = raw_artifacts.casefold()
            if (
                str(self._project_root).casefold() in lowered
                or self._project_root.as_posix().casefold() in lowered
                or any(marker in lowered for marker in _SENSITIVE_MARKERS)
            ):
                raise ValueError("training artifacts contain a forbidden path or secret marker")
        except (OSError, UnicodeError, TypeError, ValueError, ValidationError) as error:
            raise TrainingToolError("TRAINING_ARTIFACT_INVALID") from error
        return TrainingRunResult(
            run_id=run.run_id,
            role=run.role,
            spec_hash=spec.spec_hash,
            manifest_ref=manifest.manifest_ref,
            log_ref=manifest.log_ref,
            metrics_ref=manifest.metrics_ref,
            predictions_ref=manifest.predictions_ref,
            capability=TRAINING_CAPABILITY,
            real_pytorch_training=False,
            reused_existing=reused_existing,
        )

    async def run(
        self,
        spec: FrozenTrainingSpec,
        run: FrozenRunSpec,
        *,
        ctx: OperationContext,
    ) -> TrainingRunResult:
        """Execute once or validate an existing complete run with the same frozen spec."""
        if not ctx.idempotency_key:
            raise TrainingToolError("TRAINING_IDEMPOTENCY_KEY_REQUIRED")
        if run != spec.baseline and run != spec.candidate:
            raise TrainingToolError("TRAINING_RUN_NOT_IN_FROZEN_SPEC")
        spec_ref = self._store.spec_ref(spec.workflow_id, spec.revision)
        if run.command.spec_ref != spec_ref or self._store.load_spec(spec_ref) != spec:
            raise TrainingToolError("TRAINING_FROZEN_SPEC_CONFLICT")
        manifest_path = self._store.resolve_ref(self._store.manifest_ref(run.run_id))
        if manifest_path.exists():
            return self._validate_artifacts(spec, run, reused_existing=True)
        output_path = self._store.resolve_ref(self._store.run_prefix(run.run_id))
        if output_path.exists():
            raise TrainingToolError("TRAINING_ARTIFACT_INVALID")
        self._validate_entrypoint()
        actual_command = (sys.executable, *run.command.argv)
        self.commands.append(actual_command)
        self.call_count += 1
        try:
            completed = subprocess.run(
                actual_command,
                cwd=self._project_root,
                env=self._minimal_environment(),
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=run.budget.max_walltime_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise TrainingToolError("TRAINING_RUN_TIMEOUT") from error
        except OSError as error:
            raise TrainingToolError("TRAINING_EXECUTOR_FAILED") from error
        if completed.returncode != 0:
            raise TrainingToolError("TRAINING_NONZERO_EXIT")
        return self._validate_artifacts(spec, run, reused_existing=False)


__all__ = ["LocalTrainingExecutor"]
