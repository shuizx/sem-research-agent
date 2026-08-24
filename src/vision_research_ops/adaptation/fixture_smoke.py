"""Fixed subprocess-backed smoke runner for the controlled adaptation fixture."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from vision_research_ops.application.services.adaptation_models import (
    FixtureProbeOutput,
    PatchArtifactRecord,
    SmokeCommandRecord,
    SmokeResultRecord,
    SmokeStageRecord,
)
from vision_research_ops.application.services.adaptation_patch import (
    canonical_json_bytes,
    content_hash,
)
from vision_research_ops.application.services.adaptation_store import LocalAdaptationStore
from vision_research_ops.domain import JsonObject, ValidationStage, ValidationStatus
from vision_research_ops.ports import OperationContext

_STAGES = (
    ValidationStage.STATIC_POLICY,
    ValidationStage.IMPORT,
    ValidationStage.ONE_BATCH,
    ValidationStage.BOUNDED_OVERFIT,
)


class FixtureSmokeRunner:
    """Run four fixed stages without shell, network, installs, or arbitrary repositories."""

    def __init__(
        self,
        *,
        store: LocalAdaptationStore,
        clock: Callable[[], datetime],
        minimum_repair_revision: int = 0,
        timeout_seconds: int = 5,
    ) -> None:
        if (
            isinstance(minimum_repair_revision, bool)
            or not isinstance(minimum_repair_revision, int)
            or minimum_repair_revision < 0
        ):
            raise ValueError("minimum_repair_revision must be a non-negative integer")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
            raise ValueError("timeout_seconds must be an integer")
        if timeout_seconds < 1 or timeout_seconds > 30:
            raise ValueError("fixture smoke timeout must be between 1 and 30 seconds")
        self._store = store
        self._clock = clock
        self._minimum_repair_revision = minimum_repair_revision
        self._timeout_seconds = timeout_seconds
        self.call_count = 0
        self.stage_call_counts: dict[ValidationStage, int] = {}

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

    def _stage_log_ref(self, patch: PatchArtifactRecord, stage: ValidationStage) -> str:
        return f"smoke/{patch.workflow_id}/r{patch.plan_revision}/{stage.value.casefold()}.json"

    def _execute_stage(
        self,
        patch: PatchArtifactRecord,
        stage: ValidationStage,
        workspace: Path,
    ) -> SmokeStageRecord:
        started_at = self._clock()
        argv = [
            "-I",
            "fixture_probe.py",
            "--stage",
            stage.value,
            "--minimum-repair-revision",
            str(self._minimum_repair_revision),
        ]
        command = SmokeCommandRecord(
            executable_id="python-current",
            argv=argv,
            cwd_ref=patch.workspace_ref,
        )
        status = ValidationStatus.FAILED
        exit_code = 1
        evidence: JsonObject = {
            "probe_output_valid": False,
            "capability_boundary": "FIXTURE_CONTRACT_PROBE_NO_TORCH",
        }
        reason_code = "FIXTURE_PROBE_INVALID_OUTPUT"
        try:
            completed = subprocess.run(
                [sys.executable, *argv],
                cwd=workspace,
                env=self._minimal_environment(),
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self._timeout_seconds,
            )
            exit_code = completed.returncode
            try:
                output = FixtureProbeOutput.model_validate_json(completed.stdout.strip())
            except ValidationError:
                output = None
            if output is not None and output.stage is stage:
                evidence = dict(output.evidence)
                evidence["probe_output_valid"] = True
                evidence["capability_boundary"] = output.capability_boundary
                reason_code = output.reason_code or "FIXTURE_PROBE_NONZERO_EXIT"
                if output.passed and exit_code == 0:
                    status = ValidationStatus.PASSED
                    reason_code = "FIXTURE_PROBE_PASSED"
            elif completed.stderr:
                evidence["stderr_redacted"] = True
        except subprocess.TimeoutExpired:
            exit_code = 124
            reason_code = "FIXTURE_PROBE_TIMEOUT"
            evidence["timed_out"] = True

        finished_at = self._clock()
        log_ref = self._stage_log_ref(patch, stage)
        log_payload = {
            "schema_version": "1",
            "fixture_labeled": True,
            "synthetic_data_labeled": True,
            "stage": stage.value,
            "status": status.value,
            "exit_code": exit_code,
            "reason_code": reason_code,
            "command": command.model_dump(mode="json"),
            "evidence": evidence,
        }
        log_path = self._store.resolve_ref(log_ref)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(canonical_json_bytes(log_payload))
        command_digest = content_hash(canonical_json_bytes(command.model_dump(mode="json")))
        return SmokeStageRecord(
            stage=stage,
            status=status,
            exit_code=exit_code,
            command=command,
            command_digest=command_digest,
            evidence=evidence,
            log_ref=log_ref,
            started_at=started_at,
            finished_at=finished_at,
        )

    async def run(
        self,
        patch: PatchArtifactRecord,
        *,
        ctx: OperationContext,
    ) -> SmokeResultRecord:
        """Execute the fixed stage sequence, stopping on the first actual failure."""
        if not ctx.idempotency_key:
            raise ValueError("smoke execution requires an idempotency key")
        result_ref = self._store.smoke_result_ref(patch.workflow_id, patch.plan_revision)
        result_path = self._store.resolve_ref(result_ref)
        if result_path.exists():
            existing = self._store.load_smoke_result(result_ref)
            if (
                existing.patch_hash != patch.patch_hash
                or existing.attempt_id != patch.attempt_id
                or existing.base_commit_sha != patch.base_commit_sha
            ):
                raise ValueError("existing smoke result conflicts with the requested patch")
            return existing

        workspace = self._store.resolve_ref(patch.workspace_ref)
        if not workspace.is_dir():
            raise ValueError("patch workspace is unavailable for smoke validation")
        probe = workspace / "fixture_probe.py"
        if not probe.is_file() or probe.is_symlink():
            raise ValueError("controlled fixture probe is missing or unsafe")

        started_at = self._clock()
        stages: list[SmokeStageRecord] = []
        for stage in _STAGES:
            stage_result = self._execute_stage(patch, stage, workspace)
            stages.append(stage_result)
            self.stage_call_counts[stage] = self.stage_call_counts.get(stage, 0) + 1
            if stage_result.status is not ValidationStatus.PASSED:
                break
        finished_at = self._clock()
        status: Literal["PASSED", "FAILED"] = (
            "PASSED"
            if len(stages) == len(_STAGES)
            and all(item.status is ValidationStatus.PASSED for item in stages)
            else "FAILED"
        )
        result = SmokeResultRecord(
            workflow_id=patch.workflow_id,
            attempt_id=patch.attempt_id,
            plan_revision=patch.plan_revision,
            repository_id=patch.repository_id,
            base_commit_sha=patch.base_commit_sha,
            dataset_version=patch.dataset_version,
            patch_hash=patch.patch_hash,
            status=status,
            stages=stages,
            result_ref=result_ref,
            retryable=status == "FAILED",
            capability_boundary="FIXTURE_CONTRACT_PROBE_NO_TORCH",
            started_at=started_at,
            finished_at=finished_at,
        )
        self._store.write_smoke_result(result)
        self.call_count += 1
        return result


__all__ = ["FixtureSmokeRunner"]
