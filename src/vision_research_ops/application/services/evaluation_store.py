"""Write-once local store for canonical evaluation evaluation and report artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .evaluation_models import EvaluationResult, canonical_json_bytes, content_hash
from .evaluation_report import render_evaluation_report

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_RESULT_BYTES = 1_048_576


class EvaluationStoreError(Exception):
    """Sanitized persistence failure with a stable graph-facing code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class PersistedEvaluationArtifact:
    """Small result of one verified write-once operation."""

    ref: str
    content_hash: str
    reused_existing: bool


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


class LocalEvaluationStore:
    """Persist exact evaluation outputs below one ignored var root without overwriting."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        """Return the explicitly configured local var root."""
        return self._root

    @staticmethod
    def evaluation_ref(workflow_id: str) -> str:
        """Return the canonical machine-readable result reference."""
        component = _safe_component(workflow_id, name="workflow_id")
        return f"reports/{component}/evaluation.json"

    @staticmethod
    def report_ref(workflow_id: str) -> str:
        """Return the canonical Markdown report reference."""
        component = _safe_component(workflow_id, name="workflow_id")
        return f"reports/{component}/report.md"

    def resolve_ref(self, relative_ref: str) -> Path:
        """Resolve a canonical ref while proving containment under the var root."""
        safe = _safe_relative_ref(relative_ref)
        root = self._root.resolve()
        path = (root / Path(*safe.split("/"))).resolve()
        if not path.is_relative_to(root):
            raise ValueError("artifact reference escaped the configured var root")
        return path

    def _write_once(self, ref: str, payload: bytes) -> PersistedEvaluationArtifact:
        path = self.resolve_ref(ref)
        expected_hash = content_hash(payload)
        if path.exists():
            try:
                if not path.is_file() or path.stat().st_size > _MAX_RESULT_BYTES:
                    raise OSError
                existing = path.read_bytes()
            except OSError as error:
                raise EvaluationStoreError("EVALUATION_ARTIFACT_INVALID") from error
            if content_hash(existing) != expected_hash or existing != payload:
                raise EvaluationStoreError("EVALUATION_ARTIFACT_CONFLICT")
            return PersistedEvaluationArtifact(
                ref=ref,
                content_hash=expected_hash,
                reused_existing=True,
            )
        if len(payload) > _MAX_RESULT_BYTES:
            raise EvaluationStoreError("EVALUATION_ARTIFACT_INVALID")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f"{path.suffix}.tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)
        except OSError as error:
            raise EvaluationStoreError("EVALUATION_ARTIFACT_WRITE_FAILED") from error
        return PersistedEvaluationArtifact(
            ref=ref,
            content_hash=expected_hash,
            reused_existing=False,
        )

    def write_evaluation(self, result: EvaluationResult) -> PersistedEvaluationArtifact:
        """Write canonical JSON once or verify exact existing bytes and hash."""
        validated = EvaluationResult.model_validate(result.model_dump(mode="json"))
        ref = self.evaluation_ref(validated.workflow_id)
        if validated.evaluation_ref != ref:
            raise EvaluationStoreError("EVALUATION_ARTIFACT_INVALID")
        return self._write_once(ref, canonical_json_bytes(validated.model_dump(mode="json")))

    def load_evaluation(self, workflow_id: str) -> EvaluationResult:
        """Load a bounded, strictly validated canonical evaluation result."""
        ref = self.evaluation_ref(workflow_id)
        path = self.resolve_ref(ref)
        try:
            if not path.is_file() or path.stat().st_size > _MAX_RESULT_BYTES:
                raise OSError
            payload = path.read_bytes()
            result = EvaluationResult.model_validate_json(payload)
        except (OSError, ValueError) as error:
            raise EvaluationStoreError("EVALUATION_ARTIFACT_INVALID") from error
        if canonical_json_bytes(result.model_dump(mode="json")) != payload:
            raise EvaluationStoreError("EVALUATION_ARTIFACT_INVALID")
        return result

    def write_report(self, result: EvaluationResult) -> PersistedEvaluationArtifact:
        """Render from the validated model internally, then write/verify once."""
        validated = EvaluationResult.model_validate(result.model_dump(mode="json"))
        ref = self.report_ref(validated.workflow_id)
        if validated.report_ref != ref:
            raise EvaluationStoreError("EVALUATION_ARTIFACT_INVALID")
        payload = render_evaluation_report(validated).encode("utf-8")
        return self._write_once(ref, payload)


__all__ = [
    "EvaluationStoreError",
    "LocalEvaluationStore",
    "PersistedEvaluationArtifact",
]
