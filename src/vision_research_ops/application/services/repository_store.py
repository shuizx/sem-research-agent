"""Small local JSON store for the single-user Repository Agent sample."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .repository_models import RepositoryResult

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _safe_component(value: str, *, name: str) -> str:
    if _SAFE_COMPONENT_RE.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"{name} must be a safe local path component")
    return value


class LocalRepositoryStore:
    """Atomically persist one compact repository profile JSON per workflow."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        """Return the configured output root for explicit CLI or test display."""
        return self._root

    def result_path(self, workflow_id: str) -> Path:
        """Resolve the canonical profile path below the configured root."""
        component = _safe_component(workflow_id, name="workflow_id")
        return self._root / component / "repository-profile.json"

    @staticmethod
    def result_ref(workflow_id: str) -> str:
        """Return the small relative reference stored in LangGraph state."""
        component = _safe_component(workflow_id, name="workflow_id")
        return f"repositories/{component}/repository-profile.json"

    def load_result(self, workflow_id: str) -> RepositoryResult:
        """Load a persisted pending or completed Repository Agent result."""
        path = self.result_path(workflow_id)
        return RepositoryResult.model_validate_json(path.read_text(encoding="utf-8"))

    def write_result(self, result: RepositoryResult) -> str:
        """Atomically write a validated result and return its relative reference."""
        path = self.result_path(result.workflow_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(f"{encoded}\n", encoding="utf-8")
        temporary.replace(path)
        return self.result_ref(result.workflow_id)


__all__ = ["LocalRepositoryStore"]
