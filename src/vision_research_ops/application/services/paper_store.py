"""Small local JSON persistence used by the single-machine pipeline sample."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from vision_research_ops.domain import QuerySpec
from vision_research_ops.ports import RawPaperRecord

from .paper_models import (
    ResearchPaper,
    ResearchPaperAssessment,
    ResearchResult,
    ResearchWatermark,
    RetrievalWindow,
)

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _safe_component(value: str, *, name: str) -> str:
    if _SAFE_COMPONENT_RE.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"{name} must be a safe local path component")
    return value


@dataclass(slots=True)
class ResearchSession:
    """Runtime-only intermediate values intentionally excluded from checkpoints."""

    raw_records: list[RawPaperRecord] = field(default_factory=list)
    papers: list[ResearchPaper] = field(default_factory=list)
    assessments: list[ResearchPaperAssessment] = field(default_factory=list)
    retrieval_window: RetrievalWindow | None = None
    query_spec: QuerySpec | None = None
    watermark_before: datetime | None = None


class LocalResearchStore:
    """Atomically store canonical research JSON and one UTC watermark."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        """Return the configured local output root for CLI display only."""
        return self._root

    @property
    def watermark_path(self) -> Path:
        """Return the local watermark path; it is never placed in prompts or state."""
        return self._root / "watermark.json"

    def result_path(self, workflow_id: str) -> Path:
        """Resolve a validated workflow directory below the configured root."""
        component = _safe_component(workflow_id, name="workflow_id")
        return self._root / component / "papers.json"

    @staticmethod
    def result_ref(workflow_id: str) -> str:
        """Return the small repository-relative artifact reference stored in state."""
        component = _safe_component(workflow_id, name="workflow_id")
        return f"research/{component}/papers.json"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(f"{encoded}\n", encoding="utf-8")
        temporary.replace(path)

    def load_watermark(self) -> ResearchWatermark | None:
        """Read the last completed timestamp, or return None for a first run."""
        path = self.watermark_path
        if not path.exists():
            return None
        return ResearchWatermark.model_validate_json(path.read_text(encoding="utf-8"))

    def write_watermark(self, watermark: ResearchWatermark) -> None:
        """Atomically advance the successful-run watermark."""
        self._write_json(self.watermark_path, watermark.model_dump(mode="json"))

    def load_result(self, workflow_id: str) -> ResearchResult:
        """Load one previously written workflow result for gate resume/finalization."""
        path = self.result_path(workflow_id)
        return ResearchResult.model_validate_json(path.read_text(encoding="utf-8"))

    def write_result(self, result: ResearchResult) -> str:
        """Atomically persist a result and return its small relative reference."""
        self._write_json(
            self.result_path(result.workflow_id),
            result.model_dump(mode="json"),
        )
        return self.result_ref(result.workflow_id)


__all__ = ["LocalResearchStore", "ResearchSession"]
