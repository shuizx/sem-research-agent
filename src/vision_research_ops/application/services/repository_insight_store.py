"""Local JSON and Markdown evidence store for one repository insight run."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .repository_insight_models import RepositoryInsightResult, RepositoryInsightTrace

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _safe_component(value: str) -> str:
    if _SAFE_COMPONENT_RE.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError("workflow_id must be a safe local path component")
    return value


def _json_text(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


class LocalRepositoryInsightStore:
    """Persist completed advice and hash-only trace below an ignored workspace."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def run_dir(self, workflow_id: str) -> Path:
        """Return the bounded output directory for one workflow."""
        return self._root / "repository-insight" / _safe_component(workflow_id)

    @staticmethod
    def result_ref(workflow_id: str) -> str:
        return f"repository-insight/{_safe_component(workflow_id)}/repository-insight.json"

    @staticmethod
    def advice_ref(workflow_id: str) -> str:
        return f"repository-insight/{_safe_component(workflow_id)}/advice.json"

    @staticmethod
    def report_ref(workflow_id: str) -> str:
        return f"repository-insight/{_safe_component(workflow_id)}/advice.md"

    @staticmethod
    def trace_ref(workflow_id: str) -> str:
        return f"repository-insight/{_safe_component(workflow_id)}/tool-trace.json"

    def result_path(self, workflow_id: str) -> Path:
        return self.run_dir(workflow_id) / "repository-insight.json"

    def load_result(self, workflow_id: str) -> RepositoryInsightResult:
        """Load a strict completed repository insight result."""
        return RepositoryInsightResult.model_validate_json(
            self.result_path(workflow_id).read_text(encoding="utf-8")
        )

    @staticmethod
    def _markdown(result: RepositoryInsightResult) -> str:
        advice = result.advice
        read_files = "\n".join(f"- `{item.path}`" for item in result.read_files)
        evidence = "\n".join(
            f"- `{item.path}`: {item.observation}" for item in advice.code_evidence
        )
        suggestions = "\n".join(
            f"- {item.area} ({', '.join(item.target_paths)}): {item.recommendation}"
            for item in advice.suggestions
        )
        risks = "\n".join(f"- {item}" for item in advice.risks)
        checks = "\n".join(f"- {item}" for item in advice.items_to_verify)
        limitations = "\n".join(f"- {item}" for item in advice.limitations)
        return (
            "# Public Repository Adaptation Advice\n\n"
            f"- Repository: {result.repository_url}\n"
            f"- Fixed commit: `{result.resolution.commit_sha}`\n"
            f"- License: {result.metadata.license_spdx or 'UNKNOWN'}\n"
            "- Evidence mode: fixed read-only ZIP source snapshot (not a Git clone)\n\n"
            f"## Summary\n\n{advice.repository_summary}\n\n"
            f"Adaptation fit: **{advice.adaptation_fit}**\n\n"
            f"## Files read\n\n{read_files}\n\n"
            f"## Code evidence\n\n{evidence}\n\n"
            f"## Suggestions\n\n{suggestions}\n\n"
            f"## Risks\n\n{risks}\n\n"
            f"## Items to verify\n\n{checks}\n\n"
            f"## Limitations\n\n{limitations}\n\n"
            "No patch, Smoke Test, training, repository execution, or company data was used.\n"
        )

    def write_completed(
        self,
        result: RepositoryInsightResult,
        trace: RepositoryInsightTrace,
    ) -> None:
        """Write the strict artifacts atomically per file after all analysis succeeds."""
        run_dir = self.run_dir(result.workflow_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        payloads = {
            run_dir / "advice.json": _json_text(result.advice),
            run_dir / "advice.md": self._markdown(result),
            run_dir / "tool-trace.json": _json_text(trace),
            run_dir / "repository-insight.json": _json_text(result),
        }
        for path, payload in payloads.items():
            temporary = path.with_suffix(f"{path.suffix}.tmp")
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)


__all__ = ["LocalRepositoryInsightStore"]
