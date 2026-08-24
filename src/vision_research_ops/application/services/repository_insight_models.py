"""Strict application records for bounded public-repository code insight."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from vision_research_ops.domain import ArtifactRef, ContentHash, NonBlankStr, PositiveInt
from vision_research_ops.ports import RepositoryAnalysis, RepositoryMetadata, RepositoryResolution

RepositoryInsightToolName = Literal[
    "inspect_repository_summary",
    "inspect_target_profile",
    "read_repository_file",
    "submit_adaptation_advice",
]
AdaptationArea = Literal[
    "DATA_LOADING",
    "INPUT_CHANNELS",
    "LABELS",
    "MODEL_HEAD",
    "METRICS",
    "CONFIGURATION",
]

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"}
_COMMAND_MARKERS = (
    "```",
    "pip install",
    "conda install",
    "python -m",
    "python train",
    "bash ",
    "powershell ",
    "cmd /c",
    "subprocess",
    "os.system",
)
_CLAIM_MARKERS = (
    "company internal",
    "internal company",
    "公司内部数据",
    "真实公司数据",
    "guaranteed improvement",
    "guarantee improvement",
    "必然提升",
    "保证提升",
    "预计提升",
)
_LOCAL_PATH_RE = re.compile(r"(?:[a-zA-Z]:[\\/]|(?:^|\s)/(?:home|users|data|tmp)/)")


def _canonical_repository_path(value: str) -> str:
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or "%" in value
        or value.startswith("/")
        or ":" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("repository paths must be canonical POSIX relative paths")
    suffix = "." + value.rsplit(".", maxsplit=1)[-1].casefold() if "." in value else ""
    if suffix not in _ALLOWED_SUFFIXES:
        raise ValueError("repository path suffix is outside the text allowlist")
    return value


type RepositoryTextPath = Annotated[str, AfterValidator(_canonical_repository_path)]


class RepositoryInsightModel(BaseModel):
    """Strict, finite, extra-forbid base for repository insight workflow local records."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class RepositorySourceEntry(RepositoryInsightModel):
    """One readable text member in a fixed ZIP snapshot."""

    schema_version: Literal["1"] = "1"
    path: RepositoryTextPath
    size_bytes: int = Field(ge=0)


class RepositorySourceIndex(RepositoryInsightModel):
    """Bounded canonical path index; it contains no repository source text."""

    schema_version: Literal["1"] = "1"
    files: list[RepositorySourceEntry] = Field(max_length=500)

    @model_validator(mode="after")
    def _paths_are_unique(self) -> RepositorySourceIndex:
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("repository source index paths must be unique")
        if paths != sorted(paths, key=str.casefold):
            raise ValueError("repository source index paths must be deterministically sorted")
        return self


class RepositorySourceRead(RepositoryInsightModel):
    """Ephemeral bounded source text returned only to the internal LLM ToolNode."""

    schema_version: Literal["1"] = "1"
    path: RepositoryTextPath
    content: str
    returned_bytes: int = Field(ge=0, le=8 * 1024)
    original_bytes: int = Field(ge=0)
    truncated: bool
    content_hash: ContentHash


class RepositoryCodeEvidence(RepositoryInsightModel):
    """One evidence statement grounded in an actually read source path."""

    schema_version: Literal["1"] = "1"
    path: RepositoryTextPath
    observation: NonBlankStr


class RepositoryAdaptationSuggestion(RepositoryInsightModel):
    """A conceptual, non-executable adaptation suggestion."""

    schema_version: Literal["1"] = "1"
    area: AdaptationArea
    target_paths: list[RepositoryTextPath] = Field(min_length=1, max_length=3)
    recommendation: NonBlankStr
    rationale: NonBlankStr


class RepositoryAdaptationAdvice(RepositoryInsightModel):
    """Strict final LLM advice for an abstract public SEM-classification target."""

    schema_version: Literal["1"] = "1"
    repository_summary: NonBlankStr
    adaptation_fit: Literal["HIGH", "MEDIUM", "LOW", "INSUFFICIENT_EVIDENCE"]
    code_evidence: list[RepositoryCodeEvidence] = Field(min_length=1, max_length=6)
    suggestions: list[RepositoryAdaptationSuggestion] = Field(min_length=1, max_length=6)
    risks: list[NonBlankStr] = Field(min_length=1, max_length=6)
    items_to_verify: list[NonBlankStr] = Field(min_length=1, max_length=6)
    limitations: list[NonBlankStr] = Field(min_length=2, max_length=6)
    patch_generated: Literal[False] = False
    code_executed: Literal[False] = False
    training_run: Literal[False] = False
    company_data_used: Literal[False] = False
    expected_improvement_claimed: Literal[False] = False

    @model_validator(mode="after")
    def _reject_commands_and_unsupported_claims(self) -> RepositoryAdaptationAdvice:
        rendered = self.model_dump_json().casefold()
        if any(marker in rendered for marker in _COMMAND_MARKERS):
            raise ValueError("adaptation advice cannot contain shell or execution commands")
        if any(marker in rendered for marker in _CLAIM_MARKERS):
            raise ValueError("adaptation advice cannot claim company data or guaranteed gains")
        if _LOCAL_PATH_RE.search(rendered) is not None:
            raise ValueError("adaptation advice cannot expose absolute local paths")
        if "dashscope_api_key" in rendered or "bearer " in rendered:
            raise ValueError("adaptation advice cannot contain credential material")
        return self


class RepositoryInsightGeneration(RepositoryInsightModel):
    """Sanitized provenance for the tool-calling planning result."""

    schema_version: Literal["1"] = "1"
    planner_kind: Literal["SCRIPTED_TOOL_CALLING", "DASHSCOPE_TOOL_CALLING"]
    provider_id: NonBlankStr
    model_id: NonBlankStr
    prompt_version: Literal["1.0.0"] = "1.0.0"
    prompt_hash: ContentHash
    output_hash: ContentHash
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class RepositoryReadRecord(RepositoryInsightModel):
    """Persisted hash-only evidence for a source file supplied to the LLM."""

    schema_version: Literal["1"] = "1"
    path: RepositoryTextPath
    returned_bytes: int = Field(ge=0, le=8 * 1024)
    original_bytes: int = Field(ge=0)
    truncated: bool
    content_hash: ContentHash


class RepositoryInsightToolEvent(RepositoryInsightModel):
    """Hash-only record for one allowlisted ToolNode execution."""

    schema_version: Literal["1"] = "1"
    call_index: PositiveInt
    tool_name: RepositoryInsightToolName
    arguments_hash: ContentHash
    output_hash: ContentHash


class RepositoryInsightTrace(RepositoryInsightModel):
    """Small trace proving the bounded real ToolNode loop and actual reads."""

    schema_version: Literal["1"] = "1"
    events: list[RepositoryInsightToolEvent] = Field(min_length=4, max_length=10)
    read_files: list[RepositoryReadRecord] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def _trace_is_complete(self) -> RepositoryInsightTrace:
        if [event.call_index for event in self.events] != list(range(1, len(self.events) + 1)):
            raise ValueError("repository insight trace indices must be dense and ordered")
        names = {event.tool_name for event in self.events}
        required = {
            "inspect_repository_summary",
            "inspect_target_profile",
            "read_repository_file",
            "submit_adaptation_advice",
        }
        if not required.issubset(names):
            raise ValueError("repository insight trace must contain all four tools")
        paths = [item.path for item in self.read_files]
        if len(paths) != len(set(paths)):
            raise ValueError("repository insight read records must be unique")
        if sum(item.returned_bytes for item in self.read_files) > 48 * 1024:
            raise ValueError("repository insight total read budget exceeded")
        return self


class RepositoryStructureSummary(RepositoryInsightModel):
    """Compact deterministic profile displayed and supplied to the LLM."""

    schema_version: Literal["1"] = "1"
    static_supported: bool
    entrypoint_candidates: list[RepositoryTextPath] = Field(default_factory=list)
    data_loader_candidates: list[RepositoryTextPath] = Field(default_factory=list)
    dependency_files: list[RepositoryTextPath] = Field(default_factory=list)
    configuration_files: list[RepositoryTextPath] = Field(default_factory=list)
    framework_evidence: list[NonBlankStr] = Field(default_factory=list, max_length=40)
    risk_codes: list[NonBlankStr] = Field(default_factory=list, max_length=20)


class RepositoryInsightResult(RepositoryInsightModel):
    """Canonical completed local evidence for one approved public snapshot analysis."""

    schema_version: Literal["1"] = "1"
    workflow_id: NonBlankStr
    repository_url: NonBlankStr
    resolution: RepositoryResolution
    metadata: RepositoryMetadata
    snapshot: ArtifactRef
    source_index_count: int = Field(ge=0, le=500)
    structure: RepositoryStructureSummary
    advice: RepositoryAdaptationAdvice
    generation: RepositoryInsightGeneration
    read_files: list[RepositoryReadRecord] = Field(min_length=1, max_length=6)
    advice_ref: NonBlankStr
    report_ref: NonBlankStr
    trace_ref: NonBlankStr
    result_ref: NonBlankStr
    source_snapshot_only: Literal[True] = True
    git_clone_performed: Literal[False] = False
    patch_generated: Literal[False] = False
    smoke_test_run: Literal[False] = False
    training_run: Literal[False] = False
    company_data_used: Literal[False] = False

    @field_validator("advice_ref", "report_ref", "trace_ref", "result_ref")
    @classmethod
    def _relative_refs(cls, value: str) -> str:
        if (
            not value
            or value != value.strip()
            or "\\" in value
            or value.startswith("/")
            or ":" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError("result references must be canonical relative paths")
        return value

    @model_validator(mode="after")
    def _snapshot_and_advice_are_consistent(self) -> RepositoryInsightResult:
        if _COMMIT_RE.fullmatch(self.resolution.commit_sha) is None:
            raise ValueError("repository insight requires a full lowercase commit SHA")
        if self.repository_url != self.resolution.canonical_url:
            raise ValueError("result URL must match the resolved canonical repository")
        read_paths = {item.path for item in self.read_files}
        referenced = {item.path for item in self.advice.code_evidence}
        referenced.update(
            path for suggestion in self.advice.suggestions for path in suggestion.target_paths
        )
        if not referenced.issubset(read_paths):
            raise ValueError("all advice code paths must have been read")
        return self


class RepositoryInsightPlannerOutput(RepositoryInsightModel):
    """Validated output from the internal code-reading LangGraph."""

    schema_version: Literal["1"] = "1"
    advice: RepositoryAdaptationAdvice
    generation: RepositoryInsightGeneration
    trace: RepositoryInsightTrace


def structure_summary(analysis: RepositoryAnalysis) -> RepositoryStructureSummary:
    """Project the existing deterministic analysis into the small insight contract."""
    dependency_set = set(analysis.dependency_files)
    configurations = sorted(
        item.path
        for item in analysis.file_tree_summary
        if item.path.rsplit(".", maxsplit=1)[-1].casefold() in {"json", "toml", "yaml", "yml"}
        and item.path not in dependency_set
    )
    return RepositoryStructureSummary(
        static_supported=analysis.supported,
        entrypoint_candidates=analysis.entrypoint_candidates,
        data_loader_candidates=analysis.data_loader_candidates,
        dependency_files=analysis.dependency_files,
        configuration_files=configurations,
        framework_evidence=analysis.framework_evidence[:40],
        risk_codes=[item.rule_id for item in analysis.dangerous_patterns[:20]],
    )


__all__ = [
    "AdaptationArea",
    "RepositoryAdaptationAdvice",
    "RepositoryAdaptationSuggestion",
    "RepositoryCodeEvidence",
    "RepositoryInsightGeneration",
    "RepositoryInsightModel",
    "RepositoryInsightPlannerOutput",
    "RepositoryInsightResult",
    "RepositoryInsightToolEvent",
    "RepositoryInsightToolName",
    "RepositoryInsightTrace",
    "RepositoryReadRecord",
    "RepositorySourceEntry",
    "RepositorySourceIndex",
    "RepositorySourceRead",
    "RepositoryStructureSummary",
    "RepositoryTextPath",
    "structure_summary",
]
