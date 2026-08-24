"""Strict application records for the pipeline Repository Agent."""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vision_research_ops.domain import (
    ArtifactRef,
    CodeLinkEvidence,
    NonBlankStr,
    OpaqueId,
    PositiveInt,
    Reason,
    RepositorySnapshot,
    RiskFinding,
    StrictBoolean,
    StructuredFailure,
    UTCDateTime,
)
from vision_research_ops.ports import (
    RepositoryAnalysis,
    RepositoryFileSummary,
    RepositoryMetadata,
    RepositoryResolution,
)

_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


class RepositoryModel(BaseModel):
    """Strict JSON-safe base for repository workflow application-owned records."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class GitHubRepositoryLocator(RepositoryModel):
    """Canonical public GitHub repository coordinates."""

    schema_version: Literal["1"] = "1"
    canonical_url: NonBlankStr
    owner: NonBlankStr
    name: NonBlankStr


def normalize_github_repository_url(value: str) -> GitHubRepositoryLocator:
    """Accept only a credential-free HTTPS GitHub owner/repository URL."""
    if not isinstance(value, str) or value != value.strip() or "%" in value:
        raise ValueError("repository URL must be an unencoded canonical string")
    parsed = urlsplit(value)
    if (
        parsed.scheme.casefold() != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold() != "github.com"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("repository URL must be a credential-free HTTPS GitHub URL")
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) != 2:
        raise ValueError("repository URL must identify exactly one owner and repository")
    owner, name = segments
    if name.casefold().endswith(".git"):
        name = name[:-4]
    if (
        owner in {".", ".."}
        or name in {".", ".."}
        or _OWNER_RE.fullmatch(owner) is None
        or _REPOSITORY_RE.fullmatch(name) is None
    ):
        raise ValueError("repository owner or name is invalid")
    owner = owner.casefold()
    name = name.casefold()
    return GitHubRepositoryLocator(
        canonical_url=f"https://github.com/{owner}/{name}",
        owner=owner,
        name=name,
    )


class RepositoryProfile(RepositoryModel):
    """Compact explainable static profile used by the later Adaptation Agent."""

    schema_version: Literal["1"] = "1"
    profile_id: OpaqueId
    paper_id: OpaqueId
    repository_snapshot: RepositorySnapshot
    code_link_evidence: CodeLinkEvidence
    structure_type: Literal[
        "PLAIN_PYTORCH",
        "TORCHVISION_TIMM",
        "UNSUPPORTED",
    ]
    entrypoint_candidates: list[NonBlankStr] = Field(default_factory=list)
    data_loader_candidates: list[NonBlankStr] = Field(default_factory=list)
    configuration_files: list[NonBlankStr] = Field(default_factory=list)
    dependency_files: list[NonBlankStr] = Field(default_factory=list)
    model_head_evidence: list[NonBlankStr] = Field(default_factory=list)
    framework_evidence: list[NonBlankStr] = Field(default_factory=list)
    file_tree_summary: list[RepositoryFileSummary] = Field(default_factory=list)
    risk_findings: list[RiskFinding] = Field(default_factory=list)
    supported: StrictBoolean
    support_reasons: list[Reason] = Field(default_factory=list)

    @model_validator(mode="after")
    def _association_targets_profiled_repository(self) -> RepositoryProfile:
        if self.code_link_evidence.repository_url != self.repository_snapshot.canonical_url:
            raise ValueError("code-link evidence must target the profiled repository")
        return self


class RepositoryResult(RepositoryModel):
    """Canonical local JSON output for one Repository Agent workflow."""

    schema_version: Literal["1"] = "1"
    workflow_id: OpaqueId
    request_id: OpaqueId
    research_workflow_id: OpaqueId
    paper_id: OpaqueId
    requested_repository_url: NonBlankStr
    approved_repository_url: NonBlankStr | None = None
    code_link_evidence: CodeLinkEvidence
    status: Literal[
        "AWAITING_APPROVAL",
        "INGESTING",
        "COMPLETED",
        "UNSUPPORTED",
        "REJECTED",
        "FAILED",
    ]
    gate_id: OpaqueId
    gate_revision: PositiveInt
    resolution: RepositoryResolution | None = None
    metadata: RepositoryMetadata | None = None
    archive: ArtifactRef | None = None
    analysis: RepositoryAnalysis | None = None
    profile: RepositoryProfile | None = None
    failure: StructuredFailure | None = None
    created_at: UTCDateTime
    updated_at: UTCDateTime

    @model_validator(mode="after")
    def _completed_records_are_consistent(self) -> RepositoryResult:
        if self.status in {"COMPLETED", "UNSUPPORTED"} and any(
            value is None
            for value in (
                self.approved_repository_url,
                self.resolution,
                self.metadata,
                self.archive,
                self.analysis,
                self.profile,
            )
        ):
            raise ValueError("completed repository results require all static evidence")
        if self.status == "FAILED" and self.failure is None:
            raise ValueError("failed repository results require a structured failure")
        if self.status != "FAILED" and self.failure is not None:
            raise ValueError("only failed repository results may contain a failure")
        if self.status == "COMPLETED" and self.profile is not None and not self.profile.supported:
            raise ValueError("COMPLETED repository results require a supported profile")
        if self.status == "UNSUPPORTED" and self.profile is not None and self.profile.supported:
            raise ValueError("UNSUPPORTED repository results require an unsupported profile")
        return self


__all__ = [
    "GitHubRepositoryLocator",
    "RepositoryModel",
    "RepositoryProfile",
    "RepositoryResult",
    "normalize_github_repository_url",
]
