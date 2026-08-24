"""Shared, provider-neutral models for SEM Research Agent ports.

These models deliberately describe only the application-to-adapter boundary.
They do not contain provider clients, endpoints, credentials, host paths, or
free-form shell commands.  Concrete adapters and the later settings layer own
those concerns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vision_research_ops.domain import (
    ArtifactKind,
    ArtifactRef,
    CommandSpec,
    ContentHash,
    FiniteFloat,
    GitCommitSha,
    JsonObject,
    NonBlankStr,
    NonNegativeInt,
    OpaqueId,
    PatchOperation,
    PositiveInt,
    QuerySpec,
    Reason,
    ResourceRequest,
    RiskFinding,
    RunStatus,
    StrictBoolean,
    StrictInteger,
    StructuredFailure,
    UTCDateTime,
    ValidationStage,
)


class PortModel(BaseModel):
    """Strict base model for typed, serializable port boundary values."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class OperationContext(PortModel):
    """Explicit audit, idempotency, deadline, and sensitivity context for a call."""

    schema_version: Literal["1"]
    correlation_id: OpaqueId
    workflow_id: OpaqueId
    actor_id: OpaqueId
    idempotency_key: NonBlankStr | None = None
    deadline_at: UTCDateTime | None = None
    sensitivity: Literal["PUBLIC", "INTERNAL", "RESTRICTED"]

    def deadline_exceeded(self, *, now: datetime) -> bool:
        """Return whether the explicit deadline has elapsed at the supplied time."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be an aware datetime")
        if self.deadline_at is None:
            return False
        return now.astimezone(UTC) >= self.deadline_at


class PaperQuery(PortModel):
    """Provider-neutral paper search request with an explicit bounded page size."""

    schema_version: Literal["1"]
    query_id: OpaqueId
    query_spec: QuerySpec
    page_size: PositiveInt


class ExternalPaperId(PortModel):
    """An opaque paper identifier scoped to one provider."""

    schema_version: Literal["1"]
    provider_name: NonBlankStr
    value: NonBlankStr


class RawPaperRecord(PortModel):
    """Provider record retained before application-level paper normalization."""

    schema_version: Literal["1"]
    provider_name: NonBlankStr
    provider_record_id: NonBlankStr
    external_ids: list[ExternalPaperId] = Field(default_factory=list)
    raw_fields: JsonObject
    retrieved_at: UTCDateTime


class PaperSearchPage(PortModel):
    """One provider page, including cursor and retrieval provenance."""

    schema_version: Literal["1"]
    provider_name: NonBlankStr
    records: list[RawPaperRecord] = Field(default_factory=list)
    next_cursor: NonBlankStr | None = None
    provider_request_id: NonBlankStr
    retrieved_at: UTCDateTime


class RepositoryResolution(PortModel):
    """A repository URL resolved to a complete immutable commit identifier."""

    schema_version: Literal["1"]
    canonical_url: NonBlankStr
    provider: Literal["GITHUB", "LOCAL_FIXTURE"]
    owner: NonBlankStr
    name: NonBlankStr
    commit_sha: GitCommitSha


class RepositoryMetadata(PortModel):
    """Read-only repository metadata collected without executing repository code."""

    schema_version: Literal["1"]
    repository: RepositoryResolution
    license_spdx: NonBlankStr | None = None
    languages: dict[NonBlankStr, NonNegativeInt] = Field(default_factory=dict)
    default_branch: NonBlankStr | None = None
    raw_fields: JsonObject = Field(default_factory=dict)


class RepositoryPolicy(PortModel):
    """Versioned policy reference supplied to static repository analysis."""

    schema_version: Literal["1"]
    policy_id: NonBlankStr
    policy_version: NonBlankStr


class RepositoryFileSummary(PortModel):
    """Small static file-tree summary without exposing a local workspace path."""

    schema_version: Literal["1"]
    path: NonBlankStr
    size_bytes: NonNegativeInt
    kind: Literal["FILE", "DIRECTORY", "SYMLINK", "OTHER"]


class RepositoryAnalysis(PortModel):
    """Static inspection result; it contains evidence, never executed code output."""

    schema_version: Literal["1"]
    repository: RepositoryResolution
    policy: RepositoryPolicy
    file_tree_summary: list[RepositoryFileSummary] = Field(default_factory=list)
    dependency_files: list[NonBlankStr] = Field(default_factory=list)
    framework_evidence: list[NonBlankStr] = Field(default_factory=list)
    entrypoint_candidates: list[NonBlankStr] = Field(default_factory=list)
    data_loader_candidates: list[NonBlankStr] = Field(default_factory=list)
    command_candidates: list[CommandSpec] = Field(default_factory=list)
    license_spdx: NonBlankStr | None = None
    dangerous_patterns: list[RiskFinding] = Field(default_factory=list)
    supported: StrictBoolean
    support_reasons: list[Reason] = Field(default_factory=list)


class StructuredGenerationRequest[TStructured: BaseModel](PortModel):
    """Sanitized request for a Pydantic-validated LLM proposal."""

    schema_version: Literal["1"]
    task_name: NonBlankStr
    prompt_template_id: NonBlankStr
    prompt_version: NonBlankStr
    response_schema: type[TStructured]
    facts: JsonObject = Field(default_factory=dict)
    artifact_excerpts: list[ArtifactRef] = Field(default_factory=list)
    model_parameters: JsonObject = Field(default_factory=dict)
    budget_class: NonBlankStr


class GenerationUsage(PortModel):
    """Provider-neutral, bounded generation usage accounting."""

    schema_version: Literal["1"]
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    total_tokens: NonNegativeInt


class StructuredGenerationResult[TStructured: BaseModel](PortModel):
    """Validated structured generation with reproducibility-oriented provenance."""

    schema_version: Literal["1"]
    value: TStructured
    provider_id: NonBlankStr
    model_id: NonBlankStr
    usage: GenerationUsage
    latency_ms: NonNegativeInt
    prompt_hash: ContentHash
    output_hash: ContentHash
    finish_reason: NonBlankStr


class ArtifactDescriptor(PortModel):
    """Immutable artifact metadata supplied before bytes are finalized."""

    schema_version: Literal["1"]
    artifact_id: OpaqueId
    kind: ArtifactKind
    media_type: NonBlankStr
    producer: NonBlankStr
    sensitivity: Literal["PUBLIC", "INTERNAL", "RESTRICTED"]
    metadata: JsonObject = Field(default_factory=dict)


class DownloadGrant(PortModel):
    """Credential-free, time-bounded download reference issued after authorization."""

    schema_version: Literal["1"]
    grant_id: OpaqueId
    artifact_id: OpaqueId
    uri: NonBlankStr
    expires_at: UTCDateTime


class DatasetMountSpec(PortModel):
    """Opaque, read-only mount handle available only to trusted execution services."""

    schema_version: Literal["1"]
    dataset_id: OpaqueId
    version: NonBlankStr
    mount_ref: NonBlankStr
    read_only: Literal[True]


class WorkspaceRef(PortModel):
    """Opaque workspace handle that intentionally omits an arbitrary filesystem path."""

    schema_version: Literal["1"]
    workspace_id: OpaqueId
    repository_id: OpaqueId
    base_commit_sha: GitCommitSha
    operation_id: OpaqueId


class PatchDocument(PortModel):
    """Structured patch proposal applied only through the patch workspace port."""

    schema_version: Literal["1"]
    patch_id: OpaqueId
    operations: list[PatchOperation] = Field(default_factory=list)


class PatchResult(PortModel):
    """Result of applying a structured patch to an opaque workspace."""

    schema_version: Literal["1"]
    workspace: WorkspaceRef
    patch_artifact: ArtifactRef
    patch_hash: ContentHash


class ValidationStageSpec(PortModel):
    """One bounded, policy-approved validation stage without a shell command string."""

    schema_version: Literal["1"]
    validation_id: OpaqueId
    attempt_id: OpaqueId
    stage: ValidationStage
    command: CommandSpec
    timeout_seconds: PositiveInt
    environment_digest: ContentHash


class RunManifest(PortModel):
    """Minimal immutable run manifest used by executors and trackers."""

    schema_version: Literal["1"]
    run_id: OpaqueId
    experiment_id: OpaqueId
    role: Literal["BASELINE", "CANDIDATE"]
    seed: StrictInteger
    repository_commit_sha: GitCommitSha
    patch_hash: ContentHash | None = None
    environment_digest: ContentHash
    dataset_id: OpaqueId
    dataset_version: NonBlankStr
    dataset_hash: ContentHash
    split_manifest_hash: ContentHash
    preprocessing_hash: ContentHash
    config_hash: ContentHash
    argv: list[NonBlankStr] = Field(default_factory=list)
    resources: ResourceRequest
    runtime_versions: JsonObject = Field(default_factory=dict)


class FrozenRunSpec(PortModel):
    """Executor input composed from a frozen run record and structured command."""

    schema_version: Literal["1"]
    run_id: OpaqueId
    experiment_id: OpaqueId
    idempotency_key: NonBlankStr
    command: CommandSpec
    resources: ResourceRequest
    manifest: RunManifest


class ExternalRunStatus(PortModel):
    """Normalized executor status with provider-native details kept as descriptive JSON."""

    schema_version: Literal["1"]
    external_job_id: OpaqueId
    status: RunStatus
    observed_at: UTCDateTime
    raw_status: NonBlankStr
    metadata: JsonObject = Field(default_factory=dict)


class SubmissionResult(PortModel):
    """Idempotent submission acknowledgement returned by an experiment executor."""

    schema_version: Literal["1"]
    external_job_id: OpaqueId
    status: ExternalRunStatus
    submitted_at: UTCDateTime


class CancellationResult(PortModel):
    """Result of a bounded cancellation request for one external job."""

    schema_version: Literal["1"]
    external_job_id: OpaqueId
    cancelled: StrictBoolean
    status: ExternalRunStatus


class MetricPoint(PortModel):
    """One finite structured metric point; training prose is not accepted as evidence."""

    schema_version: Literal["1"]
    name: NonBlankStr
    value: FiniteFloat
    split: NonBlankStr
    step: StrictInteger | None = None
    recorded_at: UTCDateTime


class TrackerRunRef(PortModel):
    """Provider-neutral reference to a tracker run without embedded credentials."""

    schema_version: Literal["1"]
    tracker_run_id: OpaqueId
    run_id: OpaqueId
    uri: NonBlankStr


class AuditEvent(PortModel):
    """Small append-only audit event for the future UnitOfWork persistence boundary."""

    schema_version: Literal["1"]
    event_id: OpaqueId
    event_type: NonBlankStr
    occurred_at: UTCDateTime
    correlation_id: OpaqueId
    workflow_id: OpaqueId
    subject_type: NonBlankStr
    subject_id: OpaqueId
    subject_revision: PositiveInt
    payload: JsonObject = Field(default_factory=dict)
    failure: StructuredFailure | None = None


__all__ = [
    "ArtifactDescriptor",
    "AuditEvent",
    "CancellationResult",
    "DatasetMountSpec",
    "DownloadGrant",
    "ExternalPaperId",
    "ExternalRunStatus",
    "FrozenRunSpec",
    "GenerationUsage",
    "MetricPoint",
    "OperationContext",
    "PaperQuery",
    "PaperSearchPage",
    "PatchDocument",
    "PatchResult",
    "PortModel",
    "RawPaperRecord",
    "RepositoryAnalysis",
    "RepositoryFileSummary",
    "RepositoryMetadata",
    "RepositoryPolicy",
    "RepositoryResolution",
    "RunManifest",
    "StructuredGenerationRequest",
    "StructuredGenerationResult",
    "SubmissionResult",
    "TrackerRunRef",
    "ValidationStageSpec",
    "WorkspaceRef",
]
