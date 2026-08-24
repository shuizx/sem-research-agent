"""Strict, versioned foundation domain models for SEM Research Agent.

Each public model rejects unknown fields and coercive scalar inputs.  The
explicit date and datetime aliases remain responsible for their documented JSON
wire representations; all other scalars use strict aliases from ``errors``.
"""

import unicodedata
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .enums import (
    ApprovalDecision,
    ArtifactKind,
    CodeLinkConfidence,
    EvaluationConclusion,
    GateKind,
    LicenseStatus,
    NetworkPolicy,
    PatchOperationType,
    RunStatus,
    SeverityLevel,
    SplitStrategy,
    TaskType,
    ValidationStage,
    ValidationStatus,
    WorkflowStatus,
)
from .errors import (
    ContentHash,
    FiniteFloat,
    GitCommitSha,
    HumanText,
    ISODate,
    JsonObject,
    JsonValue,
    NetworkAllowlistRef,
    NonBlankStr,
    NonNegativeFiniteFloat,
    NonNegativeInt,
    OpaqueId,
    OpenUnitInterval,
    PositiveFiniteFloat,
    PositiveInt,
    ReasonCode,
    StrictBoolean,
    StructuredFailure,
    UnitInterval,
    UTCDateTime,
)

_SHELL_OR_WRAPPER_BASENAMES = {
    "bash",
    "busybox",
    "cmd",
    "csh",
    "dash",
    "doas",
    "env",
    "fish",
    "ksh",
    "powershell",
    "pwsh",
    "sh",
    "su",
    "sudo",
    "tcsh",
    "wsl",
    "xargs",
    "zsh",
}

_WINDOWS_DEVICE_BASENAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _validate_repository_relative_path(value: str) -> str:
    """Validate an unencoded canonical POSIX-style repository-relative path."""
    if value == ".":
        return value
    if value == "" or value != value.strip():
        raise ValueError("repository-relative path must not be empty or trim-ambiguous")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError("repository-relative path must not contain control characters")
    if "%" in value:
        raise ValueError("repository-relative path must not contain percent encoding")
    if "\\" in value or ":" in value:
        raise ValueError("repository-relative path must not use backslashes or colons")
    if value.startswith("/") or value.endswith("/") or "//" in value:
        raise ValueError("repository-relative path must use non-empty relative POSIX segments")

    segments = value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError("repository-relative path must not contain empty, '.' or '..' segments")
    if any(segment != segment.strip() for segment in segments):
        raise ValueError("repository-relative path segments must not be trim-ambiguous")
    if any(
        segment.split(".", maxsplit=1)[0].casefold() in _WINDOWS_DEVICE_BASENAMES
        for segment in segments
    ):
        raise ValueError("repository-relative path must not contain a Windows device name")
    return value


def _validate_non_shell_executable(value: str) -> str:
    """Reject known shells and command wrappers in identifier or path form."""
    candidate = value.strip().strip("\"'").replace("\\", "/").rstrip("/")
    basename = candidate.rsplit("/", maxsplit=1)[-1].casefold()
    if basename.endswith(".exe"):
        basename = basename.removesuffix(".exe")
    if basename in _SHELL_OR_WRAPPER_BASENAMES:
        raise ValueError("shell and command-wrapper executables are not allowed")
    return value


def _deduplicate_preserving_order(values: list[str]) -> list[str]:
    """Return the first occurrence of each already-validated string value."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _validate_image_channel_count(value: int) -> int:
    """Accept only the two MVP channel counts after strict integer validation."""
    if value not in (1, 3):
        raise ValueError("image classification datasets must declare either 1 or 3 channels")
    return value


# Keep these constraints directly on scalar schemas; do not attach ``gt`` etc.
# beside a reference to another constrained alias.
type CandidateLimit = Annotated[StrictInt, Field(ge=1, le=100)]
type NonRootJsonPointer = Annotated[
    StrictStr,
    Field(min_length=1, pattern=r"^/(?:[^~/]|~[01])*(?:/(?:[^~/]|~[01])*)*$"),
]
type RepositoryRelativePath = Annotated[
    StrictStr,
    Field(min_length=1),
    AfterValidator(_validate_repository_relative_path),
]
type ImageChannelCount = Annotated[
    StrictInt,
    Field(json_schema_extra={"enum": [1, 3]}),
    AfterValidator(_validate_image_channel_count),
]


class DomainModel(BaseModel):
    """Base for public domain models with no coercion or unknown fields."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class ArtifactRef(DomainModel):
    """Reference to an immutable artifact and its de-sensitized metadata."""

    schema_version: Literal["1"]
    artifact_id: OpaqueId
    kind: ArtifactKind
    uri: NonBlankStr
    sha256: ContentHash
    size_bytes: NonNegativeInt
    media_type: NonBlankStr
    created_at: UTCDateTime
    producer: NonBlankStr
    sensitivity: Literal["PUBLIC", "INTERNAL", "RESTRICTED"]
    metadata: JsonObject = Field(default_factory=dict)


class ProvenanceRef(DomainModel):
    """Provenance for a retrieved or generated domain record."""

    schema_version: Literal["1"]
    source_type: Literal["provider", "api", "user", "generated"]
    source_id: NonBlankStr
    source_url: NonBlankStr | None = None
    retrieved_at: UTCDateTime
    content_hash: ContentHash | None = None
    evidence_artifact_id: OpaqueId | None = None


class ResourceRequest(DomainModel):
    """Bounded compute and network capability requested for one run."""

    schema_version: Literal["1"]
    cpu_cores: PositiveFiniteFloat
    memory_mb: PositiveInt
    gpu_count: NonNegativeInt
    gpu_type: NonBlankStr | None = None
    walltime_seconds: PositiveInt
    scratch_mb: NonNegativeInt
    network_policy: NetworkPolicy
    network_allowlist_refs: list[NetworkAllowlistRef] = Field(default_factory=list)

    @field_validator("network_allowlist_refs")
    @classmethod
    def _deduplicate_network_allowlist(cls, value: list[str]) -> list[str]:
        """Normalize repeat references without changing their declared order."""
        return _deduplicate_preserving_order(value)

    @model_validator(mode="after")
    def _validate_network_policy(self) -> "ResourceRequest":
        if self.network_policy is NetworkPolicy.NONE and self.network_allowlist_refs:
            raise ValueError("network_allowlist_refs must be empty when network_policy=NONE")
        if self.network_policy is NetworkPolicy.ALLOWLIST and not self.network_allowlist_refs:
            raise ValueError(
                "network_allowlist_refs must be non-empty when network_policy=ALLOWLIST"
            )
        return self


class MetricDefinition(DomainModel):
    """Definition of one reproducible evaluation metric."""

    schema_version: Literal["1"]
    name: NonBlankStr
    direction: Literal["MAXIMIZE", "MINIMIZE"]
    aggregation: NonBlankStr
    implementation_version: NonBlankStr
    primary: StrictBoolean
    minimum_practical_delta: FiniteFloat | None = None


# ---------------------------------------------------------------------------
# Support models
# ---------------------------------------------------------------------------


class LabelSpec(DomainModel):
    """One label in a dataset profile."""

    schema_version: Literal["1"]
    label_id: OpaqueId
    name: NonBlankStr
    severity: SeverityLevel | None = None
    is_unknown: StrictBoolean = False


class SplitPolicy(DomainModel):
    """Declarative split contract; generation remains a later adaptation responsibility."""

    schema_version: Literal["1"]
    strategy: SplitStrategy
    group_keys: list[NonBlankStr] = Field(default_factory=list)
    time_key: NonBlankStr | None = None
    time_cutoff: UTCDateTime | None = None
    holdout_values: dict[NonBlankStr, list[NonBlankStr]] = Field(default_factory=dict)
    test_fraction: OpenUnitInterval | None = None
    validation_fraction: OpenUnitInterval | None = None
    seed: NonNegativeInt | None = None

    @field_validator("group_keys")
    @classmethod
    def _deduplicate_group_keys(cls, value: list[str]) -> list[str]:
        """Deduplicate group keys while retaining first-declared order."""
        return _deduplicate_preserving_order(value)

    @field_validator("holdout_values")
    @classmethod
    def _deduplicate_holdout_values(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        """Deduplicate each holdout list while retaining first-declared order."""
        return {key: _deduplicate_preserving_order(values) for key, values in value.items()}

    @model_validator(mode="after")
    def _validate_strategy(self) -> "SplitPolicy":
        if any(key not in self.group_keys for key in self.holdout_values):
            raise ValueError("holdout_values keys must be declared in group_keys")
        if (
            self.test_fraction is not None
            and self.validation_fraction is not None
            and self.test_fraction + self.validation_fraction >= 1.0
        ):
            raise ValueError("test_fraction plus validation_fraction must be less than 1")
        if self.validation_fraction is not None and self.seed is None:
            raise ValueError("a validation_fraction that selects samples requires a seed")

        has_holdout_values = any(self.holdout_values.values())
        if self.strategy is SplitStrategy.TIME_EXTRAPOLATION:
            if self.time_key is None or self.time_cutoff is None:
                raise ValueError("TIME_EXTRAPOLATION requires time_key and time_cutoff")
            if self.holdout_values:
                raise ValueError("TIME_EXTRAPOLATION requires empty holdout_values")
            if self.test_fraction is not None:
                raise ValueError("TIME_EXTRAPOLATION requires test_fraction=None")
        elif self.strategy is SplitStrategy.GROUP_HOLDOUT:
            if not self.group_keys:
                raise ValueError("GROUP_HOLDOUT requires at least one group key")
            if self.time_key is not None or self.time_cutoff is not None:
                raise ValueError("GROUP_HOLDOUT requires time_key and time_cutoff to be null")
            if (self.test_fraction is not None) == has_holdout_values:
                raise ValueError(
                    "GROUP_HOLDOUT requires exactly one of test_fraction or "
                    "non-empty holdout_values"
                )
            if self.test_fraction is not None and self.seed is None:
                raise ValueError("GROUP_HOLDOUT with test_fraction requires a seed")
        elif self.strategy is SplitStrategy.DOMAIN_HOLDOUT:
            if not self.group_keys or not has_holdout_values:
                raise ValueError("DOMAIN_HOLDOUT requires group_keys and non-empty holdout_values")
            if self.time_key is not None or self.time_cutoff is not None:
                raise ValueError("DOMAIN_HOLDOUT requires time_key and time_cutoff to be null")
            if self.test_fraction is not None:
                raise ValueError("DOMAIN_HOLDOUT requires test_fraction=None")
        elif self.strategy is SplitStrategy.SAMPLE_STRATIFIED:
            if self.group_keys or self.holdout_values:
                raise ValueError("SAMPLE_STRATIFIED requires empty group_keys and holdout_values")
            if self.time_key is not None or self.time_cutoff is not None:
                raise ValueError("SAMPLE_STRATIFIED requires time_key and time_cutoff to be null")
            if self.test_fraction is None or self.seed is None:
                raise ValueError("SAMPLE_STRATIFIED requires test_fraction and seed")
        return self


class QuerySpec(DomainModel):
    """Research retrieval query and optional inclusive calendar-date window."""

    schema_version: Literal["1"]
    keywords: list[NonBlankStr] = Field(min_length=1)
    domains: list[NonBlankStr] = Field(default_factory=list)
    date_from: ISODate | None = None
    date_to: ISODate | None = None
    excluded_terms: list[NonBlankStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_date_window(self) -> "QuerySpec":
        if (
            self.date_from is not None
            and self.date_to is not None
            and self.date_from > self.date_to
        ):
            raise ValueError("date_from must be before or equal to date_to")
        return self


class ResearchBudget(DomainModel):
    """Explicit, caller-provided ceiling for research workflow capabilities."""

    schema_version: Literal["1"]
    max_provider_pages: NonNegativeInt
    max_provider_records: NonNegativeInt
    max_llm_calls: NonNegativeInt
    max_llm_tokens: NonNegativeInt
    max_cost_estimate: NonNegativeFiniteFloat
    max_candidate_repositories: NonNegativeInt
    max_adaptation_attempts: NonNegativeInt
    max_workflow_walltime_seconds: NonNegativeInt


class TrainingBudget(DomainModel):
    """Hard run-budget limits shared by baseline and candidate execution."""

    schema_version: Literal["1"]
    max_epochs: PositiveInt | None = None
    max_steps: PositiveInt | None = None
    max_walltime_seconds: PositiveInt
    max_test_evaluations: PositiveInt

    @model_validator(mode="after")
    def _validate_training_limit(self) -> "TrainingBudget":
        if self.max_epochs is None and self.max_steps is None:
            raise ValueError("at least one of max_epochs or max_steps is required")
        return self


class AuthorRef(DomainModel):
    """A paper author and optional external identity reference."""

    schema_version: Literal["1"]
    name: NonBlankStr
    external_id: OpaqueId | None = None


class Reason(DomainModel):
    """Stable reason code with optional bounded human explanation."""

    schema_version: Literal["1"]
    code: ReasonCode
    message: HumanText | None = None


class RiskFinding(DomainModel):
    """One static-analysis finding; it does not itself authorize an action."""

    schema_version: Literal["1"]
    finding_id: OpaqueId
    rule_id: NonBlankStr
    category: ReasonCode
    severity: SeverityLevel
    description: HumanText
    location_ref: NonBlankStr | None = None


class CompatibilityGap(DomainModel):
    """Declared mismatch between repository behavior and SEM Research Agent requirements."""

    schema_version: Literal["1"]
    gap_id: OpaqueId
    area: NonBlankStr
    current_state: NonBlankStr
    required_state: NonBlankStr
    risk: SeverityLevel | None = None


class PlannedChange(DomainModel):
    """One planned repository file change."""

    schema_version: Literal["1"]
    change_id: OpaqueId
    path: NonBlankStr
    action: Literal["CREATE", "MODIFY", "DELETE"]
    reason: HumanText | None = None


class DependencyChange(DomainModel):
    """One requested dependency mutation in an adaptation plan."""

    schema_version: Literal["1"]
    dependency_id: OpaqueId
    package: NonBlankStr
    action: Literal["ADD", "REMOVE", "PIN"]
    version_constraint: NonBlankStr | None = None
    reason: HumanText | None = None


class CommandSpec(DomainModel):
    """Execution-stage structured command with a resolved trusted cwd reference."""

    schema_version: Literal["1"]
    executable_id: NonBlankStr
    argv: list[NonBlankStr] = Field(default_factory=list)
    cwd_ref: NonBlankStr
    env_refs: dict[NonBlankStr, NonBlankStr] = Field(default_factory=dict)

    @field_validator("executable_id")
    @classmethod
    def _reject_shell_or_wrapper(cls, value: str) -> str:
        """Reject shell and command-wrapper executables before resolution."""
        return _validate_non_shell_executable(value)


class GenerationRecord(DomainModel):
    """Credential-free provider/model/prompt/output provenance for generation."""

    schema_version: Literal["1"]
    provider_id: NonBlankStr
    model_id: NonBlankStr
    prompt_template_id: NonBlankStr
    prompt_version: NonBlankStr
    prompt_hash: ContentHash
    output_hash: ContentHash


class RunEntrypoint(DomainModel):
    """Frozen template entrypoint, not a shell command or resolved workspace cwd."""

    schema_version: Literal["1"]
    executable_id: NonBlankStr
    argv: list[NonBlankStr] = Field(default_factory=list)
    cwd_subpath: RepositoryRelativePath
    env_refs: dict[NonBlankStr, NonBlankStr] = Field(default_factory=dict)

    @field_validator("executable_id")
    @classmethod
    def _reject_shell_or_wrapper(cls, value: str) -> str:
        """Reject shell and command-wrapper executables in the frozen template."""
        return _validate_non_shell_executable(value)


class ModelRunTemplate(DomainModel):
    """Frozen baseline or candidate run template without duplicate resource policy."""

    schema_version: Literal["1"]
    template_id: OpaqueId
    display_name: HumanText
    repository_id: OpaqueId
    commit_sha: GitCommitSha
    patch_hash: ContentHash | None = None
    entrypoint: RunEntrypoint
    environment_digest: ContentHash
    config_hash: ContentHash


class MetricSummary(DomainModel):
    """Finite aggregate metric summary across comparable runs."""

    schema_version: Literal["1"]
    metric_name: NonBlankStr
    mean: FiniteFloat
    spread: NonNegativeFiniteFloat | None = None
    ci_lower: FiniteFloat | None = None
    ci_upper: FiniteFloat | None = None
    delta_mean: FiniteFloat | None = None

    @model_validator(mode="after")
    def _validate_confidence_interval(self) -> "MetricSummary":
        if (
            self.ci_lower is not None
            and self.ci_upper is not None
            and self.ci_lower > self.ci_upper
        ):
            raise ValueError("ci_lower must be less than or equal to ci_upper")
        return self


class PerClassSummary(DomainModel):
    """Bounded per-label performance summary."""

    schema_version: Literal["1"]
    label_id: OpaqueId
    precision: UnitInterval
    recall: UnitInterval
    f1: UnitInterval
    support: NonNegativeInt


class PatchOperation(DomainModel):
    """A SEM Research Agent structured edit; null clearing must use ``REMOVE`` instead."""

    schema_version: Literal["1"]
    op: PatchOperationType
    path: NonRootJsonPointer
    value: JsonValue | None = None
    reason: HumanText | None = None

    @model_validator(mode="after")
    def _validate_operation_value(self) -> "PatchOperation":
        if self.op in (PatchOperationType.ADD, PatchOperationType.REPLACE) and self.value is None:
            raise ValueError("ADD and REPLACE operations require a non-null value")
        if self.op is PatchOperationType.REMOVE and self.value is not None:
            raise ValueError("REMOVE operations require value=None")
        return self


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------


class DatasetProfile(DomainModel):
    """De-identified dataset description card with descriptive JSON only."""

    schema_version: Literal["1"]
    dataset_id: OpaqueId
    version: NonBlankStr
    display_name: HumanText
    task_type: TaskType
    modality: Literal["RGB", "GRAYSCALE", "PSEUDOCOLOR", "OTHER"]
    channels: ImageChannelCount
    image_shape_policy: JsonObject
    label_schema: list[LabelSpec]
    sample_counts: JsonObject
    group_keys: list[NonBlankStr]
    split_policy: SplitPolicy
    location_ref: NonBlankStr
    content_hash: ContentHash
    authorization: JsonObject
    preprocessing_contract: JsonObject
    created_at: UTCDateTime


class ResearchRequest(DomainModel):
    """Versioned research request tied to a frozen dataset version and budget."""

    schema_version: Literal["1"]
    request_id: OpaqueId
    revision: PositiveInt
    title: HumanText
    research_question: HumanText
    dataset_id: OpaqueId
    dataset_version: NonBlankStr
    query_spec: QuerySpec
    candidate_limit: CandidateLimit = 20
    budget: ResearchBudget
    requested_by: OpaqueId
    status: WorkflowStatus
    created_at: UTCDateTime
    updated_at: UTCDateTime


class PaperCandidate(DomainModel):
    """Normalized candidate paper with provenance and bounded relevance data."""

    schema_version: Literal["1"]
    paper_id: OpaqueId
    request_id: OpaqueId
    canonical_title: HumanText
    abstract_artifact_id: OpaqueId | None = None
    authors: list[AuthorRef] = Field(default_factory=list)
    external_ids: dict[NonBlankStr, NonBlankStr] = Field(default_factory=dict)
    first_published_at: UTCDateTime
    updated_at_external: UTCDateTime | None = None
    venue: HumanText | None = None
    urls: dict[NonBlankStr, NonBlankStr] = Field(default_factory=dict)
    task_tags: list[NonBlankStr] = Field(default_factory=list)
    method_tags: list[NonBlankStr] = Field(default_factory=list)
    relevance_score: UnitInterval
    score_components: dict[NonBlankStr, FiniteFloat] = Field(default_factory=dict)
    inclusion_reasons: list[Reason] = Field(default_factory=list)
    exclusion_reasons: list[Reason] = Field(default_factory=list)
    provenance: list[ProvenanceRef] = Field(min_length=1)
    selected: StrictBoolean = False


class CodeLinkEvidence(DomainModel):
    """Auditable evidence connecting a paper candidate to a repository."""

    schema_version: Literal["1"]
    evidence_id: OpaqueId
    paper_id: OpaqueId
    repository_url: NonBlankStr
    evidence_type: Literal["paper_link", "project_page", "author_repo", "readme", "search"]
    confidence: CodeLinkConfidence
    rationale_codes: list[ReasonCode] = Field(default_factory=list)
    provenance: ProvenanceRef
    verified_at: UTCDateTime


class RepositorySnapshot(DomainModel):
    """Repository metadata fixed to a complete verified Git commit SHA."""

    schema_version: Literal["1"]
    repository_id: OpaqueId
    canonical_url: NonBlankStr
    provider: Literal["GITHUB", "LOCAL_FIXTURE"]
    owner: NonBlankStr
    name: NonBlankStr
    commit_sha: GitCommitSha
    archive_artifact_id: OpaqueId | None = None
    license_spdx: NonBlankStr | None = None
    license_status: LicenseStatus
    framework: NonBlankStr | None = None
    languages: dict[NonBlankStr, NonNegativeInt] = Field(default_factory=dict)
    default_branch: NonBlankStr | None = None
    analysis_artifact_id: OpaqueId | None = None
    risk_findings: list[RiskFinding] = Field(default_factory=list)
    provenance: list[ProvenanceRef] = Field(default_factory=list)
    analyzed_at: UTCDateTime | None = None


class AdaptationPlan(DomainModel):
    """Structured, unexecuted adaptation plan for a fixed repository snapshot."""

    schema_version: Literal["1"]
    plan_id: OpaqueId
    repository_id: OpaqueId
    commit_sha: GitCommitSha
    dataset_id: OpaqueId
    dataset_version: NonBlankStr
    target_contract_version: NonBlankStr
    gaps: list[CompatibilityGap] = Field(default_factory=list)
    file_changes: list[PlannedChange] = Field(default_factory=list)
    dependency_changes: list[DependencyChange] = Field(default_factory=list)
    commands_requested: list[CommandSpec] = Field(default_factory=list)
    validation_plan: list[ValidationStage] = Field(default_factory=list)
    data_access_plan: JsonObject = Field(default_factory=dict)
    estimated_resources: ResourceRequest
    generated_by: GenerationRecord
    status: Literal["DRAFT", "APPROVED", "REJECTED", "SUPERSEDED"]
    revision: PositiveInt


class AdaptationAttempt(DomainModel):
    """Concrete patch attempt against a fixed complete base commit SHA."""

    schema_version: Literal["1"]
    attempt_id: OpaqueId
    plan_id: OpaqueId
    plan_revision: PositiveInt
    attempt_number: PositiveInt
    workspace_ref: NonBlankStr
    patch_artifact_id: OpaqueId
    patch_hash: ContentHash
    base_commit_sha: GitCommitSha
    operation_id: OpaqueId
    status: Literal["GENERATED", "VALIDATING", "PASSED", "FAILED"]
    created_at: UTCDateTime


class ValidationResult(DomainModel):
    """Status record for one bounded validation stage."""

    schema_version: Literal["1"]
    validation_id: OpaqueId
    attempt_id: OpaqueId
    stage: ValidationStage
    status: ValidationStatus
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    command_digest: ContentHash | None = None
    environment_digest: ContentHash | None = None
    started_at: UTCDateTime | None = None
    finished_at: UTCDateTime | None = None
    exit_code: StrictInt | None = None
    log_artifact_id: OpaqueId | None = None
    output_artifact_ids: list[OpaqueId] = Field(default_factory=list)
    retryable: StrictBoolean

    @model_validator(mode="after")
    def _validate_status_timing(self) -> "ValidationResult":
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at must not be earlier than started_at")
        if self.status is ValidationStatus.PENDING:
            if (
                self.started_at is not None
                or self.finished_at is not None
                or self.exit_code is not None
            ):
                raise ValueError("PENDING validation must not carry timing or exit_code")
        elif self.status is ValidationStatus.RUNNING:
            if self.started_at is None or self.finished_at is not None:
                raise ValueError("RUNNING validation requires started_at and forbids finished_at")
        elif (
            self.status
            in {
                ValidationStatus.PASSED,
                ValidationStatus.FAILED,
                ValidationStatus.SKIPPED,
                ValidationStatus.BLOCKED,
            }
            and self.finished_at is None
        ):
            raise ValueError("terminal validation status requires finished_at")
        if (
            self.status
            in {
                ValidationStatus.FAILED,
                ValidationStatus.SKIPPED,
                ValidationStatus.BLOCKED,
            }
            and not self.reason_codes
        ):
            raise ValueError("FAILED, SKIPPED, and BLOCKED validation requires reason_codes")
        return self


class ExperimentSpec(DomainModel):
    """Frozen fair-comparison specification with a typed training budget."""

    schema_version: Literal["1"]
    experiment_id: OpaqueId
    revision: PositiveInt
    request_id: OpaqueId
    dataset_id: OpaqueId
    dataset_version: NonBlankStr
    dataset_hash: ContentHash
    baseline_ref: ModelRunTemplate
    candidate_ref: ModelRunTemplate
    seeds: list[StrictInt]
    split_manifest_artifact_id: OpaqueId
    preprocessing_hash: ContentHash
    training_budget: TrainingBudget
    metrics: list[MetricDefinition]
    resources: ResourceRequest
    environment_digest: ContentHash
    approval_id: OpaqueId
    spec_hash: ContentHash
    created_at: UTCDateTime

    @model_validator(mode="after")
    def _validate_seeds_and_primary_metric(self) -> "ExperimentSpec":
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")
        if sum(metric.primary for metric in self.metrics) != 1:
            raise ValueError("an experiment spec must define exactly one primary metric")
        return self


class ExperimentRun(DomainModel):
    """Execution lifecycle record with explicit terminal-state invariants."""

    schema_version: Literal["1"]
    run_id: OpaqueId
    experiment_id: OpaqueId
    role: Literal["BASELINE", "CANDIDATE"]
    seed: StrictInt
    status: RunStatus
    idempotency_key: NonBlankStr
    executor: Literal["LOCAL", "SLURM"]
    external_job_id: OpaqueId | None = None
    attempt: NonNegativeInt = 0
    submitted_at: UTCDateTime | None = None
    started_at: UTCDateTime | None = None
    finished_at: UTCDateTime | None = None
    manifest_artifact_id: OpaqueId | None = None
    metrics_artifact_id: OpaqueId | None = None
    predictions_artifact_id: OpaqueId | None = None
    model_artifact_id: OpaqueId | None = None
    log_artifact_id: OpaqueId | None = None
    mlflow_run_id: NonBlankStr | None = None
    failure: StructuredFailure | None = None
    revision: PositiveInt

    @model_validator(mode="after")
    def _validate_run_lifecycle(self) -> "ExperimentRun":
        if (
            self.submitted_at is not None
            and self.started_at is not None
            and self.started_at < self.submitted_at
        ):
            raise ValueError("started_at must not be earlier than submitted_at")
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at must not be earlier than started_at")
        if (
            self.submitted_at is not None
            and self.finished_at is not None
            and self.finished_at < self.submitted_at
        ):
            raise ValueError("finished_at must not be earlier than submitted_at")

        terminal_statuses = {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMED_OUT,
            RunStatus.LOST,
            RunStatus.INVALID,
        }
        submitted_statuses = terminal_statuses | {RunStatus.QUEUED, RunStatus.RUNNING}
        normally_executed_terminal_statuses = {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.TIMED_OUT,
            RunStatus.LOST,
            RunStatus.INVALID,
        }
        failure_statuses = {
            RunStatus.FAILED,
            RunStatus.TIMED_OUT,
            RunStatus.LOST,
            RunStatus.INVALID,
        }

        if self.status in terminal_statuses and self.finished_at is None:
            raise ValueError("terminal experiment run status requires finished_at")
        if self.status in submitted_statuses and self.submitted_at is None:
            raise ValueError("queued, running, and terminal runs require submitted_at")
        if (
            self.status is RunStatus.RUNNING or self.status in normally_executed_terminal_statuses
        ) and self.started_at is None:
            raise ValueError("running and completed execution runs require started_at")
        if self.status in failure_statuses and self.failure is None:
            raise ValueError("failed, timed out, lost, and invalid runs require failure")
        if self.status not in failure_statuses and self.failure is not None:
            raise ValueError("only failed, timed out, lost, and invalid runs may carry failure")
        if self.status is RunStatus.SUCCEEDED and (
            self.manifest_artifact_id is None or self.metrics_artifact_id is None
        ):
            raise ValueError("SUCCEEDED run requires manifest_artifact_id and metrics_artifact_id")
        return self


class EvaluationReport(DomainModel):
    """Machine-readable evaluation outcome with validity and evidence invariants."""

    schema_version: Literal["1"]
    report_id: OpaqueId
    experiment_id: OpaqueId
    validity: Literal["VALID", "INVALID"]
    invalidity_reasons: list[ReasonCode] = Field(default_factory=list)
    baseline_run_ids: list[OpaqueId] = Field(default_factory=list)
    candidate_run_ids: list[OpaqueId] = Field(default_factory=list)
    metric_summaries: list[MetricSummary] = Field(default_factory=list)
    per_class_summaries: list[PerClassSummary] = Field(default_factory=list)
    resource_summary: JsonObject = Field(default_factory=dict)
    conclusion: EvaluationConclusion
    conclusion_reasons: list[ReasonCode] = Field(default_factory=list)
    narrative_artifact_id: OpaqueId | None = None
    evaluation_artifact_id: OpaqueId
    evaluator_version: NonBlankStr
    created_at: UTCDateTime

    @model_validator(mode="after")
    def _validate_validity(self) -> "EvaluationReport":
        if self.validity == "INVALID" and not self.invalidity_reasons:
            raise ValueError("an INVALID report must record invalidity reasons")
        if self.invalidity_reasons and self.validity != "INVALID":
            raise ValueError("invalidity reasons require validity=INVALID")
        if self.invalidity_reasons and self.conclusion is not EvaluationConclusion.INVALID:
            raise ValueError("invalidity reasons require conclusion=INVALID")
        if self.validity == "VALID":
            if self.conclusion is EvaluationConclusion.INVALID:
                raise ValueError("a VALID report cannot conclude INVALID")
            if not self.baseline_run_ids or not self.candidate_run_ids or not self.metric_summaries:
                raise ValueError(
                    "a VALID report requires baseline_run_ids, candidate_run_ids, "
                    "and metric_summaries"
                )
        return self


class Approval(DomainModel):
    """Human gate decision bound to an exact revision of one subject."""

    schema_version: Literal["1"]
    approval_id: OpaqueId
    gate_kind: GateKind
    subject_type: NonBlankStr
    subject_id: OpaqueId
    subject_revision: PositiveInt
    decision: ApprovalDecision
    edits: list[PatchOperation] = Field(default_factory=list)
    reason: HumanText
    actor_id: OpaqueId
    decided_at: UTCDateTime
    idempotency_key: NonBlankStr
    supersedes: OpaqueId | None = None

    @model_validator(mode="after")
    def _validate_edit_decision(self) -> "Approval":
        if self.decision is ApprovalDecision.EDIT and not self.edits:
            raise ValueError("an EDIT decision requires at least one structured edit")
        if self.decision in {ApprovalDecision.APPROVE, ApprovalDecision.REJECT} and self.edits:
            raise ValueError("APPROVE and REJECT decisions require empty edits")
        return self


def approval_authorizes(
    approval: Approval,
    subject_type: str,
    subject_id: str,
    subject_revision: int,
) -> bool:
    """Return whether an APPROVE decision targets exactly this subject revision."""
    return (
        approval.decision is ApprovalDecision.APPROVE
        and approval.subject_type == subject_type
        and approval.subject_id == subject_id
        and approval.subject_revision == subject_revision
    )


__all__ = [
    "AdaptationAttempt",
    "AdaptationPlan",
    "Approval",
    "ArtifactRef",
    "AuthorRef",
    "CodeLinkEvidence",
    "CommandSpec",
    "CompatibilityGap",
    "DatasetProfile",
    "DependencyChange",
    "DomainModel",
    "EvaluationReport",
    "ExperimentRun",
    "ExperimentSpec",
    "GenerationRecord",
    "LabelSpec",
    "MetricDefinition",
    "MetricSummary",
    "ModelRunTemplate",
    "PaperCandidate",
    "PatchOperation",
    "PerClassSummary",
    "PlannedChange",
    "ProvenanceRef",
    "QuerySpec",
    "Reason",
    "RepositoryRelativePath",
    "RepositorySnapshot",
    "ResearchBudget",
    "ResearchRequest",
    "ResourceRequest",
    "RiskFinding",
    "RunEntrypoint",
    "SplitPolicy",
    "TrainingBudget",
    "ValidationResult",
    "approval_authorizes",
]
