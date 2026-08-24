"""Strict application-owned records for the adaptation workflow Adaptation Agent."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from vision_research_ops.domain import (
    ContentHash,
    GenerationRecord,
    JsonObject,
    NonBlankStr,
    NonNegativeInt,
    OpaqueId,
    PositiveInt,
    StructuredFailure,
    UTCDateTime,
    ValidationStage,
    ValidationStatus,
)

GapArea = Literal[
    "INPUT_CHANNELS",
    "NUM_CLASSES",
    "LABEL_MAPPING",
    "GROUP_SPLIT",
    "METRICS_OUTPUT",
]
PatchField = Literal[
    "/input/channels",
    "/model/num_classes",
    "/data/label_mapping",
    "/data/group_split_key",
    "/metrics/names",
    "/metrics/output_file",
]
MetricName = Literal["macro_f1", "balanced_accuracy", "per_class_recall"]
PlannerToolName = Literal[
    "inspect_repository_profile",
    "inspect_dataset_contract",
    "compare_repository_dataset",
    "validate_adaptation_plan",
]

REQUIRED_GAP_AREAS: frozenset[str] = frozenset(
    {
        "INPUT_CHANNELS",
        "NUM_CLASSES",
        "LABEL_MAPPING",
        "GROUP_SPLIT",
        "METRICS_OUTPUT",
    }
)
REQUIRED_PATCH_FIELDS: frozenset[str] = frozenset(
    {
        "/input/channels",
        "/model/num_classes",
        "/data/label_mapping",
        "/data/group_split_key",
        "/metrics/names",
        "/metrics/output_file",
    }
)
REQUIRED_METRICS: tuple[str, ...] = (
    "macro_f1",
    "balanced_accuracy",
    "per_class_recall",
)

_SENSITIVE_GENERATED_ID_MARKERS = (
    "api-key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "private",
    "secret",
    "token",
)


def _reject_sensitive_generated_id(value: str) -> str:
    if any(marker in value.casefold() for marker in _SENSITIVE_GENERATED_ID_MARKERS):
        raise ValueError("generated identifiers cannot contain path or credential-like terms")
    return value


type P3GitCommitSha = Annotated[
    StrictStr,
    Field(pattern=r"^[0-9a-f]{40}$"),
]
type CanonicalGapId = Annotated[
    StrictStr,
    Field(
        min_length=5,
        max_length=64,
        pattern=r"^gap-[a-z0-9]+(?:-[a-z0-9]+){0,5}$",
    ),
    AfterValidator(_reject_sensitive_generated_id),
]
type CanonicalChangeId = Annotated[
    StrictStr,
    Field(
        min_length=8,
        max_length=64,
        pattern=r"^change-[a-z0-9]+(?:-[a-z0-9]+){0,5}$",
    ),
    AfterValidator(_reject_sensitive_generated_id),
]


class AdaptationModel(BaseModel):
    """Strict JSON-safe base for adaptation application records."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


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
        raise ValueError("artifact references must be canonical POSIX relative paths")
    return value


class CompatibilityGapProposal(AdaptationModel):
    """One schema-bound LLM compatibility observation."""

    schema_version: Literal["1"] = "1"
    gap_id: CanonicalGapId
    area: GapArea
    current_state: NonBlankStr
    required_state: NonBlankStr
    risk: Literal["LOW", "MEDIUM", "HIGH"]


class AdaptationChangeProposal(AdaptationModel):
    """One declarative change target; it is not a filesystem instruction."""

    schema_version: Literal["1"] = "1"
    change_id: CanonicalChangeId
    area: GapArea
    target_template: Literal["SEM_PLAIN_PYTORCH_CONFIG_V1"]
    target_field: PatchField
    action: Literal["SET"]
    reason: NonBlankStr


class AdaptationPlanProposal(AdaptationModel):
    """Strict LLM proposal compiled by deterministic code into one safe template."""

    schema_version: Literal["1"] = "1"
    gaps: list[CompatibilityGapProposal] = Field(min_length=5, max_length=5)
    channels: PositiveInt
    num_classes: PositiveInt
    label_mapping: dict[NonBlankStr, NonNegativeInt]
    group_split_key: NonBlankStr
    metrics: list[MetricName] = Field(min_length=3, max_length=3)
    metrics_output_file: NonBlankStr
    changes: list[AdaptationChangeProposal] = Field(min_length=6, max_length=6)
    rationale: NonBlankStr

    @field_validator("label_mapping")
    @classmethod
    def _require_dense_label_indices(cls, value: dict[str, int]) -> dict[str, int]:
        if not value:
            raise ValueError("label_mapping must not be empty")
        if sorted(value.values()) != list(range(len(value))):
            raise ValueError("label_mapping values must be dense zero-based indices")
        return value

    @field_validator("metrics")
    @classmethod
    def _require_fixed_metrics(cls, value: list[str]) -> list[str]:
        if tuple(value) != REQUIRED_METRICS:
            raise ValueError("metrics must use the fixed pipeline evaluation contract")
        return value

    @field_validator("metrics_output_file")
    @classmethod
    def _require_safe_metrics_output(cls, value: str) -> str:
        safe = _safe_relative_ref(value)
        if not safe.casefold().endswith(".json"):
            raise ValueError("metrics output must be a JSON relative path")
        if any(
            part.casefold() in {".git", ".env", "secrets", "credentials"}
            for part in safe.split("/")
        ):
            raise ValueError("metrics output cannot target control or secret files")
        return safe

    @model_validator(mode="after")
    def _require_complete_contract(self) -> AdaptationPlanProposal:
        if {gap.area for gap in self.gaps} != REQUIRED_GAP_AREAS:
            raise ValueError("gaps must cover every fixed adaptation area exactly once")
        if len({gap.area for gap in self.gaps}) != len(self.gaps):
            raise ValueError("adaptation gap areas must be unique")
        if {change.target_field for change in self.changes} != REQUIRED_PATCH_FIELDS:
            raise ValueError("changes must cover every fixed template field exactly once")
        if len({change.target_field for change in self.changes}) != len(self.changes):
            raise ValueError("adaptation change fields must be unique")
        if self.num_classes != len(self.label_mapping):
            raise ValueError("num_classes must match label_mapping size")
        return self


class AdaptationInputFacts(AdaptationModel):
    """Validated, de-identified facts used to verify an LLM proposal."""

    schema_version: Literal["1"] = "1"
    repository_id: OpaqueId
    repository_url: NonBlankStr
    base_commit_sha: P3GitCommitSha
    structure_type: Literal["PLAIN_PYTORCH"]
    license_spdx: NonBlankStr
    dataset_id: OpaqueId
    dataset_version: NonBlankStr
    dataset_content_hash: ContentHash
    modality: Literal["GRAYSCALE"]
    channels: Literal[1]
    label_names: list[NonBlankStr] = Field(min_length=2)
    group_keys: list[NonBlankStr] = Field(min_length=1)
    group_split_key: NonBlankStr
    dataset_kind: Literal["SYNTHETIC_SEM_FIXTURE"]
    repository_kind: Literal["CONTROLLED_PLAIN_PYTORCH_FIXTURE"]


class PlannerToolEvent(AdaptationModel):
    """Hash-only evidence for one allowlisted read-only planner tool call."""

    schema_version: Literal["1"] = "1"
    call_index: PositiveInt
    tool_name: PlannerToolName
    arguments_hash: ContentHash
    output_hash: ContentHash
    status: Literal["SUCCESS"] = "SUCCESS"


class AdaptationPlannerTrace(AdaptationModel):
    """De-sensitized evidence from the bounded LangGraph tool-calling loop."""

    schema_version: Literal["1"] = "1"
    trace_id: OpaqueId
    workflow_id: OpaqueId
    planner_kind: Literal["SCRIPTED_TOOL_CALLING", "DASHSCOPE_TOOL_CALLING"]
    provider_id: NonBlankStr
    model_id: NonBlankStr
    events: list[PlannerToolEvent] = Field(min_length=4, max_length=6)
    prompt_hash: ContentHash
    output_hash: ContentHash
    completed: Literal[True] = True

    @model_validator(mode="after")
    def _require_complete_tool_evidence(self) -> AdaptationPlannerTrace:
        if [event.call_index for event in self.events] != list(range(1, len(self.events) + 1)):
            raise ValueError("planner tool call indices must be dense and ordered")
        observed = {event.tool_name for event in self.events}
        required: set[str] = {
            "inspect_repository_profile",
            "inspect_dataset_contract",
            "compare_repository_dataset",
            "validate_adaptation_plan",
        }
        if not required.issubset(observed):
            raise ValueError("planner trace must contain every required tool")
        return self


class RepairRecord(AdaptationModel):
    """Evidence for the single deterministic repair opportunity."""

    schema_version: Literal["1"] = "1"
    from_revision: PositiveInt
    to_revision: PositiveInt
    reason_code: NonBlankStr
    repaired_at: UTCDateTime


class CompiledAdaptationPlan(AdaptationModel):
    """Deterministically validated plan persisted outside graph checkpoints."""

    schema_version: Literal["1"] = "1"
    workflow_id: OpaqueId
    plan_id: OpaqueId
    revision: PositiveInt
    repository_id: OpaqueId
    repository_url: NonBlankStr
    base_commit_sha: P3GitCommitSha
    dataset_id: OpaqueId
    dataset_version: NonBlankStr
    dataset_content_hash: ContentHash
    dataset_kind: Literal["SYNTHETIC_SEM_FIXTURE"]
    repository_kind: Literal["CONTROLLED_PLAIN_PYTORCH_FIXTURE"]
    proposal: AdaptationPlanProposal
    generation: GenerationRecord
    repair_revision: NonNegativeInt = 0
    repair_history: list[RepairRecord] = Field(default_factory=list)
    origin: Literal["LLM_PROPOSAL", "HUMAN_EDIT", "DETERMINISTIC_REPAIR"]
    created_at: UTCDateTime
    updated_at: UTCDateTime


class PatchChangeRecord(AdaptationModel):
    """One applied, policy-approved file change summary."""

    schema_version: Literal["1"] = "1"
    path: Literal["sem_adaptation.json"]
    operation: Literal["MODIFY"]
    field_paths: list[PatchField] = Field(min_length=6, max_length=6)
    before_hash: ContentHash
    after_hash: ContentHash


class PatchArtifactRecord(AdaptationModel):
    """Immutable evidence for one deterministic patch revision."""

    schema_version: Literal["1"] = "1"
    workflow_id: OpaqueId
    attempt_id: OpaqueId
    attempt_number: PositiveInt
    plan_id: OpaqueId
    plan_revision: PositiveInt
    repository_id: OpaqueId
    base_commit_sha: P3GitCommitSha
    dataset_version: NonBlankStr
    patch_hash: ContentHash
    workspace_ref: NonBlankStr
    patch_ref: NonBlankStr
    manifest_ref: NonBlankStr
    changes: list[PatchChangeRecord] = Field(min_length=1, max_length=1)
    fixture_labeled: Literal[True] = True
    synthetic_data_labeled: Literal[True] = True
    created_at: UTCDateTime

    @field_validator("workspace_ref", "patch_ref", "manifest_ref")
    @classmethod
    def _validate_refs(cls, value: str) -> str:
        return _safe_relative_ref(value)


class SmokeCommandRecord(AdaptationModel):
    """Sanitized command evidence for a fixed fixture probe stage."""

    schema_version: Literal["1"] = "1"
    executable_id: Literal["python-current"]
    argv: list[NonBlankStr]
    cwd_ref: NonBlankStr
    shell: Literal[False] = False
    network: Literal[False] = False
    installs_dependencies: Literal[False] = False

    @field_validator("cwd_ref")
    @classmethod
    def _validate_cwd_ref(cls, value: str) -> str:
        return _safe_relative_ref(value)


class SmokeStageRecord(AdaptationModel):
    """Actual result for one subprocess-backed fixture contract stage."""

    schema_version: Literal["1"] = "1"
    stage: ValidationStage
    status: ValidationStatus
    exit_code: int
    command: SmokeCommandRecord
    command_digest: ContentHash
    evidence: JsonObject
    log_ref: NonBlankStr
    started_at: UTCDateTime
    finished_at: UTCDateTime

    @field_validator("log_ref")
    @classmethod
    def _validate_log_ref(cls, value: str) -> str:
        return _safe_relative_ref(value)


class SmokeResultRecord(AdaptationModel):
    """Bounded smoke result with an explicit no-Torch capability label."""

    schema_version: Literal["1"] = "1"
    workflow_id: OpaqueId
    attempt_id: OpaqueId
    plan_revision: PositiveInt
    repository_id: OpaqueId
    base_commit_sha: P3GitCommitSha
    dataset_version: NonBlankStr
    patch_hash: ContentHash
    status: Literal["PASSED", "FAILED"]
    stages: list[SmokeStageRecord] = Field(min_length=1, max_length=4)
    result_ref: NonBlankStr
    retryable: bool
    capability_boundary: Literal["FIXTURE_CONTRACT_PROBE_NO_TORCH"]
    real_pytorch_training: Literal[False] = False
    shell_used: Literal[False] = False
    network_used: Literal[False] = False
    dependency_install_used: Literal[False] = False
    fixture_labeled: Literal[True] = True
    synthetic_data_labeled: Literal[True] = True
    started_at: UTCDateTime
    finished_at: UTCDateTime

    @field_validator("result_ref")
    @classmethod
    def _validate_result_ref(cls, value: str) -> str:
        return _safe_relative_ref(value)

    @model_validator(mode="after")
    def _status_matches_stages(self) -> SmokeResultRecord:
        all_passed = all(stage.status is ValidationStatus.PASSED for stage in self.stages)
        if (self.status == "PASSED") != all_passed:
            raise ValueError("smoke status must match its actual stage results")
        return self


class AttemptEvidence(AdaptationModel):
    """Small artifact-reference summary retained in adaptation evidence."""

    schema_version: Literal["1"] = "1"
    attempt_id: OpaqueId
    plan_revision: PositiveInt
    patch_hash: ContentHash
    patch_ref: NonBlankStr
    patch_manifest_ref: NonBlankStr
    smoke_ref: NonBlankStr | None = None
    smoke_status: Literal["PASSED", "FAILED"] | None = None

    @field_validator("patch_ref", "patch_manifest_ref", "smoke_ref")
    @classmethod
    def _validate_optional_refs(cls, value: str | None) -> str | None:
        return None if value is None else _safe_relative_ref(value)


class PatchReviewRecord(AdaptationModel):
    """Persisted human decision bound to one exact patch revision and hash."""

    schema_version: Literal["1"] = "1"
    approval_id: OpaqueId
    decision: Literal["APPROVE", "EDIT", "REJECT"]
    gate_id: OpaqueId
    subject_id: OpaqueId
    subject_revision: PositiveInt
    patch_hash: ContentHash
    actor_id: OpaqueId
    decided_at: UTCDateTime


class AdaptationResult(AdaptationModel):
    """Canonical local evidence index for one adaptation workflow."""

    schema_version: Literal["1"] = "1"
    workflow_id: OpaqueId
    request_id: OpaqueId
    repository_workflow_id: OpaqueId
    status: Literal[
        "INPUT_VALIDATED",
        "PLANNED",
        "PATCHED",
        "REPAIRING",
        "AWAITING_APPROVAL",
        "ACCEPTED",
        "REJECTED",
        "FAILED",
    ]
    repository_id: OpaqueId | None = None
    repository_url: NonBlankStr | None = None
    base_commit_sha: P3GitCommitSha | None = None
    dataset_id: OpaqueId | None = None
    dataset_version: NonBlankStr | None = None
    dataset_content_hash: ContentHash | None = None
    dataset_kind: Literal["SYNTHETIC_SEM_FIXTURE"] | None = None
    repository_kind: Literal["CONTROLLED_PLAIN_PYTORCH_FIXTURE"] | None = None
    generation: GenerationRecord | None = None
    gaps: list[CompatibilityGapProposal] = Field(default_factory=list)
    changes: list[AdaptationChangeProposal] = Field(default_factory=list)
    plan_id: OpaqueId | None = None
    plan_revision: PositiveInt | None = None
    plan_ref: NonBlankStr | None = None
    planner_trace_ref: NonBlankStr | None = None
    attempts: list[AttemptEvidence] = Field(default_factory=list)
    reviews: list[PatchReviewRecord] = Field(default_factory=list)
    repair_count: NonNegativeInt = 0
    gate_id: OpaqueId | None = None
    gate_revision: PositiveInt | None = None
    gate_subject_id: OpaqueId | None = None
    gate_patch_hash: ContentHash | None = None
    accepted_patch_hash: ContentHash | None = None
    approval_id: OpaqueId | None = None
    failure: StructuredFailure | None = None
    fixture_labeled: Literal[True] = True
    synthetic_data_labeled: Literal[True] = True
    created_at: UTCDateTime
    updated_at: UTCDateTime

    @field_validator("plan_ref", "planner_trace_ref")
    @classmethod
    def _validate_plan_ref(cls, value: str | None) -> str | None:
        return None if value is None else _safe_relative_ref(value)

    @model_validator(mode="after")
    def _validate_terminal_evidence(self) -> AdaptationResult:
        if self.status == "FAILED" and self.failure is None:
            raise ValueError("failed adaptation results require a structured failure")
        if self.status != "FAILED" and self.failure is not None:
            raise ValueError("only failed adaptation results may carry a terminal failure")
        if self.status == "ACCEPTED":
            if self.accepted_patch_hash is None or self.approval_id is None:
                raise ValueError("accepted adaptation results require exact approval evidence")
            if self.accepted_patch_hash != self.gate_patch_hash:
                raise ValueError("accepted patch hash must match the reviewed Gate hash")
        return self


class FixtureProbeOutput(AdaptationModel):
    """Machine-readable stdout contract emitted by the controlled probe."""

    schema_version: Literal["1"] = "1"
    stage: ValidationStage
    passed: bool
    capability_boundary: Literal["FIXTURE_CONTRACT_PROBE_NO_TORCH"]
    evidence: JsonObject
    reason_code: NonBlankStr | None = None


__all__ = [
    "REQUIRED_GAP_AREAS",
    "REQUIRED_METRICS",
    "REQUIRED_PATCH_FIELDS",
    "AdaptationChangeProposal",
    "AdaptationInputFacts",
    "AdaptationModel",
    "AdaptationPlanProposal",
    "AdaptationPlannerTrace",
    "AdaptationResult",
    "AttemptEvidence",
    "CanonicalChangeId",
    "CanonicalGapId",
    "CompatibilityGapProposal",
    "CompiledAdaptationPlan",
    "FixtureProbeOutput",
    "P3GitCommitSha",
    "PatchArtifactRecord",
    "PatchChangeRecord",
    "PatchReviewRecord",
    "PlannerToolEvent",
    "PlannerToolName",
    "RepairRecord",
    "SmokeCommandRecord",
    "SmokeResultRecord",
    "SmokeStageRecord",
]
