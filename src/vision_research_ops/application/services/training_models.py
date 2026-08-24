"""Strict training-owned models for frozen local training and its artifacts."""

from __future__ import annotations

import json
from hashlib import sha256
from math import isclose
from typing import Annotated, Literal, Self

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

from vision_research_ops.domain import (
    ContentHash,
    NonBlankStr,
    PatchOperationType,
    StructuredFailure,
    UnitInterval,
    UTCDateTime,
)

TRAINING_CAPABILITY: Literal["SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"] = (
    "SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"
)
TRAINING_ENTRYPOINT_ID: Literal["synthetic-linear-cpu-fixture"] = "synthetic-linear-cpu-fixture"
TRAINING_ENTRYPOINT_REF = "fixtures/training/synthetic_linear_cpu.py"

RunRole = Literal["BASELINE", "CANDIDATE"]
TrainingMethod = Literal["GLOBAL_STATS_LINEAR", "GRID4_LINEAR_PATCHED"]
TrainingStatus = Literal[
    "INPUT_VALIDATED",
    "AWAITING_APPROVAL",
    "EDIT_REQUESTED",
    "RUNNING",
    "SUCCEEDED",
    "REJECTED",
    "CANCELLED",
    "FAILED",
]


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
        raise ValueError("references must be canonical POSIX relative paths")
    return value


type P4GitCommitSha = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{40}$")]
type CanonicalId = Annotated[
    StrictStr,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]
type RelativeRef = Annotated[StrictStr, AfterValidator(_safe_relative_ref)]
type SmallEpochCount = Annotated[StrictInt, Field(ge=1, le=8)]
type SmallStepCount = Annotated[StrictInt, Field(ge=1, le=128)]
type SmallWalltime = Annotated[StrictInt, Field(ge=1, le=30)]
type FixtureSeed = Annotated[StrictInt, Field(ge=0, le=2_147_483_647)]
type ClassIndex = Annotated[StrictInt, Field(ge=0, le=3)]


class TrainingModel(BaseModel):
    """Strict JSON-safe base for training application records."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class TrainingBudgetSpec(TrainingModel):
    """Hard limits shared by the baseline and candidate fixture runs."""

    schema_version: Literal["1"] = "1"
    max_epochs: SmallEpochCount
    max_steps: SmallStepCount
    max_walltime_seconds: SmallWalltime


class TrainingInput(TrainingModel):
    """Exact accepted-adaptation and fixture references required before the training Gate."""

    schema_version: Literal["1"] = "1"
    adaptation_workflow_id: CanonicalId
    base_commit_sha: P4GitCommitSha
    patch_revision: Annotated[StrictInt, Field(ge=1)]
    patch_hash: ContentHash
    dataset_id: CanonicalId
    dataset_version: NonBlankStr
    dataset_content_hash: ContentHash
    dataset_ref: RelativeRef
    dataset_ref_hash: ContentHash
    split_ref: RelativeRef
    split_hash: ContentHash
    preprocess_ref: RelativeRef
    preprocess_hash: ContentHash
    seed: FixtureSeed
    budget: TrainingBudgetSpec


class TrainingCommandSpec(TrainingModel):
    """Sanitized fixed argv accepted by the local training adapter."""

    schema_version: Literal["1"] = "1"
    executable_id: Literal["python-current"] = "python-current"
    entrypoint_id: Literal["synthetic-linear-cpu-fixture"] = TRAINING_ENTRYPOINT_ID
    run_id: CanonicalId
    role: RunRole
    spec_ref: RelativeRef
    output_ref: RelativeRef
    argv: list[NonBlankStr] = Field(min_length=10, max_length=10)
    cwd_ref: Literal["."] = "."
    env_keys: list[
        Literal[
            "PYTHONIOENCODING",
            "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE",
        ]
    ] = Field(default_factory=list, min_length=3, max_length=3)
    shell: Literal[False] = False
    network: Literal[False] = False
    installs_dependencies: Literal[False] = False
    checkpoint_ref: None = None

    @model_validator(mode="after")
    def _fixed_command(self) -> Self:
        expected_output = f"runs/{self.run_id}"
        if self.output_ref != expected_output:
            raise ValueError("training output_ref must be the canonical run directory")
        expected_argv = [
            "-I",
            TRAINING_ENTRYPOINT_REF,
            "--spec-ref",
            f"var/{self.spec_ref}",
            "--run-id",
            self.run_id,
            "--role",
            self.role,
            "--output-ref",
            f"var/{self.output_ref}",
        ]
        if self.argv != expected_argv:
            raise ValueError("training argv does not match the fixed fixture template")
        if self.env_keys != [
            "PYTHONIOENCODING",
            "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE",
        ]:
            raise ValueError("training environment keys are fixed and ordered")
        return self


class FrozenRunSpec(TrainingModel):
    """One side of a frozen, budget-matched baseline/candidate pair."""

    schema_version: Literal["1"] = "1"
    run_id: CanonicalId
    role: RunRole
    method: TrainingMethod
    base_commit_sha: P4GitCommitSha
    candidate_patch_revision: Annotated[StrictInt, Field(ge=1)] | None = None
    candidate_patch_hash: ContentHash | None = None
    dataset_id: CanonicalId
    dataset_version: NonBlankStr
    dataset_content_hash: ContentHash
    dataset_ref: RelativeRef
    dataset_ref_hash: ContentHash
    split_ref: RelativeRef
    split_hash: ContentHash
    preprocess_ref: RelativeRef
    preprocess_hash: ContentHash
    method_config_ref: RelativeRef
    method_config_hash: ContentHash
    seed: FixtureSeed
    budget: TrainingBudgetSpec
    capability: Literal["SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"] = TRAINING_CAPABILITY
    real_pytorch_training: Literal[False] = False
    fixture_labeled: Literal[True] = True
    synthetic_data_labeled: Literal[True] = True
    command: TrainingCommandSpec

    @model_validator(mode="after")
    def _role_contract(self) -> Self:
        if self.command.run_id != self.run_id or self.command.role != self.role:
            raise ValueError("run command must target its exact frozen run")
        if self.role == "BASELINE":
            if self.method != "GLOBAL_STATS_LINEAR":
                raise ValueError("baseline must use the fixed global-statistics classifier")
            if self.candidate_patch_revision is not None or self.candidate_patch_hash is not None:
                raise ValueError("baseline cannot claim the candidate patch")
        else:
            if self.method != "GRID4_LINEAR_PATCHED":
                raise ValueError("candidate must use the fixed patched grid classifier")
            if self.candidate_patch_revision is None or self.candidate_patch_hash is None:
                raise ValueError("candidate requires exact patch revision and hash")
        return self


def canonical_json_bytes(value: object) -> bytes:
    """Encode a JSON value deterministically for immutable evidence hashes."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def content_hash(data: bytes) -> str:
    """Return the repository's canonical SHA-256 spelling."""
    return f"sha256:{sha256(data).hexdigest()}"


class FrozenTrainingSpec(TrainingModel):
    """Immutable, hash-checked submission reviewed by the RUN_SUBMISSION Gate."""

    schema_version: Literal["1"] = "1"
    workflow_id: CanonicalId
    adaptation_workflow_id: CanonicalId
    revision: Annotated[StrictInt, Field(ge=1)]
    base_commit_sha: P4GitCommitSha
    patch_revision: Annotated[StrictInt, Field(ge=1)]
    patch_hash: ContentHash
    adaptation_result_ref: RelativeRef
    patch_manifest_ref: RelativeRef
    smoke_result_ref: RelativeRef
    p3_approval_id: CanonicalId
    baseline: FrozenRunSpec
    candidate: FrozenRunSpec
    spec_hash: ContentHash

    @model_validator(mode="after")
    def _fair_and_hashed(self) -> Self:
        if self.baseline.role != "BASELINE" or self.candidate.role != "CANDIDATE":
            raise ValueError("frozen comparison requires baseline then candidate")
        shared_fields = (
            "base_commit_sha",
            "dataset_id",
            "dataset_version",
            "dataset_content_hash",
            "dataset_ref",
            "dataset_ref_hash",
            "split_ref",
            "split_hash",
            "preprocess_ref",
            "preprocess_hash",
            "seed",
            "budget",
            "capability",
            "real_pytorch_training",
            "fixture_labeled",
            "synthetic_data_labeled",
        )
        for field in shared_fields:
            if getattr(self.baseline, field) != getattr(self.candidate, field):
                raise ValueError(f"baseline and candidate must share frozen {field}")
        if self.baseline.method_config_ref == self.candidate.method_config_ref:
            raise ValueError("the fixed candidate config must differ from the baseline config")
        if (
            self.candidate.candidate_patch_revision != self.patch_revision
            or self.candidate.candidate_patch_hash != self.patch_hash
        ):
            raise ValueError("candidate provenance must match the accepted adaptation patch")
        payload = self.model_dump(mode="json", exclude={"spec_hash"})
        if content_hash(canonical_json_bytes(payload)) != self.spec_hash:
            raise ValueError("frozen training spec hash does not match its canonical content")
        return self


class TrainingEdit(TrainingModel):
    """One allowlisted integer edit used to regenerate the complete frozen pair."""

    schema_version: Literal["1"] = "1"
    op: Literal[PatchOperationType.REPLACE] = PatchOperationType.REPLACE
    path: Literal[
        "/budget/max_epochs",
        "/budget/max_steps",
        "/budget/max_walltime_seconds",
        "/seed",
    ]
    value: StrictInt


class TrainingReviewRecord(TrainingModel):
    """Human decision bound to one exact frozen spec revision and hash."""

    schema_version: Literal["1"] = "1"
    approval_id: CanonicalId
    decision: Literal["APPROVE", "EDIT", "REJECT"]
    gate_id: CanonicalId
    subject_id: CanonicalId
    subject_revision: Annotated[StrictInt, Field(ge=1)]
    spec_hash: ContentHash
    actor_id: CanonicalId
    decided_at: UTCDateTime


class PendingTrainingEdit(TrainingModel):
    """Sanitized edit payload persisted outside graph state until refreeze."""

    schema_version: Literal["1"] = "1"
    approval_id: CanonicalId
    edits: list[TrainingEdit] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _unique_paths(self) -> Self:
        if len({edit.path for edit in self.edits}) != len(self.edits):
            raise ValueError("training edit paths must be unique")
        return self


class LossPoint(TrainingModel):
    """One actual optimization step loss."""

    step: SmallStepCount
    epoch: SmallEpochCount
    loss: Annotated[float, Field(strict=True, ge=0.0)]


class EpochLoss(TrainingModel):
    """Mean loss for one completed or budget-truncated epoch."""

    epoch: SmallEpochCount
    steps: SmallStepCount
    mean_loss: Annotated[float, Field(strict=True, ge=0.0)]


class TrainingMetrics(TrainingModel):
    """Structured optimization evidence emitted by the trusted fixture."""

    schema_version: Literal["1"] = "1"
    run_id: CanonicalId
    role: RunRole
    status: Literal["SUCCEEDED"]
    spec_hash: ContentHash
    seed: FixtureSeed
    budget: TrainingBudgetSpec
    step_losses: list[LossPoint] = Field(min_length=1, max_length=128)
    epoch_losses: list[EpochLoss] = Field(min_length=1, max_length=8)
    initial_loss: Annotated[float, Field(strict=True, ge=0.0)]
    final_loss: Annotated[float, Field(strict=True, ge=0.0)]
    test_accuracy: UnitInterval
    prediction_count: Annotated[StrictInt, Field(ge=1, le=24)]
    capability: Literal["SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"] = TRAINING_CAPABILITY
    real_pytorch_training: Literal[False] = False

    @model_validator(mode="after")
    def _loss_contract(self) -> Self:
        if len(self.step_losses) > self.budget.max_steps:
            raise ValueError("step loss count exceeds the frozen max_steps")
        if len(self.epoch_losses) > self.budget.max_epochs:
            raise ValueError("epoch loss count exceeds the frozen max_epochs")
        if not isclose(self.initial_loss, self.step_losses[0].loss, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("initial_loss must equal the first actual step loss")
        if not isclose(self.final_loss, self.step_losses[-1].loss, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("final_loss must equal the last actual step loss")
        return self


class PredictionItem(TrainingModel):
    """One pseudonymous fixture test prediction."""

    sample_id: CanonicalId
    true_label: ClassIndex
    predicted_label: ClassIndex
    scores: list[Annotated[float, Field(strict=True, ge=0.0, le=1.0)]] = Field(
        min_length=4,
        max_length=4,
    )

    @field_validator("scores")
    @classmethod
    def _probabilities_sum_to_one(cls, value: list[float]) -> list[float]:
        if not isclose(sum(value), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("prediction scores must sum to one")
        return value


class TrainingPredictions(TrainingModel):
    """Structured test predictions for later deterministic evaluation evaluation."""

    schema_version: Literal["1"] = "1"
    run_id: CanonicalId
    role: RunRole
    spec_hash: ContentHash
    split_ref: RelativeRef
    split_hash: ContentHash
    items: list[PredictionItem] = Field(min_length=1, max_length=24)
    capability: Literal["SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"] = TRAINING_CAPABILITY
    real_pytorch_training: Literal[False] = False

    @model_validator(mode="after")
    def _unique_samples(self) -> Self:
        if len({item.sample_id for item in self.items}) != len(self.items):
            raise ValueError("prediction sample IDs must be unique")
        return self


class TrainingRunManifest(TrainingModel):
    """Traceable manifest for one controlled local fixture run."""

    schema_version: Literal["1"] = "1"
    run_id: CanonicalId
    role: RunRole
    status: Literal["SUCCEEDED"]
    spec_ref: RelativeRef
    spec_hash: ContentHash
    base_commit_sha: P4GitCommitSha
    candidate_patch_revision: Annotated[StrictInt, Field(ge=1)] | None = None
    candidate_patch_hash: ContentHash | None = None
    dataset_id: CanonicalId
    dataset_version: NonBlankStr
    dataset_content_hash: ContentHash
    dataset_ref: RelativeRef
    dataset_ref_hash: ContentHash
    split_ref: RelativeRef
    split_hash: ContentHash
    preprocess_ref: RelativeRef
    preprocess_hash: ContentHash
    method: TrainingMethod
    method_config_ref: RelativeRef
    method_config_hash: ContentHash
    seed: FixtureSeed
    budget: TrainingBudgetSpec
    command: TrainingCommandSpec
    manifest_ref: RelativeRef
    log_ref: RelativeRef
    metrics_ref: RelativeRef
    predictions_ref: RelativeRef
    capability: Literal["SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"] = TRAINING_CAPABILITY
    real_pytorch_training: Literal[False] = False
    fixture_labeled: Literal[True] = True
    synthetic_data_labeled: Literal[True] = True
    public_or_synthetic_data_only: Literal[True] = True
    shell_used: Literal[False] = False
    network_used: Literal[False] = False
    dependency_install_used: Literal[False] = False
    unknown_checkpoint_loaded: Literal[False] = False
    started_at: UTCDateTime
    finished_at: UTCDateTime

    @model_validator(mode="after")
    def _artifact_contract(self) -> Self:
        prefix = f"runs/{self.run_id}"
        expected = {
            "manifest_ref": f"{prefix}/manifest.json",
            "log_ref": f"{prefix}/train.log",
            "metrics_ref": f"{prefix}/metrics.json",
            "predictions_ref": f"{prefix}/predictions.json",
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"{field} is not canonical for this run")
        if self.command.run_id != self.run_id or self.command.role != self.role:
            raise ValueError("manifest command does not target this run")
        if self.role == "BASELINE" and (
            self.candidate_patch_revision is not None or self.candidate_patch_hash is not None
        ):
            raise ValueError("baseline manifest cannot claim the candidate patch")
        if self.role == "CANDIDATE" and (
            self.candidate_patch_revision is None or self.candidate_patch_hash is None
        ):
            raise ValueError("candidate manifest requires exact patch provenance")
        return self


class TrainingRunResult(TrainingModel):
    """Small artifact-reference result returned by the local training boundary."""

    schema_version: Literal["1"] = "1"
    run_id: CanonicalId
    role: RunRole
    spec_hash: ContentHash
    manifest_ref: RelativeRef
    log_ref: RelativeRef
    metrics_ref: RelativeRef
    predictions_ref: RelativeRef
    capability: Literal["SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"] = TRAINING_CAPABILITY
    real_pytorch_training: Literal[False] = False
    reused_existing: bool = False


class TrainingWorkflowRecord(TrainingModel):
    """Canonical local evidence index for one training workflow."""

    schema_version: Literal["1"] = "1"
    workflow_id: CanonicalId
    request_id: CanonicalId
    adaptation_workflow_id: CanonicalId
    status: TrainingStatus
    revision: Annotated[StrictInt, Field(ge=1)] | None = None
    current_spec_ref: RelativeRef | None = None
    current_spec_hash: ContentHash | None = None
    submission_id: CanonicalId | None = None
    gate_id: CanonicalId | None = None
    approval_id: CanonicalId | None = None
    reviews: list[TrainingReviewRecord] = Field(default_factory=list)
    pending_edit: PendingTrainingEdit | None = None
    baseline_result: TrainingRunResult | None = None
    candidate_result: TrainingRunResult | None = None
    cancellation_point: Literal["BEFORE_BASELINE", "BETWEEN_RUNS"] | None = None
    failure: StructuredFailure | None = None
    capability: Literal["SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"] = TRAINING_CAPABILITY
    real_pytorch_training: Literal[False] = False
    fixture_labeled: Literal[True] = True
    synthetic_data_labeled: Literal[True] = True
    created_at: UTCDateTime
    updated_at: UTCDateTime

    @model_validator(mode="after")
    def _status_evidence(self) -> Self:
        frozen_fields = (
            self.revision,
            self.current_spec_ref,
            self.current_spec_hash,
            self.submission_id,
        )
        if self.status not in {"INPUT_VALIDATED", "FAILED"} and any(
            value is None for value in frozen_fields
        ):
            raise ValueError("post-validation training records require a frozen spec identity")
        if self.status == "AWAITING_APPROVAL" and self.gate_id is None:
            raise ValueError("awaiting approval requires an exact Gate ID")
        if self.status == "EDIT_REQUESTED" and self.pending_edit is None:
            raise ValueError("edit requested status requires sanitized pending edits")
        if self.status == "SUCCEEDED" and (
            self.baseline_result is None or self.candidate_result is None
        ):
            raise ValueError("successful training requires both run results")
        if self.status == "CANCELLED" and self.cancellation_point is None:
            raise ValueError("cancelled training requires a cancellation point")
        if self.status == "FAILED" and self.failure is None:
            raise ValueError("failed training requires a structured failure")
        if self.status != "FAILED" and self.failure is not None:
            raise ValueError("only failed training records may carry a failure")
        return self


__all__ = [
    "TRAINING_CAPABILITY",
    "TRAINING_ENTRYPOINT_ID",
    "TRAINING_ENTRYPOINT_REF",
    "CanonicalId",
    "EpochLoss",
    "FrozenRunSpec",
    "FrozenTrainingSpec",
    "LossPoint",
    "P4GitCommitSha",
    "PendingTrainingEdit",
    "PredictionItem",
    "RelativeRef",
    "RunRole",
    "TrainingBudgetSpec",
    "TrainingCommandSpec",
    "TrainingEdit",
    "TrainingInput",
    "TrainingMetrics",
    "TrainingModel",
    "TrainingPredictions",
    "TrainingReviewRecord",
    "TrainingRunManifest",
    "TrainingRunResult",
    "TrainingWorkflowRecord",
    "canonical_json_bytes",
    "content_hash",
]
