"""Strict evaluation-owned models for deterministic single-pair evaluation."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Annotated, Literal, Self, TypedDict, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from vision_research_ops.domain import ContentHash, NonBlankStr, UnitInterval

from .training_models import RunRole, TrainingMethod

EVALUATION_CAPABILITY: Literal["DETERMINISTIC_SINGLE_PAIR_FIXTURE_EVALUATION"] = (
    "DETERMINISTIC_SINGLE_PAIR_FIXTURE_EVALUATION"
)
EVALUATION_POLICY_REF: Literal["fixtures/evaluation/single_pair_policy.json"] = (
    "fixtures/evaluation/single_pair_policy.json"
)
METRIC_IMPLEMENTATION_VERSION: Literal["classification-metrics-v1"] = "classification-metrics-v1"

EvaluationConclusion = Literal[
    "IMPROVED",
    "NO_CLEAR_IMPROVEMENT",
    "REGRESSED",
    "INVALID",
]
EvaluationStatus = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]
CheckStatus = Literal["PASS", "FAIL", "NOT_EVALUATED"]

ReasonCode = Literal[
    "TRAINING_WORKFLOW_UNAVAILABLE",
    "TRAINING_WORKFLOW_SCHEMA_INVALID",
    "TRAINING_WORKFLOW_NOT_SUCCEEDED",
    "TRAINING_PAIR_INCOMPLETE",
    "TRAINING_APPROVAL_PROVENANCE_INVALID",
    "FROZEN_SPEC_UNAVAILABLE",
    "FROZEN_SPEC_SCHEMA_INVALID",
    "FROZEN_SPEC_IDENTITY_MISMATCH",
    "TRAINING_ARTIFACT_MISSING",
    "TRAINING_LOG_INVALID",
    "MANIFEST_SCHEMA_INVALID",
    "METRICS_SCHEMA_INVALID",
    "PREDICTIONS_SCHEMA_INVALID",
    "RUN_ARTIFACT_IDENTITY_MISMATCH",
    "DATASET_FIXTURE_INVALID",
    "DATASET_MISMATCH",
    "SPLIT_FIXTURE_INVALID",
    "SPLIT_MISMATCH",
    "PREPROCESS_MISMATCH",
    "SEED_MISMATCH",
    "BUDGET_MISMATCH",
    "LABEL_VOCABULARY_MISMATCH",
    "TEST_SAMPLE_SET_MISMATCH",
    "TEST_SAMPLE_TRUTH_MISMATCH",
    "PREDICTED_LABEL_SCORE_MISMATCH",
    "TRAINING_METRIC_PREDICTION_MISMATCH",
    "SEVERE_CLASS_RECALL_REGRESSION",
    "MACRO_F1_REGRESSION",
    "PRACTICAL_MACRO_F1_IMPROVEMENT",
    "IMPROVEMENT_RULE_NOT_MET",
]

Limitation = Literal[
    "single_synthetic_fixture_pair",
    "single_seed_no_statistical_inference",
    "no_bootstrap_or_significance_testing",
    "not_real_sem_or_company_data",
    "not_evidence_of_production_or_business_improvement",
]
FIXED_LIMITATIONS: tuple[Limitation, ...] = (
    "single_synthetic_fixture_pair",
    "single_seed_no_statistical_inference",
    "no_bootstrap_or_significance_testing",
    "not_real_sem_or_company_data",
    "not_evidence_of_production_or_business_improvement",
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
        raise ValueError("references must be canonical POSIX relative paths")
    return value


type CanonicalId = Annotated[
    StrictStr,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]
type RelativeRef = Annotated[StrictStr, AfterValidator(_safe_relative_ref)]
type ClassId = Annotated[StrictInt, Field(ge=0, le=3)]
type SignedMetricDelta = Annotated[
    StrictFloat,
    Field(ge=-1.0, le=1.0, allow_inf_nan=False),
]
type SignedCount = Annotated[StrictInt, Field(ge=-24, le=24)]


class EvaluationModel(BaseModel):
    """Strict finite JSON base for all evaluation-owned persisted records."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


def canonical_json_bytes(value: object) -> bytes:
    """Encode one JSON value in the canonical evaluation artifact representation."""
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
    """Return a lowercase SHA-256 content identifier."""
    return f"sha256:{sha256(data).hexdigest()}"


class EvaluationPolicy(EvaluationModel):
    """Pre-registered fixture policy; thresholds cannot be changed after results."""

    schema_version: Literal["1"] = "1"
    policy_id: Literal["p5-single-pair-fixture-v1"]
    dataset_id: Literal["dataset-synthetic-sem-1"]
    dataset_version: Literal["synthetic-sem-v1"]
    label_ids: list[ClassId] = Field(min_length=4, max_length=4)
    label_names: list[NonBlankStr] = Field(min_length=4, max_length=4)
    severe_class_id: ClassId
    primary_metric: Literal["macro_f1"] = "macro_f1"
    minimum_practical_delta: Annotated[
        StrictFloat,
        Field(ge=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.01
    max_severe_recall_drop: Annotated[
        StrictFloat,
        Field(ge=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.0
    probability_sum_tolerance: Annotated[
        StrictFloat,
        Field(gt=0.0, le=1e-06, allow_inf_nan=False),
    ] = 1e-09
    metric_implementation_version: Literal["classification-metrics-v1"] = (
        METRIC_IMPLEMENTATION_VERSION
    )
    conclusion_policy_version: Literal["single-pair-threshold-v1"] = "single-pair-threshold-v1"

    @model_validator(mode="after")
    def _fixed_fixture_vocabulary(self) -> Self:
        if self.label_ids != [0, 1, 2, 3]:
            raise ValueError("the fixture label IDs must remain ordered as 0,1,2,3")
        if len(set(self.label_names)) != 4:
            raise ValueError("fixture label names must be unique")
        if self.severe_class_id != 3:
            raise ValueError("the pre-registered fixture severe_class_id is fixed at 3")
        if self.minimum_practical_delta != 0.01:
            raise ValueError("minimum_practical_delta is fixed before evaluation")
        if self.max_severe_recall_drop != 0.0:
            raise ValueError("max_severe_recall_drop is fixed before evaluation")
        if self.probability_sum_tolerance != 1e-09:
            raise ValueError("probability_sum_tolerance is fixed for training artifacts")
        return self


class EvaluationInitialInput(EvaluationModel):
    """Small identifier-only input for one evaluation graph thread."""

    schema_version: Literal["1"] = "1"
    workflow_id: CanonicalId
    thread_id: CanonicalId
    request_id: CanonicalId
    training_workflow_id: CanonicalId


class EvaluationFailure(EvaluationModel):
    """Sanitized graph infrastructure/configuration failure."""

    code: CanonicalId
    message: NonBlankStr


class EvaluationState(TypedDict, total=False):
    """Checkpoint-safe evaluation state containing no predictions or metric matrices."""

    schema_version: Literal["1"]
    workflow_id: str
    thread_id: str
    request_id: str
    training_workflow_id: str
    status: EvaluationStatus
    route: str | None
    conclusion: EvaluationConclusion | None
    evaluation_id: str | None
    evaluation_ref: str | None
    report_ref: str | None
    last_error: dict[str, str] | None


def create_evaluation_state(
    input_data: EvaluationInitialInput | dict[str, object],
) -> EvaluationState:
    """Validate and create the minimal state accepted by the evaluation graph."""
    initial = (
        input_data
        if isinstance(input_data, EvaluationInitialInput)
        else EvaluationInitialInput.model_validate(input_data)
    )
    return {
        "schema_version": "1",
        "workflow_id": initial.workflow_id,
        "thread_id": initial.thread_id,
        "request_id": initial.request_id,
        "training_workflow_id": initial.training_workflow_id,
        "status": "PENDING",
        "route": None,
        "conclusion": None,
        "evaluation_id": None,
        "evaluation_ref": None,
        "report_ref": None,
        "last_error": None,
    }


def evaluation_state_as_jsonable(state: EvaluationState) -> dict[str, object]:
    """Return a plain JSON-safe copy and reject non-finite/custom state values."""
    payload = cast(dict[str, object], dict(state))
    return cast(dict[str, object], json.loads(json.dumps(payload, allow_nan=False)))


class DatasetSample(EvaluationModel):
    """One pseudonymous label in the controlled dataset recipe."""

    group_id: CanonicalId
    label: ClassId
    noise_seed: StrictInt
    sample_id: CanonicalId


class DatasetFixture(EvaluationModel):
    """evaluation read model for the frozen synthetic dataset label vocabulary."""

    schema_version: Literal["1"]
    dataset_id: CanonicalId
    dataset_version: NonBlankStr
    fixture_kind: Literal["SYNTHETIC_SEM_IMAGE_RECIPE"]
    generator: dict[StrictStr, StrictInt]
    labels: list[NonBlankStr] = Field(min_length=4, max_length=4)
    samples: list[DatasetSample] = Field(min_length=1, max_length=64)
    synthetic_data_labeled: Literal[True]

    @model_validator(mode="after")
    def _unique_fixture_samples(self) -> Self:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("dataset label names must be unique")
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("dataset sample IDs must be unique")
        return self


class SplitFixture(EvaluationModel):
    """evaluation read model for the exact group-holdout test sample IDs."""

    schema_version: Literal["1"]
    dataset_id: CanonicalId
    group_key: CanonicalId
    group_overlap: Literal[False]
    split_kind: Literal["FIXED_GROUP_HOLDOUT"]
    train: list[CanonicalId] = Field(min_length=1, max_length=64)
    validation: list[CanonicalId] = Field(min_length=1, max_length=64)
    test: list[CanonicalId] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _disjoint_unique_splits(self) -> Self:
        combined = [*self.train, *self.validation, *self.test]
        if len(set(combined)) != len(combined):
            raise ValueError("split sample IDs must be unique across all partitions")
        return self


class ArtifactDigest(EvaluationModel):
    """Observed hash or explicit missing state for one evaluation input artifact."""

    name: CanonicalId
    ref: RelativeRef
    status: Literal["HASHED", "MISSING"]
    content_hash: ContentHash | None = None
    size_bytes: Annotated[StrictInt, Field(ge=0, le=1_048_576)] | None = None

    @model_validator(mode="after")
    def _status_matches_hash(self) -> Self:
        has_evidence = self.content_hash is not None and self.size_bytes is not None
        if (self.status == "HASHED") != has_evidence:
            raise ValueError("artifact status, content_hash, and size_bytes must agree")
        if self.status == "MISSING" and (
            self.content_hash is not None or self.size_bytes is not None
        ):
            raise ValueError("missing artifacts cannot contain digest evidence")
        return self


class ComparabilityCheck(EvaluationModel):
    """Fixed-name PASS/FAIL/NOT_EVALUATED evidence for one pair invariant."""

    name: Literal[
        "training_workflow_complete",
        "frozen_spec_integrity",
        "artifact_schema_and_identity",
        "dataset_version_content",
        "split",
        "preprocess",
        "seed",
        "budget",
        "label_vocabulary",
        "test_samples_and_truth",
    ]
    status: CheckStatus


class ComparabilityRecord(EvaluationModel):
    """Validity gate recorded before any result conclusion is allowed."""

    status: Literal["VALID", "INVALID"]
    checks: list[ComparabilityCheck] = Field(min_length=10, max_length=10)

    @model_validator(mode="after")
    def _fixed_complete_checklist(self) -> Self:
        expected = [
            "training_workflow_complete",
            "frozen_spec_integrity",
            "artifact_schema_and_identity",
            "dataset_version_content",
            "split",
            "preprocess",
            "seed",
            "budget",
            "label_vocabulary",
            "test_samples_and_truth",
        ]
        if [check.name for check in self.checks] != expected:
            raise ValueError("comparability checks must use the fixed ordered checklist")
        has_failure = any(check.status == "FAIL" for check in self.checks)
        if (self.status == "INVALID") != has_failure:
            raise ValueError("comparability status must match the fixed check results")
        if self.status == "VALID" and any(check.status != "PASS" for check in self.checks):
            raise ValueError("valid comparability requires every check to pass")
        return self


class RunProvenance(EvaluationModel):
    """training run and method identity retained in the evaluation result."""

    run_id: CanonicalId
    role: RunRole
    method: TrainingMethod
    method_config_ref: RelativeRef
    method_config_hash: ContentHash
    manifest_ref: RelativeRef
    metrics_ref: RelativeRef
    predictions_ref: RelativeRef


class TrainingProvenance(EvaluationModel):
    """Available training spec/run/commit/patch provenance, including invalid paths."""

    training_workflow_id: CanonicalId
    training_workflow_ref: RelativeRef
    frozen_spec_ref: RelativeRef | None = None
    frozen_spec_hash: ContentHash | None = None
    base_commit_sha: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{40}$")] | None = None
    patch_revision: Annotated[StrictInt, Field(ge=1)] | None = None
    patch_hash: ContentHash | None = None
    adaptation_workflow_id: CanonicalId | None = None
    adaptation_result_ref: RelativeRef | None = None
    patch_manifest_ref: RelativeRef | None = None
    smoke_result_ref: RelativeRef | None = None
    p3_approval_id: CanonicalId | None = None
    p4_approval_id: CanonicalId | None = None
    baseline: RunProvenance | None = None
    candidate: RunProvenance | None = None
    training_capability: Literal["SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"] | None = None
    real_pytorch_training: Literal[False] = False


class ClassMetrics(EvaluationModel):
    """Deterministic one-vs-rest metrics for one frozen label ID."""

    label_id: ClassId
    label_name: NonBlankStr
    precision: UnitInterval
    recall: UnitInterval
    f1: UnitInterval
    support: Annotated[StrictInt, Field(ge=0, le=24)]


class MetricSet(EvaluationModel):
    """Complete fixed-label classification result for one run."""

    implementation_version: Literal["classification-metrics-v1"] = METRIC_IMPLEMENTATION_VERSION
    label_order: list[ClassId] = Field(min_length=4, max_length=4)
    per_class: list[ClassMetrics] = Field(min_length=4, max_length=4)
    macro_f1: UnitInterval
    balanced_accuracy: UnitInterval
    accuracy: UnitInterval
    confusion_matrix: list[list[Annotated[StrictInt, Field(ge=0, le=24)]]] = Field(
        min_length=4,
        max_length=4,
    )
    sample_count: Annotated[StrictInt, Field(ge=1, le=24)]

    @model_validator(mode="after")
    def _fixed_metric_shape(self) -> Self:
        if self.label_order != [0, 1, 2, 3]:
            raise ValueError("metric label order must remain 0,1,2,3")
        if [item.label_id for item in self.per_class] != self.label_order:
            raise ValueError("per-class metrics must follow label_order")
        if len(self.confusion_matrix) != 4 or any(len(row) != 4 for row in self.confusion_matrix):
            raise ValueError("confusion matrix must be exactly 4x4")
        if sum(sum(row) for row in self.confusion_matrix) != self.sample_count:
            raise ValueError("confusion matrix total must equal sample_count")
        if sum(item.support for item in self.per_class) != self.sample_count:
            raise ValueError("per-class support must equal sample_count")
        return self


class ClassMetricDelta(EvaluationModel):
    """Candidate-minus-baseline per-class metric differences."""

    label_id: ClassId
    precision: SignedMetricDelta
    recall: SignedMetricDelta
    f1: SignedMetricDelta
    support: SignedCount


class MetricDelta(EvaluationModel):
    """Candidate-minus-baseline differences for every recorded metric."""

    label_order: list[ClassId] = Field(min_length=4, max_length=4)
    per_class: list[ClassMetricDelta] = Field(min_length=4, max_length=4)
    macro_f1: SignedMetricDelta
    balanced_accuracy: SignedMetricDelta
    accuracy: SignedMetricDelta
    severe_class_id: ClassId
    severe_class_recall: SignedMetricDelta
    confusion_matrix: list[list[SignedCount]] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def _fixed_delta_shape(self) -> Self:
        if self.label_order != [0, 1, 2, 3]:
            raise ValueError("delta label order must remain 0,1,2,3")
        if [item.label_id for item in self.per_class] != self.label_order:
            raise ValueError("per-class deltas must follow label_order")
        if len(self.confusion_matrix) != 4 or any(len(row) != 4 for row in self.confusion_matrix):
            raise ValueError("delta confusion matrix must be exactly 4x4")
        return self


class EvaluationResult(EvaluationModel):
    """Canonical evaluation result from which the Markdown report is exclusively rendered."""

    schema_version: Literal["1"] = "1"
    workflow_id: CanonicalId
    evaluation_id: CanonicalId
    evaluation_capability: Literal["DETERMINISTIC_SINGLE_PAIR_FIXTURE_EVALUATION"] = (
        EVALUATION_CAPABILITY
    )
    llm_used: Literal[False] = False
    real_company_evaluation: Literal[False] = False
    real_sem_evaluation: Literal[False] = False
    single_pair: Literal[True] = True
    synthetic_or_public_data_only: Literal[True] = True
    limitations: list[Limitation] = Field(min_length=5, max_length=5)
    policy_ref: Literal["fixtures/evaluation/single_pair_policy.json"] = EVALUATION_POLICY_REF
    policy_hash: ContentHash
    policy: EvaluationPolicy
    provenance: TrainingProvenance
    input_artifacts: list[ArtifactDigest] = Field(min_length=2, max_length=16)
    comparability: ComparabilityRecord
    baseline_metrics: MetricSet | None
    candidate_metrics: MetricSet | None
    deltas: MetricDelta | None
    conclusion: EvaluationConclusion
    reason_codes: list[ReasonCode] = Field(min_length=1, max_length=16)
    evaluation_ref: RelativeRef
    report_ref: RelativeRef

    @field_validator("limitations")
    @classmethod
    def _fixed_limitations(cls, value: list[Limitation]) -> list[Limitation]:
        if tuple(value) != FIXED_LIMITATIONS:
            raise ValueError("evaluation limitations are fixed and ordered")
        return value

    @model_validator(mode="after")
    def _result_invariants(self) -> Self:
        if self.evaluation_ref != f"reports/{self.workflow_id}/evaluation.json":
            raise ValueError("evaluation_ref must be canonical for the workflow")
        if self.report_ref != f"reports/{self.workflow_id}/report.md":
            raise ValueError("report_ref must be canonical for the workflow")
        if len({item.name for item in self.input_artifacts}) != len(self.input_artifacts):
            raise ValueError("input artifact names must be unique")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("reason codes must be unique")
        metrics = (self.baseline_metrics, self.candidate_metrics, self.deltas)
        if self.conclusion == "INVALID":
            if self.comparability.status != "INVALID" or any(item is not None for item in metrics):
                raise ValueError("INVALID results require failed comparability and no metrics")
        else:
            if self.comparability.status != "VALID" or any(item is None for item in metrics):
                raise ValueError("valid conclusions require complete comparable metrics")
        return self


__all__ = [
    "EVALUATION_CAPABILITY",
    "EVALUATION_POLICY_REF",
    "FIXED_LIMITATIONS",
    "ArtifactDigest",
    "CanonicalId",
    "CheckStatus",
    "ClassMetricDelta",
    "ClassMetrics",
    "ComparabilityCheck",
    "ComparabilityRecord",
    "DatasetFixture",
    "EvaluationConclusion",
    "EvaluationFailure",
    "EvaluationInitialInput",
    "EvaluationPolicy",
    "EvaluationResult",
    "EvaluationState",
    "MetricDelta",
    "MetricSet",
    "ReasonCode",
    "RelativeRef",
    "RunProvenance",
    "SplitFixture",
    "TrainingProvenance",
    "canonical_json_bytes",
    "content_hash",
    "create_evaluation_state",
    "evaluation_state_as_jsonable",
]
