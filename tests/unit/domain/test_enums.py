"""Enumeration tests: legal construction, illegal values, JSON serialization."""

from __future__ import annotations

import json

import pydantic
import pytest

from vision_research_ops.domain.enums import (
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
    WorkflowPhase,
    WorkflowStatus,
)

pytestmark = pytest.mark.unit

ENUM_CASES = [
    (TaskType, {"IMAGE_CLASSIFICATION"}),
    (
        WorkflowPhase,
        {
            "REQUEST_VALIDATION",
            "PAPER_RETRIEVAL",
            "CANDIDATE_RANKING",
            "AWAITING_CANDIDATE_SELECTION",
            "REPOSITORY_RESOLUTION",
            "REPOSITORY_ANALYSIS",
            "AWAITING_INGEST_APPROVAL",
            "ADAPTATION_PLANNING",
            "PATCH_GENERATION",
            "PATCH_VALIDATION",
            "AWAITING_RUN_APPROVAL",
            "EXPERIMENT_FREEZE",
            "RUN_SUBMISSION",
            "RUN_MONITORING",
            "EVALUATION",
            "REPORTING",
            "COMPLETED",
            "REJECTED",
            "FAILED",
        },
    ),
    (
        WorkflowStatus,
        {
            "PENDING",
            "RUNNING",
            "WAITING_FOR_HUMAN",
            "WAITING_FOR_EXTERNAL",
            "SUCCEEDED",
            "FAILED",
            "CANCELLED",
            "REJECTED",
        },
    ),
    (
        GateKind,
        {
            "CANDIDATE_SELECTION",
            "REPOSITORY_INGEST",
            "PATCH_ACCEPTANCE",
            "RUN_SUBMISSION",
            "CANCELLATION",
        },
    ),
    (ApprovalDecision, {"APPROVE", "EDIT", "REJECT"}),
    (
        CodeLinkConfidence,
        {"OFFICIAL_HIGH", "PROBABLE_MEDIUM", "UNVERIFIED", "CONTRADICTED"},
    ),
    (
        LicenseStatus,
        {"ALLOWLISTED", "REVIEW_REQUIRED", "DENIED", "UNKNOWN"},
    ),
    (
        ValidationStatus,
        {"PENDING", "RUNNING", "PASSED", "FAILED", "SKIPPED", "BLOCKED"},
    ),
    (
        EvaluationConclusion,
        {"IMPROVED", "NO_CLEAR_IMPROVEMENT", "REGRESSED", "INVALID"},
    ),
    (
        ValidationStage,
        {
            "STATIC_POLICY",
            "ENVIRONMENT_BUILD",
            "IMPORT",
            "ONE_BATCH",
            "BOUNDED_OVERFIT",
            "SHORT_TRAIN",
            "OUTPUT_CONTRACT",
        },
    ),
    (
        RunStatus,
        {
            "CREATED",
            "SUBMITTING",
            "QUEUED",
            "RUNNING",
            "SUCCEEDED",
            "FAILED",
            "CANCEL_REQUESTED",
            "CANCELLED",
            "TIMED_OUT",
            "LOST",
            "INVALID",
        },
    ),
    (
        ArtifactKind,
        {
            "PROVIDER_SNAPSHOT",
            "PAPER_METADATA",
            "PAPER_DOCUMENT",
            "REPOSITORY_ARCHIVE",
            "REPOSITORY_ANALYSIS",
            "ADAPTATION_PLAN",
            "PATCH",
            "BUILD_LOG",
            "VALIDATION_LOG",
            "EXPERIMENT_CONFIG",
            "RUN_MANIFEST",
            "TRAIN_LOG",
            "METRICS",
            "PREDICTIONS",
            "MODEL",
            "EVALUATION_DATA",
            "REPORT",
            "AUDIT_EXPORT",
        },
    ),
    (NetworkPolicy, {"NONE", "ALLOWLIST"}),
    (
        SplitStrategy,
        {"TIME_EXTRAPOLATION", "GROUP_HOLDOUT", "DOMAIN_HOLDOUT", "SAMPLE_STRATIFIED"},
    ),
    (SeverityLevel, {"LOW", "MEDIUM", "HIGH", "CRITICAL"}),
    (PatchOperationType, {"ADD", "REPLACE", "REMOVE"}),
]


@pytest.mark.parametrize(("enum_cls", "expected"), ENUM_CASES)
def test_enum_members_match_documentation(enum_cls: type, expected: set[str]) -> None:
    """Enum members must exactly match the documented value set."""
    actual = {member.value for member in enum_cls}
    assert actual == expected


@pytest.mark.parametrize("enum_cls", [case[0] for case in ENUM_CASES])
def test_enum_serializes_as_uppercase_string(enum_cls: type) -> None:
    """Each enum member must serialize to its uppercase string value."""
    for member in enum_cls:
        assert json.dumps(member) == json.dumps(member.value)
        assert json.loads(json.dumps(member)) == member.value
        assert str(member) == member.value


@pytest.mark.parametrize("enum_cls", [case[0] for case in ENUM_CASES])
def test_enum_is_str_subclass(enum_cls: type) -> None:
    """Enums must be usable as plain strings in serialized payloads."""
    for member in enum_cls:
        assert isinstance(member, str)


class _EnumHolder(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")


@pytest.mark.parametrize("enum_cls", [case[0] for case in ENUM_CASES])
def test_enum_rejects_unknown_value(enum_cls: type) -> None:
    """Pydantic must reject values outside the enum even if well-formed strings."""

    class _Holder(_EnumHolder):
        value: enum_cls  # type: ignore[valid-type]

    with pytest.raises(pydantic.ValidationError):
        _Holder(value="NOT_A_REAL_MEMBER")
    with pytest.raises(pydantic.ValidationError):
        _Holder(value="")
    with pytest.raises(pydantic.ValidationError):
        _Holder(value="lowercase")


def test_enum_rejects_wrong_type() -> None:
    with pytest.raises(ValueError):
        TaskType(123)  # type: ignore[arg-type]
