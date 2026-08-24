"""Domain enums for SEM Research Agent.

All enumerations are serialized as stable uppercase strings, matching
``docs/03_DOMAIN_MODEL_AND_STATE.md`` section 3.
"""

from enum import StrEnum


class TaskType(StrEnum):
    """Task family supported by the MVP (only image classification)."""

    IMAGE_CLASSIFICATION = "IMAGE_CLASSIFICATION"


class WorkflowPhase(StrEnum):
    """Phases of the research workflow lifecycle."""

    REQUEST_VALIDATION = "REQUEST_VALIDATION"
    PAPER_RETRIEVAL = "PAPER_RETRIEVAL"
    CANDIDATE_RANKING = "CANDIDATE_RANKING"
    AWAITING_CANDIDATE_SELECTION = "AWAITING_CANDIDATE_SELECTION"
    REPOSITORY_RESOLUTION = "REPOSITORY_RESOLUTION"
    REPOSITORY_ANALYSIS = "REPOSITORY_ANALYSIS"
    AWAITING_INGEST_APPROVAL = "AWAITING_INGEST_APPROVAL"
    ADAPTATION_PLANNING = "ADAPTATION_PLANNING"
    PATCH_GENERATION = "PATCH_GENERATION"
    PATCH_VALIDATION = "PATCH_VALIDATION"
    AWAITING_RUN_APPROVAL = "AWAITING_RUN_APPROVAL"
    EXPERIMENT_FREEZE = "EXPERIMENT_FREEZE"
    RUN_SUBMISSION = "RUN_SUBMISSION"
    RUN_MONITORING = "RUN_MONITORING"
    EVALUATION = "EVALUATION"
    REPORTING = "REPORTING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class WorkflowStatus(StrEnum):
    """Business-view status of a workflow."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"
    WAITING_FOR_EXTERNAL = "WAITING_FOR_EXTERNAL"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class GateKind(StrEnum):
    """Human approval gate kinds."""

    CANDIDATE_SELECTION = "CANDIDATE_SELECTION"
    REPOSITORY_INGEST = "REPOSITORY_INGEST"
    PATCH_ACCEPTANCE = "PATCH_ACCEPTANCE"
    RUN_SUBMISSION = "RUN_SUBMISSION"
    CANCELLATION = "CANCELLATION"


class ApprovalDecision(StrEnum):
    """Decisions a human reviewer can emit at a gate."""

    APPROVE = "APPROVE"
    EDIT = "EDIT"
    REJECT = "REJECT"


class CodeLinkConfidence(StrEnum):
    """Confidence of a paper-to-code association."""

    OFFICIAL_HIGH = "OFFICIAL_HIGH"
    PROBABLE_MEDIUM = "PROBABLE_MEDIUM"
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTED = "CONTRADICTED"


class LicenseStatus(StrEnum):
    """Repository license policy outcome."""

    ALLOWLISTED = "ALLOWLISTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    DENIED = "DENIED"
    UNKNOWN = "UNKNOWN"


class ValidationStatus(StrEnum):
    """Result status of a validation stage."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"


class EvaluationConclusion(StrEnum):
    """Deterministic conclusion of an evaluation report."""

    IMPROVED = "IMPROVED"
    NO_CLEAR_IMPROVEMENT = "NO_CLEAR_IMPROVEMENT"
    REGRESSED = "REGRESSED"
    INVALID = "INVALID"


class ValidationStage(StrEnum):
    """Ordered stages of the smoke validation pipeline."""

    STATIC_POLICY = "STATIC_POLICY"
    ENVIRONMENT_BUILD = "ENVIRONMENT_BUILD"
    IMPORT = "IMPORT"
    ONE_BATCH = "ONE_BATCH"
    BOUNDED_OVERFIT = "BOUNDED_OVERFIT"
    SHORT_TRAIN = "SHORT_TRAIN"
    OUTPUT_CONTRACT = "OUTPUT_CONTRACT"


class RunStatus(StrEnum):
    """Lifecycle status of an experiment run."""

    CREATED = "CREATED"
    SUBMITTING = "SUBMITTING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    LOST = "LOST"
    INVALID = "INVALID"


class ArtifactKind(StrEnum):
    """Kinds of immutable artifacts stored in the artifact store."""

    PROVIDER_SNAPSHOT = "PROVIDER_SNAPSHOT"
    PAPER_METADATA = "PAPER_METADATA"
    PAPER_DOCUMENT = "PAPER_DOCUMENT"
    REPOSITORY_ARCHIVE = "REPOSITORY_ARCHIVE"
    REPOSITORY_ANALYSIS = "REPOSITORY_ANALYSIS"
    ADAPTATION_PLAN = "ADAPTATION_PLAN"
    PATCH = "PATCH"
    BUILD_LOG = "BUILD_LOG"
    VALIDATION_LOG = "VALIDATION_LOG"
    EXPERIMENT_CONFIG = "EXPERIMENT_CONFIG"
    RUN_MANIFEST = "RUN_MANIFEST"
    TRAIN_LOG = "TRAIN_LOG"
    METRICS = "METRICS"
    PREDICTIONS = "PREDICTIONS"
    MODEL = "MODEL"
    EVALUATION_DATA = "EVALUATION_DATA"
    REPORT = "REPORT"
    AUDIT_EXPORT = "AUDIT_EXPORT"


class NetworkPolicy(StrEnum):
    """Network capability requested by a resource specification."""

    NONE = "NONE"
    ALLOWLIST = "ALLOWLIST"


class SplitStrategy(StrEnum):
    """Supported deterministic split-policy families."""

    TIME_EXTRAPOLATION = "TIME_EXTRAPOLATION"
    GROUP_HOLDOUT = "GROUP_HOLDOUT"
    DOMAIN_HOLDOUT = "DOMAIN_HOLDOUT"
    SAMPLE_STRATIFIED = "SAMPLE_STRATIFIED"


class SeverityLevel(StrEnum):
    """Unified severity vocabulary for labels, gaps, and findings."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PatchOperationType(StrEnum):
    """SEM Research Agent's structured edit operations (not a full RFC 6902 document)."""

    ADD = "ADD"
    REPLACE = "REPLACE"
    REMOVE = "REMOVE"


__all__ = [
    "ApprovalDecision",
    "ArtifactKind",
    "CodeLinkConfidence",
    "EvaluationConclusion",
    "GateKind",
    "LicenseStatus",
    "NetworkPolicy",
    "PatchOperationType",
    "RunStatus",
    "SeverityLevel",
    "SplitStrategy",
    "TaskType",
    "ValidationStage",
    "ValidationStatus",
    "WorkflowPhase",
    "WorkflowStatus",
]
