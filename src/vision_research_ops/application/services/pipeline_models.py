"""Strict integrated state, evidence summary, and write-once local persistence."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, TypedDict, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from vision_research_ops.domain import JsonValue

PIPELINE_CAPABILITY: Literal["OFFLINE_FIXTURE_RESEARCH_TO_EVALUATION_LANGGRAPH_PIPELINE_SAMPLE"] = (
    "OFFLINE_FIXTURE_RESEARCH_TO_EVALUATION_LANGGRAPH_PIPELINE_SAMPLE"
)
PIPELINE_STAGES = ("research", "repository", "adaptation", "training", "evaluation")

type PipelineStageName = Literal[
    "research",
    "repository",
    "adaptation",
    "training",
    "evaluation",
]
type PipelineScenario = Literal["happy", "smoke-failure"]
type PipelineStatus = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "STOPPED"]
type PipelineStageStatus = Literal["NOT_RUN", "SUCCEEDED", "FAILED", "STOPPED"]
type PipelineDecisionMode = Literal[
    "interactive",
    "auto_approve_sample",
    "explicit_ordered",
]
type PipelineDecisionName = Literal["approve", "edit", "reject"]

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"(?i)(?:^|[\s\"'])[a-z]:[\\/]")
_SENSITIVE_MARKERS = (
    "authorization:",
    "dashscope_api_key",
    "api_key=",
    "bearer ",
    "private key",
)


def _safe_id(value: str) -> str:
    if _SAFE_ID_RE.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError("pipeline identifier must be one safe local component")
    return value


def _relative_ref(value: str) -> str:
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or value.startswith("/")
        or ":" in value
        or "%" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("pipeline artifact reference must be canonical and relative")
    return value


type PipelineId = Annotated[StrictStr, AfterValidator(_safe_id)]
type RelativeRef = Annotated[StrictStr, AfterValidator(_relative_ref)]
type GitCommitSha = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{40}$")]
type ContentHash = Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


def canonical_json_bytes(value: object) -> bytes:
    """Encode one stable newline-terminated canonical JSON document."""
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
    """Return one lowercase SHA-256 identifier."""
    return f"sha256:{sha256(data).hexdigest()}"


def _assert_public_json(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("pipeline public JSON keys must be strings")
            _assert_public_json(item)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _assert_public_json(item)
        return
    if isinstance(value, str):
        lowered = value.casefold()
        if _WINDOWS_ABSOLUTE_RE.search(value) or any(
            marker in lowered for marker in _SENSITIVE_MARKERS
        ):
            raise ValueError("pipeline output contains an absolute path or secret marker")


class PipelineModel(BaseModel):
    """Strict JSON-only base used by all integrated records."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class PipelineFailure(PipelineModel):
    """Small sanitized reason for a failed or human-stopped pipeline run."""

    code: PipelineId
    message: Annotated[StrictStr, Field(min_length=1, max_length=240)]
    stage: PipelineStageName | None = None


class PipelineGateRecord(PipelineModel):
    """One real child interrupt and the typed decision used to resume it."""

    stage: PipelineStageName
    gate_kind: Literal[
        "CANDIDATE_SELECTION",
        "REPOSITORY_INGEST",
        "PATCH_ACCEPTANCE",
        "RUN_SUBMISSION",
    ]
    subject_id: PipelineId
    subject_revision: Annotated[StrictInt, Field(ge=1)]
    decision: Literal["APPROVE", "EDIT", "REJECT"]
    scripted_fixture_decision: StrictBool
    resume_count: Annotated[StrictInt, Field(ge=1)]
    evidence: dict[StrictStr, JsonValue] = Field(default_factory=dict)
    artifact_ref: RelativeRef | None = None

    @model_validator(mode="after")
    def _public_evidence_only(self) -> PipelineGateRecord:
        _assert_public_json(self.evidence)
        return self


class PipelineDecisionConfig(PipelineModel):
    """Canonical identity of the complete CLI decision configuration."""

    schema_version: Literal["1"] = "1"
    mode: PipelineDecisionMode
    decisions: list[PipelineDecisionName] = Field(default_factory=list, max_length=16)
    fingerprint: ContentHash

    @model_validator(mode="after")
    def _canonical_identity(self) -> PipelineDecisionConfig:
        if self.mode == "explicit_ordered":
            if not self.decisions:
                raise ValueError("explicit decision config requires an ordered decision list")
        elif self.decisions:
            raise ValueError("interactive and auto decision configs cannot carry a decision list")
        expected = pipeline_decision_config_fingerprint(
            mode=self.mode,
            decisions=self.decisions,
        )
        if self.fingerprint != expected:
            raise ValueError("decision config fingerprint does not match its canonical identity")
        return self


def pipeline_decision_config_fingerprint(
    *,
    mode: PipelineDecisionMode,
    decisions: Sequence[PipelineDecisionName],
) -> str:
    """Hash the canonical mode plus the complete ordered decision list."""
    return content_hash(
        canonical_json_bytes(
            {
                "decisions": list(decisions),
                "mode": mode,
                "schema_version": "1",
            }
        )
    )


def create_pipeline_decision_config(
    *,
    mode: PipelineDecisionMode,
    decisions: Sequence[PipelineDecisionName] = (),
) -> PipelineDecisionConfig:
    """Build one validated canonical decision configuration."""
    ordered = list(decisions)
    return PipelineDecisionConfig(
        mode=mode,
        decisions=ordered,
        fingerprint=pipeline_decision_config_fingerprint(
            mode=mode,
            decisions=ordered,
        ),
    )


class PipelineStageRecord(PipelineModel):
    """Compact status and artifact refs for one existing child Agent."""

    stage: PipelineStageName
    workflow_id: PipelineId
    status: PipelineStageStatus
    child_status: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    resume_count: Annotated[StrictInt, Field(ge=0)] = 0
    artifact_refs: list[RelativeRef] = Field(default_factory=list, max_length=24)
    failure: PipelineFailure | None = None


class PaperEvidence(PipelineModel):
    """Selected public paper and the research evidence file that supports it."""

    paper_id: PipelineId
    evidence_ref: RelativeRef


class RepositoryEvidence(PipelineModel):
    """Fixed public repository identity emitted by repository."""

    repository_id: PipelineId
    fixed_commit_sha: GitCommitSha
    profile_ref: RelativeRef


class AdaptationEvidence(PipelineModel):
    """adaptation plan, patch, and bounded Smoke references."""

    plan_ref: RelativeRef
    planner_trace_ref: RelativeRef
    planner_kind: Literal["SCRIPTED_TOOL_CALLING", "DASHSCOPE_TOOL_CALLING"]
    planner_tools: list[
        Literal[
            "inspect_repository_profile",
            "inspect_dataset_contract",
            "compare_repository_dataset",
            "validate_adaptation_plan",
        ]
    ] = Field(min_length=4, max_length=6)
    patch_ref: RelativeRef
    patch_manifest_ref: RelativeRef
    patch_hash: ContentHash
    smoke_ref: RelativeRef
    smoke_capability: Literal["FIXTURE_CONTRACT_PROBE_NO_TORCH"]
    real_pytorch_training: Literal[False] = False

    @model_validator(mode="after")
    def _complete_planner_trace(self) -> AdaptationEvidence:
        required = {
            "inspect_repository_profile",
            "inspect_dataset_contract",
            "compare_repository_dataset",
            "validate_adaptation_plan",
        }
        if not required.issubset(self.planner_tools):
            raise ValueError("adaptation evidence requires all planner tools")
        return self


class TrainingEvidence(PipelineModel):
    """training fair-pair identity and standard run artifacts."""

    workflow_ref: RelativeRef
    spec_ref: RelativeRef
    baseline_run_id: PipelineId
    candidate_run_id: PipelineId
    baseline_manifest_ref: RelativeRef
    candidate_manifest_ref: RelativeRef
    baseline_metrics_ref: RelativeRef
    candidate_metrics_ref: RelativeRef
    baseline_predictions_ref: RelativeRef
    candidate_predictions_ref: RelativeRef
    capability: Literal["SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"]
    real_pytorch_training: Literal[False] = False


class EvaluationMetricEvidence(PipelineModel):
    """Small values copied verbatim from the validated canonical evaluation result."""

    baseline_macro_f1: StrictFloat | None = None
    candidate_macro_f1: StrictFloat | None = None
    macro_f1_delta: StrictFloat | None = None
    baseline_balanced_accuracy: StrictFloat | None = None
    candidate_balanced_accuracy: StrictFloat | None = None
    balanced_accuracy_delta: StrictFloat | None = None
    severe_recall_delta: StrictFloat | None = None


class EvaluationEvidence(PipelineModel):
    """evaluation canonical JSON/report refs and its unmodified deterministic conclusion."""

    evaluation_ref: RelativeRef
    report_ref: RelativeRef
    conclusion: Literal[
        "IMPROVED",
        "NO_CLEAR_IMPROVEMENT",
        "REGRESSED",
        "INVALID",
    ]
    metrics: EvaluationMetricEvidence
    capability: Literal["DETERMINISTIC_SINGLE_PAIR_FIXTURE_EVALUATION"]
    llm_used: Literal[False] = False
    real_company_evaluation: Literal[False] = False


class PipelineSummary(PipelineModel):
    """Canonical evidence index for one complete top-level run."""

    schema_version: Literal["3"] = "3"
    capability: Literal["OFFLINE_FIXTURE_RESEARCH_TO_EVALUATION_LANGGRAPH_PIPELINE_SAMPLE"] = (
        PIPELINE_CAPABILITY
    )
    workflow_id: PipelineId
    mode: Literal["fixture"] = "fixture"
    adaptation_planner_mode: Literal["scripted", "dashscope"]
    scenario: PipelineScenario
    status: PipelineStatus
    fixture_labeled: Literal[True] = True
    scripted_fixture_decisions: StrictBool
    decision_config: PipelineDecisionConfig
    synthetic_or_public_data_only: Literal[True] = True
    real_pytorch_training: Literal[False] = False
    real_company_evaluation: Literal[False] = False
    stages: list[PipelineStageRecord] = Field(min_length=5, max_length=5)
    gates: list[PipelineGateRecord] = Field(default_factory=list, max_length=16)
    resume_count: Annotated[StrictInt, Field(ge=0)]
    paper: PaperEvidence | None = None
    repository: RepositoryEvidence | None = None
    adaptation: AdaptationEvidence | None = None
    training: TrainingEvidence | None = None
    evaluation: EvaluationEvidence | None = None
    conclusion: (
        Literal[
            "IMPROVED",
            "NO_CLEAR_IMPROVEMENT",
            "REGRESSED",
            "INVALID",
        ]
        | None
    ) = None
    limitations: list[Annotated[StrictStr, Field(min_length=1, max_length=240)]] = Field(
        min_length=3,
        max_length=8,
    )
    failure_reason: PipelineFailure | None = None
    summary_ref: RelativeRef
    created_at: datetime

    @model_validator(mode="after")
    def _consistent_summary(self) -> PipelineSummary:
        if [item.stage for item in self.stages] != list(PIPELINE_STAGES):
            raise ValueError("pipeline stages must be complete and ordered")
        if self.resume_count != sum(item.resume_count for item in self.stages):
            raise ValueError("pipeline resume_count must equal its stage totals")
        if self.resume_count != len(self.gates):
            raise ValueError("pipeline resume_count must equal its recorded Gate count")
        if any(
            gate.scripted_fixture_decision != self.scripted_fixture_decisions for gate in self.gates
        ):
            raise ValueError("pipeline Gate decision provenance must be consistent")
        decision_config_is_scripted = self.decision_config.mode != "interactive"
        if self.scripted_fixture_decisions != decision_config_is_scripted:
            raise ValueError("pipeline decision config mode must match scripted provenance")
        if self.decision_config.mode == "explicit_ordered":
            consumed = [gate.decision.casefold() for gate in self.gates]
            if consumed != self.decision_config.decisions[: len(consumed)]:
                raise ValueError("pipeline Gates must match the ordered decision config prefix")
        elif self.decision_config.mode == "auto_approve_sample" and any(
            gate.decision != "APPROVE" for gate in self.gates
        ):
            raise ValueError("auto-approved fixture Gates must all be approved")
        if self.adaptation is not None:
            expected_planner = (
                "SCRIPTED_TOOL_CALLING"
                if self.adaptation_planner_mode == "scripted"
                else "DASHSCOPE_TOOL_CALLING"
            )
            if self.adaptation.planner_kind != expected_planner:
                raise ValueError("adaptation evidence does not match planner mode")
        if self.status == "SUCCEEDED":
            if self.evaluation is None or self.conclusion != self.evaluation.conclusion:
                raise ValueError(
                    "successful pipeline summary requires the exact evaluation conclusion"
                )
            if any(
                evidence is None
                for evidence in (
                    self.paper,
                    self.repository,
                    self.adaptation,
                    self.training,
                )
            ):
                raise ValueError("successful pipeline summary requires the full evidence chain")
            required_gates = {
                "CANDIDATE_SELECTION",
                "REPOSITORY_INGEST",
                "PATCH_ACCEPTANCE",
                "RUN_SUBMISSION",
            }
            approved_gates = {gate.gate_kind for gate in self.gates if gate.decision == "APPROVE"}
            if not required_gates.issubset(approved_gates):
                raise ValueError("successful pipeline summary requires all exact Gate approvals")
            if any(item.status != "SUCCEEDED" for item in self.stages):
                raise ValueError("successful pipeline summary requires all five stages")
            if self.failure_reason is not None:
                raise ValueError("successful pipeline summary cannot carry a failure")
        elif self.failure_reason is None:
            raise ValueError("failed or stopped pipeline summary requires a reason")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("pipeline created_at must be timezone-aware")
        _assert_public_json(self.model_dump(mode="json"))
        return self


class PipelineInitialInput(PipelineModel):
    """Validated identifier-only input for the integrated top-level graph."""

    schema_version: Literal["1"] = "1"
    workflow_id: PipelineId
    thread_id: PipelineId
    scenario: PipelineScenario
    scripted_fixture_decisions: StrictBool


class PipelineState(TypedDict, total=False):
    """Small JSON-safe state; full child evidence remains in local artifacts."""

    schema_version: Literal["1"]
    workflow_id: str
    thread_id: str
    scenario: PipelineScenario
    scripted_fixture_decisions: bool
    phase: str
    status: PipelineStatus
    route: Literal["CONTINUE", "STOP", "DONE"] | None
    stage_records: list[dict[str, object]]
    gate_records: list[dict[str, object]]
    resume_count: int
    artifact_refs: dict[str, str]
    conclusion: str | None
    summary_ref: str | None
    failure_reason: dict[str, object] | None


def create_pipeline_state(
    input_data: PipelineInitialInput | dict[str, object],
) -> PipelineState:
    """Create the minimal validated top-level graph state."""
    initial = (
        input_data
        if isinstance(input_data, PipelineInitialInput)
        else PipelineInitialInput.model_validate(input_data)
    )
    return {
        "schema_version": "1",
        "workflow_id": initial.workflow_id,
        "thread_id": initial.thread_id,
        "scenario": initial.scenario,
        "scripted_fixture_decisions": initial.scripted_fixture_decisions,
        "phase": "PENDING",
        "status": "PENDING",
        "route": None,
        "stage_records": [],
        "gate_records": [],
        "resume_count": 0,
        "artifact_refs": {},
        "conclusion": None,
        "summary_ref": None,
        "failure_reason": None,
    }


def pipeline_state_as_jsonable(state: PipelineState | dict[str, object]) -> dict[str, object]:
    """Round-trip top-level state through strict ordinary JSON values."""
    payload = cast(dict[str, object], dict(state))
    _assert_public_json(payload)
    return cast(dict[str, object], json.loads(json.dumps(payload, allow_nan=False)))


class PipelineStoreError(Exception):
    """Sanitized write-once summary persistence failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class LocalPipelineSummaryStore:
    """Write or verify one canonical summary below the selected workspace var root."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        """Return the explicitly selected workspace var root."""
        return self._root

    @staticmethod
    def summary_ref(workflow_id: str) -> str:
        """Return the canonical relative summary reference."""
        return f"sample/{_safe_id(workflow_id)}/summary.json"

    def resolve_ref(self, relative_ref: str) -> Path:
        """Resolve a canonical reference and prove containment below the var root."""
        safe = _relative_ref(relative_ref)
        root = self._root.resolve()
        path = (root / Path(*safe.split("/"))).resolve()
        if not path.is_relative_to(root):
            raise PipelineStoreError("PIPELINE_ARTIFACT_REF_INVALID")
        return path

    def load_summary(self, workflow_id: str) -> PipelineSummary:
        """Load and verify exact canonical summary bytes."""
        ref = self.summary_ref(workflow_id)
        path = self.resolve_ref(ref)
        try:
            payload = path.read_bytes()
            summary = PipelineSummary.model_validate_json(payload)
        except (OSError, ValueError) as error:
            raise PipelineStoreError("PIPELINE_SUMMARY_INVALID") from error
        if (
            summary.summary_ref != ref
            or canonical_json_bytes(summary.model_dump(mode="json")) != payload
        ):
            raise PipelineStoreError("PIPELINE_SUMMARY_INVALID")
        return summary

    def write_summary(self, summary: PipelineSummary) -> bool:
        """Write once; return True only when exact existing bytes were reused."""
        validated = PipelineSummary.model_validate(summary.model_dump(mode="python"))
        expected_ref = self.summary_ref(validated.workflow_id)
        if validated.summary_ref != expected_ref:
            raise PipelineStoreError("PIPELINE_SUMMARY_INVALID")
        payload = canonical_json_bytes(validated.model_dump(mode="json"))
        path = self.resolve_ref(expected_ref)
        if path.exists():
            try:
                existing = path.read_bytes()
            except OSError as error:
                raise PipelineStoreError("PIPELINE_SUMMARY_INVALID") from error
            if existing != payload or content_hash(existing) != content_hash(payload):
                raise PipelineStoreError("PIPELINE_SUMMARY_CONFLICT")
            return True
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f"{path.suffix}.tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)
        except OSError as error:
            raise PipelineStoreError("PIPELINE_SUMMARY_WRITE_FAILED") from error
        return False


__all__ = [
    "PIPELINE_CAPABILITY",
    "PIPELINE_STAGES",
    "AdaptationEvidence",
    "EvaluationEvidence",
    "EvaluationMetricEvidence",
    "LocalPipelineSummaryStore",
    "PaperEvidence",
    "PipelineDecisionConfig",
    "PipelineDecisionMode",
    "PipelineDecisionName",
    "PipelineFailure",
    "PipelineGateRecord",
    "PipelineInitialInput",
    "PipelineScenario",
    "PipelineStageName",
    "PipelineStageRecord",
    "PipelineState",
    "PipelineStatus",
    "PipelineStoreError",
    "PipelineSummary",
    "RepositoryEvidence",
    "TrainingEvidence",
    "canonical_json_bytes",
    "content_hash",
    "create_pipeline_decision_config",
    "create_pipeline_state",
    "pipeline_decision_config_fingerprint",
    "pipeline_state_as_jsonable",
]
