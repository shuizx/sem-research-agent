"""Strict Research Agent models kept outside the stable foundation domain contract."""

from __future__ import annotations

from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from vision_research_ops.domain import (
    GenerationRecord,
    HumanText,
    JsonObject,
    NonBlankStr,
    OpaqueId,
    PaperCandidate,
    ProvenanceRef,
    QuerySpec,
    Reason,
    StrictBoolean,
    UnitInterval,
    UTCDateTime,
)

PublicAbstract = Annotated[StrictStr, Field(min_length=1, max_length=8000)]


class ResearchModel(BaseModel):
    """Strict JSON-safe base for research workflow application-owned records."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class ProblemProfile(ResearchModel):
    """Public, de-identified problem facts supplied to paper analysis."""

    schema_version: Literal["1"] = "1"
    profile_id: OpaqueId
    task_type: Literal["image_classification"]
    domain: Literal["wafer_sem_defect_classification"]
    modality: Literal["SEM_GRAYSCALE"]
    image_characteristics: list[NonBlankStr]
    data_risks: list[NonBlankStr]
    preferred_metrics: list[NonBlankStr]
    compute_budget: Literal["local_short_training"]

    def prompt_facts(self) -> JsonObject:
        """Return only public problem facts, intentionally omitting internal IDs."""
        return cast(
            JsonObject,
            {
                "task_type": self.task_type,
                "domain": self.domain,
                "modality": self.modality,
                "image_characteristics": list(self.image_characteristics),
                "data_risks": list(self.data_risks),
                "preferred_metrics": list(self.preferred_metrics),
                "compute_budget": self.compute_budget,
            },
        )


def default_sem_problem_profile() -> ProblemProfile:
    """Return the fixed pipeline problem description used by the first sample."""
    return ProblemProfile(
        profile_id="problem-wafer-sem-v1",
        task_type="image_classification",
        domain="wafer_sem_defect_classification",
        modality="SEM_GRAYSCALE",
        image_characteristics=[
            "grayscale microscopy images",
            "fine texture and morphology cues",
            "possible high-resolution inputs",
        ],
        data_risks=[
            "class imbalance and rare defects",
            "wafer or lot leakage requires group-aware splitting",
            "raw images and internal paths must not be sent to an external LLM",
        ],
        preferred_metrics=["macro_f1", "balanced_accuracy", "per_class_recall"],
        compute_budget="local_short_training",
    )


class RetrievalWindow(ResearchModel):
    """Exact UTC interval used in addition to the provider's day-level query."""

    schema_version: Literal["1"] = "1"
    start_at: UTCDateTime
    end_at: UTCDateTime

    @model_validator(mode="after")
    def _ordered(self) -> RetrievalWindow:
        if self.start_at >= self.end_at:
            raise ValueError("retrieval window start_at must be earlier than end_at")
        return self


class ResearchWatermark(ResearchModel):
    """Last successfully completed research retrieval timestamp."""

    schema_version: Literal["1"] = "1"
    last_successful_run_at: UTCDateTime


class ResearchPaper(ResearchModel):
    """Normalized public paper data retained outside LangGraph state."""

    schema_version: Literal["1"] = "1"
    paper_id: OpaqueId
    provider_name: NonBlankStr
    provider_record_ids: list[NonBlankStr] = Field(min_length=1)
    arxiv_id: NonBlankStr | None = None
    doi: NonBlankStr | None = None
    title: HumanText
    abstract: PublicAbstract
    authors: list[HumanText]
    categories: list[NonBlankStr]
    published_at: UTCDateTime
    updated_at: UTCDateTime
    entry_url: NonBlankStr
    pdf_url: NonBlankStr | None = None
    comment: HumanText | None = None
    code_urls: list[NonBlankStr] = Field(default_factory=list)
    provenance: list[ProvenanceRef] = Field(min_length=1)


class HardFilterDecision(ResearchModel):
    """Deterministic eligibility result produced before any LLM call."""

    schema_version: Literal["1"] = "1"
    eligible: StrictBoolean
    classification_match: StrictBoolean
    python_match: StrictBoolean
    pytorch_match: StrictBoolean
    code_available: StrictBoolean
    matched_terms: dict[NonBlankStr, list[NonBlankStr]] = Field(default_factory=dict)
    reasons: list[Reason] = Field(default_factory=list)

    @model_validator(mode="after")
    def _eligibility_matches_components(self) -> HardFilterDecision:
        expected = self.classification_match and self.python_match and self.pytorch_match
        if self.eligible is not expected:
            raise ValueError(
                "eligible must equal the classification, Python, and PyTorch hard-filter components"
            )
        return self


class ApplicabilityEvidence(ResearchModel):
    """One bounded, source-addressed statement supporting the LLM decision."""

    schema_version: Literal["1"] = "1"
    dimension: Literal["TASK", "MODALITY", "DATA", "CODE", "COMPUTE"]
    source_field: Literal["title", "abstract", "categories", "comment", "problem_profile"]
    statement: HumanText


class PaperApplicabilityDecision(ResearchModel):
    """Validated LLM judgment for SEM defect-classification applicability."""

    schema_version: Literal["1"] = "1"
    summary: HumanText
    applicable: StrictBoolean
    recommendation: Literal["HIGH", "MEDIUM", "LOW", "REJECT"]
    relevance_score: UnitInterval
    confidence: UnitInterval
    task_match: UnitInterval
    modality_match: UnitInterval
    data_match: UnitInterval
    code_match: UnitInterval
    compute_fit: UnitInterval
    evidence: list[ApplicabilityEvidence] = Field(min_length=1, max_length=10)
    risks: list[HumanText] = Field(default_factory=list, max_length=10)
    rationale: HumanText

    @model_validator(mode="after")
    def _recommendation_consistency(self) -> PaperApplicabilityDecision:
        if self.recommendation == "REJECT" and self.applicable:
            raise ValueError("REJECT recommendation cannot be applicable")
        if self.recommendation in {"HIGH", "MEDIUM"} and not self.applicable:
            raise ValueError("HIGH/MEDIUM recommendation requires applicable=true")
        return self


class ResearchPaperAssessment(ResearchModel):
    """Paper, deterministic filter, optional LLM decision, and final candidate."""

    schema_version: Literal["1"] = "1"
    paper: ResearchPaper
    hard_filter: HardFilterDecision
    applicability: PaperApplicabilityDecision | None = None
    generation: GenerationRecord | None = None
    candidate: PaperCandidate
    selected: StrictBoolean = False

    @model_validator(mode="after")
    def _generation_pair(self) -> ResearchPaperAssessment:
        if (self.applicability is None) != (self.generation is None):
            raise ValueError("applicability and generation provenance must appear together")
        if self.candidate.paper_id != self.paper.paper_id:
            raise ValueError("candidate paper_id must match the normalized paper")
        if self.selected != self.candidate.selected:
            raise ValueError("assessment and candidate selected values must match")
        return self


class ResearchResult(ResearchModel):
    """Canonical local JSON output for one Research Agent workflow."""

    schema_version: Literal["1"] = "1"
    workflow_id: OpaqueId
    request_id: OpaqueId
    problem_profile: ProblemProfile
    retrieval_window: RetrievalWindow
    watermark_before: UTCDateTime | None = None
    query_spec: QuerySpec
    assessments: list[ResearchPaperAssessment]
    recommended_paper_ids: list[OpaqueId] = Field(default_factory=list)
    selected_paper_ids: list[OpaqueId] = Field(default_factory=list)
    status: Literal["AWAITING_SELECTION", "COMPLETED", "REJECTED", "NO_CANDIDATES"]
    gate_id: OpaqueId | None = None
    gate_revision: int | None = Field(default=None, ge=1)
    created_at: UTCDateTime
    updated_at: UTCDateTime


__all__ = [
    "ApplicabilityEvidence",
    "HardFilterDecision",
    "PaperApplicabilityDecision",
    "ProblemProfile",
    "PublicAbstract",
    "ResearchModel",
    "ResearchPaper",
    "ResearchPaperAssessment",
    "ResearchResult",
    "ResearchWatermark",
    "RetrievalWindow",
    "default_sem_problem_profile",
]
