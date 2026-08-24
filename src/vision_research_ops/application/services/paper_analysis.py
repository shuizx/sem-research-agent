"""Deterministic paper filtering and structured LLM request composition."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from vision_research_ops.domain import (
    AuthorRef,
    GenerationRecord,
    JsonObject,
    PaperCandidate,
    Reason,
)
from vision_research_ops.ports import (
    OperationContext,
    StructuredGenerationRequest,
    StructuredLLM,
)
from vision_research_ops.prompts.paper_applicability import (
    PROMPT_TEMPLATE_ID,
    PROMPT_VERSION,
)

from .paper_models import (
    HardFilterDecision,
    PaperApplicabilityDecision,
    ProblemProfile,
    ResearchPaper,
    ResearchPaperAssessment,
)

_CLASSIFICATION_TERMS = (
    "classification",
    "classifier",
    "recognition",
    "categorization",
    "defect class",
)
_PYTHON_TERMS = ("python", "pytorch", "torchvision")
_PYTORCH_TERMS = ("pytorch", "torchvision", "torch.nn")


def _matches(text: str, terms: Iterable[str]) -> list[str]:
    lowered = text.casefold()
    return [term for term in terms if term in lowered]


def hard_filter_paper(paper: ResearchPaper) -> HardFilterDecision:
    """Apply daily classification/Python/PyTorch eligibility without an LLM."""
    public_text = " ".join([paper.title, paper.abstract, paper.comment or "", *paper.code_urls])
    classification_terms = _matches(public_text, _CLASSIFICATION_TERMS)
    python_terms = _matches(public_text, _PYTHON_TERMS)
    pytorch_terms = _matches(public_text, _PYTORCH_TERMS)
    code_terms = [url for url in paper.code_urls if url.startswith(("https://", "http://"))]
    components = {
        "classification": classification_terms,
        "python": python_terms,
        "pytorch": pytorch_terms,
        "code": code_terms,
    }
    eligibility_components = ("classification", "python", "pytorch")
    missing = [name for name in eligibility_components if not components[name]]
    reasons = [
        Reason(
            schema_version="1",
            code=f"RESEARCH_HARD_MISSING_{name.upper()}",
            message=f"Public metadata lacks deterministic {name} evidence.",
        )
        for name in missing
    ]
    return HardFilterDecision(
        eligible=not missing,
        classification_match=bool(classification_terms),
        python_match=bool(python_terms),
        pytorch_match=bool(pytorch_terms),
        code_available=bool(code_terms),
        matched_terms=components,
        reasons=reasons,
    )


def applicability_facts(
    paper: ResearchPaper,
    problem: ProblemProfile,
) -> JsonObject:
    """Build the only public facts allowed to cross the external LLM boundary."""
    return cast(
        JsonObject,
        {
            "problem": problem.prompt_facts(),
            "paper": {
                "arxiv_id": paper.arxiv_id,
                "doi": paper.doi,
                "title": paper.title,
                "abstract": paper.abstract,
                "categories": list(paper.categories),
                "comment": paper.comment,
                "code_urls": list(paper.code_urls),
            },
        },
    )


def applicability_request(
    paper: ResearchPaper,
    problem: ProblemProfile,
) -> StructuredGenerationRequest[PaperApplicabilityDecision]:
    """Create a schema-bound request with no workflow, dataset, path, image, or key."""
    return StructuredGenerationRequest[PaperApplicabilityDecision](
        schema_version="1",
        task_name="paper_applicability",
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_version=PROMPT_VERSION,
        response_schema=PaperApplicabilityDecision,
        facts=applicability_facts(paper, problem),
        artifact_excerpts=[],
        model_parameters={"temperature": 0},
        budget_class="pipeline_research_small",
    )


def _candidate(
    *,
    paper: ResearchPaper,
    request_id: str,
    hard_filter: HardFilterDecision,
    decision: PaperApplicabilityDecision | None,
    selected: bool = False,
    budget_skipped: bool = False,
) -> PaperCandidate:
    inclusion_reasons: list[Reason] = []
    exclusion_reasons = list(hard_filter.reasons)
    task_tags = ["image_classification"] if hard_filter.classification_match else []
    method_tags = ["pytorch"] if hard_filter.pytorch_match else []
    relevance_score = 0.0
    score_components: dict[str, float] = {}

    if hard_filter.eligible:
        inclusion_reasons.append(
            Reason(
                schema_version="1",
                code="RESEARCH_HARD_FILTER_ELIGIBLE",
                message="Public metadata satisfies the classification/Python/PyTorch prefilter.",
            )
        )
    if budget_skipped:
        exclusion_reasons.append(
            Reason(
                schema_version="1",
                code="RESEARCH_BUDGET_LLM_SKIPPED",
                message="The configured LLM-call budget was exhausted.",
            )
        )
    if decision is not None:
        relevance_score = decision.relevance_score
        score_components = {
            "task_match": decision.task_match,
            "modality_match": decision.modality_match,
            "data_match": decision.data_match,
            "code_match": decision.code_match,
            "compute_fit": decision.compute_fit,
            "confidence": decision.confidence,
        }
        reason = Reason(
            schema_version="1",
            code=(
                "RESEARCH_LLM_APPLICABLE" if decision.applicable else "RESEARCH_LLM_NOT_APPLICABLE"
            ),
            message=decision.rationale,
        )
        if decision.applicable:
            inclusion_reasons.append(reason)
        else:
            exclusion_reasons.append(reason)

    urls = {"entry": paper.entry_url}
    if paper.pdf_url is not None:
        urls["pdf"] = paper.pdf_url
    for index, url in enumerate(paper.code_urls, start=1):
        urls[f"code_{index}"] = url
    external_ids: dict[str, str] = {}
    if paper.arxiv_id is not None:
        external_ids["arxiv"] = paper.arxiv_id
    if paper.doi is not None:
        external_ids["doi"] = paper.doi
    return PaperCandidate(
        schema_version="1",
        paper_id=paper.paper_id,
        request_id=request_id,
        canonical_title=paper.title,
        abstract_artifact_id=None,
        authors=[AuthorRef(schema_version="1", name=name) for name in paper.authors],
        external_ids=external_ids,
        first_published_at=paper.published_at,
        updated_at_external=paper.updated_at,
        venue=None,
        urls=urls,
        task_tags=task_tags,
        method_tags=method_tags,
        relevance_score=relevance_score,
        score_components=score_components,
        inclusion_reasons=inclusion_reasons,
        exclusion_reasons=exclusion_reasons,
        provenance=paper.provenance,
        selected=selected,
    )


def unscored_assessment(
    paper: ResearchPaper,
    *,
    request_id: str,
) -> ResearchPaperAssessment:
    """Create an assessment after deterministic filtering and before LLM analysis."""
    hard_filter = hard_filter_paper(paper)
    return ResearchPaperAssessment(
        paper=paper,
        hard_filter=hard_filter,
        applicability=None,
        generation=None,
        candidate=_candidate(
            paper=paper,
            request_id=request_id,
            hard_filter=hard_filter,
            decision=None,
        ),
    )


async def score_assessments(
    assessments: list[ResearchPaperAssessment],
    *,
    problem: ProblemProfile,
    request_id: str,
    llm: StructuredLLM,
    max_llm_calls: int,
    ctx: OperationContext,
    include_ineligible: bool = False,
) -> list[ResearchPaperAssessment]:
    """Score eligible papers, optionally including a user-requested single-paper analysis."""
    scored: list[ResearchPaperAssessment] = []
    used_calls = 0
    for assessment in assessments:
        if not assessment.hard_filter.eligible and not include_ineligible:
            scored.append(assessment)
            continue
        if used_calls >= max_llm_calls:
            scored.append(
                assessment.model_copy(
                    update={
                        "candidate": _candidate(
                            paper=assessment.paper,
                            request_id=request_id,
                            hard_filter=assessment.hard_filter,
                            decision=None,
                            budget_skipped=True,
                        )
                    }
                )
            )
            continue
        result = await llm.generate(
            applicability_request(assessment.paper, problem),
            ctx=ctx,
        )
        used_calls += 1
        generation = GenerationRecord(
            schema_version="1",
            provider_id=result.provider_id,
            model_id=result.model_id,
            prompt_template_id=PROMPT_TEMPLATE_ID,
            prompt_version=PROMPT_VERSION,
            prompt_hash=result.prompt_hash,
            output_hash=result.output_hash,
        )
        scored.append(
            assessment.model_copy(
                update={
                    "applicability": result.value,
                    "generation": generation,
                    "candidate": _candidate(
                        paper=assessment.paper,
                        request_id=request_id,
                        hard_filter=assessment.hard_filter,
                        decision=result.value,
                    ),
                }
            )
        )
    return scored


def rank_assessments(
    assessments: list[ResearchPaperAssessment],
) -> list[ResearchPaperAssessment]:
    """Sort scored papers by recommendation evidence with deterministic tie breaks."""
    return sorted(
        assessments,
        key=lambda item: (
            -item.candidate.relevance_score,
            -item.paper.published_at.timestamp(),
            item.paper.paper_id,
        ),
    )


def recommended_paper_ids(
    assessments: list[ResearchPaperAssessment],
    *,
    limit: int,
) -> list[str]:
    """Select only schema-validated applicable papers up to the caller limit."""
    if limit < 0:
        raise ValueError("recommendation limit must not be negative")
    return [
        item.paper.paper_id
        for item in rank_assessments(assessments)
        if item.applicability is not None and item.applicability.applicable
    ][:limit]


__all__ = [
    "applicability_facts",
    "applicability_request",
    "hard_filter_paper",
    "rank_assessments",
    "recommended_paper_ids",
    "score_assessments",
    "unscored_assessment",
]
