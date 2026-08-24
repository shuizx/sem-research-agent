"""Deterministic fixtures for the research workflow Research Agent matrix."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.fakes import DelegateStep, ScriptedPaperProvider, ScriptedStructuredLLM
from vision_research_ops.application.research_runtime import ResearchDependencies
from vision_research_ops.application.runtime import InMemoryApprovalRecorder
from vision_research_ops.application.services.paper_models import default_sem_problem_profile
from vision_research_ops.application.services.paper_store import LocalResearchStore
from vision_research_ops.application.state import WorkflowState, create_initial_state
from vision_research_ops.domain import QuerySpec, ResearchBudget, ResearchRequest, WorkflowStatus
from vision_research_ops.ports import ExternalPaperId, PaperSearchPage, RawPaperRecord

FIXED_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


def raw_paper(
    *,
    record_id: str = "2608.01234v1",
    arxiv_id: str = "2608.01234v1",
    doi: str | None = "10.1000/sem.2026.1",
    title: str = "PyTorch Classification of Wafer SEM Defects",
    abstract: str = (
        "Image classification for wafer SEM defects with a public Python PyTorch implementation "
        "at https://github.com/example/sem-classifier."
    ),
    published_at: str = "2026-08-11T10:00:00Z",
    updated_at: str | None = None,
    code_urls: list[str] | None = None,
    comment: str = "Python PyTorch reference implementation is public.",
) -> RawPaperRecord:
    """Build one valid provider record with overrideable identity evidence."""
    external_ids = [ExternalPaperId(schema_version="1", provider_name="arxiv", value=arxiv_id)]
    if doi is not None:
        external_ids.append(ExternalPaperId(schema_version="1", provider_name="doi", value=doi))
    return RawPaperRecord(
        schema_version="1",
        provider_name="arxiv",
        provider_record_id=record_id,
        external_ids=external_ids,
        raw_fields={
            "arxiv_id": arxiv_id,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "authors": ["Ada Researcher"],
            "categories": ["cs.CV"],
            "published_at": published_at,
            "updated_at": updated_at or published_at,
            "entry_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "comment": comment,
            "code_urls": (
                ["https://github.com/example/sem-classifier"] if code_urls is None else code_urls
            ),
        },
        retrieved_at=FIXED_NOW,
    )


def excluded_paper() -> RawPaperRecord:
    """Build a normal CV paper that deterministic rules exclude before LLM use."""
    return raw_paper(
        record_id="2608.05678v1",
        arxiv_id="2608.05678v1",
        doi=None,
        title="Unsupervised Segmentation of Natural Photographs",
        abstract="We segment natural RGB photographs without a released implementation.",
        code_urls=[],
    )


def valid_decision(*, applicable: bool = True) -> dict[str, object]:
    """Return one strict scripted PaperApplicabilityDecision payload."""
    return {
        "schema_version": "1",
        "summary": "The paper presents a PyTorch classification method for wafer SEM defects.",
        "applicable": applicable,
        "recommendation": "HIGH" if applicable else "REJECT",
        "relevance_score": 0.91 if applicable else 0.2,
        "confidence": 0.84,
        "task_match": 0.95,
        "modality_match": 0.9,
        "data_match": 0.8,
        "code_match": 0.95,
        "compute_fit": 0.75,
        "evidence": [
            {
                "schema_version": "1",
                "dimension": "TASK",
                "source_field": "abstract",
                "statement": "The paper explicitly studies classification of wafer SEM defects.",
            },
            {
                "schema_version": "1",
                "dimension": "CODE",
                "source_field": "comment",
                "statement": "The metadata links a public PyTorch implementation.",
            },
        ],
        "risks": ["A small local experiment may not reproduce the reported scale."],
        "rationale": (
            "The task, microscopy modality, and public PyTorch code support a bounded trial."
            if applicable
            else "The available evidence does not justify a SEM classification trial."
        ),
    }


def make_request(
    *,
    max_provider_pages: int = 2,
    max_provider_records: int = 20,
    max_llm_calls: int = 5,
) -> ResearchRequest:
    """Build the immutable request used by graph tests."""
    return ResearchRequest(
        schema_version="1",
        request_id="request-research-1",
        revision=1,
        title="Daily SEM paper research",
        research_question="Which new CV papers merit a wafer SEM classification experiment?",
        dataset_id="problem-wafer-sem-v1",
        dataset_version="profile-v1",
        query_spec=QuerySpec(
            schema_version="1",
            keywords=["defect classification", "microscopy classification"],
            domains=["cs.CV"],
            excluded_terms=["survey"],
        ),
        candidate_limit=5,
        budget=ResearchBudget(
            schema_version="1",
            max_provider_pages=max_provider_pages,
            max_provider_records=max_provider_records,
            max_llm_calls=max_llm_calls,
            max_llm_tokens=4000,
            max_cost_estimate=1.0,
            max_candidate_repositories=5,
            max_adaptation_attempts=1,
            max_workflow_walltime_seconds=120,
        ),
        requested_by="pipeline-user",
        status=WorkflowStatus.PENDING,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
    )


@dataclass(slots=True)
class ResearchHarness:
    """Explicit graph dependencies and observable scripted adapters."""

    dependencies: ResearchDependencies
    provider: ScriptedPaperProvider
    llm: ScriptedStructuredLLM
    recorder: InMemoryApprovalRecorder
    store: LocalResearchStore


@pytest.fixture
def make_research_harness(tmp_path: Path) -> Callable[..., ResearchHarness]:
    """Create independent Research Agent dependencies rooted in a temporary directory."""

    def factory(
        *,
        pages: Mapping[str | None, PaperSearchPage] | None = None,
        llm_outputs: Mapping[str, object] | None = None,
        request: ResearchRequest | None = None,
        root: Path | None = None,
    ) -> ResearchHarness:
        configured_request = request or make_request()
        configured_pages = pages or {
            None: PaperSearchPage(
                schema_version="1",
                provider_name="arxiv",
                records=[raw_paper(), excluded_paper()],
                next_cursor=None,
                provider_request_id="arxiv-page-1",
                retrieved_at=FIXED_NOW,
            )
        }
        provider = ScriptedPaperProvider(
            provider_name="arxiv",
            pages=configured_pages,
            script={
                "paper.search": [
                    DelegateStep() for _ in range(configured_request.budget.max_provider_pages)
                ]
            },
        )
        llm = ScriptedStructuredLLM(
            outputs=llm_outputs or {"paper_applicability": valid_decision()},
            provider_id="scripted-dashscope",
            model_id="qwen-plus-fixture",
            script={
                "llm.generate": [
                    DelegateStep() for _ in range(configured_request.budget.max_llm_calls)
                ]
            },
        )
        recorder = InMemoryApprovalRecorder()
        store = LocalResearchStore(root or (tmp_path / "research"))
        dependencies = ResearchDependencies(
            request=configured_request,
            problem_profile=default_sem_problem_profile(),
            paper_provider=provider,
            structured_llm=llm,
            store=store,
            approval_recorder=recorder,
            page_size=2,
            overlap_minutes=60,
            initial_lookback_hours=24,
            clock=lambda: FIXED_NOW,
        )
        return ResearchHarness(dependencies, provider, llm, recorder, store)

    return factory


@pytest.fixture
def research_initial_state() -> WorkflowState:
    """Return the small checkpoint state used by every research graph test."""
    return create_initial_state(
        {
            "schema_version": "1",
            "workflow_id": "workflow-research-1",
            "thread_id": "thread-research-1",
            "request_id": "request-research-1",
            "dataset_profile_id": "problem-wafer-sem-v1",
        }
    )
