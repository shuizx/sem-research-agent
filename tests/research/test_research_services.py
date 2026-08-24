"""Deterministic retrieval, normalization, filtering, prompt, and adapter tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from vision_research_ops.adapters.papers import ArxivPaperProvider
from vision_research_ops.application.services.paper_analysis import (
    applicability_facts,
    hard_filter_paper,
)
from vision_research_ops.application.services.paper_retrieval import (
    collect_provider_records,
    compute_retrieval_window,
    normalize_and_deduplicate,
    normalize_raw_paper,
    query_for_window,
)
from vision_research_ops.domain import QuerySpec
from vision_research_ops.ports import OperationContext, PaperQuery, PaperSearchPage

from .conftest import FIXED_NOW, excluded_paper, raw_paper


def _ctx() -> OperationContext:
    return OperationContext(
        schema_version="1",
        correlation_id="corr-service-test",
        workflow_id="workflow-service-test",
        actor_id="pipeline-user",
        sensitivity="PUBLIC",
    )


def test_watermark_window_uses_overlap_and_initial_lookback() -> None:
    """A prior success adds overlap; a first run uses the bounded lookback."""
    previous = datetime(2026, 8, 11, 8, 0, tzinfo=UTC)
    overlapped = compute_retrieval_window(
        now=FIXED_NOW,
        last_successful_run_at=previous,
        overlap=timedelta(minutes=60),
        initial_lookback=timedelta(hours=24),
    )
    assert overlapped.start_at == datetime(2026, 8, 11, 7, 0, tzinfo=UTC)
    assert overlapped.end_at == FIXED_NOW

    first = compute_retrieval_window(
        now=FIXED_NOW,
        last_successful_run_at=None,
        overlap=timedelta(minutes=60),
        initial_lookback=timedelta(hours=24),
    )
    assert first.start_at == FIXED_NOW - timedelta(hours=24)
    query = query_for_window(
        QuerySpec(schema_version="1", keywords=["classification"]),
        first,
    )
    assert query.date_from.isoformat() == "2026-08-10"
    assert query.date_to.isoformat() == "2026-08-11"


@pytest.mark.asyncio
async def test_provider_pagination_honors_page_and_record_limits() -> None:
    """Collection stops at configured ceilings even when another cursor exists."""
    from tests.fakes import DelegateStep, ScriptedPaperProvider

    pages = {
        None: PaperSearchPage(
            schema_version="1",
            provider_name="arxiv",
            records=[raw_paper(record_id="a", arxiv_id="2608.00001v1")],
            next_cursor="page-2",
            provider_request_id="request-1",
            retrieved_at=FIXED_NOW,
        ),
        "page-2": PaperSearchPage(
            schema_version="1",
            provider_name="arxiv",
            records=[raw_paper(record_id="b", arxiv_id="2608.00002v1")],
            next_cursor=None,
            provider_request_id="request-2",
            retrieved_at=FIXED_NOW,
        ),
    }
    provider = ScriptedPaperProvider(
        provider_name="arxiv",
        pages=pages,
        script={"paper.search": [DelegateStep()]},
    )
    records, pages_used = await collect_provider_records(
        provider,
        query_id="query-pagination",
        query_spec=QuerySpec(schema_version="1", keywords=["classification"]),
        max_pages=1,
        max_records=10,
        page_size=1,
        ctx=_ctx(),
    )
    assert [record.provider_record_id for record in records] == ["a"]
    assert pages_used == 1
    assert provider.call_count("paper.search") == 1


def test_arxiv_doi_and_title_duplicates_merge_with_stable_order() -> None:
    """Identity-connected records merge and retain the newest public metadata."""
    duplicate = raw_paper(
        record_id="2608.01234v2",
        arxiv_id="2608.01234v2",
        doi="https://doi.org/10.1000/SEM.2026.1",
        title="  PyTorch classification—of wafer SEM defects ",
        updated_at="2026-08-11T11:00:00Z",
        code_urls=[
            "https://github.com/example/sem-classifier",
            "https://github.com/example/sem-classifier-docs",
        ],
    )
    other = raw_paper(
        record_id="2608.09999v1",
        arxiv_id="2608.09999v1",
        doi=None,
        title="Classification Baseline for Microscopy",
        published_at="2026-08-11T08:00:00Z",
    )
    papers = normalize_and_deduplicate([other, raw_paper(), duplicate])
    assert len(papers) == 2
    merged = papers[0]
    assert merged.paper_id == "paper-arxiv-2608.01234"
    assert merged.provider_record_ids == ["2608.01234v1", "2608.01234v2"]
    assert merged.doi == "10.1000/sem.2026.1"
    assert merged.code_urls == [
        "https://github.com/example/sem-classifier",
        "https://github.com/example/sem-classifier-docs",
    ]


def test_hard_filter_excludes_without_llm_and_prompt_facts_are_sanitized() -> None:
    """Rules are deterministic and prompt facts omit internal/path/secret fields."""
    eligible = normalize_raw_paper(raw_paper())
    excluded = normalize_raw_paper(excluded_paper())
    assert hard_filter_paper(eligible).eligible is True
    decision = hard_filter_paper(excluded)
    assert decision.eligible is False
    assert decision.classification_match is False
    assert decision.code_available is False

    from vision_research_ops.application.services.paper_models import (
        default_sem_problem_profile,
    )

    facts = applicability_facts(eligible, default_sem_problem_profile())
    encoded = json.dumps(facts, ensure_ascii=False)
    for forbidden in (
        "workflow_id",
        "request_id",
        "dataset_id",
        "profile_id",
        "location_ref",
        "DASHSCOPE_API_KEY",
        "C:\\",
        "/home/",
        "image_bytes",
    ):
        assert forbidden not in encoded
    assert "2608.01234" in encoded
    assert "wafer_sem_defect_classification" in encoded


def test_missing_code_is_an_engineering_fact_not_a_daily_hard_exclusion() -> None:
    """code availability does not change classification/Python/PyTorch eligibility."""
    paper = normalize_raw_paper(
        raw_paper(
            code_urls=[],
            abstract="Image classification for wafer SEM defects using Python and PyTorch.",
            comment="No public code link appears in this metadata.",
        )
    )
    decision = hard_filter_paper(paper)
    assert decision.eligible is True
    assert decision.classification_match is True
    assert decision.python_match is True
    assert decision.pytorch_match is True
    assert decision.code_available is False
    assert "RESEARCH_HARD_MISSING_CODE" not in {reason.code for reason in decision.reasons}


@pytest.mark.asyncio
async def test_arxiv_atom_adapter_parses_public_metadata_and_bounded_query() -> None:
    """The live adapter surface is exercised offline using a recorded Atom fixture."""
    payload = (Path(__file__).parent / "fixtures" / "arxiv_feed.xml").read_bytes()
    requested: list[str] = []

    def transport(url: str, timeout: int) -> bytes:
        assert timeout == 7
        requested.append(url)
        return payload

    provider = ArxivPaperProvider(
        timeout_seconds=7,
        transport=transport,
        clock=lambda: FIXED_NOW,
    )
    query = PaperQuery(
        schema_version="1",
        query_id="query-arxiv-fixture",
        query_spec=QuerySpec(
            schema_version="1",
            keywords=["defect classification"],
            domains=["cs.CV"],
            date_from=datetime(2026, 8, 10, tzinfo=UTC).date(),
            date_to=datetime(2026, 8, 11, tzinfo=UTC).date(),
        ),
        page_size=2,
    )
    page = await provider.search(query, cursor=None, ctx=_ctx())
    assert len(page.records) == 2
    assert page.records[0].provider_record_id == "2608.01234v1"
    assert page.records[0].raw_fields["code_urls"] == ["https://github.com/example/sem-classifier"]
    assert page.next_cursor is None
    params = parse_qs(urlparse(requested[0]).query)
    assert params["start"] == ["0"]
    assert params["max_results"] == ["2"]
    assert "submittedDate:[202608100000 TO 202608112359]" in params["search_query"][0]
