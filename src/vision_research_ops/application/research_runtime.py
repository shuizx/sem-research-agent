"""Runtime-only dependencies for the research workflow Research Agent graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from vision_research_ops.domain import ResearchRequest
from vision_research_ops.ports import PaperProvider, StructuredLLM

from .runtime import ApprovalRecorder
from .services.paper_models import ProblemProfile
from .services.paper_store import LocalResearchStore, ResearchSession


def research_utc_now() -> datetime:
    """Return an aware UTC time for injected research dependencies."""
    return datetime.now(UTC)


@dataclass(slots=True)
class ResearchDependencies:
    """Injected provider, LLM, local store, request, and single-run session."""

    request: ResearchRequest
    problem_profile: ProblemProfile
    paper_provider: PaperProvider
    structured_llm: StructuredLLM
    store: LocalResearchStore
    approval_recorder: ApprovalRecorder
    page_size: int = 25
    overlap_minutes: int = 60
    initial_lookback_hours: int = 24
    clock: Callable[[], datetime] = research_utc_now
    session: ResearchSession = field(default_factory=ResearchSession)

    def __post_init__(self) -> None:
        if self.page_size < 1 or self.page_size > 100:
            raise ValueError("research page_size must be between 1 and 100")
        if self.overlap_minutes < 0:
            raise ValueError("research overlap_minutes must not be negative")
        if self.initial_lookback_hours <= 0:
            raise ValueError("research initial_lookback_hours must be positive")
        if not callable(self.clock):
            raise TypeError("research clock must be callable")


__all__ = ["ResearchDependencies", "research_utc_now"]
