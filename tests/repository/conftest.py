"""Deterministic repository workflow repository, research result, and graph fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from vision_research_ops.adapters.repositories import (
    GitHubRepositoryProvider,
    ZipStaticRepositoryAnalyzer,
)
from vision_research_ops.application.repository_runtime import RepositoryDependencies
from vision_research_ops.application.runtime import InMemoryApprovalRecorder
from vision_research_ops.application.services.paper_analysis import unscored_assessment
from vision_research_ops.application.services.paper_models import (
    ResearchPaper,
    ResearchResult,
    RetrievalWindow,
    default_sem_problem_profile,
)
from vision_research_ops.application.services.paper_store import LocalResearchStore
from vision_research_ops.application.services.repository_store import LocalRepositoryStore
from vision_research_ops.application.state import WorkflowState, create_initial_state
from vision_research_ops.domain import ProvenanceRef, QuerySpec

FIXED_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
FIXED_SHA = "a" * 40


class FixtureGitHubTransport:
    """Serve bounded GitHub API fixtures while retaining observable read calls."""

    def __init__(
        self,
        archive: bytes,
        *,
        commit_sha: str = FIXED_SHA,
        license_spdx: str | None = "MIT",
    ) -> None:
        self.archive = archive
        self.commit_sha = commit_sha
        self.license_spdx = license_spdx
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout: int) -> bytes:
        assert timeout > 0
        self.calls.append((url, dict(headers)))
        if "/commits/" in url:
            return json.dumps({"sha": self.commit_sha}).encode()
        if url.endswith("/languages"):
            return json.dumps({"Python": 4096}).encode()
        if "/zipball/" in url:
            return self.archive
        license_value = None if self.license_spdx is None else {"spdx_id": self.license_spdx}
        return json.dumps(
            {
                "archived": False,
                "default_branch": "main",
                "fork": False,
                "license": license_value,
            }
        ).encode()

    @property
    def call_count(self) -> int:
        """Return the total number of simulated external reads."""
        return len(self.calls)


def repository_archive(*, dangerous_source: bool = False) -> bytes:
    """Build a reproducible zip from the visible plain-PyTorch fixture."""
    fixture_root = Path(__file__).parent / "fixtures" / "plain_pytorch"
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        for source in sorted(path for path in fixture_root.rglob("*") if path.is_file()):
            relative = source.relative_to(fixture_root).as_posix()
            content = source.read_text(encoding="utf-8")
            if dangerous_source and relative == "train.py":
                content += "\nimport os\nos.system('unsafe')\n"
            info = ZipInfo(f"example-sem-classifier/{relative}")
            info.date_time = (2026, 1, 1, 0, 0, 0)
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, content.encode("utf-8"))
    return buffer.getvalue()


def regression_repository_archive() -> bytes:
    """Build a PyTorch regression layout that must not pass the classification policy."""
    source = """from torch import nn
from torch.utils.data import DataLoader, Dataset


class RegressionDataset(Dataset):
    pass


class Regressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Linear(4, 1)


def train() -> None:
    DataLoader(RegressionDataset(), batch_size=4)
"""
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        info = ZipInfo("example-regressor/train.py")
        info.date_time = (2026, 1, 1, 0, 0, 0)
        info.compress_type = ZIP_DEFLATED
        archive.writestr(info, source.encode("utf-8"))
    return buffer.getvalue()


def write_selected_research_result(
    store: LocalResearchStore,
    *,
    code_urls: list[str] | None = None,
) -> None:
    """Persist the smallest honest completed research result consumed by repository."""
    paper = ResearchPaper(
        paper_id="paper-arxiv-2608.01234",
        provider_name="arxiv",
        provider_record_ids=["2608.01234v1"],
        arxiv_id="2608.01234v1",
        doi="10.1000/sem.2026.1",
        title="PyTorch Classification of Wafer SEM Defects",
        abstract=(
            "Image classification for wafer SEM defects with a public Python PyTorch "
            "implementation."
        ),
        authors=["Ada Researcher"],
        categories=["cs.CV"],
        published_at=FIXED_NOW,
        updated_at=FIXED_NOW,
        entry_url="https://arxiv.org/abs/2608.01234v1",
        pdf_url="https://arxiv.org/pdf/2608.01234v1",
        comment="The paper metadata links the authors' public implementation.",
        code_urls=code_urls or ["https://github.com/example/sem-classifier"],
        provenance=[
            ProvenanceRef(
                schema_version="1",
                source_type="provider",
                source_id="arxiv:2608.01234v1",
                source_url="https://arxiv.org/abs/2608.01234v1",
                retrieved_at=FIXED_NOW,
            )
        ],
    )
    assessment = unscored_assessment(paper, request_id="request-research-1")
    assessment = assessment.model_copy(
        update={
            "selected": True,
            "candidate": assessment.candidate.model_copy(update={"selected": True}),
        }
    )
    store.write_result(
        ResearchResult(
            workflow_id="workflow-research-1",
            request_id="request-research-1",
            problem_profile=default_sem_problem_profile(),
            retrieval_window=RetrievalWindow(
                start_at=FIXED_NOW.replace(hour=11),
                end_at=FIXED_NOW,
            ),
            query_spec=QuerySpec(
                schema_version="1",
                keywords=["SEM defect classification"],
                domains=["cs.CV"],
            ),
            assessments=[assessment],
            recommended_paper_ids=[paper.paper_id],
            selected_paper_ids=[paper.paper_id],
            status="COMPLETED",
            gate_id="gate-candidate-selection-workflow-research-1-r1",
            gate_revision=1,
            created_at=FIXED_NOW,
            updated_at=FIXED_NOW,
        )
    )


@dataclass(slots=True)
class RepositoryHarness:
    """Injected dependencies and observable repository tools."""

    dependencies: RepositoryDependencies
    transport: FixtureGitHubTransport
    recorder: InMemoryApprovalRecorder
    store: LocalRepositoryStore
    research_store: LocalResearchStore


@pytest.fixture
def make_repository_harness(tmp_path: Path) -> Callable[..., RepositoryHarness]:
    """Create a production-adapter graph harness with fixture-only HTTP bytes."""

    def factory(
        *,
        root: Path | None = None,
        transport: FixtureGitHubTransport | None = None,
        code_urls: list[str] | None = None,
        license_spdx: str | None = "MIT",
        dangerous_source: bool = False,
    ) -> RepositoryHarness:
        base = root or tmp_path
        research_store = LocalResearchStore(base / "research")
        write_selected_research_result(research_store, code_urls=code_urls)
        configured_transport = transport or FixtureGitHubTransport(
            repository_archive(dangerous_source=dangerous_source),
            license_spdx=license_spdx,
        )
        artifact_root = base / "repository-artifacts"
        provider = GitHubRepositoryProvider(
            artifact_root=artifact_root,
            transport=configured_transport,
            clock=lambda: FIXED_NOW,
        )
        recorder = InMemoryApprovalRecorder()
        store = LocalRepositoryStore(base / "repositories")
        dependencies = RepositoryDependencies(
            research_store=research_store,
            research_workflow_id="workflow-research-1",
            selected_paper_id="paper-arxiv-2608.01234",
            repository_provider=provider,
            static_analyzer=ZipStaticRepositoryAnalyzer(artifact_root=artifact_root),
            store=store,
            approval_recorder=recorder,
            clock=lambda: FIXED_NOW,
        )
        return RepositoryHarness(
            dependencies=dependencies,
            transport=configured_transport,
            recorder=recorder,
            store=store,
            research_store=research_store,
        )

    return factory


@pytest.fixture
def repository_initial_state() -> WorkflowState:
    """Return the small checkpoint state used by Repository Agent tests."""
    return create_initial_state(
        {
            "schema_version": "1",
            "workflow_id": "workflow-repository-1",
            "thread_id": "thread-repository-1",
            "request_id": "request-repository-1",
            "dataset_profile_id": "problem-wafer-sem-v1",
        }
    )
