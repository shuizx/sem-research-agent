"""Controlled repository, dataset, patch, smoke, and graph fixtures for adaptation workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import pytest

from vision_research_ops.adaptation import (
    FixturePatchTool,
    FixtureSmokeRunner,
    FixtureToolCallingAdaptationPlanner,
)
from vision_research_ops.application.adaptation_runtime import AdaptationDependencies
from vision_research_ops.application.runtime import InMemoryApprovalRecorder
from vision_research_ops.application.services.adaptation_store import LocalAdaptationStore
from vision_research_ops.application.services.repository_models import (
    RepositoryProfile,
    RepositoryResult,
)
from vision_research_ops.application.services.repository_store import LocalRepositoryStore
from vision_research_ops.application.state import WorkflowState, create_initial_state
from vision_research_ops.domain import (
    ArtifactKind,
    ArtifactRef,
    CodeLinkConfidence,
    CodeLinkEvidence,
    DatasetProfile,
    LicenseStatus,
    ProvenanceRef,
    RepositorySnapshot,
)
from vision_research_ops.ports import (
    RepositoryAnalysis,
    RepositoryFileSummary,
    RepositoryMetadata,
    RepositoryPolicy,
    RepositoryResolution,
)

FIXED_NOW = datetime(2026, 8, 12, 8, 0, tzinfo=UTC)
FIXED_SHA = "a" * 40
REPOSITORY_WORKFLOW_ID = "workflow-repository-p3-fixture"


def repository_result(*, supported: bool = True, commit_sha: str = FIXED_SHA) -> RepositoryResult:
    """Build the smallest completed repository evidence for the exact controlled fixture."""
    resolution = RepositoryResolution(
        schema_version="1",
        canonical_url="https://github.com/example/sem-classifier",
        provider="GITHUB",
        owner="example",
        name="sem-classifier",
        commit_sha=commit_sha,
    )
    provenance = ProvenanceRef(
        schema_version="1",
        source_type="api",
        source_id=f"github:example/sem-classifier@{commit_sha}",
        source_url=resolution.canonical_url,
        retrieved_at=FIXED_NOW,
        content_hash="sha256:" + "c" * 64,
        evidence_artifact_id="artifact-repository-fixture",
    )
    evidence = CodeLinkEvidence(
        schema_version="1",
        evidence_id="code-link-p3-fixture",
        paper_id="paper-p3-fixture",
        repository_url=resolution.canonical_url,
        evidence_type="paper_link",
        confidence=CodeLinkConfidence.OFFICIAL_HIGH,
        rationale_codes=["PAPER_METADATA_PUBLIC_CODE_LINK"],
        provenance=provenance,
        verified_at=FIXED_NOW,
    )
    file_names = ["LICENSE", "config.yaml", "data.py", "model.py", "train.py"]
    file_tree = [
        RepositoryFileSummary(schema_version="1", path=name, size_bytes=128, kind="FILE")
        for name in file_names
    ]
    analysis = RepositoryAnalysis(
        schema_version="1",
        repository=resolution,
        policy=RepositoryPolicy(
            schema_version="1",
            policy_id="pipeline-pytorch-classification",
            policy_version="1",
        ),
        file_tree_summary=file_tree,
        dependency_files=[],
        framework_evidence=[
            "PYTORCH_IMPORT:model.py",
            "MODEL_HEAD:model.py",
            "IMAGE_CLASSIFICATION:train.py",
        ],
        entrypoint_candidates=["train.py"],
        data_loader_candidates=["data.py"],
        command_candidates=[],
        license_spdx="MIT",
        dangerous_patterns=[],
        supported=supported,
        support_reasons=[],
    )
    snapshot = RepositorySnapshot(
        schema_version="1",
        repository_id=f"repo-example-sem-classifier-{commit_sha[:12]}",
        canonical_url=resolution.canonical_url,
        provider="GITHUB",
        owner="example",
        name="sem-classifier",
        commit_sha=commit_sha,
        archive_artifact_id="artifact-repository-fixture",
        license_spdx="MIT",
        license_status=LicenseStatus.ALLOWLISTED,
        framework="PyTorch",
        languages={"Python": 4096},
        default_branch="main",
        risk_findings=[],
        provenance=[provenance],
        analyzed_at=FIXED_NOW,
    )
    profile = RepositoryProfile(
        profile_id="profile-p3-fixture",
        paper_id="paper-p3-fixture",
        repository_snapshot=snapshot,
        code_link_evidence=evidence,
        structure_type="PLAIN_PYTORCH" if supported else "UNSUPPORTED",
        entrypoint_candidates=["train.py"],
        data_loader_candidates=["data.py"],
        configuration_files=["config.yaml"],
        dependency_files=[],
        model_head_evidence=["model.py"],
        framework_evidence=analysis.framework_evidence,
        file_tree_summary=file_tree,
        risk_findings=[],
        supported=supported,
        support_reasons=[],
    )
    status = "COMPLETED" if supported else "UNSUPPORTED"
    return RepositoryResult(
        workflow_id=REPOSITORY_WORKFLOW_ID,
        request_id="request-repository-p3-fixture",
        research_workflow_id="workflow-research-p3-fixture",
        paper_id="paper-p3-fixture",
        requested_repository_url=resolution.canonical_url,
        approved_repository_url=resolution.canonical_url,
        code_link_evidence=evidence,
        status=status,
        gate_id="gate-repository-p3-fixture-r1",
        gate_revision=1,
        resolution=resolution,
        metadata=RepositoryMetadata(
            schema_version="1",
            repository=resolution,
            license_spdx="MIT",
            languages={"Python": 4096},
            default_branch="main",
            raw_fields={},
        ),
        archive=ArtifactRef(
            schema_version="1",
            artifact_id="artifact-repository-fixture",
            kind=ArtifactKind.REPOSITORY_ARCHIVE,
            uri="snapshots/p3-fixture.zip",
            sha256="sha256:" + "c" * 64,
            size_bytes=1024,
            media_type="application/zip",
            created_at=FIXED_NOW,
            producer="p2-fixture",
            sensitivity="PUBLIC",
            metadata={"fixture": True},
        ),
        analysis=analysis,
        profile=profile,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
    )


def load_dataset_profile() -> DatasetProfile:
    """Generate the sample profile from its committed synthetic image directory."""
    from vision_research_ops.application.services.dataset_profiling import profile_dataset

    root = Path(__file__).parents[2]
    return profile_dataset(
        root / "fixtures" / "datasets" / "synthetic_sem_images",
        output_root=root / "var" / "test-dataset-profiles",
    ).profile


@dataclass(slots=True)
class AdaptationHarness:
    """Injected dependencies and observable adaptation boundary tools."""

    dependencies: AdaptationDependencies
    llm: FixtureToolCallingAdaptationPlanner
    patch_tool: FixturePatchTool
    smoke_tool: FixtureSmokeRunner
    store: LocalAdaptationStore
    repository_store: LocalRepositoryStore
    recorder: InMemoryApprovalRecorder


@pytest.fixture
def make_adaptation_harness(tmp_path: Path) -> Callable[..., AdaptationHarness]:
    """Create one fully offline adaptation graph harness."""

    def factory(
        *,
        root: Path | None = None,
        supported_repository: bool = True,
        repository_commit_sha: str = FIXED_SHA,
        llm_mode: str = "success",
        minimum_repair_revision: int = 0,
    ) -> AdaptationHarness:
        var_root = (root or tmp_path) / "var"
        repository_store = LocalRepositoryStore(var_root / "repositories")
        repository_store.write_result(
            repository_result(
                supported=supported_repository,
                commit_sha=repository_commit_sha,
            )
        )
        store = LocalAdaptationStore(var_root)
        fixture_root = Path(__file__).parents[2] / "fixtures" / "repositories" / "plain_pytorch"
        llm = FixtureToolCallingAdaptationPlanner(
            mode=cast(Literal["success", "provider_failure", "schema_failure"], llm_mode)
        )
        patch_tool = FixturePatchTool(
            fixture_root=fixture_root,
            store=store,
            clock=lambda: FIXED_NOW,
        )
        smoke_tool = FixtureSmokeRunner(
            store=store,
            clock=lambda: FIXED_NOW,
            minimum_repair_revision=minimum_repair_revision,
        )
        recorder = InMemoryApprovalRecorder()
        dependencies = AdaptationDependencies(
            repository_store=repository_store,
            repository_workflow_id=REPOSITORY_WORKFLOW_ID,
            dataset_profile=load_dataset_profile(),
            planner=llm,
            patch_tool=patch_tool,
            smoke_tool=smoke_tool,
            store=store,
            approval_recorder=recorder,
            clock=lambda: FIXED_NOW,
        )
        return AdaptationHarness(
            dependencies=dependencies,
            llm=llm,
            patch_tool=patch_tool,
            smoke_tool=smoke_tool,
            store=store,
            repository_store=repository_store,
            recorder=recorder,
        )

    return factory


@pytest.fixture
def adaptation_initial_state() -> WorkflowState:
    """Return the small checkpoint-safe initial state for adaptation graph tests."""
    return create_initial_state(
        {
            "schema_version": "1",
            "workflow_id": "workflow-adaptation-1",
            "thread_id": "thread-adaptation-1",
            "request_id": "request-adaptation-1",
            "dataset_profile_id": "dataset-synthetic-sem-1",
        }
    )
