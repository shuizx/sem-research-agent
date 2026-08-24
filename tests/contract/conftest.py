"""Small strict factories shared by deterministic port contract port-contract tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from vision_research_ops.domain import (
    ArtifactKind,
    ArtifactRef,
    CommandSpec,
    NetworkPolicy,
    QuerySpec,
    ResourceRequest,
    RunStatus,
)
from vision_research_ops.ports import (
    ArtifactDescriptor,
    CancellationResult,
    ExternalPaperId,
    ExternalRunStatus,
    FrozenRunSpec,
    OperationContext,
    PaperQuery,
    PaperSearchPage,
    RawPaperRecord,
    RepositoryMetadata,
    RepositoryResolution,
    RunManifest,
    SubmissionResult,
    TrackerRunRef,
)

SHA256 = "sha256:" + "a" * 64
COMMIT_SHA = "a" * 40
TIMESTAMP = "2026-08-09T00:00:00Z"


@pytest.fixture
def make_context() -> Callable[..., OperationContext]:
    """Build an explicit deterministic port call context."""

    def factory(
        *,
        idempotency_key: str | None = "idem_1",
        deadline_at: str | None = None,
        correlation_id: str = "corr_1",
    ) -> OperationContext:
        return OperationContext(
            schema_version="1",
            correlation_id=correlation_id,
            workflow_id="wf_1",
            actor_id="actor_1",
            idempotency_key=idempotency_key,
            deadline_at=deadline_at,
            sensitivity="PUBLIC",
        )

    return factory


@pytest.fixture
def make_query() -> Callable[[], PaperQuery]:
    """Build a bounded provider-neutral paper query."""

    def factory() -> PaperQuery:
        return PaperQuery(
            schema_version="1",
            query_id="query_1",
            query_spec=QuerySpec(schema_version="1", keywords=["defect"]),
            page_size=10,
        )

    return factory


@pytest.fixture
def make_raw_record() -> Callable[..., RawPaperRecord]:
    """Build one raw paper record before application normalization."""

    def factory(*, provider_record_id: str = "paper_1") -> RawPaperRecord:
        return RawPaperRecord(
            schema_version="1",
            provider_name="scripted-paper",
            provider_record_id=provider_record_id,
            external_ids=[
                ExternalPaperId(
                    schema_version="1",
                    provider_name="scripted-paper",
                    value=provider_record_id,
                )
            ],
            raw_fields={"title": "Synthetic defect classifier"},
            retrieved_at=TIMESTAMP,
        )

    return factory


@pytest.fixture
def make_page() -> Callable[..., PaperSearchPage]:
    """Build one paper provider page, including the valid empty page case."""

    def factory(
        *,
        records: list[RawPaperRecord] | None = None,
        next_cursor: str | None = None,
        request_id: str = "provider_request_1",
    ) -> PaperSearchPage:
        return PaperSearchPage(
            schema_version="1",
            provider_name="scripted-paper",
            records=[] if records is None else records,
            next_cursor=next_cursor,
            provider_request_id=request_id,
            retrieved_at=TIMESTAMP,
        )

    return factory


@pytest.fixture
def make_artifact_ref() -> Callable[..., ArtifactRef]:
    """Build an immutable test artifact reference."""

    def factory(*, artifact_id: str = "art_1") -> ArtifactRef:
        return ArtifactRef(
            schema_version="1",
            artifact_id=artifact_id,
            kind=ArtifactKind.REPOSITORY_ARCHIVE,
            uri=f"fake://artifacts/{artifact_id}",
            sha256=SHA256,
            size_bytes=3,
            media_type="application/octet-stream",
            created_at=TIMESTAMP,
            producer="test-fake",
            sensitivity="PUBLIC",
        )

    return factory


@pytest.fixture
def make_descriptor() -> Callable[..., ArtifactDescriptor]:
    """Build typed immutable metadata supplied before fake artifact finalization."""

    def factory(*, artifact_id: str = "art_1") -> ArtifactDescriptor:
        return ArtifactDescriptor(
            schema_version="1",
            artifact_id=artifact_id,
            kind=ArtifactKind.REPOSITORY_ARCHIVE,
            media_type="application/octet-stream",
            producer="test-fake",
            sensitivity="PUBLIC",
        )

    return factory


@pytest.fixture
def make_resolution() -> Callable[..., RepositoryResolution]:
    """Build a fixed full-SHA repository resolution."""

    def factory(*, url: str = "https://example.invalid/org/repo") -> RepositoryResolution:
        return RepositoryResolution(
            schema_version="1",
            canonical_url=url,
            provider="LOCAL_FIXTURE",
            owner="org",
            name="repo",
            commit_sha=COMMIT_SHA,
        )

    return factory


@pytest.fixture
def make_metadata() -> Callable[[RepositoryResolution], RepositoryMetadata]:
    """Build read-only repository metadata for a supplied resolution."""

    def factory(resolution: RepositoryResolution) -> RepositoryMetadata:
        return RepositoryMetadata(
            schema_version="1",
            repository=resolution,
            license_spdx="MIT",
            languages={"Python": 100},
            default_branch="main",
        )

    return factory


@pytest.fixture
def make_resource_request() -> Callable[[], ResourceRequest]:
    """Build a no-network resource request suitable for a frozen fake run."""

    def factory() -> ResourceRequest:
        return ResourceRequest(
            schema_version="1",
            cpu_cores=1.0,
            memory_mb=1024,
            gpu_count=0,
            walltime_seconds=60,
            scratch_mb=0,
            network_policy=NetworkPolicy.NONE,
        )

    return factory


@pytest.fixture
def make_command() -> Callable[[], CommandSpec]:
    """Build a safe structured command specification."""

    def factory() -> CommandSpec:
        return CommandSpec(
            schema_version="1",
            executable_id="python",
            argv=["train.py"],
            cwd_ref="fake/workspace",
        )

    return factory


@pytest.fixture
def make_manifest() -> Callable[[], RunManifest]:
    """Build the minimal immutable manifest required by executor and tracker ports."""

    def factory() -> RunManifest:
        return RunManifest(
            schema_version="1",
            run_id="run_1",
            experiment_id="exp_1",
            role="BASELINE",
            seed=1,
            repository_commit_sha=COMMIT_SHA,
            environment_digest=SHA256,
            dataset_id="ds_1",
            dataset_version="v1",
            dataset_hash=SHA256,
            split_manifest_hash=SHA256,
            preprocessing_hash=SHA256,
            config_hash=SHA256,
            resources=ResourceRequest(
                schema_version="1",
                cpu_cores=1.0,
                memory_mb=1024,
                gpu_count=0,
                walltime_seconds=60,
                scratch_mb=0,
                network_policy=NetworkPolicy.NONE,
            ),
        )

    return factory


@pytest.fixture
def make_frozen_run(
    make_command: Callable[[], CommandSpec],
    make_manifest: Callable[[], RunManifest],
    make_resource_request: Callable[[], ResourceRequest],
) -> Callable[..., FrozenRunSpec]:
    """Build a frozen run with the idempotency key owned by executor submission."""

    def factory(*, idempotency_key: str = "run_idempotency_1") -> FrozenRunSpec:
        return FrozenRunSpec(
            schema_version="1",
            run_id="run_1",
            experiment_id="exp_1",
            idempotency_key=idempotency_key,
            command=make_command(),
            resources=make_resource_request(),
            manifest=make_manifest(),
        )

    return factory


@pytest.fixture
def make_external_status() -> Callable[..., ExternalRunStatus]:
    """Build a normalized external status record."""

    def factory(*, status: RunStatus = RunStatus.QUEUED) -> ExternalRunStatus:
        return ExternalRunStatus(
            schema_version="1",
            external_job_id="job_1",
            status=status,
            observed_at=TIMESTAMP,
            raw_status="QUEUED",
        )

    return factory


@pytest.fixture
def make_submission(
    make_external_status: Callable[..., ExternalRunStatus],
) -> Callable[[], SubmissionResult]:
    """Build a deterministic successful executor acknowledgement."""

    def factory() -> SubmissionResult:
        return SubmissionResult(
            schema_version="1",
            external_job_id="job_1",
            status=make_external_status(),
            submitted_at=TIMESTAMP,
        )

    return factory


@pytest.fixture
def make_cancellation(
    make_external_status: Callable[..., ExternalRunStatus],
) -> Callable[[], CancellationResult]:
    """Build a deterministic successful cancellation result."""

    def factory() -> CancellationResult:
        return CancellationResult(
            schema_version="1",
            external_job_id="job_1",
            cancelled=True,
            status=make_external_status(status=RunStatus.CANCELLED),
        )

    return factory


@pytest.fixture
def make_tracker_ref() -> Callable[[], TrackerRunRef]:
    """Build a credential-free fake tracker reference."""

    def factory() -> TrackerRunRef:
        return TrackerRunRef(
            schema_version="1",
            tracker_run_id="tracker_1",
            run_id="run_1",
            uri="fake://tracker/tracker_1",
        )

    return factory
