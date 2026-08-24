"""Deterministic fixtures for the vertical-slice workflow LangGraph acceptance matrix."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from tests.fakes import DelegateStep, ScriptedExperimentExecutor
from vision_research_ops.application.runtime import (
    InMemoryApprovalRecorder,
    WorkflowDependencies,
)
from vision_research_ops.application.state import WorkflowState, create_initial_state
from vision_research_ops.domain import CommandSpec, NetworkPolicy, ResourceRequest, RunStatus
from vision_research_ops.ports import (
    ExternalRunStatus,
    FrozenRunSpec,
    RunManifest,
    SubmissionResult,
)

SHA256 = "sha256:" + "a" * 64
COMMIT_SHA = "a" * 40
FIXTURE_NOW = datetime(2026, 8, 9, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class GraphHarness:
    """A graph's explicit fake executor, recorder, and runtime dependencies."""

    dependencies: WorkflowDependencies
    executor: ScriptedExperimentExecutor
    recorder: InMemoryApprovalRecorder


def make_frozen_run(*, experiment_id: str = "exp_fixture_1") -> FrozenRunSpec:
    """Build the smallest valid frozen run accepted by the fake executor."""
    resources = ResourceRequest(
        schema_version="1",
        cpu_cores=1.0,
        memory_mb=1024,
        gpu_count=0,
        walltime_seconds=60,
        scratch_mb=0,
        network_policy=NetworkPolicy.NONE,
    )
    command = CommandSpec(
        schema_version="1",
        executable_id="python",
        argv=["train.py", "--fixture"],
        cwd_ref="fixture/workspace",
    )
    manifest = RunManifest(
        schema_version="1",
        run_id="run_fixture_1",
        experiment_id=experiment_id,
        role="CANDIDATE",
        seed=1,
        repository_commit_sha=COMMIT_SHA,
        environment_digest=SHA256,
        dataset_id="dataset_fixture_1",
        dataset_version="v1",
        dataset_hash=SHA256,
        split_manifest_hash=SHA256,
        preprocessing_hash=SHA256,
        config_hash=SHA256,
        argv=["train.py", "--fixture"],
        resources=resources,
    )
    return FrozenRunSpec(
        schema_version="1",
        run_id="run_fixture_1",
        experiment_id=experiment_id,
        idempotency_key="fixture-run-submission-1",
        command=command,
        resources=resources,
        manifest=manifest,
    )


def make_submission() -> SubmissionResult:
    """Build a successful deterministic fake executor acknowledgement."""
    status = ExternalRunStatus(
        schema_version="1",
        external_job_id="job_fixture_1",
        status=RunStatus.QUEUED,
        observed_at=FIXTURE_NOW,
        raw_status="QUEUED",
    )
    return SubmissionResult(
        schema_version="1",
        external_job_id="job_fixture_1",
        status=status,
        submitted_at=FIXTURE_NOW,
    )


@pytest.fixture
def make_harness() -> Callable[..., GraphHarness]:
    """Create independent dependency sets without state leaking across tests."""

    def factory(
        *,
        executor: ScriptedExperimentExecutor | None = None,
        run_experiment_id: str = "exp_fixture_1",
    ) -> GraphHarness:
        fake_executor = executor or ScriptedExperimentExecutor(
            submissions={"run_fixture_1": make_submission()},
            script={"executor.submit": [DelegateStep()]},
        )
        recorder = InMemoryApprovalRecorder()
        dependencies = WorkflowDependencies(
            executor=fake_executor,
            run_spec=make_frozen_run(experiment_id=run_experiment_id),
            fixture_paper_candidate_ids=("paper_fixture_1", "paper_fixture_2"),
            fixture_repository_snapshot_id="repo_snapshot_fixture_1",
            fixture_repository_id="repo_fixture_1",
            fixture_plan_id="plan_fixture_1",
            fixture_experiment_id="exp_fixture_1",
            fixture_report_id="report_fixture_1",
            approval_recorder=recorder,
            clock=lambda: FIXTURE_NOW,
        )
        return GraphHarness(
            dependencies=dependencies,
            executor=fake_executor,
            recorder=recorder,
        )

    return factory


@pytest.fixture
def initial_state() -> WorkflowState:
    """Build a strict valid state whose thread is safe for InMemorySaver tests."""
    return create_initial_state(
        {
            "schema_version": "1",
            "workflow_id": "workflow_fixture_1",
            "thread_id": "thread_fixture_1",
            "request_id": "request_fixture_1",
            "dataset_profile_id": "dataset_fixture_1",
        }
    )
