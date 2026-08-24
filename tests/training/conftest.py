"""Controlled accepted-adaptation and local project harness for training tests."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vision_research_ops.adapters.execution.local_training import LocalTrainingExecutor
from vision_research_ops.application.runtime import InMemoryApprovalRecorder
from vision_research_ops.application.services.adaptation_models import (
    AttemptEvidence,
    PatchArtifactRecord,
    PatchChangeRecord,
    PatchReviewRecord,
    SmokeCommandRecord,
    SmokeResultRecord,
    SmokeStageRecord,
)
from vision_research_ops.application.services.adaptation_store import LocalAdaptationStore
from vision_research_ops.application.services.training_freeze import (
    DATASET_REF,
    PREPROCESS_REF,
    SPLIT_REF,
)
from vision_research_ops.application.services.training_models import (
    TrainingBudgetSpec,
    TrainingInput,
    content_hash,
)
from vision_research_ops.application.services.training_store import LocalTrainingStore
from vision_research_ops.application.state import (
    InitialWorkflowInput,
    WorkflowState,
    create_initial_state,
)
from vision_research_ops.application.training_runtime import TrainingDependencies
from vision_research_ops.domain import ValidationStage, ValidationStatus

FIXED_NOW = datetime(2026, 8, 12, 8, 0, tzinfo=UTC)
BASE_COMMIT = "a" * 40
DATASET_CONTENT_HASH = f"sha256:{'b' * 64}"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SequenceCancellation:
    """Return a short deterministic cancellation sequence at run boundaries."""

    def __init__(self, values: tuple[bool, ...] = (False, False)) -> None:
        self._values = values
        self.call_count = 0

    def is_cancelled(self) -> bool:
        index = min(self.call_count, len(self._values) - 1)
        self.call_count += 1
        return self._values[index]


@dataclass(slots=True)
class TrainingHarness:
    """One real local executor plus exact evidence and injected graph dependencies."""

    project_root: Path
    adaptation_store: LocalAdaptationStore
    store: LocalTrainingStore
    trainer: LocalTrainingExecutor
    recorder: InMemoryApprovalRecorder
    cancellation: SequenceCancellation
    training_input: TrainingInput | dict[str, object]
    dependencies: TrainingDependencies


def _fixture_hash(project_root: Path, ref: str) -> str:
    path = project_root / Path(*ref.split("/"))
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    return content_hash(normalized)


def _seed_accepted_p3(store: LocalAdaptationStore) -> str:
    workflow_id = "adaptation-training-fixture"
    patch_ref = f"patches/{workflow_id}/r1/change.patch"
    patch_bytes = b"--- a/sem_adaptation.json\n+++ b/sem_adaptation.json\n@@ fixture accepted\n"
    patch_path = store.resolve_ref(patch_ref)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(patch_bytes)
    patch_hash = content_hash(patch_bytes)
    patch = PatchArtifactRecord(
        workflow_id=workflow_id,
        attempt_id="adaptation-attempt-training-r1",
        attempt_number=1,
        plan_id="adaptation-plan-training",
        plan_revision=1,
        repository_id="repository-training-fixture",
        base_commit_sha=BASE_COMMIT,
        dataset_version="synthetic-sem-v1",
        patch_hash=patch_hash,
        workspace_ref=f"workspaces/{workflow_id}/r1",
        patch_ref=patch_ref,
        manifest_ref=store.patch_manifest_ref(workflow_id, 1),
        changes=[
            PatchChangeRecord(
                path="sem_adaptation.json",
                operation="MODIFY",
                field_paths=[
                    "/input/channels",
                    "/model/num_classes",
                    "/data/label_mapping",
                    "/data/group_split_key",
                    "/metrics/names",
                    "/metrics/output_file",
                ],
                before_hash=f"sha256:{'c' * 64}",
                after_hash=f"sha256:{'d' * 64}",
            )
        ],
        created_at=FIXED_NOW,
    )
    store.write_patch_record(patch)
    stages: list[SmokeStageRecord] = []
    for stage in (
        ValidationStage.STATIC_POLICY,
        ValidationStage.IMPORT,
        ValidationStage.ONE_BATCH,
        ValidationStage.BOUNDED_OVERFIT,
    ):
        command = SmokeCommandRecord(
            executable_id="python-current",
            argv=["-I", "fixture_probe.py", "--stage", stage.value],
            cwd_ref=patch.workspace_ref,
        )
        stages.append(
            SmokeStageRecord(
                stage=stage,
                status=ValidationStatus.PASSED,
                exit_code=0,
                command=command,
                command_digest=content_hash(stage.value.encode()),
                evidence={"fixture_stage_passed": True},
                log_ref=f"smoke/{workflow_id}/r1/{stage.value.casefold()}.json",
                started_at=FIXED_NOW,
                finished_at=FIXED_NOW,
            )
        )
    smoke = SmokeResultRecord(
        workflow_id=workflow_id,
        attempt_id=patch.attempt_id,
        plan_revision=1,
        repository_id=patch.repository_id,
        base_commit_sha=BASE_COMMIT,
        dataset_version="synthetic-sem-v1",
        patch_hash=patch_hash,
        status="PASSED",
        stages=stages,
        result_ref=store.smoke_result_ref(workflow_id, 1),
        retryable=False,
        capability_boundary="FIXTURE_CONTRACT_PROBE_NO_TORCH",
        started_at=FIXED_NOW,
        finished_at=FIXED_NOW,
    )
    store.write_smoke_result(smoke)
    from vision_research_ops.application.services.adaptation_models import AdaptationResult

    result = AdaptationResult(
        workflow_id=workflow_id,
        request_id="request-adaptation-training",
        repository_workflow_id="repository-workflow-training",
        status="ACCEPTED",
        repository_id=patch.repository_id,
        repository_url="https://github.com/sem-research-agent/plain-pytorch-fixture",
        base_commit_sha=BASE_COMMIT,
        dataset_id="dataset-synthetic-sem-1",
        dataset_version="synthetic-sem-v1",
        dataset_content_hash=DATASET_CONTENT_HASH,
        dataset_kind="SYNTHETIC_SEM_FIXTURE",
        repository_kind="CONTROLLED_PLAIN_PYTORCH_FIXTURE",
        plan_id=patch.plan_id,
        plan_revision=1,
        plan_ref=f"adaptations/{workflow_id}/plan-r1.json",
        attempts=[
            AttemptEvidence(
                attempt_id=patch.attempt_id,
                plan_revision=1,
                patch_hash=patch_hash,
                patch_ref=patch.patch_ref,
                patch_manifest_ref=patch.manifest_ref,
                smoke_ref=smoke.result_ref,
                smoke_status="PASSED",
            )
        ],
        reviews=[
            PatchReviewRecord(
                approval_id="approval-p3-training",
                decision="APPROVE",
                gate_id="gate-p3-training",
                subject_id="adaptation-patch-training",
                subject_revision=1,
                patch_hash=patch_hash,
                actor_id="pipeline-reviewer",
                decided_at=FIXED_NOW,
            )
        ],
        gate_id="gate-p3-training",
        gate_revision=1,
        gate_subject_id="adaptation-patch-training",
        gate_patch_hash=patch_hash,
        accepted_patch_hash=patch_hash,
        approval_id="approval-p3-training",
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
    )
    store.write_result(result)
    return patch_hash


@pytest.fixture
def training_initial_state() -> WorkflowState:
    """Return a strict small state for one training workflow thread."""
    return create_initial_state(
        InitialWorkflowInput(
            workflow_id="workflow-training-1",
            thread_id="thread-training-1",
            request_id="request-training-1",
            dataset_profile_id="dataset-synthetic-sem-1",
        )
    )


@pytest.fixture
def make_training_harness() -> Callable[..., TrainingHarness]:
    """Build a real fixture project and exact accepted adaptation evidence."""

    def _make(
        *,
        root: Path,
        cancellation_values: tuple[bool, ...] = (False, False),
        input_updates: dict[str, object] | None = None,
    ) -> TrainingHarness:
        project_root = root / "project"
        fixture_root = project_root / "fixtures" / "training"
        if not fixture_root.exists():
            fixture_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(PROJECT_ROOT / "fixtures" / "training", fixture_root)
        var_root = project_root / "var"
        adaptation_store = LocalAdaptationStore(var_root)
        result_path = adaptation_store.resolve_ref(
            adaptation_store.result_ref("adaptation-training-fixture")
        )
        patch_hash = (
            _seed_accepted_p3(adaptation_store)
            if not result_path.exists()
            else str(
                adaptation_store.load_result("adaptation-training-fixture").accepted_patch_hash
            )
        )
        training_input = TrainingInput(
            adaptation_workflow_id="adaptation-training-fixture",
            base_commit_sha=BASE_COMMIT,
            patch_revision=1,
            patch_hash=patch_hash,
            dataset_id="dataset-synthetic-sem-1",
            dataset_version="synthetic-sem-v1",
            dataset_content_hash=DATASET_CONTENT_HASH,
            dataset_ref=DATASET_REF,
            dataset_ref_hash=_fixture_hash(project_root, DATASET_REF),
            split_ref=SPLIT_REF,
            split_hash=_fixture_hash(project_root, SPLIT_REF),
            preprocess_ref=PREPROCESS_REF,
            preprocess_hash=_fixture_hash(project_root, PREPROCESS_REF),
            seed=17,
            budget=TrainingBudgetSpec(
                max_epochs=4,
                max_steps=48,
                max_walltime_seconds=10,
            ),
        )
        raw_input: TrainingInput | dict[str, object] = training_input
        if input_updates:
            raw_input = {**training_input.model_dump(mode="json"), **input_updates}
        store = LocalTrainingStore(var_root)
        trainer = LocalTrainingExecutor(project_root=project_root, store=store)
        recorder = InMemoryApprovalRecorder()
        cancellation = SequenceCancellation(cancellation_values)
        dependencies = TrainingDependencies(
            adaptation_reader=adaptation_store,
            training_input=raw_input,
            project_root=project_root,
            store=store,
            trainer=trainer,
            approval_recorder=recorder,
            cancellation=cancellation,
            clock=lambda: FIXED_NOW,
        )
        return TrainingHarness(
            project_root=project_root,
            adaptation_store=adaptation_store,
            store=store,
            trainer=trainer,
            recorder=recorder,
            cancellation=cancellation,
            training_input=raw_input,
            dependencies=dependencies,
        )

    return _make
