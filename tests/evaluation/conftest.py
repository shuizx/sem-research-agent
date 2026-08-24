"""Strict training artifact pair and isolated local project harness for evaluation tests."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from vision_research_ops.application.evaluation_runtime import EvaluationDependencies
from vision_research_ops.application.services.evaluation_models import (
    EvaluationInitialInput,
    EvaluationState,
    create_evaluation_state,
)
from vision_research_ops.application.services.evaluation_store import LocalEvaluationStore
from vision_research_ops.application.services.training_freeze import (
    BASELINE_CONFIG_REF,
    CANDIDATE_CONFIG_REF,
    DATASET_REF,
    PREPROCESS_REF,
    SPLIT_REF,
)
from vision_research_ops.application.services.training_models import (
    EpochLoss,
    FrozenRunSpec,
    FrozenTrainingSpec,
    LossPoint,
    PredictionItem,
    TrainingBudgetSpec,
    TrainingCommandSpec,
    TrainingMetrics,
    TrainingPredictions,
    TrainingReviewRecord,
    TrainingRunManifest,
    TrainingRunResult,
    TrainingWorkflowRecord,
    canonical_json_bytes,
    content_hash,
)
from vision_research_ops.application.services.training_store import LocalTrainingStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXED_NOW = datetime(2026, 8, 12, 9, 0, tzinfo=UTC)
TRAINING_WORKFLOW_ID = "workflow-training-evaluation-fixture"
EVALUATION_WORKFLOW_ID = "workflow-evaluation-1"
TEST_SAMPLE_IDS = ("syn-018", "syn-019", "syn-020", "syn-021", "syn-022", "syn-023")
TEST_TRUTH = (2, 3, 0, 1, 2, 3)
DEFAULT_BASELINE = (0, 3, 0, 0, 2, 3)
DEFAULT_CANDIDATE = TEST_TRUTH


@dataclass(slots=True)
class EvaluationHarness:
    """One strict training pair, evaluation dependencies, and minimal initial graph state."""

    project_root: Path
    var_root: Path
    training_store: LocalTrainingStore
    evaluation_store: LocalEvaluationStore
    dependencies: EvaluationDependencies
    state: EvaluationState
    spec: FrozenTrainingSpec


def _fixture_hash(project_root: Path, ref: str) -> str:
    payload = (
        (project_root / Path(*ref.split("/")))
        .read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .encode("utf-8")
    )
    return content_hash(payload)


def _command(run_id: str, role: str, spec_ref: str) -> TrainingCommandSpec:
    output_ref = f"runs/{run_id}"
    return TrainingCommandSpec(
        run_id=run_id,
        role=role,
        spec_ref=spec_ref,
        output_ref=output_ref,
        argv=[
            "-I",
            "fixtures/training/synthetic_linear_cpu.py",
            "--spec-ref",
            f"var/{spec_ref}",
            "--run-id",
            run_id,
            "--role",
            role,
            "--output-ref",
            f"var/{output_ref}",
        ],
        env_keys=[
            "PYTHONIOENCODING",
            "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE",
        ],
    )


def _frozen_spec(project_root: Path) -> FrozenTrainingSpec:
    workflow_id = TRAINING_WORKFLOW_ID
    spec_ref = LocalTrainingStore.spec_ref(workflow_id, 1)
    budget = TrainingBudgetSpec(
        max_epochs=2,
        max_steps=12,
        max_walltime_seconds=10,
    )
    base_commit = "a" * 40
    patch_hash = f"sha256:{'c' * 64}"
    shared: dict[str, object] = {
        "base_commit_sha": base_commit,
        "dataset_id": "dataset-synthetic-sem-1",
        "dataset_version": "synthetic-sem-v1",
        "dataset_content_hash": f"sha256:{'b' * 64}",
        "dataset_ref": DATASET_REF,
        "dataset_ref_hash": _fixture_hash(project_root, DATASET_REF),
        "split_ref": SPLIT_REF,
        "split_hash": _fixture_hash(project_root, SPLIT_REF),
        "preprocess_ref": PREPROCESS_REF,
        "preprocess_hash": _fixture_hash(project_root, PREPROCESS_REF),
        "seed": 17,
        "budget": budget.model_dump(mode="json"),
    }
    baseline = FrozenRunSpec.model_validate(
        {
            **shared,
            "run_id": "evaluation-baseline-run",
            "role": "BASELINE",
            "method": "GLOBAL_STATS_LINEAR",
            "method_config_ref": BASELINE_CONFIG_REF,
            "method_config_hash": _fixture_hash(project_root, BASELINE_CONFIG_REF),
            "command": _command(
                "evaluation-baseline-run",
                "BASELINE",
                spec_ref,
            ).model_dump(mode="json"),
        }
    )
    candidate = FrozenRunSpec.model_validate(
        {
            **shared,
            "run_id": "evaluation-candidate-run",
            "role": "CANDIDATE",
            "method": "GRID4_LINEAR_PATCHED",
            "candidate_patch_revision": 1,
            "candidate_patch_hash": patch_hash,
            "method_config_ref": CANDIDATE_CONFIG_REF,
            "method_config_hash": _fixture_hash(project_root, CANDIDATE_CONFIG_REF),
            "command": _command(
                "evaluation-candidate-run",
                "CANDIDATE",
                spec_ref,
            ).model_dump(mode="json"),
        }
    )
    payload: dict[str, object] = {
        "schema_version": "1",
        "workflow_id": workflow_id,
        "adaptation_workflow_id": "adaptation-evaluation-fixture",
        "revision": 1,
        "base_commit_sha": base_commit,
        "patch_revision": 1,
        "patch_hash": patch_hash,
        "adaptation_result_ref": "adaptations/adaptation-evaluation-fixture/adaptation.json",
        "patch_manifest_ref": "patches/adaptation-evaluation-fixture/r1/manifest.json",
        "smoke_result_ref": "smoke/adaptation-evaluation-fixture/r1/result.json",
        "p3_approval_id": "approval-p3-evaluation-fixture",
        "baseline": baseline.model_dump(mode="json"),
        "candidate": candidate.model_dump(mode="json"),
    }
    payload["spec_hash"] = content_hash(canonical_json_bytes(payload))
    return FrozenTrainingSpec.model_validate(payload)


def _scores(predicted_label: int) -> list[float]:
    scores = [0.1, 0.1, 0.1, 0.1]
    scores[predicted_label] = 0.7
    return scores


def _write_model(store: LocalTrainingStore, ref: str, value: BaseModel) -> None:
    payload = value.model_dump(mode="json")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    path = store.resolve_ref(ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{encoded}\n", encoding="utf-8")


def _write_training_log(
    store: LocalTrainingStore,
    ref: str,
    run: FrozenRunSpec,
    metrics: TrainingMetrics,
) -> None:
    events: list[dict[str, object]] = [
        {"event": "run_started", "role": run.role, "run_id": run.run_id},
        *[
            {
                "epoch": item.epoch,
                "event": "epoch_completed",
                "mean_loss": item.mean_loss,
                "steps": item.steps,
            }
            for item in metrics.epoch_losses
        ],
        {
            "capability": metrics.capability,
            "event": "run_completed",
            "real_pytorch_training": False,
        },
    ]
    encoded = "\n".join(
        json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for event in events
    )
    path = store.resolve_ref(ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{encoded}\n", encoding="utf-8")


def _run_artifacts(
    store: LocalTrainingStore,
    spec: FrozenTrainingSpec,
    run: FrozenRunSpec,
    predicted_labels: Sequence[int],
) -> TrainingRunResult:
    items = [
        PredictionItem(
            sample_id=sample_id,
            true_label=true_label,
            predicted_label=predicted_label,
            scores=_scores(predicted_label),
        )
        for sample_id, true_label, predicted_label in zip(
            TEST_SAMPLE_IDS,
            TEST_TRUTH,
            predicted_labels,
            strict=True,
        )
    ]
    accuracy = sum(item.true_label == item.predicted_label for item in items) / len(items)
    predictions = TrainingPredictions(
        run_id=run.run_id,
        role=run.role,
        spec_hash=spec.spec_hash,
        split_ref=run.split_ref,
        split_hash=run.split_hash,
        items=items,
    )
    metrics = TrainingMetrics(
        run_id=run.run_id,
        role=run.role,
        status="SUCCEEDED",
        spec_hash=spec.spec_hash,
        seed=run.seed,
        budget=run.budget,
        step_losses=[LossPoint(step=1, epoch=1, loss=1.0)],
        epoch_losses=[EpochLoss(epoch=1, steps=1, mean_loss=1.0)],
        initial_loss=1.0,
        final_loss=1.0,
        test_accuracy=accuracy,
        prediction_count=len(items),
    )
    manifest = TrainingRunManifest(
        run_id=run.run_id,
        role=run.role,
        status="SUCCEEDED",
        spec_ref=run.command.spec_ref,
        spec_hash=spec.spec_hash,
        base_commit_sha=run.base_commit_sha,
        candidate_patch_revision=run.candidate_patch_revision,
        candidate_patch_hash=run.candidate_patch_hash,
        dataset_id=run.dataset_id,
        dataset_version=run.dataset_version,
        dataset_content_hash=run.dataset_content_hash,
        dataset_ref=run.dataset_ref,
        dataset_ref_hash=run.dataset_ref_hash,
        split_ref=run.split_ref,
        split_hash=run.split_hash,
        preprocess_ref=run.preprocess_ref,
        preprocess_hash=run.preprocess_hash,
        method=run.method,
        method_config_ref=run.method_config_ref,
        method_config_hash=run.method_config_hash,
        seed=run.seed,
        budget=run.budget,
        command=run.command,
        manifest_ref=store.manifest_ref(run.run_id),
        log_ref=store.log_ref(run.run_id),
        metrics_ref=store.metrics_ref(run.run_id),
        predictions_ref=store.predictions_ref(run.run_id),
        started_at=FIXED_NOW,
        finished_at=FIXED_NOW,
    )
    _write_model(store, manifest.manifest_ref, manifest)
    _write_model(store, manifest.metrics_ref, metrics)
    _write_model(store, manifest.predictions_ref, predictions)
    _write_training_log(store, manifest.log_ref, run, metrics)
    return TrainingRunResult(
        run_id=run.run_id,
        role=run.role,
        spec_hash=spec.spec_hash,
        manifest_ref=manifest.manifest_ref,
        log_ref=manifest.log_ref,
        metrics_ref=manifest.metrics_ref,
        predictions_ref=manifest.predictions_ref,
    )


def _seed_pair(
    project_root: Path,
    store: LocalTrainingStore,
    *,
    baseline_labels: Sequence[int],
    candidate_labels: Sequence[int],
) -> FrozenTrainingSpec:
    spec = _frozen_spec(project_root)
    store.write_spec(spec)
    baseline_result = _run_artifacts(store, spec, spec.baseline, baseline_labels)
    candidate_result = _run_artifacts(store, spec, spec.candidate, candidate_labels)
    review = TrainingReviewRecord(
        approval_id="approval-p4-evaluation-fixture",
        decision="APPROVE",
        gate_id="gate-p4-evaluation-fixture",
        subject_id="submission-p4-evaluation-fixture",
        subject_revision=1,
        spec_hash=spec.spec_hash,
        actor_id="pipeline-reviewer",
        decided_at=FIXED_NOW,
    )
    record = TrainingWorkflowRecord(
        workflow_id=TRAINING_WORKFLOW_ID,
        request_id="request-training-evaluation-fixture",
        adaptation_workflow_id=spec.adaptation_workflow_id,
        status="SUCCEEDED",
        revision=1,
        current_spec_ref=store.spec_ref(TRAINING_WORKFLOW_ID, 1),
        current_spec_hash=spec.spec_hash,
        submission_id="submission-p4-evaluation-fixture",
        approval_id=review.approval_id,
        reviews=[review],
        baseline_result=baseline_result,
        candidate_result=candidate_result,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
    )
    store.write_workflow(record)
    return spec


@pytest.fixture
def make_evaluation_harness() -> Callable[..., EvaluationHarness]:
    """Build an isolated project with exact structured training baseline/candidate outputs."""

    def _make(
        *,
        root: Path,
        baseline_labels: Sequence[int] = DEFAULT_BASELINE,
        candidate_labels: Sequence[int] = DEFAULT_CANDIDATE,
        evaluation_thread_id: str = "thread-evaluation-1",
    ) -> EvaluationHarness:
        project_root = root / "project"
        shutil.copytree(
            PROJECT_ROOT / "fixtures" / "training",
            project_root / "fixtures" / "training",
            dirs_exist_ok=True,
        )
        shutil.copytree(
            PROJECT_ROOT / "fixtures" / "evaluation",
            project_root / "fixtures" / "evaluation",
            dirs_exist_ok=True,
        )
        var_root = project_root / "var"
        training_store = LocalTrainingStore(var_root)
        spec = _seed_pair(
            project_root,
            training_store,
            baseline_labels=baseline_labels,
            candidate_labels=candidate_labels,
        )
        evaluation_store = LocalEvaluationStore(var_root)
        dependencies = EvaluationDependencies(
            training_reader=training_store,
            project_root=project_root,
            store=evaluation_store,
        )
        state = create_evaluation_state(
            EvaluationInitialInput(
                workflow_id=EVALUATION_WORKFLOW_ID,
                thread_id=evaluation_thread_id,
                request_id="request-evaluation-1",
                training_workflow_id=TRAINING_WORKFLOW_ID,
            )
        )
        return EvaluationHarness(
            project_root=project_root,
            var_root=var_root,
            training_store=training_store,
            evaluation_store=evaluation_store,
            dependencies=dependencies,
            state=state,
            spec=spec,
        )

    return _make
