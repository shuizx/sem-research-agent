"""Validate training artifacts and assemble canonical evaluation results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from ..evaluation_runtime import EvaluationDependencies
from .evaluation_metrics import compute_metric_delta, compute_metrics, decide_conclusion
from .evaluation_models import (
    EVALUATION_POLICY_REF,
    FIXED_LIMITATIONS,
    ArtifactDigest,
    CheckStatus,
    ComparabilityCheck,
    ComparabilityRecord,
    DatasetFixture,
    EvaluationInitialInput,
    EvaluationPolicy,
    EvaluationResult,
    EvaluationState,
    ReasonCode,
    RunProvenance,
    SplitFixture,
    TrainingProvenance,
    canonical_json_bytes,
    content_hash,
)
from .evaluation_training_log import validate_training_log
from .training_freeze import EXPECTED_FIXTURE_HASHES
from .training_models import (
    FrozenRunSpec,
    FrozenTrainingSpec,
    TrainingMetrics,
    TrainingPredictions,
    TrainingRunManifest,
    TrainingRunResult,
    TrainingWorkflowRecord,
)

_MAX_INPUT_BYTES = 1_048_576
type _CheckName = Literal[
    "training_workflow_complete",
    "frozen_spec_integrity",
    "artifact_schema_and_identity",
    "dataset_version_content",
    "split",
    "preprocess",
    "seed",
    "budget",
    "label_vocabulary",
    "test_samples_and_truth",
]
_CHECK_NAMES: tuple[_CheckName, ...] = (
    "training_workflow_complete",
    "frozen_spec_integrity",
    "artifact_schema_and_identity",
    "dataset_version_content",
    "split",
    "preprocess",
    "seed",
    "budget",
    "label_vocabulary",
    "test_samples_and_truth",
)
_REASON_ORDER: tuple[ReasonCode, ...] = (
    "TRAINING_WORKFLOW_UNAVAILABLE",
    "TRAINING_WORKFLOW_SCHEMA_INVALID",
    "TRAINING_WORKFLOW_NOT_SUCCEEDED",
    "TRAINING_PAIR_INCOMPLETE",
    "TRAINING_APPROVAL_PROVENANCE_INVALID",
    "FROZEN_SPEC_UNAVAILABLE",
    "FROZEN_SPEC_SCHEMA_INVALID",
    "FROZEN_SPEC_IDENTITY_MISMATCH",
    "TRAINING_ARTIFACT_MISSING",
    "TRAINING_LOG_INVALID",
    "MANIFEST_SCHEMA_INVALID",
    "METRICS_SCHEMA_INVALID",
    "PREDICTIONS_SCHEMA_INVALID",
    "RUN_ARTIFACT_IDENTITY_MISMATCH",
    "DATASET_FIXTURE_INVALID",
    "DATASET_MISMATCH",
    "SPLIT_FIXTURE_INVALID",
    "SPLIT_MISMATCH",
    "PREPROCESS_MISMATCH",
    "SEED_MISMATCH",
    "BUDGET_MISMATCH",
    "LABEL_VOCABULARY_MISMATCH",
    "TEST_SAMPLE_SET_MISMATCH",
    "TEST_SAMPLE_TRUTH_MISMATCH",
    "PREDICTED_LABEL_SCORE_MISMATCH",
    "TRAINING_METRIC_PREDICTION_MISMATCH",
    "SEVERE_CLASS_RECALL_REGRESSION",
    "MACRO_F1_REGRESSION",
    "PRACTICAL_MACRO_F1_IMPROVEMENT",
    "IMPROVEMENT_RULE_NOT_MET",
)


class EvaluationEngineError(Exception):
    """Sanitized configuration failure outside the training validity conclusion."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class EvaluationComputation:
    """One deterministic result plus whether both training artifact sets loaded."""

    result: EvaluationResult
    outputs_loaded: bool


@dataclass(frozen=True, slots=True)
class _LoadedRun:
    spec: FrozenRunSpec
    result: TrainingRunResult
    manifest: TrainingRunManifest
    metrics: TrainingMetrics
    predictions: TrainingPredictions


class _BuildEvidence:
    def __init__(self) -> None:
        self.artifacts: list[ArtifactDigest] = []
        self.checks: dict[_CheckName, CheckStatus] = {
            name: "NOT_EVALUATED" for name in _CHECK_NAMES
        }
        self.reasons: list[ReasonCode] = []

    def add_reason(self, reason: ReasonCode, *, check: _CheckName) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)
        self.checks[check] = "FAIL"

    def pass_check(self, check: _CheckName) -> None:
        if self.checks[check] != "FAIL":
            self.checks[check] = "PASS"

    def add_artifact(self, *, name: str, ref: str, payload: bytes | None) -> None:
        self.artifacts.append(
            ArtifactDigest(
                name=name,
                ref=ref,
                status="MISSING" if payload is None else "HASHED",
                content_hash=None if payload is None else content_hash(payload),
                size_bytes=None if payload is None else len(payload),
            )
        )

    def comparability(self) -> ComparabilityRecord:
        checks = [ComparabilityCheck(name=name, status=self.checks[name]) for name in _CHECK_NAMES]
        return ComparabilityRecord(
            status="INVALID" if any(item.status == "FAIL" for item in checks) else "VALID",
            checks=checks,
        )

    def ordered_reasons(self) -> list[ReasonCode]:
        return [reason for reason in _REASON_ORDER if reason in self.reasons]


def _normalized_text_bytes(payload: bytes) -> bytes:
    try:
        return payload.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    except UnicodeDecodeError as error:
        raise EvaluationEngineError("EVALUATION_POLICY_INVALID") from error


def _safe_project_file(project_root: Path, ref: str) -> Path:
    if (
        not ref
        or ref != ref.strip()
        or "\\" in ref
        or ref.startswith("/")
        or ":" in ref
        or "%" in ref
        or any(part in {"", ".", ".."} for part in ref.split("/"))
    ):
        raise ValueError("project artifact ref is not canonical")
    root = project_root.resolve()
    path = (root / Path(*ref.split("/"))).resolve()
    if (
        not path.is_relative_to(root)
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size > _MAX_INPUT_BYTES
    ):
        raise ValueError("project artifact is missing or unsafe")
    return path


def _read_policy(dependencies: EvaluationDependencies) -> tuple[EvaluationPolicy, bytes]:
    try:
        path = _safe_project_file(dependencies.project_root, dependencies.policy_ref)
        payload = _normalized_text_bytes(path.read_bytes())
        return EvaluationPolicy.model_validate_json(payload), payload
    except (OSError, TypeError, ValueError, ValidationError) as error:
        raise EvaluationEngineError("EVALUATION_POLICY_INVALID") from error


def _read_var(
    evidence: _BuildEvidence,
    dependencies: EvaluationDependencies,
    *,
    name: str,
    ref: str,
) -> bytes | None:
    try:
        path = dependencies.training_reader.resolve_ref(ref)
        if not path.is_file() or path.stat().st_size > _MAX_INPUT_BYTES:
            raise OSError
        payload = path.read_bytes()
    except (OSError, TypeError, ValueError):
        evidence.add_artifact(name=name, ref=ref, payload=None)
        return None
    evidence.add_artifact(name=name, ref=ref, payload=payload)
    return payload


def _read_project(
    evidence: _BuildEvidence,
    dependencies: EvaluationDependencies,
    *,
    name: str,
    ref: str,
) -> bytes | None:
    try:
        payload = _normalized_text_bytes(
            _safe_project_file(dependencies.project_root, ref).read_bytes()
        )
    except (EvaluationEngineError, OSError, TypeError, ValueError):
        evidence.add_artifact(name=name, ref=ref, payload=None)
        return None
    evidence.add_artifact(name=name, ref=ref, payload=payload)
    return payload


def _run_provenance(spec: FrozenRunSpec, result: TrainingRunResult) -> RunProvenance:
    return RunProvenance(
        run_id=spec.run_id,
        role=spec.role,
        method=spec.method,
        method_config_ref=spec.method_config_ref,
        method_config_hash=spec.method_config_hash,
        manifest_ref=result.manifest_ref,
        metrics_ref=result.metrics_ref,
        predictions_ref=result.predictions_ref,
    )


def _provenance(
    training_workflow_id: str,
    training_workflow_ref: str,
    *,
    record: TrainingWorkflowRecord | None = None,
    spec: FrozenTrainingSpec | None = None,
) -> TrainingProvenance:
    baseline = None
    candidate = None
    if (
        spec is not None
        and record is not None
        and record.baseline_result is not None
        and record.candidate_result is not None
    ):
        baseline = _run_provenance(spec.baseline, record.baseline_result)
        candidate = _run_provenance(spec.candidate, record.candidate_result)
    return TrainingProvenance(
        training_workflow_id=training_workflow_id,
        training_workflow_ref=training_workflow_ref,
        frozen_spec_ref=None if record is None else record.current_spec_ref,
        frozen_spec_hash=None if spec is None else spec.spec_hash,
        base_commit_sha=None if spec is None else spec.base_commit_sha,
        patch_revision=None if spec is None else spec.patch_revision,
        patch_hash=None if spec is None else spec.patch_hash,
        adaptation_workflow_id=None if spec is None else spec.adaptation_workflow_id,
        adaptation_result_ref=None if spec is None else spec.adaptation_result_ref,
        patch_manifest_ref=None if spec is None else spec.patch_manifest_ref,
        smoke_result_ref=None if spec is None else spec.smoke_result_ref,
        p3_approval_id=None if spec is None else spec.p3_approval_id,
        p4_approval_id=None if record is None else record.approval_id,
        baseline=baseline,
        candidate=candidate,
        training_capability=None if spec is None else spec.baseline.capability,
    )


def _evaluation_id(
    workflow_id: str,
    policy_hash: str,
    provenance: TrainingProvenance,
    artifacts: list[ArtifactDigest],
    comparability: ComparabilityRecord,
    reasons: list[ReasonCode],
) -> str:
    payload = {
        "workflow_id": workflow_id,
        "policy_hash": policy_hash,
        "provenance": provenance.model_dump(mode="json"),
        "input_artifacts": [item.model_dump(mode="json") for item in artifacts],
        "comparability": comparability.model_dump(mode="json"),
        "invalid_reason_codes": reasons,
    }
    digest = content_hash(canonical_json_bytes(payload)).removeprefix("sha256:")
    return f"evaluation-{digest[:24]}"


def _invalid_result(
    *,
    workflow_id: str,
    policy: EvaluationPolicy,
    policy_hash: str,
    provenance: TrainingProvenance,
    evidence: _BuildEvidence,
) -> EvaluationResult:
    comparability = evidence.comparability()
    reasons = evidence.ordered_reasons()
    if not reasons:
        raise EvaluationEngineError("EVALUATION_INTERNAL_INVALID_STATE")
    return EvaluationResult(
        workflow_id=workflow_id,
        evaluation_id=_evaluation_id(
            workflow_id,
            policy_hash,
            provenance,
            evidence.artifacts,
            comparability,
            reasons,
        ),
        limitations=list(FIXED_LIMITATIONS),
        policy_hash=policy_hash,
        policy=policy,
        provenance=provenance,
        input_artifacts=evidence.artifacts,
        comparability=comparability,
        baseline_metrics=None,
        candidate_metrics=None,
        deltas=None,
        conclusion="INVALID",
        reason_codes=reasons,
        evaluation_ref=f"reports/{workflow_id}/evaluation.json",
        report_ref=f"reports/{workflow_id}/report.md",
    )


def _valid_result(
    *,
    workflow_id: str,
    policy: EvaluationPolicy,
    policy_hash: str,
    provenance: TrainingProvenance,
    evidence: _BuildEvidence,
    baseline: _LoadedRun,
    candidate: _LoadedRun,
) -> EvaluationResult:
    baseline_metrics = compute_metrics(baseline.predictions, policy)
    candidate_metrics = compute_metrics(candidate.predictions, policy)
    delta = compute_metric_delta(baseline_metrics, candidate_metrics, policy)
    conclusion, reasons = decide_conclusion(delta, policy)
    comparability = evidence.comparability()
    reason_list = list(reasons)
    evaluation_id = _evaluation_id(
        workflow_id,
        policy_hash,
        provenance,
        evidence.artifacts,
        comparability,
        [],
    )
    return EvaluationResult(
        workflow_id=workflow_id,
        evaluation_id=evaluation_id,
        limitations=list(FIXED_LIMITATIONS),
        policy_hash=policy_hash,
        policy=policy,
        provenance=provenance,
        input_artifacts=evidence.artifacts,
        comparability=comparability,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        deltas=delta,
        conclusion=conclusion,
        reason_codes=reason_list,
        evaluation_ref=f"reports/{workflow_id}/evaluation.json",
        report_ref=f"reports/{workflow_id}/report.md",
    )


def _load_run(
    evidence: _BuildEvidence,
    dependencies: EvaluationDependencies,
    *,
    name: Literal["baseline", "candidate"],
    spec: FrozenRunSpec,
    result: TrainingRunResult,
) -> _LoadedRun | None:
    manifest_payload = _read_var(
        evidence,
        dependencies,
        name=f"{name}_manifest",
        ref=result.manifest_ref,
    )
    metrics_payload = _read_var(
        evidence,
        dependencies,
        name=f"{name}_metrics",
        ref=result.metrics_ref,
    )
    predictions_payload = _read_var(
        evidence,
        dependencies,
        name=f"{name}_predictions",
        ref=result.predictions_ref,
    )
    if manifest_payload is None or metrics_payload is None or predictions_payload is None:
        evidence.add_reason("TRAINING_ARTIFACT_MISSING", check="artifact_schema_and_identity")
        return None
    try:
        manifest = TrainingRunManifest.model_validate_json(manifest_payload)
    except (TypeError, ValueError, ValidationError):
        evidence.add_reason("MANIFEST_SCHEMA_INVALID", check="artifact_schema_and_identity")
        return None
    try:
        metrics = TrainingMetrics.model_validate_json(metrics_payload)
    except (TypeError, ValueError, ValidationError):
        evidence.add_reason("METRICS_SCHEMA_INVALID", check="artifact_schema_and_identity")
        return None
    log_payload = _read_var(
        evidence,
        dependencies,
        name=f"{name}_log",
        ref=manifest.log_ref,
    )
    if log_payload is None:
        evidence.add_reason("TRAINING_LOG_INVALID", check="artifact_schema_and_identity")
        return None
    try:
        validate_training_log(log_payload, run=spec, metrics=metrics)
    except (UnicodeError, ValueError):
        evidence.add_reason("TRAINING_LOG_INVALID", check="artifact_schema_and_identity")
        return None
    try:
        predictions = TrainingPredictions.model_validate_json(predictions_payload)
    except (TypeError, ValueError, ValidationError):
        evidence.add_reason("PREDICTIONS_SCHEMA_INVALID", check="artifact_schema_and_identity")
        return None
    loaded = _LoadedRun(
        spec=spec,
        result=result,
        manifest=manifest,
        metrics=metrics,
        predictions=predictions,
    )
    if not _run_identity_is_exact(loaded):
        evidence.add_reason(
            "RUN_ARTIFACT_IDENTITY_MISMATCH",
            check="artifact_schema_and_identity",
        )
        return None
    for item in predictions.items:
        if item.scores[item.predicted_label] < max(item.scores):
            evidence.add_reason(
                "PREDICTED_LABEL_SCORE_MISMATCH",
                check="artifact_schema_and_identity",
            )
            return None
    observed_accuracy = sum(
        item.predicted_label == item.true_label for item in predictions.items
    ) / len(predictions.items)
    if metrics.prediction_count != len(predictions.items) or not isclose(
        metrics.test_accuracy, observed_accuracy, rel_tol=0.0, abs_tol=1e-12
    ):
        evidence.add_reason(
            "TRAINING_METRIC_PREDICTION_MISMATCH",
            check="artifact_schema_and_identity",
        )
        return None
    return loaded


def _run_identity_is_exact(run: _LoadedRun) -> bool:
    spec = run.spec
    result = run.result
    manifest = run.manifest
    metrics = run.metrics
    predictions = run.predictions
    expected_refs = (
        result.run_id == spec.run_id,
        result.role == spec.role,
        result.manifest_ref == manifest.manifest_ref,
        result.log_ref == manifest.log_ref,
        result.metrics_ref == manifest.metrics_ref,
        result.predictions_ref == manifest.predictions_ref,
    )
    frozen_fields = (
        manifest.run_id == spec.run_id,
        manifest.role == spec.role,
        manifest.spec_ref == spec.command.spec_ref,
        manifest.base_commit_sha == spec.base_commit_sha,
        manifest.candidate_patch_revision == spec.candidate_patch_revision,
        manifest.candidate_patch_hash == spec.candidate_patch_hash,
        manifest.dataset_id == spec.dataset_id,
        manifest.dataset_version == spec.dataset_version,
        manifest.dataset_content_hash == spec.dataset_content_hash,
        manifest.dataset_ref == spec.dataset_ref,
        manifest.dataset_ref_hash == spec.dataset_ref_hash,
        manifest.split_ref == spec.split_ref,
        manifest.split_hash == spec.split_hash,
        manifest.preprocess_ref == spec.preprocess_ref,
        manifest.preprocess_hash == spec.preprocess_hash,
        manifest.method == spec.method,
        manifest.method_config_ref == spec.method_config_ref,
        manifest.method_config_hash == spec.method_config_hash,
        manifest.seed == spec.seed,
        manifest.budget == spec.budget,
        manifest.command == spec.command,
        manifest.capability == spec.capability == result.capability,
        not manifest.real_pytorch_training,
        not result.real_pytorch_training,
    )
    metric_fields = (
        metrics.run_id == spec.run_id,
        metrics.role == spec.role,
        metrics.spec_hash == result.spec_hash,
        metrics.seed == spec.seed,
        metrics.budget == spec.budget,
        metrics.capability == spec.capability,
        not metrics.real_pytorch_training,
    )
    prediction_fields = (
        predictions.run_id == spec.run_id,
        predictions.role == spec.role,
        predictions.spec_hash == result.spec_hash,
        predictions.split_ref == spec.split_ref,
        predictions.split_hash == spec.split_hash,
        predictions.capability == spec.capability,
        not predictions.real_pytorch_training,
    )
    return all((*expected_refs, *frozen_fields, *metric_fields, *prediction_fields))


def _exact_pair_identity(
    evidence: _BuildEvidence,
    spec: FrozenTrainingSpec,
    record: TrainingWorkflowRecord,
    baseline: _LoadedRun,
    candidate: _LoadedRun,
) -> None:
    expected_hash = spec.spec_hash
    if not all(
        value == expected_hash
        for value in (
            record.current_spec_hash,
            baseline.result.spec_hash,
            baseline.manifest.spec_hash,
            baseline.metrics.spec_hash,
            baseline.predictions.spec_hash,
            candidate.result.spec_hash,
            candidate.manifest.spec_hash,
            candidate.metrics.spec_hash,
            candidate.predictions.spec_hash,
        )
    ):
        evidence.add_reason(
            "RUN_ARTIFACT_IDENTITY_MISMATCH",
            check="artifact_schema_and_identity",
        )
        return
    evidence.pass_check("artifact_schema_and_identity")


def _validate_project_fixtures(
    evidence: _BuildEvidence,
    dependencies: EvaluationDependencies,
    policy: EvaluationPolicy,
    baseline: _LoadedRun,
    candidate: _LoadedRun,
) -> tuple[DatasetFixture | None, SplitFixture | None]:
    baseline_spec = baseline.spec
    candidate_spec = candidate.spec
    dataset_payload = _read_project(
        evidence,
        dependencies,
        name="dataset_fixture",
        ref=baseline_spec.dataset_ref,
    )
    dataset: DatasetFixture | None = None
    if dataset_payload is None:
        evidence.add_reason("DATASET_FIXTURE_INVALID", check="dataset_version_content")
    else:
        try:
            dataset = DatasetFixture.model_validate_json(dataset_payload)
        except (TypeError, ValueError, ValidationError):
            evidence.add_reason("DATASET_FIXTURE_INVALID", check="dataset_version_content")
        observed = content_hash(dataset_payload)
        dataset_mismatch = (
            observed != EXPECTED_FIXTURE_HASHES.get(baseline_spec.dataset_ref)
            or observed != baseline_spec.dataset_ref_hash
            or observed != candidate_spec.dataset_ref_hash
            or (
                dataset is not None
                and not all(
                    (
                        dataset.dataset_id == baseline_spec.dataset_id == candidate_spec.dataset_id,
                        dataset.dataset_version
                        == baseline_spec.dataset_version
                        == candidate_spec.dataset_version,
                        baseline_spec.dataset_content_hash == candidate_spec.dataset_content_hash,
                        dataset.dataset_id == policy.dataset_id,
                        dataset.dataset_version == policy.dataset_version,
                    )
                )
            )
        )
        if dataset_mismatch:
            evidence.add_reason("DATASET_MISMATCH", check="dataset_version_content")
        elif dataset is not None:
            evidence.pass_check("dataset_version_content")

    split_payload = _read_project(
        evidence,
        dependencies,
        name="split_fixture",
        ref=baseline_spec.split_ref,
    )
    split: SplitFixture | None = None
    if split_payload is None:
        evidence.add_reason("SPLIT_FIXTURE_INVALID", check="split")
    else:
        try:
            split = SplitFixture.model_validate_json(split_payload)
        except (TypeError, ValueError, ValidationError):
            evidence.add_reason("SPLIT_FIXTURE_INVALID", check="split")
        observed = content_hash(split_payload)
        split_mismatch = (
            observed != EXPECTED_FIXTURE_HASHES.get(baseline_spec.split_ref)
            or observed != baseline_spec.split_hash
            or observed != candidate_spec.split_hash
            or baseline.predictions.split_ref != candidate.predictions.split_ref
            or baseline.predictions.split_hash != candidate.predictions.split_hash
            or (split is not None and split.dataset_id != baseline_spec.dataset_id)
        )
        if split_mismatch:
            evidence.add_reason("SPLIT_MISMATCH", check="split")
        elif split is not None:
            evidence.pass_check("split")

    preprocess_payload = _read_project(
        evidence,
        dependencies,
        name="preprocess_fixture",
        ref=baseline_spec.preprocess_ref,
    )
    if preprocess_payload is None:
        evidence.add_reason("PREPROCESS_MISMATCH", check="preprocess")
    else:
        observed = content_hash(preprocess_payload)
        try:
            decoded = json.loads(preprocess_payload)
        except (TypeError, ValueError):
            decoded = None
        if (
            observed != EXPECTED_FIXTURE_HASHES.get(baseline_spec.preprocess_ref)
            or observed != baseline_spec.preprocess_hash
            or observed != candidate_spec.preprocess_hash
            or not isinstance(decoded, dict)
            or decoded.get("schema_version") != "1"
        ):
            evidence.add_reason("PREPROCESS_MISMATCH", check="preprocess")
        else:
            evidence.pass_check("preprocess")

    for prefix, run in (("baseline", baseline), ("candidate", candidate)):
        payload = _read_project(
            evidence,
            dependencies,
            name=f"{prefix}_method_config",
            ref=run.spec.method_config_ref,
        )
        if (
            payload is None
            or content_hash(payload) != EXPECTED_FIXTURE_HASHES.get(run.spec.method_config_ref)
            or content_hash(payload) != run.spec.method_config_hash
        ):
            evidence.add_reason(
                "RUN_ARTIFACT_IDENTITY_MISMATCH",
                check="artifact_schema_and_identity",
            )
    return dataset, split


def _validate_pair_comparability(
    evidence: _BuildEvidence,
    policy: EvaluationPolicy,
    dataset: DatasetFixture | None,
    split: SplitFixture | None,
    baseline: _LoadedRun,
    candidate: _LoadedRun,
) -> None:
    baseline_spec = baseline.spec
    candidate_spec = candidate.spec
    if baseline_spec.seed != candidate_spec.seed:
        evidence.add_reason("SEED_MISMATCH", check="seed")
    else:
        evidence.pass_check("seed")
    if baseline_spec.budget != candidate_spec.budget:
        evidence.add_reason("BUDGET_MISMATCH", check="budget")
    else:
        evidence.pass_check("budget")

    if (
        dataset is None
        or policy.label_ids != list(range(len(dataset.labels)))
        or policy.label_names != dataset.labels
    ):
        evidence.add_reason("LABEL_VOCABULARY_MISMATCH", check="label_vocabulary")
    else:
        evidence.pass_check("label_vocabulary")

    baseline_items = {item.sample_id: item for item in baseline.predictions.items}
    candidate_items = {item.sample_id: item for item in candidate.predictions.items}
    expected_ids = set() if split is None else set(split.test)
    if split is None or set(baseline_items) != expected_ids or set(candidate_items) != expected_ids:
        evidence.add_reason("TEST_SAMPLE_SET_MISMATCH", check="test_samples_and_truth")
        return
    dataset_truth = (
        {} if dataset is None else {item.sample_id: item.label for item in dataset.samples}
    )
    if any(
        sample_id not in dataset_truth
        or baseline_items[sample_id].true_label != dataset_truth[sample_id]
        or candidate_items[sample_id].true_label != dataset_truth[sample_id]
        or baseline_items[sample_id].true_label != candidate_items[sample_id].true_label
        for sample_id in expected_ids
    ):
        evidence.add_reason("TEST_SAMPLE_TRUTH_MISMATCH", check="test_samples_and_truth")
    else:
        evidence.pass_check("test_samples_and_truth")


def _initial_from_state(state: EvaluationState) -> EvaluationInitialInput:
    try:
        return EvaluationInitialInput.model_validate(
            {
                "schema_version": state.get("schema_version"),
                "workflow_id": state.get("workflow_id"),
                "thread_id": state.get("thread_id"),
                "request_id": state.get("request_id"),
                "training_workflow_id": state.get("training_workflow_id"),
            }
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise EvaluationEngineError("EVALUATION_REQUEST_INVALID") from error


def compute_evaluation(
    state: EvaluationState,
    dependencies: EvaluationDependencies,
) -> EvaluationComputation:
    """Return a complete valid or INVALID result without mutating any artifact."""
    initial = _initial_from_state(state)
    policy, policy_payload = _read_policy(dependencies)
    policy_hash = content_hash(policy_payload)
    evidence = _BuildEvidence()
    evidence.add_artifact(
        name="evaluation_policy",
        ref=EVALUATION_POLICY_REF,
        payload=policy_payload,
    )
    workflow_ref = f"training/{initial.training_workflow_id}/training.json"
    workflow_payload = _read_var(
        evidence,
        dependencies,
        name="training_workflow",
        ref=workflow_ref,
    )
    if workflow_payload is None:
        evidence.add_reason(
            "TRAINING_WORKFLOW_UNAVAILABLE",
            check="training_workflow_complete",
        )
        result = _invalid_result(
            workflow_id=initial.workflow_id,
            policy=policy,
            policy_hash=policy_hash,
            provenance=_provenance(initial.training_workflow_id, workflow_ref),
            evidence=evidence,
        )
        return EvaluationComputation(result=result, outputs_loaded=False)
    try:
        record = TrainingWorkflowRecord.model_validate_json(workflow_payload)
    except (TypeError, ValueError, ValidationError):
        evidence.add_reason(
            "TRAINING_WORKFLOW_SCHEMA_INVALID",
            check="training_workflow_complete",
        )
        result = _invalid_result(
            workflow_id=initial.workflow_id,
            policy=policy,
            policy_hash=policy_hash,
            provenance=_provenance(initial.training_workflow_id, workflow_ref),
            evidence=evidence,
        )
        return EvaluationComputation(result=result, outputs_loaded=False)
    if record.workflow_id != initial.training_workflow_id or record.status != "SUCCEEDED":
        evidence.add_reason(
            "TRAINING_WORKFLOW_NOT_SUCCEEDED",
            check="training_workflow_complete",
        )
    if (
        record.baseline_result is None
        or record.candidate_result is None
        or record.current_spec_ref is None
        or record.current_spec_hash is None
        or record.revision is None
    ):
        evidence.add_reason("TRAINING_PAIR_INCOMPLETE", check="training_workflow_complete")
    if evidence.checks["training_workflow_complete"] == "FAIL":
        result = _invalid_result(
            workflow_id=initial.workflow_id,
            policy=policy,
            policy_hash=policy_hash,
            provenance=_provenance(
                initial.training_workflow_id,
                workflow_ref,
                record=record,
            ),
            evidence=evidence,
        )
        return EvaluationComputation(result=result, outputs_loaded=False)
    evidence.pass_check("training_workflow_complete")
    assert record.current_spec_ref is not None
    assert record.current_spec_hash is not None
    assert record.revision is not None
    assert record.baseline_result is not None
    assert record.candidate_result is not None

    spec_payload = _read_var(
        evidence,
        dependencies,
        name="frozen_training_spec",
        ref=record.current_spec_ref,
    )
    if spec_payload is None:
        evidence.add_reason("FROZEN_SPEC_UNAVAILABLE", check="frozen_spec_integrity")
        result = _invalid_result(
            workflow_id=initial.workflow_id,
            policy=policy,
            policy_hash=policy_hash,
            provenance=_provenance(
                initial.training_workflow_id,
                workflow_ref,
                record=record,
            ),
            evidence=evidence,
        )
        return EvaluationComputation(result=result, outputs_loaded=False)
    try:
        spec = FrozenTrainingSpec.model_validate_json(spec_payload)
    except (TypeError, ValueError, ValidationError):
        evidence.add_reason("FROZEN_SPEC_SCHEMA_INVALID", check="frozen_spec_integrity")
        result = _invalid_result(
            workflow_id=initial.workflow_id,
            policy=policy,
            policy_hash=policy_hash,
            provenance=_provenance(
                initial.training_workflow_id,
                workflow_ref,
                record=record,
            ),
            evidence=evidence,
        )
        return EvaluationComputation(result=result, outputs_loaded=False)
    if (
        spec.workflow_id != record.workflow_id
        or spec.revision != record.revision
        or spec.spec_hash != record.current_spec_hash
        or record.baseline_result.run_id != spec.baseline.run_id
        or record.candidate_result.run_id != spec.candidate.run_id
        or record.baseline_result.role != "BASELINE"
        or record.candidate_result.role != "CANDIDATE"
    ):
        evidence.add_reason("FROZEN_SPEC_IDENTITY_MISMATCH", check="frozen_spec_integrity")
    review = next(
        (
            item
            for item in record.reviews
            if record.approval_id is not None
            and item.approval_id == record.approval_id
            and item.decision == "APPROVE"
            and item.subject_id == record.submission_id
            and item.subject_revision == spec.revision
            and item.spec_hash == spec.spec_hash
        ),
        None,
    )
    if review is None:
        evidence.add_reason(
            "TRAINING_APPROVAL_PROVENANCE_INVALID",
            check="frozen_spec_integrity",
        )
    provenance = _provenance(
        initial.training_workflow_id,
        workflow_ref,
        record=record,
        spec=spec,
    )
    if evidence.checks["frozen_spec_integrity"] == "FAIL":
        result = _invalid_result(
            workflow_id=initial.workflow_id,
            policy=policy,
            policy_hash=policy_hash,
            provenance=provenance,
            evidence=evidence,
        )
        return EvaluationComputation(result=result, outputs_loaded=False)
    evidence.pass_check("frozen_spec_integrity")

    baseline = _load_run(
        evidence,
        dependencies,
        name="baseline",
        spec=spec.baseline,
        result=record.baseline_result,
    )
    candidate = _load_run(
        evidence,
        dependencies,
        name="candidate",
        spec=spec.candidate,
        result=record.candidate_result,
    )
    if baseline is None or candidate is None:
        result = _invalid_result(
            workflow_id=initial.workflow_id,
            policy=policy,
            policy_hash=policy_hash,
            provenance=provenance,
            evidence=evidence,
        )
        return EvaluationComputation(result=result, outputs_loaded=False)
    _exact_pair_identity(evidence, spec, record, baseline, candidate)
    if evidence.checks["artifact_schema_and_identity"] == "FAIL":
        result = _invalid_result(
            workflow_id=initial.workflow_id,
            policy=policy,
            policy_hash=policy_hash,
            provenance=provenance,
            evidence=evidence,
        )
        return EvaluationComputation(result=result, outputs_loaded=False)

    dataset, split = _validate_project_fixtures(
        evidence,
        dependencies,
        policy,
        baseline,
        candidate,
    )
    _validate_pair_comparability(
        evidence,
        policy,
        dataset,
        split,
        baseline,
        candidate,
    )
    if any(check == "FAIL" for check in evidence.checks.values()):
        result = _invalid_result(
            workflow_id=initial.workflow_id,
            policy=policy,
            policy_hash=policy_hash,
            provenance=provenance,
            evidence=evidence,
        )
        return EvaluationComputation(result=result, outputs_loaded=True)
    result = _valid_result(
        workflow_id=initial.workflow_id,
        policy=policy,
        policy_hash=policy_hash,
        provenance=provenance,
        evidence=evidence,
        baseline=baseline,
        candidate=candidate,
    )
    return EvaluationComputation(result=result, outputs_loaded=True)


__all__ = ["EvaluationComputation", "EvaluationEngineError", "compute_evaluation"]
