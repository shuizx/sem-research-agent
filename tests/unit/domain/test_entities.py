"""Core entity construction and explicitly accepted state-invariant tests."""

from __future__ import annotations

import pydantic
import pytest

from vision_research_ops.domain import (
    ApprovalDecision,
    DatasetProfile,
    EvaluationConclusion,
    MetricDefinition,
    PatchOperationType,
    RunStatus,
    TaskType,
    ValidationStatus,
    WorkflowStatus,
    approval_authorizes,
)

pytestmark = pytest.mark.unit


def _primary(*, name: str = "macro_f1") -> MetricDefinition:
    return MetricDefinition(
        schema_version="1",
        name=name,
        direction="MAXIMIZE",
        aggregation="macro",
        implementation_version="1",
        primary=True,
    )


def _secondary(name: str = "balanced_accuracy") -> MetricDefinition:
    return MetricDefinition(
        schema_version="1",
        name=name,
        direction="MAXIMIZE",
        aggregation="macro",
        implementation_version="1",
        primary=False,
    )


def test_dataset_profile_happy_path_and_typed_contract(make_dataset_profile) -> None:
    profile = make_dataset_profile()
    assert profile.task_type is TaskType.IMAGE_CLASSIFICATION
    assert profile.channels in (1, 3)
    assert profile.image_shape_policy == {"fixed": True, "shape": [64, 64]}


@pytest.mark.parametrize("channels", [0, 2, 3.0, True])
def test_dataset_profile_channels_are_limited_and_strict(
    make_dataset_profile, channels: object
) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_dataset_profile(channels=channels)


def test_dataset_profile_requires_split_and_json_description_fields(make_dataset_profile) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_dataset_profile(split_policy=None)
    with pytest.raises(pydantic.ValidationError):
        make_dataset_profile(sample_counts={"train": float("nan")})


def test_research_request_happy_path_and_candidate_limit(make_research_request) -> None:
    request = make_research_request()
    assert request.candidate_limit == 20
    assert request.status is WorkflowStatus.PENDING
    assert make_research_request(candidate_limit=1).candidate_limit == 1
    assert make_research_request(candidate_limit=100).candidate_limit == 100
    for bad in (0, 101, 1.0, "20", True):
        with pytest.raises(pydantic.ValidationError):
            make_research_request(candidate_limit=bad)


def test_paper_candidate_requires_provenance_and_bounded_relevance(make_paper_candidate) -> None:
    assert make_paper_candidate(relevance_score=0.0).relevance_score == 0.0
    assert make_paper_candidate(relevance_score=1.0).relevance_score == 1.0
    for kwargs in ({"provenance": []}, {"relevance_score": 1.01}, {"relevance_score": "0.9"}):
        with pytest.raises(pydantic.ValidationError):
            make_paper_candidate(**kwargs)


def test_repository_snapshot_requires_strict_language_byte_counts(make_repository_snapshot) -> None:
    snapshot = make_repository_snapshot(languages={"Python": 2048})
    assert snapshot.languages == {"Python": 2048}
    for bad in (True, 1.0, "2048"):
        with pytest.raises(pydantic.ValidationError):
            make_repository_snapshot(languages={"Python": bad})


def test_experiment_spec_requires_unique_seeds_primary_metric_and_training_budget(
    make_experiment_spec,
) -> None:
    spec = make_experiment_spec()
    assert spec.seeds == [1, 2, 3]
    with pytest.raises(pydantic.ValidationError):
        make_experiment_spec(seeds=[])
    with pytest.raises(pydantic.ValidationError):
        make_experiment_spec(seeds=[1, 2, 2])
    with pytest.raises(pydantic.ValidationError):
        make_experiment_spec(seeds=[1.0])
    with pytest.raises(pydantic.ValidationError):
        make_experiment_spec(metrics=[_primary(), _primary(name="accuracy")])
    with pytest.raises(pydantic.ValidationError):
        make_experiment_spec(metrics=[_secondary()])
    with pytest.raises(pydantic.ValidationError):
        make_experiment_spec(training_budget={"schema_version": "1"})


def test_validation_result_pending_and_running_state_branches(make_validation_result) -> None:
    pending = make_validation_result(
        status=ValidationStatus.PENDING,
        started_at=None,
        finished_at=None,
        exit_code=None,
    )
    assert pending.status is ValidationStatus.PENDING
    with pytest.raises(pydantic.ValidationError):
        make_validation_result(status=ValidationStatus.PENDING, started_at="2026-08-06T08:00:00Z")
    running = make_validation_result(
        status=ValidationStatus.RUNNING,
        started_at="2026-08-06T08:00:00Z",
        finished_at=None,
    )
    assert running.status is ValidationStatus.RUNNING
    with pytest.raises(pydantic.ValidationError):
        make_validation_result(status=ValidationStatus.RUNNING, started_at=None, finished_at=None)
    with pytest.raises(pydantic.ValidationError):
        make_validation_result(status=ValidationStatus.RUNNING)


@pytest.mark.parametrize(
    "status",
    [
        ValidationStatus.PASSED,
        ValidationStatus.FAILED,
        ValidationStatus.SKIPPED,
        ValidationStatus.BLOCKED,
    ],
)
def test_validation_result_terminal_status_requires_finished_at(
    make_validation_result, status
) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_validation_result(status=status, finished_at=None)


@pytest.mark.parametrize(
    "status",
    [ValidationStatus.FAILED, ValidationStatus.SKIPPED, ValidationStatus.BLOCKED],
)
def test_validation_result_failure_like_status_requires_reason_codes(
    make_validation_result, status
) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_validation_result(status=status, reason_codes=[])
    valid = make_validation_result(status=status, reason_codes=["VALIDATION_IMPORT_FAILED"])
    assert valid.reason_codes == ["VALIDATION_IMPORT_FAILED"]


def test_validation_result_enforces_timestamp_order(make_validation_result) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_validation_result(
            started_at="2026-08-06T08:01:00Z", finished_at="2026-08-06T08:00:00Z"
        )


def test_experiment_run_created_and_queued_branches(make_experiment_run) -> None:
    assert make_experiment_run().status is RunStatus.CREATED
    queued = make_experiment_run(status=RunStatus.QUEUED, submitted_at="2026-08-06T08:00:00Z")
    assert queued.submitted_at is not None
    with pytest.raises(pydantic.ValidationError):
        make_experiment_run(status=RunStatus.QUEUED)


def test_experiment_run_running_branch_requires_submitted_and_started(make_experiment_run) -> None:
    running = make_experiment_run(
        status=RunStatus.RUNNING,
        submitted_at="2026-08-06T08:00:00Z",
        started_at="2026-08-06T08:01:00Z",
    )
    assert running.status is RunStatus.RUNNING
    with pytest.raises(pydantic.ValidationError):
        make_experiment_run(status=RunStatus.RUNNING, submitted_at="2026-08-06T08:00:00Z")


def test_experiment_run_succeeded_branch_requires_artifacts_and_times(make_experiment_run) -> None:
    succeeded = make_experiment_run(
        status=RunStatus.SUCCEEDED,
        submitted_at="2026-08-06T08:00:00Z",
        started_at="2026-08-06T08:01:00Z",
        finished_at="2026-08-06T08:02:00Z",
        manifest_artifact_id="manifest_1",
        metrics_artifact_id="metrics_1",
    )
    assert succeeded.status is RunStatus.SUCCEEDED
    with pytest.raises(pydantic.ValidationError):
        make_experiment_run(
            status=RunStatus.SUCCEEDED,
            submitted_at="2026-08-06T08:00:00Z",
            started_at="2026-08-06T08:01:00Z",
            finished_at="2026-08-06T08:02:00Z",
        )


@pytest.mark.parametrize(
    "status",
    [RunStatus.FAILED, RunStatus.TIMED_OUT, RunStatus.LOST, RunStatus.INVALID],
)
def test_experiment_run_failure_terminal_branches_require_failure(
    make_experiment_run, make_structured_failure, status
) -> None:
    common = {
        "status": status,
        "submitted_at": "2026-08-06T08:00:00Z",
        "started_at": "2026-08-06T08:01:00Z",
        "finished_at": "2026-08-06T08:02:00Z",
    }
    with pytest.raises(pydantic.ValidationError):
        make_experiment_run(**common)
    failed = make_experiment_run(**common, failure=make_structured_failure())
    assert failed.failure is not None


def test_experiment_run_cancelled_does_not_carry_failure(
    make_experiment_run, make_structured_failure
) -> None:
    cancelled = make_experiment_run(
        status=RunStatus.CANCELLED,
        submitted_at="2026-08-06T08:00:00Z",
        finished_at="2026-08-06T08:01:00Z",
    )
    assert cancelled.status is RunStatus.CANCELLED
    with pytest.raises(pydantic.ValidationError):
        make_experiment_run(
            status=RunStatus.CANCELLED,
            submitted_at="2026-08-06T08:00:00Z",
            finished_at="2026-08-06T08:01:00Z",
            failure=make_structured_failure(),
        )


def test_experiment_run_requires_terminal_finished_and_time_order(
    make_experiment_run, make_structured_failure
) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_experiment_run(
            status=RunStatus.FAILED,
            submitted_at="2026-08-06T08:00:00Z",
            started_at="2026-08-06T08:01:00Z",
            failure=make_structured_failure(),
        )
    with pytest.raises(pydantic.ValidationError):
        make_experiment_run(
            status=RunStatus.FAILED,
            submitted_at="2026-08-06T08:00:00Z",
            started_at="2026-08-06T08:02:00Z",
            finished_at="2026-08-06T08:01:00Z",
            failure=make_structured_failure(),
        )
    with pytest.raises(pydantic.ValidationError):
        make_experiment_run(
            status=RunStatus.CANCELLED,
            submitted_at="2026-08-06T08:01:00Z",
            finished_at="2026-08-06T08:00:00Z",
        )
    with pytest.raises(pydantic.ValidationError):
        make_experiment_run(
            status=RunStatus.FAILED,
            submitted_at="2026-08-06T08:01:00Z",
            started_at="2026-08-06T08:00:00Z",
            finished_at="2026-08-06T08:02:00Z",
            failure=make_structured_failure(),
        )


def test_evaluation_report_valid_branch_requires_runs_and_metrics(make_evaluation_report) -> None:
    assert make_evaluation_report().validity == "VALID"
    for field in ("baseline_run_ids", "candidate_run_ids", "metric_summaries"):
        with pytest.raises(pydantic.ValidationError):
            make_evaluation_report(**{field: []})


def test_evaluation_report_invalidity_and_conclusion_branches(make_evaluation_report) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_evaluation_report(validity="INVALID", invalidity_reasons=[])
    invalid = make_evaluation_report(
        validity="INVALID",
        invalidity_reasons=["EVAL_SPLIT_HASH_MISMATCH"],
        conclusion=EvaluationConclusion.INVALID,
    )
    assert invalid.conclusion is EvaluationConclusion.INVALID
    with pytest.raises(pydantic.ValidationError):
        make_evaluation_report(
            validity="VALID",
            invalidity_reasons=["EVAL_SPLIT_HASH_MISMATCH"],
        )
    with pytest.raises(pydantic.ValidationError):
        make_evaluation_report(validity="VALID", conclusion=EvaluationConclusion.INVALID)
    with pytest.raises(pydantic.ValidationError):
        make_evaluation_report(
            validity="INVALID",
            invalidity_reasons=["EVAL_SPLIT_HASH_MISMATCH"],
            conclusion=EvaluationConclusion.NO_CLEAR_IMPROVEMENT,
        )


def test_approval_edit_and_non_edit_invariants(make_approval, make_patch_operation) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_approval(decision=ApprovalDecision.EDIT, edits=[])
    edit = make_approval(
        decision=ApprovalDecision.EDIT,
        edits=[make_patch_operation(path="/seeds", value=[1, 2, 3])],
    )
    assert edit.decision is ApprovalDecision.EDIT
    for decision in (ApprovalDecision.APPROVE, ApprovalDecision.REJECT):
        with pytest.raises(pydantic.ValidationError):
            make_approval(
                decision=decision,
                edits=[make_patch_operation(op=PatchOperationType.REMOVE, path="/x", value=None)],
            )


def test_approval_reason_bound_and_exact_revision_authorization(make_approval) -> None:
    approval = make_approval(subject_type="experiment_spec", subject_id="exp_1", subject_revision=3)
    assert approval_authorizes(approval, "experiment_spec", "exp_1", 3)
    assert not approval_authorizes(approval, "experiment_spec", "exp_1", 4)
    assert not approval_authorizes(
        make_approval(decision=ApprovalDecision.REJECT), "experiment_spec", "exp_1", 3
    )
    assert make_approval(reason="x" * 1024)
    with pytest.raises(pydantic.ValidationError):
        make_approval(reason="")
    with pytest.raises(pydantic.ValidationError):
        make_approval(reason="x" * 1025)


def test_core_entities_json_roundtrip(
    make_dataset_profile, make_experiment_spec, make_evaluation_report
) -> None:
    for entity in (make_dataset_profile(), make_experiment_spec(), make_evaluation_report()):
        assert type(entity).model_validate_json(entity.model_dump_json()) == entity


def test_dataset_profile_required_fields_are_not_silently_defaulted(make_dataset_profile) -> None:
    payload = make_dataset_profile().model_dump()
    del payload["label_schema"]
    with pytest.raises(pydantic.ValidationError):
        DatasetProfile(**payload)
