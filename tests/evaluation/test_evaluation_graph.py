"""deterministic graph, invalid boundaries, artifacts, and report tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from vision_research_ops.application.services.evaluation_models import (
    EvaluationResult,
    EvaluationState,
    canonical_json_bytes,
    evaluation_state_as_jsonable,
)
from vision_research_ops.application.workflows.core import workflow_config
from vision_research_ops.application.workflows.evaluation import build_evaluation_graph

from .conftest import (
    DEFAULT_BASELINE,
    DEFAULT_CANDIDATE,
    EVALUATION_WORKFLOW_ID,
    EvaluationHarness,
)


async def _run(harness: EvaluationHarness) -> dict[str, object]:
    result = await build_evaluation_graph().ainvoke(
        harness.state,
        config=workflow_config(cast(str, harness.state["thread_id"])),
        context=harness.dependencies,
    )
    return cast(dict[str, object], result)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _write_json(path: Path, value: Mapping[str, object], *, allow_nan: bool = False) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=allow_nan,
        )
        + "\n",
        encoding="utf-8",
    )


def _candidate_path(harness: EvaluationHarness, name: str) -> Path:
    return harness.training_store.resolve_ref(f"runs/{harness.spec.candidate.run_id}/{name}.json")


def _candidate_log_path(harness: EvaluationHarness) -> Path:
    return harness.training_store.resolve_ref(f"runs/{harness.spec.candidate.run_id}/train.log")


@pytest.mark.asyncio
@pytest.mark.graph
async def test_happy_graph_writes_canonical_evaluation_then_exact_template_report(
    make_evaluation_harness: Callable[..., EvaluationHarness],
    tmp_path: Path,
) -> None:
    """a real graph preserves provenance, numbers, claims, and small state."""
    harness = make_evaluation_harness(root=tmp_path / "happy")
    final = await _run(harness)

    assert final["status"] == "COMPLETED"
    assert final["conclusion"] == "IMPROVED"
    assert final["evaluation_ref"] == "reports/workflow-evaluation-1/evaluation.json"
    assert final["report_ref"] == "reports/workflow-evaluation-1/report.md"
    result = harness.evaluation_store.load_evaluation(EVALUATION_WORKFLOW_ID)
    assert result.conclusion == "IMPROVED"
    assert result.comparability.status == "VALID"
    assert all(check.status == "PASS" for check in result.comparability.checks)
    assert result.baseline_metrics is not None
    assert result.candidate_metrics is not None
    assert result.deltas is not None
    assert result.deltas.macro_f1 > 0.01
    assert result.deltas.balanced_accuracy >= 0.0
    assert result.deltas.severe_class_recall == 0.0
    assert result.evaluation_capability == "DETERMINISTIC_SINGLE_PAIR_FIXTURE_EVALUATION"
    assert result.llm_used is False
    assert result.real_company_evaluation is False
    assert result.real_sem_evaluation is False
    assert result.provenance.training_capability == "SYNTHETIC_FIXTURE_LINEAR_CPU_NO_TORCH"
    assert result.provenance.real_pytorch_training is False
    assert result.provenance.base_commit_sha == "a" * 40
    assert result.provenance.patch_hash == f"sha256:{'c' * 64}"
    assert result.provenance.baseline is not None
    assert result.provenance.candidate is not None
    assert result.provenance.baseline.method == "GLOBAL_STATS_LINEAR"
    assert result.provenance.candidate.method == "GRID4_LINEAR_PATCHED"
    assert result.provenance.baseline.method_config_hash != (
        result.provenance.candidate.method_config_hash
    )
    assert len(result.input_artifacts) == 16
    assert all(artifact.status == "HASHED" for artifact in result.input_artifacts)
    assert all(artifact.size_bytes is not None for artifact in result.input_artifacts)
    log_digests = {
        artifact.name: artifact
        for artifact in result.input_artifacts
        if artifact.name in {"baseline_log", "candidate_log"}
    }
    assert set(log_digests) == {"baseline_log", "candidate_log"}
    assert log_digests["baseline_log"].ref == (f"runs/{harness.spec.baseline.run_id}/train.log")
    assert log_digests["candidate_log"].ref == (f"runs/{harness.spec.candidate.run_id}/train.log")
    assert all(item.content_hash is not None for item in log_digests.values())
    assert all(item.size_bytes is not None and item.size_bytes > 0 for item in log_digests.values())

    evaluation_path = harness.evaluation_store.resolve_ref(result.evaluation_ref)
    raw_evaluation = evaluation_path.read_bytes()
    assert raw_evaluation == canonical_json_bytes(result.model_dump(mode="json"))
    assert EvaluationResult.model_validate_json(raw_evaluation) == result
    report = harness.evaluation_store.resolve_ref(result.report_ref).read_text(encoding="utf-8")
    assert f"Conclusion: **{result.conclusion}**" in report
    assert f"| Macro-F1 | {json.dumps(result.baseline_metrics.macro_f1)} |" in report
    assert json.dumps(result.deltas.confusion_matrix, separators=(",", ":")) in report
    assert "LLM used: `false`" in report
    assert "Real company evaluation: `false`" in report
    assert "does not represent improvement on real SEM" in report
    assert "no LLM can modify its numbers or conclusion" in report
    reports_dir = harness.var_root / "reports" / EVALUATION_WORKFLOW_ID
    assert sorted(path.name for path in reports_dir.iterdir()) == [
        "evaluation.json",
        "report.md",
    ]
    state_json = json.dumps(
        evaluation_state_as_jsonable(cast(EvaluationState, final)),
        allow_nan=False,
    )
    assert str(harness.project_root) not in state_json
    assert "scores" not in state_json
    assert "confusion_matrix" not in state_json


@pytest.mark.asyncio
@pytest.mark.graph
@pytest.mark.parametrize(
    ("baseline", "candidate", "expected"),
    [
        (DEFAULT_BASELINE, DEFAULT_CANDIDATE, "IMPROVED"),
        (DEFAULT_CANDIDATE, DEFAULT_CANDIDATE, "NO_CLEAR_IMPROVEMENT"),
        (DEFAULT_CANDIDATE, (2, 0, 0, 1, 2, 3), "REGRESSED"),
    ],
)
async def test_valid_positive_neutral_and_negative_results_are_all_persisted(
    make_evaluation_harness: Callable[..., EvaluationHarness],
    tmp_path: Path,
    baseline: Sequence[int],
    candidate: Sequence[int],
    expected: str,
) -> None:
    """valid negative/no-clear branches retain full metrics and reports."""
    harness = make_evaluation_harness(
        root=tmp_path / expected.casefold(),
        baseline_labels=baseline,
        candidate_labels=candidate,
    )
    final = await _run(harness)
    result = harness.evaluation_store.load_evaluation(EVALUATION_WORKFLOW_ID)
    assert final["conclusion"] == expected
    assert result.conclusion == expected
    assert result.baseline_metrics is not None
    assert result.candidate_metrics is not None
    assert result.deltas is not None
    report = harness.evaluation_store.resolve_ref(result.report_ref).read_text(encoding="utf-8")
    if expected != "IMPROVED":
        assert "was not hidden or rewritten as an improvement" in report


def _workflow_not_succeeded(harness: EvaluationHarness) -> None:
    path = harness.training_store.resolve_ref(
        harness.training_store.workflow_ref(harness.spec.workflow_id)
    )
    payload = _read_json(path)
    payload["status"] = "RUNNING"
    _write_json(path, payload)


def _dataset_mismatch(harness: EvaluationHarness) -> None:
    path = _candidate_path(harness, "manifest")
    payload = _read_json(path)
    payload["dataset_content_hash"] = f"sha256:{'d' * 64}"
    _write_json(path, payload)


def _split_mismatch(harness: EvaluationHarness) -> None:
    path = _candidate_path(harness, "predictions")
    payload = _read_json(path)
    payload["split_hash"] = f"sha256:{'d' * 64}"
    _write_json(path, payload)


def _preprocess_mismatch(harness: EvaluationHarness) -> None:
    path = _candidate_path(harness, "manifest")
    payload = _read_json(path)
    payload["preprocess_hash"] = f"sha256:{'d' * 64}"
    _write_json(path, payload)


def _seed_mismatch(harness: EvaluationHarness) -> None:
    for name in ("manifest", "metrics"):
        path = _candidate_path(harness, name)
        payload = _read_json(path)
        payload["seed"] = 18
        _write_json(path, payload)


def _budget_mismatch(harness: EvaluationHarness) -> None:
    for name in ("manifest", "metrics"):
        path = _candidate_path(harness, name)
        payload = _read_json(path)
        budget = cast(dict[str, object], payload["budget"])
        budget["max_epochs"] = 3
        _write_json(path, payload)


def _sample_set_mismatch(harness: EvaluationHarness) -> None:
    path = _candidate_path(harness, "predictions")
    payload = _read_json(path)
    items = cast(list[dict[str, object]], payload["items"])
    items[0]["sample_id"] = "syn-missing-replacement"
    _write_json(path, payload)


def _sample_truth_mismatch(harness: EvaluationHarness) -> None:
    path = _candidate_path(harness, "predictions")
    payload = _read_json(path)
    items = cast(list[dict[str, object]], payload["items"])
    items[0]["true_label"] = 1
    _write_json(path, payload)
    metrics_path = _candidate_path(harness, "metrics")
    metrics = _read_json(metrics_path)
    metrics["test_accuracy"] = 5 / 6
    _write_json(metrics_path, metrics)


@pytest.mark.asyncio
@pytest.mark.graph
@pytest.mark.parametrize(
    "mutation",
    [
        _workflow_not_succeeded,
        _dataset_mismatch,
        _split_mismatch,
        _preprocess_mismatch,
        _seed_mismatch,
        _budget_mismatch,
        _sample_set_mismatch,
        _sample_truth_mismatch,
    ],
)
async def test_incomplete_or_noncomparable_p4_pair_is_invalid_not_regressed(
    make_evaluation_harness: Callable[..., EvaluationHarness],
    tmp_path: Path,
    mutation: Callable[[EvaluationHarness], None],
) -> None:
    """every exact-pair failure follows the graph's INVALID branch."""
    harness = make_evaluation_harness(root=tmp_path / mutation.__name__)
    mutation(harness)
    final = await _run(harness)
    result = harness.evaluation_store.load_evaluation(EVALUATION_WORKFLOW_ID)
    assert final["status"] == "COMPLETED"
    assert final["conclusion"] == "INVALID"
    assert result.conclusion == "INVALID"
    assert result.comparability.status == "INVALID"
    assert result.reason_codes
    assert result.baseline_metrics is None
    assert result.candidate_metrics is None
    assert result.deltas is None
    assert "REGRESSED" not in result.reason_codes
    report = harness.evaluation_store.resolve_ref(result.report_ref).read_text(encoding="utf-8")
    assert "Metrics were not computed" in report
    assert "was not hidden or rewritten as an improvement" in report


def _duplicate_sample(payload: dict[str, object]) -> None:
    items = cast(list[dict[str, object]], payload["items"])
    items.append(dict(items[0]))


def _unknown_label(payload: dict[str, object]) -> None:
    items = cast(list[dict[str, object]], payload["items"])
    items[0]["true_label"] = 9


def _nonfinite_score(payload: dict[str, object]) -> None:
    items = cast(list[dict[str, object]], payload["items"])
    scores = cast(list[float], items[0]["scores"])
    scores[0] = float("nan")


def _bad_probability_length(payload: dict[str, object]) -> None:
    items = cast(list[dict[str, object]], payload["items"])
    items[0]["scores"] = [0.5, 0.5]


def _probability_out_of_range(payload: dict[str, object]) -> None:
    items = cast(list[dict[str, object]], payload["items"])
    items[0]["scores"] = [1.1, -0.1, 0.0, 0.0]


def _probability_bad_sum(payload: dict[str, object]) -> None:
    items = cast(list[dict[str, object]], payload["items"])
    items[0]["scores"] = [0.4, 0.1, 0.1, 0.1]


def _predicted_label_not_score_max(payload: dict[str, object]) -> None:
    items = cast(list[dict[str, object]], payload["items"])
    predicted_label = cast(int, items[0]["predicted_label"])
    wrong_max = (predicted_label + 1) % 4
    scores = [0.1, 0.1, 0.1, 0.1]
    scores[wrong_max] = 0.7
    items[0]["scores"] = scores


@pytest.mark.asyncio
@pytest.mark.graph
@pytest.mark.parametrize(
    ("mutation", "reason", "allow_nan"),
    [
        (_duplicate_sample, "PREDICTIONS_SCHEMA_INVALID", False),
        (_unknown_label, "PREDICTIONS_SCHEMA_INVALID", False),
        (_nonfinite_score, "PREDICTIONS_SCHEMA_INVALID", True),
        (_bad_probability_length, "PREDICTIONS_SCHEMA_INVALID", False),
        (_probability_out_of_range, "PREDICTIONS_SCHEMA_INVALID", False),
        (_probability_bad_sum, "PREDICTIONS_SCHEMA_INVALID", False),
        (_predicted_label_not_score_max, "PREDICTED_LABEL_SCORE_MISMATCH", False),
    ],
)
async def test_prediction_boundary_rejects_bad_ids_labels_and_probabilities(
    make_evaluation_harness: Callable[..., EvaluationHarness],
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    reason: str,
    allow_nan: bool,
) -> None:
    """duplicate/unknown/non-finite/bad probability predictions fail deterministically."""
    harness = make_evaluation_harness(root=tmp_path / mutation.__name__)
    path = _candidate_path(harness, "predictions")
    payload = _read_json(path)
    mutation(payload)
    _write_json(path, payload, allow_nan=allow_nan)
    final = await _run(harness)
    result = harness.evaluation_store.load_evaluation(EVALUATION_WORKFLOW_ID)
    assert final["conclusion"] == "INVALID"
    assert reason in result.reason_codes
    assert result.baseline_metrics is None
    assert result.candidate_metrics is None


@pytest.mark.asyncio
@pytest.mark.graph
async def test_missing_required_p4_artifact_is_a_complete_invalid_result(
    make_evaluation_harness: Callable[..., EvaluationHarness],
    tmp_path: Path,
) -> None:
    """a missing metrics file remains visible in hashes and the INVALID report."""
    harness = make_evaluation_harness(root=tmp_path / "missing-metrics")
    missing_path = _candidate_path(harness, "metrics")
    missing_path.unlink()
    final = await _run(harness)
    result = harness.evaluation_store.load_evaluation(EVALUATION_WORKFLOW_ID)
    assert final["conclusion"] == "INVALID"
    assert "TRAINING_ARTIFACT_MISSING" in result.reason_codes
    digest = next(item for item in result.input_artifacts if item.name == "candidate_metrics")
    assert digest.status == "MISSING"
    assert digest.content_hash is None
    assert digest.size_bytes is None


@pytest.mark.asyncio
@pytest.mark.graph
async def test_missing_training_log_is_a_complete_invalid_result(
    make_evaluation_harness: Callable[..., EvaluationHarness],
    tmp_path: Path,
) -> None:
    """a missing manifest-referenced log is explicit INVALID evidence."""
    harness = make_evaluation_harness(root=tmp_path / "missing-log")
    _candidate_log_path(harness).unlink()

    final = await _run(harness)
    result = harness.evaluation_store.load_evaluation(EVALUATION_WORKFLOW_ID)

    assert final["conclusion"] == "INVALID"
    assert result.comparability.status == "INVALID"
    assert "TRAINING_LOG_INVALID" in result.reason_codes
    digest = next(item for item in result.input_artifacts if item.name == "candidate_log")
    assert digest.status == "MISSING"
    assert digest.content_hash is None
    assert digest.size_bytes is None
    assert result.baseline_metrics is None
    assert result.candidate_metrics is None


@pytest.mark.asyncio
@pytest.mark.graph
@pytest.mark.parametrize("mutation", ["malformed_jsonl", "mismatched_identity"])
async def test_malformed_or_mismatched_training_log_is_invalid(
    make_evaluation_harness: Callable[..., EvaluationHarness],
    tmp_path: Path,
    mutation: str,
) -> None:
    """damaged JSONL or a conflicting run identity never reaches metrics."""
    harness = make_evaluation_harness(root=tmp_path / mutation)
    path = _candidate_log_path(harness)
    if mutation == "malformed_jsonl":
        path.write_text('{"event":\n', encoding="utf-8")
    else:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        events[0]["run_id"] = "wrong-candidate-run"
        path.write_text(
            "\n".join(json.dumps(event, sort_keys=True, separators=(",", ":")) for event in events)
            + "\n",
            encoding="utf-8",
        )

    final = await _run(harness)
    result = harness.evaluation_store.load_evaluation(EVALUATION_WORKFLOW_ID)

    assert final["conclusion"] == "INVALID"
    assert result.comparability.status == "INVALID"
    assert "TRAINING_LOG_INVALID" in result.reason_codes
    digest = next(item for item in result.input_artifacts if item.name == "candidate_log")
    assert digest.status == "HASHED"
    assert digest.content_hash is not None
    assert digest.size_bytes is not None and digest.size_bytes > 0
    assert result.baseline_metrics is None
    assert result.candidate_metrics is None


@pytest.mark.asyncio
@pytest.mark.graph
@pytest.mark.parametrize(
    ("artifact", "field", "value"),
    [
        ("manifest", "run_id", "wrong-candidate-run"),
        ("metrics", "role", "BASELINE"),
        ("predictions", "spec_hash", f"sha256:{'e' * 64}"),
    ],
)
async def test_run_role_and_spec_identity_mismatches_are_invalid(
    make_evaluation_harness: Callable[..., EvaluationHarness],
    tmp_path: Path,
    artifact: str,
    field: str,
    value: object,
) -> None:
    """schema-valid role/run/spec mismatches never enter metric calculation."""
    harness = make_evaluation_harness(root=tmp_path / f"identity-{artifact}-{field}")
    path = _candidate_path(harness, artifact)
    payload = _read_json(path)
    payload[field] = value
    _write_json(path, payload)
    final = await _run(harness)
    result = harness.evaluation_store.load_evaluation(EVALUATION_WORKFLOW_ID)
    assert final["conclusion"] == "INVALID"
    assert (
        "RUN_ARTIFACT_IDENTITY_MISMATCH" in result.reason_codes
        or f"{artifact.upper()}_SCHEMA_INVALID" in result.reason_codes
    )
    assert result.baseline_metrics is None
