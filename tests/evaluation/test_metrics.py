"""hand-computed metrics and fixed conclusion threshold tests."""

from __future__ import annotations

import pytest

from vision_research_ops.application.services.evaluation_metrics import (
    compute_metrics,
    decide_conclusion,
)
from vision_research_ops.application.services.evaluation_models import (
    ClassMetricDelta,
    EvaluationPolicy,
    MetricDelta,
)
from vision_research_ops.application.services.training_models import (
    PredictionItem,
    TrainingPredictions,
)

from .conftest import PROJECT_ROOT


def _policy() -> EvaluationPolicy:
    return EvaluationPolicy.model_validate_json(
        (PROJECT_ROOT / "fixtures" / "evaluation" / "single_pair_policy.json").read_bytes()
    )


def _prediction(sample_id: str, true_label: int, predicted_label: int) -> PredictionItem:
    scores = [0.1, 0.1, 0.1, 0.1]
    scores[predicted_label] = 0.7
    return PredictionItem(
        sample_id=sample_id,
        true_label=true_label,
        predicted_label=predicted_label,
        scores=scores,
    )


@pytest.mark.unit
def test_hand_computed_fixed_label_metrics_are_exact() -> None:
    """per-class values, averages, accuracy, and matrix match a hand example."""
    predictions = TrainingPredictions(
        run_id="hand-metric-run",
        role="BASELINE",
        spec_hash=f"sha256:{'a' * 64}",
        split_ref="fixtures/training/synthetic_sem_split.json",
        split_hash=f"sha256:{'b' * 64}",
        items=[
            _prediction("sample-0", 0, 0),
            _prediction("sample-1", 0, 1),
            _prediction("sample-2", 1, 1),
            _prediction("sample-3", 2, 0),
            _prediction("sample-4", 3, 3),
        ],
    )
    metrics = compute_metrics(predictions, _policy())

    assert metrics.confusion_matrix == [
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
    assert metrics.sample_count == 5
    assert metrics.accuracy == pytest.approx(3 / 5)
    assert metrics.balanced_accuracy == pytest.approx(5 / 8)
    assert metrics.macro_f1 == pytest.approx(13 / 24)
    expected = [
        (0.5, 0.5, 0.5, 2),
        (0.5, 1.0, 2 / 3, 1),
        (0.0, 0.0, 0.0, 1),
        (1.0, 1.0, 1.0, 1),
    ]
    for observed, values in zip(metrics.per_class, expected, strict=True):
        precision, recall, f1, support = values
        assert observed.precision == pytest.approx(precision)
        assert observed.recall == pytest.approx(recall)
        assert observed.f1 == pytest.approx(f1)
        assert observed.support == support


def _delta(
    *,
    macro_f1: float,
    balanced: float,
    severe_recall: float,
    accuracy: float = 0.0,
) -> MetricDelta:
    return MetricDelta(
        label_order=[0, 1, 2, 3],
        per_class=[
            ClassMetricDelta(
                label_id=label_id,
                precision=0.0,
                recall=severe_recall if label_id == 3 else 0.0,
                f1=0.0,
                support=0,
            )
            for label_id in range(4)
        ],
        macro_f1=macro_f1,
        balanced_accuracy=balanced,
        accuracy=accuracy,
        severe_class_id=3,
        severe_class_recall=severe_recall,
        confusion_matrix=[[0, 0, 0, 0] for _ in range(4)],
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("delta", "conclusion", "reason"),
    [
        (
            _delta(macro_f1=0.01, balanced=0.0, severe_recall=0.0),
            "IMPROVED",
            "PRACTICAL_MACRO_F1_IMPROVEMENT",
        ),
        (
            _delta(macro_f1=0.009, balanced=0.1, severe_recall=0.0),
            "NO_CLEAR_IMPROVEMENT",
            "IMPROVEMENT_RULE_NOT_MET",
        ),
        (
            _delta(macro_f1=-0.01, balanced=0.0, severe_recall=0.0),
            "REGRESSED",
            "MACRO_F1_REGRESSION",
        ),
        (
            _delta(macro_f1=0.2, balanced=0.2, severe_recall=-0.0001),
            "REGRESSED",
            "SEVERE_CLASS_RECALL_REGRESSION",
        ),
    ],
)
def test_pre_registered_valid_conclusion_branches_are_threshold_exact(
    delta: MetricDelta,
    conclusion: str,
    reason: str,
) -> None:
    """equality boundaries and any severe recall drop are reproducible."""
    observed, reasons = decide_conclusion(delta, _policy())
    assert observed == conclusion
    assert reason in reasons


@pytest.mark.unit
def test_accuracy_alone_cannot_claim_improvement() -> None:
    """even a maximal Accuracy delta cannot replace the primary metric rule."""
    conclusion, reasons = decide_conclusion(
        _delta(
            macro_f1=0.0,
            balanced=0.0,
            severe_recall=0.0,
            accuracy=1.0,
        ),
        _policy(),
    )
    assert conclusion == "NO_CLEAR_IMPROVEMENT"
    assert reasons == ("IMPROVEMENT_RULE_NOT_MET",)


@pytest.mark.unit
def test_zero_support_uses_fixed_zero_division_value() -> None:
    """absent labels still occupy the fixed vocabulary with all-zero metrics."""
    predictions = TrainingPredictions(
        run_id="zero-support-run",
        role="BASELINE",
        spec_hash=f"sha256:{'a' * 64}",
        split_ref="fixtures/training/synthetic_sem_split.json",
        split_hash=f"sha256:{'b' * 64}",
        items=[_prediction("only-sample", 0, 1)],
    )
    metrics = compute_metrics(predictions, _policy())
    for label_id in (2, 3):
        observed = metrics.per_class[label_id]
        assert observed.support == 0
        assert observed.precision == 0.0
        assert observed.recall == 0.0
        assert observed.f1 == 0.0
