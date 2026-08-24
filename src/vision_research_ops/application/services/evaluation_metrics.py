"""Pure deterministic metrics and pre-registered conclusion policy for evaluation."""

from __future__ import annotations

from .evaluation_models import (
    ClassMetricDelta,
    ClassMetrics,
    EvaluationConclusion,
    EvaluationPolicy,
    MetricDelta,
    MetricSet,
    ReasonCode,
)
from .training_models import PredictionItem, TrainingPredictions


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a finite ratio with the fixed zero-division value of 0.0."""
    return 0.0 if denominator == 0 else numerator / denominator


def compute_metrics(
    predictions: TrainingPredictions,
    policy: EvaluationPolicy,
) -> MetricSet:
    """Compute fixed-label precision/recall/F1, averages, accuracy, and confusion."""
    labels = policy.label_ids
    label_to_index = {label: index for index, label in enumerate(labels)}
    confusion = [[0 for _ in labels] for _ in labels]
    items: list[PredictionItem] = sorted(predictions.items, key=lambda item: item.sample_id)
    for item in items:
        confusion[label_to_index[item.true_label]][label_to_index[item.predicted_label]] += 1

    per_class: list[ClassMetrics] = []
    for index, (label_id, label_name) in enumerate(
        zip(policy.label_ids, policy.label_names, strict=True)
    ):
        true_positive = confusion[index][index]
        predicted_count = sum(row[index] for row in confusion)
        support = sum(confusion[index])
        precision = _safe_ratio(true_positive, predicted_count)
        recall = _safe_ratio(true_positive, support)
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        per_class.append(
            ClassMetrics(
                label_id=label_id,
                label_name=label_name,
                precision=precision,
                recall=recall,
                f1=f1,
                support=support,
            )
        )
    sample_count = len(items)
    return MetricSet(
        label_order=list(labels),
        per_class=per_class,
        macro_f1=sum(item.f1 for item in per_class) / len(per_class),
        balanced_accuracy=sum(item.recall for item in per_class) / len(per_class),
        accuracy=sum(confusion[index][index] for index in range(len(labels))) / sample_count,
        confusion_matrix=confusion,
        sample_count=sample_count,
    )


def compute_metric_delta(
    baseline: MetricSet,
    candidate: MetricSet,
    policy: EvaluationPolicy,
) -> MetricDelta:
    """Return candidate-minus-baseline differences for every recorded metric."""
    if baseline.label_order != candidate.label_order or baseline.label_order != policy.label_ids:
        raise ValueError("metric delta requires the same frozen label order")
    per_class: list[ClassMetricDelta] = []
    for baseline_class, candidate_class in zip(
        baseline.per_class,
        candidate.per_class,
        strict=True,
    ):
        if baseline_class.label_id != candidate_class.label_id:
            raise ValueError("metric delta requires matching per-class label IDs")
        per_class.append(
            ClassMetricDelta(
                label_id=baseline_class.label_id,
                precision=candidate_class.precision - baseline_class.precision,
                recall=candidate_class.recall - baseline_class.recall,
                f1=candidate_class.f1 - baseline_class.f1,
                support=candidate_class.support - baseline_class.support,
            )
        )
    severe_index = policy.label_ids.index(policy.severe_class_id)
    confusion_delta = [
        [
            candidate.confusion_matrix[row][column] - baseline.confusion_matrix[row][column]
            for column in range(4)
        ]
        for row in range(4)
    ]
    return MetricDelta(
        label_order=list(policy.label_ids),
        per_class=per_class,
        macro_f1=candidate.macro_f1 - baseline.macro_f1,
        balanced_accuracy=candidate.balanced_accuracy - baseline.balanced_accuracy,
        accuracy=candidate.accuracy - baseline.accuracy,
        severe_class_id=policy.severe_class_id,
        severe_class_recall=(
            candidate.per_class[severe_index].recall - baseline.per_class[severe_index].recall
        ),
        confusion_matrix=confusion_delta,
    )


def decide_conclusion(
    delta: MetricDelta,
    policy: EvaluationPolicy,
) -> tuple[EvaluationConclusion, tuple[ReasonCode, ...]]:
    """Apply the fixed severe-recall and practical Macro-F1 rules in order."""
    regression_reasons: list[ReasonCode] = []
    severe_drop = -delta.severe_class_recall
    if severe_drop > policy.max_severe_recall_drop:
        regression_reasons.append("SEVERE_CLASS_RECALL_REGRESSION")
    if delta.macro_f1 <= -policy.minimum_practical_delta:
        regression_reasons.append("MACRO_F1_REGRESSION")
    if regression_reasons:
        return "REGRESSED", tuple(regression_reasons)
    if (
        delta.macro_f1 >= policy.minimum_practical_delta
        and delta.balanced_accuracy >= 0.0
        and severe_drop <= policy.max_severe_recall_drop
    ):
        return "IMPROVED", ("PRACTICAL_MACRO_F1_IMPROVEMENT",)
    return "NO_CLEAR_IMPROVEMENT", ("IMPROVEMENT_RULE_NOT_MET",)


__all__ = ["compute_metric_delta", "compute_metrics", "decide_conclusion"]
