"""Deterministic Markdown renderer with no LLM or narrative override input."""

from __future__ import annotations

import json

from .evaluation_models import EvaluationResult, MetricSet


def _number(value: float | int) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _matrix_lines(title: str, metrics: MetricSet) -> list[str]:
    return [
        f"### {title} confusion matrix",
        "",
        "Rows are true labels and columns are predicted labels in order `0,1,2,3`.",
        "",
        "```json",
        json.dumps(metrics.confusion_matrix, separators=(",", ":"), allow_nan=False),
        "```",
        "",
    ]


def render_evaluation_report(result: EvaluationResult) -> str:
    """Render only values present in a revalidated canonical evaluation model."""
    evaluation = EvaluationResult.model_validate(result.model_dump(mode="json"))
    policy = evaluation.policy
    lines = [
        "# SEM Research Agent Deterministic Evaluation Report",
        "",
        f"- Evaluation ID: `{evaluation.evaluation_id}`",
        f"- Conclusion: **{evaluation.conclusion}**",
        f"- Capability: `{evaluation.evaluation_capability}`",
        f"- Training capability: `{evaluation.provenance.training_capability or 'unavailable'}`",
        "- LLM used: `false`",
        "- Real PyTorch training: `false`",
        "- Real SEM evaluation: `false`",
        "- Real company evaluation: `false`",
        "",
        "## Pre-registered policy",
        "",
        f"- Policy ID: `{policy.policy_id}`",
        f"- Primary metric: `{policy.primary_metric}`",
        f"- Label order: `{','.join(str(label) for label in policy.label_ids)}`",
        f"- Severe class ID: `{policy.severe_class_id}`",
        f"- Minimum practical Macro-F1 delta: `{_number(policy.minimum_practical_delta)}`",
        (f"- Maximum severe-class recall drop: `{_number(policy.max_severe_recall_drop)}`"),
        f"- Metric implementation: `{policy.metric_implementation_version}`",
        "",
        "## training provenance and method difference",
        "",
        f"- Training workflow: `{evaluation.provenance.training_workflow_id}`",
        f"- Frozen spec: `{evaluation.provenance.frozen_spec_ref or 'unavailable'}`",
        f"- Frozen spec hash: `{evaluation.provenance.frozen_spec_hash or 'unavailable'}`",
        f"- Base commit: `{evaluation.provenance.base_commit_sha or 'unavailable'}`",
        f"- Candidate patch revision: `{evaluation.provenance.patch_revision or 'unavailable'}`",
        f"- Candidate patch hash: `{evaluation.provenance.patch_hash or 'unavailable'}`",
    ]
    if evaluation.provenance.baseline is not None:
        lines.append(f"- Baseline method: `{evaluation.provenance.baseline.method}`")
    else:
        lines.append("- Baseline method: `unavailable`")
    if evaluation.provenance.candidate is not None:
        lines.append(f"- Candidate method: `{evaluation.provenance.candidate.method}`")
    else:
        lines.append("- Candidate method: `unavailable`")
    lines.extend(
        [
            "- Method-specific differences are allowed and recorded; data, split, preprocessing, "
            "seed, and budget must remain exact.",
            "",
            "## Comparability",
            "",
            f"Overall: **{evaluation.comparability.status}**",
            "",
            "| Check | Status |",
            "|---|---|",
        ]
    )
    for check in evaluation.comparability.checks:
        lines.append(f"| `{check.name}` | {check.status} |")
    lines.extend(
        [
            "",
            "## Input artifact hashes",
            "",
            "| Artifact | Relative ref | Status | Bytes | SHA-256 |",
            "|---|---|---|---:|---|",
        ]
    )
    for artifact in evaluation.input_artifacts:
        lines.append(
            "| "
            f"`{artifact.name}` | `{_cell(artifact.ref)}` | {artifact.status} | "
            f"{artifact.size_bytes if artifact.size_bytes is not None else 'unavailable'} | "
            f"`{artifact.content_hash or 'unavailable'}` |"
        )
    lines.extend(["", "## Metrics", ""])
    if (
        evaluation.baseline_metrics is None
        or evaluation.candidate_metrics is None
        or evaluation.deltas is None
    ):
        lines.extend(
            [
                "Metrics were not computed because the validity/comparability gate failed.",
                "",
            ]
        )
    else:
        baseline = evaluation.baseline_metrics
        candidate = evaluation.candidate_metrics
        delta = evaluation.deltas
        lines.extend(
            [
                "| Metric | Baseline | Candidate | Candidate - baseline |",
                "|---|---:|---:|---:|",
                (
                    f"| Macro-F1 | {_number(baseline.macro_f1)} | "
                    f"{_number(candidate.macro_f1)} | {_number(delta.macro_f1)} |"
                ),
                (
                    f"| Balanced accuracy | {_number(baseline.balanced_accuracy)} | "
                    f"{_number(candidate.balanced_accuracy)} | "
                    f"{_number(delta.balanced_accuracy)} |"
                ),
                (
                    f"| Accuracy | {_number(baseline.accuracy)} | "
                    f"{_number(candidate.accuracy)} | {_number(delta.accuracy)} |"
                ),
                (
                    f"| Severe class `{delta.severe_class_id}` recall | "
                    f"{_number(baseline.per_class[delta.severe_class_id].recall)} | "
                    f"{_number(candidate.per_class[delta.severe_class_id].recall)} | "
                    f"{_number(delta.severe_class_recall)} |"
                ),
                "",
                "### Per-class metrics",
                "",
                (
                    "| Label | B precision | B recall | B F1 | B support | C precision | "
                    "C recall | C F1 | C support | Δ precision | Δ recall | Δ F1 | Δ support |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for baseline_class, candidate_class, class_delta in zip(
            baseline.per_class,
            candidate.per_class,
            delta.per_class,
            strict=True,
        ):
            lines.append(
                f"| `{baseline_class.label_id}:{_cell(baseline_class.label_name)}` | "
                f"{_number(baseline_class.precision)} | {_number(baseline_class.recall)} | "
                f"{_number(baseline_class.f1)} | {_number(baseline_class.support)} | "
                f"{_number(candidate_class.precision)} | {_number(candidate_class.recall)} | "
                f"{_number(candidate_class.f1)} | {_number(candidate_class.support)} | "
                f"{_number(class_delta.precision)} | {_number(class_delta.recall)} | "
                f"{_number(class_delta.f1)} | {_number(class_delta.support)} |"
            )
        lines.extend([""])
        lines.extend(_matrix_lines("Baseline", baseline))
        lines.extend(_matrix_lines("Candidate", candidate))
        lines.extend(
            [
                "### Candidate-minus-baseline confusion matrix",
                "",
                "```json",
                json.dumps(delta.confusion_matrix, separators=(",", ":"), allow_nan=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Deterministic conclusion and reason codes",
            "",
            f"Conclusion: **{evaluation.conclusion}**",
            "",
        ]
    )
    for reason in evaluation.reason_codes:
        lines.append(f"- `{reason}`")
    if evaluation.conclusion == "IMPROVED":
        lines.extend(
            [
                "",
                "The fixed improvement rule passed; this does not broaden the data or training "
                "capability stated above.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "This negative, invalid, or no-clear-improvement result is retained exactly; it "
                "was not hidden or rewritten as an improvement.",
            ]
        )
    lines.extend(["", "## Limitations", ""])
    for limitation in evaluation.limitations:
        lines.append(f"- `{limitation}`")
    lines.extend(
        [
            "",
            "This single synthetic fixture pair does not represent improvement on real SEM "
            "images, company data, production systems, or business outcomes.",
            "",
            "The report is rendered deterministically from `evaluation.json`; no LLM can modify "
            "its numbers or conclusion.",
            "",
            f"Machine-readable result: `{evaluation.evaluation_ref}`",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["render_evaluation_report"]
