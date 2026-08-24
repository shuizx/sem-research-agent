"""Explicit scripted LLM for offline adaptation tests and pipeline demos."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Literal, TypeVar, cast

from pydantic import BaseModel, ValidationError

from vision_research_ops.application.services.adaptation_models import (
    AdaptationChangeProposal,
    AdaptationPlanProposal,
    CompatibilityGapProposal,
)
from vision_research_ops.ports import (
    GenerationUsage,
    LLMError,
    OperationContext,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    StructuredOutputValidationError,
    make_failure,
)

TStructured = TypeVar("TStructured", bound=BaseModel)


def _hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


def fixture_adaptation_proposal(facts: dict[str, object]) -> AdaptationPlanProposal:
    """Build the deterministic fixture proposal used by scripted planning tests."""
    dataset = facts.get("dataset_facts")
    if not isinstance(dataset, dict):
        raise ValueError("fixture adaptation facts require dataset_facts")
    labels = dataset.get("label_names")
    group_keys = dataset.get("group_keys")
    channels = dataset.get("channels")
    if (
        not isinstance(labels, list)
        or not all(isinstance(label, str) for label in labels)
        or not isinstance(group_keys, list)
        or not group_keys
        or not isinstance(group_keys[0], str)
        or isinstance(channels, bool)
        or not isinstance(channels, int)
    ):
        raise ValueError("fixture adaptation facts are malformed")
    label_names = cast(list[str], labels)
    split_group_keys = cast(list[str], group_keys)
    channel_count = channels

    gap_data = (
        (
            "gap-input-channels",
            "INPUT_CHANNELS",
            "The public fixture config does not declare grayscale channel handling.",
            f"Use exactly {channel_count} grayscale channel.",
        ),
        (
            "gap-num-classes",
            "NUM_CLASSES",
            "The fixture classifier head uses its repository default.",
            f"Set the head to {len(label_names)} de-identified classes.",
        ),
        (
            "gap-label-mapping",
            "LABEL_MAPPING",
            "The fixture has no frozen dataset label order.",
            "Use a dense deterministic mapping from the profile label order.",
        ),
        (
            "gap-group-split",
            "GROUP_SPLIT",
            "The fixture does not enforce group-disjoint splitting.",
            f"Split by the de-identified {split_group_keys[0]} group key.",
        ),
        (
            "gap-metrics-output",
            "METRICS_OUTPUT",
            "The fixture exposes no standard evaluation output contract.",
            "Emit Macro-F1, Balanced Accuracy, and per-class recall JSON.",
        ),
    )
    fields = (
        ("change-input-channels", "INPUT_CHANNELS", "/input/channels"),
        ("change-num-classes", "NUM_CLASSES", "/model/num_classes"),
        ("change-label-mapping", "LABEL_MAPPING", "/data/label_mapping"),
        ("change-group-split", "GROUP_SPLIT", "/data/group_split_key"),
        ("change-metric-names", "METRICS_OUTPUT", "/metrics/names"),
        ("change-metric-output", "METRICS_OUTPUT", "/metrics/output_file"),
    )
    return AdaptationPlanProposal(
        gaps=[
            CompatibilityGapProposal(
                gap_id=gap_id,
                area=cast(
                    Literal[
                        "INPUT_CHANNELS",
                        "NUM_CLASSES",
                        "LABEL_MAPPING",
                        "GROUP_SPLIT",
                        "METRICS_OUTPUT",
                    ],
                    area,
                ),
                current_state=current,
                required_state=required,
                risk="MEDIUM",
            )
            for gap_id, area, current, required in gap_data
        ],
        channels=channel_count,
        num_classes=len(label_names),
        label_mapping={label: index for index, label in enumerate(label_names)},
        group_split_key=split_group_keys[0],
        metrics=["macro_f1", "balanced_accuracy", "per_class_recall"],
        metrics_output_file="outputs/metrics.json",
        changes=[
            AdaptationChangeProposal(
                change_id=change_id,
                area=cast(
                    Literal[
                        "INPUT_CHANNELS",
                        "NUM_CLASSES",
                        "LABEL_MAPPING",
                        "GROUP_SPLIT",
                        "METRICS_OUTPUT",
                    ],
                    area,
                ),
                target_template="SEM_PLAIN_PYTORCH_CONFIG_V1",
                target_field=cast(
                    Literal[
                        "/input/channels",
                        "/model/num_classes",
                        "/data/label_mapping",
                        "/data/group_split_key",
                        "/metrics/names",
                        "/metrics/output_file",
                    ],
                    field,
                ),
                action="SET",
                reason="Compile the dataset contract with the fixed SEM template.",
            )
            for change_id, area, field in fields
        ],
        rationale=(
            "The controlled fixture needs only deterministic configuration adaptation; "
            "no dependency, shell, or arbitrary source edit is requested."
        ),
    )


class FixtureAdaptationLLM:
    """Produce labeled schema-valid proposals or explicit scripted failures."""

    def __init__(
        self,
        *,
        mode: Literal["success", "provider_failure", "schema_failure"] = "success",
    ) -> None:
        self.mode = mode
        self.requests: list[StructuredGenerationRequest[BaseModel]] = []

    @property
    def call_count(self) -> int:
        """Return the number of actual structured generation calls."""
        return len(self.requests)

    async def generate(
        self,
        request: StructuredGenerationRequest[TStructured],
        *,
        ctx: OperationContext,
    ) -> StructuredGenerationResult[TStructured]:
        """Build the proposal only from sanitized structured facts."""
        self.requests.append(cast(StructuredGenerationRequest[BaseModel], request))
        if request.task_name != "adaptation_plan":
            raise LLMError(
                make_failure(
                    code="FIXTURE_LLM_TASK_UNSUPPORTED",
                    category="FIXTURE",
                    message="The offline fixture LLM supports only adaptation planning.",
                    retryable=False,
                    ctx=ctx,
                )
            )
        if self.mode == "provider_failure":
            raise LLMError(
                make_failure(
                    code="FIXTURE_LLM_PROVIDER_FAILED",
                    category="LLM_PROVIDER",
                    message="The scripted adaptation provider failed explicitly.",
                    retryable=False,
                    ctx=ctx,
                )
            )
        if self.mode == "schema_failure":
            raise StructuredOutputValidationError(request.task_name, ctx)

        try:
            raw = fixture_adaptation_proposal(cast(dict[str, object], request.facts))
            value = request.response_schema.model_validate(raw.model_dump(mode="json"))
        except (ValidationError, ValueError):
            raise StructuredOutputValidationError(request.task_name, ctx) from None
        output = value.model_dump(mode="json")
        return StructuredGenerationResult[TStructured](
            schema_version="1",
            value=value,
            provider_id="offline-fixture",
            model_id="scripted-adaptation-plan-v1",
            usage=GenerationUsage(
                schema_version="1",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            ),
            latency_ms=0,
            prompt_hash=_hash(
                {
                    "template_id": request.prompt_template_id,
                    "version": request.prompt_version,
                    "facts": request.facts,
                }
            ),
            output_hash=_hash(output),
            finish_reason="FIXTURE",
        )


__all__ = ["FixtureAdaptationLLM", "fixture_adaptation_proposal"]
