"""Explicit offline fixture LLM for the manual pipeline research sample."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from vision_research_ops.application.services.paper_models import (
    ApplicabilityEvidence,
    PaperApplicabilityDecision,
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


class FixtureStructuredLLM:
    """Return one labeled deterministic decision; never masquerade as a live model."""

    async def generate(
        self,
        request: StructuredGenerationRequest[TStructured],
        *,
        ctx: OperationContext,
    ) -> StructuredGenerationResult[TStructured]:
        """Validate the fixture output through the caller's requested schema."""
        if request.task_name != "paper_applicability":
            raise LLMError(
                make_failure(
                    code="FIXTURE_LLM_TASK_UNSUPPORTED",
                    category="FIXTURE",
                    message="The offline fixture LLM supports only paper applicability.",
                    retryable=False,
                    ctx=ctx,
                )
            )
        paper_facts = request.facts.get("paper")
        is_mismatched_fixture = isinstance(
            paper_facts, dict
        ) and "Unsupervised Segmentation" in str(paper_facts.get("title", ""))
        raw = PaperApplicabilityDecision(
            summary=(
                "The paper studies unsupervised segmentation for natural RGB photographs."
                if is_mismatched_fixture
                else "The paper studies PyTorch image classification for wafer SEM defects."
            ),
            applicable=not is_mismatched_fixture,
            recommendation="REJECT" if is_mismatched_fixture else "HIGH",
            relevance_score=0.1 if is_mismatched_fixture else 0.9,
            confidence=0.8,
            task_match=0.1 if is_mismatched_fixture else 0.95,
            modality_match=0.1 if is_mismatched_fixture else 0.9,
            data_match=0.1 if is_mismatched_fixture else 0.8,
            code_match=0.0 if is_mismatched_fixture else 0.95,
            compute_fit=0.75,
            evidence=[
                ApplicabilityEvidence(
                    schema_version="1",
                    dimension="TASK",
                    source_field="abstract",
                    statement=(
                        "Fixture metadata describes natural-image segmentation rather than "
                        "SEM classification."
                        if is_mismatched_fixture
                        else "Fixture metadata explicitly describes SEM defect classification."
                    ),
                ),
                ApplicabilityEvidence(
                    schema_version="1",
                    dimension="CODE",
                    source_field="comment",
                    statement=(
                        "Fixture metadata contains no public code URL."
                        if is_mismatched_fixture
                        else "Fixture metadata includes a public PyTorch repository URL."
                    ),
                ),
            ],
            risks=["This is scripted fixture reasoning, not a live model judgment."],
            rationale=(
                "The fixture studies a different task and modality, so it is not a "
                "SEM classification target."
                if is_mismatched_fixture
                else "Fixture evidence supports samplenstrating the bounded Research Agent path."
            ),
        )
        try:
            value = request.response_schema.model_validate(raw.model_dump(mode="json"))
        except ValidationError:
            raise StructuredOutputValidationError(request.task_name, ctx) from None
        output = value.model_dump(mode="json")
        return StructuredGenerationResult[TStructured](
            schema_version="1",
            value=value,
            provider_id="offline-fixture",
            model_id="scripted-paper-applicability-v1",
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


__all__ = ["FixtureStructuredLLM"]
