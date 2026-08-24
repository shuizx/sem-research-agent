"""schema recovery bounded live-schema recovery tests with no network or real key."""

from __future__ import annotations

from typing import cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from pydantic import ValidationError

from vision_research_ops.adapters.llm.dashscope import DashScopeStructuredLLM
from vision_research_ops.application.services.paper_models import (
    ApplicabilityEvidence,
    PaperApplicabilityDecision,
)
from vision_research_ops.ports import (
    LLMError,
    OperationContext,
    StructuredGenerationRequest,
    StructuredOutputValidationError,
)
from vision_research_ops.prompts.paper_applicability import (
    PROMPT_TEMPLATE_ID,
    PROMPT_VERSION,
)


class _ScriptedStructuredRunnable:
    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> object:
        self.calls.append(messages)
        response = next(self._responses)
        if isinstance(response, BaseException):
            raise response
        return response


class _StubChatModel:
    def __init__(self, runnable: _ScriptedStructuredRunnable) -> None:
        self.runnable = runnable
        self.schema: object = None
        self.options: dict[str, object] = {}

    def with_structured_output(self, schema: object, **options: object) -> object:
        self.schema = schema
        self.options = options
        return self.runnable


def _decision() -> PaperApplicabilityDecision:
    return PaperApplicabilityDecision(
        schema_version="1",
        summary="The paper evaluates a bounded classification method on public images.",
        applicable=True,
        recommendation="MEDIUM",
        relevance_score=0.7,
        confidence=0.7,
        task_match=0.8,
        modality_match=0.6,
        data_match=0.6,
        code_match=0.4,
        compute_fit=0.8,
        evidence=[
            ApplicabilityEvidence(
                schema_version="1",
                dimension="TASK",
                source_field="abstract",
                statement="The abstract describes an image-classification evaluation.",
            )
        ],
        risks=["The reported image modality differs from SEM."],
        rationale="The task is relevant, but microscopy transfer still needs validation.",
    )


def _missing_summary_error() -> ValidationError:
    payload = _decision().model_dump(mode="json")
    payload.pop("summary")
    payload["PUBLIC_MODEL_TEXT_SENTINEL"] = "must never enter the repair prompt"
    with pytest.raises(ValidationError) as captured:
        PaperApplicabilityDecision.model_validate(payload)
    return captured.value


def _raw(*, input_tokens: int, output_tokens: int) -> AIMessage:
    return AIMessage(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={"finish_reason": "stop"},
    )


def _request() -> StructuredGenerationRequest[PaperApplicabilityDecision]:
    return StructuredGenerationRequest[PaperApplicabilityDecision](
        schema_version="1",
        task_name="paper_applicability",
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_version=PROMPT_VERSION,
        response_schema=PaperApplicabilityDecision,
        facts={
            "paper": {
                "title": "Public paper title",
                "abstract": "PUBLIC_ABSTRACT_SENTINEL",
            },
            "problem": {"task": "image_classification"},
        },
        artifact_excerpts=[],
        model_parameters={"temperature": 0},
        budget_class="pipeline_research_small",
    )


def _context() -> OperationContext:
    return OperationContext(
        schema_version="1",
        correlation_id="correlation-schema-recovery",
        workflow_id="workflow-schema-recovery",
        actor_id="pipeline-user",
        sensitivity="PUBLIC",
    )


def _adapter(
    responses: list[object],
) -> tuple[DashScopeStructuredLLM, _ScriptedStructuredRunnable, _StubChatModel]:
    runnable = _ScriptedStructuredRunnable(responses)
    client = _StubChatModel(runnable)
    adapter = DashScopeStructuredLLM(
        cast(BaseChatModel, client),
        model_id="qwen-test",
    )
    return adapter, runnable, client


@pytest.mark.asyncio
async def test_valid_first_response_uses_one_provider_attempt() -> None:
    """an already-valid result keeps the normal one-call path."""
    decision = _decision()
    adapter, runnable, client = _adapter(
        [
            {
                "parsed": decision,
                "parsing_error": None,
                "raw": _raw(input_tokens=2, output_tokens=3),
            }
        ]
    )

    result = await adapter.generate(_request(), ctx=_context())

    assert result.value == decision
    assert result.usage.input_tokens == 2
    assert result.usage.output_tokens == 3
    assert result.usage.total_tokens == 5
    assert result.finish_reason == "stop"
    assert len(runnable.calls) == 1
    assert client.schema is PaperApplicabilityDecision
    assert client.options == {"method": "function_calling", "include_raw": True}


@pytest.mark.asyncio
async def test_schema_failure_retries_once_with_only_sanitized_hints() -> None:
    """one invalid result receives one field/type-only correction."""
    decision = _decision()
    invalid = {
        "parsed": None,
        "parsing_error": _missing_summary_error(),
        "raw": _raw(input_tokens=2, output_tokens=1),
    }
    adapter, runnable, _client = _adapter(
        [
            invalid,
            {
                "parsed": decision,
                "parsing_error": None,
                "raw": _raw(input_tokens=4, output_tokens=2),
            },
        ]
    )

    repaired = await adapter.generate(_request(), ctx=_context())
    normal_adapter, _normal_runnable, _normal_client = _adapter(
        [
            {
                "parsed": decision,
                "parsing_error": None,
                "raw": _raw(input_tokens=4, output_tokens=2),
            }
        ]
    )
    normal = await normal_adapter.generate(_request(), ctx=_context())

    assert repaired.value == decision
    assert repaired.usage.input_tokens == 6
    assert repaired.usage.output_tokens == 3
    assert repaired.usage.total_tokens == 9
    assert repaired.prompt_hash != normal.prompt_hash
    assert len(runnable.calls) == 2
    repair_messages = [
        message for message in runnable.calls[1] if isinstance(message, SystemMessage)
    ]
    assert len(repair_messages) == 2
    repair_text = str(repair_messages[1].content)
    assert "summary:missing" in repair_text
    assert "response:extra_forbidden" in repair_text
    assert "PUBLIC_MODEL_TEXT_SENTINEL" not in repair_text
    assert "PUBLIC_ABSTRACT_SENTINEL" not in repair_text
    assert "Public paper title" not in repair_text


@pytest.mark.asyncio
async def test_second_schema_failure_stops_without_a_third_attempt() -> None:
    """two invalid structured responses fail closed with exactly two calls."""
    invalid = {
        "parsed": None,
        "parsing_error": _missing_summary_error(),
        "raw": _raw(input_tokens=1, output_tokens=1),
    }
    adapter, runnable, _client = _adapter([invalid, invalid])

    with pytest.raises(StructuredOutputValidationError):
        await adapter.generate(_request(), ctx=_context())

    assert len(runnable.calls) == 2


@pytest.mark.asyncio
async def test_provider_failure_is_not_treated_as_schema_repair() -> None:
    """transport/provider errors retain their original one-call failure path."""
    adapter, runnable, _client = _adapter([RuntimeError("provider unavailable")])

    with pytest.raises(LLMError) as captured:
        await adapter.generate(_request(), ctx=_context())

    assert captured.value.failure.code == "LLM_PROVIDER_REQUEST_FAILED"
    assert len(runnable.calls) == 1
