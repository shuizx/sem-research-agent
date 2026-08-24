"""DashScope OpenAI-compatible structured LLM adapter."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from time import monotonic
from typing import TypeVar, cast

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from vision_research_ops.ports import (
    GenerationUsage,
    LLMError,
    OperationContext,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    StructuredOutputValidationError,
    make_failure,
)
from vision_research_ops.prompts.paper_applicability import (
    PROMPT_TEMPLATE_ID,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
)
from vision_research_ops.settings import Settings

TStructured = TypeVar("TStructured", bound=BaseModel)

_MAX_SCHEMA_ATTEMPTS = 2
_GENERIC_SCHEMA_HINT = "response:structured_output_invalid"


class _SchemaResponseError(Exception):
    """Internal validation signal containing field/type hints but no model values."""

    def __init__(self, hints: tuple[str, ...]) -> None:
        super().__init__("structured response did not validate")
        self.hints = hints


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: object) -> str:
    return f"sha256:{sha256(_canonical_bytes(value)).hexdigest()}"


def _nonnegative_token(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _usage(raw: object) -> GenerationUsage:
    metadata: object = raw.usage_metadata if isinstance(raw, AIMessage) else None
    if not isinstance(metadata, dict):
        metadata = {}
    return GenerationUsage(
        schema_version="1",
        input_tokens=_nonnegative_token(metadata.get("input_tokens")),
        output_tokens=_nonnegative_token(metadata.get("output_tokens")),
        total_tokens=_nonnegative_token(metadata.get("total_tokens")),
    )


def _add_usage(left: GenerationUsage, right: GenerationUsage) -> GenerationUsage:
    return GenerationUsage(
        schema_version="1",
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )


def _validation_error(value: object) -> ValidationError | None:
    """Find a nested Pydantic error without rendering provider-controlled values."""
    current = value
    seen: set[int] = set()
    for _ in range(4):
        if isinstance(current, ValidationError):
            return current
        if not isinstance(current, BaseException) or id(current) in seen:
            return None
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _schema_field_names(response_schema: type[BaseModel]) -> frozenset[str]:
    """Collect only declared JSON-Schema property names for safe error locations."""
    field_names: set[str] = set()
    pending: list[object] = [response_schema.model_json_schema()]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            properties = current.get("properties")
            if isinstance(properties, dict):
                field_names.update(key for key in properties if isinstance(key, str))
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return frozenset(field_names)


def _safe_location(raw_location: object, field_names: frozenset[str]) -> str:
    if not isinstance(raw_location, (list, tuple)):
        return "response"
    safe_parts: list[str] = []
    for part in raw_location[:8]:
        if isinstance(part, str) and part in field_names:
            safe_parts.append(part)
        elif isinstance(part, int) and not isinstance(part, bool) and part >= 0:
            safe_parts.append(str(part))
        else:
            break
    return ".".join(safe_parts) or "response"


def _validation_hints(
    value: object,
    response_schema: type[BaseModel],
) -> tuple[str, ...]:
    """Return bounded schema locations/types without input values or error messages."""
    error = _validation_error(value)
    if error is None:
        return (_GENERIC_SCHEMA_HINT,)
    hints: list[str] = []
    field_names = _schema_field_names(response_schema)
    for item in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )[:8]:
        location = _safe_location(item.get("loc", ()), field_names)
        error_type = item.get("type")
        stable_type = error_type if isinstance(error_type, str) else "validation_error"
        hints.append(f"{location}:{stable_type}")
    return tuple(hints) or (_GENERIC_SCHEMA_HINT,)


def _validated_response[TResponse: BaseModel](
    response: object,
    response_schema: type[TResponse],
) -> tuple[TResponse, dict[str, object], object]:
    if not isinstance(response, dict):
        raise _SchemaResponseError((_GENERIC_SCHEMA_HINT,))
    raw = response.get("raw")
    parsing_error = response.get("parsing_error")
    if parsing_error is not None:
        raise _SchemaResponseError(_validation_hints(parsing_error, response_schema))
    try:
        value = response_schema.model_validate(response.get("parsed"))
        value_json = value.model_dump(mode="json")
        _canonical_bytes(value_json)
    except ValidationError as error:
        raise _SchemaResponseError(_validation_hints(error, response_schema)) from None
    except (RecursionError, TypeError, UnicodeError, ValueError):
        raise _SchemaResponseError((_GENERIC_SCHEMA_HINT,)) from None
    return value, cast(dict[str, object], value_json), raw


def _repair_instruction(hints: tuple[str, ...]) -> str:
    joined = ", ".join(hints)
    return (
        "The previous function-call arguments failed strict schema validation at these "
        f"field/type locations: {joined}. Return a fresh function-call result for the same "
        "facts and the same response schema. Supply every required field, use only declared "
        "enum values, keep scores within 0..1, and make recommendation consistent with "
        "applicable. Do not add explanations or extra fields."
    )


def _finish_reason(raw: object) -> str:
    metadata: object = raw.response_metadata if isinstance(raw, AIMessage) else None
    if isinstance(metadata, dict):
        value = metadata.get("finish_reason")
        if isinstance(value, str) and value.strip():
            return value
    return "UNKNOWN"


class DashScopeStructuredLLM:
    """Use ChatOpenAI transport while enforcing the requested Pydantic schema."""

    def __init__(self, client: BaseChatModel, *, model_id: str) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be blank")
        self._client = client
        self._model_id = model_id

    @staticmethod
    def _prompt(request: StructuredGenerationRequest[BaseModel]) -> str:
        key = (request.prompt_template_id, request.prompt_version)
        if key != (PROMPT_TEMPLATE_ID, PROMPT_VERSION):
            raise ValueError("structured LLM request references an unknown prompt version")
        return SYSTEM_PROMPT

    async def generate(
        self,
        request: StructuredGenerationRequest[TStructured],
        *,
        ctx: OperationContext,
    ) -> StructuredGenerationResult[TStructured]:
        """Return validated output or a de-sensitized provider/schema failure."""
        if ctx.deadline_exceeded(now=datetime.now(UTC)):
            raise LLMError(
                make_failure(
                    code="LLM_REQUEST_DEADLINE_EXCEEDED",
                    category="TIMEOUT",
                    message="The structured LLM deadline elapsed before generation.",
                    retryable=True,
                    ctx=ctx,
                )
            )
        try:
            prompt = self._prompt(cast(StructuredGenerationRequest[BaseModel], request))
        except ValueError:
            raise LLMError(
                make_failure(
                    code="LLM_PROMPT_VERSION_UNKNOWN",
                    category="CONFIGURATION",
                    message="The requested structured prompt version is not registered.",
                    retryable=False,
                    ctx=ctx,
                )
            ) from None

        facts_json = _canonical_bytes(request.facts).decode("utf-8")
        structured = self._client.with_structured_output(
            request.response_schema,
            method="function_calling",
            include_raw=True,
        )
        started = monotonic()
        usage = GenerationUsage(
            schema_version="1",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )
        repair_hints: tuple[str, ...] | None = None
        raw: object = None
        value: TStructured
        value_json: dict[str, object]
        for attempt in range(_MAX_SCHEMA_ATTEMPTS):
            messages: list[SystemMessage | HumanMessage] = [SystemMessage(content=prompt)]
            if repair_hints is not None:
                messages.append(SystemMessage(content=_repair_instruction(repair_hints)))
            messages.append(HumanMessage(content=facts_json))
            try:
                response = await structured.ainvoke(messages)
            except Exception:
                raise LLMError(
                    make_failure(
                        code="LLM_PROVIDER_REQUEST_FAILED",
                        category="LLM_PROVIDER",
                        message="The structured LLM provider request failed.",
                        retryable=True,
                        ctx=ctx,
                    )
                ) from None
            response_raw = response.get("raw") if isinstance(response, dict) else None
            usage = _add_usage(usage, _usage(response_raw))
            try:
                value, value_json, raw = _validated_response(
                    response,
                    request.response_schema,
                )
            except _SchemaResponseError as error:
                if attempt + 1 >= _MAX_SCHEMA_ATTEMPTS:
                    raise StructuredOutputValidationError(request.task_name, ctx) from None
                if ctx.deadline_exceeded(now=datetime.now(UTC)):
                    raise LLMError(
                        make_failure(
                            code="LLM_REQUEST_DEADLINE_EXCEEDED",
                            category="TIMEOUT",
                            message="The structured LLM deadline elapsed before schema repair.",
                            retryable=True,
                            ctx=ctx,
                        )
                    ) from None
                repair_hints = error.hints
                continue
            break
        else:  # pragma: no cover - the bounded loop either returns or raises above
            raise StructuredOutputValidationError(request.task_name, ctx)

        latency_ms = max(0, int((monotonic() - started) * 1000))
        prompt_payload: dict[str, object] = {
            "template_id": request.prompt_template_id,
            "version": request.prompt_version,
            "system": prompt,
            "facts": request.facts,
        }
        if repair_hints is not None:
            prompt_payload["schema_repair"] = {
                "attempts": _MAX_SCHEMA_ATTEMPTS,
                "validation_hints": list(repair_hints),
            }

        return StructuredGenerationResult[TStructured](
            schema_version="1",
            value=value,
            provider_id="dashscope-openai-compatible",
            model_id=self._model_id,
            usage=usage,
            latency_ms=latency_ms,
            prompt_hash=_content_hash(prompt_payload),
            output_hash=_content_hash(value_json),
            finish_reason=_finish_reason(raw),
        )


def build_dashscope_chat_model(settings: Settings) -> ChatOpenAI:
    """Build the injected OpenAI-compatible chat transport without reading env here."""
    api_key = settings.dashscope_api_key
    if api_key is None:
        settings.require_dashscope_api_key()
        raise AssertionError("unreachable")
    return ChatOpenAI(
        api_key=api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=0,
        max_retries=1,
        http_client=httpx.Client(trust_env=False),
        http_async_client=httpx.AsyncClient(trust_env=False),
        http_socket_options=(),
    )


def build_dashscope_llm(settings: Settings) -> DashScopeStructuredLLM:
    """Build the schema-bound LLM adapter over the shared chat transport."""
    client = build_dashscope_chat_model(settings)
    return DashScopeStructuredLLM(client, model_id=settings.llm_model)


__all__ = ["DashScopeStructuredLLM", "build_dashscope_chat_model", "build_dashscope_llm"]
