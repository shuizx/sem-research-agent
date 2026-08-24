"""Stable, de-sensitized exceptions for provider-neutral port calls."""

from __future__ import annotations

from hashlib import sha256

from vision_research_ops.domain import JsonObject, StructuredFailure

from ._security import redact_json_object, redact_message
from .common import OperationContext


def _message_hash(message: str) -> str:
    """Return the domain content-hash spelling for a controlled error message."""
    return f"sha256:{sha256(message.encode('utf-8')).hexdigest()}"


def make_failure(
    *,
    code: str,
    category: str,
    message: str,
    retryable: bool,
    ctx: OperationContext | None,
    details: JsonObject | None = None,
) -> StructuredFailure:
    """Build a structured failure without retaining a raw provider exception."""
    safe_details = {} if details is None else redact_json_object(details)
    safe_message = redact_message(message, secret_values=details)
    return StructuredFailure(
        schema_version="1",
        code=code,
        category=category,
        message=safe_message,
        message_hash=_message_hash(safe_message),
        retryable=retryable,
        correlation_id=None if ctx is None else ctx.correlation_id,
        details=safe_details,
    )


class PortError(Exception):
    """Exception wrapper whose stable public payload is ``StructuredFailure`` only."""

    def __init__(self, failure: StructuredFailure) -> None:
        safe_message = redact_message(failure.message, secret_values=failure.details)
        self.failure = failure.model_copy(
            update={
                "message": safe_message,
                "message_hash": _message_hash(safe_message),
                "details": redact_json_object(failure.details),
            }
        )
        super().__init__(self.failure.code)


class ProviderError(PortError):
    """Failure returned by a paper or repository provider boundary."""


class LLMError(PortError):
    """Failure returned by the structured LLM boundary."""


class ArtifactError(PortError):
    """Failure returned by an immutable artifact-store boundary."""


class ExecutorError(PortError):
    """Failure returned by a bounded experiment-executor boundary."""


class TrackerError(PortError):
    """Failure returned by an experiment-tracker boundary."""


class PersistenceError(PortError):
    """Failure returned by the UnitOfWork persistence boundary."""


class OperationTimeoutError(PortError):
    """Explicit deadline or scripted timeout; callers can inspect retryability."""

    def __init__(self, operation: str, ctx: OperationContext) -> None:
        super().__init__(
            make_failure(
                code="PORT_OPERATION_TIMEOUT",
                category="TIMEOUT",
                message="The bounded port operation exceeded its deadline.",
                retryable=True,
                ctx=ctx,
                details={"operation": operation},
            )
        )


class OperationCancelledError(PortError):
    """Explicit cancellation observed at a port boundary."""

    def __init__(self, operation: str, ctx: OperationContext) -> None:
        super().__init__(
            make_failure(
                code="PORT_OPERATION_CANCELLED",
                category="CANCELLATION",
                message="The bounded port operation was cancelled.",
                retryable=False,
                ctx=ctx,
                details={"operation": operation},
            )
        )


class CapabilityNotSupportedError(PortError):
    """Fail-closed response for an operation absent from a port declaration."""

    def __init__(self, capability: str, ctx: OperationContext) -> None:
        super().__init__(
            make_failure(
                code="PORT_CAPABILITY_NOT_SUPPORTED",
                category="CAPABILITY",
                message="The requested port capability is not supported.",
                retryable=False,
                ctx=ctx,
                details={"capability": capability},
            )
        )


class IdempotencyKeyRequiredError(PortError):
    """Fail-closed response when a side effect lacks an explicit idempotency key."""

    def __init__(self, operation: str, ctx: OperationContext) -> None:
        super().__init__(
            make_failure(
                code="OPERATION_IDEMPOTENCY_KEY_REQUIRED",
                category="IDEMPOTENCY",
                message="The side-effect operation requires an idempotency key.",
                retryable=False,
                ctx=ctx,
                details={"operation": operation},
            )
        )


class IdempotencyConflictError(PortError):
    """Conflict for an already-used key paired with a different canonical payload."""

    def __init__(self, operation: str, ctx: OperationContext) -> None:
        super().__init__(
            make_failure(
                code="IDEMPOTENCY_KEY_REUSED_WITH_DIFFERENT_PAYLOAD",
                category="IDEMPOTENCY_CONFLICT",
                message="The idempotency key was already used with a different payload.",
                retryable=False,
                ctx=ctx,
                details={"operation": operation},
            )
        )


class StructuredOutputValidationError(LLMError):
    """Provider-neutral schema failure that never exposes the raw model response."""

    def __init__(self, task_name: str, ctx: OperationContext) -> None:
        super().__init__(
            make_failure(
                code="LLM_SCHEMA_VALIDATION_FAILED",
                category="LLM_SCHEMA",
                message="The structured LLM output failed schema validation.",
                retryable=True,
                ctx=ctx,
                details={"task_name": task_name},
            )
        )


class FakeScriptExhaustedError(PortError):
    """Test-only fail-closed response for an unscripted fake interaction."""

    def __init__(self, operation: str, ctx: OperationContext) -> None:
        super().__init__(
            make_failure(
                code="FAKE_SCRIPT_EXHAUSTED",
                category="TEST_FAKE",
                message="No scripted outcome was configured for the fake port operation.",
                retryable=False,
                ctx=ctx,
                details={"operation": operation},
            )
        )


__all__ = [
    "ArtifactError",
    "CapabilityNotSupportedError",
    "ExecutorError",
    "FakeScriptExhaustedError",
    "IdempotencyConflictError",
    "IdempotencyKeyRequiredError",
    "LLMError",
    "OperationCancelledError",
    "OperationTimeoutError",
    "PersistenceError",
    "PortError",
    "ProviderError",
    "StructuredOutputValidationError",
    "TrackerError",
    "make_failure",
]
