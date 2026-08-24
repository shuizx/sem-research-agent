"""Provider-neutral structured LLM port with no client or settings dependency."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from .common import OperationContext, StructuredGenerationRequest, StructuredGenerationResult

TStructured = TypeVar("TStructured", bound=BaseModel)


@runtime_checkable
class StructuredLLM(Protocol):
    """Generate only Pydantic-validated structured proposals from sanitized inputs."""

    async def generate(
        self,
        request: StructuredGenerationRequest[TStructured],
        *,
        ctx: OperationContext,
    ) -> StructuredGenerationResult[TStructured]:
        """Return validated output or a structured schema/provider failure."""


__all__ = ["StructuredLLM"]
