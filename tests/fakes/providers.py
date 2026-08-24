"""Deterministic scripted paper, repository, and structured-LLM fake adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from vision_research_ops.domain import ArtifactRef
from vision_research_ops.ports import (
    LLMError,
    OperationContext,
    PaperQuery,
    PaperSearchPage,
    ProviderError,
    RawPaperRecord,
    RepositoryAnalysis,
    RepositoryMetadata,
    RepositoryPolicy,
    RepositoryResolution,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    StructuredOutputValidationError,
    make_failure,
)
from vision_research_ops.ports._security import canonical_json_bytes, canonical_json_hash
from vision_research_ops.ports.common import ExternalPaperId, GenerationUsage

from .script import IdempotencyLedger, ScriptedPort, ScriptStep, require_scripted_instance

TStructured = TypeVar("TStructured", bound=BaseModel)


def _require_optional_raw_paper(
    value: object,
    *,
    operation: str,
    ctx: OperationContext,
) -> RawPaperRecord | None:
    """Validate a scripted paper lookup result, including its documented ``None`` case."""
    if value is None:
        return None
    return require_scripted_instance(
        value,
        RawPaperRecord,
        operation=operation,
        ctx=ctx,
        error_type=ProviderError,
    )


class ScriptedPaperProvider(ScriptedPort):
    """PaperProvider fake with explicit pages, raw records, and no success fallback."""

    provider_name: str

    def __init__(
        self,
        *,
        provider_name: str = "scripted-paper",
        pages: Mapping[str | None, PaperSearchPage] | None = None,
        records: Mapping[tuple[str, str], RawPaperRecord] | None = None,
        script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        clock: Callable[[], datetime] | None = None,
        supported_operations: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            port_name="PaperProvider",
            supported_operations=supported_operations
            or ("paper.search", "paper.get_by_external_id"),
            script=script,
            clock=clock,
        )
        self.provider_name = provider_name
        self._pages = dict(pages or {})
        self._records = dict(records or {})

    async def search(
        self,
        query: PaperQuery,
        *,
        cursor: str | None,
        ctx: OperationContext,
    ) -> PaperSearchPage:
        """Return an explicitly scripted provider page, including empty pages."""
        operation = "paper.search"
        payload = {"query": query, "cursor": cursor}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        def default() -> PaperSearchPage:
            try:
                return self._pages[cursor]
            except KeyError as exc:
                raise ProviderError(
                    make_failure(
                        code="RETRIEVAL_PROVIDER_PAGE_NOT_SCRIPTED",
                        category="TEST_FAKE",
                        message="No deterministic paper page is configured for this cursor.",
                        retryable=False,
                        ctx=ctx,
                        details={"cursor": "<initial>" if cursor is None else cursor},
                    )
                ) from exc

        return await self._consume(
            operation=operation,
            payload=payload,
            ctx=ctx,
            default=default,
            validate_return=lambda value: require_scripted_instance(
                value,
                PaperSearchPage,
                operation=operation,
                ctx=ctx,
                error_type=ProviderError,
            ),
            error_type=ProviderError,
        )

    async def get_by_external_id(
        self,
        external_id: ExternalPaperId,
        *,
        ctx: OperationContext,
    ) -> RawPaperRecord | None:
        """Return a scripted raw record or an explicitly configured ``None`` result."""
        operation = "paper.get_by_external_id"
        payload = {"external_id": external_id}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        def default() -> RawPaperRecord | None:
            return self._records.get((external_id.provider_name, external_id.value))

        return await self._consume(
            operation=operation,
            payload=payload,
            ctx=ctx,
            default=default,
            validate_return=lambda value: _require_optional_raw_paper(
                value,
                operation=operation,
                ctx=ctx,
            ),
            error_type=ProviderError,
        )


class ScriptedRepositoryProvider(ScriptedPort):
    """RepositoryProvider fake with idempotent snapshots and static maps only."""

    def __init__(
        self,
        *,
        resolutions: Mapping[tuple[str, str | None], RepositoryResolution] | None = None,
        metadata: Mapping[str, RepositoryMetadata] | None = None,
        snapshots: Mapping[str, ArtifactRef] | None = None,
        script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        clock: Callable[[], datetime] | None = None,
        supported_operations: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            port_name="RepositoryProvider",
            supported_operations=supported_operations
            or ("repository.resolve", "repository.fetch_metadata", "repository.snapshot"),
            script=script,
            clock=clock,
        )
        self._resolutions = dict(resolutions or {})
        self._metadata = dict(metadata or {})
        self._snapshots = dict(snapshots or {})
        self._snapshot_ledger: IdempotencyLedger[ArtifactRef] = IdempotencyLedger()

    @property
    def snapshot_effect_count(self) -> int:
        """Return the number of distinct successful snapshot side effects."""
        return self._snapshot_ledger.effect_counts["repository.snapshot"]

    async def resolve(
        self,
        repository_url: str,
        revision: str | None,
        *,
        ctx: OperationContext,
    ) -> RepositoryResolution:
        """Resolve only a preconfigured repository reference to a complete SHA."""
        operation = "repository.resolve"
        payload = {"repository_url": repository_url, "revision": revision}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        def default() -> RepositoryResolution:
            try:
                return self._resolutions[(repository_url, revision)]
            except KeyError as exc:
                raise ProviderError(
                    make_failure(
                        code="REPOSITORY_RESOLUTION_NOT_FOUND",
                        category="NOT_FOUND",
                        message="No deterministic repository resolution is configured.",
                        retryable=False,
                        ctx=ctx,
                    )
                ) from exc

        return await self._consume(
            operation=operation,
            payload=payload,
            ctx=ctx,
            default=default,
            validate_return=lambda value: require_scripted_instance(
                value,
                RepositoryResolution,
                operation=operation,
                ctx=ctx,
                error_type=ProviderError,
            ),
            error_type=ProviderError,
        )

    async def fetch_metadata(
        self,
        repository: RepositoryResolution,
        *,
        ctx: OperationContext,
    ) -> RepositoryMetadata:
        """Return preconfigured metadata without executing repository code."""
        operation = "repository.fetch_metadata"
        payload = {"repository": repository}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        def default() -> RepositoryMetadata:
            try:
                return self._metadata[repository.commit_sha]
            except KeyError as exc:
                raise ProviderError(
                    make_failure(
                        code="REPOSITORY_METADATA_NOT_FOUND",
                        category="NOT_FOUND",
                        message="No deterministic repository metadata is configured.",
                        retryable=False,
                        ctx=ctx,
                    )
                ) from exc

        return await self._consume(
            operation=operation,
            payload=payload,
            ctx=ctx,
            default=default,
            validate_return=lambda value: require_scripted_instance(
                value,
                RepositoryMetadata,
                operation=operation,
                ctx=ctx,
                error_type=ProviderError,
            ),
            error_type=ProviderError,
        )

    async def snapshot(
        self,
        repository: RepositoryResolution,
        *,
        ctx: OperationContext,
    ) -> ArtifactRef:
        """Return a safely replayable snapshot only after an explicit scripted step."""
        operation = "repository.snapshot"
        payload = {"repository": repository}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        async def execute() -> ArtifactRef:
            def default() -> ArtifactRef:
                try:
                    return self._snapshots[repository.commit_sha]
                except KeyError as exc:
                    raise ProviderError(
                        make_failure(
                            code="REPOSITORY_SNAPSHOT_NOT_FOUND",
                            category="NOT_FOUND",
                            message="No deterministic repository snapshot is configured.",
                            retryable=False,
                            ctx=ctx,
                        )
                    ) from exc

            return await self._consume(
                operation=operation,
                payload=payload,
                ctx=ctx,
                default=default,
                validate_return=lambda value: require_scripted_instance(
                    value,
                    ArtifactRef,
                    operation=operation,
                    ctx=ctx,
                    error_type=ProviderError,
                ),
                error_type=ProviderError,
            )

        return await self._snapshot_ledger.replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )


class ScriptedStaticRepositoryAnalyzer(ScriptedPort):
    """StaticRepositoryAnalyzer fake that only returns prebuilt evidence models."""

    def __init__(
        self,
        *,
        analyses: Mapping[tuple[str, str, str], RepositoryAnalysis] | None = None,
        script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        clock: Callable[[], datetime] | None = None,
        supported_operations: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            port_name="StaticRepositoryAnalyzer",
            supported_operations=supported_operations or ("repository.analyze",),
            script=script,
            clock=clock,
        )
        self._analyses = dict(analyses or {})

    async def analyze(
        self,
        repository_archive: ArtifactRef,
        policy: RepositoryPolicy,
        *,
        ctx: OperationContext,
    ) -> RepositoryAnalysis:
        """Return scripted static analysis and never inspect a filesystem or archive."""
        operation = "repository.analyze"
        payload = {"repository_archive": repository_archive, "policy": policy}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        def default() -> RepositoryAnalysis:
            key = (repository_archive.artifact_id, policy.policy_id, policy.policy_version)
            try:
                return self._analyses[key]
            except KeyError as exc:
                raise ProviderError(
                    make_failure(
                        code="REPOSITORY_ANALYSIS_NOT_FOUND",
                        category="NOT_FOUND",
                        message="No deterministic repository analysis is configured.",
                        retryable=False,
                        ctx=ctx,
                    )
                ) from exc

        return await self._consume(
            operation=operation,
            payload=payload,
            ctx=ctx,
            default=default,
            validate_return=lambda value: require_scripted_instance(
                value,
                RepositoryAnalysis,
                operation=operation,
                ctx=ctx,
                error_type=ProviderError,
            ),
            error_type=ProviderError,
        )


class ScriptedStructuredLLM(ScriptedPort):
    """StructuredLLM fake that validates scripted raw values against the requested schema."""

    def __init__(
        self,
        *,
        outputs: Mapping[str, object] | None = None,
        provider_id: str = "scripted-llm",
        model_id: str = "scripted-model",
        usage: GenerationUsage | None = None,
        script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        clock: Callable[[], datetime] | None = None,
        supported_operations: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            port_name="StructuredLLM",
            supported_operations=supported_operations or ("llm.generate",),
            script=script,
            clock=clock,
        )
        self._outputs = dict(outputs or {})
        self._provider_id = provider_id
        self._model_id = model_id
        self._usage = usage or GenerationUsage(
            schema_version="1",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )

    async def generate(
        self,
        request: StructuredGenerationRequest[TStructured],
        *,
        ctx: OperationContext,
    ) -> StructuredGenerationResult[TStructured]:
        """Validate one repeatable scripted output or raise a de-sensitized schema failure."""
        operation = "llm.generate"
        payload = {"request": request}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        def default() -> StructuredGenerationResult[TStructured]:
            try:
                raw_value = self._outputs[request.task_name]
            except KeyError as exc:
                raise LLMError(
                    make_failure(
                        code="LLM_SCRIPTED_OUTPUT_NOT_FOUND",
                        category="TEST_FAKE",
                        message="No deterministic structured LLM output is configured.",
                        retryable=False,
                        ctx=ctx,
                        details={"task_name": request.task_name},
                    )
                ) from exc
            try:
                value = request.response_schema.model_validate(raw_value)
                return self._build_result(value, request=request)
            except (RecursionError, TypeError, UnicodeError, ValidationError, ValueError):
                raise StructuredOutputValidationError(request.task_name, ctx) from None

        return await self._consume(
            operation=operation,
            payload=payload,
            ctx=ctx,
            default=default,
            validate_return=lambda value: self._validate_scripted_result(
                value,
                request=request,
                operation=operation,
                ctx=ctx,
            ),
            error_type=LLMError,
        )

    def _build_result(
        self,
        value: TStructured,
        *,
        request: StructuredGenerationRequest[TStructured],
    ) -> StructuredGenerationResult[TStructured]:
        """Build a finite-JSON-validated fake result from a validated Pydantic value."""
        value_json = value.model_dump(mode="json")
        canonical_json_bytes(value_json)
        prompt_hash = canonical_json_hash(
            {
                "task_name": request.task_name,
                "prompt_template_id": request.prompt_template_id,
                "prompt_version": request.prompt_version,
                "facts": request.facts,
            }
        )
        return StructuredGenerationResult[TStructured](
            schema_version="1",
            value=value,
            provider_id=self._provider_id,
            model_id=self._model_id,
            usage=self._usage,
            latency_ms=0,
            prompt_hash=prompt_hash,
            output_hash=canonical_json_hash(value_json),
            finish_reason="STOP",
        )

    def _validate_scripted_result(
        self,
        value: object,
        *,
        request: StructuredGenerationRequest[TStructured],
        operation: str,
        ctx: OperationContext,
    ) -> StructuredGenerationResult[TStructured]:
        """Validate direct scripted LLM results without trusting generic type erasure."""
        result = require_scripted_instance(
            value,
            StructuredGenerationResult,
            operation=operation,
            ctx=ctx,
            error_type=LLMError,
        )
        try:
            validated_value = request.response_schema.model_validate(result.value)
            value_json = validated_value.model_dump(mode="json")
            canonical_json_bytes(value_json)
        except (RecursionError, TypeError, UnicodeError, ValidationError, ValueError):
            raise StructuredOutputValidationError(request.task_name, ctx) from None
        return StructuredGenerationResult[TStructured](
            schema_version="1",
            value=validated_value,
            provider_id=result.provider_id,
            model_id=result.model_id,
            usage=result.usage,
            latency_ms=result.latency_ms,
            prompt_hash=result.prompt_hash,
            output_hash=canonical_json_hash(value_json),
            finish_reason=result.finish_reason,
        )


__all__ = [
    "ScriptedPaperProvider",
    "ScriptedRepositoryProvider",
    "ScriptedStaticRepositoryAnalyzer",
    "ScriptedStructuredLLM",
]
