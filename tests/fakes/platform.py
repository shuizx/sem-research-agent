"""Deterministic scripted artifact, dataset, workspace, executor, and tracker fakes."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from datetime import datetime, timedelta
from hashlib import sha256

from vision_research_ops.domain import (
    ArtifactRef,
    DatasetProfile,
    RepositorySnapshot,
    RunStatus,
    ValidationResult,
)
from vision_research_ops.ports import (
    ArtifactDescriptor,
    ArtifactError,
    CancellationResult,
    DatasetMountSpec,
    DownloadGrant,
    ExecutorError,
    ExternalRunStatus,
    FrozenRunSpec,
    IdempotencyKeyRequiredError,
    MetricPoint,
    OperationContext,
    PatchDocument,
    PatchResult,
    PortError,
    RunManifest,
    SubmissionResult,
    TrackerError,
    TrackerRunRef,
    ValidationStageSpec,
    WorkspaceRef,
    make_failure,
)
from vision_research_ops.ports.base import AsyncBinaryReader

from .script import (
    IdempotencyLedger,
    ScriptedPort,
    ScriptStep,
    _ScriptReservation,
    require_scripted_async_binary_reader,
    require_scripted_instance,
    require_scripted_none,
)

_EXPECTED_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class InMemoryAsyncReader:
    """One-chunk asynchronous reader for immutable bytes retained by a fake store."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._iterated = False

    async def read(self) -> bytes:
        """Return all retained bytes without altering the immutable value."""
        return self._data

    def __aiter__(self) -> InMemoryAsyncReader:
        """Return the reader as an asynchronous one-chunk iterator."""
        return self

    async def __anext__(self) -> bytes:
        """Yield the immutable payload once."""
        if self._iterated:
            raise StopAsyncIteration
        self._iterated = True
        return self._data


class InMemoryArtifactStore(ScriptedPort):
    """ArtifactStore fake with explicit scripts, hash checks, immutability, and replay."""

    def __init__(
        self,
        *,
        script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        clock: Callable[[], datetime] | None = None,
        supported_operations: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            port_name="ArtifactStore",
            supported_operations=supported_operations
            or (
                "artifact.put_bytes",
                "artifact.open",
                "artifact.stat",
                "artifact.issue_download",
            ),
            script=script,
            clock=clock,
        )
        self._bytes: dict[str, bytes] = {}
        self._refs: dict[str, ArtifactRef] = {}
        self._put_ledger: IdempotencyLedger[ArtifactRef] = IdempotencyLedger()
        self._grant_ledger: IdempotencyLedger[DownloadGrant] = IdempotencyLedger()

    @property
    def put_effect_count(self) -> int:
        """Return the number of successful distinct artifact-finalization effects."""
        return self._put_ledger.effect_counts["artifact.put_bytes"]

    async def put_bytes(
        self,
        data: AsyncIterator[bytes] | bytes,
        descriptor: ArtifactDescriptor,
        *,
        expected_sha256: str | None,
        ctx: OperationContext,
    ) -> ArtifactRef:
        """Preflight first, then materialize and finalize explicitly permitted bytes once."""
        operation = "artifact.put_bytes"
        preflight_payload = {
            "data": data if isinstance(data, bytes) else {"stream": "async_iterator"},
            "descriptor": descriptor,
            "expected_sha256": expected_sha256,
        }
        record_index = self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=preflight_payload,
            ctx=ctx,
        )
        if ctx.idempotency_key is None:
            raise IdempotencyKeyRequiredError(operation, ctx)
        reservation: _ScriptReservation | None = None
        try:
            self._validate_put_preflight(
                data=data,
                descriptor=descriptor,
                expected_sha256=expected_sha256,
                ctx=ctx,
            )
            if not self._put_ledger.has_replay_state(ctx.idempotency_key):
                reservation = self._reserve_success_step(
                    operation=operation,
                    reservation_key=ctx.idempotency_key,
                    ctx=ctx,
                    error_type=ArtifactError,
                )
            content = data if isinstance(data, bytes) else await self._collect_bytes(data, ctx)
            payload = {
                "data": content,
                "descriptor": descriptor,
                "expected_sha256": expected_sha256,
            }
            self._replace_call_payload(record_index, payload)
            stored = await self._store_permitted_content(
                content=content,
                descriptor=descriptor,
                expected_sha256=expected_sha256,
                ctx=ctx,
                reservation=reservation,
            )
            if reservation is not None:
                self._release_step_reservation(
                    operation=operation,
                    reservation_key=ctx.idempotency_key,
                    reservation=reservation,
                )
            return stored
        except BaseException:
            if reservation is not None:
                self._release_step_reservation(
                    operation=operation,
                    reservation_key=ctx.idempotency_key,
                    reservation=reservation,
                )
            raise

    def _validate_put_preflight(
        self,
        *,
        data: object,
        descriptor: object,
        expected_sha256: object,
        ctx: OperationContext,
    ) -> None:
        """Reject invalid scalar or stream-shape inputs before iterating their content."""
        if not isinstance(data, bytes) and not self._is_safe_async_stream_shape(data):
            raise ArtifactError(
                make_failure(
                    code="ARTIFACT_STREAM_TYPE_INVALID",
                    category="VALIDATION",
                    message="Artifact content must be bytes or an asynchronous byte iterator.",
                    retryable=False,
                    ctx=ctx,
                )
            )
        if not isinstance(descriptor, ArtifactDescriptor):
            raise ArtifactError(
                make_failure(
                    code="ARTIFACT_DESCRIPTOR_INVALID",
                    category="VALIDATION",
                    message="Artifact metadata must use the typed artifact descriptor contract.",
                    retryable=False,
                    ctx=ctx,
                )
            )
        if expected_sha256 is None:
            return
        if type(expected_sha256) is not str or not _EXPECTED_SHA256_PATTERN.fullmatch(
            expected_sha256
        ):
            raise ArtifactError(
                make_failure(
                    code="ARTIFACT_EXPECTED_SHA256_INVALID",
                    category="VALIDATION",
                    message=(
                        "The expected artifact SHA-256 must use canonical sha256 lowercase form."
                    ),
                    retryable=False,
                    ctx=ctx,
                )
            )

    @staticmethod
    def _is_safe_async_stream_shape(data: object) -> bool:
        """Probe a stream shape without surfacing a malicious attribute exception."""
        try:
            return callable(data.__aiter__) and callable(data.__anext__)
        except Exception:
            return False

    async def _store_permitted_content(
        self,
        *,
        content: bytes,
        descriptor: ArtifactDescriptor,
        expected_sha256: str | None,
        ctx: OperationContext,
        reservation: _ScriptReservation | None,
    ) -> ArtifactRef:
        """Finalize preflight-approved bytes without retaining raw content in call records."""
        operation = "artifact.put_bytes"
        payload = {
            "data": content,
            "descriptor": descriptor,
            "expected_sha256": expected_sha256,
        }

        async def execute() -> ArtifactRef:
            def default() -> ArtifactRef:
                content_hash = f"sha256:{sha256(content).hexdigest()}"
                if expected_sha256 is not None and expected_sha256 != content_hash:
                    raise ArtifactError(
                        make_failure(
                            code="ARTIFACT_HASH_MISMATCH",
                            category="ARTIFACT",
                            message="Artifact bytes do not match the expected SHA-256 hash.",
                            retryable=False,
                            ctx=ctx,
                        )
                    )
                existing = self._bytes.get(descriptor.artifact_id)
                if existing is not None and existing != content:
                    raise ArtifactError(
                        make_failure(
                            code="ARTIFACT_IMMUTABLE_CONTENT_CONFLICT",
                            category="ARTIFACT",
                            message=(
                                "A finalized artifact ID cannot be overwritten "
                                "with different bytes."
                            ),
                            retryable=False,
                            ctx=ctx,
                        )
                    )
                if existing is not None:
                    return self._refs[descriptor.artifact_id]
                reference = ArtifactRef(
                    schema_version="1",
                    artifact_id=descriptor.artifact_id,
                    kind=descriptor.kind,
                    uri=f"fake://artifacts/{descriptor.artifact_id}",
                    sha256=content_hash,
                    size_bytes=len(content),
                    media_type=descriptor.media_type,
                    created_at=self._clock(),
                    producer=descriptor.producer,
                    sensitivity=descriptor.sensitivity,
                    metadata=descriptor.metadata,
                )
                self._bytes[descriptor.artifact_id] = content
                self._refs[descriptor.artifact_id] = reference
                return reference

            def validate_return(value: object) -> ArtifactRef:
                return require_scripted_instance(
                    value,
                    ArtifactRef,
                    operation=operation,
                    ctx=ctx,
                    error_type=ArtifactError,
                )

            if reservation is not None:
                return await self._consume_reserved_step(
                    operation=operation,
                    reservation_key=ctx.idempotency_key or "",
                    reservation=reservation,
                    default=default,
                    validate_return=validate_return,
                )
            return await self._consume(
                operation=operation,
                payload=payload,
                ctx=ctx,
                default=default,
                validate_return=validate_return,
                error_type=ArtifactError,
            )

        return await self._put_ledger.replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )

    async def open(self, artifact_id: str, *, ctx: OperationContext) -> AsyncBinaryReader:
        """Return an explicit scripted reader for a finalized in-memory artifact."""
        operation = "artifact.open"
        payload = {"artifact_id": artifact_id}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        def default() -> AsyncBinaryReader:
            try:
                return InMemoryAsyncReader(self._bytes[artifact_id])
            except KeyError as exc:
                raise ArtifactError(
                    make_failure(
                        code="ARTIFACT_NOT_FOUND",
                        category="NOT_FOUND",
                        message="The requested artifact does not exist.",
                        retryable=False,
                        ctx=ctx,
                    )
                ) from exc

        return await self._consume(
            operation=operation,
            payload=payload,
            ctx=ctx,
            default=default,
            validate_return=lambda value: require_scripted_async_binary_reader(
                value,
                operation=operation,
                ctx=ctx,
                error_type=ArtifactError,
            ),
            error_type=ArtifactError,
        )

    async def stat(self, artifact_id: str, *, ctx: OperationContext) -> ArtifactRef:
        """Return explicitly scripted immutable metadata for a finalized artifact."""
        operation = "artifact.stat"
        payload = {"artifact_id": artifact_id}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        def default() -> ArtifactRef:
            try:
                return self._refs[artifact_id]
            except KeyError as exc:
                raise ArtifactError(
                    make_failure(
                        code="ARTIFACT_NOT_FOUND",
                        category="NOT_FOUND",
                        message="The requested artifact does not exist.",
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
                error_type=ArtifactError,
            ),
            error_type=ArtifactError,
        )

    async def issue_download(
        self,
        artifact_id: str,
        ttl_seconds: int,
        *,
        ctx: OperationContext,
    ) -> DownloadGrant:
        """Issue an idempotent fixed-clock grant for an existing artifact only."""
        operation = "artifact.issue_download"
        payload = {"artifact_id": artifact_id, "ttl_seconds": ttl_seconds}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        async def execute() -> DownloadGrant:
            def default() -> DownloadGrant:
                if type(ttl_seconds) is not int or ttl_seconds <= 0:
                    raise ArtifactError(
                        make_failure(
                            code="ARTIFACT_DOWNLOAD_TTL_INVALID",
                            category="VALIDATION",
                            message="Download grant TTL must be a positive integer.",
                            retryable=False,
                            ctx=ctx,
                        )
                    )
                try:
                    reference = self._refs[artifact_id]
                except KeyError as exc:
                    raise ArtifactError(
                        make_failure(
                            code="ARTIFACT_NOT_FOUND",
                            category="NOT_FOUND",
                            message="The requested artifact does not exist.",
                            retryable=False,
                            ctx=ctx,
                        )
                    ) from exc
                now = self._clock()
                return DownloadGrant(
                    schema_version="1",
                    grant_id=f"grant_{artifact_id}_{ctx.actor_id}",
                    artifact_id=artifact_id,
                    uri=reference.uri,
                    expires_at=now + timedelta(seconds=ttl_seconds),
                )

            return await self._consume(
                operation=operation,
                payload=payload,
                ctx=ctx,
                default=default,
                validate_return=lambda value: require_scripted_instance(
                    value,
                    DownloadGrant,
                    operation=operation,
                    ctx=ctx,
                    error_type=ArtifactError,
                ),
                error_type=ArtifactError,
            )

        return await self._grant_ledger.replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )

    async def _collect_bytes(
        self,
        data: AsyncIterator[bytes] | bytes,
        ctx: OperationContext,
    ) -> bytes:
        """Collect a bounded test payload without touching a filesystem or network."""
        if isinstance(data, bytes):
            return data
        chunks: list[bytes] = []
        try:
            async for chunk in data:
                if not isinstance(chunk, bytes):
                    raise ArtifactError(
                        make_failure(
                            code="ARTIFACT_CHUNK_TYPE_INVALID",
                            category="VALIDATION",
                            message="Artifact streams must yield bytes chunks.",
                            retryable=False,
                            ctx=ctx,
                        )
                    )
                chunks.append(chunk)
        except PortError:
            raise
        except Exception:
            raise ArtifactError(
                make_failure(
                    code="ARTIFACT_STREAM_READ_FAILED",
                    category="ARTIFACT",
                    message="The artifact stream could not be read safely.",
                    retryable=False,
                    ctx=ctx,
                )
            ) from None
        return b"".join(chunks)


class ScriptedDatasetCatalog(ScriptedPort):
    """DatasetCatalog fake that returns only supplied profile and opaque mount values."""

    def __init__(
        self,
        *,
        profiles: Mapping[tuple[str, str], DatasetProfile] | None = None,
        mount_specs: Mapping[tuple[str, str], DatasetMountSpec] | None = None,
        script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        clock: Callable[[], datetime] | None = None,
        supported_operations: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            port_name="DatasetCatalog",
            supported_operations=supported_operations
            or ("dataset.get_profile", "dataset.get_mount_spec"),
            script=script,
            clock=clock,
        )
        self._profiles = dict(profiles or {})
        self._mount_specs = dict(mount_specs or {})

    async def get_profile(
        self,
        dataset_id: str,
        version: str,
        *,
        ctx: OperationContext,
    ) -> DatasetProfile:
        """Return a supplied de-identified profile or an explicit not-found error."""
        operation = "dataset.get_profile"
        payload = {"dataset_id": dataset_id, "version": version}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        def default() -> DatasetProfile:
            try:
                return self._profiles[(dataset_id, version)]
            except KeyError as exc:
                raise ArtifactError(
                    make_failure(
                        code="DATASET_PROFILE_NOT_FOUND",
                        category="NOT_FOUND",
                        message="No deterministic dataset profile is configured.",
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
                DatasetProfile,
                operation=operation,
                ctx=ctx,
                error_type=ArtifactError,
            ),
            error_type=ArtifactError,
        )

    async def get_mount_spec(
        self,
        dataset_id: str,
        version: str,
        *,
        ctx: OperationContext,
    ) -> DatasetMountSpec:
        """Return a supplied opaque mount reference without exposing a host path."""
        operation = "dataset.get_mount_spec"
        payload = {"dataset_id": dataset_id, "version": version}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        def default() -> DatasetMountSpec:
            try:
                return self._mount_specs[(dataset_id, version)]
            except KeyError as exc:
                raise ArtifactError(
                    make_failure(
                        code="DATASET_MOUNT_NOT_FOUND",
                        category="NOT_FOUND",
                        message="No deterministic dataset mount specification is configured.",
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
                DatasetMountSpec,
                operation=operation,
                ctx=ctx,
                error_type=ArtifactError,
            ),
            error_type=ArtifactError,
        )


class ScriptedPatchWorkspace(ScriptedPort):
    """PatchWorkspace fake using prebuilt opaque handles and immutable patch artifacts."""

    def __init__(
        self,
        *,
        workspaces: Mapping[str, WorkspaceRef] | None = None,
        patch_results: Mapping[tuple[str, str], PatchResult] | None = None,
        exports: Mapping[str, ArtifactRef] | None = None,
        script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        clock: Callable[[], datetime] | None = None,
        supported_operations: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            port_name="PatchWorkspace",
            supported_operations=supported_operations
            or (
                "workspace.create",
                "workspace.apply_patch",
                "workspace.export_patch",
                "workspace.destroy",
            ),
            script=script,
            clock=clock,
        )
        self._workspaces = dict(workspaces or {})
        self._patch_results = dict(patch_results or {})
        self._exports = dict(exports or {})
        self._destroyed: set[str] = set()
        self._ledgers: dict[str, IdempotencyLedger[object]] = {
            "workspace.create": IdempotencyLedger(),
            "workspace.apply_patch": IdempotencyLedger(),
            "workspace.export_patch": IdempotencyLedger(),
            "workspace.destroy": IdempotencyLedger(),
        }

    async def create(
        self,
        repository: RepositorySnapshot,
        operation_id: str,
        *,
        ctx: OperationContext,
    ) -> WorkspaceRef:
        """Create an explicitly scripted opaque workspace for a fixed snapshot."""
        operation = "workspace.create"
        payload = {"repository": repository, "operation_id": operation_id}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        async def execute() -> WorkspaceRef:
            def default() -> WorkspaceRef:
                try:
                    return self._workspaces[operation_id]
                except KeyError as exc:
                    raise ArtifactError(
                        make_failure(
                            code="WORKSPACE_REFERENCE_NOT_FOUND",
                            category="NOT_FOUND",
                            message="No deterministic workspace is configured for this operation.",
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
                    WorkspaceRef,
                    operation=operation,
                    ctx=ctx,
                    error_type=ArtifactError,
                ),
                error_type=ArtifactError,
            )

        return await self._ledgers[operation].replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )

    async def apply_patch(
        self,
        workspace: WorkspaceRef,
        patch: PatchDocument,
        *,
        ctx: OperationContext,
    ) -> PatchResult:
        """Apply only an explicitly configured structured patch result."""
        operation = "workspace.apply_patch"
        payload = {"workspace": workspace, "patch": patch}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        async def execute() -> PatchResult:
            def default() -> PatchResult:
                try:
                    return self._patch_results[(workspace.workspace_id, patch.patch_id)]
                except KeyError as exc:
                    raise ArtifactError(
                        make_failure(
                            code="WORKSPACE_PATCH_RESULT_NOT_FOUND",
                            category="NOT_FOUND",
                            message="No deterministic patch result is configured.",
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
                    PatchResult,
                    operation=operation,
                    ctx=ctx,
                    error_type=ArtifactError,
                ),
                error_type=ArtifactError,
            )

        return await self._ledgers[operation].replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )

    async def export_patch(self, workspace: WorkspaceRef, *, ctx: OperationContext) -> ArtifactRef:
        """Export only the preconfigured immutable patch artifact."""
        operation = "workspace.export_patch"
        payload = {"workspace": workspace}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        async def execute() -> ArtifactRef:
            def default() -> ArtifactRef:
                try:
                    return self._exports[workspace.workspace_id]
                except KeyError as exc:
                    raise ArtifactError(
                        make_failure(
                            code="WORKSPACE_EXPORT_NOT_FOUND",
                            category="NOT_FOUND",
                            message="No deterministic workspace export is configured.",
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
                    error_type=ArtifactError,
                ),
                error_type=ArtifactError,
            )

        return await self._ledgers[operation].replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )

    async def destroy(self, workspace: WorkspaceRef, *, ctx: OperationContext) -> None:
        """Destroy an opaque workspace through an idempotent explicit script step."""
        operation = "workspace.destroy"
        payload = {"workspace": workspace}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        async def execute() -> None:
            def default() -> None:
                self._destroyed.add(workspace.workspace_id)

            return await self._consume(
                operation=operation,
                payload=payload,
                ctx=ctx,
                default=default,
                validate_return=lambda value: require_scripted_none(
                    value,
                    operation=operation,
                    ctx=ctx,
                    error_type=ArtifactError,
                ),
                error_type=ArtifactError,
            )

        await self._ledgers[operation].replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )


class ScriptedValidationRunner(ScriptedPort):
    """ValidationRunner fake that returns prebuilt domain ValidationResult values only."""

    def __init__(
        self,
        *,
        results: Mapping[str, ValidationResult] | None = None,
        script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        clock: Callable[[], datetime] | None = None,
        supported_operations: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            port_name="ValidationRunner",
            supported_operations=supported_operations or ("validation.run_stage",),
            script=script,
            clock=clock,
        )
        self._results = dict(results or {})
        self._ledger: IdempotencyLedger[ValidationResult] = IdempotencyLedger()

    async def run_stage(
        self,
        spec: ValidationStageSpec,
        *,
        ctx: OperationContext,
    ) -> ValidationResult:
        """Run exactly one explicit scripted stage without a subprocess or sleep."""
        operation = "validation.run_stage"
        payload = {"spec": spec}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        async def execute() -> ValidationResult:
            def default() -> ValidationResult:
                try:
                    return self._results[spec.validation_id]
                except KeyError as exc:
                    raise ExecutorError(
                        make_failure(
                            code="VALIDATION_RESULT_NOT_FOUND",
                            category="NOT_FOUND",
                            message="No deterministic validation result is configured.",
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
                    ValidationResult,
                    operation=operation,
                    ctx=ctx,
                    error_type=ExecutorError,
                ),
                error_type=ExecutorError,
            )

        return await self._ledger.replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )


class ScriptedExperimentExecutor(ScriptedPort):
    """ExperimentExecutor fake with distinct submit, poll, and cancel script queues."""

    executor_name = "scripted-executor"

    def __init__(
        self,
        *,
        submissions: Mapping[str, SubmissionResult] | None = None,
        statuses: Mapping[str, ExternalRunStatus] | None = None,
        cancellations: Mapping[str, CancellationResult] | None = None,
        script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        clock: Callable[[], datetime] | None = None,
        supported_operations: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            port_name="ExperimentExecutor",
            supported_operations=supported_operations
            or ("executor.submit", "executor.get_status", "executor.cancel"),
            script=script,
            clock=clock,
        )
        self._submissions = dict(submissions or {})
        self._statuses = dict(statuses or {})
        self._cancellations = dict(cancellations or {})
        self._submit_ledger: IdempotencyLedger[SubmissionResult] = IdempotencyLedger()
        self._cancel_ledger: IdempotencyLedger[CancellationResult] = IdempotencyLedger()

    @property
    def submit_effect_count(self) -> int:
        """Return how many distinct external submit effects the fake observed."""
        return self._submit_ledger.effect_counts["executor.submit"]

    async def submit(self, run: FrozenRunSpec, *, ctx: OperationContext) -> SubmissionResult:
        """Submit a frozen run once per run idempotency key and payload fingerprint."""
        operation = "executor.submit"
        payload = {"run": run}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        async def execute() -> SubmissionResult:
            def default() -> SubmissionResult:
                try:
                    return self._submissions[run.run_id]
                except KeyError as exc:
                    raise ExecutorError(
                        make_failure(
                            code="EXECUTOR_SUBMISSION_NOT_FOUND",
                            category="NOT_FOUND",
                            message="No deterministic submission result is configured.",
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
                    SubmissionResult,
                    operation=operation,
                    ctx=ctx,
                    error_type=ExecutorError,
                ),
                error_type=ExecutorError,
            )

        return await self._submit_ledger.replay_or_execute(
            operation=operation,
            idempotency_key=run.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )

    async def get_status(self, external_job_id: str, *, ctx: OperationContext) -> ExternalRunStatus:
        """Perform one explicit scripted status lookup with no polling loop."""
        operation = "executor.get_status"
        payload = {"external_job_id": external_job_id}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        def default() -> ExternalRunStatus:
            try:
                return self._statuses[external_job_id]
            except KeyError as exc:
                raise ExecutorError(
                    make_failure(
                        code="EXECUTOR_JOB_NOT_FOUND",
                        category="NOT_FOUND",
                        message="No deterministic external job status is configured.",
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
                ExternalRunStatus,
                operation=operation,
                ctx=ctx,
                error_type=ExecutorError,
            ),
            error_type=ExecutorError,
        )

    async def cancel(self, external_job_id: str, *, ctx: OperationContext) -> CancellationResult:
        """Issue a safely replayable cancellation request for one preconfigured job."""
        operation = "executor.cancel"
        payload = {"external_job_id": external_job_id}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        async def execute() -> CancellationResult:
            def default() -> CancellationResult:
                try:
                    return self._cancellations[external_job_id]
                except KeyError as exc:
                    raise ExecutorError(
                        make_failure(
                            code="EXECUTOR_CANCELLATION_NOT_FOUND",
                            category="NOT_FOUND",
                            message="No deterministic cancellation result is configured.",
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
                    CancellationResult,
                    operation=operation,
                    ctx=ctx,
                    error_type=ExecutorError,
                ),
                error_type=ExecutorError,
            )

        return await self._cancel_ledger.replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )


class InMemoryExperimentTracker(ScriptedPort):
    """ExperimentTracker fake that captures structured evidence without MLflow or files."""

    def __init__(
        self,
        *,
        runs: Mapping[str, TrackerRunRef] | None = None,
        script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        clock: Callable[[], datetime] | None = None,
        supported_operations: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            port_name="ExperimentTracker",
            supported_operations=supported_operations
            or (
                "tracker.create_run",
                "tracker.log_metrics",
                "tracker.log_artifact_refs",
                "tracker.finalize",
            ),
            script=script,
            clock=clock,
        )
        self._runs = dict(runs or {})
        self.metrics: dict[str, list[MetricPoint]] = {}
        self.artifacts: dict[str, list[ArtifactRef]] = {}
        self.final_statuses: dict[str, RunStatus] = {}
        self._create_ledger: IdempotencyLedger[TrackerRunRef] = IdempotencyLedger()
        self._metrics_ledger: IdempotencyLedger[None] = IdempotencyLedger()
        self._artifacts_ledger: IdempotencyLedger[None] = IdempotencyLedger()
        self._finalize_ledger: IdempotencyLedger[None] = IdempotencyLedger()

    async def create_run(self, manifest: RunManifest, *, ctx: OperationContext) -> TrackerRunRef:
        """Create or replay an explicitly supplied tracker reference for one manifest."""
        operation = "tracker.create_run"
        payload = {"manifest": manifest}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        async def execute() -> TrackerRunRef:
            def default() -> TrackerRunRef:
                try:
                    return self._runs[manifest.run_id]
                except KeyError as exc:
                    raise TrackerError(
                        make_failure(
                            code="TRACKER_RUN_REFERENCE_NOT_FOUND",
                            category="NOT_FOUND",
                            message="No deterministic tracker run reference is configured.",
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
                    TrackerRunRef,
                    operation=operation,
                    ctx=ctx,
                    error_type=TrackerError,
                ),
                error_type=TrackerError,
            )

        return await self._create_ledger.replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )

    async def log_metrics(
        self,
        run: TrackerRunRef,
        metrics: list[MetricPoint],
        *,
        ctx: OperationContext,
    ) -> None:
        """Append one explicitly scripted metric batch exactly once per idempotency key."""
        operation = "tracker.log_metrics"
        payload = {"run": run, "metrics": metrics}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        async def execute() -> None:
            def default() -> None:
                self.metrics.setdefault(run.tracker_run_id, []).extend(metrics)

            return await self._consume(
                operation=operation,
                payload=payload,
                ctx=ctx,
                default=default,
                validate_return=lambda value: require_scripted_none(
                    value,
                    operation=operation,
                    ctx=ctx,
                    error_type=TrackerError,
                ),
                error_type=TrackerError,
            )

        await self._metrics_ledger.replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )

    async def log_artifact_refs(
        self,
        run: TrackerRunRef,
        artifacts: list[ArtifactRef],
        *,
        ctx: OperationContext,
    ) -> None:
        """Attach an explicitly scripted immutable artifact batch exactly once."""
        operation = "tracker.log_artifact_refs"
        payload = {"run": run, "artifacts": artifacts}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        async def execute() -> None:
            def default() -> None:
                self.artifacts.setdefault(run.tracker_run_id, []).extend(artifacts)

            return await self._consume(
                operation=operation,
                payload=payload,
                ctx=ctx,
                default=default,
                validate_return=lambda value: require_scripted_none(
                    value,
                    operation=operation,
                    ctx=ctx,
                    error_type=TrackerError,
                ),
                error_type=TrackerError,
            )

        await self._artifacts_ledger.replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )

    async def finalize(
        self,
        run: TrackerRunRef,
        status: RunStatus,
        *,
        ctx: OperationContext,
    ) -> None:
        """Finalize a tracker record through an explicit idempotent fake side effect."""
        operation = "tracker.finalize"
        payload = {"run": run, "status": status}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        async def execute() -> None:
            def default() -> None:
                self.final_statuses[run.tracker_run_id] = status

            return await self._consume(
                operation=operation,
                payload=payload,
                ctx=ctx,
                default=default,
                validate_return=lambda value: require_scripted_none(
                    value,
                    operation=operation,
                    ctx=ctx,
                    error_type=TrackerError,
                ),
                error_type=TrackerError,
            )

        await self._finalize_ledger.replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )


__all__ = [
    "InMemoryArtifactStore",
    "InMemoryAsyncReader",
    "InMemoryExperimentTracker",
    "ScriptedDatasetCatalog",
    "ScriptedExperimentExecutor",
    "ScriptedPatchWorkspace",
    "ScriptedValidationRunner",
]
