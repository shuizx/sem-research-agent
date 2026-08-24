"""Private-boundary regressions for private fake boundaries and concrete port contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeVar

import pytest

import tests.fakes as fake_exports
from tests.contract.base import assert_concrete_protocol_signatures
from tests.fakes import (
    CancelledStep,
    DelegateStep,
    FailureStep,
    IdempotencyLedger,
    InMemoryArtifactStore,
    InMemoryAsyncReader,
    InMemoryUnitOfWork,
    ReturnStep,
    ScriptedAuditRepository,
    ScriptedEntityRepository,
    ScriptedStructuredLLM,
    TimeoutStep,
)
from tests.fakes import script as script_module
from vision_research_ops.domain import (
    ResearchBudget,
    ResearchRequest,
    StructuredFailure,
    WorkflowStatus,
)
from vision_research_ops.ports import (
    ArtifactError,
    ArtifactStore,
    AuditEvent,
    CapabilityNotSupportedError,
    FakeScriptExhaustedError,
    IdempotencyConflictError,
    OperationCancelledError,
    OperationContext,
    OperationTimeoutError,
    PersistenceError,
    PortError,
    StructuredGenerationRequest,
    make_failure,
)

TIMESTAMP = "2026-08-09T00:00:00Z"


class _SignatureBase:
    """Module-scoped base result used by postponed-annotation contract regressions."""


class _SignatureChild(_SignatureBase):
    """Module-scoped subtype used to prove invariant containers remain strict."""


_T_ARBITRARY = TypeVar("_T_ARBITRARY")


class ObservableByteStream:
    """A deterministic stream with optional Event-controlled first-chunk blocking."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
        failure: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._started = started
        self._release = release
        self._failure = failure
        self._index = 0
        self.iterations = 0

    def __aiter__(self) -> ObservableByteStream:
        """Return the observable stream without consuming a byte chunk."""
        return self

    async def __anext__(self) -> bytes:
        """Yield one deterministic chunk, failure, or cancellation observation point."""
        self.iterations += 1
        if self._index == 0 and self._started is not None:
            self._started.set()
            assert self._release is not None
            await self._release.wait()
        if self._failure is not None:
            failure = self._failure
            self._failure = None
            raise failure
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class MaliciousItemsMapping(Mapping[str, object]):
    """Mapping whose ``items`` method proves canonicalization never leaks raw exceptions."""

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        raise RuntimeError("password=round3-malicious-mapping-secret")


def _request(*, title: str) -> ResearchRequest:
    """Create one valid entity for UoW facade lifecycle regressions."""
    return ResearchRequest(
        schema_version="1",
        request_id="round3_request",
        revision=1,
        title=title,
        research_question="Can stale fake repository facades be rejected?",
        dataset_id="round3_dataset",
        dataset_version="v1",
        query_spec={"schema_version": "1", "keywords": ["synthetic"]},
        budget=ResearchBudget(
            schema_version="1",
            max_provider_pages=1,
            max_provider_records=1,
            max_llm_calls=1,
            max_llm_tokens=1,
            max_cost_estimate=0.0,
            max_candidate_repositories=1,
            max_adaptation_attempts=1,
            max_workflow_walltime_seconds=1,
        ),
        requested_by="round3_actor",
        status=WorkflowStatus.PENDING,
        created_at=TIMESTAMP,
        updated_at=TIMESTAMP,
    )


def _audit_event(*, event_id: str) -> AuditEvent:
    """Create one small append-only event for facade lifecycle regressions."""
    return AuditEvent(
        schema_version="1",
        event_id=event_id,
        event_type="REQUEST_SAVED",
        occurred_at=TIMESTAMP,
        correlation_id="corr_1",
        workflow_id="wf_1",
        subject_type="research_request",
        subject_id="round3_request",
        subject_revision=1,
        payload={"action": "save"},
    )


def _structured_failure(*, ctx: OperationContext | None, message: str) -> StructuredFailure:
    """Build one controlled test failure with no raw error retention."""
    return make_failure(
        code="ROUND3_TEST_FAILURE",
        category="TEST_FAKE",
        message=message,
        retryable=False,
        ctx=ctx,
    )


@pytest.mark.contract
@pytest.mark.asyncio
async def test_redaction_covers_quoted_json_url_newlines_prefixes_and_call_records(
    make_context: Callable[..., OperationContext],
) -> None:
    """All public observations redact controlled credential syntaxes without suffix leakage."""
    long_secret = "prefix" + "a" * 128
    prefix_values = ["prefix" + "a" * length for length in range(1, 129)]
    message = (
        f'{{"password":"{long_secret}"}} '
        f'"authorization": "Bearer {long_secret}" '
        f"'api_key'='{long_secret}' "
        f"https://example.invalid/?token={long_secret}&page=1 "
        f"credential\n=\n{long_secret}"
    )
    failure = make_failure(
        code="ROUND3_TEST_REDACTION",
        category="TEST_FAKE",
        message=message,
        retryable=False,
        ctx=make_context(idempotency_key=None),
        details={"token": prefix_values, "nested": [{"PASSWORD": long_secret}]},
    )
    for value in prefix_values:
        assert value not in failure.message
        assert value not in repr(failure.details)
    assert "[REDACTED]" in failure.message
    assert failure.details["token"] == "[REDACTED]"
    assert failure.details["nested"][0]["PASSWORD"] == "[REDACTED]"

    for no_details_message in (
        f'{{"password":"{long_secret}"}}',
        f'"authorization": "Bearer {long_secret}"',
        f"'api_key'='{long_secret}'",
        f"API KEY: {long_secret}",
        f"access_token={long_secret}",
        f"https://example.invalid/?token={long_secret}&page=1",
        f"credential\n=\n{long_secret}",
    ):
        no_details = make_failure(
            code="ROUND3_TEST_NO_DETAILS",
            category="TEST_FAKE",
            message=no_details_message,
            retryable=False,
            ctx=make_context(idempotency_key=None),
        )
        assert long_secret not in no_details.message

    direct = PortError(
        StructuredFailure(
            schema_version="1",
            code="ROUND3_DIRECT_REDACTION",
            category="TEST_FAKE",
            message=f'{{"password":"{long_secret}"}}',
            message_hash="sha256:" + "a" * 64,
            retryable=False,
            details={},
        )
    )
    for public_value in (
        str(direct),
        repr(direct),
        direct.failure.message,
        repr(direct.failure.details),
    ):
        assert long_secret not in public_value

    malformed_details = make_failure(
        code="ROUND3_TEST_MALFORMED_DETAILS",
        category="TEST_FAKE",
        message="password=round3-malicious-mapping-secret",
        retryable=False,
        ctx=make_context(idempotency_key=None),
        details=MaliciousItemsMapping(),
    )
    assert "round3-malicious-mapping-secret" not in repr(malformed_details)

    fake = ScriptedStructuredLLM(
        script={"llm.generate": [FailureStep(failure)]},
    )
    request = StructuredGenerationRequest[StructuredFailure](
        schema_version="1",
        task_name="round3-redaction",
        prompt_template_id="round3/redaction",
        prompt_version="1",
        response_schema=StructuredFailure,
        facts={"headers": [{"authorization": f"Bearer {long_secret}"}]},
        budget_class="test",
    )
    with pytest.raises(PortError):
        await fake.generate(request, ctx=make_context(idempotency_key=None))
    assert long_secret not in repr(fake.calls[0])
    assert long_secret not in repr(fake.calls[0].payload)


@pytest.mark.contract
@pytest.mark.asyncio
async def test_ledger_keeps_snapshots_private_and_rejects_loop_bound_replay_results(
    make_context: Callable[..., OperationContext],
) -> None:
    """Task, Future, and coroutine results never enter public or cross-loop replay state."""
    assert "IdempotencyLedgerSnapshot" not in script_module.__all__
    assert "IdempotencyLedgerSnapshot" not in fake_exports.__all__
    ledger: IdempotencyLedger[object] = IdempotencyLedger()
    assert not hasattr(ledger, "snapshot")
    assert not hasattr(ledger, "restore")
    assert not hasattr(ledger, "clone")
    assert "fingerprint" not in repr(ledger).casefold()
    context = make_context(idempotency_key="round3-loop-bound")
    created_tasks: list[asyncio.Task[str]] = []

    async def value() -> str:
        return "not-replayable"

    async def returns_task() -> object:
        task = asyncio.create_task(value())
        created_tasks.append(task)
        return task

    async def returns_future() -> object:
        return asyncio.get_running_loop().create_future()

    async def returns_event() -> object:
        return asyncio.Event()

    async def returns_coroutine() -> object:
        return value()

    for execute in (returns_task, returns_future, returns_event, returns_coroutine):
        with pytest.raises(PortError) as raised:
            await ledger.replay_or_execute(
                operation="round3.loop-bound",
                idempotency_key="round3-loop-bound",
                payload={"request": "same"},
                ctx=context,
                execute=execute,
            )
        assert raised.value.failure.code == "FAKE_IDEMPOTENCY_RESULT_NOT_REPLAYABLE"
        assert ledger.has_replay_state("round3-loop-bound") is False
        assert ledger.effect_counts["round3.loop-bound"] == 0

    assert await created_tasks[0] == "not-replayable"

    async def replayable() -> object:
        return "safe"

    assert (
        await ledger.replay_or_execute(
            operation="round3.loop-bound",
            idempotency_key="round3-loop-bound",
            payload={"request": "same"},
            ctx=context,
            execute=replayable,
        )
        == "safe"
    )
    assert ledger.effect_counts["round3.loop-bound"] == 1


@pytest.mark.contract
@pytest.mark.asyncio
async def test_ledger_canonicalization_rejects_a_malicious_mapping_without_leaking_its_secret(
    make_context: Callable[..., OperationContext],
) -> None:
    """Ordinary Mapping failures become a safe typed validation error before any effect runs."""
    ledger: IdempotencyLedger[str] = IdempotencyLedger()

    async def forbidden_effect() -> str:
        raise AssertionError("malicious payload must fail before the side effect")

    with pytest.raises(PortError) as raised:
        await ledger.replay_or_execute(
            operation="round3.malicious-mapping",
            idempotency_key="round3-mapping-key",
            payload=MaliciousItemsMapping(),
            ctx=make_context(idempotency_key="round3-mapping-key"),
            execute=forbidden_effect,
        )
    assert raised.value.failure.code == "PORT_PAYLOAD_CANONICALIZATION_FAILED"
    assert "round3-malicious-mapping-secret" not in repr(raised.value)


@pytest.mark.contract
@pytest.mark.asyncio
async def test_unit_of_work_old_entity_and_audit_facades_cannot_be_reactivated(
    make_context: Callable[..., OperationContext],
) -> None:
    """A shared backing creates fresh session facades while old UoW references stay terminal."""
    original = _request(title="original")
    first_update = _request(title="first")
    second_update = _request(title="second")
    repository = ScriptedEntityRepository(
        ResearchRequest,
        entities={"round3_request": original},
        revisions={"round3_request": 1},
        script={"persistence.save": [DelegateStep(), DelegateStep()]},
    )
    audits = ScriptedAuditRepository(
        script={"persistence.audit.append": [DelegateStep(), DelegateStep()]}
    )
    first = InMemoryUnitOfWork(
        research_requests=repository,
        audits=audits,
        lifecycle_script={"uow.enter": [DelegateStep()], "uow.commit": [DelegateStep()]},
    )
    async with first as active:
        old_repository = active.research_requests
        old_audits = active.audits
        await old_repository.save(
            "round3_request",
            first_update,
            expected_revision=1,
            ctx=make_context(idempotency_key="round3-first-entity"),
        )
        await old_audits.append(
            _audit_event(event_id="round3-event-1"),
            ctx=make_context(idempotency_key="round3-first-audit"),
        )
        await active.commit()

    second = InMemoryUnitOfWork(
        research_requests=repository,
        audits=audits,
        lifecycle_script={"uow.enter": [DelegateStep()], "uow.commit": [DelegateStep()]},
    )
    async with second as active:
        assert active.research_requests is not old_repository
        assert active.audits is not old_audits
        with pytest.raises(PersistenceError) as stale_entity:
            await old_repository.save(
                "round3_request",
                second_update,
                expected_revision=2,
                ctx=make_context(idempotency_key="round3-stale-entity"),
            )
        with pytest.raises(PersistenceError) as stale_audit:
            await old_audits.append(
                _audit_event(event_id="round3-stale-event"),
                ctx=make_context(idempotency_key="round3-stale-audit"),
            )
        assert stale_entity.value.failure.code == "PERSISTENCE_UNIT_OF_WORK_NOT_ACTIVE"
        assert stale_audit.value.failure.code == "PERSISTENCE_UNIT_OF_WORK_NOT_ACTIVE"
        await active.research_requests.save(
            "round3_request",
            second_update,
            expected_revision=2,
            ctx=make_context(idempotency_key="round3-second-entity"),
        )
        await active.audits.append(
            _audit_event(event_id="round3-event-2"),
            ctx=make_context(idempotency_key="round3-second-audit"),
        )
        await active.commit()

    assert repository.backing_entity("round3_request") == second_update
    assert repository.backing_revision_of("round3_request") == 3
    assert [event.event_id for event in audits.backing_events] == [
        "round3-event-1",
        "round3-event-2",
    ]


@pytest.mark.contract
@pytest.mark.asyncio
async def test_artifact_reservation_blocks_another_new_key_before_stream_consumption(
    make_context: Callable[..., OperationContext],
    make_descriptor: Callable[..., object],
) -> None:
    """One DelegateStep is atomically reserved, so a competing new key reads zero chunks."""
    store = InMemoryArtifactStore(script={"artifact.put_bytes": [DelegateStep()]})
    started = asyncio.Event()
    release = asyncio.Event()
    owner_stream = ObservableByteStream([b"owner"], started=started, release=release)
    owner = asyncio.create_task(
        store.put_bytes(
            owner_stream,
            make_descriptor(artifact_id="round3-owner"),
            expected_sha256=None,
            ctx=make_context(idempotency_key="round3-owner-key"),
        )
    )
    await started.wait()
    loser_stream = ObservableByteStream([b"loser"])
    with pytest.raises(FakeScriptExhaustedError):
        await store.put_bytes(
            loser_stream,
            make_descriptor(artifact_id="round3-loser"),
            expected_sha256=None,
            ctx=make_context(idempotency_key="round3-loser-key"),
        )
    assert loser_stream.iterations == 0
    release.set()
    assert (await owner).artifact_id == "round3-owner"
    assert store.put_effect_count == 1


@pytest.mark.contract
@pytest.mark.asyncio
async def test_artifact_same_key_reservation_replays_or_conflicts_without_extra_steps(
    make_context: Callable[..., OperationContext],
    make_descriptor: Callable[..., object],
) -> None:
    """Concurrent same-key content shares one step; different content has no second effect."""
    replay_store = InMemoryArtifactStore(script={"artifact.put_bytes": [DelegateStep()]})
    started = asyncio.Event()
    release = asyncio.Event()
    first = asyncio.create_task(
        replay_store.put_bytes(
            ObservableByteStream([b"same"], started=started, release=release),
            make_descriptor(artifact_id="round3-same"),
            expected_sha256=None,
            ctx=make_context(idempotency_key="round3-same-key"),
        )
    )
    await started.wait()
    second = await replay_store.put_bytes(
        ObservableByteStream([b"same"]),
        make_descriptor(artifact_id="round3-same"),
        expected_sha256=None,
        ctx=make_context(idempotency_key="round3-same-key"),
    )
    release.set()
    assert await first == second
    assert replay_store.put_effect_count == 1

    conflict_store = InMemoryArtifactStore(script={"artifact.put_bytes": [DelegateStep()]})
    conflict_started = asyncio.Event()
    conflict_release = asyncio.Event()
    delayed = asyncio.create_task(
        conflict_store.put_bytes(
            ObservableByteStream(
                [b"first"],
                started=conflict_started,
                release=conflict_release,
            ),
            make_descriptor(artifact_id="round3-conflict"),
            expected_sha256=None,
            ctx=make_context(idempotency_key="round3-conflict-key"),
        )
    )
    await conflict_started.wait()
    winner = await conflict_store.put_bytes(
        ObservableByteStream([b"second"]),
        make_descriptor(artifact_id="round3-conflict"),
        expected_sha256=None,
        ctx=make_context(idempotency_key="round3-conflict-key"),
    )
    conflict_release.set()
    with pytest.raises(IdempotencyConflictError):
        await delayed
    assert winner.artifact_id == "round3-conflict"
    assert conflict_store.put_effect_count == 1


@pytest.mark.contract
@pytest.mark.asyncio
async def test_artifact_releases_pre_stream_reservations_after_error_or_cancellation(
    make_context: Callable[..., OperationContext],
    make_descriptor: Callable[..., object],
) -> None:
    """A stream failure or cancellation restores an unstarted step for deterministic retry."""
    error_store = InMemoryArtifactStore(script={"artifact.put_bytes": [DelegateStep()]})
    raw_secret = "round3-stream-token"
    with pytest.raises(ArtifactError) as read_error:
        await error_store.put_bytes(
            ObservableByteStream([], failure=RuntimeError(f"token={raw_secret}")),
            make_descriptor(artifact_id="round3-error"),
            expected_sha256=None,
            ctx=make_context(idempotency_key="round3-error-key"),
        )
    assert read_error.value.failure.code == "ARTIFACT_STREAM_READ_FAILED"
    assert raw_secret not in repr(read_error.value)
    assert (
        await error_store.put_bytes(
            b"recovered",
            make_descriptor(artifact_id="round3-error"),
            expected_sha256=None,
            ctx=make_context(idempotency_key="round3-error-key"),
        )
    ).artifact_id == "round3-error"

    cancelled_store = InMemoryArtifactStore(script={"artifact.put_bytes": [DelegateStep()]})
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.create_task(
        cancelled_store.put_bytes(
            ObservableByteStream([b"cancel"], started=started, release=release),
            make_descriptor(artifact_id="round3-cancel"),
            expected_sha256=None,
            ctx=make_context(idempotency_key="round3-cancel-key"),
        )
    )
    await started.wait()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert (
        await cancelled_store.put_bytes(
            b"recovered",
            make_descriptor(artifact_id="round3-cancel"),
            expected_sha256=None,
            ctx=make_context(idempotency_key="round3-cancel-key"),
        )
    ).artifact_id == "round3-cancel"


@pytest.mark.contract
@pytest.mark.asyncio
async def test_artifact_rejects_malicious_stream_shape_and_terminal_steps_before_reading(
    make_context: Callable[..., OperationContext],
    make_descriptor: Callable[..., object],
) -> None:
    """Property errors and terminal scripts are typed, redacted, and consume zero stream chunks."""
    raw_secret = "round3-shape-token"

    class BadAiter:
        @property
        def __aiter__(self) -> object:
            raise RuntimeError(f"authorization=Bearer {raw_secret}")

        @property
        def __anext__(self) -> object:
            raise RuntimeError(f"password={raw_secret}")

    class BadAnext:
        def __aiter__(self) -> BadAnext:
            return self

        @property
        def __anext__(self) -> object:
            raise RuntimeError(f"credential={raw_secret}")

    for stream in (BadAiter(), BadAnext()):
        store = InMemoryArtifactStore(script={"artifact.put_bytes": [DelegateStep()]})
        with pytest.raises(ArtifactError) as malformed:
            await store.put_bytes(
                stream,
                make_descriptor(artifact_id="round3-malformed"),
                expected_sha256=None,
                ctx=make_context(idempotency_key="round3-malformed-key"),
            )
        assert malformed.value.failure.code == "ARTIFACT_STREAM_TYPE_INVALID"
        assert raw_secret not in repr(malformed.value)

    preflight_cases = (
        (
            InMemoryArtifactStore(),
            make_context(idempotency_key="round3-exhausted-key"),
            FakeScriptExhaustedError,
        ),
        (
            InMemoryArtifactStore(
                script={"artifact.put_bytes": [DelegateStep()]},
                supported_operations=("artifact.open",),
            ),
            make_context(idempotency_key="round3-unsupported-key"),
            CapabilityNotSupportedError,
        ),
        (
            InMemoryArtifactStore(script={"artifact.put_bytes": [DelegateStep()]}),
            make_context(
                idempotency_key="round3-expired-key",
                deadline_at="2025-01-01T00:00:00Z",
            ),
            OperationTimeoutError,
        ),
    )
    for store, context, expected_error in preflight_cases:
        stream = ObservableByteStream([b"must-not-read"])
        with pytest.raises(expected_error):
            await store.put_bytes(
                stream,
                make_descriptor(artifact_id="round3-preflight"),
                expected_sha256=None,
                ctx=context,
            )
        assert stream.iterations == 0

    terminal_steps: tuple[object, ...] = (
        FailureStep(
            _structured_failure(
                ctx=None,
                message=f"token={raw_secret}",
            )
        ),
        TimeoutStep(),
        CancelledStep(),
    )
    expected_errors = (ArtifactError, OperationTimeoutError, OperationCancelledError)
    for step, expected_error in zip(terminal_steps, expected_errors, strict=True):
        stream = ObservableByteStream([b"must-not-read"])
        store = InMemoryArtifactStore(script={"artifact.put_bytes": [step]})
        with pytest.raises(expected_error) as raised:
            await store.put_bytes(
                stream,
                make_descriptor(artifact_id="round3-terminal"),
                expected_sha256=None,
                ctx=make_context(idempotency_key="round3-terminal-key"),
            )
        assert stream.iterations == 0
        assert raw_secret not in repr(raised.value)


@pytest.mark.contract
@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["not-bytes", None])
async def test_validating_reader_rejects_non_bytes_read_and_chunk_values(
    value: object,
    make_context: Callable[..., OperationContext],
) -> None:
    """The proxy validates both ``read`` and iterator chunks instead of only method shapes."""

    class Reader:
        async def read(self) -> object:
            return value

        def __aiter__(self) -> Reader:
            return self

        async def __anext__(self) -> object:
            return value

    read_store = InMemoryArtifactStore(script={"artifact.open": [ReturnStep(Reader())]})
    reader = await read_store.open("unused", ctx=make_context(idempotency_key=None))
    with pytest.raises(ArtifactError) as read_error:
        await reader.read()
    assert read_error.value.failure.code == "ARTIFACT_READER_BYTES_INVALID"

    chunk_store = InMemoryArtifactStore(script={"artifact.open": [ReturnStep(Reader())]})
    chunk_reader = await chunk_store.open("unused", ctx=make_context(idempotency_key=None))
    with pytest.raises(ArtifactError) as chunk_error:
        await chunk_reader.__anext__()
    assert chunk_error.value.failure.code == "ARTIFACT_READER_BYTES_INVALID"


@pytest.mark.contract
@pytest.mark.asyncio
async def test_validating_reader_caches_iterator_and_wraps_property_or_runtime_errors(
    make_context: Callable[..., OperationContext],
) -> None:
    """Raw reader changes, property failures, and exceptions cannot bypass the proxy boundary."""
    raw_secret = "round3-reader-token"

    class FirstIterator:
        def __init__(self) -> None:
            self._used = False

        async def __anext__(self) -> bytes:
            if self._used:
                raise StopAsyncIteration
            self._used = True
            return b"first"

    class SwitchingReader:
        def __init__(self) -> None:
            self.aiter_calls = 0
            self._first = FirstIterator()

        async def read(self) -> bytes:
            return b"read"

        def __aiter__(self) -> FirstIterator:
            self.aiter_calls += 1
            if self.aiter_calls == 1:
                return self._first
            raise RuntimeError(f"secret={raw_secret}")

        async def __anext__(self) -> bytes:
            raise StopAsyncIteration

    switching = SwitchingReader()
    store = InMemoryArtifactStore(script={"artifact.open": [ReturnStep(switching)]})
    reader = await store.open("unused", ctx=make_context(idempotency_key=None))
    assert await reader.read() == b"read"
    assert [chunk async for chunk in reader] == [b"first"]
    assert switching.aiter_calls == 1

    class PropertyReader:
        @property
        def read(self) -> object:
            raise RuntimeError(f"token={raw_secret}")

        def __aiter__(self) -> PropertyReader:
            return self

        async def __anext__(self) -> bytes:
            raise StopAsyncIteration

    malformed_store = InMemoryArtifactStore(
        script={"artifact.open": [ReturnStep(PropertyReader())]}
    )
    with pytest.raises(ArtifactError) as malformed:
        await malformed_store.open("unused", ctx=make_context(idempotency_key=None))
    assert raw_secret not in repr(malformed.value)

    class AiterPropertyReader:
        async def read(self) -> bytes:
            return b"unused"

        @property
        def __aiter__(self) -> object:
            raise RuntimeError(f"authorization=Bearer {raw_secret}")

        async def __anext__(self) -> bytes:
            raise StopAsyncIteration

    class AnextPropertyReader:
        async def read(self) -> bytes:
            return b"unused"

        def __aiter__(self) -> AnextPropertyReader:
            return self

        @property
        def __anext__(self) -> object:
            raise RuntimeError(f"credential={raw_secret}")

    for malformed_reader in (AiterPropertyReader(), AnextPropertyReader()):
        malformed_store = InMemoryArtifactStore(
            script={"artifact.open": [ReturnStep(malformed_reader)]}
        )
        with pytest.raises(ArtifactError) as malformed:
            await malformed_store.open("unused", ctx=make_context(idempotency_key=None))
        assert raw_secret not in repr(malformed.value)

    class RuntimeReader:
        async def read(self) -> bytes:
            raise RuntimeError(f"authorization=Bearer {raw_secret}")

        def __aiter__(self) -> RuntimeReader:
            return self

        async def __anext__(self) -> bytes:
            raise RuntimeError(f"password={raw_secret}")

    runtime_store = InMemoryArtifactStore(script={"artifact.open": [ReturnStep(RuntimeReader())]})
    runtime_reader = await runtime_store.open("unused", ctx=make_context(idempotency_key=None))
    with pytest.raises(ArtifactError) as read_error:
        await runtime_reader.read()
    with pytest.raises(ArtifactError) as next_error:
        await runtime_reader.__anext__()
    assert raw_secret not in repr(read_error.value)
    assert raw_secret not in repr(next_error.value)


@pytest.mark.contract
@pytest.mark.asyncio
async def test_validating_reader_preserves_cancellation_stop_iteration_and_legal_bytes(
    make_context: Callable[..., OperationContext],
) -> None:
    """Cancellation and StopAsyncIteration retain their semantics while legal bytes still work."""
    started = asyncio.Event()
    release = asyncio.Event()

    class CancelReader:
        async def read(self) -> bytes:
            started.set()
            await release.wait()
            return b"late"

        def __aiter__(self) -> CancelReader:
            return self

        async def __anext__(self) -> bytes:
            raise StopAsyncIteration

    cancelled_store = InMemoryArtifactStore(script={"artifact.open": [ReturnStep(CancelReader())]})
    cancelled_reader = await cancelled_store.open("unused", ctx=make_context(idempotency_key=None))
    pending = asyncio.create_task(cancelled_reader.read())
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    legal_store = InMemoryArtifactStore(
        script={"artifact.open": [ReturnStep(InMemoryAsyncReader(b"legal"))]}
    )
    legal_reader = await legal_store.open("unused", ctx=make_context(idempotency_key=None))
    assert await legal_reader.read() == b"legal"
    assert [chunk async for chunk in legal_reader] == [b"legal"]
    with pytest.raises(StopAsyncIteration):
        await legal_reader.__anext__()


@pytest.mark.contract
def test_return_annotation_checks_reject_any_typevars_invariant_containers_and_wrong_unions() -> (
    None
):
    """Concrete signature checks remain strict without TypeVar or container variance loopholes."""

    class ReturnProtocol(Protocol):
        async def result(self, *, ctx: object) -> _SignatureBase:
            """Return a base result."""

    class BadReturnPaper:
        async def result(self, *, ctx: object) -> int:
            del ctx
            return 1

    class AnyReturn:
        async def result(self, *, ctx: object) -> Any:
            del ctx
            return None

    class TypeVarReturn:
        async def result(self, *, ctx: object) -> _T_ARBITRARY:
            del ctx
            raise AssertionError("annotation test does not invoke fake methods")

    class ListProtocol(Protocol):
        async def result(self, *, ctx: object) -> list[_SignatureBase]:
            """Return invariant list content."""

    class ListChildReturn:
        async def result(self, *, ctx: object) -> list[_SignatureChild]:
            del ctx
            return []

    class DictProtocol(Protocol):
        async def result(self, *, ctx: object) -> dict[str, _SignatureBase]:
            """Return invariant dictionary content."""

    class DictChildReturn:
        async def result(self, *, ctx: object) -> dict[str, _SignatureChild]:
            del ctx
            return {}

    class OptionalProtocol(Protocol):
        async def result(self, *, ctx: object) -> _SignatureBase | None:
            """Return the declared Optional type."""

    class WrongUnionReturn:
        async def result(self, *, ctx: object) -> _SignatureBase | str:
            del ctx
            return "wrong"

    class MissingReturn:
        async def result(self, *, ctx: object):
            del ctx
            return _SignatureBase()

    cases: tuple[tuple[type[object], object], ...] = (
        (ReturnProtocol, BadReturnPaper()),
        (ReturnProtocol, AnyReturn()),
        (ReturnProtocol, TypeVarReturn()),
        (ListProtocol, ListChildReturn()),
        (DictProtocol, DictChildReturn()),
        (OptionalProtocol, WrongUnionReturn()),
        (ReturnProtocol, MissingReturn()),
    )
    for protocol, fake in cases:
        with pytest.raises(AssertionError):
            assert_concrete_protocol_signatures(protocol, fake)


@pytest.mark.contract
def test_artifact_store_protocol_still_has_a_concrete_auto_discovered_fake() -> None:
    """A smoke assertion keeps the reader proxy implementation under the Protocol suite."""
    assert isinstance(InMemoryArtifactStore(), ArtifactStore)
