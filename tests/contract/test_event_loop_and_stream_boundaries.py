"""Event-loop and stream boundary regressions for deterministic ports."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import fields
from typing import Literal, cast

import pytest
from pydantic import BaseModel

from tests.contract.base import assert_concrete_protocol_signatures
from tests.fakes import (
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
)
from vision_research_ops.domain import (
    ResearchBudget,
    ResearchRequest,
    StructuredFailure,
    WorkflowStatus,
)
from vision_research_ops.ports import (
    ArtifactDescriptor,
    ArtifactError,
    AuditEvent,
    IdempotencyConflictError,
    OperationContext,
    PersistenceError,
    PortError,
    StructuredGenerationRequest,
    StructuredLLM,
    UnitOfWork,
    make_failure,
)
from vision_research_ops.ports.base import AsyncBinaryReader

TIMESTAMP = "2026-08-09T00:00:00Z"


class TextProposal(BaseModel):
    """Small structured result model used only by redaction call-record regressions."""

    schema_version: Literal["1"]
    value: str


class ObservableStream:
    """An asynchronous stream whose observed iteration count proves preflight behavior."""

    def __init__(self, chunks: list[bytes], *, failure: BaseException | None = None) -> None:
        self._chunks = chunks
        self._failure = failure
        self._index = 0
        self.iterations = 0

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Return this test stream without consuming an item."""
        return self

    async def __anext__(self) -> bytes:
        """Yield the next byte chunk or one explicit test failure."""
        self.iterations += 1
        if self._failure is not None:
            failure = self._failure
            self._failure = None
            raise failure
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


def _request(*, title: str) -> ResearchRequest:
    """Create one valid small entity for UnitOfWork rollback regressions."""
    return ResearchRequest(
        schema_version="1",
        request_id="round2_request",
        revision=1,
        title=title,
        research_question="Can a synthetic transaction be restored?",
        dataset_id="round2_dataset",
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
        requested_by="round2_actor",
        status=WorkflowStatus.PENDING,
        created_at=TIMESTAMP,
        updated_at=TIMESTAMP,
    )


def _audit_event() -> AuditEvent:
    """Create an append-only event used to prove staged audit behavior."""
    return AuditEvent(
        schema_version="1",
        event_id="round2_event",
        event_type="REQUEST_SAVED",
        occurred_at=TIMESTAMP,
        correlation_id="corr_1",
        workflow_id="wf_1",
        subject_type="research_request",
        subject_id="round2_request",
        subject_revision=1,
        payload={"action": "save"},
    )


@pytest.mark.contract
def test_ledger_reuses_one_instance_across_sequential_event_loops(
    make_context: Callable[..., OperationContext],
) -> None:
    """A completed first-loop contention leaves no loop-bound task, lock, or event behind."""
    ledger: IdempotencyLedger[str] = IdempotencyLedger()
    executions = 0
    context = make_context(idempotency_key="sequential-loop-key")

    async def first_loop() -> list[str]:
        nonlocal executions
        started = asyncio.Event()
        release = asyncio.Event()

        async def execute() -> str:
            nonlocal executions
            executions += 1
            started.set()
            await release.wait()
            return "first-loop-result"

        first = asyncio.create_task(
            ledger.replay_or_execute(
                operation="round2.loop",
                idempotency_key="sequential-loop-key",
                payload={"request": "same"},
                ctx=context,
                execute=execute,
            )
        )
        await started.wait()
        replay = asyncio.create_task(
            ledger.replay_or_execute(
                operation="round2.loop",
                idempotency_key="sequential-loop-key",
                payload={"request": "same"},
                ctx=context,
                execute=execute,
            )
        )
        await ledger.wait_for_in_flight_join()
        release.set()
        return await asyncio.gather(first, replay)

    async def second_loop() -> str:
        async def forbidden_effect() -> str:
            raise AssertionError("completed idempotency replay must not execute again")

        return await ledger.replay_or_execute(
            operation="round2.loop",
            idempotency_key="sequential-loop-key",
            payload={"request": "same"},
            ctx=context,
            execute=forbidden_effect,
        )

    assert asyncio.run(first_loop()) == ["first-loop-result", "first-loop-result"]
    assert asyncio.run(second_loop()) == "first-loop-result"
    assert executions == 1
    assert ledger.effect_counts["round2.loop"] == 1
    assert ledger.in_flight_count == 0


@pytest.mark.contract
def test_ledger_fails_closed_for_an_active_different_event_loop(
    make_context: Callable[..., OperationContext],
) -> None:
    """A second live loop never awaits a foreign task or starts a second side effect."""
    ledger: IdempotencyLedger[str] = IdempotencyLedger()
    started = threading.Event()
    release = threading.Event()
    owner_result: dict[str, object] = {}
    executions = 0
    context = make_context(idempotency_key="cross-loop-key")

    def owner() -> None:
        async def run() -> str:
            nonlocal executions

            async def execute() -> str:
                nonlocal executions
                executions += 1
                started.set()
                await asyncio.to_thread(release.wait)
                return "owner-result"

            return await ledger.replay_or_execute(
                operation="round2.cross-loop",
                idempotency_key="cross-loop-key",
                payload={"request": "same"},
                ctx=context,
                execute=execute,
            )

        try:
            owner_result["value"] = asyncio.run(run())
        except BaseException as error:  # pragma: no cover - failure evidence is asserted below
            owner_result["error"] = error

    thread = threading.Thread(target=owner)
    thread.start()
    try:
        assert started.wait(timeout=2)

        async def foreign_loop() -> None:
            async def forbidden_effect() -> str:
                raise AssertionError("foreign event loop must not execute the side effect")

            with pytest.raises(PortError) as raised:
                await ledger.replay_or_execute(
                    operation="round2.cross-loop",
                    idempotency_key="cross-loop-key",
                    payload={"request": "same"},
                    ctx=context,
                    execute=forbidden_effect,
                )
            assert raised.value.failure.code == "PORT_IDEMPOTENCY_CROSS_LOOP_UNSUPPORTED"
            assert "different event loop" not in str(raised.value).casefold()

        asyncio.run(foreign_loop())
        assert executions == 1
    finally:
        release.set()
        thread.join(timeout=2)
    assert thread.is_alive() is False
    assert owner_result == {"value": "owner-result"}
    assert ledger.effect_counts["round2.cross-loop"] == 1


@pytest.mark.contract
@pytest.mark.asyncio
async def test_ledger_consumes_shared_task_failure_after_all_waiters_cancel(
    make_context: Callable[..., OperationContext],
) -> None:
    """A background failure after shielded waiters cancel cannot reach the loop handler."""
    raw_secret = "round2-unretrieved-token"
    ledger: IdempotencyLedger[str] = IdempotencyLedger()
    started = asyncio.Event()
    release = asyncio.Event()
    loop = asyncio.get_running_loop()
    observed: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: observed.append(context))
    context = make_context(idempotency_key="cancelled-waiters-key")
    executions = 0

    async def execute() -> str:
        nonlocal executions
        executions += 1
        started.set()
        await release.wait()
        raise RuntimeError(f"token={raw_secret}")

    try:
        first = asyncio.create_task(
            ledger.replay_or_execute(
                operation="round2.cancelled-waiters",
                idempotency_key="cancelled-waiters-key",
                payload={"request": "same"},
                ctx=context,
                execute=execute,
            )
        )
        await started.wait()
        second = asyncio.create_task(
            ledger.replay_or_execute(
                operation="round2.cancelled-waiters",
                idempotency_key="cancelled-waiters-key",
                payload={"request": "same"},
                ctx=context,
                execute=execute,
            )
        )
        await ledger.wait_for_in_flight_join()
        first.cancel()
        second.cancel()
        results = await asyncio.gather(first, second, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        release.set()
        await ledger.wait_for_idle()
        assert ledger.in_flight_count == 0
        assert executions == 1
        assert observed == []
        assert raw_secret not in repr(observed)
    finally:
        loop.set_exception_handler(previous_handler)


@pytest.mark.contract
@pytest.mark.asyncio
async def test_redaction_sanitizes_pair_sequences_messages_and_call_records(
    make_context: Callable[..., OperationContext],
) -> None:
    """Every public failure and call history removes nested credential values consistently."""
    raw_secret = "round2-raw-password"
    nested_details = {
        "headers": [("Authorization", f"Bearer {raw_secret}")],
        "mixed": ({"Api-Key": raw_secret}, [("PASSWORD", raw_secret)]),
        "credential_value": raw_secret,
    }
    message = (
        f"password={raw_secret}; Authorization: Bearer {raw_secret}; "
        f"token {raw_secret}; api-key={raw_secret}"
    )
    failure = make_failure(
        code="ROUND2_TEST_REDACTION",
        category="TEST_FAKE",
        message=message,
        retryable=False,
        ctx=make_context(idempotency_key=None),
        details=nested_details,
    )
    assert raw_secret not in failure.message
    assert raw_secret not in repr(failure.details)
    assert failure.details["headers"][0][1] == "[REDACTED]"
    assert failure.details["mixed"][0]["Api-Key"] == "[REDACTED]"
    assert failure.details["mixed"][1][0][1] == "[REDACTED]"

    unsafe = StructuredFailure(
        schema_version="1",
        code="ROUND2_UNSAFE_FAILURE",
        category="TEST_FAKE",
        message=f"Authorization: Bearer {raw_secret}",
        message_hash="sha256:" + "b" * 64,
        retryable=False,
        details={"token": raw_secret},
    )
    wrapped = ArtifactError(unsafe)
    assert raw_secret not in wrapped.failure.message
    assert raw_secret not in repr(wrapped)
    assert raw_secret not in repr(wrapped.failure.details)

    request = StructuredGenerationRequest[TextProposal](
        schema_version="1",
        task_name="redaction-call-record",
        prompt_template_id="redaction/v1",
        prompt_version="1",
        response_schema=TextProposal,
        facts={"headers": [["Authorization", f"Bearer {raw_secret}"]]},
        budget_class="test",
    )
    fake = ScriptedStructuredLLM(
        script={"llm.generate": [FailureStep(unsafe)]},
    )
    with pytest.raises(PortError):
        await fake.generate(request, ctx=make_context(idempotency_key=None))
    record = fake.calls[0]
    assert raw_secret not in repr(record)
    assert "payload_fingerprint" not in {field.name for field in fields(record)}
    assert "digest" not in repr(record).casefold()


@pytest.mark.contract
@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), object()])
async def test_ledger_rejects_noncanonical_payloads_without_public_fingerprints(
    value: object,
    make_context: Callable[..., OperationContext],
) -> None:
    """Unsafe canonical JSON cannot become a replay key or leak through the error boundary."""
    ledger: IdempotencyLedger[str] = IdempotencyLedger()

    async def forbidden_effect() -> str:
        raise AssertionError("invalid payload must fail before the side effect")

    with pytest.raises(PortError) as raised:
        await ledger.replay_or_execute(
            operation="round2.canonical",
            idempotency_key="canonical-key",
            payload={"nested": [{"value": value}]},
            ctx=make_context(idempotency_key="canonical-key"),
            execute=forbidden_effect,
        )
    assert raised.value.failure.code == "PORT_PAYLOAD_CANONICALIZATION_FAILED"
    assert "fingerprint" not in repr(raised.value).casefold()


@pytest.mark.contract
@pytest.mark.asyncio
async def test_secret_only_idempotency_change_conflicts_without_exposing_a_digest(
    make_context: Callable[..., OperationContext],
) -> None:
    """Comparison preserves secret-value differences while public observations remain redacted."""
    ledger: IdempotencyLedger[str] = IdempotencyLedger()
    context = make_context(idempotency_key="secret-conflict-key")

    async def effect() -> str:
        return "recorded"

    assert (
        await ledger.replay_or_execute(
            operation="round2.secret-conflict",
            idempotency_key="secret-conflict-key",
            payload={"headers": [("Authorization", "Bearer one-secret")]},
            ctx=context,
            execute=effect,
        )
        == "recorded"
    )
    with pytest.raises(IdempotencyConflictError):
        await ledger.replay_or_execute(
            operation="round2.secret-conflict",
            idempotency_key="secret-conflict-key",
            payload={"headers": [("Authorization", "Bearer two-secret")]},
            ctx=context,
            execute=effect,
        )


@pytest.mark.contract
@pytest.mark.asyncio
async def test_unit_of_work_closes_writes_after_commit_and_cannot_reenter(
    make_context: Callable[..., OperationContext],
) -> None:
    """Commit publishes once and prevents both staged writes and reuse of the same UoW."""
    original = _request(title="original")
    committed = _request(title="committed")
    unexpected = _request(title="unexpected")
    unentered_repository = ScriptedEntityRepository(
        ResearchRequest,
        entities={"round2_request": original},
        revisions={"round2_request": 1},
        script={"persistence.save": [DelegateStep()]},
    )
    with pytest.raises(PersistenceError) as unentered_write:
        await unentered_repository.save(
            "round2_request",
            committed,
            expected_revision=1,
            ctx=make_context(idempotency_key="unentered-write-key"),
        )
    assert unentered_write.value.failure.code == "PERSISTENCE_UNIT_OF_WORK_NOT_ACTIVE"
    repository = ScriptedEntityRepository(
        ResearchRequest,
        entities={"round2_request": original},
        revisions={"round2_request": 1},
        script={"persistence.save": [DelegateStep()]},
    )
    audits = ScriptedAuditRepository(script={"persistence.audit.append": [DelegateStep()]})
    uow = InMemoryUnitOfWork(
        research_requests=repository,
        audits=audits,
        context=make_context(idempotency_key="commit-key"),
        lifecycle_script={"uow.enter": [DelegateStep()], "uow.commit": [DelegateStep()]},
    )
    async with uow as active:
        await active.research_requests.save(
            "round2_request",
            committed,
            expected_revision=1,
            ctx=make_context(idempotency_key="commit-key"),
        )
        await active.audits.append(
            _audit_event(),
            ctx=make_context(idempotency_key="commit-audit-key"),
        )
        await active.commit()
        with pytest.raises(PersistenceError):
            await active.research_requests.save(
                "round2_request",
                unexpected,
                expected_revision=2,
                ctx=make_context(idempotency_key="post-commit-write"),
            )
        with pytest.raises(PersistenceError):
            await active.rollback()
    assert repository.backing_entity("round2_request") == committed
    assert repository.backing_revision_of("round2_request") == 2
    assert repository.backing_effect_count() == 1
    assert audits.backing_events == [_audit_event()]
    assert uow.lifecycle_state == "CLOSED"
    with pytest.raises(PersistenceError) as reentered:
        await uow.__aenter__()
    assert reentered.value.failure.code == "PERSISTENCE_UNIT_OF_WORK_REUSED"


@pytest.mark.contract
@pytest.mark.asyncio
async def test_unit_of_work_commit_and_rollback_failure_clear_every_staged_state(
    make_context: Callable[..., OperationContext],
) -> None:
    """A failed publish or rollback cannot leak staged entity, audit, revision, or ledger state."""
    raw_secret = "round2-uow-token"
    original = _request(title="original")
    changed = _request(title="changed")
    repository = ScriptedEntityRepository(
        ResearchRequest,
        entities={"round2_request": original},
        revisions={"round2_request": 1},
        script={"persistence.save": [DelegateStep(), DelegateStep(), DelegateStep()]},
    )

    class FailingAuditRepository(ScriptedAuditRepository):
        """Simulate a late raw publish failure after a preceding repository has committed."""

        def commit_transaction(self) -> None:
            super().commit_transaction()
            raise RuntimeError(f"password={raw_secret}")

    failing_audits = FailingAuditRepository(script={"persistence.audit.append": [DelegateStep()]})
    failed_commit = InMemoryUnitOfWork(
        research_requests=repository,
        audits=failing_audits,
        context=make_context(idempotency_key="failed-commit-key"),
        lifecycle_script={"uow.enter": [DelegateStep()], "uow.commit": [DelegateStep()]},
    )
    with pytest.raises(PersistenceError) as commit_error:
        async with failed_commit as active:
            await active.research_requests.save(
                "round2_request",
                changed,
                expected_revision=1,
                ctx=make_context(idempotency_key="failed-commit-key"),
            )
            await active.audits.append(
                _audit_event(),
                ctx=make_context(idempotency_key="failed-commit-audit-key"),
            )
            await active.commit()
    assert commit_error.value.failure.code == "PERSISTENCE_UNIT_OF_WORK_COMMIT_FAILED"
    assert raw_secret not in repr(commit_error.value)
    assert repository.backing_entity("round2_request") == original
    assert repository.backing_revision_of("round2_request") == 1
    assert repository.backing_effect_count() == 0
    assert failing_audits.backing_events == []
    assert failing_audits.backing_effect_count() == 0

    rollback_failure = make_failure(
        code="ROUND2_ROLLBACK_FAILURE",
        category="TEST_FAKE",
        message=f"token={raw_secret}",
        retryable=False,
        ctx=None,
    )
    failed_rollback = InMemoryUnitOfWork(
        research_requests=repository,
        context=make_context(idempotency_key="rollback-key"),
        lifecycle_script={
            "uow.enter": [DelegateStep()],
            "uow.rollback": [FailureStep(rollback_failure)],
        },
    )
    with pytest.raises(PersistenceError) as rollback_error:
        async with failed_rollback as active:
            await active.research_requests.save(
                "round2_request",
                changed,
                expected_revision=1,
                ctx=make_context(idempotency_key="rollback-key"),
            )
    assert raw_secret not in repr(rollback_error.value)
    assert repository.backing_entity("round2_request") == original
    assert repository.backing_revision_of("round2_request") == 1
    assert repository.backing_effect_count() == 0

    retry = InMemoryUnitOfWork(
        research_requests=repository,
        context=make_context(idempotency_key="rollback-key"),
        lifecycle_script={"uow.enter": [DelegateStep()], "uow.commit": [DelegateStep()]},
    )
    async with retry as active:
        await active.research_requests.save(
            "round2_request",
            changed,
            expected_revision=1,
            ctx=make_context(idempotency_key="rollback-key"),
        )
        await active.commit()
    assert repository.backing_entity("round2_request") == changed
    assert repository.backing_effect_count() == 1


@pytest.mark.contract
@pytest.mark.asyncio
async def test_artifact_preflight_validates_expected_hash_before_stream_consumption(
    make_context: Callable[..., OperationContext],
    make_descriptor: Callable[..., ArtifactDescriptor],
) -> None:
    """Invalid expected hashes reject with zero reads and leave their scripted success available."""
    fake = InMemoryArtifactStore(script={"artifact.put_bytes": [DelegateStep()]})
    wrong_type = ObservableStream([b"round2-content"])
    with pytest.raises(ArtifactError) as type_error:
        await fake.put_bytes(
            wrong_type,
            make_descriptor(artifact_id="round2-invalid-hash"),
            expected_sha256=cast(str | None, 123),
            ctx=make_context(idempotency_key="round2-invalid-hash-key"),
        )
    assert type_error.value.failure.code == "ARTIFACT_EXPECTED_SHA256_INVALID"
    assert wrong_type.iterations == 0

    malformed = ObservableStream([b"round2-content"])
    with pytest.raises(ArtifactError):
        await fake.put_bytes(
            malformed,
            make_descriptor(artifact_id="round2-malformed-hash"),
            expected_sha256="sha256:" + "A" * 64,
            ctx=make_context(idempotency_key="round2-malformed-hash-key"),
        )
    assert malformed.iterations == 0

    valid = ObservableStream([b"round2-content"])
    result = await fake.put_bytes(
        valid,
        make_descriptor(artifact_id="round2-valid-hash"),
        expected_sha256=None,
        ctx=make_context(idempotency_key="round2-valid-hash-key"),
    )
    assert result.artifact_id == "round2-valid-hash"
    assert fake.put_effect_count == 1


@pytest.mark.contract
@pytest.mark.asyncio
async def test_artifact_stream_replay_conflict_and_records_are_script_and_secret_safe(
    make_context: Callable[..., OperationContext],
    make_descriptor: Callable[..., ArtifactDescriptor],
) -> None:
    """A one-step stream fake replays matching content and rejects mismatches safely."""
    fake = InMemoryArtifactStore(script={"artifact.put_bytes": [DelegateStep()]})
    context = make_context(idempotency_key="round2-stream-key")
    descriptor = make_descriptor(artifact_id="round2-stream-artifact")
    first_stream = ObservableStream([b"round2-content"])
    first = await fake.put_bytes(first_stream, descriptor, expected_sha256=None, ctx=context)
    replay_stream = ObservableStream([b"round2-content"])
    replay = await fake.put_bytes(replay_stream, descriptor, expected_sha256=None, ctx=context)
    assert replay == first
    assert fake.put_effect_count == 1
    assert replay_stream.iterations > 0

    conflicting_stream = ObservableStream([b"round2-other-content"])
    with pytest.raises(IdempotencyConflictError):
        await fake.put_bytes(conflicting_stream, descriptor, expected_sha256=None, ctx=context)
    assert fake.put_effect_count == 1
    assert all("round2-content" not in repr(record.payload) for record in fake.calls)
    assert all("payload_fingerprint" not in repr(record) for record in fake.calls)
    assert all("digest" not in repr(record).casefold() for record in fake.calls)

    concurrent = InMemoryArtifactStore(script={"artifact.put_bytes": [DelegateStep()]})
    concurrent_context = make_context(idempotency_key="round2-concurrent-stream-key")
    concurrent_descriptor = make_descriptor(artifact_id="round2-concurrent-stream-artifact")
    left, right = await asyncio.gather(
        concurrent.put_bytes(
            ObservableStream([b"same-content"]),
            concurrent_descriptor,
            expected_sha256=None,
            ctx=concurrent_context,
        ),
        concurrent.put_bytes(
            ObservableStream([b"same-content"]),
            concurrent_descriptor,
            expected_sha256=None,
            ctx=concurrent_context,
        ),
    )
    assert left == right
    assert concurrent.put_effect_count == 1


@pytest.mark.contract
@pytest.mark.asyncio
async def test_artifact_open_rejects_bad_concrete_reader_shapes_before_returning(
    make_context: Callable[..., OperationContext],
) -> None:
    """Scripted reader returns require concrete async methods and valid iterator structure."""
    raw_secret = "round2-reader-token"

    class BadRead:
        def read(self, size: int) -> bytes:
            return b""

        def __aiter__(self) -> BadRead:
            return self

        async def __anext__(self) -> bytes:
            raise StopAsyncIteration

    class BadAiterSignature:
        async def read(self) -> bytes:
            return b""

        def __aiter__(self, required: int) -> BadAiterSignature:
            del required
            return self

        async def __anext__(self) -> bytes:
            raise StopAsyncIteration

    class BadAiterReturn:
        async def read(self) -> bytes:
            return b""

        def __aiter__(self) -> object:
            return object()

        async def __anext__(self) -> bytes:
            raise StopAsyncIteration

    class BadAnext:
        async def read(self) -> bytes:
            return b""

        def __aiter__(self) -> BadAnext:
            return self

        def __anext__(self) -> bytes:
            return raw_secret.encode("utf-8")

    for bad_reader in (BadRead(), BadAiterSignature(), BadAiterReturn(), BadAnext()):
        fake = InMemoryArtifactStore(script={"artifact.open": [ReturnStep(bad_reader)]})
        with pytest.raises(ArtifactError) as raised:
            await fake.open("unused", ctx=make_context(idempotency_key=None))
        assert raised.value.failure.code == "TEST_FAKE_SCRIPTED_RETURN_TYPE_INVALID"
        assert raw_secret not in repr(raised.value)


@pytest.mark.contract
def test_protocol_return_checks_accept_concrete_reader_and_self_covariance() -> None:
    """The shared assertion accepts the current concrete reader and UoW return types."""
    assert_concrete_protocol_signatures(AsyncBinaryReader, InMemoryAsyncReader(b""))
    assert_concrete_protocol_signatures(UnitOfWork, InMemoryUnitOfWork())
    assert isinstance(ScriptedStructuredLLM(), StructuredLLM)
