"""Idempotency and transaction boundary regressions for deterministic fakes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal

import pytest
from pydantic import BaseModel

from tests.fakes import (
    CancelledStep,
    DelegateStep,
    FailureStep,
    FrozenClock,
    IdempotencyLedger,
    InMemoryArtifactStore,
    InMemoryAsyncReader,
    InMemoryExperimentTracker,
    InMemoryUnitOfWork,
    ReturnStep,
    ScriptedAuditRepository,
    ScriptedEntityRepository,
    ScriptedPaperProvider,
    ScriptedRepositoryProvider,
    ScriptedStructuredLLM,
    TimeoutStep,
    safe_payload,
)
from vision_research_ops.domain import (
    ResearchBudget,
    ResearchRequest,
    RunStatus,
    StructuredFailure,
    WorkflowStatus,
)
from vision_research_ops.ports import (
    ArtifactDescriptor,
    ArtifactError,
    AuditEvent,
    CapabilityNotSupportedError,
    FakeScriptExhaustedError,
    IdempotencyConflictError,
    LLMError,
    OperationCancelledError,
    OperationContext,
    OperationTimeoutError,
    PaperQuery,
    PaperSearchPage,
    PortError,
    ProviderError,
    RepositoryResolution,
    StructuredGenerationRequest,
    StructuredOutputValidationError,
    TrackerError,
    TrackerRunRef,
    make_failure,
)

TIMESTAMP = "2026-08-09T00:00:00Z"


class Proposal(BaseModel):
    """Strict structured output used by deterministic scripted LLM tests."""

    schema_version: Literal["1"]
    recommendation: str


class PermissiveNumber(BaseModel):
    """Deliberately permissive schema proving post-validation finite JSON enforcement."""

    number: float


class CountingAsyncIterator:
    """Asynchronous byte stream whose iteration count makes preflight observable."""

    def __init__(self, chunks: list[bytes], *, failure: BaseException | None = None) -> None:
        self._chunks = chunks
        self._failure = failure
        self._index = 0
        self.iterations = 0

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Return the stream itself without consuming a chunk."""
        return self

    async def __anext__(self) -> bytes:
        """Yield a configured chunk or an explicit read-time failure."""
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


def _make_request(*, title: str = "original") -> ResearchRequest:
    """Build a small valid entity for staged persistence regression tests."""
    return ResearchRequest(
        schema_version="1",
        request_id="request_1",
        revision=1,
        title=title,
        research_question="Can the synthetic fixture be reproduced?",
        dataset_id="dataset_1",
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
        requested_by="actor_1",
        status=WorkflowStatus.PENDING,
        created_at=TIMESTAMP,
        updated_at=TIMESTAMP,
    )


def _make_audit_event() -> AuditEvent:
    """Build a small append-only event for staged audit regression tests."""
    return AuditEvent(
        schema_version="1",
        event_id="event_1",
        event_type="REQUEST_SAVED",
        occurred_at=TIMESTAMP,
        correlation_id="corr_1",
        workflow_id="wf_1",
        subject_type="research_request",
        subject_id="request_1",
        subject_revision=1,
        payload={"action": "save"},
    )


def _make_generation_request() -> StructuredGenerationRequest[Proposal]:
    """Build a bounded provider-neutral LLM request without real provider configuration."""
    return StructuredGenerationRequest[Proposal](
        schema_version="1",
        task_name="contract-check",
        prompt_template_id="contract-check/v1",
        prompt_version="1",
        response_schema=Proposal,
        facts={"topic": "synthetic"},
        budget_class="test",
    )


@pytest.mark.contract
@pytest.mark.asyncio
async def test_idempotency_ledger_shares_concurrent_same_payload_execution(
    make_context: Callable[..., OperationContext],
) -> None:
    """Concurrent equivalent calls share one task and increment the effect counter once."""
    ledger: IdempotencyLedger[str] = IdempotencyLedger()
    started = asyncio.Event()
    release = asyncio.Event()
    executions = 0
    context = make_context(idempotency_key="same_key")

    async def execute() -> str:
        nonlocal executions
        executions += 1
        started.set()
        await release.wait()
        return "submitted"

    first = asyncio.create_task(
        ledger.replay_or_execute(
            operation="test.submit",
            idempotency_key="same_key",
            payload={"request": "same"},
            ctx=context,
            execute=execute,
        )
    )
    await started.wait()
    second = asyncio.create_task(
        ledger.replay_or_execute(
            operation="test.submit",
            idempotency_key="same_key",
            payload={"request": "same"},
            ctx=context,
            execute=execute,
        )
    )
    await ledger.wait_for_in_flight_join()
    assert executions == 1
    assert ledger.in_flight_count == 1
    release.set()
    assert await asyncio.gather(first, second) == ["submitted", "submitted"]
    assert ledger.effect_counts["test.submit"] == 1
    assert ledger.in_flight_count == 0


@pytest.mark.contract
@pytest.mark.asyncio
async def test_idempotency_ledger_rejects_concurrent_conflict_without_second_effect(
    make_context: Callable[..., OperationContext],
) -> None:
    """A conflicting payload sees the in-flight key and cannot execute its own side effect."""
    ledger: IdempotencyLedger[str] = IdempotencyLedger()
    started = asyncio.Event()
    release = asyncio.Event()
    executions = 0
    context = make_context(idempotency_key="same_key")

    async def execute() -> str:
        nonlocal executions
        executions += 1
        started.set()
        await release.wait()
        return "submitted"

    owner = asyncio.create_task(
        ledger.replay_or_execute(
            operation="test.submit",
            idempotency_key="same_key",
            payload={"request": "first"},
            ctx=context,
            execute=execute,
        )
    )
    await started.wait()
    with pytest.raises(IdempotencyConflictError):
        await ledger.replay_or_execute(
            operation="test.submit",
            idempotency_key="same_key",
            payload={"request": "different"},
            ctx=context,
            execute=execute,
        )
    assert executions == 1
    release.set()
    assert await owner == "submitted"
    assert ledger.effect_counts["test.submit"] == 1


@pytest.mark.contract
@pytest.mark.asyncio
async def test_idempotency_ledger_does_not_serialize_independent_keys(
    make_context: Callable[..., OperationContext],
) -> None:
    """Different keys can enter their independent side effects before either one completes."""
    ledger: IdempotencyLedger[str] = IdempotencyLedger()
    both_started = asyncio.Event()
    release = asyncio.Event()
    started: set[str] = set()

    def effect_for(name: str) -> Callable[[], Awaitable[str]]:
        async def execute() -> str:
            started.add(name)
            if len(started) == 2:
                both_started.set()
            await release.wait()
            return name

        return execute

    first = asyncio.create_task(
        ledger.replay_or_execute(
            operation="test.parallel",
            idempotency_key="first_key",
            payload={"request": "first"},
            ctx=make_context(idempotency_key="first_key"),
            execute=effect_for("first"),
        )
    )
    second = asyncio.create_task(
        ledger.replay_or_execute(
            operation="test.parallel",
            idempotency_key="second_key",
            payload={"request": "second"},
            ctx=make_context(idempotency_key="second_key"),
            execute=effect_for("second"),
        )
    )
    await both_started.wait()
    assert ledger.in_flight_count == 2
    release.set()
    assert set(await asyncio.gather(first, second)) == {"first", "second"}
    assert ledger.effect_counts["test.parallel"] == 2


@pytest.mark.contract
@pytest.mark.asyncio
async def test_idempotency_ledger_cleans_failure_and_cancellation_for_safe_retry(
    make_context: Callable[..., OperationContext],
) -> None:
    """Failures are not cached and every terminal outcome clears the in-flight marker."""
    context = make_context(idempotency_key="retry_key")
    ledger: IdempotencyLedger[str] = IdempotencyLedger()

    async def fail() -> str:
        raise RuntimeError("synthetic execution failure")

    with pytest.raises(RuntimeError):
        await ledger.replay_or_execute(
            operation="test.submit",
            idempotency_key="retry_key",
            payload={"request": "same"},
            ctx=context,
            execute=fail,
        )
    assert ledger.in_flight_count == 0
    assert ledger.effect_counts["test.submit"] == 0

    async def cancel() -> str:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await ledger.replay_or_execute(
            operation="test.submit",
            idempotency_key="retry_key",
            payload={"request": "same"},
            ctx=context,
            execute=cancel,
        )
    assert ledger.in_flight_count == 0

    async def succeed() -> str:
        return "retried"

    assert (
        await ledger.replay_or_execute(
            operation="test.submit",
            idempotency_key="retry_key",
            payload={"request": "same"},
            ctx=context,
            execute=succeed,
        )
        == "retried"
    )
    assert ledger.effect_counts["test.submit"] == 1


@pytest.mark.contract
@pytest.mark.asyncio
async def test_secret_redaction_is_recursive_and_fingerprint_preserves_sensitive_differences(
    make_context: Callable[..., OperationContext],
) -> None:
    """Redact observations recursively while retaining comparison differences only in hashes."""
    raw_secret_one = "synthetic-secret-one"
    raw_secret_two = "synthetic-secret-two"
    details = {
        "Nested": {
            "Authorization": raw_secret_one,
            "items": [{"PASSWORD": raw_secret_one}, {"apiKey": raw_secret_one}],
        },
        "TOKEN": raw_secret_one,
        "credential_value": raw_secret_one,
    }
    failure = make_failure(
        code="TEST_FAKE_REDACTION_CHECK",
        category="TEST_FAKE",
        message="A synthetic failure was redacted.",
        retryable=False,
        ctx=make_context(idempotency_key=None),
        details=details,
    )
    assert failure.details["Nested"]["Authorization"] == "[REDACTED]"
    assert failure.details["Nested"]["items"][0]["PASSWORD"] == "[REDACTED]"
    assert failure.details["Nested"]["items"][1]["apiKey"] == "[REDACTED]"
    assert failure.details["TOKEN"] == "[REDACTED]"
    assert failure.details["credential_value"] == "[REDACTED]"
    assert raw_secret_one not in repr(failure.details)

    untrusted_failure = StructuredFailure(
        schema_version="1",
        code="TEST_FAKE_UNTRUSTED_DETAILS",
        category="TEST_FAKE",
        message="A controlled failure wrapper must redact details.",
        message_hash="sha256:" + "b" * 64,
        retryable=False,
        details={"nested": {"Authorization": raw_secret_one}},
    )
    wrapped = ArtifactError(untrusted_failure)
    assert wrapped.failure.details["nested"]["Authorization"] == "[REDACTED]"
    assert raw_secret_one not in repr(wrapped.failure.details)

    safe = safe_payload({"outer": [{"Authorization": raw_secret_one}]})
    assert safe["outer"][0]["Authorization"] == "[REDACTED]"
    assert raw_secret_one not in repr(safe)
    ledger: IdempotencyLedger[str] = IdempotencyLedger()

    async def effect() -> str:
        return "recorded"

    context = make_context(idempotency_key="sensitive_key")
    assert (
        await ledger.replay_or_execute(
            operation="test.secret",
            idempotency_key="sensitive_key",
            payload={"Authorization": raw_secret_one},
            ctx=context,
            execute=effect,
        )
        == "recorded"
    )
    with pytest.raises(IdempotencyConflictError):
        await ledger.replay_or_execute(
            operation="test.secret",
            idempotency_key="sensitive_key",
            payload={"Authorization": raw_secret_two},
            ctx=context,
            execute=effect,
        )


@pytest.mark.contract
@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
async def test_idempotency_ledger_rejects_non_finite_json(
    value: float,
    make_context: Callable[..., OperationContext],
) -> None:
    """Canonical payload comparison fails closed without exposing a public digest."""
    ledger: IdempotencyLedger[str] = IdempotencyLedger()

    async def effect() -> str:
        return "unreachable"

    with pytest.raises(PortError) as raised:
        await ledger.replay_or_execute(
            operation="test.finite-json",
            idempotency_key="finite-json-key",
            payload={"nested": [{"value": value}]},
            ctx=make_context(idempotency_key="finite-json-key"),
            execute=effect,
        )
    assert raised.value.failure.code == "PORT_PAYLOAD_CANONICALIZATION_FAILED"


@pytest.mark.contract
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "step",
    [
        FailureStep(
            StructuredFailure(
                schema_version="1",
                code="TEST_FAKE_ARTIFACT_FAILURE",
                category="TEST_FAKE",
                message="Synthetic artifact preflight failure.",
                message_hash="sha256:" + "a" * 64,
                retryable=False,
                details={},
            )
        ),
        TimeoutStep(),
        CancelledStep(),
    ],
)
async def test_artifact_script_preflight_rejects_before_reading_stream(
    step: object,
    make_context: Callable[..., OperationContext],
    make_descriptor: Callable[..., ArtifactDescriptor],
) -> None:
    """Failure, timeout, and cancellation script steps consume zero stream chunks."""
    stream = CountingAsyncIterator([b"synthetic-content"])
    fake = InMemoryArtifactStore(script={"artifact.put_bytes": [step]})
    with pytest.raises((ArtifactError, OperationTimeoutError, OperationCancelledError)):
        await fake.put_bytes(
            stream,
            make_descriptor(artifact_id="preflight_artifact"),
            expected_sha256=None,
            ctx=make_context(idempotency_key="preflight_key"),
        )
    assert stream.iterations == 0
    assert fake.put_effect_count == 0
    assert "synthetic-content" not in repr(fake.calls[0].payload)


@pytest.mark.contract
@pytest.mark.asyncio
async def test_artifact_capability_deadline_and_exhaustion_preflight_do_not_read_stream(
    make_context: Callable[..., OperationContext],
    make_descriptor: Callable[..., ArtifactDescriptor],
) -> None:
    """Every non-executable artifact preflight failure leaves the async iterator untouched."""
    unsupported_stream = CountingAsyncIterator([b"synthetic-content"])
    unsupported = InMemoryArtifactStore(
        supported_operations=("artifact.open",),
        script={"artifact.put_bytes": [DelegateStep()]},
    )
    with pytest.raises(CapabilityNotSupportedError):
        await unsupported.put_bytes(
            unsupported_stream,
            make_descriptor(artifact_id="unsupported_artifact"),
            expected_sha256=None,
            ctx=make_context(idempotency_key="unsupported_key"),
        )
    assert unsupported_stream.iterations == 0

    expired_stream = CountingAsyncIterator([b"synthetic-content"])
    expired = InMemoryArtifactStore(
        clock=FrozenClock(datetime(2026, 1, 1, tzinfo=UTC)),
        script={"artifact.put_bytes": [DelegateStep()]},
    )
    with pytest.raises(OperationTimeoutError):
        await expired.put_bytes(
            expired_stream,
            make_descriptor(artifact_id="expired_artifact"),
            expected_sha256=None,
            ctx=make_context(
                idempotency_key="expired_key",
                deadline_at="2025-12-31T23:59:59Z",
            ),
        )
    assert expired_stream.iterations == 0

    exhausted_stream = CountingAsyncIterator([b"synthetic-content"])
    exhausted = InMemoryArtifactStore()
    with pytest.raises(FakeScriptExhaustedError) as raised:
        await exhausted.put_bytes(
            exhausted_stream,
            make_descriptor(artifact_id="exhausted_artifact"),
            expected_sha256=None,
            ctx=make_context(idempotency_key="exhausted_key"),
        )
    assert raised.value.failure.code == "FAKE_SCRIPT_EXHAUSTED"
    assert exhausted_stream.iterations == 0


@pytest.mark.contract
@pytest.mark.asyncio
async def test_artifact_stream_is_summarized_only_after_preflight_and_cancel_does_not_finalize(
    make_context: Callable[..., OperationContext],
    make_descriptor: Callable[..., ArtifactDescriptor],
) -> None:
    """Successful streams record only hash/size; read-time cancellation has no artifact effect."""
    stream = CountingAsyncIterator([b"synthetic-content"])
    fake = InMemoryArtifactStore(script={"artifact.put_bytes": [DelegateStep()]})
    reference = await fake.put_bytes(
        stream,
        make_descriptor(artifact_id="stream_artifact"),
        expected_sha256=None,
        ctx=make_context(idempotency_key="stream_key"),
    )
    assert stream.iterations > 0
    assert fake.calls[0].payload["data"]["size_bytes"] == len(b"synthetic-content")
    assert "synthetic-content" not in repr(fake.calls[0].payload)
    assert fake.put_effect_count == 1
    assert reference.artifact_id == "stream_artifact"

    cancelled_context = make_context(idempotency_key="cancelled_stream_key")
    cancelled_stream = CountingAsyncIterator(
        [],
        failure=OperationCancelledError("artifact.put_bytes", cancelled_context),
    )
    cancelled_fake = InMemoryArtifactStore(script={"artifact.put_bytes": [DelegateStep()]})
    with pytest.raises(OperationCancelledError):
        await cancelled_fake.put_bytes(
            cancelled_stream,
            make_descriptor(artifact_id="cancelled_stream_artifact"),
            expected_sha256=None,
            ctx=cancelled_context,
        )
    assert cancelled_fake.put_effect_count == 0
    assert "cancelled_stream_artifact" not in cancelled_fake._refs


@pytest.mark.contract
@pytest.mark.asyncio
async def test_scripted_returns_are_validated_for_models_none_and_async_reader(
    make_context: Callable[..., OperationContext],
    make_descriptor: Callable[..., ArtifactDescriptor],
    make_page: Callable[..., PaperSearchPage],
    make_query: Callable[[], PaperQuery],
    make_resolution: Callable[[], RepositoryResolution],
) -> None:
    """Concrete fakes reject wrong ReturnSteps and preserve documented special return forms."""
    context = make_context(idempotency_key=None)
    page = make_page()
    paper = ScriptedPaperProvider(script={"paper.search": [ReturnStep(page)]})
    assert await paper.search(make_query(), cursor=None, ctx=context) == page
    bad_paper = ScriptedPaperProvider(script={"paper.search": [ReturnStep("not-a-page")]})
    with pytest.raises(ProviderError) as bad_paper_error:
        await bad_paper.search(make_query(), cursor=None, ctx=context)
    assert bad_paper_error.value.failure.code == "TEST_FAKE_SCRIPTED_RETURN_TYPE_INVALID"

    resolution = make_resolution()
    repository = ScriptedRepositoryProvider(script={"repository.resolve": [ReturnStep(resolution)]})
    assert await repository.resolve("ignored", None, ctx=context) == resolution
    bad_repository = ScriptedRepositoryProvider(
        script={"repository.resolve": [ReturnStep("not-a-resolution")]}
    )
    with pytest.raises(ProviderError):
        await bad_repository.resolve("ignored", None, ctx=context)

    seed_llm = ScriptedStructuredLLM(
        outputs={"contract-check": {"schema_version": "1", "recommendation": "keep"}},
        script={"llm.generate": [DelegateStep()]},
    )
    request = _make_generation_request()
    generation = await seed_llm.generate(request, ctx=context)
    scripted_llm = ScriptedStructuredLLM(script={"llm.generate": [ReturnStep(generation)]})
    assert await scripted_llm.generate(request, ctx=context) == generation
    invalid_llm = ScriptedStructuredLLM(script={"llm.generate": [ReturnStep("not-a-result")]})
    with pytest.raises(LLMError) as invalid_llm_error:
        await invalid_llm.generate(request, ctx=context)
    assert invalid_llm_error.value.failure.code == "TEST_FAKE_SCRIPTED_RETURN_TYPE_INVALID"

    reader = InMemoryAsyncReader(b"reader-content")
    store = InMemoryArtifactStore(script={"artifact.open": [ReturnStep(reader)]})
    validated_reader = await store.open("unused", ctx=context)
    assert validated_reader is not reader
    assert await validated_reader.read() == b"reader-content"
    bad_reader = InMemoryArtifactStore(script={"artifact.open": [ReturnStep("not-a-reader")]})
    with pytest.raises(ArtifactError):
        await bad_reader.open("unused", ctx=context)

    tracker = InMemoryExperimentTracker(script={"tracker.finalize": [ReturnStep(None)]})
    tracker_ref = TrackerRunRef(
        schema_version="1",
        tracker_run_id="tracker_1",
        run_id="run_1",
        uri="fake://tracker/tracker_1",
    )
    assert await tracker.finalize(tracker_ref, RunStatus.SUCCEEDED, ctx=make_context()) is None
    bad_tracker = InMemoryExperimentTracker(script={"tracker.finalize": [ReturnStep("not-none")]})
    with pytest.raises(TrackerError):
        await bad_tracker.finalize(tracker_ref, RunStatus.SUCCEEDED, ctx=make_context())


@pytest.mark.contract
@pytest.mark.asyncio
@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
async def test_permissive_llm_schema_still_rejects_non_finite_json(
    number: float,
    make_context: Callable[..., OperationContext],
) -> None:
    """Post-validation canonical JSON rejects non-finite values accepted by a permissive model."""
    request = StructuredGenerationRequest[PermissiveNumber](
        schema_version="1",
        task_name="finite-json",
        prompt_template_id="finite/v1",
        prompt_version="1",
        response_schema=PermissiveNumber,
        budget_class="test",
    )
    fake = ScriptedStructuredLLM(
        outputs={"finite-json": {"number": number}},
        script={"llm.generate": [DelegateStep()]},
    )
    with pytest.raises(StructuredOutputValidationError) as raised:
        await fake.generate(request, ctx=make_context(idempotency_key=None))
    assert raised.value.failure.code == "LLM_SCHEMA_VALIDATION_FAILED"
    assert repr(number) not in str(raised.value)


@pytest.mark.contract
@pytest.mark.asyncio
async def test_unit_of_work_commit_publishes_staged_entity_revision_audit_and_idempotency(
    make_context: Callable[..., OperationContext],
) -> None:
    """Commit is the only path that makes staged entity, audit, and replay state durable."""
    original = _make_request()
    updated = _make_request(title="committed")
    repository = ScriptedEntityRepository(
        ResearchRequest,
        entities={"request_1": original},
        revisions={"request_1": 1},
        script={"persistence.get": [DelegateStep()], "persistence.save": [DelegateStep()]},
    )
    audits = ScriptedAuditRepository(script={"persistence.audit.append": [DelegateStep()]})
    context = make_context(idempotency_key="entity_key")
    uow = InMemoryUnitOfWork(
        research_requests=repository,
        audits=audits,
        context=context,
        lifecycle_script={"uow.enter": [DelegateStep()], "uow.commit": [DelegateStep()]},
    )
    async with uow as active:
        loaded = await active.research_requests.get(
            "request_1",
            ctx=make_context(idempotency_key=None),
        )
        assert loaded == original
        await active.research_requests.save(
            "request_1",
            updated,
            expected_revision=1,
            ctx=context,
        )
        await active.audits.append(
            _make_audit_event(),
            ctx=make_context(idempotency_key="audit_key"),
        )
        assert active.research_requests.revision_of("request_1") == 2
        assert repository.backing_revision_of("request_1") == 1
        assert repository.backing_entity("request_1") == original
        assert audits.backing_events == []
        assert repository.backing_effect_count() == 0
        assert audits.backing_effect_count() == 0
        await active.commit()
    assert repository.backing_entity("request_1") == updated
    assert repository.backing_revision_of("request_1") == 2
    assert audits.backing_events == [_make_audit_event()]
    assert repository.backing_effect_count() == 1
    assert audits.backing_effect_count() == 1
    assert uow.calls == ["uow.enter", "uow.commit", "uow.exit"]


@pytest.mark.contract
@pytest.mark.asyncio
async def test_unit_of_work_explicit_rollback_restores_revision_audit_and_idempotency_for_retry(
    make_context: Callable[..., OperationContext],
) -> None:
    """Rollback discards every staged mutation and permits the same idempotency key to retry."""
    original = _make_request()
    updated = _make_request(title="retried")
    repository = ScriptedEntityRepository(
        ResearchRequest,
        entities={"request_1": original},
        revisions={"request_1": 1},
        script={"persistence.save": [DelegateStep()]},
    )
    audits = ScriptedAuditRepository(script={"persistence.audit.append": [DelegateStep()]})
    context = make_context(idempotency_key="rollback_key")
    first = InMemoryUnitOfWork(
        research_requests=repository,
        audits=audits,
        context=context,
        lifecycle_script={"uow.enter": [DelegateStep()], "uow.rollback": [DelegateStep()]},
    )
    async with first as active:
        await active.research_requests.save(
            "request_1",
            updated,
            expected_revision=1,
            ctx=context,
        )
        await active.audits.append(
            _make_audit_event(),
            ctx=make_context(idempotency_key="rollback_audit_key"),
        )
        await active.rollback()
    assert repository.backing_entity("request_1") == original
    assert repository.backing_revision_of("request_1") == 1
    assert audits.backing_events == []
    assert repository.backing_effect_count() == 0
    assert audits.backing_effect_count() == 0
    assert first.calls == ["uow.enter", "uow.rollback", "uow.exit"]

    repository.append_script("persistence.save", DelegateStep())
    audits.append_script("persistence.audit.append", DelegateStep())
    retry = InMemoryUnitOfWork(
        research_requests=repository,
        audits=audits,
        context=context,
        lifecycle_script={"uow.enter": [DelegateStep()], "uow.commit": [DelegateStep()]},
    )
    async with retry as active:
        await active.research_requests.save(
            "request_1",
            updated,
            expected_revision=1,
            ctx=context,
        )
        await active.audits.append(
            _make_audit_event(),
            ctx=make_context(idempotency_key="rollback_audit_key"),
        )
        await active.commit()
    assert repository.backing_entity("request_1") == updated
    assert repository.backing_revision_of("request_1") == 2
    assert audits.backing_events == [_make_audit_event()]
    assert repository.backing_effect_count() == 1
    assert audits.backing_effect_count() == 1


@pytest.mark.contract
@pytest.mark.asyncio
async def test_unit_of_work_exception_and_normal_uncommitted_exit_fail_closed(
    make_context: Callable[..., OperationContext],
) -> None:
    """Both exceptional and ordinary exits without commit roll back staged state."""
    original = _make_request()
    updated = _make_request(title="not-committed")

    async def exercise(
        *,
        raise_inside: bool,
    ) -> tuple[ScriptedEntityRepository[ResearchRequest], InMemoryUnitOfWork]:
        repository = ScriptedEntityRepository(
            ResearchRequest,
            entities={"request_1": original},
            revisions={"request_1": 1},
            script={"persistence.save": [DelegateStep()]},
        )
        uow = InMemoryUnitOfWork(
            research_requests=repository,
            context=make_context(idempotency_key="exit_key"),
            lifecycle_script={"uow.enter": [DelegateStep()], "uow.rollback": [DelegateStep()]},
        )
        if raise_inside:
            with pytest.raises(RuntimeError, match="synthetic application failure"):
                async with uow as active:
                    await active.research_requests.save(
                        "request_1",
                        updated,
                        expected_revision=1,
                        ctx=make_context(idempotency_key="exit_key"),
                    )
                    raise RuntimeError("synthetic application failure")
        else:
            async with uow as active:
                await active.research_requests.save(
                    "request_1",
                    updated,
                    expected_revision=1,
                    ctx=make_context(idempotency_key="exit_key"),
                )
        return repository, uow

    exceptional_repository, exceptional_uow = await exercise(raise_inside=True)
    normal_repository, normal_uow = await exercise(raise_inside=False)
    for repository, uow in (
        (exceptional_repository, exceptional_uow),
        (normal_repository, normal_uow),
    ):
        assert repository.backing_entity("request_1") == original
        assert repository.backing_revision_of("request_1") == 1
        assert repository.backing_effect_count() == 0
        assert uow.committed is False
        assert uow.rolled_back is True
        assert uow.calls == ["uow.enter", "uow.rollback", "uow.exit"]
