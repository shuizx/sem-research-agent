"""Behavior contracts for deterministic scripted deterministic port contract fakes."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from tests.contract.base import (
    assert_replay_and_conflict,
    assert_structured_failure,
)
from tests.fakes import (
    CancelledStep,
    DelegateStep,
    FailureStep,
    FrozenClock,
    InMemoryArtifactStore,
    InMemoryExperimentTracker,
    InMemoryUnitOfWork,
    ScriptedExperimentExecutor,
    ScriptedPaperProvider,
    ScriptedRepositoryProvider,
    ScriptedStructuredLLM,
    TimeoutStep,
)
from vision_research_ops.domain import ArtifactRef, RunStatus
from vision_research_ops.ports import (
    ArtifactDescriptor,
    ArtifactError,
    CancellationResult,
    CapabilityNotSupportedError,
    ExternalPaperId,
    FakeScriptExhaustedError,
    FrozenRunSpec,
    GenerationUsage,
    IdempotencyConflictError,
    LLMError,
    MetricPoint,
    OperationCancelledError,
    OperationContext,
    OperationTimeoutError,
    PaperQuery,
    PaperSearchPage,
    ProviderError,
    RawPaperRecord,
    RepositoryResolution,
    RunManifest,
    StructuredGenerationRequest,
    StructuredOutputValidationError,
    SubmissionResult,
    TrackerRunRef,
    make_failure,
)

SHA256 = "sha256:" + "a" * 64
TIMESTAMP = "2026-08-09T00:00:00Z"


class Proposal(BaseModel):
    """Strict output schema used to prove scripted LLM validation behavior."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1"]
    recommendation: str


@pytest.mark.contract
@pytest.mark.asyncio
async def test_paper_fake_handles_success_empty_page_pagination_and_records_calls(
    make_context: Callable[..., OperationContext],
    make_page: Callable[..., PaperSearchPage],
    make_query: Callable[[], PaperQuery],
    make_raw_record: Callable[..., RawPaperRecord],
) -> None:
    """Paper fake returns configured pages and records stable sequence and input details."""
    raw = make_raw_record()
    first_page = make_page(records=[raw], next_cursor="cursor_2")
    empty_page = make_page(records=[], next_cursor=None, request_id="provider_request_2")
    fake = ScriptedPaperProvider(
        pages={None: first_page, "cursor_2": empty_page},
        records={("scripted-paper", "paper_1"): raw},
        script={
            "paper.search": [DelegateStep(), DelegateStep()],
            "paper.get_by_external_id": [DelegateStep()],
        },
    )
    ctx = make_context(idempotency_key=None)
    query = make_query()

    result_page = await fake.search(query, cursor=None, ctx=ctx)
    result_empty = await fake.search(query, cursor="cursor_2", ctx=ctx)
    result_record = await fake.get_by_external_id(
        ExternalPaperId(schema_version="1", provider_name="scripted-paper", value="paper_1"),
        ctx=ctx,
    )

    assert result_page.records == [raw]
    assert result_page.next_cursor == "cursor_2"
    assert result_empty.records == []
    assert result_empty.next_cursor is None
    assert result_record == raw
    fake.assert_call_order("paper.search", "paper.search", "paper.get_by_external_id")
    assert fake.calls[0].payload["cursor"] is None
    assert fake.calls[1].payload["cursor"] == "cursor_2"
    assert fake.port_name == "PaperProvider"


@pytest.mark.contract
@pytest.mark.asyncio
async def test_paper_fake_reports_timeout_cancellation_and_retry_semantics(
    make_context: Callable[..., OperationContext],
    make_query: Callable[[], PaperQuery],
) -> None:
    """Timeout, cancellation, retryable, and non-retryable outcomes remain structured."""
    retryable = make_failure(
        code="RETRIEVAL_PROVIDER_RATE_LIMITED",
        category="PROVIDER",
        message="Rate limited.",
        retryable=True,
        ctx=None,
    )
    non_retryable = make_failure(
        code="RETRIEVAL_PROVIDER_REQUEST_INVALID",
        category="PROVIDER",
        message="Request invalid.",
        retryable=False,
        ctx=None,
    )
    fake = ScriptedPaperProvider(
        script={
            "paper.search": [
                TimeoutStep(),
                CancelledStep(),
                FailureStep(retryable),
                FailureStep(non_retryable),
            ]
        }
    )
    ctx = make_context(correlation_id="corr_fail", idempotency_key=None)
    query = make_query()

    with pytest.raises(OperationTimeoutError) as timed_out:
        await fake.search(query, cursor=None, ctx=ctx)
    assert_structured_failure(
        timed_out.value,
        code="PORT_OPERATION_TIMEOUT",
        retryable=True,
        correlation_id="corr_fail",
    )
    with pytest.raises(OperationCancelledError) as cancelled:
        await fake.search(query, cursor=None, ctx=ctx)
    assert_structured_failure(
        cancelled.value,
        code="PORT_OPERATION_CANCELLED",
        retryable=False,
        correlation_id="corr_fail",
    )
    with pytest.raises(ProviderError) as retryable_error:
        await fake.search(query, cursor=None, ctx=ctx)
    assert_structured_failure(
        retryable_error.value,
        code="RETRIEVAL_PROVIDER_RATE_LIMITED",
        retryable=True,
        correlation_id="corr_fail",
    )
    with pytest.raises(ProviderError) as non_retryable_error:
        await fake.search(query, cursor=None, ctx=ctx)
    assert_structured_failure(
        non_retryable_error.value,
        code="RETRIEVAL_PROVIDER_REQUEST_INVALID",
        retryable=False,
        correlation_id="corr_fail",
    )


@pytest.mark.contract
@pytest.mark.asyncio
async def test_paper_fake_enforces_deadline_and_capability_fail_closed(
    make_context: Callable[..., OperationContext],
    make_query: Callable[[], PaperQuery],
) -> None:
    """Context deadlines and omitted capabilities fail before any fake fallback occurs."""
    clock = FrozenClock(datetime(2026, 1, 1, tzinfo=UTC))
    deadline_fake = ScriptedPaperProvider(
        script={"paper.search": [DelegateStep()]},
        clock=clock,
    )
    expired_context = make_context(
        idempotency_key=None,
        deadline_at="2025-12-31T23:59:59Z",
    )
    with pytest.raises(OperationTimeoutError):
        await deadline_fake.search(make_query(), cursor=None, ctx=expired_context)

    unsupported = ScriptedPaperProvider(
        supported_operations=["paper.get_by_external_id"],
        script={"paper.search": [DelegateStep()]},
    )
    with pytest.raises(CapabilityNotSupportedError) as unsupported_error:
        await unsupported.search(make_query(), cursor=None, ctx=make_context(idempotency_key=None))
    assert unsupported_error.value.failure.code == "PORT_CAPABILITY_NOT_SUPPORTED"


@pytest.mark.contract
@pytest.mark.asyncio
async def test_unscripted_fake_calls_fail_closed_instead_of_returning_default_success(
    make_context: Callable[..., OperationContext],
    make_query: Callable[[], PaperQuery],
) -> None:
    """Configured data alone never permits a fake method to silently succeed."""
    fake = ScriptedPaperProvider()
    with pytest.raises(FakeScriptExhaustedError) as exhausted:
        await fake.search(make_query(), cursor=None, ctx=make_context(idempotency_key=None))
    assert exhausted.value.failure.code == "FAKE_SCRIPT_EXHAUSTED"


@pytest.mark.contract
@pytest.mark.asyncio
async def test_repository_snapshot_replays_and_rejects_conflicting_payloads(
    make_artifact_ref: Callable[..., ArtifactRef],
    make_context: Callable[..., OperationContext],
    make_resolution: Callable[..., RepositoryResolution],
) -> None:
    """Snapshot side effects are safe to replay and reject key reuse with a new payload."""
    first_resolution = make_resolution()
    conflicting_resolution = make_resolution(url="https://example.invalid/org/other")
    expected = make_artifact_ref(artifact_id="snapshot_1")
    fake = ScriptedRepositoryProvider(
        snapshots={first_resolution.commit_sha: expected},
        script={"repository.snapshot": [DelegateStep()]},
    )
    ctx = make_context(idempotency_key="snapshot_idempotency")

    await assert_replay_and_conflict(
        first_call=lambda: fake.snapshot(first_resolution, ctx=ctx),
        replay_call=lambda: fake.snapshot(first_resolution, ctx=ctx),
        conflicting_call=lambda: fake.snapshot(conflicting_resolution, ctx=ctx),
        error_type=IdempotencyConflictError,
        expected_result=expected,
    )
    assert fake.snapshot_effect_count == 1
    assert fake.call_count("repository.snapshot") == 3


@pytest.mark.contract
@pytest.mark.asyncio
async def test_artifact_fake_is_immutable_hash_checked_and_replay_safe(
    make_context: Callable[..., OperationContext],
    make_descriptor: Callable[..., ArtifactDescriptor],
) -> None:
    """Artifact bytes have exact hash behavior, immutable IDs, and idempotent finalization."""
    descriptor = make_descriptor(artifact_id="artifact_1")
    context = make_context(idempotency_key="artifact_idempotency")
    fake = InMemoryArtifactStore(
        script={
            "artifact.put_bytes": [DelegateStep()],
            "artifact.stat": [DelegateStep()],
            "artifact.open": [DelegateStep()],
        }
    )

    first = await fake.put_bytes(b"abc", descriptor, expected_sha256=None, ctx=context)
    replay = await fake.put_bytes(b"abc", descriptor, expected_sha256=None, ctx=context)
    assert replay == first
    assert fake.put_effect_count == 1
    with pytest.raises(IdempotencyConflictError) as conflict:
        await fake.put_bytes(b"different", descriptor, expected_sha256=None, ctx=context)
    assert conflict.value.failure.code == "IDEMPOTENCY_KEY_REUSED_WITH_DIFFERENT_PAYLOAD"

    read_context = make_context(idempotency_key=None)
    assert await fake.stat("artifact_1", ctx=read_context) == first
    reader = await fake.open("artifact_1", ctx=read_context)
    assert await reader.read() == b"abc"

    mismatch = InMemoryArtifactStore(script={"artifact.put_bytes": [DelegateStep()]})
    with pytest.raises(ArtifactError) as hash_mismatch:
        await mismatch.put_bytes(
            b"abc",
            make_descriptor(artifact_id="artifact_2"),
            expected_sha256=SHA256,
            ctx=make_context(idempotency_key="hash_mismatch"),
        )
    assert hash_mismatch.value.failure.code == "ARTIFACT_HASH_MISMATCH"


@pytest.mark.contract
@pytest.mark.asyncio
async def test_executor_supports_replay_status_cancellation_and_call_order(
    make_cancellation: Callable[[], CancellationResult],
    make_context: Callable[..., OperationContext],
    make_frozen_run: Callable[..., FrozenRunSpec],
    make_submission: Callable[[], SubmissionResult],
) -> None:
    """Executor fake models one submit effect, bounded status, and explicit cancellation."""
    run = make_frozen_run()
    submission = make_submission()
    cancellation = make_cancellation()
    fake = ScriptedExperimentExecutor(
        submissions={"run_1": submission},
        cancellations={"job_1": cancellation},
        script={
            "executor.submit": [DelegateStep()],
            "executor.get_status": [CancelledStep()],
            "executor.cancel": [DelegateStep()],
        },
    )

    submitted = await fake.submit(run, ctx=make_context(idempotency_key=None))
    replay = await fake.submit(run, ctx=make_context(idempotency_key=None))
    assert submitted == submission
    assert replay == submission
    assert fake.submit_effect_count == 1
    with pytest.raises(OperationCancelledError):
        await fake.get_status("job_1", ctx=make_context(idempotency_key=None))
    result = await fake.cancel("job_1", ctx=make_context(idempotency_key="cancel_1"))
    assert result == cancellation
    fake.assert_call_order(
        "executor.submit",
        "executor.submit",
        "executor.get_status",
        "executor.cancel",
    )


@pytest.mark.contract
@pytest.mark.asyncio
async def test_tracker_replays_structured_evidence_without_mlflow(
    make_artifact_ref: Callable[..., ArtifactRef],
    make_context: Callable[..., OperationContext],
    make_manifest: Callable[[], RunManifest],
    make_tracker_ref: Callable[[], TrackerRunRef],
) -> None:
    """Tracker fake captures finite metrics, artifacts, and final status deterministically."""
    manifest = make_manifest()
    tracker_ref = make_tracker_ref()
    fake = InMemoryExperimentTracker(
        runs={"run_1": tracker_ref},
        script={
            "tracker.create_run": [DelegateStep()],
            "tracker.log_metrics": [DelegateStep()],
            "tracker.log_artifact_refs": [DelegateStep()],
            "tracker.finalize": [DelegateStep()],
        },
    )
    created = await fake.create_run(manifest, ctx=make_context(idempotency_key="tracker_create"))
    replay = await fake.create_run(manifest, ctx=make_context(idempotency_key="tracker_create"))
    assert created == tracker_ref
    assert replay == tracker_ref
    metric = MetricPoint(
        schema_version="1",
        name="macro_f1",
        value=0.5,
        split="test",
        recorded_at=TIMESTAMP,
    )
    await fake.log_metrics(created, [metric], ctx=make_context(idempotency_key="tracker_metrics"))
    await fake.log_artifact_refs(
        created,
        [make_artifact_ref(artifact_id="metric_artifact")],
        ctx=make_context(idempotency_key="tracker_artifacts"),
    )
    await fake.finalize(
        created,
        RunStatus.SUCCEEDED,
        ctx=make_context(idempotency_key="tracker_final"),
    )
    assert fake.metrics[created.tracker_run_id] == [metric]
    assert fake.artifacts[created.tracker_run_id][0].artifact_id == "metric_artifact"
    assert fake.final_statuses[created.tracker_run_id] is RunStatus.SUCCEEDED


@pytest.mark.contract
@pytest.mark.asyncio
async def test_structured_llm_is_repeatable_schema_validated_and_secret_safe(
    make_context: Callable[..., OperationContext],
) -> None:
    """LLM fake validates Pydantic output and redacts secret-like fake call inputs."""
    request = StructuredGenerationRequest[Proposal](
        schema_version="1",
        task_name="candidate_ranking",
        prompt_template_id="candidate/v1",
        prompt_version="1",
        response_schema=Proposal,
        facts={"topic": "defect"},
        budget_class="ranking",
    )
    fake = ScriptedStructuredLLM(
        outputs={"candidate_ranking": {"schema_version": "1", "recommendation": "keep"}},
        usage=GenerationUsage(schema_version="1", input_tokens=2, output_tokens=3, total_tokens=5),
        script={"llm.generate": [DelegateStep(), DelegateStep()]},
    )
    first = await fake.generate(request, ctx=make_context(idempotency_key=None))
    second = await fake.generate(request, ctx=make_context(idempotency_key=None))
    assert first.value.recommendation == "keep"
    assert second == first
    assert first.usage.total_tokens == 5

    schema_failure = ScriptedStructuredLLM(
        outputs={"candidate_ranking": {"schema_version": "1", "recommendation": 7}},
        script={"llm.generate": [DelegateStep()]},
    )
    with pytest.raises(StructuredOutputValidationError) as malformed:
        await schema_failure.generate(request, ctx=make_context(idempotency_key=None))
    assert malformed.value.failure.code == "LLM_SCHEMA_VALIDATION_FAILED"
    assert malformed.value.failure.retryable is True

    secret_request = request.model_copy(update={"facts": {"api_key": "never-log-this"}})
    secret_failure = ScriptedStructuredLLM(
        script={
            "llm.generate": [
                FailureStep(
                    make_failure(
                        code="LLM_PROVIDER_UNAVAILABLE",
                        category="PROVIDER",
                        message="Provider unavailable.",
                        retryable=True,
                        ctx=None,
                    )
                )
            ]
        }
    )
    with pytest.raises(LLMError) as raised:
        await secret_failure.generate(secret_request, ctx=make_context(idempotency_key=None))
    assert "never-log-this" not in repr(secret_failure.calls[0].payload)
    assert "never-log-this" not in str(raised.value)


@pytest.mark.contract
@pytest.mark.asyncio
async def test_unit_of_work_fake_requires_explicit_lifecycle_and_commits(
    make_context: Callable[..., OperationContext],
) -> None:
    """The persistence fake supports deterministic context-manager and commit behavior."""
    uow = InMemoryUnitOfWork(
        lifecycle_script={
            "uow.enter": [DelegateStep()],
            "uow.commit": [DelegateStep()],
        },
        context=make_context(idempotency_key=None),
    )
    async with uow as active:
        await active.commit()
    assert uow.committed is True
    assert uow.rolled_back is False
    assert uow.calls == ["uow.enter", "uow.commit", "uow.exit"]
