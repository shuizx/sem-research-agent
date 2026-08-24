"""Reusable deterministic scripting, recording, and idempotency support for fakes."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeVar, cast

from vision_research_ops.domain import JsonObject, StructuredFailure
from vision_research_ops.ports import (
    CapabilityNotSupportedError,
    FakeScriptExhaustedError,
    IdempotencyConflictError,
    IdempotencyKeyRequiredError,
    OperationCancelledError,
    OperationContext,
    OperationTimeoutError,
    PortError,
    make_failure,
)
from vision_research_ops.ports._security import canonical_json_hash, redact_json_object
from vision_research_ops.ports.base import AsyncBinaryReader

T = TypeVar("T")
_LOOP_BOUND_RESULT_TYPES = (
    asyncio.Barrier,
    asyncio.BoundedSemaphore,
    asyncio.Condition,
    asyncio.Event,
    asyncio.Lock,
    asyncio.Queue,
    asyncio.Semaphore,
)


@dataclass(frozen=True)
class DelegateStep:
    """Explicitly delegate a scripted call to the fake's deterministic implementation."""


@dataclass(frozen=True)
class ReturnStep[T]:
    """Explicitly return a predetermined value without invoking fake storage logic."""

    value: T


@dataclass(frozen=True)
class FailureStep:
    """Explicitly raise a supplied stable structured failure."""

    failure: StructuredFailure


@dataclass(frozen=True)
class TimeoutStep:
    """Explicitly model a retryable timeout without using wall-clock sleeps."""


@dataclass(frozen=True)
class CancelledStep:
    """Explicitly model cancellation observed at a port boundary."""


type ScriptStep[T] = DelegateStep | ReturnStep[T] | FailureStep | TimeoutStep | CancelledStep
type ReturnValidator[T] = Callable[[object], T]


@dataclass(frozen=True)
class CallRecord:
    """De-sensitized deterministic record of a fake-port method invocation."""

    index: int
    operation: str
    correlation_id: str
    workflow_id: str
    idempotency_key: str | None
    payload: JsonObject


class FrozenClock:
    """Manually controlled clock used to test deadlines without sleeping."""

    def __init__(self, now: datetime) -> None:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("FrozenClock requires an aware datetime")
        self._now = now.astimezone(UTC)

    def __call__(self) -> datetime:
        """Return the currently configured deterministic instant."""
        return self._now

    def set(self, now: datetime) -> None:
        """Set the deterministic instant used by later fake calls."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("FrozenClock requires an aware datetime")
        self._now = now.astimezone(UTC)


def safe_payload(payload: Mapping[str, object]) -> JsonObject:
    """Return a recursively de-sensitized payload suitable for call-record assertions."""
    return redact_json_object(payload)


def _payload_fingerprint(payload: Mapping[str, object]) -> str:
    """Hash complete canonical JSON for private idempotency comparison only."""
    return canonical_json_hash(payload)


def require_scripted_instance[T](
    value: object,
    expected_type: type[T],
    *,
    operation: str,
    ctx: OperationContext,
    error_type: type[PortError],
) -> T:
    """Validate a ``ReturnStep`` value before exposing it as a concrete fake result."""
    if not isinstance(value, expected_type):
        raise error_type(
            make_failure(
                code="TEST_FAKE_SCRIPTED_RETURN_TYPE_INVALID",
                category="TEST_FAKE",
                message="The scripted fake return value does not match the port result type.",
                retryable=False,
                ctx=ctx,
                details={"operation": operation, "expected_type": expected_type.__name__},
            )
        )
    return value


def require_scripted_none(
    value: object,
    *,
    operation: str,
    ctx: OperationContext,
    error_type: type[PortError],
) -> None:
    """Validate a scripted ``None`` result for a side-effect-only fake method."""
    if value is not None:
        raise error_type(
            make_failure(
                code="TEST_FAKE_SCRIPTED_RETURN_TYPE_INVALID",
                category="TEST_FAKE",
                message="The scripted fake return value does not match the port result type.",
                retryable=False,
                ctx=ctx,
                details={"operation": operation, "expected_type": "None"},
            )
        )


class _ValidatingAsyncBinaryReader:
    """Private proxy that enforces bytes at every scripted reader boundary."""

    def __init__(
        self,
        *,
        read_method: Callable[[], Awaitable[object]],
        next_method: Callable[[], Awaitable[object]],
        operation: str,
        ctx: OperationContext,
        error_type: type[PortError],
    ) -> None:
        self._read_method = read_method
        self._next_method = next_method
        self._operation = operation
        self._ctx = ctx
        self._error_type = error_type

    async def read(self) -> bytes:
        """Read and validate the complete artifact payload as exact ``bytes``."""
        try:
            value = await self._read_method()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._raise_runtime_error("ARTIFACT_READER_READ_FAILED")
        return self._require_bytes(value)

    def __aiter__(self) -> _ValidatingAsyncBinaryReader:
        """Return this proxy so a changing raw iterator cannot bypass validation."""
        return self

    async def __anext__(self) -> bytes:
        """Yield and validate one cached raw iterator chunk."""
        try:
            value = await self._next_method()
        except StopAsyncIteration:
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            self._raise_runtime_error("ARTIFACT_READER_NEXT_FAILED")
        return self._require_bytes(value)

    def _require_bytes(self, value: object) -> bytes:
        """Reject non-bytes reader content without exposing its value or repr."""
        if type(value) is bytes:
            return value
        self._raise_runtime_error("ARTIFACT_READER_BYTES_INVALID")

    def _raise_runtime_error(self, code: str) -> None:
        """Raise one stable typed reader failure without retaining raw reader data."""
        raise self._error_type(
            make_failure(
                code=code,
                category="ARTIFACT",
                message="The scripted artifact reader returned an invalid binary result.",
                retryable=False,
                ctx=self._ctx,
                details={"operation": self._operation},
            )
        )


def require_scripted_async_binary_reader(
    value: object,
    *,
    operation: str,
    ctx: OperationContext,
    error_type: type[PortError],
) -> AsyncBinaryReader:
    """Return a cached validating proxy for one concrete scripted binary reader."""
    try:
        read_method = _require_zero_argument_method(value, "read", is_async=True)
        aiter_method = _require_zero_argument_method(value, "__aiter__", is_async=False)
        _require_zero_argument_method(value, "__anext__", is_async=True)
        iterator = aiter_method()
        next_method = _require_zero_argument_method(iterator, "__anext__", is_async=True)
    except asyncio.CancelledError:
        raise
    except Exception:
        _raise_reader_return_type_error(operation=operation, ctx=ctx, error_type=error_type)
    return cast(
        AsyncBinaryReader,
        _ValidatingAsyncBinaryReader(
            read_method=cast(Callable[[], Awaitable[object]], read_method),
            next_method=cast(Callable[[], Awaitable[object]], next_method),
            operation=operation,
            ctx=ctx,
            error_type=error_type,
        ),
    )


def _require_zero_argument_method(
    value: object,
    method_name: str,
    *,
    is_async: bool,
) -> Callable[..., object]:
    """Return one safe bound method or raise without evaluating an object's repr."""
    method = getattr(value, method_name)
    if not callable(method) or inspect.iscoroutinefunction(method) is not is_async:
        raise TypeError("scripted reader method shape is invalid")
    try:
        if inspect.signature(method).parameters:
            raise TypeError("scripted reader method has parameters")
    except (TypeError, ValueError):
        raise TypeError("scripted reader method signature is invalid") from None
    return method


def _raise_reader_return_type_error(
    *,
    operation: str,
    ctx: OperationContext,
    error_type: type[PortError],
) -> None:
    """Raise one generic safe failure without retaining a malformed object's repr."""
    raise error_type(
        make_failure(
            code="TEST_FAKE_SCRIPTED_RETURN_TYPE_INVALID",
            category="TEST_FAKE",
            message="The scripted fake return value does not match the port result type.",
            retryable=False,
            ctx=ctx,
            details={"operation": operation, "expected_type": "AsyncBinaryReader"},
        )
    )


@dataclass
class _ScriptReservation:
    """Private pre-stream reservation of one successful scripted outcome."""

    step: DelegateStep | ReturnStep[object]
    holders: int = 1
    execution_started: bool = False


@dataclass
class _ScriptState:
    """Shared mutable script state for facades that use one fake backing store."""

    script: dict[str, deque[ScriptStep[object]]]
    calls: list[CallRecord]
    guard: threading.RLock
    reservations: dict[tuple[str, str], _ScriptReservation]


class ScriptedPort:
    """Fail-closed scripted fake base with deterministic records and test-only support flags."""

    def __init__(
        self,
        *,
        port_name: str,
        supported_operations: Iterable[str],
        script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        clock: Callable[[], datetime] | None = None,
        _script_state: _ScriptState | None = None,
    ) -> None:
        self.port_name = port_name
        self._supported_operations = frozenset(supported_operations)
        self._script_state = _script_state or _ScriptState(
            script={operation: deque(steps) for operation, steps in (script or {}).items()},
            calls=[],
            guard=threading.RLock(),
            reservations={},
        )
        self._script = self._script_state.script
        self._clock = clock or FrozenClock(datetime(2026, 1, 1, tzinfo=UTC))
        self.calls = self._script_state.calls

    def append_script(self, operation: str, *steps: ScriptStep[object]) -> None:
        """Append explicit outcomes for future calls to one named operation."""
        with self._script_state.guard:
            self._script.setdefault(operation, deque()).extend(steps)

    def call_count(self, operation: str) -> int:
        """Return how many times the named operation was invoked, including replays."""
        with self._script_state.guard:
            return sum(record.operation == operation for record in self.calls)

    def assert_call_order(self, *operations: str) -> None:
        """Assert the complete observed operation order for deterministic graph tests."""
        with self._script_state.guard:
            actual = [record.operation for record in self.calls]
        if actual != list(operations):
            raise AssertionError(f"expected call order {list(operations)!r}, got {actual!r}")

    def _prepare(
        self,
        *,
        operation: str,
        supported_operation: str,
        payload: Mapping[str, object],
        ctx: OperationContext,
    ) -> int:
        safe = safe_payload(payload)
        with self._script_state.guard:
            record = CallRecord(
                index=len(self.calls) + 1,
                operation=operation,
                correlation_id=ctx.correlation_id,
                workflow_id=ctx.workflow_id,
                idempotency_key=ctx.idempotency_key,
                payload=safe,
            )
            self.calls.append(record)
        if supported_operation not in self._supported_operations:
            raise CapabilityNotSupportedError(supported_operation, ctx)
        if ctx.deadline_exceeded(now=self._clock()):
            raise OperationTimeoutError(operation, ctx)
        return record.index

    def _replace_call_payload(self, record_index: int, payload: Mapping[str, object]) -> None:
        """Replace one call observation after a permitted stream has been safely summarized."""
        with self._script_state.guard:
            record = self.calls[record_index - 1]
            self.calls[record_index - 1] = CallRecord(
                index=record.index,
                operation=record.operation,
                correlation_id=record.correlation_id,
                workflow_id=record.workflow_id,
                idempotency_key=record.idempotency_key,
                payload=safe_payload(payload),
            )

    def _peek_step(
        self,
        *,
        operation: str,
        ctx: OperationContext,
    ) -> ScriptStep[object]:
        """Inspect the next scripted outcome under the shared queue lock."""
        with self._script_state.guard:
            try:
                return self._script[operation][0]
            except (KeyError, IndexError) as exc:
                raise FakeScriptExhaustedError(operation, ctx) from exc

    def _take_step(
        self,
        *,
        operation: str,
        ctx: OperationContext,
        error_type: type[PortError] = PortError,
    ) -> ScriptStep[object]:
        """Atomically consume one scripted outcome before a delegated side effect."""
        with self._script_state.guard:
            try:
                step = self._script[operation].popleft()
            except (KeyError, IndexError) as exc:
                raise FakeScriptExhaustedError(operation, ctx) from exc
        self._raise_terminal_step(step=step, operation=operation, ctx=ctx, error_type=error_type)
        return step

    def _reserve_success_step(
        self,
        *,
        operation: str,
        reservation_key: str,
        ctx: OperationContext,
        error_type: type[PortError] = PortError,
    ) -> _ScriptReservation:
        """Atomically reserve one successful step before a stream may be consumed."""
        reservation_id = (operation, reservation_key)
        with self._script_state.guard:
            existing = self._script_state.reservations.get(reservation_id)
            if existing is not None:
                existing.holders += 1
                return existing
            try:
                step = self._script[operation].popleft()
            except (KeyError, IndexError) as exc:
                raise FakeScriptExhaustedError(operation, ctx) from exc
            if isinstance(step, DelegateStep | ReturnStep):
                reservation = _ScriptReservation(step=step)
                self._script_state.reservations[reservation_id] = reservation
                return reservation
        self._raise_terminal_step(step=step, operation=operation, ctx=ctx, error_type=error_type)
        raise AssertionError("terminal script steps always raise")  # pragma: no cover

    def _release_step_reservation(
        self,
        *,
        operation: str,
        reservation_key: str,
        reservation: _ScriptReservation,
    ) -> None:
        """Release an unstarted reservation and restore its step when no holder remains."""
        reservation_id = (operation, reservation_key)
        with self._script_state.guard:
            current = self._script_state.reservations.get(reservation_id)
            if current is not reservation or reservation.execution_started:
                return
            reservation.holders -= 1
            if reservation.holders == 0:
                self._script_state.reservations.pop(reservation_id, None)
                self._script.setdefault(operation, deque()).appendleft(reservation.step)

    async def _consume_reserved_step[T](
        self,
        *,
        operation: str,
        reservation_key: str,
        reservation: _ScriptReservation,
        default: Callable[[], T | Awaitable[T]],
        validate_return: ReturnValidator[T],
    ) -> T:
        """Commit one reservation immediately before its one ledger-governed effect starts."""
        reservation_id = (operation, reservation_key)
        with self._script_state.guard:
            current = self._script_state.reservations.get(reservation_id)
            if current is not reservation or reservation.execution_started:
                raise RuntimeError("script reservation is unavailable for delegated execution")
            reservation.execution_started = True
            self._script_state.reservations.pop(reservation_id, None)
        return await self._consume_step(
            step=reservation.step,
            default=default,
            validate_return=validate_return,
        )

    @staticmethod
    def _raise_terminal_step(
        *,
        step: ScriptStep[object],
        operation: str,
        ctx: OperationContext,
        error_type: type[PortError],
    ) -> None:
        """Translate terminal scripted outcomes into their stable typed boundary errors."""
        if isinstance(step, FailureStep):
            failure = step.failure
            if failure.correlation_id != ctx.correlation_id:
                failure = failure.model_copy(update={"correlation_id": ctx.correlation_id})
            raise error_type(failure)
        if isinstance(step, TimeoutStep):
            raise OperationTimeoutError(operation, ctx)
        if isinstance(step, CancelledStep):
            raise OperationCancelledError(operation, ctx)

    async def _consume_step[T](
        self,
        *,
        step: ScriptStep[object],
        default: Callable[[], T | Awaitable[T]],
        validate_return: ReturnValidator[T],
    ) -> T:
        """Resolve a preflighted delegate or runtime-validated scripted return value."""
        if isinstance(step, ReturnStep):
            return validate_return(step.value)
        if not isinstance(step, DelegateStep):  # pragma: no cover - terminal steps already raise
            raise AssertionError("unknown scripted fake step")
        result = default()
        if isinstance(result, Awaitable):
            result = await result
        return validate_return(result)

    async def _consume[T](
        self,
        *,
        operation: str,
        payload: Mapping[str, object],
        ctx: OperationContext,
        default: Callable[[], T | Awaitable[T]],
        validate_return: ReturnValidator[T],
        error_type: type[PortError] = PortError,
    ) -> T:
        """Consume one configured outcome; unscripted success is never inferred."""
        del payload
        step = self._take_step(operation=operation, ctx=ctx, error_type=error_type)
        return await self._consume_step(
            step=step,
            default=default,
            validate_return=validate_return,
        )


@dataclass(frozen=True)
class _IdempotencyEntry[T]:
    """One successful fake side effect retained for safe replay."""

    payload_fingerprint: str
    result: T


@dataclass(frozen=True)
class _IdempotencyLedgerSnapshot[T]:
    """Private successful-effect copy used only by transactional fake storage."""

    entries: dict[str, _IdempotencyEntry[T]]
    effect_counts: dict[str, int]


@dataclass(frozen=True)
class _InFlight[T]:
    """One shared in-flight execution for a single idempotency key."""

    payload_fingerprint: str
    task: asyncio.Task[T]
    loop: asyncio.AbstractEventLoop


class IdempotencyLedger[T]:
    """Concurrent fake-only successful-effect ledger; failures are deliberately not cached."""

    def __init__(self) -> None:
        self._entries: dict[str, _IdempotencyEntry[T]] = {}
        self._in_flight: dict[str, _InFlight[T]] = {}
        self._guard = threading.RLock()
        self._in_flight_joined = threading.Event()
        self._in_flight_settled = threading.Event()
        self._in_flight_settled.set()
        self.effect_counts: dict[str, int] = defaultdict(int)

    def _snapshot(self) -> _IdempotencyLedgerSnapshot[T]:
        """Copy committed replay state for private fake-transaction staging only."""
        with self._guard:
            if self._in_flight:
                raise RuntimeError("cannot snapshot an idempotency ledger with in-flight effects")
            return _IdempotencyLedgerSnapshot(dict(self._entries), dict(self.effect_counts))

    def _restore(self, snapshot: _IdempotencyLedgerSnapshot[T]) -> None:
        """Restore private fake-transaction state without exposing comparison hashes."""
        with self._guard:
            if self._in_flight:
                raise RuntimeError("cannot restore an idempotency ledger with in-flight effects")
            self._entries = dict(snapshot.entries)
            self.effect_counts = defaultdict(int, snapshot.effect_counts)

    def _clone(self) -> IdempotencyLedger[T]:
        """Create private staged replay state for the persistence fake only."""
        clone = IdempotencyLedger[T]()
        clone._restore(self._snapshot())
        return clone

    async def wait_for_in_flight_join(self) -> None:
        """Wait for a test caller to join an in-flight execution without using sleeps."""
        await asyncio.to_thread(self._in_flight_joined.wait)

    async def wait_for_idle(self) -> None:
        """Wait until all active fake side effects have settled and task errors were consumed."""
        await asyncio.to_thread(self._in_flight_settled.wait)

    @property
    def in_flight_count(self) -> int:
        """Return the number of currently shared side effects for deterministic cleanup checks."""
        with self._guard:
            return len(self._in_flight)

    def has_replay_state(self, idempotency_key: str | None) -> bool:
        """Return whether a key is completed or active without exposing its private digest."""
        if idempotency_key is None:
            return False
        with self._guard:
            return idempotency_key in self._entries or idempotency_key in self._in_flight

    async def replay_or_execute(
        self,
        *,
        operation: str,
        idempotency_key: str | None,
        payload: Mapping[str, object],
        ctx: OperationContext,
        execute: Callable[[], Awaitable[T]],
    ) -> T:
        """Replay, conflict, or share one side effect without globally serializing operations."""
        if idempotency_key is None:
            raise IdempotencyKeyRequiredError(operation, ctx)
        fingerprint = self._fingerprint_or_failure(operation=operation, payload=payload, ctx=ctx)
        current_loop = asyncio.get_running_loop()
        with self._guard:
            existing = self._entries.get(idempotency_key)
            if existing is not None:
                if existing.payload_fingerprint != fingerprint:
                    raise IdempotencyConflictError(operation, ctx)
                return existing.result
            in_flight = self._in_flight.get(idempotency_key)
            if in_flight is not None:
                if in_flight.payload_fingerprint != fingerprint:
                    raise IdempotencyConflictError(operation, ctx)
                if in_flight.loop is not current_loop:
                    raise PortError(
                        make_failure(
                            code="PORT_IDEMPOTENCY_CROSS_LOOP_UNSUPPORTED",
                            category="IDEMPOTENCY",
                            message=(
                                "The idempotency operation is active on a different event loop."
                            ),
                            retryable=True,
                            ctx=ctx,
                            details={"operation": operation},
                        )
                    )
                self._in_flight_joined.set()
                task = in_flight.task
            else:
                self._in_flight_joined.clear()
                self._in_flight_settled.clear()
                task = current_loop.create_task(
                    self._execute_and_record(
                        operation=operation,
                        idempotency_key=idempotency_key,
                        fingerprint=fingerprint,
                        ctx=ctx,
                        execute=execute,
                    )
                )
                task.add_done_callback(self._consume_task_exception)
                task.add_done_callback(self._mark_in_flight_settled)
                self._in_flight[idempotency_key] = _InFlight(
                    payload_fingerprint=fingerprint,
                    task=task,
                    loop=current_loop,
                )
        return await asyncio.shield(task)

    async def _execute_and_record(
        self,
        *,
        operation: str,
        idempotency_key: str,
        fingerprint: str,
        ctx: OperationContext,
        execute: Callable[[], Awaitable[T]],
    ) -> T:
        """Run one side effect and clear its in-flight marker on every terminal outcome."""
        try:
            result = await execute()
            self._ensure_replayable_result(
                operation=operation,
                result=result,
                ctx=ctx,
            )
            with self._guard:
                self._entries[idempotency_key] = _IdempotencyEntry(
                    payload_fingerprint=fingerprint,
                    result=result,
                )
                self.effect_counts[operation] += 1
            return result
        finally:
            with self._guard:
                current_task = asyncio.current_task()
                in_flight = self._in_flight.get(idempotency_key)
                if in_flight is not None and in_flight.task is current_task:
                    self._in_flight.pop(idempotency_key, None)

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[object]) -> None:
        """Consume terminal task exceptions when every shielded waiter was cancelled."""
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    def _mark_in_flight_settled(self, task: asyncio.Task[object]) -> None:
        """Signal safe task settlement only after its exception-consumption callback runs."""
        del task
        with self._guard:
            if not self._in_flight:
                self._in_flight_settled.set()

    @staticmethod
    def _ensure_replayable_result(
        *,
        operation: str,
        result: object,
        ctx: OperationContext,
    ) -> None:
        """Reject loop-bound or one-shot results before successful replay state is written."""
        if not isinstance(
            result, (asyncio.Future, *_LOOP_BOUND_RESULT_TYPES)
        ) and not inspect.isawaitable(result):
            return
        if inspect.iscoroutine(result):
            result.close()
        raise PortError(
            make_failure(
                code="FAKE_IDEMPOTENCY_RESULT_NOT_REPLAYABLE",
                category="TEST_FAKE",
                message="The fake side effect returned a value that cannot be replayed safely.",
                retryable=False,
                ctx=ctx,
                details={"operation": operation},
            )
        )

    @staticmethod
    def _fingerprint_or_failure(
        *,
        operation: str,
        payload: Mapping[str, object],
        ctx: OperationContext,
    ) -> str:
        """Fail closed if a private idempotency comparison cannot be canonicalized."""
        try:
            return _payload_fingerprint(payload)
        except Exception as exc:
            del exc
            raise PortError(
                make_failure(
                    code="PORT_PAYLOAD_CANONICALIZATION_FAILED",
                    category="VALIDATION",
                    message="The port payload cannot be compared safely for idempotency.",
                    retryable=False,
                    ctx=ctx,
                    details={"operation": operation},
                )
            ) from None


__all__ = [
    "CallRecord",
    "CancelledStep",
    "DelegateStep",
    "FailureStep",
    "FrozenClock",
    "IdempotencyLedger",
    "ReturnStep",
    "ScriptStep",
    "ScriptedPort",
    "TimeoutStep",
    "require_scripted_async_binary_reader",
    "require_scripted_instance",
    "require_scripted_none",
    "safe_payload",
]
