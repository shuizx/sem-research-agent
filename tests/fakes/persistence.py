"""Deterministic staged in-memory persistence fakes for the foundation UnitOfWork boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TypeVar

from pydantic import BaseModel

from vision_research_ops.domain import (
    AdaptationPlan,
    Approval,
    ExperimentSpec,
    PaperCandidate,
    RepositorySnapshot,
    ResearchRequest,
)
from vision_research_ops.ports import (
    AuditEvent,
    OperationCancelledError,
    OperationContext,
    OperationTimeoutError,
    PersistenceError,
    PortError,
    make_failure,
)

from .script import (
    CancelledStep,
    DelegateStep,
    FailureStep,
    IdempotencyLedger,
    ReturnStep,
    ScriptedPort,
    ScriptStep,
    TimeoutStep,
    _ScriptState,
    require_scripted_instance,
    require_scripted_none,
)

TEntity = TypeVar("TEntity", bound=BaseModel)


class _RepositoryWriteState(Enum):
    """Private write gate used while a repository is owned by a fake UoW."""

    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    TERMINAL = "TERMINAL"


class _UnitOfWorkState(Enum):
    """One-shot transaction lifecycle for the in-memory persistence fake."""

    NEW = "NEW"
    ACTIVE = "ACTIVE"
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"
    CLOSED = "CLOSED"


@dataclass
class _EntityState[TEntity: BaseModel]:
    """Committed or staged entity/revision/idempotency state for one repository fake."""

    entities: dict[str, TEntity]
    revisions: dict[str, int]
    ledger: IdempotencyLedger[TEntity]


@dataclass
class _EntityBacking[TEntity: BaseModel]:
    """Committed entity state shared by fresh, session-bound repository facades."""

    state: _EntityState[TEntity]


class ScriptedEntityRepository[TEntity: BaseModel](ScriptedPort):
    """Conditional-update repository fake with staged UnitOfWork-aware state."""

    def __init__(
        self,
        entity_type: type[TEntity],
        *,
        entities: Mapping[str, TEntity] | None = None,
        revisions: Mapping[str, int] | None = None,
        script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        clock: Callable[[], datetime] | None = None,
        _backing: _EntityBacking[TEntity] | None = None,
        _script_state: _ScriptState | None = None,
    ) -> None:
        super().__init__(
            port_name="EntityRepository",
            supported_operations=("persistence.get", "persistence.save"),
            script=script,
            clock=clock,
            _script_state=_script_state,
        )
        self._entity_type = entity_type
        self._backing = _backing or _EntityBacking(
            state=_EntityState(
                entities=dict(entities or {}),
                revisions=dict(revisions or {}),
                ledger=IdempotencyLedger(),
            )
        )
        self._staged: _EntityState[TEntity] | None = None
        self._write_state = _RepositoryWriteState.INACTIVE

    async def get(self, entity_id: str, *, ctx: OperationContext) -> TEntity | None:
        """Read one seeded entity through an explicit scripted outcome."""
        operation = "persistence.get"
        payload = {"entity_id": entity_id}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )

        def default() -> TEntity | None:
            return self._state.entities.get(entity_id)

        def validate_return(value: object) -> TEntity | None:
            if value is None:
                return None
            return require_scripted_instance(
                value,
                self._entity_type,
                operation=operation,
                ctx=ctx,
                error_type=PersistenceError,
            )

        return await self._consume(
            operation=operation,
            payload=payload,
            ctx=ctx,
            default=default,
            validate_return=validate_return,
            error_type=PersistenceError,
        )

    async def save(
        self,
        entity_id: str,
        entity: TEntity,
        *,
        expected_revision: int | None,
        ctx: OperationContext,
    ) -> TEntity:
        """Stage or persist one entity only when its expected revision matches."""
        operation = "persistence.save"
        self._ensure_writable(ctx)
        payload = {
            "entity_id": entity_id,
            "entity": entity,
            "expected_revision": expected_revision,
        }
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )
        state = self._state

        async def execute() -> TEntity:
            def default() -> TEntity:
                current = state.revisions.get(entity_id)
                if expected_revision is not None and current != expected_revision:
                    raise PersistenceError(
                        make_failure(
                            code="PERSISTENCE_REVISION_CONFLICT",
                            category="REVISION_CONFLICT",
                            message=(
                                "The stored entity revision differs from the expected revision."
                            ),
                            retryable=False,
                            ctx=ctx,
                            details={
                                "expected_revision": expected_revision,
                                "actual_revision": current,
                            },
                        )
                    )
                state.entities[entity_id] = entity
                state.revisions[entity_id] = 1 if current is None else current + 1
                return entity

            return await self._consume(
                operation=operation,
                payload=payload,
                ctx=ctx,
                default=default,
                validate_return=lambda value: require_scripted_instance(
                    value,
                    self._entity_type,
                    operation=operation,
                    ctx=ctx,
                    error_type=PersistenceError,
                ),
                error_type=PersistenceError,
            )

        return await state.ledger.replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )

    def begin_transaction(self) -> None:
        """Create a private staged copy; the backing state remains untouched until commit."""
        if self._staged is not None or self._write_state is not _RepositoryWriteState.INACTIVE:
            raise RuntimeError("repository already has an active fake transaction")
        self._staged = _EntityState(
            entities=dict(self._backing.state.entities),
            revisions=dict(self._backing.state.revisions),
            ledger=self._backing.state.ledger._clone(),
        )
        self._write_state = _RepositoryWriteState.ACTIVE

    def new_session_facade(self) -> ScriptedEntityRepository[TEntity]:
        """Return a fresh non-reusable facade over this repository's committed backing state."""
        return type(self)(
            self._entity_type,
            _backing=self._backing,
            _script_state=self._script_state,
            clock=self._clock,
        )

    def commit_transaction(self) -> None:
        """Publish the completed staged entity, revision, and idempotency state."""
        if self._staged is None:
            raise RuntimeError("repository has no active fake transaction")
        self._backing.state = self._staged
        self._staged = None
        self._write_state = _RepositoryWriteState.TERMINAL

    def rollback_transaction(self) -> None:
        """Discard staged entity, revision, and idempotency state without touching backing."""
        self._staged = None
        self._write_state = _RepositoryWriteState.TERMINAL

    def backing_snapshot(self) -> _EntityState[TEntity]:
        """Return the unmodified committed state for atomic fake-UoW recovery only."""
        return self._backing.state

    def restore_backing(self, backing: _EntityState[TEntity]) -> None:
        """Restore a previous backing reference after a failed multi-repository commit."""
        self._backing.state = backing
        self._staged = None
        self._write_state = _RepositoryWriteState.TERMINAL

    @property
    def has_active_transaction(self) -> bool:
        """Return whether this fake repository currently owns staged transaction state."""
        return self._staged is not None and self._write_state is _RepositoryWriteState.ACTIVE

    @property
    def _state(self) -> _EntityState[TEntity]:
        """Return the active staged state when present, otherwise the committed backing state."""
        return self._backing.state if self._staged is None else self._staged

    def _ensure_writable(self, ctx: OperationContext) -> None:
        """Reject writes after a UoW terminal state instead of mutating backing storage."""
        if self._write_state is _RepositoryWriteState.ACTIVE:
            return
        raise PersistenceError(
            make_failure(
                code="PERSISTENCE_UNIT_OF_WORK_NOT_ACTIVE",
                category="LIFECYCLE",
                message="The fake UnitOfWork no longer permits repository writes.",
                retryable=False,
                ctx=ctx,
            )
        )

    def revision_of(self, entity_id: str) -> int | None:
        """Return the revision visible to the active transaction view."""
        return self._state.revisions.get(entity_id)

    def backing_revision_of(self, entity_id: str) -> int | None:
        """Return the revision durably visible outside a staged fake transaction."""
        return self._backing.state.revisions.get(entity_id)

    def backing_entity(self, entity_id: str) -> TEntity | None:
        """Return committed backing state for rollback and persistence regression assertions."""
        return self._backing.state.entities.get(entity_id)

    def backing_effect_count(self, operation: str = "persistence.save") -> int:
        """Return committed idempotency effects for rollback regression assertions."""
        return self._backing.state.ledger.effect_counts[operation]


@dataclass
class _AuditState:
    """Committed or staged append-only events and their replay state."""

    events: list[AuditEvent]
    ledger: IdempotencyLedger[None]


@dataclass
class _AuditBacking:
    """Committed audit state shared by fresh, session-bound audit facades."""

    state: _AuditState


class ScriptedAuditRepository(ScriptedPort):
    """Append-only audit fake whose events and replay state stage with a UnitOfWork."""

    def __init__(
        self,
        *,
        script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        clock: Callable[[], datetime] | None = None,
        _backing: _AuditBacking | None = None,
        _script_state: _ScriptState | None = None,
    ) -> None:
        super().__init__(
            port_name="AuditRepository",
            supported_operations=("persistence.audit.append",),
            script=script,
            clock=clock,
            _script_state=_script_state,
        )
        self._backing = _backing or _AuditBacking(
            state=_AuditState(events=[], ledger=IdempotencyLedger())
        )
        self._staged: _AuditState | None = None
        self._write_state = _RepositoryWriteState.INACTIVE

    @property
    def events(self) -> list[AuditEvent]:
        """Expose events visible within the current fake transaction view."""
        return self._state.events

    @property
    def backing_events(self) -> list[AuditEvent]:
        """Expose committed events for rollback regression assertions."""
        return self._backing.state.events

    async def append(self, event: AuditEvent, *, ctx: OperationContext) -> None:
        """Append one immutable audit event exactly once per idempotency key."""
        operation = "persistence.audit.append"
        self._ensure_writable(ctx)
        payload = {"event": event}
        self._prepare(
            operation=operation,
            supported_operation=operation,
            payload=payload,
            ctx=ctx,
        )
        state = self._state

        async def execute() -> None:
            def default() -> None:
                state.events.append(event)

            return await self._consume(
                operation=operation,
                payload=payload,
                ctx=ctx,
                default=default,
                validate_return=lambda value: require_scripted_none(
                    value,
                    operation=operation,
                    ctx=ctx,
                    error_type=PersistenceError,
                ),
                error_type=PersistenceError,
            )

        await state.ledger.replay_or_execute(
            operation=operation,
            idempotency_key=ctx.idempotency_key,
            payload=payload,
            ctx=ctx,
            execute=execute,
        )

    def begin_transaction(self) -> None:
        """Create a private staged event/idempotency copy for one UnitOfWork."""
        if self._staged is not None or self._write_state is not _RepositoryWriteState.INACTIVE:
            raise RuntimeError("audit repository already has an active fake transaction")
        self._staged = _AuditState(
            events=list(self._backing.state.events),
            ledger=self._backing.state.ledger._clone(),
        )
        self._write_state = _RepositoryWriteState.ACTIVE

    def new_session_facade(self) -> ScriptedAuditRepository:
        """Return a fresh non-reusable facade over this audit repository's backing state."""
        return type(self)(
            _backing=self._backing,
            _script_state=self._script_state,
            clock=self._clock,
        )

    def commit_transaction(self) -> None:
        """Publish staged events and replay state after an explicit fake commit."""
        if self._staged is None:
            raise RuntimeError("audit repository has no active fake transaction")
        self._backing.state = self._staged
        self._staged = None
        self._write_state = _RepositoryWriteState.TERMINAL

    def rollback_transaction(self) -> None:
        """Discard staged audit and replay state without modifying committed evidence."""
        self._staged = None
        self._write_state = _RepositoryWriteState.TERMINAL

    def backing_snapshot(self) -> _AuditState:
        """Return committed audit state for atomic fake-UoW recovery only."""
        return self._backing.state

    def restore_backing(self, backing: _AuditState) -> None:
        """Restore committed audit state after a failed multi-repository publish."""
        self._backing.state = backing
        self._staged = None
        self._write_state = _RepositoryWriteState.TERMINAL

    @property
    def has_active_transaction(self) -> bool:
        """Return whether this fake audit repository currently owns staged transaction state."""
        return self._staged is not None and self._write_state is _RepositoryWriteState.ACTIVE

    @property
    def _state(self) -> _AuditState:
        """Return the active staged audit state or committed backing state."""
        return self._backing.state if self._staged is None else self._staged

    def _ensure_writable(self, ctx: OperationContext) -> None:
        """Reject append after a UoW terminal state instead of changing backing evidence."""
        if self._write_state is _RepositoryWriteState.ACTIVE:
            return
        raise PersistenceError(
            make_failure(
                code="PERSISTENCE_UNIT_OF_WORK_NOT_ACTIVE",
                category="LIFECYCLE",
                message="The fake UnitOfWork no longer permits repository writes.",
                retryable=False,
                ctx=ctx,
            )
        )

    def backing_effect_count(self, operation: str = "persistence.audit.append") -> int:
        """Return committed audit idempotency effects for rollback assertions."""
        return self._backing.state.ledger.effect_counts[operation]


class InMemoryOperationRepository:
    """Deliberately behavior-free fake for the deferred persistence boundary operation contract."""


class InMemoryUnitOfWork:
    """One-shot staged transaction fake; it does not claim database isolation levels."""

    def __init__(
        self,
        *,
        research_requests: ScriptedEntityRepository[ResearchRequest] | None = None,
        papers: ScriptedEntityRepository[PaperCandidate] | None = None,
        repositories: ScriptedEntityRepository[RepositorySnapshot] | None = None,
        adaptations: ScriptedEntityRepository[AdaptationPlan] | None = None,
        experiments: ScriptedEntityRepository[ExperimentSpec] | None = None,
        approvals: ScriptedEntityRepository[Approval] | None = None,
        audits: ScriptedAuditRepository | None = None,
        operations: InMemoryOperationRepository | None = None,
        lifecycle_script: Mapping[str, Iterable[ScriptStep[object]]] | None = None,
        context: OperationContext | None = None,
    ) -> None:
        research_request_backing = research_requests or ScriptedEntityRepository(ResearchRequest)
        paper_backing = papers or ScriptedEntityRepository(PaperCandidate)
        repository_backing = repositories or ScriptedEntityRepository(RepositorySnapshot)
        adaptation_backing = adaptations or ScriptedEntityRepository(AdaptationPlan)
        experiment_backing = experiments or ScriptedEntityRepository(ExperimentSpec)
        approval_backing = approvals or ScriptedEntityRepository(Approval)
        audit_backing = audits or ScriptedAuditRepository()
        self.research_requests = research_request_backing.new_session_facade()
        self.papers = paper_backing.new_session_facade()
        self.repositories = repository_backing.new_session_facade()
        self.adaptations = adaptation_backing.new_session_facade()
        self.experiments = experiment_backing.new_session_facade()
        self.approvals = approval_backing.new_session_facade()
        self.audits = audit_backing.new_session_facade()
        self.operations = operations or InMemoryOperationRepository()
        self._transactional_repositories = (
            self.research_requests,
            self.papers,
            self.repositories,
            self.adaptations,
            self.experiments,
            self.approvals,
            self.audits,
        )
        self._lifecycle_script = {
            operation: list(steps) for operation, steps in (lifecycle_script or {}).items()
        }
        self._context = context or OperationContext(
            schema_version="1",
            correlation_id="fake_uow_correlation",
            workflow_id="fake_uow_workflow",
            actor_id="fake_uow_actor",
            sensitivity="INTERNAL",
        )
        self.calls: list[str] = []
        self.committed = False
        self.rolled_back = False
        self._state = _UnitOfWorkState.NEW

    @property
    def lifecycle_state(self) -> str:
        """Expose the deterministic terminal-safe lifecycle label for fake assertions."""
        return self._state.value

    async def __aenter__(self) -> InMemoryUnitOfWork:
        """Enter one private staging scope; a terminal instance is never reusable."""
        if self._state is not _UnitOfWorkState.NEW:
            raise self._lifecycle_error(
                code="PERSISTENCE_UNIT_OF_WORK_REUSED",
                message="The fake UnitOfWork instance cannot be entered more than once.",
            )
        try:
            self._consume_lifecycle("uow.enter")
            for repository in self._transactional_repositories:
                repository.begin_transaction()
        except BaseException as error:
            self._terminate_failed_transaction()
            self._raise_safe_transaction_error(
                error,
                code="PERSISTENCE_UNIT_OF_WORK_ENTER_FAILED",
                message="The fake UnitOfWork could not enter a transaction.",
            )
        self._state = _UnitOfWorkState.ACTIVE
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Close once, preserving only explicitly committed backing state."""
        del exc_type, exc, traceback
        try:
            if self._state is _UnitOfWorkState.ACTIVE:
                await self.rollback()
        finally:
            self._discard_staged_state()
            self._state = _UnitOfWorkState.CLOSED
            self.calls.append("uow.exit")

    async def commit(self) -> None:
        """Publish all staged repositories atomically, then permanently close writes."""
        self._ensure_active("uow.commit")
        backups = [
            (repository, repository.backing_snapshot())
            for repository in self._transactional_repositories
        ]
        try:
            self._consume_lifecycle("uow.commit")
            self._ensure_all_staged()
            for repository in self._transactional_repositories:
                repository.commit_transaction()
        except BaseException as error:
            for repository, backing in backups:
                repository.restore_backing(backing)
            self._terminate_failed_transaction()
            self._raise_safe_transaction_error(
                error,
                code="PERSISTENCE_UNIT_OF_WORK_COMMIT_FAILED",
                message="The fake UnitOfWork could not commit its staged changes.",
            )
        self.committed = True
        self._state = _UnitOfWorkState.COMMITTED

    async def rollback(self) -> None:
        """Discard every staged state even when the scripted rollback outcome fails."""
        self._ensure_active("uow.rollback")
        try:
            self._consume_lifecycle("uow.rollback")
        except BaseException as error:
            self._terminate_failed_transaction()
            self._raise_safe_transaction_error(
                error,
                code="PERSISTENCE_UNIT_OF_WORK_ROLLBACK_FAILED",
                message="The fake UnitOfWork could not complete rollback.",
            )
        self._discard_staged_state()
        self.rolled_back = True
        self._state = _UnitOfWorkState.ROLLED_BACK

    def _ensure_active(self, operation: str) -> None:
        """Reject commit, rollback, and staged writes outside the active lifecycle state."""
        if self._state is _UnitOfWorkState.ACTIVE:
            return
        raise self._lifecycle_error(
            code="PERSISTENCE_UNIT_OF_WORK_NOT_ACTIVE",
            message="The fake UnitOfWork is not inside an active transaction scope.",
            details={"operation": operation, "state": self._state.value},
        )

    def _ensure_all_staged(self) -> None:
        """Fail before publishing if a repository lost its private staged state."""
        if all(
            repository.has_active_transaction for repository in self._transactional_repositories
        ):
            return
        raise self._lifecycle_error(
            code="PERSISTENCE_UNIT_OF_WORK_TRANSACTION_STATE_INVALID",
            message="The fake UnitOfWork lost staged transaction state before commit.",
        )

    def _discard_staged_state(self) -> None:
        """Clear all entity, revision, audit, and ledger staging without touching backing state."""
        for repository in self._transactional_repositories:
            repository.rollback_transaction()

    def _terminate_failed_transaction(self) -> None:
        """Fail closed and clear staging after any enter, commit, or rollback failure."""
        try:
            self._discard_staged_state()
        finally:
            self.committed = False
            self.rolled_back = True
            self._state = _UnitOfWorkState.FAILED

    def _raise_safe_transaction_error(
        self,
        error: BaseException,
        *,
        code: str,
        message: str,
    ) -> None:
        """Preserve typed safe errors and replace all raw transaction exceptions."""
        if isinstance(error, (PortError, KeyboardInterrupt, SystemExit)):
            raise error
        raise self._lifecycle_error(code=code, message=message) from None

    def _lifecycle_error(
        self,
        *,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> PersistenceError:
        """Build a deterministic de-sensitized lifecycle failure."""
        return PersistenceError(
            make_failure(
                code=code,
                category="LIFECYCLE",
                message=message,
                retryable=False,
                ctx=self._context,
                details={} if details is None else details,
            )
        )

    def _consume_lifecycle(self, operation: str) -> None:
        """Consume a deterministic lifecycle outcome without hidden success fallback."""
        self.calls.append(operation)
        try:
            step = self._lifecycle_script[operation].pop(0)
        except (KeyError, IndexError) as exc:
            raise self._lifecycle_error(
                code="PERSISTENCE_LIFECYCLE_NOT_SCRIPTED",
                message="No deterministic UnitOfWork lifecycle outcome is configured.",
                details={"operation": operation},
            ) from exc
        if isinstance(step, FailureStep):
            failure = step.failure
            if failure.correlation_id != self._context.correlation_id:
                failure = failure.model_copy(
                    update={"correlation_id": self._context.correlation_id}
                )
            raise PersistenceError(failure)
        if isinstance(step, TimeoutStep):
            raise OperationTimeoutError(operation, self._context)
        if isinstance(step, CancelledStep):
            raise OperationCancelledError(operation, self._context)
        if isinstance(step, ReturnStep):
            require_scripted_none(
                step.value,
                operation=operation,
                ctx=self._context,
                error_type=PersistenceError,
            )
            return
        if isinstance(step, DelegateStep):
            return
        raise AssertionError("unknown UnitOfWork lifecycle script step")


__all__ = [
    "InMemoryOperationRepository",
    "InMemoryUnitOfWork",
    "ScriptedAuditRepository",
    "ScriptedEntityRepository",
]
