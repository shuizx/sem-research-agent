"""Business persistence boundary and UnitOfWork protocol.

deterministic port contract fixes the dependency boundary and conditional-update shape only.  The
database schema, operation-record layout, migrations, and concrete repository
methods remain owned by persistence boundary.
"""

from __future__ import annotations

from typing import Protocol, Self, TypeVar, runtime_checkable

from vision_research_ops.domain import (
    AdaptationPlan,
    Approval,
    ExperimentSpec,
    PaperCandidate,
    RepositorySnapshot,
    ResearchRequest,
)

from .common import AuditEvent, OperationContext

TEntity = TypeVar("TEntity")


@runtime_checkable
class EntityRepository(Protocol[TEntity]):
    """Minimal typed repository shape with explicit conditional-update support."""

    async def get(self, entity_id: str, *, ctx: OperationContext) -> TEntity | None:
        """Load one entity by its opaque ID without leaking storage details."""

    async def save(
        self,
        entity_id: str,
        entity: TEntity,
        *,
        expected_revision: int | None,
        ctx: OperationContext,
    ) -> TEntity:
        """Persist an entity subject to the expected revision when supplied."""


@runtime_checkable
class AuditRepository(Protocol):
    """Append small audit events; large evidence belongs in the artifact store."""

    async def append(self, event: AuditEvent, *, ctx: OperationContext) -> None:
        """Append one immutable audit event."""


@runtime_checkable
class ResearchRequestRepository(EntityRepository[ResearchRequest], Protocol):
    """Typed repository boundary for versioned research requests."""


@runtime_checkable
class PaperRepository(EntityRepository[PaperCandidate], Protocol):
    """Typed repository boundary for normalized paper candidates."""


@runtime_checkable
class RepositorySnapshotRepository(EntityRepository[RepositorySnapshot], Protocol):
    """Typed repository boundary for fixed repository snapshots."""


@runtime_checkable
class AdaptationRepository(EntityRepository[AdaptationPlan], Protocol):
    """Typed repository boundary for structured adaptation plans."""


@runtime_checkable
class ExperimentRepository(EntityRepository[ExperimentSpec], Protocol):
    """Typed repository boundary for frozen experiment specifications."""


@runtime_checkable
class ApprovalRepository(EntityRepository[Approval], Protocol):
    """Typed repository boundary for audit-bound human approvals."""


class OperationRepository(Protocol):
    """Deferred persistence boundary boundary; its records and operations are not yet specified."""


@runtime_checkable
class UnitOfWork(Protocol):
    """Transaction boundary exposing typed business repositories and audit storage."""

    research_requests: ResearchRequestRepository
    papers: PaperRepository
    repositories: RepositorySnapshotRepository
    adaptations: AdaptationRepository
    experiments: ExperimentRepository
    approvals: ApprovalRepository
    audits: AuditRepository
    operations: OperationRepository

    async def __aenter__(self) -> Self:
        """Enter the transaction scope."""

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Close the transaction scope, rolling back all uncommitted work."""

    async def commit(self) -> None:
        """Atomically commit pending business and audit changes."""

    async def rollback(self) -> None:
        """Discard pending changes without replacing existing evidence."""


__all__ = [
    "AdaptationRepository",
    "ApprovalRepository",
    "AuditRepository",
    "EntityRepository",
    "ExperimentRepository",
    "OperationRepository",
    "PaperRepository",
    "RepositorySnapshotRepository",
    "ResearchRequestRepository",
    "UnitOfWork",
]
