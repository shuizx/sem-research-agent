"""Provider-neutral repository resolution and static-inspection ports."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vision_research_ops.domain import ArtifactRef

from .common import (
    OperationContext,
    RepositoryAnalysis,
    RepositoryMetadata,
    RepositoryPolicy,
    RepositoryResolution,
)


@runtime_checkable
class RepositoryProvider(Protocol):
    """Resolve, inspect, and snapshot a repository through explicit policy boundaries."""

    async def resolve(
        self,
        repository_url: str,
        revision: str | None,
        *,
        ctx: OperationContext,
    ) -> RepositoryResolution:
        """Resolve a permitted repository reference to a complete immutable commit SHA."""

    async def fetch_metadata(
        self,
        repository: RepositoryResolution,
        *,
        ctx: OperationContext,
    ) -> RepositoryMetadata:
        """Read repository metadata without importing, installing, or executing code."""

    async def snapshot(
        self,
        repository: RepositoryResolution,
        *,
        ctx: OperationContext,
    ) -> ArtifactRef:
        """Write an approved repository snapshot idempotently as an immutable artifact."""


@runtime_checkable
class StaticRepositoryAnalyzer(Protocol):
    """Analyze an archive statically; implementations must never execute repository code."""

    async def analyze(
        self,
        repository_archive: ArtifactRef,
        policy: RepositoryPolicy,
        *,
        ctx: OperationContext,
    ) -> RepositoryAnalysis:
        """Return bounded static evidence and support assessment for one archive."""


__all__ = ["RepositoryProvider", "StaticRepositoryAnalyzer"]
