"""Immutable artifact and trusted dataset-catalog port interfaces."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from vision_research_ops.domain import ArtifactRef, DatasetProfile

from .base import AsyncBinaryReader
from .common import ArtifactDescriptor, DatasetMountSpec, DownloadGrant, OperationContext


@runtime_checkable
class ArtifactStore(Protocol):
    """Store immutable, content-addressed artifacts behind authorized references."""

    async def put_bytes(
        self,
        data: AsyncIterator[bytes] | bytes,
        descriptor: ArtifactDescriptor,
        *,
        expected_sha256: str | None,
        ctx: OperationContext,
    ) -> ArtifactRef:
        """Finalize bytes once, optionally verifying the caller's expected content hash."""

    async def open(self, artifact_id: str, *, ctx: OperationContext) -> AsyncBinaryReader:
        """Open an immutable artifact through an asynchronous binary reader."""

    async def stat(self, artifact_id: str, *, ctx: OperationContext) -> ArtifactRef:
        """Return immutable artifact metadata without reading its content."""

    async def issue_download(
        self,
        artifact_id: str,
        ttl_seconds: int,
        *,
        ctx: OperationContext,
    ) -> DownloadGrant:
        """Issue a sensitivity-checked, credential-free download reference."""


@runtime_checkable
class DatasetCatalog(Protocol):
    """Read de-identified profiles and trusted-executor-only mount references."""

    async def get_profile(
        self,
        dataset_id: str,
        version: str,
        *,
        ctx: OperationContext,
    ) -> DatasetProfile:
        """Return a de-identified immutable dataset profile."""

    async def get_mount_spec(
        self,
        dataset_id: str,
        version: str,
        *,
        ctx: OperationContext,
    ) -> DatasetMountSpec:
        """Return an opaque mount handle only for trusted execution composition."""


__all__ = ["ArtifactStore", "DatasetCatalog"]
