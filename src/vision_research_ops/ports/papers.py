"""Provider-neutral academic-paper retrieval port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .common import ExternalPaperId, OperationContext, PaperQuery, PaperSearchPage, RawPaperRecord


@runtime_checkable
class PaperProvider(Protocol):
    """Retrieve raw, paginated academic records without normalizing business entities."""

    provider_name: str

    async def search(
        self,
        query: PaperQuery,
        *,
        cursor: str | None,
        ctx: OperationContext,
    ) -> PaperSearchPage:
        """Return one provider page and its opaque continuation cursor."""

    async def get_by_external_id(
        self,
        external_id: ExternalPaperId,
        *,
        ctx: OperationContext,
    ) -> RawPaperRecord | None:
        """Return a raw provider record, or ``None`` if the ID is not present."""


__all__ = ["PaperProvider"]
