"""Small shared Protocols used by concrete SEM Research Agent port interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AsyncBinaryReader(Protocol):
    """Minimal asynchronous binary reader returned by an artifact store."""

    async def read(self) -> bytes:
        """Read the complete immutable artifact content."""

    def __aiter__(self) -> AsyncBinaryReader:
        """Return the asynchronous byte-chunk iterator."""

    async def __anext__(self) -> bytes:
        """Yield the next byte chunk or raise ``StopAsyncIteration``."""


__all__ = ["AsyncBinaryReader"]
