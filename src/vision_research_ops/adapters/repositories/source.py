"""Bounded text indexing and reading from an approved ZIP snapshot."""

from __future__ import annotations

import stat
from hashlib import sha256
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo

from vision_research_ops.application.services.repository_insight_models import (
    RepositorySourceEntry,
    RepositorySourceIndex,
    RepositorySourceRead,
)
from vision_research_ops.domain import ArtifactRef

_MAX_INDEX_FILES = 500
_MAX_READ_BYTES = 8 * 1024
_TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"}


def _safe_member_path(name: str) -> PurePosixPath:
    if "\\" in name or "%" in name:
        raise ValueError("archive member path is not canonical POSIX")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("archive member path escapes the repository")
    return path


def _is_symlink(info: ZipInfo) -> bool:
    return stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)


class BoundedZipSourceReader:
    """Read only allowlisted text members; never extract, import, install, or execute."""

    def __init__(self, *, artifact_root: Path) -> None:
        self._artifact_root = artifact_root.resolve()

    def _artifact_path(self, artifact: ArtifactRef) -> Path:
        relative = Path(artifact.uri)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("repository archive URI must stay below the artifact root")
        resolved = (self._artifact_root / relative).resolve()
        if not resolved.is_relative_to(self._artifact_root):
            raise ValueError("repository archive URI escapes the artifact root")
        payload = resolved.read_bytes()
        if len(payload) != artifact.size_bytes:
            raise ValueError("repository archive size does not match its artifact evidence")
        if f"sha256:{sha256(payload).hexdigest()}" != artifact.sha256:
            raise ValueError("repository archive hash does not match its artifact evidence")
        return resolved

    @staticmethod
    def _members(archive: ZipFile) -> dict[str, ZipInfo]:
        infos = archive.infolist()
        if len(infos) > _MAX_INDEX_FILES:
            raise ValueError("repository archive has too many members for bounded insight")
        safe = {info: _safe_member_path(info.filename) for info in infos}
        roots = {path.parts[0] for path in safe.values() if path.parts}
        strip_root = len(roots) == 1
        result: dict[str, ZipInfo] = {}
        for info, path in safe.items():
            parts = path.parts[1:] if strip_root else path.parts
            if not parts or info.is_dir() or _is_symlink(info):
                continue
            relative = PurePosixPath(*parts).as_posix()
            if PurePosixPath(relative).suffix.casefold() not in _TEXT_SUFFIXES:
                continue
            if relative in result:
                raise ValueError("repository archive contains duplicate canonical paths")
            result[relative] = info
        if len(result) > _MAX_INDEX_FILES:
            raise ValueError("repository archive text index exceeds the bounded file count")
        return result

    def index(self, artifact: ArtifactRef) -> RepositorySourceIndex:
        """Return a canonical path and size index without exposing source contents."""
        try:
            with ZipFile(self._artifact_path(artifact)) as archive:
                members = self._members(archive)
                entries = [
                    RepositorySourceEntry(path=path, size_bytes=members[path].file_size)
                    for path in sorted(members, key=str.casefold)
                ]
            return RepositorySourceIndex(files=entries)
        except (BadZipFile, KeyError, OSError, UnicodeError, ValueError):
            raise ValueError("bounded repository source index failed") from None

    def read(
        self,
        artifact: ArtifactRef,
        index: RepositorySourceIndex,
        path: str,
    ) -> RepositorySourceRead:
        """Return at most 8 KiB from one canonical path already present in the index."""
        entry = next((item for item in index.files if item.path == path), None)
        if entry is None:
            raise ValueError("repository path is not present in the approved source index")
        try:
            with ZipFile(self._artifact_path(artifact)) as archive:
                info = self._members(archive).get(path)
                if info is None or info.file_size != entry.size_bytes:
                    raise ValueError("repository source member changed after indexing")
                with archive.open(info) as stream:
                    payload = stream.read(_MAX_READ_BYTES + 1)
        except (BadZipFile, KeyError, OSError, RuntimeError, ValueError):
            raise ValueError("bounded repository source read failed") from None
        returned = payload[:_MAX_READ_BYTES]
        content = returned.decode("utf-8", errors="replace")
        if "\x00" in content:
            raise ValueError("repository source member is not ordinary text")
        return RepositorySourceRead(
            path=entry.path,
            content=content,
            returned_bytes=len(returned),
            original_bytes=entry.size_bytes,
            truncated=entry.size_bytes > len(returned),
            content_hash=f"sha256:{sha256(returned).hexdigest()}",
        )


__all__ = ["BoundedZipSourceReader"]
