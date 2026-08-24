"""Small, explicit local memory for the conversational pipeline Agent."""

from __future__ import annotations

import json
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .conversation_intent import normalize_github_target

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_CONTEXT_BYTES = 64 * 1024


def _non_blank(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("context text must be non-blank and trimmed")
    return value


def _relative_ref(value: str) -> str:
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or "%" in value
        or value.startswith("/")
        or ":" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("context references must be canonical relative paths")
    return value


def _public_code_url(value: str) -> str:
    if not value or value != value.strip() or "\\" in value or "%" in value:
        raise ValueError("code URLs must be canonical public HTTPS URLs")
    parsed = urlsplit(value)
    hostname = parsed.hostname
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("code URLs must be canonical public HTTPS URLs")
    try:
        ip_address(hostname)
    except ValueError:
        pass
    else:
        raise ValueError("code URLs cannot target an IP address")
    if "." not in hostname or hostname.casefold().endswith((".local", ".internal")):
        raise ValueError("code URLs must use a public hostname")
    if any(part in {".", ".."} for part in parsed.path.split("/")):
        raise ValueError("code URLs cannot contain dot segments")
    github_url = normalize_github_target(value)
    if github_url is not None:
        return github_url
    return urlunsplit(("https", parsed.netloc.casefold(), parsed.path, "", ""))


class ConversationContextModel(BaseModel):
    """Strict base for the small, public context document."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class CurrentPaperContext(ConversationContextModel):
    """Public paper facts required by ``/current`` and ``/continue``."""

    schema_version: Literal["1"] = "1"
    paper_id: str = Field(max_length=160)
    title: str = Field(max_length=512)
    arxiv_id: str | None = Field(default=None, max_length=32)
    code_urls: list[str] = Field(default_factory=list, max_length=8)
    recommendation: Literal["HIGH", "MEDIUM", "LOW", "REJECT"]
    artifact_ref: str = Field(max_length=512)

    @field_validator("paper_id", "title")
    @classmethod
    def _text_is_trimmed(cls, value: str) -> str:
        return _non_blank(value)

    @field_validator("arxiv_id")
    @classmethod
    def _arxiv_id_is_canonical(cls, value: str | None) -> str | None:
        if value is not None and _ARXIV_ID_RE.fullmatch(value) is None:
            raise ValueError("arxiv_id must be a canonical unversioned identifier")
        return value

    @field_validator("code_urls")
    @classmethod
    def _code_urls_are_public_and_unique(cls, value: list[str]) -> list[str]:
        validated = [_public_code_url(item) for item in value]
        if len(validated) != len(set(validated)):
            raise ValueError("code URLs must be unique")
        return validated

    @field_validator("artifact_ref")
    @classmethod
    def _artifact_is_relative(cls, value: str) -> str:
        return _relative_ref(value)


class CurrentRepositoryContext(ConversationContextModel):
    """Public repository facts from one strictly completed insight result."""

    schema_version: Literal["1"] = "1"
    repository_url: str = Field(max_length=300)
    commit_sha: str = Field(max_length=40)
    license_spdx: str = Field(max_length=80)
    adaptation_fit: Literal["HIGH", "MEDIUM", "LOW", "INSUFFICIENT_EVIDENCE"]
    read_files: list[str] = Field(min_length=1, max_length=6)
    result_ref: str = Field(max_length=512)

    @field_validator("repository_url")
    @classmethod
    def _repository_url_is_canonical(cls, value: str) -> str:
        if normalize_github_target(value) != value:
            raise ValueError("repository_url must be a canonical public GitHub URL")
        return value

    @field_validator("commit_sha")
    @classmethod
    def _commit_is_full_sha(cls, value: str) -> str:
        if _COMMIT_RE.fullmatch(value) is None:
            raise ValueError("commit_sha must be a full lowercase Git SHA")
        return value

    @field_validator("license_spdx")
    @classmethod
    def _license_is_trimmed(cls, value: str) -> str:
        return _non_blank(value)

    @field_validator("read_files")
    @classmethod
    def _read_files_are_relative_and_unique(cls, value: list[str]) -> list[str]:
        validated = [_relative_ref(item) for item in value]
        if len(validated) != len(set(validated)):
            raise ValueError("read files must be unique")
        return validated

    @field_validator("result_ref")
    @classmethod
    def _result_is_relative(cls, value: str) -> str:
        return _relative_ref(value)


class ConversationContext(ConversationContextModel):
    """Versioned persisted working context; intentionally not a chat transcript."""

    schema_version: Literal["1"] = "1"
    current_paper: CurrentPaperContext | None = None
    current_repository: CurrentRepositoryContext | None = None
    last_successful_artifact_ref: str | None = Field(default=None, max_length=512)
    updated_at: datetime

    @field_validator("last_successful_artifact_ref")
    @classmethod
    def _last_artifact_is_relative(cls, value: str | None) -> str | None:
        return None if value is None else _relative_ref(value)

    @field_validator("updated_at")
    @classmethod
    def _updated_at_is_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None:
            raise ValueError("updated_at must be timezone-aware UTC")
        if offset.total_seconds() != 0:
            raise ValueError("updated_at must use UTC")
        return value.astimezone(UTC)

    @classmethod
    def empty(cls, *, updated_at: datetime | None = None) -> ConversationContext:
        """Build an explicit empty context without touching the filesystem."""
        return cls(updated_at=updated_at or datetime.now(UTC))

    @property
    def has_working_context(self) -> bool:
        """Return whether the document contains any successful working fact."""
        return any(
            (
                self.current_paper is not None,
                self.current_repository is not None,
                self.last_successful_artifact_ref is not None,
            )
        )


@dataclass(frozen=True, slots=True)
class ConversationContextLoad:
    """A load outcome that never exposes a local path or parser exception."""

    context: ConversationContext
    restored: bool
    warning: str | None = None


class ConversationContextStoreError(RuntimeError):
    """Stable public failure raised when an atomic context write does not complete."""


class LocalConversationContextStore:
    """Read and atomically replace one small context JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> ConversationContextLoad:
        """Load a valid whole document, or return an empty context with a safe warning."""
        if not self._path.exists():
            return ConversationContextLoad(
                context=ConversationContext.empty(),
                restored=False,
            )
        try:
            if self._path.stat().st_size > _MAX_CONTEXT_BYTES:
                raise ValueError("context document exceeds the bounded size")
            context = ConversationContext.model_validate_json(
                self._path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValidationError, ValueError):
            return ConversationContextLoad(
                context=ConversationContext.empty(),
                restored=False,
                warning=(
                    "会话记忆无法恢复: 本地上下文损坏或版本不兼容; "
                    "已从空上下文启动, 可使用 /clear 重置。"
                ),
            )
        return ConversationContextLoad(
            context=context,
            restored=context.has_working_context,
        )

    def save(self, context: ConversationContext) -> None:
        """Write canonical JSON through a same-directory temporary file and replace."""
        try:
            validated = ConversationContext.model_validate(context.model_dump(mode="python"))
        except (TypeError, ValidationError, ValueError):
            raise ConversationContextStoreError("CONVERSATION_CONTEXT_WRITE_FAILED") from None
        payload = (
            json.dumps(
                validated.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        if len(payload.encode("utf-8")) > _MAX_CONTEXT_BYTES:
            raise ConversationContextStoreError("CONVERSATION_CONTEXT_WRITE_FAILED")
        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(self._path)
        except OSError:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise ConversationContextStoreError("CONVERSATION_CONTEXT_WRITE_FAILED") from None


__all__ = [
    "ConversationContext",
    "ConversationContextLoad",
    "ConversationContextStoreError",
    "CurrentPaperContext",
    "CurrentRepositoryContext",
    "LocalConversationContextStore",
]
