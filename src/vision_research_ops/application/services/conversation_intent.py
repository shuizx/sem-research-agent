"""Strict, bounded intent routing for the conversational pipeline CLI."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$", re.IGNORECASE)
_GITHUB_RE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_URL_TOKEN_RE = re.compile(r"https?://[^\s]+")
_ARXIV_ID_TOKEN_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}\uff0c\u3002\uff1b\uff1a\uff01\uff1f\uff09\u3011\u300b"


class ConversationIntentName(StrEnum):
    """The complete, closed set of user-visible conversation intents."""

    RESEARCH_LATEST = "RESEARCH_LATEST"
    ANALYZE_ARXIV_PAPER = "ANALYZE_ARXIV_PAPER"
    ANALYZE_GITHUB_REPOSITORY = "ANALYZE_GITHUB_REPOSITORY"
    RUN_PIPELINE_SAMPLE = "RUN_PIPELINE_SAMPLE"
    SHOW_HELP = "SHOW_HELP"
    SHOW_STATUS = "SHOW_STATUS"
    SHOW_CURRENT_PAPER = "SHOW_CURRENT_PAPER"
    CONTINUE_CURRENT_PAPER = "CONTINUE_CURRENT_PAPER"
    EXIT = "EXIT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class ConversationIntent(BaseModel):
    """Schema-validated routing result with no tool or free-text execution fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    intent: ConversationIntentName
    arxiv_id: str | None = Field(default=None, max_length=32)
    github_url: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _fields_match_intent(self) -> ConversationIntent:
        if self.intent is ConversationIntentName.ANALYZE_ARXIV_PAPER:
            if self.arxiv_id is None:
                raise ValueError("ANALYZE_ARXIV_PAPER requires arxiv_id")
            if self.github_url is not None:
                raise ValueError("arXiv analysis cannot contain github_url")
        elif self.intent is ConversationIntentName.ANALYZE_GITHUB_REPOSITORY:
            if self.github_url is None:
                raise ValueError("ANALYZE_GITHUB_REPOSITORY requires github_url")
            if self.arxiv_id is not None:
                raise ValueError("GitHub preview cannot contain arxiv_id")
        elif self.arxiv_id is not None or self.github_url is not None:
            raise ValueError("only URL-targeted intents may contain a target")
        return self


class ConversationRoutingError(RuntimeError):
    """An explicit provider or schema failure from the optional LLM router."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class StructuredIntentRouter(Protocol):
    """Narrow protocol used only for ambiguous natural-language routing."""

    async def route(self, message: str) -> ConversationIntent:
        """Return one schema-validated fixed intent or raise an explicit routing error."""


def normalize_arxiv_target(value: str) -> str | None:
    """Accept only canonical arXiv IDs or canonical abs/pdf URLs and remove versions."""
    candidate = value.strip()
    if _ARXIV_ID_RE.fullmatch(candidate):
        return re.sub(r"v\d+$", "", candidate, flags=re.IGNORECASE)
    parsed = urlparse(candidate)
    if parsed.scheme != "https" or parsed.netloc != "arxiv.org" or parsed.query or parsed.fragment:
        return None
    parts = parsed.path.split("/")
    if len(parts) != 3 or parts[1] not in {"abs", "pdf"}:
        return None
    paper_id = parts[2][:-4] if parts[1] == "pdf" and parts[2].endswith(".pdf") else parts[2]
    if _ARXIV_ID_RE.fullmatch(paper_id) is None:
        return None
    return re.sub(r"v\d+$", "", paper_id, flags=re.IGNORECASE)


def normalize_github_target(value: str) -> str | None:
    """Accept only a canonical public GitHub owner/repository URL, without I/O."""
    candidate = value.strip()
    if _GITHUB_RE.fullmatch(candidate) is None:
        return None
    owner, repository = candidate.removeprefix("https://github.com/").split("/", maxsplit=1)
    if owner in {".", ".."} or repository in {".", ".."}:
        return None
    return f"https://github.com/{owner.casefold()}/{repository.casefold()}"


def _command_intent(message: str) -> ConversationIntent | None:
    command = message.strip().casefold()
    if command in {"/help", "help", "帮助"}:
        return ConversationIntent(intent=ConversationIntentName.SHOW_HELP)
    if command in {"/status", "状态"} or command.startswith("/debug"):
        return ConversationIntent(intent=ConversationIntentName.SHOW_STATUS)
    if command in {"/current", "当前候选", "当前论文"}:
        return ConversationIntent(intent=ConversationIntentName.SHOW_CURRENT_PAPER)
    if command in {"/continue", "继续找代码"}:
        return ConversationIntent(intent=ConversationIntentName.CONTINUE_CURRENT_PAPER)
    if command in {"/exit", "exit", "退出", "quit"}:
        return ConversationIntent(intent=ConversationIntentName.EXIT)
    if command in {"/research", "开始检索文献", "检索最新论文", "最新论文"}:
        return ConversationIntent(intent=ConversationIntentName.RESEARCH_LATEST)
    if command in {"/pipeline", "运行完整流程", "运行完整研究流程"}:
        return ConversationIntent(intent=ConversationIntentName.RUN_PIPELINE_SAMPLE)
    return None


def _supported_targets(message: str) -> list[ConversationIntent]:
    """Collect distinct canonical arXiv/GitHub targets without choosing among them."""
    targets: dict[tuple[str, str], ConversationIntent] = {}
    url_matches = list(_URL_TOKEN_RE.finditer(message))
    for match in url_matches:
        url = match.group(0).rstrip(_TRAILING_URL_PUNCTUATION)
        arxiv_id = normalize_arxiv_target(url)
        if arxiv_id is not None:
            targets[("arxiv", arxiv_id)] = ConversationIntent(
                intent=ConversationIntentName.ANALYZE_ARXIV_PAPER,
                arxiv_id=arxiv_id,
            )
            continue
        github_url = normalize_github_target(url)
        if github_url is not None:
            targets[("github", github_url)] = ConversationIntent(
                intent=ConversationIntentName.ANALYZE_GITHUB_REPOSITORY,
                github_url=github_url,
            )
    for match in _ARXIV_ID_TOKEN_RE.finditer(message):
        if any(
            url_match.start() <= match.start() and match.end() <= url_match.end()
            for url_match in url_matches
        ):
            continue
        arxiv_id = normalize_arxiv_target(match.group(0))
        assert arxiv_id is not None
        targets[("arxiv", arxiv_id)] = ConversationIntent(
            intent=ConversationIntentName.ANALYZE_ARXIV_PAPER,
            arxiv_id=arxiv_id,
        )
    return list(targets.values())


def has_ambiguous_supported_targets(message: str) -> bool:
    """Return true when one turn contains multiple distinct supported targets."""
    return len(_supported_targets(message.strip())) > 1


def deterministic_intent(message: str) -> ConversationIntent | None:
    """Recognize commands and canonical URL/ID inputs without invoking an LLM."""
    stripped = message.strip()
    if not stripped:
        return ConversationIntent(intent=ConversationIntentName.OUT_OF_SCOPE)
    command = _command_intent(stripped)
    if command is not None:
        return command
    targets = _supported_targets(stripped)
    if len(targets) > 1:
        return ConversationIntent(intent=ConversationIntentName.OUT_OF_SCOPE)
    if targets:
        return targets[0]
    if "github.com" in stripped.casefold():
        return ConversationIntent(
            intent=ConversationIntentName.ANALYZE_GITHUB_REPOSITORY,
            github_url=stripped,
        )
    if "arxiv" in stripped.casefold() or "arxiv.org" in stripped.casefold():
        return ConversationIntent(
            intent=ConversationIntentName.ANALYZE_ARXIV_PAPER,
            arxiv_id=stripped,
        )
    return None


class FixtureIntentRouter:
    """Offline structured-router substitute used by normal tests and fixture CLI runs."""

    async def route(self, message: str) -> ConversationIntent:
        text = message.casefold()
        if "当前候选" in text or "当前论文" in text:
            return ConversationIntent(intent=ConversationIntentName.SHOW_CURRENT_PAPER)
        if "继续找代码" in text or "继续查代码" in text:
            return ConversationIntent(intent=ConversationIntentName.CONTINUE_CURRENT_PAPER)
        if "检索" in text or "最新论文" in text or "research" in text:
            return ConversationIntent(intent=ConversationIntentName.RESEARCH_LATEST)
        if "完整流程" in text or "pipeline" in text or "研究流程" in text:
            return ConversationIntent(intent=ConversationIntentName.RUN_PIPELINE_SAMPLE)
        if "帮助" in text or "help" in text:
            return ConversationIntent(intent=ConversationIntentName.SHOW_HELP)
        return ConversationIntent(intent=ConversationIntentName.OUT_OF_SCOPE)


__all__ = [
    "ConversationIntent",
    "ConversationIntentName",
    "ConversationRoutingError",
    "FixtureIntentRouter",
    "StructuredIntentRouter",
    "deterministic_intent",
    "has_ambiguous_supported_targets",
    "normalize_arxiv_target",
    "normalize_github_target",
]
