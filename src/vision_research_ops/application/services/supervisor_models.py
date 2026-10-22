"""Task-local, bounded supervisor contracts; no execution authority is model-owned."""

from __future__ import annotations

import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .conversation_intent import normalize_github_target


class SupervisorError(RuntimeError):
    """A safe, stable supervisor failure code."""


class ModelCallBudget:
    """Count request invocations before execution, including failed requests."""

    def __init__(self, limit: int = 24) -> None:
        if limit < 1:
            raise ValueError("model budget must be positive")
        self.limit = limit
        self.used = 0

    def consume(self) -> None:
        """Reserve one call or stop before invoking the provider."""
        if self.used >= self.limit:
            raise SupervisorError("SUPERVISOR_MODEL_BUDGET_EXHAUSTED")
        self.used += 1


class WorkerTask(BaseModel):
    """Independent worker input, limited to a goal and a validated public target."""

    model_config = ConfigDict(extra="forbid", strict=True)
    worker: Literal["research", "code"]
    goal: str = Field(default="", max_length=2000)
    target: str | None = Field(default=None, max_length=300)


class WorkerResult(BaseModel):
    """Compact terminal observation, never a transcript or an approval."""

    model_config = ConfigDict(extra="forbid", strict=True)
    worker: Literal["research", "code"]
    status: Literal["success", "rejected", "failed", "no_results"]
    code_urls: list[str] = Field(default_factory=list, max_length=5)
    artifact_ref: str | None = Field(default=None, max_length=300)
    summary: str = Field(
        default="Worker completed with the stated terminal status.", max_length=600
    )
    reason: str | None = Field(default=None, max_length=120)
    paper_id: str | None = Field(default=None, max_length=160)

    @field_validator("reason", mode="before")
    @classmethod
    def _stable_reason(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,119}", value):
            return value
        return "WORKER_FAILED"

    @field_validator("code_urls")
    @classmethod
    def _urls(cls, value: list[str]) -> list[str]:
        if any(normalize_github_target(url) != url for url in value):
            raise ValueError("worker handoff requires canonical GitHub targets")
        return value

    @field_validator("artifact_ref")
    @classmethod
    def _ref(cls, value: str | None) -> str | None:
        if value is not None and (
            ":" in value
            or "\\" in value
            or value.startswith("/")
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError("artifact reference must remain local and relative")
        return value


class SupervisorProposal(BaseModel):
    """Closed action schema used for both scripted and live decisions."""

    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["research", "code", "finish"]
    target: str | None = Field(default=None, max_length=300)


class SupervisorPolicy(Protocol):
    """Select a single action after observing compact worker results."""

    async def decide(
        self, goal: str, observations: list[WorkerResult], budget: ModelCallBudget
    ) -> SupervisorProposal:
        """Return a strict proposal, never an approval or executable command."""
