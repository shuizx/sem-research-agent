"""Central settings for optional live Research Agent integrations."""

from __future__ import annotations

from collections.abc import Mapping
from os import environ
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

DASHSCOPE_OPENAI_BASE_URL: Literal["https://dashscope.aliyuncs.com/compatible-mode/v1"] = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
ARXIV_API_URL = "https://export.arxiv.org/api/query"
LOCAL_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_local_env(path: Path | None = None) -> bool:
    """Load an ignored local env file without replacing process-level configuration."""
    return load_dotenv(dotenv_path=path or LOCAL_ENV_FILE, override=False)


class Settings(BaseModel):
    """Explicitly parsed configuration injected into adapters and workflows."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
    )

    dashscope_api_key: SecretStr | None = None
    llm_model: str = "qwen-plus"
    llm_base_url: Literal["https://dashscope.aliyuncs.com/compatible-mode/v1"] = (
        DASHSCOPE_OPENAI_BASE_URL
    )
    research_output_root: Path = Path("var/research")
    research_overlap_minutes: int = Field(default=60, ge=0, le=24 * 60)
    research_initial_lookback_hours: int = Field(default=24, ge=1, le=24 * 30)
    arxiv_timeout_seconds: int = Field(default=20, ge=1, le=120)

    @field_validator("llm_model")
    @classmethod
    def _nonblank_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("VRO_LLM_MODEL must not be blank")
        return value

    @classmethod
    def from_env(cls, source: Mapping[str, str] | None = None) -> Settings:
        """Parse the small supported environment surface in one trusted layer."""
        values = environ if source is None else source

        def parse_int(name: str, default: int) -> int:
            raw = values.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer") from exc

        api_key = values.get("DASHSCOPE_API_KEY")
        return cls(
            dashscope_api_key=None if not api_key else SecretStr(api_key),
            llm_model=values.get("VRO_LLM_MODEL", "qwen-plus"),
            research_output_root=Path(values.get("VRO_RESEARCH_OUTPUT_ROOT", "var/research")),
            research_overlap_minutes=parse_int("VRO_RESEARCH_OVERLAP_MINUTES", 60),
            research_initial_lookback_hours=parse_int(
                "VRO_RESEARCH_INITIAL_LOOKBACK_HOURS",
                24,
            ),
            arxiv_timeout_seconds=parse_int("VRO_ARXIV_TIMEOUT_SECONDS", 20),
        )

    def require_dashscope_api_key(self) -> str:
        """Return the live key or fail explicitly without exposing a secret value."""
        if self.dashscope_api_key is None:
            raise ValueError("DASHSCOPE_API_KEY is required for live LLM mode")
        return self.dashscope_api_key.get_secret_value()


__all__ = [
    "ARXIV_API_URL",
    "DASHSCOPE_OPENAI_BASE_URL",
    "LOCAL_ENV_FILE",
    "Settings",
    "load_local_env",
]
