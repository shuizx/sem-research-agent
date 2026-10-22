"""Provider configuration wiring and secret boundaries without live credentials or requests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from vision_research_ops.adapters.llm.dashscope import build_dashscope_chat_model
from vision_research_ops.settings import DASHSCOPE_OPENAI_BASE_URL, Settings, load_local_env

SHU_URL = "https://claw.shu.edu.cn/api/v1"
SHU_MODEL = "Qwen3.8-MAX"


def shu_settings(**extra: str) -> Settings:
    """Build isolated settings using visibly synthetic credentials only."""
    return Settings.from_env(
        {
            "VRO_LLM_BASE_URL": SHU_URL,
            "VRO_LLM_MODEL": SHU_MODEL,
            "VRO_LLM_API_KEY": "synthetic-shu-key",
            "DASHSCOPE_API_KEY": "synthetic-legacy-key",
            **extra,
        }
    )


def test_generic_config_and_secret_selection() -> None:
    settings = shu_settings()
    assert settings.llm_base_url == SHU_URL
    assert settings.llm_model == SHU_MODEL
    assert settings.llm_provider_id == "openai-compatible"
    assert settings.require_llm_api_key().get_secret_value() == "synthetic-shu-key"
    for rendered in (repr(settings), settings.model_dump_json()):
        assert "synthetic-shu-key" not in rendered
        assert "synthetic-legacy-key" not in rendered
    assert shu_settings(VRO_LLM_BASE_URL=SHU_URL + "/").llm_base_url == SHU_URL


def test_legacy_default_and_explicit_generic_key_precedence() -> None:
    legacy = Settings.from_env({"DASHSCOPE_API_KEY": "synthetic-legacy-key"})
    assert legacy.llm_base_url == DASHSCOPE_OPENAI_BASE_URL
    assert legacy.llm_provider_id == "dashscope-openai-compatible"
    assert legacy.require_llm_api_key() == SecretStr("synthetic-legacy-key")
    assert legacy.require_dashscope_api_key() == "synthetic-legacy-key"
    generic = shu_settings(VRO_LLM_BASE_URL=DASHSCOPE_OPENAI_BASE_URL)
    assert generic.require_llm_api_key() == SecretStr("synthetic-shu-key")


@pytest.mark.parametrize("key", ["", "   "])
def test_custom_endpoint_never_uses_legacy_key(key: str, monkeypatch) -> None:
    settings = shu_settings(VRO_LLM_API_KEY=key)
    monkeypatch.setattr(
        "vision_research_ops.adapters.llm.dashscope.ChatOpenAI",
        lambda **_: pytest.fail("must fail before creating a client"),
    )
    with pytest.raises(ValueError, match="VRO_LLM_API_KEY") as error:
        build_dashscope_chat_model(settings)
    assert "synthetic-legacy-key" not in str(error.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://claw.shu.edu.cn/api/v1",
        "https://user:synthetic-url-secret@claw.shu.edu.cn/api/v1",
        SHU_URL + "?key=synthetic-url-secret",
        SHU_URL + "#synthetic-url-secret",
        "claw.shu.edu.cn/api/v1",
    ],
)
def test_invalid_endpoint_rejected_without_echoing_input(url: str) -> None:
    with pytest.raises(ValidationError, match="VRO_LLM_BASE_URL") as error:
        shu_settings(VRO_LLM_BASE_URL=url)
    assert "synthetic-url-secret" not in str(error.value)


def test_same_name_process_configuration_overrides_dotenv(tmp_path: Path, monkeypatch) -> None:
    file = tmp_path / ".env"
    file.write_text(
        "VRO_LLM_API_KEY=synthetic-file-key\n"
        "VRO_LLM_BASE_URL=https://example.org/v1\nVRO_LLM_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VRO_LLM_API_KEY", "synthetic-process-key")
    monkeypatch.setenv("VRO_LLM_BASE_URL", SHU_URL)
    monkeypatch.setenv("VRO_LLM_MODEL", SHU_MODEL)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    assert load_local_env(file)
    settings = Settings.from_env()
    assert settings.llm_base_url == SHU_URL
    assert settings.llm_model == SHU_MODEL
    assert settings.require_llm_api_key() == SecretStr("synthetic-process-key")


def test_factory_preserves_selected_endpoint_model_and_tls(monkeypatch) -> None:
    captured = {}
    http_options = []

    def fake_client(**kwargs):
        http_options.append(kwargs)
        return object()

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("vision_research_ops.adapters.llm.dashscope.ChatOpenAI", fake_chat)
    monkeypatch.setattr("vision_research_ops.adapters.llm.dashscope.httpx.Client", fake_client)
    monkeypatch.setattr("vision_research_ops.adapters.llm.dashscope.httpx.AsyncClient", fake_client)
    build_dashscope_chat_model(shu_settings())
    assert captured["base_url"] == SHU_URL
    assert captured["model"] == SHU_MODEL
    assert captured["api_key"] == SecretStr("synthetic-shu-key")
    assert captured["temperature"] == 0
    assert http_options == [{"trust_env": False}, {"trust_env": False}]


@pytest.mark.parametrize(
    ("module_name", "factory_name", "planner_name"),
    [
        ("dashscope", "build_dashscope_llm", "DashScopeStructuredLLM"),
        ("adaptation", "build_dashscope_adaptation_planner", "LangGraphAdaptationPlanner"),
        (
            "repository_insight",
            "build_dashscope_repository_insight_planner",
            "LangGraphRepositoryInsightPlanner",
        ),
        ("research", "build_dashscope_research_planner", "LangGraphResearchPlanner"),
    ],
)
def test_factories_inject_truthful_provider_metadata(
    module_name: str, factory_name: str, planner_name: str, monkeypatch
) -> None:
    from importlib import import_module

    module = import_module(f"vision_research_ops.adapters.llm.{module_name}")
    captured = {}

    def fake_planner(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(module, "build_dashscope_chat_model", lambda _settings: object())
    monkeypatch.setattr(module, planner_name, fake_planner)
    getattr(module, factory_name)(shu_settings())
    assert captured["provider_id"] == "openai-compatible"
    assert captured["model_id"] == SHU_MODEL


def test_launcher_and_template_expose_config_names_not_keys() -> None:
    script = Path("scripts/run-agent.ps1").read_text(encoding="utf-8")
    template = Path(".env.example").read_text(encoding="utf-8")
    assert "IsNullOrWhiteSpace($env:VRO_LLM_API_KEY)" in script
    assert "IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)" in script
    assert 'Write-Host "$env:VRO_LLM_API_KEY"' not in script
    assert SHU_URL in template and SHU_MODEL in template
    assert "synthetic-shu-key" not in template
