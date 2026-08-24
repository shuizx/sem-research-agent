"""Configuration boundary and manual offline Research Agent entry tests."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
from pydantic import SecretStr

from vision_research_ops.adapters.llm import build_dashscope_llm
from vision_research_ops.application.services.paper_models import ResearchResult
from vision_research_ops.cli.research import parse_options, run
from vision_research_ops.settings import DASHSCOPE_OPENAI_BASE_URL, Settings, load_local_env


def test_local_env_loads_secret_without_overriding_process_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A local ignored file is convenient, while explicit process values stay authoritative."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DASHSCOPE_API_KEY=file-secret\nVRO_LLM_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("VRO_LLM_MODEL", "process-model")

    assert load_local_env(env_file)
    settings = Settings.from_env()

    assert settings.require_dashscope_api_key() == "file-secret"
    assert settings.llm_model == "process-model"
    assert "file-secret" not in repr(settings)


def test_committed_env_example_contains_no_credential() -> None:
    """The tracked template documents names without looking like a usable secret."""
    content = Path(".env.example").read_text(encoding="utf-8")
    assert "DASHSCOPE_API_KEY=replace-with-your-dashscope-api-key" in content
    assert "VRO_LLM_MODEL=qwen-plus" in content
    assert "sk-" not in content


@pytest.mark.parametrize(
    "module_name",
    [
        "vision_research_ops.cli.research",
        "vision_research_ops.cli.pipeline",
    ],
)
def test_cli_main_loads_local_env_before_running(module_name: str, monkeypatch) -> None:
    """Both user-facing entry points load .env, while their service functions remain pure."""
    module = import_module(module_name)
    events: list[str] = []

    monkeypatch.setattr(module, "parse_options", lambda _argv: object())
    monkeypatch.setattr(module, "load_local_env", lambda: events.append("env"))

    async def fake_run(_options: object) -> int:
        events.append("run")
        return 0

    monkeypatch.setattr(module, "run", fake_run)
    assert module.main([]) == 0
    assert events == ["env", "run"]


def test_settings_parse_only_supported_environment_and_redact_key() -> None:
    """The central settings layer matches OpenAI-compatible's DashScope-compatible scheme."""
    settings = Settings.from_env(
        {
            "DASHSCOPE_API_KEY": "test-secret-value",
            "VRO_LLM_MODEL": "qwen-plus",
            "VRO_RESEARCH_OUTPUT_ROOT": "var/example",
            "VRO_RESEARCH_OVERLAP_MINUTES": "30",
            "VRO_RESEARCH_INITIAL_LOOKBACK_HOURS": "48",
            "VRO_ARXIV_TIMEOUT_SECONDS": "15",
            "VRO_DATASET_ROOT": "  ",
        }
    )
    assert settings.llm_base_url == DASHSCOPE_OPENAI_BASE_URL
    assert settings.require_dashscope_api_key() == "test-secret-value"
    assert "test-secret-value" not in repr(settings)
    assert settings.research_output_root == Path("var/example")
    assert settings.research_overlap_minutes == 30
    assert settings.research_initial_lookback_hours == 48
    assert settings.arxiv_timeout_seconds == 15
    assert settings.dataset_root is None


def test_live_llm_factory_requires_key_and_passes_secret_to_chat_openai(monkeypatch) -> None:
    """Live construction is explicit, fixed to DashScope, and never performs a request."""
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        build_dashscope_llm(Settings())

    captured: dict[str, object] = {}
    sync_client = object()
    async_client = object()
    http_client_options: list[dict[str, object]] = []

    def fake_http_client(**kwargs):
        http_client_options.append(kwargs)
        return sync_client

    def fake_async_http_client(**kwargs):
        http_client_options.append(kwargs)
        return async_client

    def fake_chat_open_ai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "vision_research_ops.adapters.llm.dashscope.ChatOpenAI",
        fake_chat_open_ai,
    )
    monkeypatch.setattr(
        "vision_research_ops.adapters.llm.dashscope.httpx.Client",
        fake_http_client,
    )
    monkeypatch.setattr(
        "vision_research_ops.adapters.llm.dashscope.httpx.AsyncClient",
        fake_async_http_client,
    )
    build_dashscope_llm(Settings(dashscope_api_key=SecretStr("not-a-real-key")))
    assert isinstance(captured["api_key"], SecretStr)
    assert captured["base_url"] == DASHSCOPE_OPENAI_BASE_URL
    assert captured["model"] == "qwen-plus"
    assert captured["temperature"] == 0
    assert captured["http_client"] is sync_client
    assert captured["http_async_client"] is async_client
    assert captured["http_socket_options"] == ()
    assert http_client_options == [{"trust_env": False}, {"trust_env": False}]


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_manual_fixture_cli_runs_human_gated_graph_offline(tmp_path: Path) -> None:
    """The documented default sample produces honest fixture-labeled evidence JSON."""
    output_root = tmp_path / "cli-results"
    options = parse_options(
        [
            "--mode",
            "fixture",
            "--fixture-xml",
            "tests/research/fixtures/arxiv_feed.xml",
            "--output-root",
            str(output_root),
            "--workflow-id",
            "workflow-cli-fixture",
            "--decision",
            "approve",
        ]
    )
    output: list[str] = []
    assert await run(options, output_fn=output.append) == 0
    result = ResearchResult.model_validate_json(
        (output_root / "workflow-cli-fixture" / "papers.json").read_text(encoding="utf-8")
    )
    assert result.status == "COMPLETED"
    assert result.selected_paper_ids == ["paper-arxiv-2608.01234"]
    selected = next(item for item in result.assessments if item.selected)
    assert selected.generation is not None
    assert selected.generation.provider_id == "offline-fixture"
    assert "scripted fixture reasoning" in selected.applicability.risks[0].casefold()
    assert any('"recommended_papers"' in message for message in output)
