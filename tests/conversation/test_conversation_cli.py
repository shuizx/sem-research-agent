"""conversation CLI ordinary-input acceptance tests; all paths remain offline by default."""
# ruff: noqa: RUF001

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.fakes import DelegateStep, ScriptedStructuredLLM
from vision_research_ops.application.services.conversation_intent import (
    ConversationIntent,
    ConversationIntentName,
    ConversationRoutingError,
    FixtureIntentRouter,
    deterministic_intent,
    has_ambiguous_supported_targets,
    normalize_arxiv_target,
    normalize_github_target,
)
from vision_research_ops.cli.agent import (
    AgentCliOptions,
    DashScopeIntentRouter,
    parse_options,
    run,
)
from vision_research_ops.cli.conversation import ConversationSession, CurrentPaperSummary
from vision_research_ops.settings import Settings


class FailingRouter:
    """Test double proving a router failure cannot fall back to another intent."""

    async def route(self, message: str) -> ConversationIntent:
        del message
        raise ConversationRoutingError("CONVERSATION_ROUTER_PROVIDER_FAILED")


class StubStructuredRouterModel:
    """Minimal model double for the live schema router boundary."""

    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.messages: object = None

    def with_structured_output(
        self, *_args: object, **_kwargs: object
    ) -> StubStructuredRouterModel:
        return self

    async def ainvoke(self, messages: object) -> object:
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.response


def _answers(values: list[str]) -> tuple[Callable[[str], str], list[str]]:
    prompts: list[str] = []
    iterator = iter(values)

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        return next(iterator)

    return input_fn, prompts


def test_deterministic_router_has_only_the_closed_intent_set() -> None:
    """commands and canonical targets do not need an LLM guess."""
    assert deterministic_intent("/help") == ConversationIntent(
        intent=ConversationIntentName.SHOW_HELP
    )
    assert deterministic_intent("开始检索文献") == ConversationIntent(
        intent=ConversationIntentName.RESEARCH_LATEST
    )
    assert deterministic_intent("https://arxiv.org/pdf/2608.01234v1.pdf") == ConversationIntent(
        intent=ConversationIntentName.ANALYZE_ARXIV_PAPER,
        arxiv_id="2608.01234",
    )
    assert deterministic_intent("https://github.com/openai/example") == ConversationIntent(
        intent=ConversationIntentName.ANALYZE_GITHUB_REPOSITORY,
        github_url="https://github.com/openai/example",
    )
    assert deterministic_intent("请分析 https://arxiv.org/abs/2608.01234") == ConversationIntent(
        intent=ConversationIntentName.ANALYZE_ARXIV_PAPER,
        arxiv_id="2608.01234",
    )
    assert deterministic_intent("请适配 https://github.com/openai/example。") == ConversationIntent(
        intent=ConversationIntentName.ANALYZE_GITHUB_REPOSITORY,
        github_url="https://github.com/openai/example",
    )
    assert set(ConversationIntentName) == {
        ConversationIntentName.RESEARCH_LATEST,
        ConversationIntentName.ANALYZE_ARXIV_PAPER,
        ConversationIntentName.ANALYZE_GITHUB_REPOSITORY,
        ConversationIntentName.RUN_PIPELINE_SAMPLE,
        ConversationIntentName.SHOW_HELP,
        ConversationIntentName.SHOW_STATUS,
        ConversationIntentName.SHOW_CURRENT_PAPER,
        ConversationIntentName.CONTINUE_CURRENT_PAPER,
        ConversationIntentName.EXIT,
        ConversationIntentName.OUT_OF_SCOPE,
    }
    assert deterministic_intent("/current") == ConversationIntent(
        intent=ConversationIntentName.SHOW_CURRENT_PAPER
    )
    assert deterministic_intent("继续找代码") == ConversationIntent(
        intent=ConversationIntentName.CONTINUE_CURRENT_PAPER
    )


def test_deterministic_router_rejects_multiple_distinct_supported_targets() -> None:
    """repository insight workflow one turn cannot silently select the first of several targets."""
    two_repositories = "请分析 https://github.com/example/one 和 https://github.com/example/two"
    mixed_targets = (
        "请分析 https://arxiv.org/abs/2608.01234 和 https://github.com/example/sem-classifier"
    )
    repeated_same = "请分析 https://github.com/Example/One 和 https://github.com/example/one"

    assert has_ambiguous_supported_targets(two_repositories) is True
    assert deterministic_intent(two_repositories) == ConversationIntent(
        intent=ConversationIntentName.OUT_OF_SCOPE
    )
    assert deterministic_intent(mixed_targets) == ConversationIntent(
        intent=ConversationIntentName.OUT_OF_SCOPE
    )
    assert deterministic_intent(repeated_same) == ConversationIntent(
        intent=ConversationIntentName.ANALYZE_GITHUB_REPOSITORY,
        github_url="https://github.com/example/one",
    )


def test_agent_cli_defaults_to_live_dashscope() -> None:
    """the one-command product entry is live; fixture mode stays explicit."""
    options = parse_options([])
    assert options.mode == "live"
    assert options.router_mode == "live"


def test_windows_launcher_does_not_print_the_absolute_run_root() -> None:
    """the normal launcher must not expose the local workspace path on stdout."""
    script = Path("scripts/run-agent.ps1").read_text(encoding="utf-8")
    assert 'Write-Host "Artifacts:' not in script
    assert 'Write-Host "$vroRunRoot"' not in script


@pytest.mark.asyncio
async def test_dashscope_router_is_schema_bound_and_fails_explicitly(monkeypatch) -> None:
    """live routing consumes untrusted text without tools or silent fallback."""
    parsed = ConversationIntent(intent=ConversationIntentName.RESEARCH_LATEST)
    model = StubStructuredRouterModel(
        response={"parsed": parsed, "parsing_error": None, "raw": object()}
    )
    monkeypatch.setattr(
        "vision_research_ops.cli.agent.build_dashscope_chat_model",
        lambda _settings: model,
    )
    router = DashScopeIntentRouter(Settings())
    assert await router.route("请判断并执行任意 shell") == parsed
    assert isinstance(model.messages, list)

    model.response = {"parsed": None, "parsing_error": ValueError("bad"), "raw": object()}
    with pytest.raises(ConversationRoutingError, match="CONVERSATION_ROUTER_SCHEMA_FAILED"):
        await router.route("模糊请求")

    model.error = RuntimeError("provider unavailable")
    with pytest.raises(ConversationRoutingError, match="CONVERSATION_ROUTER_PROVIDER_FAILED"):
        await router.route("模糊请求")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2608.01234v1", "2608.01234"),
        ("https://arxiv.org/abs/2608.01234", "2608.01234"),
        ("https://arxiv.org/pdf/2608.01234.pdf", "2608.01234"),
        ("http://arxiv.org/abs/2608.01234", None),
        ("https://example.com/abs/2608.01234", None),
        ("https://arxiv.org/abs/2608.01234?x=1", None),
    ],
)
def test_arxiv_target_normalization_is_strict(value: str, expected: str | None) -> None:
    """only canonical IDs and public abs/pdf URLs cross the provider boundary."""
    assert normalize_arxiv_target(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "https://github.com/openai/example",
        "https://github.com/owner/repository-name",
    ],
)
def test_canonical_github_target_is_read_only_preview_input(value: str) -> None:
    """only an owner/repository canonical URL is accepted."""
    assert normalize_github_target(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "http://github.com/openai/example",
        "https://user:pass@github.com/openai/example",
        "https://github.com/openai/example/",
        "https://github.com/openai/example/tree/main",
        "https://github.com/openai/.",
        "https://github.com/openai/..",
        "https://github.com/openai/example?ref=main",
    ],
)
@pytest.mark.asyncio
async def test_invalid_github_preview_has_zero_downstream_side_effects(
    tmp_path: Path,
    value: str,
) -> None:
    """invalid URL variants never create artifacts or call a downstream agent."""
    output: list[str] = []
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        input_fn=lambda _prompt: "approve",
        output_fn=output.append,
    )
    assert await session.handle(value) is True
    assert "GitHub 代码分析失败" in output[-1]
    assert not (tmp_path / "conversation").exists()


@pytest.mark.asyncio
async def test_fixture_research_is_human_gated_and_default_output_is_concise(
    tmp_path: Path,
) -> None:
    """research is reused, its real Gate is retained, and JSON stays hidden."""
    output: list[str] = []
    input_fn, prompts = _answers(["approve"])
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        input_fn=input_fn,
        output_fn=output.append,
    )
    assert await session.handle("请开始检索今天的论文") is True
    assert any("人工 Gate：CANDIDATE_SELECTION" in line for line in output)
    assert any("论文检索完成" in line for line in output)
    assert not any(line.lstrip().startswith("{") for line in output)
    assert prompts == ["决定 [approve/edit/reject]: "]
    assert (tmp_path / "conversation" / "research").exists()


@pytest.mark.asyncio
async def test_research_current_continue_and_status_are_a_single_safe_continuation(
    tmp_path: Path,
) -> None:
    """one approved paper remains usable within this session only."""
    output: list[str] = []
    input_fn, _prompts = _answers(["approve", "approve"])
    workspace = tmp_path / "conversation"
    session = ConversationSession(
        workspace=workspace,
        mode="fixture",
        router=FixtureIntentRouter(),
        input_fn=input_fn,
        output_fn=output.append,
    )
    await session.handle("开始检索文献")
    await session.handle("当前候选")
    current = output[-1]
    await session.handle("继续找代码")
    insight = output[-1]
    await session.handle("/status")
    status = output[-1]

    assert "PyTorch Classification of Wafer SEM Defects" in current
    assert "research/" in current
    assert "公开 GitHub 代码分析完成" in insight
    assert "固定 commit：" in insight
    assert "已读文件：" in insight
    assert "不是 git clone" in insight
    assert "未生成 patch，未运行 Smoke、训练或第三方代码" in insight
    assert "当前论文=PyTorch Classification of Wafer SEM Defects" in status
    assert set(path.name for path in workspace.iterdir()) == {
        "research",
        "repository-insight",
        "snapshots",
    }


@pytest.mark.asyncio
async def test_failed_analysis_does_not_overwrite_current_paper_context(tmp_path: Path) -> None:
    """a failed turn retains a prior successful single-paper summary."""
    output: list[str] = []
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        output_fn=output.append,
    )
    await session.handle("https://arxiv.org/abs/2608.01234")
    await session.handle("https://arxiv.org/abs/2608.99999")
    assert output[-1] == "arXiv 分析失败：未找到 2608.99999。"
    await session.handle("/current")
    assert "PyTorch Classification of Wafer SEM Defects" in output[-1]
    assert "papers/2608.01234/assessment.json" in output[-1]


@pytest.mark.asyncio
async def test_continue_requires_a_unique_canonical_github_code_url(tmp_path: Path) -> None:
    """absent, multiple, and non-canonical URLs cannot guess a preview target."""
    output: list[str] = []
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        output_fn=output.append,
    )
    await session.handle("/continue")
    assert "暂无当前论文" in output[-1]

    session._current_paper = CurrentPaperSummary(
        paper_id="paper-example",
        title="Example",
        arxiv_id="2608.01234",
        code_urls=["https://github.com/example/one", "https://github.com/example/two"],
        recommendation="HIGH",
        artifact_ref="papers/2608.01234/assessment.json",
    )
    await session.handle("/continue")
    assert "没有唯一的公开 code URL" in output[-1]

    session._current_paper.code_urls = ["https://github.com/example/one/tree/main"]
    await session.handle("/continue")
    assert "不是规范 GitHub 仓库" in output[-1]
    assert not (tmp_path / "conversation").exists()


@pytest.mark.asyncio
async def test_research_reject_is_reported_as_a_human_decision(tmp_path: Path) -> None:
    """rejecting a candidate is not presented as a successful selection."""
    output: list[str] = []
    input_fn, _prompts = _answers(["reject"])
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        input_fn=input_fn,
        output_fn=output.append,
    )
    await session.handle("开始检索文献")
    assert "论文候选已按人工决定拒绝" in output[-1]


@pytest.mark.asyncio
async def test_research_edit_preserves_the_typed_follow_up_prompt(tmp_path: Path) -> None:
    """edit details pass through to the existing typed Approval builder."""
    output: list[str] = []
    input_fn, prompts = _answers(["edit", "paper-arxiv-2608.01234"])
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        input_fn=input_fn,
        output_fn=output.append,
    )
    await session.handle("开始检索文献")
    assert prompts == [
        "决定 [approve/edit/reject]: ",
        "输入要保留的 paper_id (多个 ID 用英文逗号分隔): ",
    ]
    assert "已选择 1 篇" in output[-1]


@pytest.mark.asyncio
async def test_fixture_arxiv_analysis_uses_provider_and_structured_applicability(
    tmp_path: Path,
) -> None:
    """a single fixture paper is fetched by ID and summarized without raw JSON."""
    output: list[str] = []
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        output_fn=output.append,
    )
    await session.handle("https://arxiv.org/abs/2608.01234v1")
    assert "PyTorch Classification of Wafer SEM Defects" in output[-1]
    assert "内容概要：" in output[-1]
    assert "目标论文：是" in output[-1]
    assert "Recommendation：HIGH" in output[-1]
    assert "判断理由：" in output[-1]
    assert "代码状态：当前元数据中已发现公开代码链接" in output[-1]
    assert "风险：" in output[-1]
    assert "产物：papers/2608.01234/assessment.json" in output[-1]
    assert not output[-1].lstrip().startswith("{")
    assert (tmp_path / "conversation" / "papers" / "2608.01234" / "assessment.json").is_file()


@pytest.mark.asyncio
async def test_fixture_arxiv_analysis_scores_non_target_without_code(tmp_path: Path) -> None:
    """direct analysis calls the LLM and fully explains a mismatched paper."""
    output: list[str] = []
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        fixture_xml=Path("tests/conversation/fixtures/arxiv_single_non_target.xml"),
        output_fn=output.append,
    )
    await session.handle("https://arxiv.org/abs/2608.05678")
    assert "论文：Unsupervised Segmentation of Natural Photographs" in output[-1]
    assert "内容概要：" in output[-1]
    assert "目标论文：否" in output[-1]
    assert "Recommendation：REJECT" in output[-1]
    assert "判断理由：" in output[-1]
    assert "代码状态：当前元数据中未发现公开代码链接" in output[-1]
    assert "风险：" in output[-1]
    assert "产物：papers/2608.05678/assessment.json" in output[-1]
    assert (tmp_path / "conversation" / "papers" / "2608.05678" / "assessment.json").is_file()


@pytest.mark.asyncio
async def test_failed_single_paper_schema_does_not_overwrite_existing_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a schema failure leaves a previous artifact and current-paper context intact."""
    output: list[str] = []
    workspace = tmp_path / "conversation"
    session = ConversationSession(
        workspace=workspace,
        mode="fixture",
        router=FixtureIntentRouter(),
        output_fn=output.append,
    )
    await session.handle("https://arxiv.org/abs/2608.01234")
    artifact = workspace / "papers" / "2608.01234" / "assessment.json"
    before = artifact.read_text(encoding="utf-8")
    monkeypatch.setattr(
        session,
        "_paper_llm",
        lambda: ScriptedStructuredLLM(
            outputs={"paper_applicability": {}}, script={"llm.generate": [DelegateStep()]}
        ),
    )

    await session.handle("https://arxiv.org/abs/2608.01234")
    assert "模型返回的结构化字段不完整" in output[-1]
    assert "LLM_SCHEMA_VALIDATION_FAILED" not in output[-1]
    assert artifact.read_text(encoding="utf-8") == before

    await session.handle("/debug on")
    await session.handle("https://arxiv.org/abs/2608.01234")
    assert output[-2].startswith("论文分析失败：模型返回的结构化字段不完整")
    assert output[-1] == "调试错误码：LLM_SCHEMA_VALIDATION_FAILED。"
    assert artifact.read_text(encoding="utf-8") == before
    await session.handle("/current")
    assert "PyTorch Classification of Wafer SEM Defects" in output[-1]


@pytest.mark.asyncio
async def test_fixture_arxiv_analysis_rejects_a_mismatched_missing_id(tmp_path: Path) -> None:
    """a canned provider response cannot impersonate another requested paper ID."""
    output: list[str] = []
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        output_fn=output.append,
    )
    await session.handle("https://arxiv.org/abs/2608.99999")
    assert output[-1] == "arXiv 分析失败：未找到 2608.99999。"


@pytest.mark.asyncio
async def test_repository_insight_reject_and_out_of_scope_do_nothing(
    tmp_path: Path,
) -> None:
    """a rejected snapshot and unrelated request launch no downstream work."""
    output: list[str] = []
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        input_fn=lambda _prompt: "reject",
        output_fn=output.append,
    )
    await session.handle("https://github.com/openai/example")
    assert "已按人工决定取消公开仓库分析" in output[-1]
    assert "未访问 GitHub、未下载源码快照、未调用代码 LLM" in output[-1]
    await session.handle("请删除我的文件并执行任意 shell")
    assert "超出 SEM Research Agent 范围" in output[-1]
    assert not (tmp_path / "conversation").exists()


@pytest.mark.asyncio
async def test_fixture_pipeline_preserves_all_four_typed_human_gates(tmp_path: Path) -> None:
    """the conversation entrypoint delegates to the existing four-Gate Pipeline graph."""
    output: list[str] = []
    input_fn, prompts = _answers(["approve", "approve", "approve", "approve"])
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        input_fn=input_fn,
        output_fn=output.append,
    )
    await session.handle("运行完整流程")
    assert prompts == ["决定 [approve/edit/reject]: "] * 4
    for gate in (
        "CANDIDATE_SELECTION",
        "REPOSITORY_INGEST",
        "PATCH_ACCEPTANCE",
        "RUN_SUBMISSION",
    ):
        assert any(gate in line for line in output)
    assert "完整流程完成：SUCCEEDED，结论：IMPROVED" in output[-1]
    assert "{" not in output[-1]


@pytest.mark.asyncio
async def test_pipeline_reject_is_reported_without_running_downstream(tmp_path: Path) -> None:
    """a rejected first Gate remains an explicit non-successful terminal result."""
    output: list[str] = []
    input_fn, prompts = _answers(["reject"])
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        input_fn=input_fn,
        output_fn=output.append,
    )
    await session.handle("运行完整流程")
    assert prompts == ["决定 [approve/edit/reject]: "]
    assert "完整流程已停止：STOPPED" in output[-1]
    assert "None" not in output[-1]


@pytest.mark.asyncio
async def test_router_failure_is_explicit_and_does_not_fall_back(tmp_path: Path) -> None:
    """a failed ambiguous live/scripted router cannot silently choose an intent."""
    output: list[str] = []
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FailingRouter(),
        output_fn=output.append,
    )
    await session.handle("一个无法由规则识别的普通请求")
    assert output[-1] == "意图路由失败：CONVERSATION_ROUTER_PROVIDER_FAILED。未启动下游工作流。"
    assert not (tmp_path / "conversation").exists()


@pytest.mark.asyncio
async def test_default_cli_supports_three_consecutive_turns_and_exit(tmp_path: Path) -> None:
    """default REPL prompts with sem-agent> and keeps normal output compact."""
    output: list[str] = []
    input_fn, prompts = _answers(
        ["/help", "https://github.com/openai/example", "reject", "/status", "/exit"]
    )
    options = AgentCliOptions(
        mode="fixture",
        router_mode="fixture",
        workspace=tmp_path / "conversation",
        debug_json=False,
    )
    assert await run(options, input_fn=input_fn, output_fn=output.append) == 0
    assert prompts == [
        "sem-agent> ",
        "sem-agent> ",
        "决定 [approve/reject]: ",
        "sem-agent> ",
        "sem-agent> ",
    ]
    assert any("已按人工决定取消公开仓库分析" in line for line in output)
    assert any(
        line.startswith("状态：") and "last_intent=ANALYZE_GITHUB_REPOSITORY" in line
        for line in output
    )
    assert output[-1] == "已退出 SEM Research Agent 对话。"
    assert not any(line.lstrip().startswith("{") for line in output)


@pytest.mark.asyncio
async def test_invalid_gate_decision_is_reprompted_without_ending_session(tmp_path: Path) -> None:
    """ordinary input mistakes cannot bypass or crash the candidate Gate."""
    output: list[str] = []
    input_fn, prompts = _answers(["yes", "approve"])
    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        input_fn=input_fn,
        output_fn=output.append,
    )
    assert await session.handle("开始检索文献") is True
    assert prompts == ["决定 [approve/edit/reject]: "] * 2
    assert "请输入 approve、edit 或 reject。" in output
    assert any("论文检索完成" in line for line in output)
