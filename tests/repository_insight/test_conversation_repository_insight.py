"""conversation continuation and failure-preservation checks for repository insight workflow."""
# ruff: noqa: RUF001

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from vision_research_ops.application.services.conversation_context import (
    LocalConversationContextStore,
)
from vision_research_ops.application.services.conversation_intent import (
    ConversationIntent,
    ConversationIntentName,
    FixtureIntentRouter,
)
from vision_research_ops.cli.conversation import ConversationSession, CurrentPaperSummary
from vision_research_ops.ports import LLMError, make_failure


class RecordingRouter:
    """Prove ambiguous deterministic input never reaches the LLM intent router."""

    def __init__(self) -> None:
        self.call_count = 0

    async def route(self, message: str) -> ConversationIntent:
        del message
        self.call_count += 1
        return ConversationIntent(intent=ConversationIntentName.OUT_OF_SCOPE)


@pytest.mark.asyncio
async def test_ambiguous_github_targets_have_zero_router_gate_or_repository_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_repository_insight_harness,
) -> None:
    """regression: two different repositories are rejected before every side effect."""
    harness = make_repository_insight_harness(root=tmp_path / "conversation")
    router = RecordingRouter()
    dependency_builds = 0
    input_calls = 0

    def dependencies(**_kwargs: object) -> object:
        nonlocal dependency_builds
        dependency_builds += 1
        return harness.dependencies

    def input_fn(_prompt: str) -> str:
        nonlocal input_calls
        input_calls += 1
        raise AssertionError("ambiguous input must not reach a human Gate")

    monkeypatch.setattr(
        "vision_research_ops.cli.conversation.build_repository_insight_dependencies",
        dependencies,
    )
    output: list[str] = []
    session = ConversationSession(
        workspace=harness.workspace,
        mode="fixture",
        router=router,
        input_fn=input_fn,
        output_fn=output.append,
    )

    assert await session.handle(
        "比较 https://github.com/example/one 和 https://github.com/example/two"
    )
    assert output == [
        "目标不明确：一次只提供一个 arXiv 论文或一个规范 GitHub 仓库地址；"
        "本轮未启动任何下游工作流。"
    ]
    assert router.call_count == 0
    assert dependency_builds == 0
    assert input_calls == 0
    assert harness.transport.call_count == 0
    assert not harness.workspace.exists()


@pytest.mark.asyncio
async def test_direct_url_and_current_paper_continue_use_the_same_canonical_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """direct and paper-context URLs call the exact same repository insight method."""
    seen: list[str | None] = []

    async def record(value: str | None) -> None:
        seen.append(value)

    session = ConversationSession(
        workspace=tmp_path / "conversation",
        mode="fixture",
        router=FixtureIntentRouter(),
        output_fn=lambda _message: None,
    )
    monkeypatch.setattr(session, "_run_repository_insight", record)
    await session.handle("https://github.com/example/sem-classifier")
    session._current_paper = CurrentPaperSummary(
        paper_id="paper-public-1",
        title="Public SEM classification paper",
        arxiv_id="2608.01234",
        code_urls=["https://github.com/example/sem-classifier"],
        recommendation="HIGH",
        artifact_ref="papers/2608.01234/assessment.json",
    )
    await session.handle("/continue")
    assert seen == [
        "https://github.com/example/sem-classifier",
        "https://github.com/example/sem-classifier",
    ]


@pytest.mark.asyncio
async def test_code_llm_failure_is_explicit_and_preserves_existing_conversation_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_repository_insight_harness,
) -> None:
    """a schema/provider failure cannot replace current paper or last good artifact."""
    harness = make_repository_insight_harness()

    class FailingPlanner:
        async def analyze(self, **_kwargs: object) -> object:
            raise LLMError(
                make_failure(
                    code="REPOSITORY_INSIGHT_SCHEMA_FAILED",
                    category="REPOSITORY_INSIGHT_LLM",
                    message="The strict repository advice schema was not satisfied.",
                    retryable=False,
                    ctx=None,
                )
            )

    harness.dependencies.planner = cast(object, FailingPlanner())
    monkeypatch.setattr(
        "vision_research_ops.cli.conversation.build_repository_insight_dependencies",
        lambda **_kwargs: harness.dependencies,
    )
    output: list[str] = []
    session = ConversationSession(
        workspace=harness.workspace,
        mode="fixture",
        router=FixtureIntentRouter(),
        input_fn=lambda _prompt: "approve",
        output_fn=output.append,
    )
    current = CurrentPaperSummary(
        paper_id="paper-public-1",
        title="Previously analyzed paper",
        arxiv_id="2608.01234",
        code_urls=["https://github.com/example/sem-classifier"],
        recommendation="HIGH",
        artifact_ref="papers/2608.01234/assessment.json",
    )
    session._current_paper = current
    session._last_artifact_ref = current.artifact_ref

    await session.handle("https://github.com/example/sem-classifier")
    assert "REPOSITORY_INSIGHT_SCHEMA_FAILED" in output[-1]
    assert "当前论文上下文和最近成功产物保持不变" in output[-1]
    assert session._current_paper is current
    assert session._last_artifact_ref == current.artifact_ref
    assert not list(harness.workspace.glob("repository-insight/*/repository-insight.json"))


@pytest.mark.asyncio
async def test_completed_repository_restores_reject_preserves_and_new_paper_clears_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_repository_insight_harness,
) -> None:
    """context persistence persist only completed insight and never bind it to a new paper."""
    harness = make_repository_insight_harness(root=tmp_path / "run-one")
    monkeypatch.setattr(
        "vision_research_ops.cli.conversation.build_repository_insight_dependencies",
        lambda **_kwargs: harness.dependencies,
    )
    context_path = tmp_path / "sessions" / "context.json"
    output: list[str] = []
    session = ConversationSession(
        workspace=harness.workspace,
        context_path=context_path,
        mode="fixture",
        router=FixtureIntentRouter(),
        input_fn=lambda _prompt: "approve",
        output_fn=output.append,
    )
    await session.handle("2608.01234")
    await session.handle("https://github.com/example/sem-classifier")

    persisted = LocalConversationContextStore(context_path).load().context
    assert persisted.current_paper is not None
    assert persisted.current_repository is not None
    assert persisted.current_repository.repository_url == (
        "https://github.com/example/sem-classifier"
    )
    assert len(persisted.current_repository.commit_sha) == 40
    assert persisted.current_repository.read_files
    assert persisted.last_successful_artifact_ref == persisted.current_repository.result_ref

    provider_calls = harness.transport.call_count
    router = RecordingRouter()
    restored_output: list[str] = []
    restored = ConversationSession(
        workspace=tmp_path / "run-two",
        context_path=context_path,
        mode="fixture",
        router=router,
        input_fn=lambda _prompt: (_ for _ in ()).throw(
            AssertionError("current/status must not ask for approval")
        ),
        output_fn=restored_output.append,
    )
    await restored.handle("/current")
    await restored.handle("/status")
    assert router.call_count == 0
    assert harness.transport.call_count == provider_calls
    assert "当前仓库：https://github.com/example/sem-classifier" in restored_output[-2]
    assert "当前仓库=https://github.com/example/sem-classifier" in restored_output[-1]
    assert not (tmp_path / "run-two").exists()

    before_reject = context_path.read_bytes()
    rejected = ConversationSession(
        workspace=tmp_path / "run-reject",
        context_path=context_path,
        mode="fixture",
        router=FixtureIntentRouter(),
        input_fn=lambda _prompt: "reject",
        output_fn=lambda _message: None,
    )
    await rejected.handle("https://github.com/example/sem-classifier")
    assert context_path.read_bytes() == before_reject
    assert harness.transport.call_count == provider_calls

    new_paper = ConversationSession(
        workspace=tmp_path / "run-three",
        context_path=context_path,
        mode="fixture",
        router=FixtureIntentRouter(),
        fixture_xml=Path("tests/conversation/fixtures/arxiv_single_non_target.xml"),
        output_fn=lambda _message: None,
    )
    await new_paper.handle("2608.05678")
    replaced = LocalConversationContextStore(context_path).load().context
    assert replaced.current_paper is not None
    assert replaced.current_paper.arxiv_id == "2608.05678"
    assert replaced.current_repository is None
    assert replaced.last_successful_artifact_ref == "papers/2608.05678/assessment.json"
