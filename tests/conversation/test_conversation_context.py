"""context persistence bounded, restartable conversation-context acceptance tests."""
# ruff: noqa: RUF001

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from vision_research_ops.application.services.conversation_context import (
    ConversationContext,
    ConversationContextStoreError,
    CurrentPaperContext,
    CurrentRepositoryContext,
    LocalConversationContextStore,
)
from vision_research_ops.application.services.conversation_intent import (
    ConversationIntent,
    ConversationIntentName,
    FixtureIntentRouter,
)
from vision_research_ops.cli.agent import (
    DEBUG_CONTEXT_PATH,
    LIVE_CONTEXT_PATH,
    parse_options,
)
from vision_research_ops.cli.conversation import ConversationSession

NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


class CountingRouter:
    """Router double proving deterministic memory commands need no LLM routing."""

    def __init__(self) -> None:
        self.call_count = 0

    async def route(self, message: str) -> ConversationIntent:
        del message
        self.call_count += 1
        return ConversationIntent(intent=ConversationIntentName.OUT_OF_SCOPE)


def _paper() -> CurrentPaperContext:
    return CurrentPaperContext(
        paper_id="paper-arxiv-2608.01234",
        title="PyTorch Classification of Wafer SEM Defects",
        arxiv_id="2608.01234",
        code_urls=["https://github.com/Example/SEM-Classifier"],
        recommendation="HIGH",
        artifact_ref="papers/2608.01234/assessment.json",
    )


def _repository() -> CurrentRepositoryContext:
    return CurrentRepositoryContext(
        repository_url="https://github.com/example/sem-classifier",
        commit_sha="a" * 40,
        license_spdx="MIT",
        adaptation_fit="HIGH",
        read_files=["README.md", "train.py"],
        result_ref="repository-insight/run-1/repository-insight.json",
    )


def test_context_schema_is_strict_small_versioned_and_round_trips(tmp_path: Path) -> None:
    """only the explicit public working facts survive canonical JSON round-trip."""
    context = ConversationContext(
        current_paper=_paper(),
        current_repository=_repository(),
        last_successful_artifact_ref="repository-insight/run-1/repository-insight.json",
        updated_at=NOW,
    )
    store = LocalConversationContextStore(tmp_path / "session" / "context.json")
    store.save(context)
    loaded = store.load()

    assert loaded.context == context
    assert loaded.restored is True
    assert loaded.warning is None
    assert loaded.context.current_paper is not None
    assert loaded.context.current_paper.code_urls == ["https://github.com/example/sem-classifier"]
    payload = json.loads((tmp_path / "session" / "context.json").read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema_version",
        "current_paper",
        "current_repository",
        "last_successful_artifact_ref",
        "updated_at",
    }
    assert set(payload["current_paper"]) == {
        "schema_version",
        "paper_id",
        "title",
        "arxiv_id",
        "code_urls",
        "recommendation",
        "artifact_ref",
    }
    assert set(payload["current_repository"]) == {
        "schema_version",
        "repository_url",
        "commit_sha",
        "license_spdx",
        "adaptation_fit",
        "read_files",
        "result_ref",
    }
    rendered = json.dumps(payload, ensure_ascii=False).casefold()
    for forbidden in (
        "abstract",
        "readme_content",
        "source_code",
        "prompt",
        "model_response",
        "toolmessage",
        "api_key",
        "user_profile",
        "pending_gate",
    ):
        assert forbidden not in rendered


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": "2",
            "current_paper": None,
            "current_repository": None,
            "last_successful_artifact_ref": None,
            "updated_at": "2026-08-24T12:00:00Z",
        },
        {
            "schema_version": "1",
            "current_paper": None,
            "current_repository": None,
            "last_successful_artifact_ref": "C:/private/result.json",
            "updated_at": "2026-08-24T12:00:00Z",
        },
        {
            "schema_version": "1",
            "current_paper": None,
            "current_repository": None,
            "last_successful_artifact_ref": None,
            "updated_at": "2026-08-24T12:00:00Z",
            "chat_history": ["secret"],
        },
    ],
)
def test_context_rejects_unknown_version_absolute_refs_and_extra_fields(
    payload: dict[str, object],
) -> None:
    """strict validation rejects data outside the bounded document."""
    with pytest.raises(ValidationError):
        ConversationContext.model_validate_json(json.dumps(payload))


def test_nested_context_rejects_noncanonical_paths_and_short_commit() -> None:
    """nested paper/repository facts remain bounded at their own boundary."""
    with pytest.raises(ValidationError):
        CurrentPaperContext(
            paper_id="paper-1",
            title="Paper",
            arxiv_id="2608.01234",
            code_urls=[],
            recommendation="HIGH",
            artifact_ref="C:/private/assessment.json",
        )
    with pytest.raises(ValidationError):
        CurrentRepositoryContext(
            repository_url="https://github.com/example/repository",
            commit_sha="abc123",
            license_spdx="MIT",
            adaptation_fit="HIGH",
            read_files=["../private.py"],
            result_ref="repository-insight/run/result.json",
        )


def test_missing_and_corrupt_context_are_safe_and_clear_is_recoverable(tmp_path: Path) -> None:
    """missing creates nothing; corrupt input is not overwritten until /clear."""
    context_path = tmp_path / "sessions" / "context.json"
    store = LocalConversationContextStore(context_path)
    missing = store.load()
    assert missing.restored is False
    assert missing.warning is None
    assert not context_path.parent.exists()

    context_path.parent.mkdir(parents=True)
    corrupt = b'{"schema_version":"99","api_key":"do-not-repeat"}'
    context_path.write_bytes(corrupt)
    output: list[str] = []
    router = CountingRouter()
    session = ConversationSession(
        workspace=tmp_path / "artifacts",
        context_path=context_path,
        mode="fixture",
        router=router,
        output_fn=output.append,
        clock=lambda: NOW,
    )
    startup = "\n".join(session.introduction())
    assert "会话记忆无法恢复" in startup
    assert str(tmp_path) not in startup
    assert "do-not-repeat" not in startup
    assert context_path.read_bytes() == corrupt


@pytest.mark.asyncio
async def test_corrupt_context_clear_is_deterministic_and_keeps_artifacts(tmp_path: Path) -> None:
    """/clear repairs the context without routing or deleting business evidence."""
    context_path = tmp_path / "sessions" / "context.json"
    context_path.parent.mkdir(parents=True)
    context_path.write_text("not-json", encoding="utf-8")
    artifact = tmp_path / "artifacts" / "papers" / "kept.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("evidence", encoding="utf-8")
    router = CountingRouter()
    output: list[str] = []
    session = ConversationSession(
        workspace=tmp_path / "artifacts",
        context_path=context_path,
        mode="fixture",
        router=router,
        output_fn=output.append,
        clock=lambda: NOW,
    )

    await session.handle("/clear")
    await session.handle("/clear")
    assert router.call_count == 0
    assert artifact.read_text(encoding="utf-8") == "evidence"
    loaded = LocalConversationContextStore(context_path).load()
    assert loaded.warning is None
    assert loaded.context.has_working_context is False
    assert output[-1].startswith("会话记忆已清除")


@pytest.mark.asyncio
async def test_clear_write_failure_preserves_context_and_reports_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/clear cannot report success unless the persisted document was replaced."""
    context_path = tmp_path / "sessions" / "context.json"
    initial = ConversationContext(
        current_paper=_paper(),
        last_successful_artifact_ref=_paper().artifact_ref,
        updated_at=NOW,
    )
    store = LocalConversationContextStore(context_path)
    store.save(initial)
    before = context_path.read_bytes()
    output: list[str] = []
    session = ConversationSession(
        workspace=tmp_path / "artifacts",
        context_path=context_path,
        mode="fixture",
        router=CountingRouter(),
        output_fn=output.append,
        clock=lambda: NOW,
    )
    assert session._context_store is not None

    def fail_save(_context: ConversationContext) -> None:
        raise ConversationContextStoreError("CONVERSATION_CONTEXT_WRITE_FAILED")

    monkeypatch.setattr(session._context_store, "save", fail_save)
    await session.handle("/clear")
    assert output[-1].startswith("会话记忆清除失败")
    assert context_path.read_bytes() == before
    await session.handle("/current")
    assert "PyTorch Classification of Wafer SEM Defects" in output[-1]


@pytest.mark.asyncio
async def test_successful_paper_restores_without_router_or_provider_calls(tmp_path: Path) -> None:
    """a successful paper survives a fresh session using only the local context."""
    context_path = tmp_path / "sessions" / "context.json"
    workspace = tmp_path / "run-one"
    first_output: list[str] = []
    first = ConversationSession(
        workspace=workspace,
        context_path=context_path,
        mode="fixture",
        router=FixtureIntentRouter(),
        output_fn=first_output.append,
        clock=lambda: NOW,
    )
    await first.handle("https://arxiv.org/abs/2608.01234")
    assert "论文：PyTorch Classification" in first_output[-1]
    assert context_path.is_file()

    router = CountingRouter()
    second_output: list[str] = []
    second = ConversationSession(
        workspace=tmp_path / "run-two",
        context_path=context_path,
        mode="fixture",
        router=router,
        output_fn=second_output.append,
        clock=lambda: NOW,
    )
    assert any("已恢复上次成功" in line for line in second.introduction())
    await second.handle("/current")
    await second.handle("/status")

    assert router.call_count == 0
    assert "PyTorch Classification of Wafer SEM Defects" in second_output[-2]
    assert "当前仓库：暂无" in second_output[-2]
    assert "当前论文=PyTorch Classification of Wafer SEM Defects" in second_output[-1]
    assert not (tmp_path / "run-two").exists()


@pytest.mark.asyncio
async def test_failed_paper_does_not_replace_last_successful_persisted_context(
    tmp_path: Path,
) -> None:
    """not-found/schema/provider-style failure paths preserve last success bytes."""
    context_path = tmp_path / "sessions" / "context.json"
    output: list[str] = []
    session = ConversationSession(
        workspace=tmp_path / "artifacts",
        context_path=context_path,
        mode="fixture",
        router=FixtureIntentRouter(),
        output_fn=output.append,
        clock=lambda: NOW,
    )
    await session.handle("2608.01234")
    before = context_path.read_bytes()
    await session.handle("2608.99999")

    assert output[-1] == "arXiv 分析失败：未找到 2608.99999。"
    assert context_path.read_bytes() == before
    assert LocalConversationContextStore(context_path).load().context.current_paper == _paper()


@pytest.mark.asyncio
async def test_context_write_failure_is_explicit_but_business_result_remains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a failed atomic save cannot turn a completed analysis into fake recovery."""
    context_path = tmp_path / "sessions" / "context.json"
    output: list[str] = []
    session = ConversationSession(
        workspace=tmp_path / "artifacts",
        context_path=context_path,
        mode="fixture",
        router=FixtureIntentRouter(),
        output_fn=output.append,
        clock=lambda: NOW,
    )

    def fail_save(_context: ConversationContext) -> None:
        raise ConversationContextStoreError("CONVERSATION_CONTEXT_WRITE_FAILED")

    assert session._context_store is not None
    monkeypatch.setattr(session._context_store, "save", fail_save)
    await session.handle("2608.01234")

    assert any("论文：PyTorch Classification" in line for line in output)
    assert output[-1].startswith("本次业务结果已完成，但会话记忆未保存")
    assert (tmp_path / "artifacts" / "papers" / "2608.01234" / "assessment.json").is_file()
    assert not context_path.exists()


def test_agent_default_context_paths_are_stable_isolated_and_injectable() -> None:
    """live/debug defaults do not follow timestamp artifact workspaces."""
    live = parse_options([])
    debug = parse_options(["--mode", "fixture", "--router", "fixture"])
    custom = parse_options(["--context-path", "var/test-session/context.json"])

    assert live.context_path == LIVE_CONTEXT_PATH
    assert debug.context_path == DEBUG_CONTEXT_PATH
    assert live.context_path != debug.context_path
    assert custom.context_path == Path("var/test-session/context.json")
    assert live.workspace == debug.workspace == Path("var/conversation")
