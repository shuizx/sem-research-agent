"""Ordinary Chinese prose next to public URLs and invalid-target safe responses."""
# ruff: noqa: RUF001

import pytest

from vision_research_ops.application.services.conversation_intent import (
    ConversationIntentName,
    FixtureIntentRouter,
    deterministic_intent,
    has_ambiguous_supported_targets,
)
from vision_research_ops.application.services.supervisor_models import SupervisorProposal
from vision_research_ops.cli.conversation import ConversationSession


@pytest.mark.parametrize(
    "message,field,expected",
    [
        ("仅分析论文 https://arxiv.org/abs/2607.07965，不分析代码", "arxiv_id", "2607.07965"),
        ("分析 https://arxiv.org/pdf/2607.07965v2.pdf；仅分析论文", "arxiv_id", "2607.07965"),
        ("分析“https://arxiv.org/abs/2607.07965”这篇论文", "arxiv_id", "2607.07965"),
        (
            "分析“https://github.com/kuangliu/pytorch-cifar”这个仓库",
            "github_url",
            "https://github.com/kuangliu/pytorch-cifar",
        ),
        (
            "分析 https://github.com/Kuangliu/Pytorch-Cifar，给出建议",
            "github_url",
            "https://github.com/kuangliu/pytorch-cifar",
        ),
        (
            "分析https://github.com/Kuangliu/Pytorch-Cifar的代码",
            "github_url",
            "https://github.com/kuangliu/pytorch-cifar",
        ),
    ],
)
def test_chinese_url_boundaries(message, field, expected):
    assert getattr(deterministic_intent(message), field) == expected


def test_chinese_separated_multiple_targets_still_require_selection():
    message = "https://arxiv.org/abs/2607.07965，以及https://github.com/kuangliu/pytorch-cifar"
    assert has_ambiguous_supported_targets(message)
    assert deterministic_intent(message).intent == ConversationIntentName.OUT_OF_SCOPE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "https://arxiv.org/abs/2607.07965?view=full",
        "https://[arxiv.org/abs/2607.07965",
        "https://github.com/kuangliu/pytorch-cifar/tree/main，分析这个路径",
    ],
)
async def test_invalid_target_is_safe_and_does_not_invoke_supervisor(tmp_path, message):
    class NoCalls:
        async def decide(self, *args):
            pytest.fail("invalid explicit target cannot reach a live supervisor")

    intent = deterministic_intent(message)
    assert intent.intent == ConversationIntentName.OUT_OF_SCOPE
    assert intent.arxiv_id is intent.github_url is None
    outputs = []
    session = ConversationSession(
        workspace=tmp_path / "work",
        mode="fixture",
        router=FixtureIntentRouter(),
        supervisor_policy=NoCalls(),
        input_fn=lambda _: pytest.fail("no gate"),
        output_fn=outputs.append,
    )
    assert await session.handle(message)
    assert "失败" in outputs[-1]
    assert not (tmp_path / "work").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "帮我检索 arxiv 上最新的计算机视觉论文",
        "请分析 arxiv 上的这篇论文，但我还没有提供它的地址或正确编号，请先告诉我如何提供地址。",
    ],
)
async def test_arxiv_mention_without_url_reaches_supervisor(tmp_path, message):
    calls = []

    class RecordingPolicy:
        async def decide(self, goal, observations, budget):
            calls.append((goal, observations))
            return SupervisorProposal(action="finish")

    assert deterministic_intent(message) is None
    session = ConversationSession(
        workspace=tmp_path / "work",
        mode="fixture",
        router=FixtureIntentRouter(),
        supervisor_policy=RecordingPolicy(),
        input_fn=lambda _: pytest.fail("no gate"),
        output_fn=lambda _: None,
    )
    assert await session.handle(message)
    assert calls == [(message, [])]
