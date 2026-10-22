"""C1-C7 ordinary synthetic-profile and supervisor-context integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vision_research_ops.adapters.llm.supervisor import ScriptedSupervisorPolicy
from vision_research_ops.application.services.conversation_intent import FixtureIntentRouter
from vision_research_ops.application.services.dataset_profiling import profile_dataset
from vision_research_ops.application.services.target_profile import load_target_profile
from vision_research_ops.cli.agent import AgentCliOptions, run
from vision_research_ops.cli.conversation import ConversationSession
from vision_research_ops.settings import Settings


def generated(tmp_path: Path) -> Path:
    """Generate evidence from the existing synthetic profiler, not a second profiler."""
    return profile_dataset(
        Path("fixtures/datasets/synthetic_sem_images"), output_root=tmp_path / "profiles"
    ).output_path


def session(tmp_path: Path, *, profile: Path | None = None, output=None):
    return ConversationSession(
        workspace=tmp_path / "work",
        mode="fixture",
        router=FixtureIntentRouter(),
        supervisor_policy=ScriptedSupervisorPolicy(),
        settings=Settings(dataset_profile_path=profile),
        context_path=tmp_path / "context.json",
        input_fn=lambda _: "approve",
        output_fn=output or (lambda _: None),
    )


@pytest.mark.asyncio
async def test_generated_profiles_change_actual_tool_evidence(tmp_path: Path) -> None:
    """Two valid synthetic profile facts change the actual Code target-tool output hash."""
    path = generated(tmp_path)
    first = load_target_profile(path)
    s = session(tmp_path, profile=path)
    await s.handle("https://github.com/example/sem-classifier")
    first_context = json.loads((tmp_path / "context.json").read_text())
    trace1 = next((tmp_path / "work" / "repository-insight").rglob("tool-trace.json"), None)
    # The Code trace location is recorded alongside its strict result.
    result_files = list((tmp_path / "work").rglob("*.json"))
    target_hashes1 = [
        event["output_hash"]
        for file in result_files
        for event in json.loads(file.read_text()).get("events", [])
        if event.get("tool_name") == "inspect_target_profile"
    ]
    assert target_hashes1
    del trace1
    payload = json.loads(path.read_text())
    payload["channels"] = 3
    payload["modality"] = "RGB"
    payload["label_schema"] = payload["label_schema"][:2]
    payload["split_policy"]["test_fraction"] = 0.2
    path.write_text(json.dumps(payload))
    second = load_target_profile(path)
    assert first != second
    assert second.input_channels == 3 and second.class_count == 2
    restarted = session(tmp_path, profile=path)
    assert restarted._current_repository is None
    assert restarted._last_artifact_ref is None
    await restarted.handle("https://github.com/example/sem-classifier")
    target_hashes2 = [
        event["output_hash"]
        for file in (tmp_path / "work").rglob("*.json")
        for event in json.loads(file.read_text()).get("events", [])
        if event.get("tool_name") == "inspect_target_profile"
    ]
    assert len(set(target_hashes2)) == 2
    assert first_context["target_profile"]["input_channels"] == 1
    safe = json.dumps(second.tool_facts())
    for forbidden in (
        "dataset-synthetic",
        "label-bridge",
        "wafer_group",
        "dataset-handle",
        str(tmp_path),
    ):
        assert forbidden not in safe
    restored = session(tmp_path)
    assert restored._target_profile == second
    assert restored._current_repository is not None
    await restored.handle("/clear")
    assert session(tmp_path)._target_profile is None
    assert session(tmp_path)._current_repository is None


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [True, False])
async def test_invalid_explicit_profile_fails_before_router_or_model(tmp_path: Path, missing: bool):
    path = tmp_path / "private-path.json"
    if not missing:
        path.write_text('{"secret": "not a DatasetProfile"}')
    output = []
    code = await run(
        AgentCliOptions(
            mode="fixture",
            router_mode="fixture",
            workspace=tmp_path / "work",
            debug_json=False,
            dataset_profile_path=path,
        ),
        settings=Settings(),
        input_fn=lambda _: pytest.fail("must not enter REPL"),
        output_fn=output.append,
    )
    assert code == 2
    assert "DATASET_PROFILE_INVALID_OR_MISSING" in output[0]
    assert str(tmp_path) not in str(output)
    assert not (tmp_path / "work").exists()


@pytest.mark.asyncio
async def test_changed_explicit_file_cannot_restore_stale_snapshot(tmp_path: Path):
    path = generated(tmp_path)
    output = []
    s = session(tmp_path, profile=path, output=output.append)
    path.write_text("{}")
    await s.handle("https://github.com/example/sem-classifier")
    assert "DATASET_PROFILE_INVALID_OR_MISSING" in output[-1]
    assert not (tmp_path / "work").exists()
    with pytest.raises(ValueError, match=r"^DATASET_PROFILE_INVALID_OR_MISSING$"):
        session(tmp_path, profile=path)


@pytest.mark.asyncio
async def test_followup_uses_request_budget_and_asks_when_ambiguous(tmp_path: Path):
    from vision_research_ops.application.services.supervisor_models import SupervisorError

    s = session(tmp_path)
    await s.handle("检索最新论文")

    class ExhaustBeforeDispatch:
        async def decide(self, goal, observations, budget):
            for _ in range(25):
                budget.consume()
            raise AssertionError("must fail at the shared default-24 limit")

    s._supervisor_policy = ExhaustBeforeDispatch()
    await s.handle("/continue")
    traces = [json.loads(p.read_text()) for p in (tmp_path / "work" / "supervisor").glob("*.json")]
    exhausted = [
        t
        for t in traces
        if t["failure_code"] == str(SupervisorError("SUPERVISOR_MODEL_BUDGET_EXHAUSTED"))
    ]
    assert len(exhausted) == 1
    assert exhausted[0]["model_calls"] == 24 and exhausted[0]["delegations"] == 0
    output = []
    fresh = session(tmp_path / "fresh", output=output.append)
    for goal in ("继续", "分析当前论文", "分析当前仓库", "/continue"):
        await fresh.handle(goal)
    assert len(output) == 4
    assert not (tmp_path / "fresh" / "work").exists()


@pytest.mark.asyncio
async def test_continue_restart_current_repository_and_trace(tmp_path: Path):
    output = []
    s = session(tmp_path, output=output.append)
    await s.handle("/continue")
    assert not (tmp_path / "work").exists()
    await s.handle("检索最新论文")
    assert s._current_paper is not None
    assert s._current_repository is None
    await session(tmp_path).handle("/continue")
    restored = session(tmp_path)
    assert restored._current_repository.repository_url == s._current_paper.code_urls[0]
    await restored.handle("分析当前仓库")
    traces = list((tmp_path / "work" / "supervisor").glob("*.json"))
    assert len(traces) == 3
    for path in traces:
        trace = json.loads(path.read_text())
        assert trace["delegations"] == 1
        assert trace["model_calls"] == 0  # explicitly scripted transports
        assert "summary" not in path.read_text()
        assert str(tmp_path) not in path.read_text()
    # New paper with no code must not inherit the unrelated current repository.
    await restored.handle("https://arxiv.org/abs/2608.01234")
    assert restored._current_repository is None
    restored._current_paper.code_urls = []
    before = len(list((tmp_path / "work" / "supervisor").glob("*.json")))
    await restored.handle("/continue")
    assert len(list((tmp_path / "work" / "supervisor").glob("*.json"))) == before
