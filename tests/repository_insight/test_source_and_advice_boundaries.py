"""bounded source and strict advice tests for repository insight workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from vision_research_ops.adapters.repositories import (
    BoundedZipSourceReader,
    GitHubRepositoryProvider,
)
from vision_research_ops.application.services.repository_insight_models import (
    RepositoryAdaptationAdvice,
    RepositoryAdaptationSuggestion,
    RepositoryCodeEvidence,
)
from vision_research_ops.application.services.repository_insight_planner import (
    validate_advice_paths,
)
from vision_research_ops.ports import OperationContext
from vision_research_ops.repository_insight.fixture_repository import (
    FixtureGitHubInsightTransport,
)


def _ctx() -> OperationContext:
    return OperationContext(
        schema_version="1",
        correlation_id="corr-source-reader",
        workflow_id="workflow-source-reader",
        actor_id="pipeline-user",
        idempotency_key="source-reader",
        sensitivity="PUBLIC",
    )


@pytest.mark.asyncio
async def test_source_reader_indexes_canonical_text_and_returns_at_most_8_kib(
    tmp_path: Path,
) -> None:
    """reading stays inside one hash-verified ZIP and never extracts it."""
    provider = GitHubRepositoryProvider(
        artifact_root=tmp_path,
        transport=FixtureGitHubInsightTransport(),
    )
    resolution = await provider.resolve(
        "https://github.com/example/sem-classifier",
        None,
        ctx=_ctx(),
    )
    snapshot = await provider.snapshot(resolution, ctx=_ctx())
    reader = BoundedZipSourceReader(artifact_root=tmp_path)
    index = reader.index(snapshot)
    assert "train.py" in {item.path for item in index.files}
    assert "LICENSE" not in {item.path for item in index.files}
    source = reader.read(snapshot, index, "train.py")
    assert source.returned_bytes <= 8 * 1024
    assert "CrossEntropyLoss" in source.content
    assert list(tmp_path.rglob("train.py")) == []
    with pytest.raises(ValueError):
        reader.read(snapshot, index, "../train.py")


def test_advice_requires_all_evidence_and_targets_to_have_been_read() -> None:
    """strict output cannot invent an unseen source path or execution claim."""
    advice = RepositoryAdaptationAdvice(
        repository_summary="A small public image-classification layout was inspected.",
        adaptation_fit="MEDIUM",
        code_evidence=[
            RepositoryCodeEvidence(path="train.py", observation="A classification loss is used.")
        ],
        suggestions=[
            RepositoryAdaptationSuggestion(
                area="INPUT_CHANNELS",
                target_paths=["model.py"],
                recommendation="Review the input stem for one-channel SEM images.",
                rationale="The public target uses grayscale imagery.",
            )
        ],
        risks=["Only a bounded source subset was inspected."],
        items_to_verify=["Verify tensor shape assumptions."],
        limitations=[
            "No code was executed or patched.",
            "No training was run and improvement is not guaranteed.",
        ],
    )
    with pytest.raises(ValueError, match="was not read"):
        validate_advice_paths(advice, {"train.py"})
    validate_advice_paths(advice, {"train.py", "model.py"})

    with pytest.raises(ValueError, match="shell or execution commands"):
        RepositoryAdaptationAdvice.model_validate(
            advice.model_dump(mode="python")
            | {"risks": ["Run pip install before reviewing the repository."]}
        )
