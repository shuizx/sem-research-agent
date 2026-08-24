"""Fixed repository workflow URL, GitHub boundary, and static analyzer tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from vision_research_ops.adapters.repositories import (
    GitHubRepositoryProvider,
    ZipStaticRepositoryAnalyzer,
)
from vision_research_ops.application.services.repository_models import (
    normalize_github_repository_url,
)
from vision_research_ops.ports import OperationContext, ProviderError, RepositoryPolicy

from .conftest import (
    FIXED_NOW,
    FixtureGitHubTransport,
    regression_repository_archive,
    repository_archive,
)


def _ctx() -> OperationContext:
    return OperationContext(
        schema_version="1",
        correlation_id="corr-repository-adapter",
        workflow_id="workflow-repository-1",
        actor_id="pipeline-user",
        idempotency_key="repository-adapter-test",
        sensitivity="PUBLIC",
    )


def test_github_url_normalization_is_narrow_and_canonical() -> None:
    """Only a credential-free owner/repository HTTPS URL crosses the provider boundary."""
    locator = normalize_github_repository_url("https://github.com/Example/Sem-Classifier.git")
    assert locator.canonical_url == "https://github.com/example/sem-classifier"
    assert locator.owner == "example"
    assert locator.name == "sem-classifier"


@pytest.mark.parametrize(
    "value",
    [
        "http://github.com/example/repo",
        "https://user:secret@github.com/example/repo",
        "https://github.com/example/repo/issues",
        "https://github.com/example/repo?tab=readme",
        "https://github.com/example/%2e%2e",
        "https://github.com/foo/.",
        "https://github.com/foo/..",
        "https://github.com/foo/..git",
        "https://evil.example/example/repo",
    ],
)
def test_github_url_normalization_rejects_non_repository_targets(value: str) -> None:
    """Repository links cannot smuggle credentials, alternate hosts, or extra paths."""
    with pytest.raises(ValueError):
        normalize_github_repository_url(value)


@pytest.mark.asyncio
async def test_github_provider_pins_full_sha_metadata_and_relative_snapshot(
    tmp_path: Path,
) -> None:
    """The real adapter uses fixed API endpoints and never exposes a host artifact path."""
    transport = FixtureGitHubTransport(repository_archive())
    provider = GitHubRepositoryProvider(
        artifact_root=tmp_path,
        transport=transport,
        clock=lambda: FIXED_NOW,
    )
    repository = await provider.resolve(
        "https://github.com/example/sem-classifier",
        None,
        ctx=_ctx(),
    )
    metadata = await provider.fetch_metadata(repository, ctx=_ctx())
    snapshot = await provider.snapshot(repository, ctx=_ctx())

    assert repository.commit_sha == "a" * 40
    assert metadata.license_spdx == "MIT"
    assert metadata.languages == {"Python": 4096}
    assert snapshot.uri.startswith("snapshots/")
    assert Path(snapshot.uri).is_absolute() is False
    assert (tmp_path / snapshot.uri).is_file()
    assert all(
        url.startswith("https://api.github.com/repos/example/sem-classifier")
        for url, _ in transport.calls
    )


@pytest.mark.asyncio
async def test_github_provider_rejects_abbreviated_or_invalid_commit_sha(tmp_path: Path) -> None:
    """A branch or abbreviated provider response cannot become an immutable snapshot."""
    transport = FixtureGitHubTransport(repository_archive(), commit_sha="abc123")
    provider = GitHubRepositoryProvider(artifact_root=tmp_path, transport=transport)
    with pytest.raises(ProviderError) as raised:
        await provider.resolve(
            "https://github.com/example/sem-classifier",
            None,
            ctx=_ctx(),
        )
    assert raised.value.failure.code == "GITHUB_COMMIT_SHA_INVALID"


@pytest.mark.asyncio
async def test_static_analyzer_recognizes_plain_pytorch_without_obeying_readme(
    tmp_path: Path,
) -> None:
    """README prompt-like prose remains inert while source evidence is profiled."""
    transport = FixtureGitHubTransport(repository_archive())
    provider = GitHubRepositoryProvider(
        artifact_root=tmp_path,
        transport=transport,
        clock=lambda: FIXED_NOW,
    )
    repository = await provider.resolve(
        "https://github.com/example/sem-classifier",
        None,
        ctx=_ctx(),
    )
    snapshot = await provider.snapshot(repository, ctx=_ctx())
    analyzer = ZipStaticRepositoryAnalyzer(artifact_root=tmp_path)
    analysis = await analyzer.analyze(
        snapshot,
        RepositoryPolicy(
            schema_version="1",
            policy_id="pipeline-pytorch-classification",
            policy_version="1",
        ),
        ctx=_ctx(),
    )

    assert analysis.supported is True
    assert analysis.entrypoint_candidates == ["train.py"]
    assert analysis.data_loader_candidates == ["data.py", "train.py"]
    assert analysis.dependency_files == ["requirements.txt"]
    assert "STRUCTURE:PLAIN_PYTORCH" in analysis.framework_evidence
    assert analysis.command_candidates == []
    assert analysis.dangerous_patterns == []
    assert analysis.license_spdx == "MIT"


@pytest.mark.asyncio
async def test_static_analyzer_reports_source_execution_primitive_as_unsupported(
    tmp_path: Path,
) -> None:
    """Ordinary static policy blocks a fixture source file containing os.system."""
    transport = FixtureGitHubTransport(repository_archive(dangerous_source=True))
    provider = GitHubRepositoryProvider(artifact_root=tmp_path, transport=transport)
    repository = await provider.resolve(
        "https://github.com/example/sem-classifier",
        None,
        ctx=_ctx(),
    )
    snapshot = await provider.snapshot(repository, ctx=_ctx())
    analysis = await ZipStaticRepositoryAnalyzer(artifact_root=tmp_path).analyze(
        snapshot,
        RepositoryPolicy(
            schema_version="1",
            policy_id="pipeline-pytorch-classification",
            policy_version="1",
        ),
        ctx=_ctx(),
    )
    assert analysis.supported is False
    assert [item.rule_id for item in analysis.dangerous_patterns] == ["OS_SYSTEM"]


@pytest.mark.asyncio
async def test_static_analyzer_rejects_generic_pytorch_regression_layout(
    tmp_path: Path,
) -> None:
    """DataLoader plus an arbitrary linear output is not classification evidence."""
    transport = FixtureGitHubTransport(regression_repository_archive())
    provider = GitHubRepositoryProvider(artifact_root=tmp_path, transport=transport)
    repository = await provider.resolve(
        "https://github.com/example/regressor",
        None,
        ctx=_ctx(),
    )
    snapshot = await provider.snapshot(repository, ctx=_ctx())
    analysis = await ZipStaticRepositoryAnalyzer(artifact_root=tmp_path).analyze(
        snapshot,
        RepositoryPolicy(
            schema_version="1",
            policy_id="pipeline-pytorch-classification",
            policy_version="1",
        ),
        ctx=_ctx(),
    )
    assert analysis.supported is False
    assert not any(
        item.startswith(("CLASSIFICATION_LOSS:", "MODEL_HEAD:"))
        for item in analysis.framework_evidence
    )
