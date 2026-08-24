"""Concrete runtime and signature contracts for fake-backed ports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import pytest

from tests.contract.base import (
    assert_concrete_protocol_signatures,
    assert_public_protocol_registry,
)
from tests.fakes import (
    InMemoryArtifactStore,
    InMemoryAsyncReader,
    InMemoryExperimentTracker,
    InMemoryUnitOfWork,
    ScriptedAuditRepository,
    ScriptedDatasetCatalog,
    ScriptedEntityRepository,
    ScriptedExperimentExecutor,
    ScriptedPaperProvider,
    ScriptedPatchWorkspace,
    ScriptedRepositoryProvider,
    ScriptedStaticRepositoryAnalyzer,
    ScriptedStructuredLLM,
    ScriptedValidationRunner,
)
from vision_research_ops import ports
from vision_research_ops.domain import ResearchRequest
from vision_research_ops.ports import (
    AdaptationRepository,
    ApprovalRepository,
    ArtifactStore,
    AsyncBinaryReader,
    AuditRepository,
    DatasetCatalog,
    EntityRepository,
    ExperimentExecutor,
    ExperimentRepository,
    ExperimentTracker,
    OperationRepository,
    PaperProvider,
    PaperRepository,
    PatchWorkspace,
    RepositoryProvider,
    RepositorySnapshotRepository,
    ResearchRequestRepository,
    StaticRepositoryAnalyzer,
    StructuredLLM,
    UnitOfWork,
    ValidationRunner,
)


def _public_port_protocols() -> tuple[type[object], ...]:
    """Discover the exported production Protocol set without maintaining a hand list."""
    return tuple(
        candidate
        for name in ports.__all__
        if isinstance(candidate := getattr(ports, name), type)
        and getattr(candidate, "_is_protocol", False)
    )


FAKE_REGISTRY: dict[type[object], Callable[[], object]] = {
    PaperProvider: ScriptedPaperProvider,
    RepositoryProvider: ScriptedRepositoryProvider,
    StaticRepositoryAnalyzer: ScriptedStaticRepositoryAnalyzer,
    StructuredLLM: ScriptedStructuredLLM,
    ArtifactStore: InMemoryArtifactStore,
    DatasetCatalog: ScriptedDatasetCatalog,
    PatchWorkspace: ScriptedPatchWorkspace,
    ValidationRunner: ScriptedValidationRunner,
    ExperimentExecutor: ScriptedExperimentExecutor,
    ExperimentTracker: InMemoryExperimentTracker,
    EntityRepository: lambda: ScriptedEntityRepository(ResearchRequest),
    AuditRepository: ScriptedAuditRepository,
    ResearchRequestRepository: lambda: ScriptedEntityRepository(ResearchRequest),
    PaperRepository: lambda: ScriptedEntityRepository(ResearchRequest),
    RepositorySnapshotRepository: lambda: ScriptedEntityRepository(ResearchRequest),
    AdaptationRepository: lambda: ScriptedEntityRepository(ResearchRequest),
    ExperimentRepository: lambda: ScriptedEntityRepository(ResearchRequest),
    ApprovalRepository: lambda: ScriptedEntityRepository(ResearchRequest),
    UnitOfWork: InMemoryUnitOfWork,
    AsyncBinaryReader: lambda: InMemoryAsyncReader(b""),
}


@pytest.mark.contract
def test_all_port_fakes_satisfy_runtime_protocols_as_a_supplemental_check() -> None:
    """Runtime Protocol checks supplement, but do not replace, concrete signature assertions."""
    assert isinstance(ScriptedPaperProvider(), PaperProvider)
    assert isinstance(ScriptedRepositoryProvider(), RepositoryProvider)
    assert isinstance(ScriptedStaticRepositoryAnalyzer(), StaticRepositoryAnalyzer)
    assert isinstance(ScriptedStructuredLLM(), StructuredLLM)
    assert isinstance(InMemoryArtifactStore(), ArtifactStore)
    assert isinstance(ScriptedDatasetCatalog(), DatasetCatalog)
    assert isinstance(ScriptedPatchWorkspace(), PatchWorkspace)
    assert isinstance(ScriptedValidationRunner(), ValidationRunner)
    assert isinstance(ScriptedExperimentExecutor(), ExperimentExecutor)
    assert isinstance(InMemoryExperimentTracker(), ExperimentTracker)
    assert isinstance(InMemoryUnitOfWork(), UnitOfWork)
    assert isinstance(InMemoryAsyncReader(b""), AsyncBinaryReader)


@pytest.mark.contract
def test_every_public_non_deferred_protocol_has_an_auto_discovered_concrete_fake() -> None:
    """Registry completeness and every concrete signature derive from the exported Protocol set."""
    report = assert_public_protocol_registry(
        FAKE_REGISTRY,
        public_protocols=_public_port_protocols(),
        deferred_protocols=(OperationRepository,),
    )
    assert report.protocol_count == len(FAKE_REGISTRY)
    assert report.implementation_count > 0
    assert report.method_count > report.protocol_count


@pytest.mark.contract
def test_bad_concrete_signature_is_rejected() -> None:
    """The reusable signature assertion fails for a fake with positional ``ctx``."""

    class BadPaper:
        provider_name = "bad-paper"

        async def search(self, query: object, cursor: object, ctx: object) -> object:
            return None

        async def get_by_external_id(self, external_id: object, *, ctx: object) -> object:
            return None

    with pytest.raises(AssertionError, match="positional/keyword-only"):
        assert_concrete_protocol_signatures(PaperProvider, BadPaper())


@pytest.mark.contract
def test_bad_concrete_return_annotation_is_rejected() -> None:
    """Resolved return annotations catch a fake that claims to return the wrong type."""

    class BadReturnPaper:
        provider_name = "bad-return-paper"

        async def search(
            self,
            query: object,
            *,
            cursor: str | None,
            ctx: object,
        ) -> int:
            return 1

        async def get_by_external_id(self, external_id: object, *, ctx: object) -> object | None:
            return None

    with pytest.raises(AssertionError, match="return annotation"):
        assert_concrete_protocol_signatures(PaperProvider, BadReturnPaper())


@pytest.mark.contract
def test_automatic_method_discovery_rejects_a_protocol_method_missing_from_its_fake() -> None:
    """A new Protocol method cannot evade the shared suite through a stale hand-maintained list."""

    class ProtocolWithExtraMethod(Protocol):
        async def present(self, *, ctx: object) -> str:
            """Return one value."""

        async def extra(self, *, ctx: object) -> str:
            """Return one extra value."""

    class MissingExtraMethod:
        async def present(self, *, ctx: object) -> str:
            del ctx
            return "present"

    with pytest.raises(AssertionError, match=r"missing.*extra"):
        assert_concrete_protocol_signatures(ProtocolWithExtraMethod, MissingExtraMethod())


@pytest.mark.contract
def test_registry_rejects_an_unregistered_new_public_protocol() -> None:
    """A coordinator must register every future exported concrete port before tests can pass."""

    class FuturePort(Protocol):
        async def future(self, *, ctx: object) -> str:
            """Return a future capability result."""

    with pytest.raises(AssertionError, match="registry mismatch"):
        assert_public_protocol_registry(
            FAKE_REGISTRY,
            public_protocols=(*_public_port_protocols(), FuturePort),
            deferred_protocols=(OperationRepository,),
        )


@pytest.mark.contract
def test_capability_mechanism_is_not_a_public_port_requirement() -> None:
    """Coordinator decision CD-P0T03-01 keeps test support flags out of public Protocols."""
    assert "CapabilityAware" not in ports.__all__
    assert "CapabilityDeclaration" not in ports.__all__
    assert "PortCapability" not in ports.__all__
    for protocol in (
        PaperProvider,
        RepositoryProvider,
        StaticRepositoryAnalyzer,
        StructuredLLM,
        ArtifactStore,
        DatasetCatalog,
        PatchWorkspace,
        ValidationRunner,
        ExperimentExecutor,
        ExperimentTracker,
        UnitOfWork,
    ):
        assert "supports" not in protocol.__dict__
        assert "capability_declaration" not in getattr(protocol, "__annotations__", {})


@pytest.mark.contract
def test_operation_repository_remains_deferred_and_non_runtime_checkable() -> None:
    """Coordinator decision CD-P0T03-02 forbids an empty runtime Protocol pseudo-check."""
    assert OperationRepository._is_runtime_protocol is False
    with pytest.raises(TypeError):
        isinstance(object(), OperationRepository)
