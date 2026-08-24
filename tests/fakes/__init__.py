"""Deterministic scripted fakes for SEM Research Agent application and graph tests."""

from .persistence import (
    InMemoryOperationRepository,
    InMemoryUnitOfWork,
    ScriptedAuditRepository,
    ScriptedEntityRepository,
)
from .platform import (
    InMemoryArtifactStore,
    InMemoryAsyncReader,
    InMemoryExperimentTracker,
    ScriptedDatasetCatalog,
    ScriptedExperimentExecutor,
    ScriptedPatchWorkspace,
    ScriptedValidationRunner,
)
from .providers import (
    ScriptedPaperProvider,
    ScriptedRepositoryProvider,
    ScriptedStaticRepositoryAnalyzer,
    ScriptedStructuredLLM,
)
from .script import (
    CallRecord,
    CancelledStep,
    DelegateStep,
    FailureStep,
    FrozenClock,
    IdempotencyLedger,
    ReturnStep,
    ScriptedPort,
    TimeoutStep,
    safe_payload,
)

__all__ = [
    "CallRecord",
    "CancelledStep",
    "DelegateStep",
    "FailureStep",
    "FrozenClock",
    "IdempotencyLedger",
    "InMemoryArtifactStore",
    "InMemoryAsyncReader",
    "InMemoryExperimentTracker",
    "InMemoryOperationRepository",
    "InMemoryUnitOfWork",
    "ReturnStep",
    "ScriptedAuditRepository",
    "ScriptedDatasetCatalog",
    "ScriptedEntityRepository",
    "ScriptedExperimentExecutor",
    "ScriptedPaperProvider",
    "ScriptedPatchWorkspace",
    "ScriptedPort",
    "ScriptedRepositoryProvider",
    "ScriptedStaticRepositoryAnalyzer",
    "ScriptedStructuredLLM",
    "ScriptedValidationRunner",
    "TimeoutStep",
    "safe_payload",
]
