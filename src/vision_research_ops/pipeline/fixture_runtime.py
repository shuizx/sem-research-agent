"""Offline fixture assembly and research-to-evaluation child-graph driver."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Literal, Protocol, cast
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command
from pydantic import BaseModel

from vision_research_ops.adaptation import (
    FixturePatchTool,
    FixtureSmokeRunner,
    FixtureToolCallingAdaptationPlanner,
)
from vision_research_ops.adapters.execution.local_training import LocalTrainingExecutor
from vision_research_ops.adapters.llm import FixtureStructuredLLM
from vision_research_ops.adapters.papers import ArxivPaperProvider
from vision_research_ops.adapters.repositories import (
    GitHubRepositoryProvider,
    ZipStaticRepositoryAnalyzer,
)
from vision_research_ops.application.adaptation_runtime import (
    AdaptationDependencies,
    AdaptationPlanner,
)
from vision_research_ops.application.evaluation_runtime import EvaluationDependencies
from vision_research_ops.application.pipeline_runtime import (
    DecisionProvider,
    PipelineStageOutcome,
)
from vision_research_ops.application.repository_runtime import RepositoryDependencies
from vision_research_ops.application.research_runtime import ResearchDependencies
from vision_research_ops.application.runtime import InMemoryApprovalRecorder
from vision_research_ops.application.services.adaptation_models import AdaptationResult
from vision_research_ops.application.services.adaptation_store import LocalAdaptationStore
from vision_research_ops.application.services.evaluation_models import (
    EvaluationInitialInput,
    EvaluationResult,
    create_evaluation_state,
)
from vision_research_ops.application.services.evaluation_store import LocalEvaluationStore
from vision_research_ops.application.services.paper_models import default_sem_problem_profile
from vision_research_ops.application.services.paper_store import LocalResearchStore
from vision_research_ops.application.services.pipeline_models import (
    AdaptationEvidence,
    EvaluationEvidence,
    EvaluationMetricEvidence,
    PaperEvidence,
    PipelineDecisionConfig,
    PipelineFailure,
    PipelineGateRecord,
    PipelineScenario,
    PipelineStageName,
    PipelineStageRecord,
    PipelineState,
    PipelineSummary,
    RepositoryEvidence,
    TrainingEvidence,
)
from vision_research_ops.application.services.repository_store import LocalRepositoryStore
from vision_research_ops.application.services.training_freeze import (
    DATASET_REF,
    PREPROCESS_REF,
    SPLIT_REF,
)
from vision_research_ops.application.services.training_models import (
    TrainingBudgetSpec,
    TrainingInput,
    content_hash,
)
from vision_research_ops.application.services.training_store import LocalTrainingStore
from vision_research_ops.application.state import create_initial_state
from vision_research_ops.application.training_runtime import (
    NeverCancelled,
    TrainingDependencies,
)
from vision_research_ops.application.workflows.adaptation import build_adaptation_graph
from vision_research_ops.application.workflows.core import workflow_config
from vision_research_ops.application.workflows.evaluation import build_evaluation_graph
from vision_research_ops.application.workflows.repository import build_repository_graph
from vision_research_ops.application.workflows.research import build_research_graph
from vision_research_ops.application.workflows.training import build_training_graph
from vision_research_ops.domain import (
    Approval,
    DatasetProfile,
    JsonValue,
    QuerySpec,
    ResearchBudget,
    ResearchRequest,
    WorkflowStatus,
)

FIXTURE_NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
FIXED_REPOSITORY_SHA = "a" * 40
_P2_ARCHIVE_FILES = frozenset(
    {"LICENSE", "README.md", "config.yaml", "data.py", "model.py", "train.py"}
)
_LIMITATIONS = [
    "This is a single-user offline pipeline prototype, not a production platform.",
    "Paper, repository, data, and decisions are explicit public or synthetic fixtures.",
    "Smoke validates a no-Torch fixture contract and does not prove arbitrary repository safety.",
    "Training is a standard-library synthetic linear CPU fixture, not real PyTorch training.",
    "The deterministic single-pair result does not establish real SEM or company improvement.",
]
_CHECKPOINT_TYPE_ALLOWLIST = (
    ("vision_research_ops.domain.enums", "WorkflowPhase"),
    ("vision_research_ops.domain.enums", "WorkflowStatus"),
)


class _ChildGraph(Protocol):
    async def ainvoke(
        self,
        input: object,
        config: RunnableConfig | None = None,
        *,
        context: object | None = None,
    ) -> object:
        """Invoke one compiled child graph."""


type _GraphFactory = Callable[[BaseCheckpointSaver[str]], _ChildGraph]


@dataclass(frozen=True, slots=True)
class _ChildStage:
    workflow_id: str
    thread_id: str
    initial_state: object
    dependencies: object
    graph_factory: _GraphFactory


class FixtureGitHubTransport:
    """Serve deterministic GitHub API bytes without any network access."""

    def __init__(self, archive: bytes) -> None:
        self._archive = archive
        self.calls: list[str] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout: int) -> bytes:
        if timeout < 1 or "Authorization" in headers:
            raise ValueError("fixture transport received an unsafe request")
        self.calls.append(url)
        if "/commits/" in url:
            return json.dumps({"sha": FIXED_REPOSITORY_SHA}).encode("utf-8")
        if url.endswith("/languages"):
            return json.dumps({"Python": 4096}).encode("utf-8")
        if "/zipball/" in url:
            return self._archive
        return json.dumps(
            {
                "archived": False,
                "default_branch": "main",
                "fork": False,
                "license": {"spdx_id": "MIT"},
            }
        ).encode("utf-8")


class _FixtureProject:
    """Copy only fixed inputs below the user-selected workspace var directory."""

    def __init__(self, *, source_root: Path, var_root: Path) -> None:
        self.source_root = source_root.resolve()
        self.project_root = (var_root / "fixture-project").resolve()
        if not self.project_root.is_relative_to(var_root.resolve()):
            raise ValueError("fixture project escaped the workspace var root")

    def _copy_tree(self, relative: str) -> None:
        source = self.source_root / Path(*relative.split("/"))
        destination = self.project_root / Path(*relative.split("/"))
        if not source.is_dir() or source.is_symlink():
            raise ValueError("controlled fixture directory is unavailable")
        for item in sorted(path for path in source.rglob("*") if path.is_file()):
            if item.is_symlink():
                raise ValueError("controlled fixture inputs cannot contain symlinks")
            target = destination / item.relative_to(source)
            payload = item.read_bytes()
            if target.exists():
                if not target.is_file() or target.is_symlink() or target.read_bytes() != payload:
                    raise ValueError("existing fixture-project input conflicts with source")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(f"{target.suffix}.tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)

    def prepare_training(self) -> None:
        """Materialize only the training allowlisted fixture inputs."""
        self._copy_tree("fixtures/training")

    def prepare_evaluation(self) -> None:
        """Materialize the evaluation policy after training has completed."""
        self.prepare_training()
        self._copy_tree("fixtures/evaluation")


def _research_request(request_id: str) -> ResearchRequest:
    return ResearchRequest(
        schema_version="1",
        request_id=request_id,
        revision=1,
        title="Daily SEM paper research",
        research_question="Which new CV papers merit a wafer SEM classification experiment?",
        dataset_id="problem-wafer-sem-v1",
        dataset_version="profile-v1",
        query_spec=QuerySpec(
            schema_version="1",
            keywords=[
                "SEM defect classification",
                "microscopy image classification",
                "industrial anomaly classification",
            ],
            domains=["cs.CV"],
            excluded_terms=["survey"],
        ),
        candidate_limit=5,
        budget=ResearchBudget(
            schema_version="1",
            max_provider_pages=2,
            max_provider_records=20,
            max_llm_calls=5,
            max_llm_tokens=6000,
            max_cost_estimate=1.0,
            max_candidate_repositories=5,
            max_adaptation_attempts=1,
            max_workflow_walltime_seconds=180,
        ),
        requested_by="pipeline-user",
        status=WorkflowStatus.PENDING,
        created_at=FIXTURE_NOW,
        updated_at=FIXTURE_NOW,
    )


def _repository_archive(fixture_root: Path) -> bytes:
    if not fixture_root.is_dir() or fixture_root.is_symlink():
        raise ValueError("controlled repository fixture is unavailable")
    observed = {
        path.relative_to(fixture_root).as_posix()
        for path in fixture_root.rglob("*")
        if path.is_file()
    }
    if not _P2_ARCHIVE_FILES.issubset(observed):
        raise ValueError("controlled repository fixture lacks repository profile inputs")
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        for relative in sorted(_P2_ARCHIVE_FILES):
            source = fixture_root / Path(*relative.split("/"))
            if source.is_symlink():
                raise ValueError("controlled repository fixture cannot contain symlinks")
            info = ZipInfo(f"example-sem-classifier/{relative}")
            info.date_time = (2026, 1, 1, 0, 0, 0)
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, source.read_bytes())
    return buffer.getvalue()


def _fixture_hash(project_root: Path, ref: str) -> str:
    path = project_root / Path(*ref.split("/"))
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    return content_hash(normalized)


def _status_text(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _interrupt_payload(result: Mapping[str, object]) -> dict[str, object] | None:
    interrupts = result.get("__interrupt__")
    if not isinstance(interrupts, list | tuple) or len(interrupts) != 1:
        return None
    value = getattr(interrupts[0], "value", None)
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _child_failure(
    stage: PipelineStageName,
    state: Mapping[str, object],
) -> PipelineFailure:
    value = state.get("last_error")
    if isinstance(value, BaseModel):
        raw = value.model_dump(mode="json")
    elif isinstance(value, dict):
        raw = value
    else:
        raw = {}
    code = raw.get("code")
    message = raw.get("message")
    return PipelineFailure(
        code=code if isinstance(code, str) else f"PIPELINE_{stage.upper()}_FAILED",
        message=(
            message[:240]
            if isinstance(message, str) and message.strip()
            else f"The {stage} child Agent did not complete successfully."
        ),
        stage=stage,
    )


class FixturePipelineStageDriver:
    """Drive real child graphs with fixed offline providers and actual resume calls."""

    def __init__(
        self,
        *,
        workspace: Path,
        source_root: Path,
        scenario: PipelineScenario,
        decision_provider: DecisionProvider,
        decision_config: PipelineDecisionConfig,
        event_sink: Callable[[str], None],
        adaptation_planner: AdaptationPlanner | None = None,
        adaptation_planner_mode: Literal["scripted", "dashscope"] = "scripted",
    ) -> None:
        self.workspace = workspace.resolve()
        self.var_root = (self.workspace / "var").resolve()
        if not self.var_root.is_relative_to(self.workspace):
            raise ValueError("pipeline var root escaped the selected workspace")
        self.source_root = source_root.resolve()
        self.scenario = scenario
        self.decision_provider = decision_provider
        self.decision_config = PipelineDecisionConfig.model_validate(
            decision_config.model_dump(mode="python")
        )
        if (self.decision_config.mode != "interactive") != (
            self.decision_provider.scripted_fixture_decisions
        ):
            raise ValueError("decision config mode does not match the DecisionProvider")
        self.event_sink = event_sink
        self.adaptation_planner = adaptation_planner or FixtureToolCallingAdaptationPlanner()
        self.adaptation_planner_mode = adaptation_planner_mode
        if adaptation_planner_mode == "dashscope" and adaptation_planner is None:
            raise ValueError("dashscope planner mode requires an injected live planner")
        self.research_store = LocalResearchStore(self.var_root / "research")
        self.repository_store = LocalRepositoryStore(self.var_root / "repositories")
        self.adaptation_store = LocalAdaptationStore(self.var_root)
        self.fixture_project = _FixtureProject(
            source_root=self.source_root,
            var_root=self.var_root,
        )
        self._training_store: LocalTrainingStore | None = None
        self._evaluation_store: LocalEvaluationStore | None = None

    @property
    def scripted_fixture_decisions(self) -> bool:
        """Expose the explicit decision provenance used by summary generation."""
        return self.decision_provider.scripted_fixture_decisions

    @staticmethod
    def _stage_workflow_id(pipeline_workflow_id: str, stage: PipelineStageName) -> str:
        return f"{pipeline_workflow_id}-{stage}"

    @staticmethod
    def _thread_id(workflow_id: str) -> str:
        return f"thread-{workflow_id}"

    def _training_ref(self, ref: str) -> str:
        return f"fixture-project/var/{ref}"

    def _emit(self, payload: dict[str, object]) -> None:
        self.event_sink(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )

    def _research_stage(self, pipeline_workflow_id: str) -> _ChildStage:
        workflow_id = self._stage_workflow_id(pipeline_workflow_id, "research")
        thread_id = self._thread_id(workflow_id)
        request_id = f"request-{workflow_id}"
        fixture_path = self.source_root / "fixtures" / "pipeline" / "arxiv_feed.xml"

        def transport(_url: str, _timeout_seconds: int) -> bytes:
            return fixture_path.read_bytes()

        dependencies = ResearchDependencies(
            request=_research_request(request_id),
            problem_profile=default_sem_problem_profile(),
            paper_provider=ArxivPaperProvider(
                timeout_seconds=20,
                transport=transport,
                clock=lambda: FIXTURE_NOW,
            ),
            structured_llm=FixtureStructuredLLM(),
            store=self.research_store,
            approval_recorder=InMemoryApprovalRecorder(),
            page_size=10,
            overlap_minutes=60,
            initial_lookback_hours=24,
            clock=lambda: FIXTURE_NOW,
        )
        initial = create_initial_state(
            {
                "schema_version": "1",
                "workflow_id": workflow_id,
                "thread_id": thread_id,
                "request_id": request_id,
                "dataset_profile_id": "problem-wafer-sem-v1",
            }
        )
        return _ChildStage(
            workflow_id=workflow_id,
            thread_id=thread_id,
            initial_state=initial,
            dependencies=dependencies,
            graph_factory=lambda saver: cast(
                _ChildGraph,
                build_research_graph(checkpointer=saver),
            ),
        )

    def _repository_stage(self, pipeline_workflow_id: str) -> _ChildStage:
        research_id = self._stage_workflow_id(pipeline_workflow_id, "research")
        research = self.research_store.load_result(research_id)
        if research.status != "COMPLETED" or len(research.selected_paper_ids) != 1:
            raise ValueError("repository stage requires one selected research paper")
        workflow_id = self._stage_workflow_id(pipeline_workflow_id, "repository")
        thread_id = self._thread_id(workflow_id)
        request_id = f"request-{workflow_id}"
        artifact_root = self.var_root / "repository-artifacts"
        fixture_root = self.source_root / "fixtures" / "repositories" / "plain_pytorch"
        transport = FixtureGitHubTransport(_repository_archive(fixture_root))
        dependencies = RepositoryDependencies(
            research_store=self.research_store,
            research_workflow_id=research_id,
            selected_paper_id=research.selected_paper_ids[0],
            repository_provider=GitHubRepositoryProvider(
                artifact_root=artifact_root,
                transport=transport,
                clock=lambda: FIXTURE_NOW,
            ),
            static_analyzer=ZipStaticRepositoryAnalyzer(artifact_root=artifact_root),
            store=self.repository_store,
            approval_recorder=InMemoryApprovalRecorder(),
            clock=lambda: FIXTURE_NOW,
        )
        initial = create_initial_state(
            {
                "schema_version": "1",
                "workflow_id": workflow_id,
                "thread_id": thread_id,
                "request_id": request_id,
                "dataset_profile_id": "problem-wafer-sem-v1",
            }
        )
        return _ChildStage(
            workflow_id=workflow_id,
            thread_id=thread_id,
            initial_state=initial,
            dependencies=dependencies,
            graph_factory=lambda saver: cast(
                _ChildGraph,
                build_repository_graph(checkpointer=saver),
            ),
        )

    def _dataset_profile(self) -> DatasetProfile:
        path = self.source_root / "fixtures" / "datasets" / "synthetic_sem_profile.json"
        return DatasetProfile.model_validate_json(path.read_text(encoding="utf-8"))

    def _adaptation_stage(self, pipeline_workflow_id: str) -> _ChildStage:
        workflow_id = self._stage_workflow_id(pipeline_workflow_id, "adaptation")
        repository_id = self._stage_workflow_id(pipeline_workflow_id, "repository")
        thread_id = self._thread_id(workflow_id)
        request_id = f"request-{workflow_id}"
        patch_tool = FixturePatchTool(
            fixture_root=self.source_root / "fixtures" / "repositories" / "plain_pytorch",
            store=self.adaptation_store,
            clock=lambda: FIXTURE_NOW,
        )
        smoke_tool = FixtureSmokeRunner(
            store=self.adaptation_store,
            clock=lambda: FIXTURE_NOW,
            minimum_repair_revision=2 if self.scenario == "smoke-failure" else 0,
        )
        dependencies = AdaptationDependencies(
            repository_store=self.repository_store,
            repository_workflow_id=repository_id,
            dataset_profile=self._dataset_profile(),
            planner=self.adaptation_planner,
            patch_tool=patch_tool,
            smoke_tool=smoke_tool,
            store=self.adaptation_store,
            approval_recorder=InMemoryApprovalRecorder(),
            clock=lambda: FIXTURE_NOW,
        )
        initial = create_initial_state(
            {
                "schema_version": "1",
                "workflow_id": workflow_id,
                "thread_id": thread_id,
                "request_id": request_id,
                "dataset_profile_id": "dataset-synthetic-sem-1",
            }
        )
        return _ChildStage(
            workflow_id=workflow_id,
            thread_id=thread_id,
            initial_state=initial,
            dependencies=dependencies,
            graph_factory=lambda saver: cast(
                _ChildGraph,
                build_adaptation_graph(checkpointer=saver),
            ),
        )

    def _training_input(self, pipeline_workflow_id: str, project_root: Path) -> TrainingInput:
        adaptation_id = self._stage_workflow_id(pipeline_workflow_id, "adaptation")
        result = self.adaptation_store.load_result(adaptation_id)
        if (
            result.status != "ACCEPTED"
            or result.base_commit_sha is None
            or result.plan_revision is None
            or result.accepted_patch_hash is None
        ):
            raise ValueError("training stage requires exact accepted adaptation evidence")
        dataset = self._dataset_profile()
        return TrainingInput(
            adaptation_workflow_id=adaptation_id,
            base_commit_sha=result.base_commit_sha,
            patch_revision=result.plan_revision,
            patch_hash=result.accepted_patch_hash,
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            dataset_content_hash=dataset.content_hash,
            dataset_ref=DATASET_REF,
            dataset_ref_hash=_fixture_hash(project_root, DATASET_REF),
            split_ref=SPLIT_REF,
            split_hash=_fixture_hash(project_root, SPLIT_REF),
            preprocess_ref=PREPROCESS_REF,
            preprocess_hash=_fixture_hash(project_root, PREPROCESS_REF),
            seed=17,
            budget=TrainingBudgetSpec(
                max_epochs=4,
                max_steps=48,
                max_walltime_seconds=10,
            ),
        )

    def _training_stage(self, pipeline_workflow_id: str) -> _ChildStage:
        self.fixture_project.prepare_training()
        project_root = self.fixture_project.project_root
        workflow_id = self._stage_workflow_id(pipeline_workflow_id, "training")
        thread_id = self._thread_id(workflow_id)
        request_id = f"request-{workflow_id}"
        store = LocalTrainingStore(project_root / "var")
        self._training_store = store
        dependencies = TrainingDependencies(
            adaptation_reader=self.adaptation_store,
            training_input=self._training_input(pipeline_workflow_id, project_root),
            project_root=project_root,
            store=store,
            trainer=LocalTrainingExecutor(project_root=project_root, store=store),
            approval_recorder=InMemoryApprovalRecorder(),
            cancellation=NeverCancelled(),
            clock=lambda: FIXTURE_NOW,
        )
        initial = create_initial_state(
            {
                "schema_version": "1",
                "workflow_id": workflow_id,
                "thread_id": thread_id,
                "request_id": request_id,
                "dataset_profile_id": "dataset-synthetic-sem-1",
            }
        )
        return _ChildStage(
            workflow_id=workflow_id,
            thread_id=thread_id,
            initial_state=initial,
            dependencies=dependencies,
            graph_factory=lambda saver: cast(
                _ChildGraph,
                build_training_graph(checkpointer=saver),
            ),
        )

    def _evaluation_stage(self, pipeline_workflow_id: str) -> _ChildStage:
        self.fixture_project.prepare_evaluation()
        project_root = self.fixture_project.project_root
        training_store = self._training_store or LocalTrainingStore(project_root / "var")
        self._training_store = training_store
        evaluation_store = LocalEvaluationStore(project_root / "var")
        self._evaluation_store = evaluation_store
        workflow_id = self._stage_workflow_id(pipeline_workflow_id, "evaluation")
        training_id = self._stage_workflow_id(pipeline_workflow_id, "training")
        thread_id = self._thread_id(workflow_id)
        initial = create_evaluation_state(
            EvaluationInitialInput(
                workflow_id=workflow_id,
                thread_id=thread_id,
                request_id=f"request-{workflow_id}",
                training_workflow_id=training_id,
            )
        )
        dependencies = EvaluationDependencies(
            training_reader=training_store,
            project_root=project_root,
            store=evaluation_store,
        )
        return _ChildStage(
            workflow_id=workflow_id,
            thread_id=thread_id,
            initial_state=initial,
            dependencies=dependencies,
            graph_factory=lambda saver: cast(
                _ChildGraph,
                build_evaluation_graph(checkpointer=saver),
            ),
        )

    def _assemble(
        self,
        pipeline_workflow_id: str,
        stage: PipelineStageName,
    ) -> _ChildStage:
        if stage == "research":
            return self._research_stage(pipeline_workflow_id)
        if stage == "repository":
            return self._repository_stage(pipeline_workflow_id)
        if stage == "adaptation":
            return self._adaptation_stage(pipeline_workflow_id)
        if stage == "training":
            return self._training_stage(pipeline_workflow_id)
        return self._evaluation_stage(pipeline_workflow_id)

    def _gate_evidence(
        self,
        *,
        stage: PipelineStageName,
        stage_workflow_id: str,
        payload: dict[str, object],
    ) -> tuple[dict[str, object], str | None]:
        if stage == "research":
            papers = payload.get("recommended_papers")
            first = papers[0] if isinstance(papers, list) and papers else None
            evidence = {
                "recommended_count": len(papers) if isinstance(papers, list) else 0,
                "paper_id": first.get("paper_id") if isinstance(first, dict) else None,
                "title": first.get("title") if isinstance(first, dict) else None,
                "relevance_score": (
                    first.get("relevance_score") if isinstance(first, dict) else None
                ),
            }
            return evidence, self.research_store.result_ref(stage_workflow_id)
        if stage == "repository":
            return {
                "repository_url": payload.get("repository_url"),
                "confidence": payload.get("confidence"),
                "evidence_type": payload.get("evidence_type"),
            }, self.repository_store.result_ref(stage_workflow_id)
        if stage == "adaptation":
            return {
                "patch_hash": payload.get("patch_hash"),
                "smoke_capability": payload.get("smoke_capability_boundary"),
                "real_pytorch_training": payload.get("real_pytorch_training"),
            }, cast(str | None, payload.get("smoke_ref"))
        return {
            "frozen_spec_hash": payload.get("frozen_spec_hash"),
            "seed": payload.get("seed"),
            "budget": payload.get("budget"),
            "capability": payload.get("capability"),
            "real_pytorch_training": payload.get("real_pytorch_training"),
        }, self._training_ref(
            LocalTrainingStore.spec_ref(
                stage_workflow_id,
                cast(int, payload.get("subject_revision")),
            )
        )

    def _stage_artifacts(
        self,
        stage: PipelineStageName,
        workflow_id: str,
    ) -> list[str]:
        if stage == "research":
            path = self.research_store.result_path(workflow_id)
            return [self.research_store.result_ref(workflow_id)] if path.is_file() else []
        if stage == "repository":
            path = self.repository_store.result_path(workflow_id)
            return [self.repository_store.result_ref(workflow_id)] if path.is_file() else []
        if stage == "adaptation":
            path = self.adaptation_store.resolve_ref(self.adaptation_store.result_ref(workflow_id))
            if not path.is_file():
                return []
            result = self.adaptation_store.load_result(workflow_id)
            refs = [self.adaptation_store.result_ref(workflow_id)]
            if result.plan_ref is not None:
                refs.append(result.plan_ref)
            for attempt in result.attempts:
                refs.extend([attempt.patch_ref, attempt.patch_manifest_ref])
                if attempt.smoke_ref is not None:
                    refs.append(attempt.smoke_ref)
            return list(dict.fromkeys(refs))
        if stage == "training":
            store = self._training_store
            if store is None:
                return []
            path = store.resolve_ref(store.workflow_ref(workflow_id))
            if not path.is_file():
                return []
            record = store.load_workflow(workflow_id)
            refs = [store.workflow_ref(workflow_id)]
            if record.current_spec_ref is not None:
                refs.append(record.current_spec_ref)
            for run in (record.baseline_result, record.candidate_result):
                if run is not None:
                    refs.extend(
                        [run.manifest_ref, run.log_ref, run.metrics_ref, run.predictions_ref]
                    )
            return [self._training_ref(ref) for ref in refs]
        evaluation_store = self._evaluation_store
        if evaluation_store is None:
            return []
        evaluation_ref = evaluation_store.evaluation_ref(workflow_id)
        if not evaluation_store.resolve_ref(evaluation_ref).is_file():
            return []
        return [
            self._training_ref(evaluation_ref),
            self._training_ref(evaluation_store.report_ref(workflow_id)),
        ]

    def _outcome(
        self,
        *,
        stage: PipelineStageName,
        workflow_id: str,
        result: Mapping[str, object],
        gates: list[PipelineGateRecord],
    ) -> PipelineStageOutcome:
        child_status = _status_text(result.get("status", "UNKNOWN"))
        conclusion = _status_text(result.get("conclusion")) if result.get("conclusion") else None
        if stage == "evaluation":
            succeeded = child_status == "COMPLETED" and conclusion != "INVALID"
            stopped = False
        else:
            succeeded = child_status == WorkflowStatus.SUCCEEDED.value
            stopped = child_status in {
                WorkflowStatus.REJECTED.value,
                WorkflowStatus.CANCELLED.value,
            }
        failure = None
        if stopped:
            failure = PipelineFailure(
                code="PIPELINE_HUMAN_REJECTED",
                message="A human Gate stopped this stage; downstream side effects were not run.",
                stage=stage,
            )
        elif not succeeded:
            failure = _child_failure(stage, result)
        status: Literal["SUCCEEDED", "STOPPED", "FAILED"] = (
            "SUCCEEDED" if succeeded else "STOPPED" if stopped else "FAILED"
        )
        record = PipelineStageRecord(
            stage=stage,
            workflow_id=workflow_id,
            status=status,
            child_status=child_status,
            resume_count=len(gates),
            artifact_refs=self._stage_artifacts(stage, workflow_id),
            failure=failure,
        )
        return PipelineStageOutcome(record=record, gates=tuple(gates), conclusion=conclusion)

    async def run_stage(
        self,
        *,
        pipeline_workflow_id: str,
        stage: PipelineStageName,
    ) -> PipelineStageOutcome:
        """Invoke a child, consume only real interrupts, and resume with a new graph instance."""
        workflow_id = self._stage_workflow_id(pipeline_workflow_id, stage)
        self._emit({"event": "stage_started", "stage": stage, "workflow_id": workflow_id})
        gates: list[PipelineGateRecord] = []
        try:
            child = self._assemble(pipeline_workflow_id, stage)
            saver: BaseCheckpointSaver[str] = InMemorySaver(
                serde=JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPE_ALLOWLIST)
            )
            graph = child.graph_factory(saver)
            raw = await graph.ainvoke(
                child.initial_state,
                config=workflow_config(child.thread_id),
                context=child.dependencies,
            )
            if not isinstance(raw, dict):
                raise TypeError("child graph result must be a mapping")
            result = cast(dict[str, object], raw)
            payload = _interrupt_payload(result)
            while payload is not None:
                occurrence = len(gates) + 1
                evidence, artifact_ref = self._gate_evidence(
                    stage=stage,
                    stage_workflow_id=child.workflow_id,
                    payload=payload,
                )
                self._emit(
                    {
                        "event": "gate_pending",
                        "stage": stage,
                        "gate_kind": payload.get("gate_kind"),
                        "subject_id": payload.get("subject_id"),
                        "subject_revision": payload.get("subject_revision"),
                        "evidence": evidence,
                        "artifact_ref": artifact_ref,
                    }
                )
                approval: Approval = self.decision_provider.decide(
                    workflow_id=pipeline_workflow_id,
                    stage=stage,
                    payload=payload,
                    occurrence=occurrence,
                    decided_at=FIXTURE_NOW,
                )
                gate_kind_value = payload.get("gate_kind")
                if gate_kind_value not in {
                    "CANDIDATE_SELECTION",
                    "REPOSITORY_INGEST",
                    "PATCH_ACCEPTANCE",
                    "RUN_SUBMISSION",
                }:
                    raise ValueError("child interrupt has an unsupported Gate kind")
                json_evidence = cast(
                    dict[str, JsonValue],
                    json.loads(json.dumps(evidence, allow_nan=False)),
                )
                gate = PipelineGateRecord(
                    stage=stage,
                    gate_kind=gate_kind_value,
                    subject_id=approval.subject_id,
                    subject_revision=approval.subject_revision,
                    decision=approval.decision.value,
                    scripted_fixture_decision=(self.decision_provider.scripted_fixture_decisions),
                    resume_count=occurrence,
                    evidence=json_evidence,
                    artifact_ref=artifact_ref,
                )
                gates.append(gate)
                self._emit(
                    {
                        "event": "gate_decision",
                        "stage": stage,
                        "gate_kind": gate.gate_kind,
                        "decision": gate.decision,
                        "scripted_fixture_decision": gate.scripted_fixture_decision,
                    }
                )
                graph = child.graph_factory(saver)
                resumed = await graph.ainvoke(
                    Command(resume=approval.model_dump(mode="json")),
                    config=workflow_config(child.thread_id),
                    context=child.dependencies,
                )
                if not isinstance(resumed, dict):
                    raise TypeError("resumed child graph result must be a mapping")
                result = cast(dict[str, object], resumed)
                payload = _interrupt_payload(result)
            outcome = self._outcome(
                stage=stage,
                workflow_id=child.workflow_id,
                result=result,
                gates=gates,
            )
        except Exception as error:
            if isinstance(error, KeyboardInterrupt):
                raise
            outcome = PipelineStageOutcome(
                record=PipelineStageRecord(
                    stage=stage,
                    workflow_id=workflow_id,
                    status="FAILED",
                    child_status="DRIVER_FAILED",
                    resume_count=len(gates),
                    artifact_refs=self._stage_artifacts(stage, workflow_id),
                    failure=PipelineFailure(
                        code=f"PIPELINE_{stage.upper()}_DRIVER_FAILED",
                        message="The fixture child graph or its bounded runtime failed explicitly.",
                        stage=stage,
                    ),
                ),
                gates=tuple(gates),
            )
        self._emit(
            {
                "event": "stage_finished",
                "stage": stage,
                "status": outcome.record.status,
                "child_status": outcome.record.child_status,
                "resume_count": outcome.record.resume_count,
            }
        )
        return outcome

    def _stage_records(self, state: PipelineState) -> list[PipelineStageRecord]:
        observed = {
            item.stage: item
            for item in (
                PipelineStageRecord.model_validate(raw) for raw in state.get("stage_records", [])
            )
        }
        return [
            observed.get(stage)
            or PipelineStageRecord(
                stage=stage,
                workflow_id=self._stage_workflow_id(state["workflow_id"], stage),
                status="NOT_RUN",
                child_status="NOT_RUN",
            )
            for stage in cast(
                tuple[PipelineStageName, ...],
                (
                    "research",
                    "repository",
                    "adaptation",
                    "training",
                    "evaluation",
                ),
            )
        ]

    def _paper_evidence(self, workflow_id: str) -> PaperEvidence | None:
        child_id = self._stage_workflow_id(workflow_id, "research")
        if not self.research_store.result_path(child_id).is_file():
            return None
        result = self.research_store.load_result(child_id)
        if len(result.selected_paper_ids) != 1:
            return None
        return PaperEvidence(
            paper_id=result.selected_paper_ids[0],
            evidence_ref=self.research_store.result_ref(child_id),
        )

    def _repository_evidence(self, workflow_id: str) -> RepositoryEvidence | None:
        child_id = self._stage_workflow_id(workflow_id, "repository")
        if not self.repository_store.result_path(child_id).is_file():
            return None
        result = self.repository_store.load_result(child_id)
        if result.profile is None:
            return None
        snapshot = result.profile.repository_snapshot
        return RepositoryEvidence(
            repository_id=snapshot.repository_id,
            fixed_commit_sha=snapshot.commit_sha,
            profile_ref=self.repository_store.result_ref(child_id),
        )

    def _adaptation_evidence(self, workflow_id: str) -> AdaptationEvidence | None:
        child_id = self._stage_workflow_id(workflow_id, "adaptation")
        result_path = self.adaptation_store.resolve_ref(self.adaptation_store.result_ref(child_id))
        if not result_path.is_file():
            return None
        result: AdaptationResult = self.adaptation_store.load_result(child_id)
        if result.plan_ref is None or result.planner_trace_ref is None or not result.attempts:
            return None
        attempt = result.attempts[-1]
        if attempt.smoke_ref is None:
            return None
        patch = self.adaptation_store.load_patch_record(attempt.patch_manifest_ref)
        smoke = self.adaptation_store.load_smoke_result(attempt.smoke_ref)
        planner_trace = self.adaptation_store.load_planner_trace(result.planner_trace_ref)
        return AdaptationEvidence(
            plan_ref=result.plan_ref,
            planner_trace_ref=result.planner_trace_ref,
            planner_kind=planner_trace.planner_kind,
            planner_tools=[event.tool_name for event in planner_trace.events],
            patch_ref=attempt.patch_ref,
            patch_manifest_ref=attempt.patch_manifest_ref,
            patch_hash=patch.patch_hash,
            smoke_ref=attempt.smoke_ref,
            smoke_capability=smoke.capability_boundary,
            real_pytorch_training=False,
        )

    def _training_evidence(self, workflow_id: str) -> TrainingEvidence | None:
        child_id = self._stage_workflow_id(workflow_id, "training")
        store = self._training_store or LocalTrainingStore(
            self.fixture_project.project_root / "var"
        )
        workflow_ref = store.workflow_ref(child_id)
        if not store.resolve_ref(workflow_ref).is_file():
            return None
        record = store.load_workflow(child_id)
        if (
            record.status != "SUCCEEDED"
            or record.current_spec_ref is None
            or record.baseline_result is None
            or record.candidate_result is None
        ):
            return None
        baseline = record.baseline_result
        candidate = record.candidate_result
        return TrainingEvidence(
            workflow_ref=self._training_ref(workflow_ref),
            spec_ref=self._training_ref(record.current_spec_ref),
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            baseline_manifest_ref=self._training_ref(baseline.manifest_ref),
            candidate_manifest_ref=self._training_ref(candidate.manifest_ref),
            baseline_metrics_ref=self._training_ref(baseline.metrics_ref),
            candidate_metrics_ref=self._training_ref(candidate.metrics_ref),
            baseline_predictions_ref=self._training_ref(baseline.predictions_ref),
            candidate_predictions_ref=self._training_ref(candidate.predictions_ref),
            capability=baseline.capability,
            real_pytorch_training=False,
        )

    def _evaluation_evidence(self, workflow_id: str) -> EvaluationEvidence | None:
        child_id = self._stage_workflow_id(workflow_id, "evaluation")
        store = self._evaluation_store or LocalEvaluationStore(
            self.fixture_project.project_root / "var"
        )
        ref = store.evaluation_ref(child_id)
        if not store.resolve_ref(ref).is_file():
            return None
        result: EvaluationResult = store.load_evaluation(child_id)
        metrics = EvaluationMetricEvidence()
        if (
            result.baseline_metrics is not None
            and result.candidate_metrics is not None
            and result.deltas is not None
        ):
            metrics = EvaluationMetricEvidence(
                baseline_macro_f1=result.baseline_metrics.macro_f1,
                candidate_macro_f1=result.candidate_metrics.macro_f1,
                macro_f1_delta=result.deltas.macro_f1,
                baseline_balanced_accuracy=result.baseline_metrics.balanced_accuracy,
                candidate_balanced_accuracy=result.candidate_metrics.balanced_accuracy,
                balanced_accuracy_delta=result.deltas.balanced_accuracy,
                severe_recall_delta=result.deltas.severe_class_recall,
            )
        return EvaluationEvidence(
            evaluation_ref=self._training_ref(result.evaluation_ref),
            report_ref=self._training_ref(result.report_ref),
            conclusion=result.conclusion,
            metrics=metrics,
            capability=result.evaluation_capability,
            llm_used=False,
            real_company_evaluation=False,
        )

    def build_summary(self, state: PipelineState) -> PipelineSummary:
        """Re-read child artifacts and preserve evaluation numbers/conclusion without rewriting."""
        workflow_id = state["workflow_id"]
        stages = self._stage_records(state)
        gates = [PipelineGateRecord.model_validate(raw) for raw in state.get("gate_records", [])]
        failure_raw = state.get("failure_reason")
        failure = PipelineFailure.model_validate(failure_raw) if failure_raw is not None else None
        evaluation = self._evaluation_evidence(workflow_id)
        conclusion = evaluation.conclusion if evaluation is not None else None
        return PipelineSummary(
            workflow_id=workflow_id,
            adaptation_planner_mode=self.adaptation_planner_mode,
            scenario=self.scenario,
            status=state["status"],
            scripted_fixture_decisions=self.scripted_fixture_decisions,
            decision_config=self.decision_config,
            stages=stages,
            gates=gates,
            resume_count=sum(item.resume_count for item in stages),
            paper=self._paper_evidence(workflow_id),
            repository=self._repository_evidence(workflow_id),
            adaptation=self._adaptation_evidence(workflow_id),
            training=self._training_evidence(workflow_id),
            evaluation=evaluation,
            conclusion=conclusion,
            limitations=list(_LIMITATIONS),
            failure_reason=failure,
            summary_ref=f"sample/{workflow_id}/summary.json",
            created_at=FIXTURE_NOW,
        )


__all__ = [
    "FIXED_REPOSITORY_SHA",
    "FIXTURE_NOW",
    "FixtureGitHubTransport",
    "FixturePipelineStageDriver",
]
