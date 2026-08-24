"""Shared strict-valid factories for every public foundation domain model."""

import pytest

from vision_research_ops.domain import (
    AdaptationAttempt,
    AdaptationPlan,
    Approval,
    ArtifactRef,
    AuthorRef,
    CodeLinkEvidence,
    CommandSpec,
    CompatibilityGap,
    DatasetProfile,
    DependencyChange,
    EvaluationReport,
    ExperimentRun,
    ExperimentSpec,
    GenerationRecord,
    LabelSpec,
    MetricDefinition,
    MetricSummary,
    ModelRunTemplate,
    PaperCandidate,
    PatchOperation,
    PerClassSummary,
    PlannedChange,
    ProvenanceRef,
    QuerySpec,
    Reason,
    RepositorySnapshot,
    ResearchBudget,
    ResearchRequest,
    ResourceRequest,
    RiskFinding,
    RunEntrypoint,
    SplitPolicy,
    StructuredFailure,
    TrainingBudget,
    ValidationResult,
)
from vision_research_ops.domain.enums import (
    ApprovalDecision,
    ArtifactKind,
    CodeLinkConfidence,
    EvaluationConclusion,
    GateKind,
    LicenseStatus,
    NetworkPolicy,
    PatchOperationType,
    RunStatus,
    SeverityLevel,
    SplitStrategy,
    TaskType,
    ValidationStage,
    ValidationStatus,
    WorkflowStatus,
)

SHA256 = "sha256:" + "a" * 64
COMMIT_SHA = "a" * 40


def _ts() -> str:
    return "2026-08-06T08:00:00Z"


def _later_ts() -> str:
    return "2026-08-06T08:01:00Z"


@pytest.fixture
def make_artifact_ref():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "artifact_id": "art_1",
            "kind": ArtifactKind.PATCH,
            "uri": "artifacts/workflows/wf_1/patch.bin",
            "sha256": SHA256,
            "size_bytes": 128,
            "media_type": "application/octet-stream",
            "created_at": _ts(),
            "producer": "patch:1",
            "sensitivity": "INTERNAL",
        }
        defaults.update(over)
        return ArtifactRef(**defaults)

    return _make


@pytest.fixture
def make_provenance_ref():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "source_type": "provider",
            "source_id": "arxiv_1234.5678",
            "source_url": "https://arxiv.org/abs/1234.5678",
            "retrieved_at": _ts(),
        }
        defaults.update(over)
        return ProvenanceRef(**defaults)

    return _make


@pytest.fixture
def make_resource_request():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "cpu_cores": 2.0,
            "memory_mb": 4096,
            "gpu_count": 0,
            "walltime_seconds": 3600,
            "scratch_mb": 2048,
            "network_policy": NetworkPolicy.NONE,
        }
        defaults.update(over)
        return ResourceRequest(**defaults)

    return _make


@pytest.fixture
def make_metric_definition():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "name": "macro_f1",
            "direction": "MAXIMIZE",
            "aggregation": "macro",
            "implementation_version": "1.0.0",
            "primary": True,
        }
        defaults.update(over)
        return MetricDefinition(**defaults)

    return _make


@pytest.fixture
def make_label_spec():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "label_id": "lb_1",
            "name": "defect_a",
            "is_unknown": False,
        }
        defaults.update(over)
        return LabelSpec(**defaults)

    return _make


@pytest.fixture
def make_split_policy():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "strategy": SplitStrategy.GROUP_HOLDOUT,
            "group_keys": ["wafer_id"],
            "test_fraction": 0.2,
            "seed": 7,
        }
        defaults.update(over)
        return SplitPolicy(**defaults)

    return _make


@pytest.fixture
def make_query_spec():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "keywords": ["defect", "classification"],
        }
        defaults.update(over)
        return QuerySpec(**defaults)

    return _make


@pytest.fixture
def make_research_budget():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "max_provider_pages": 10,
            "max_provider_records": 100,
            "max_llm_calls": 50,
            "max_llm_tokens": 100000,
            "max_cost_estimate": 10.0,
            "max_candidate_repositories": 20,
            "max_adaptation_attempts": 2,
            "max_workflow_walltime_seconds": 7200,
        }
        defaults.update(over)
        return ResearchBudget(**defaults)

    return _make


@pytest.fixture
def make_training_budget():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "max_epochs": 10,
            "max_steps": None,
            "max_walltime_seconds": 3600,
            "max_test_evaluations": 1,
        }
        defaults.update(over)
        return TrainingBudget(**defaults)

    return _make


@pytest.fixture
def make_author_ref():
    def _make(**over):
        defaults = {"schema_version": "1", "name": "Alice Example"}
        defaults.update(over)
        return AuthorRef(**defaults)

    return _make


@pytest.fixture
def make_reason():
    def _make(**over):
        defaults = {"schema_version": "1", "code": "RETRIEVAL_PROVIDER_RATE_LIMITED"}
        defaults.update(over)
        return Reason(**defaults)

    return _make


@pytest.fixture
def make_risk_finding():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "finding_id": "find_1",
            "rule_id": "network_access",
            "category": "SECURITY_NETWORK_ACCESS",
            "severity": SeverityLevel.HIGH,
            "description": "Network access candidate requires policy review.",
        }
        defaults.update(over)
        return RiskFinding(**defaults)

    return _make


@pytest.fixture
def make_compatibility_gap():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "gap_id": "gap_1",
            "area": "DATA_LOADING",
            "current_state": "reads-directory",
            "required_state": "uses-dataset-adapter",
        }
        defaults.update(over)
        return CompatibilityGap(**defaults)

    return _make


@pytest.fixture
def make_planned_change():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "change_id": "chg_1",
            "path": "train.py",
            "action": "MODIFY",
        }
        defaults.update(over)
        return PlannedChange(**defaults)

    return _make


@pytest.fixture
def make_dependency_change():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "dependency_id": "dep_1",
            "package": "torch",
            "action": "ADD",
        }
        defaults.update(over)
        return DependencyChange(**defaults)

    return _make


@pytest.fixture
def make_command_spec():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "executable_id": "python",
            "argv": ["train.py", "--epochs", "1"],
            "cwd_ref": "scratch/worktree_1",
        }
        defaults.update(over)
        return CommandSpec(**defaults)

    return _make


@pytest.fixture
def make_generation_record():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "provider_id": "fake-provider",
            "model_id": "model-x",
            "prompt_template_id": "adapt/v1",
            "prompt_version": "1",
            "prompt_hash": SHA256,
            "output_hash": SHA256,
        }
        defaults.update(over)
        return GenerationRecord(**defaults)

    return _make


@pytest.fixture
def make_run_entrypoint():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "executable_id": "python",
            "argv": ["train.py", "--epochs", "1"],
            "cwd_subpath": "src",
        }
        defaults.update(over)
        return RunEntrypoint(**defaults)

    return _make


@pytest.fixture
def make_model_run_template():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "template_id": "tmpl_1",
            "display_name": "baseline",
            "repository_id": "repo_1",
            "commit_sha": COMMIT_SHA,
            "entrypoint": RunEntrypoint(
                schema_version="1",
                executable_id="python",
                argv=["train.py"],
                cwd_subpath=".",
            ),
            "environment_digest": SHA256,
            "config_hash": SHA256,
        }
        defaults.update(over)
        return ModelRunTemplate(**defaults)

    return _make


@pytest.fixture
def make_metric_summary():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "metric_name": "macro_f1",
            "mean": 0.81,
        }
        defaults.update(over)
        return MetricSummary(**defaults)

    return _make


@pytest.fixture
def make_per_class_summary():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "label_id": "defect_a",
            "precision": 0.82,
            "recall": 0.77,
            "f1": 0.79,
            "support": 42,
        }
        defaults.update(over)
        return PerClassSummary(**defaults)

    return _make


@pytest.fixture
def make_patch_operation():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "op": PatchOperationType.REPLACE,
            "path": "/epochs",
            "value": 1,
        }
        defaults.update(over)
        return PatchOperation(**defaults)

    return _make


@pytest.fixture
def make_dataset_profile():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "dataset_id": "ds_1",
            "version": "v1",
            "display_name": "synthetic defects",
            "task_type": TaskType.IMAGE_CLASSIFICATION,
            "modality": "GRAYSCALE",
            "channels": 1,
            "image_shape_policy": {"fixed": True, "shape": [64, 64]},
            "label_schema": [
                LabelSpec(schema_version="1", label_id="lb_1", name="ok"),
                LabelSpec(schema_version="1", label_id="lb_2", name="defect"),
            ],
            "sample_counts": {"train": 800, "test": 200},
            "group_keys": ["wafer_id"],
            "split_policy": SplitPolicy(
                schema_version="1",
                strategy=SplitStrategy.GROUP_HOLDOUT,
                group_keys=["wafer_id"],
                test_fraction=0.2,
                seed=7,
            ),
            "location_ref": "catalog/ds_1@v1",
            "content_hash": SHA256,
            "authorization": {"owner": "org", "purpose": "research"},
            "preprocessing_contract": {"decode": "gray"},
            "created_at": _ts(),
        }
        defaults.update(over)
        return DatasetProfile(**defaults)

    return _make


@pytest.fixture
def make_research_request():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "request_id": "rr_1",
            "revision": 1,
            "title": "defect classification",
            "research_question": "Does method X improve macro-f1?",
            "dataset_id": "ds_1",
            "dataset_version": "v1",
            "query_spec": QuerySpec(schema_version="1", keywords=["defect"]),
            "budget": ResearchBudget(
                schema_version="1",
                max_provider_pages=10,
                max_provider_records=100,
                max_llm_calls=50,
                max_llm_tokens=100000,
                max_cost_estimate=10.0,
                max_candidate_repositories=20,
                max_adaptation_attempts=2,
                max_workflow_walltime_seconds=7200,
            ),
            "requested_by": "actor_1",
            "status": WorkflowStatus.PENDING,
            "created_at": _ts(),
            "updated_at": _ts(),
        }
        defaults.update(over)
        return ResearchRequest(**defaults)

    return _make


@pytest.fixture
def make_paper_candidate():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "paper_id": "pp_1",
            "request_id": "rr_1",
            "canonical_title": "A title",
            "first_published_at": _ts(),
            "relevance_score": 0.9,
            "provenance": [
                ProvenanceRef(
                    schema_version="1",
                    source_type="provider",
                    source_id="x",
                    retrieved_at=_ts(),
                )
            ],
        }
        defaults.update(over)
        return PaperCandidate(**defaults)

    return _make


@pytest.fixture
def make_code_link_evidence():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "evidence_id": "ev_1",
            "paper_id": "pp_1",
            "repository_url": "https://github.com/org/repo",
            "evidence_type": "project_page",
            "confidence": CodeLinkConfidence.OFFICIAL_HIGH,
            "provenance": ProvenanceRef(
                schema_version="1", source_type="provider", source_id="x", retrieved_at=_ts()
            ),
            "verified_at": _ts(),
        }
        defaults.update(over)
        return CodeLinkEvidence(**defaults)

    return _make


@pytest.fixture
def make_repository_snapshot():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "repository_id": "repo_1",
            "canonical_url": "https://github.com/org/repo",
            "provider": "GITHUB",
            "owner": "org",
            "name": "repo",
            "commit_sha": COMMIT_SHA,
            "license_status": LicenseStatus.ALLOWLISTED,
        }
        defaults.update(over)
        return RepositorySnapshot(**defaults)

    return _make


@pytest.fixture
def make_adaptation_plan():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "plan_id": "plan_1",
            "repository_id": "repo_1",
            "commit_sha": COMMIT_SHA,
            "dataset_id": "ds_1",
            "dataset_version": "v1",
            "target_contract_version": "adapter-v1",
            "estimated_resources": ResourceRequest(
                schema_version="1",
                cpu_cores=1.0,
                memory_mb=2048,
                gpu_count=0,
                walltime_seconds=600,
                scratch_mb=512,
                network_policy=NetworkPolicy.NONE,
            ),
            "generated_by": GenerationRecord(
                schema_version="1",
                provider_id="fake-provider",
                model_id="m",
                prompt_template_id="p",
                prompt_version="1",
                prompt_hash=SHA256,
                output_hash=SHA256,
            ),
            "status": "DRAFT",
            "revision": 1,
        }
        defaults.update(over)
        return AdaptationPlan(**defaults)

    return _make


@pytest.fixture
def make_adaptation_attempt():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "attempt_id": "att_1",
            "plan_id": "plan_1",
            "plan_revision": 1,
            "attempt_number": 1,
            "workspace_ref": "scratch/worktree_1",
            "patch_artifact_id": "art_patch_1",
            "patch_hash": SHA256,
            "base_commit_sha": COMMIT_SHA,
            "operation_id": "op_1",
            "status": "GENERATED",
            "created_at": _ts(),
        }
        defaults.update(over)
        return AdaptationAttempt(**defaults)

    return _make


@pytest.fixture
def make_validation_result():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "validation_id": "val_1",
            "attempt_id": "att_1",
            "stage": ValidationStage.STATIC_POLICY,
            "status": ValidationStatus.PASSED,
            "started_at": _ts(),
            "finished_at": _later_ts(),
            "retryable": False,
        }
        defaults.update(over)
        return ValidationResult(**defaults)

    return _make


@pytest.fixture
def make_experiment_spec():
    def _make(**over):
        baseline_entrypoint = RunEntrypoint(
            schema_version="1",
            executable_id="python",
            argv=["train.py"],
            cwd_subpath=".",
        )
        defaults = {
            "schema_version": "1",
            "experiment_id": "exp_1",
            "revision": 1,
            "request_id": "rr_1",
            "dataset_id": "ds_1",
            "dataset_version": "v1",
            "dataset_hash": SHA256,
            "baseline_ref": ModelRunTemplate(
                schema_version="1",
                template_id="tmpl_b",
                display_name="baseline",
                repository_id="repo_1",
                commit_sha=COMMIT_SHA,
                entrypoint=baseline_entrypoint,
                environment_digest=SHA256,
                config_hash=SHA256,
            ),
            "candidate_ref": ModelRunTemplate(
                schema_version="1",
                template_id="tmpl_c",
                display_name="candidate",
                repository_id="repo_1",
                commit_sha=COMMIT_SHA,
                entrypoint=RunEntrypoint(
                    schema_version="1",
                    executable_id="python",
                    argv=["train.py"],
                    cwd_subpath=".",
                ),
                environment_digest=SHA256,
                config_hash=SHA256,
            ),
            "seeds": [1, 2, 3],
            "split_manifest_artifact_id": "art_split",
            "preprocessing_hash": SHA256,
            "training_budget": TrainingBudget(
                schema_version="1",
                max_epochs=10,
                max_walltime_seconds=3600,
                max_test_evaluations=1,
            ),
            "metrics": [
                MetricDefinition(
                    schema_version="1",
                    name="macro_f1",
                    direction="MAXIMIZE",
                    aggregation="macro",
                    implementation_version="1.0.0",
                    primary=True,
                )
            ],
            "resources": ResourceRequest(
                schema_version="1",
                cpu_cores=2.0,
                memory_mb=8192,
                gpu_count=1,
                walltime_seconds=86400,
                scratch_mb=4096,
                network_policy=NetworkPolicy.NONE,
            ),
            "environment_digest": SHA256,
            "approval_id": "appr_1",
            "spec_hash": SHA256,
            "created_at": _ts(),
        }
        defaults.update(over)
        return ExperimentSpec(**defaults)

    return _make


@pytest.fixture
def make_experiment_run():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "run_id": "run_1",
            "experiment_id": "exp_1",
            "role": "BASELINE",
            "seed": 1,
            "status": RunStatus.CREATED,
            "idempotency_key": "key_1",
            "executor": "LOCAL",
            "revision": 1,
        }
        defaults.update(over)
        return ExperimentRun(**defaults)

    return _make


@pytest.fixture
def make_evaluation_report():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "report_id": "rep_1",
            "experiment_id": "exp_1",
            "validity": "VALID",
            "baseline_run_ids": ["run_base_1"],
            "candidate_run_ids": ["run_candidate_1"],
            "metric_summaries": [
                MetricSummary(schema_version="1", metric_name="macro_f1", mean=0.81)
            ],
            "conclusion": EvaluationConclusion.NO_CLEAR_IMPROVEMENT,
            "evaluation_artifact_id": "art_eval",
            "evaluator_version": "1.0.0",
            "created_at": _ts(),
        }
        defaults.update(over)
        return EvaluationReport(**defaults)

    return _make


@pytest.fixture
def make_approval():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "approval_id": "appr_1",
            "gate_kind": GateKind.RUN_SUBMISSION,
            "subject_type": "experiment_spec",
            "subject_id": "exp_1",
            "subject_revision": 3,
            "decision": ApprovalDecision.APPROVE,
            "reason": "resources verified",
            "actor_id": "actor_1",
            "decided_at": _ts(),
            "idempotency_key": "appr_key_1",
        }
        defaults.update(over)
        return Approval(**defaults)

    return _make


@pytest.fixture
def make_structured_failure():
    def _make(**over):
        defaults = {
            "schema_version": "1",
            "code": "RUN_RESOURCE_POLICY_EXCEEDED",
            "category": "RUN",
            "message": "resources exceed budget",
            "message_hash": SHA256,
            "retryable": False,
        }
        defaults.update(over)
        return StructuredFailure(**defaults)

    return _make
