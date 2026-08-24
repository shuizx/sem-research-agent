"""Validate accepted adaptation evidence and freeze a fair training training pair."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from vision_research_ops.domain import Approval, PatchOperationType, ValidationStage

from .adaptation_models import AdaptationResult, PatchArtifactRecord, SmokeResultRecord
from .training_models import (
    FrozenRunSpec,
    FrozenTrainingSpec,
    PendingTrainingEdit,
    RunRole,
    TrainingBudgetSpec,
    TrainingCommandSpec,
    TrainingInput,
    canonical_json_bytes,
    content_hash,
)

DATASET_REF = "fixtures/training/synthetic_sem_dataset.json"
SPLIT_REF = "fixtures/training/synthetic_sem_split.json"
PREPROCESS_REF = "fixtures/training/grayscale_preprocess.json"
BASELINE_CONFIG_REF = "fixtures/training/baseline_config.json"
CANDIDATE_CONFIG_REF = "fixtures/training/candidate_config.json"

# Filled from the committed controlled fixture bytes. Runtime input must match both
# the caller-provided digest and this source allowlist.
EXPECTED_FIXTURE_HASHES: dict[str, str] = {
    DATASET_REF: "sha256:50b6b4544b44e1b445509c9e709554bd4de652b73bfa83b48cdd134548329e21",
    SPLIT_REF: "sha256:2783930e9ce943dc8bbc8f6fe0b51cbd2ce1d80d7d57e1d5a6c56d91bc6d257e",
    PREPROCESS_REF: "sha256:208ec445f7bea802a1ac2e7b1591237602e0d80a8bdcf47496892c5185f8566c",
    BASELINE_CONFIG_REF: "sha256:3a93f0f626f45b25beee62037a94b44c071b3c06640305c7b0033679bc9794f2",
    CANDIDATE_CONFIG_REF: "sha256:78309e8141cdfdc59608f4d02064afd40e840ecc7bdc6a9b3cb987cd81bbec54",
}


class AcceptedAdaptationReader(Protocol):
    """Narrow read boundary for exact adaptation result, patch, and Smoke evidence."""

    def load_result(self, workflow_id: str) -> AdaptationResult:
        """Load one strict adaptation workflow result."""

    def load_patch_record(self, ref: str) -> PatchArtifactRecord:
        """Load one exact adaptation patch manifest."""

    def load_smoke_result(self, ref: str) -> SmokeResultRecord:
        """Load one exact adaptation Smoke result."""

    def resolve_ref(self, relative_ref: str) -> Path:
        """Resolve a adaptation-local artifact ref under its trusted var root."""


@dataclass(frozen=True, slots=True)
class AcceptedTrainingEvidence:
    """Runtime-only validated evidence used to compile a frozen submission."""

    result: AdaptationResult
    patch: PatchArtifactRecord
    smoke: SmokeResultRecord
    patch_manifest_ref: str
    smoke_result_ref: str
    p3_approval_id: str


def _trusted_file_hash(project_root: Path, ref: str) -> str:
    root = project_root.resolve()
    path = (root / Path(*ref.split("/"))).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError("controlled training fixture is missing or unsafe")
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    observed = content_hash(normalized)
    if observed != EXPECTED_FIXTURE_HASHES[ref]:
        raise ValueError("controlled training fixture content is not allowlisted")
    return observed


def validate_training_input(
    raw_input: TrainingInput | Mapping[str, object],
    *,
    reader: AcceptedAdaptationReader,
    project_root: Path,
) -> tuple[TrainingInput, AcceptedTrainingEvidence]:
    """Fail closed unless exact accepted adaptation and fixed fixture evidence matches."""
    payload = (
        raw_input.model_dump(mode="json")
        if isinstance(raw_input, TrainingInput)
        else dict(raw_input)
    )
    request = TrainingInput.model_validate(payload)
    if (
        request.dataset_ref != DATASET_REF
        or request.split_ref != SPLIT_REF
        or request.preprocess_ref != PREPROCESS_REF
    ):
        raise ValueError("training input must use the sole controlled fixture references")
    requested_hashes = {
        request.dataset_ref: request.dataset_ref_hash,
        request.split_ref: request.split_hash,
        request.preprocess_ref: request.preprocess_hash,
    }
    for ref, requested_hash in requested_hashes.items():
        if _trusted_file_hash(project_root, ref) != requested_hash:
            raise ValueError("training fixture reference hash does not match its content")
    _trusted_file_hash(project_root, BASELINE_CONFIG_REF)
    _trusted_file_hash(project_root, CANDIDATE_CONFIG_REF)

    result = reader.load_result(request.adaptation_workflow_id)
    required = (
        result.status == "ACCEPTED"
        and result.base_commit_sha == request.base_commit_sha
        and result.plan_revision == request.patch_revision
        and result.accepted_patch_hash == request.patch_hash
        and result.gate_patch_hash == request.patch_hash
        and result.dataset_id == request.dataset_id
        and result.dataset_version == request.dataset_version
        and result.dataset_content_hash == request.dataset_content_hash
        and result.repository_kind == "CONTROLLED_PLAIN_PYTORCH_FIXTURE"
        and result.dataset_kind == "SYNTHETIC_SEM_FIXTURE"
        and result.approval_id is not None
        and result.gate_subject_id is not None
        and result.gate_revision == request.patch_revision
    )
    if not required:
        raise ValueError("adaptation result is not an exact accepted training input")
    attempt = next(
        (
            item
            for item in result.attempts
            if item.plan_revision == request.patch_revision
            and item.patch_hash == request.patch_hash
            and item.smoke_status == "PASSED"
            and item.smoke_ref is not None
        ),
        None,
    )
    if attempt is None or attempt.smoke_ref is None:
        raise ValueError("accepted patch lacks exact passed Smoke evidence")
    patch = reader.load_patch_record(attempt.patch_manifest_ref)
    smoke = reader.load_smoke_result(attempt.smoke_ref)
    if (
        patch.plan_revision != request.patch_revision
        or patch.patch_hash != request.patch_hash
        or patch.base_commit_sha != request.base_commit_sha
        or patch.dataset_version != request.dataset_version
        or smoke.plan_revision != request.patch_revision
        or smoke.patch_hash != request.patch_hash
        or smoke.base_commit_sha != request.base_commit_sha
        or smoke.dataset_version != request.dataset_version
        or smoke.status != "PASSED"
        or smoke.real_pytorch_training
        or [stage.stage for stage in smoke.stages]
        != [
            ValidationStage.STATIC_POLICY,
            ValidationStage.IMPORT,
            ValidationStage.ONE_BATCH,
            ValidationStage.BOUNDED_OVERFIT,
        ]
    ):
        raise ValueError("adaptation patch or Smoke evidence does not match the accepted revision")
    if content_hash(reader.resolve_ref(patch.patch_ref).read_bytes()) != request.patch_hash:
        raise ValueError("accepted patch bytes do not match the reviewed hash")
    review = next(
        (
            item
            for item in result.reviews
            if item.approval_id == result.approval_id
            and item.decision == "APPROVE"
            and item.subject_id == result.gate_subject_id
            and item.subject_revision == request.patch_revision
            and item.patch_hash == request.patch_hash
        ),
        None,
    )
    if review is None:
        raise ValueError("adaptation result lacks an exact APPROVE review")
    return request, AcceptedTrainingEvidence(
        result=result,
        patch=patch,
        smoke=smoke,
        patch_manifest_ref=attempt.patch_manifest_ref,
        smoke_result_ref=attempt.smoke_ref,
        p3_approval_id=review.approval_id,
    )


def _run_id(workflow_id: str, revision: int, role: str) -> str:
    digest = content_hash(f"{workflow_id}:r{revision}:{role}".encode())
    return f"training-{role.casefold()}-{digest.removeprefix('sha256:')[:20]}"


def _command(run_id: str, role: RunRole, spec_ref: str) -> TrainingCommandSpec:
    output_ref = f"runs/{run_id}"
    return TrainingCommandSpec(
        run_id=run_id,
        role=role,
        spec_ref=spec_ref,
        output_ref=output_ref,
        argv=[
            "-I",
            "fixtures/training/synthetic_linear_cpu.py",
            "--spec-ref",
            f"var/{spec_ref}",
            "--run-id",
            run_id,
            "--role",
            role,
            "--output-ref",
            f"var/{output_ref}",
        ],
        env_keys=[
            "PYTHONIOENCODING",
            "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE",
        ],
    )


def freeze_training_spec(
    *,
    workflow_id: str,
    request: TrainingInput,
    evidence: AcceptedTrainingEvidence,
    revision: int,
    budget: TrainingBudgetSpec | None = None,
    seed: int | None = None,
) -> FrozenTrainingSpec:
    """Freeze one fair pair whose only run difference is candidate config/patch."""
    selected_budget = request.budget if budget is None else budget
    selected_seed = request.seed if seed is None else seed
    spec_ref = f"training/{workflow_id}/spec-r{revision}.json"
    baseline_id = _run_id(workflow_id, revision, "BASELINE")
    candidate_id = _run_id(workflow_id, revision, "CANDIDATE")
    shared: dict[str, object] = {
        "base_commit_sha": request.base_commit_sha,
        "dataset_id": request.dataset_id,
        "dataset_version": request.dataset_version,
        "dataset_content_hash": request.dataset_content_hash,
        "dataset_ref": request.dataset_ref,
        "dataset_ref_hash": request.dataset_ref_hash,
        "split_ref": request.split_ref,
        "split_hash": request.split_hash,
        "preprocess_ref": request.preprocess_ref,
        "preprocess_hash": request.preprocess_hash,
        "seed": selected_seed,
        "budget": selected_budget.model_dump(mode="json"),
    }
    baseline = FrozenRunSpec.model_validate(
        {
            **shared,
            "run_id": baseline_id,
            "role": "BASELINE",
            "method": "GLOBAL_STATS_LINEAR",
            "method_config_ref": BASELINE_CONFIG_REF,
            "method_config_hash": EXPECTED_FIXTURE_HASHES[BASELINE_CONFIG_REF],
            "command": _command(baseline_id, "BASELINE", spec_ref).model_dump(mode="json"),
        }
    )
    candidate = FrozenRunSpec.model_validate(
        {
            **shared,
            "run_id": candidate_id,
            "role": "CANDIDATE",
            "method": "GRID4_LINEAR_PATCHED",
            "candidate_patch_revision": request.patch_revision,
            "candidate_patch_hash": request.patch_hash,
            "method_config_ref": CANDIDATE_CONFIG_REF,
            "method_config_hash": EXPECTED_FIXTURE_HASHES[CANDIDATE_CONFIG_REF],
            "command": _command(candidate_id, "CANDIDATE", spec_ref).model_dump(mode="json"),
        }
    )
    payload: dict[str, object] = {
        "schema_version": "1",
        "workflow_id": workflow_id,
        "adaptation_workflow_id": request.adaptation_workflow_id,
        "revision": revision,
        "base_commit_sha": request.base_commit_sha,
        "patch_revision": request.patch_revision,
        "patch_hash": request.patch_hash,
        "adaptation_result_ref": f"adaptations/{request.adaptation_workflow_id}/adaptation.json",
        "patch_manifest_ref": evidence.patch_manifest_ref,
        "smoke_result_ref": evidence.smoke_result_ref,
        "p3_approval_id": evidence.p3_approval_id,
        "baseline": baseline.model_dump(mode="json"),
        "candidate": candidate.model_dump(mode="json"),
    }
    payload["spec_hash"] = content_hash(canonical_json_bytes(payload))
    return FrozenTrainingSpec.model_validate(payload)


def apply_training_edits(
    request: TrainingInput,
    pending: PendingTrainingEdit,
) -> tuple[TrainingBudgetSpec, int]:
    """Apply only four bounded integer fields to both sides of the next revision."""
    values = request.budget.model_dump(mode="json")
    seed = request.seed
    for edit in pending.edits:
        if edit.op is not PatchOperationType.REPLACE:
            raise ValueError("training edits only support REPLACE")
        if edit.path == "/seed":
            seed = edit.value
        else:
            values[edit.path.removeprefix("/budget/")] = edit.value
    try:
        budget = TrainingBudgetSpec.model_validate(values)
        validated_seed = TrainingInput.model_validate(
            {
                **request.model_dump(mode="json"),
                "budget": budget.model_dump(mode="json"),
                "seed": seed,
            }
        ).seed
    except ValidationError as error:
        raise ValueError("training edit exceeded the fixed fixture bounds") from error
    return budget, validated_seed


def submission_id(spec: FrozenTrainingSpec) -> str:
    """Bind the human subject identity to the exact frozen spec hash."""
    return f"training-submission-{spec.spec_hash.removeprefix('sha256:')[:24]}"


def training_gate_id(spec: FrozenTrainingSpec) -> str:
    """Return a stable Gate ID bound to revision and spec content."""
    return f"gate-run-submission-{spec.spec_hash.removeprefix('sha256:')[:20]}-r{spec.revision}"


def approval_edits(approval: Approval) -> PendingTrainingEdit:
    """Sanitize the existing strict Approval edits into training's tiny edit grammar."""
    edits: list[dict[str, object]] = []
    for edit in approval.edits:
        if edit.op is not PatchOperationType.REPLACE or not isinstance(edit.value, int):
            raise ValueError("training Gate edits must be bounded integer replacements")
        edits.append({"op": edit.op, "path": edit.path, "value": edit.value})
    return PendingTrainingEdit.model_validate({"approval_id": approval.approval_id, "edits": edits})


__all__ = [
    "BASELINE_CONFIG_REF",
    "CANDIDATE_CONFIG_REF",
    "DATASET_REF",
    "EXPECTED_FIXTURE_HASHES",
    "PREPROCESS_REF",
    "SPLIT_REF",
    "AcceptedAdaptationReader",
    "AcceptedTrainingEvidence",
    "apply_training_edits",
    "approval_edits",
    "freeze_training_spec",
    "submission_id",
    "training_gate_id",
    "validate_training_input",
]
