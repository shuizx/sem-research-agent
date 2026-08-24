"""Sanitized LLM request composition and deterministic plan compilation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import cast

from pydantic import ValidationError

from vision_research_ops.domain import (
    Approval,
    GenerationRecord,
    JsonObject,
    LicenseStatus,
    PatchOperationType,
    SplitStrategy,
    TaskType,
)
from vision_research_ops.ports import StructuredGenerationRequest, StructuredGenerationResult
from vision_research_ops.prompts.adaptation import (
    PROMPT_TEMPLATE_ID,
    PROMPT_VERSION,
    UNTRUSTED_CONTENT_NOTICE,
)

from .adaptation_models import (
    REQUIRED_METRICS,
    AdaptationInputFacts,
    AdaptationPlanProposal,
    CompiledAdaptationPlan,
)
from .repository_models import RepositoryResult

ALLOWED_FIXTURE_REPOSITORY_URL = "https://github.com/example/sem-classifier"
_REQUIRED_PROFILE_FILES = frozenset({"config.yaml", "data.py", "model.py", "train.py"})
_ALLOWED_PROFILE_FILES = frozenset(
    {
        "LICENSE",
        "README.md",
        "config.yaml",
        "data.py",
        "model.py",
        "requirements.txt",
        "train.py",
    }
)
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")
_GROUP_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"(?i)(?:^|[\s\"'])[a-z]:[\\/]")
_EDITABLE_PLAN_PATHS = frozenset(
    {
        "/channels",
        "/num_classes",
        "/label_mapping",
        "/group_split_key",
        "/metrics",
        "/metrics_output_file",
    }
)


def validate_adaptation_inputs(
    repository: RepositoryResult,
    dataset: object,
) -> AdaptationInputFacts:
    """Fail closed unless repository and the de-identified dataset match the sole fixture."""
    from vision_research_ops.domain import DatasetProfile

    if not isinstance(dataset, DatasetProfile):
        raise TypeError("dataset input must be a validated DatasetProfile")
    if repository.status != "COMPLETED" or repository.profile is None:
        raise ValueError("adaptation requires a COMPLETED repository repository result")
    profile = repository.profile
    snapshot = profile.repository_snapshot
    if not profile.supported:
        raise ValueError("adaptation requires a supported repository repository profile")
    if profile.structure_type != "PLAIN_PYTORCH":
        raise ValueError("only the allowlisted PLAIN_PYTORCH fixture is supported")
    if snapshot.canonical_url != ALLOWED_FIXTURE_REPOSITORY_URL:
        raise ValueError("repository is not the controlled adaptation fixture")
    if snapshot.license_status is not LicenseStatus.ALLOWLISTED:
        raise ValueError("repository license must be allowlisted before adaptation")
    if snapshot.license_spdx not in {"MIT", "Apache-2.0", "BSD-3-Clause"}:
        raise ValueError("repository SPDX identifier is not in the adaptation fixture allowlist")
    profiled_files = {item.path for item in profile.file_tree_summary if item.kind == "FILE"}
    if not _REQUIRED_PROFILE_FILES.issubset(profiled_files):
        raise ValueError("repository profile does not match the controlled fixture layout")
    if not profiled_files.issubset(_ALLOWED_PROFILE_FILES):
        raise ValueError("repository profile contains files outside the controlled fixture")
    if any(
        "\\" in path
        or path.startswith("/")
        or ":" in path
        or "%" in path
        or any(part in {"", ".", "..", ".git", ".env"} for part in path.split("/"))
        for path in profiled_files
    ):
        raise ValueError("repository profile contains a non-canonical or protected path")
    if not set(profile.dependency_files).issubset({"requirements.txt"}):
        raise ValueError("repository profile contains an unapproved dependency file")
    if "train.py" not in profile.entrypoint_candidates:
        raise ValueError("repository profile lacks the fixed fixture entrypoint")

    if dataset.task_type is not TaskType.IMAGE_CLASSIFICATION:
        raise ValueError("adaptation supports image classification only")
    if dataset.modality != "GRAYSCALE" or dataset.channels != 1:
        raise ValueError("the adaptation fixture requires one-channel grayscale input")
    if len(dataset.label_schema) < 2:
        raise ValueError("the dataset profile requires at least two labels")
    label_names = [label.name for label in dataset.label_schema]
    if len(set(label_names)) != len(label_names):
        raise ValueError("de-identified label names must be unique")
    if any(_LABEL_RE.fullmatch(name) is None for name in label_names):
        raise ValueError("label names must be bounded de-identified display tokens")
    if _VERSION_RE.fullmatch(dataset.version) is None:
        raise ValueError("dataset version must be a bounded de-identified token")
    if any(_GROUP_RE.fullmatch(key) is None for key in dataset.group_keys):
        raise ValueError("dataset group keys must be bounded de-identified tokens")
    if dataset.split_policy.strategy is not SplitStrategy.GROUP_HOLDOUT:
        raise ValueError("the adaptation fixture requires a group-holdout split")
    if not dataset.split_policy.group_keys:
        raise ValueError("the dataset split requires a de-identified group key")
    if not set(dataset.split_policy.group_keys).issubset(dataset.group_keys):
        raise ValueError("split group keys must be declared by the dataset profile")
    authorization = dataset.authorization
    if (
        authorization.get("source_kind") != "SYNTHETIC"
        or authorization.get("profile_use_allowed") is not True
    ):
        raise ValueError("adaptation accepts only the authorized synthetic dataset fixture")

    return AdaptationInputFacts(
        repository_id=snapshot.repository_id,
        repository_url=snapshot.canonical_url,
        base_commit_sha=snapshot.commit_sha,
        structure_type="PLAIN_PYTORCH",
        license_spdx=snapshot.license_spdx,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        dataset_content_hash=dataset.content_hash,
        modality="GRAYSCALE",
        channels=1,
        label_names=label_names,
        group_keys=list(dataset.group_keys),
        group_split_key=dataset.split_policy.group_keys[0],
        dataset_kind="SYNTHETIC_SEM_FIXTURE",
        repository_kind="CONTROLLED_PLAIN_PYTORCH_FIXTURE",
    )


def adaptation_prompt_facts(facts: AdaptationInputFacts) -> JsonObject:
    """Return the only de-identified facts allowed across the LLM boundary."""
    value: JsonObject = {
        "content_notice": UNTRUSTED_CONTENT_NOTICE,
        "repository_facts": {
            "structure_type": facts.structure_type,
            "commit_sha": facts.base_commit_sha,
            "license_spdx": facts.license_spdx,
            "fixture_layout": ["config.yaml", "data.py", "model.py", "train.py"],
        },
        "dataset_facts": {
            "source_kind": "SYNTHETIC_SEM_FIXTURE",
            "version": facts.dataset_version,
            "modality": facts.modality,
            "channels": facts.channels,
            "label_names": list(facts.label_names),
            "group_keys": list(facts.group_keys),
            "split_strategy": "GROUP_HOLDOUT",
        },
        "required_contract": {
            "template": "SEM_PLAIN_PYTORCH_CONFIG_V1",
            "metrics": list(REQUIRED_METRICS),
            "dependency_changes_allowed": False,
            "shell_allowed": False,
        },
    }
    _assert_sanitized(value)
    return value


def _assert_sanitized(value: object) -> None:
    serialized = str(value).casefold()
    forbidden = (
        "location_ref",
        "authorization",
        "dataset_id",
        "workflow_id",
        "request_id",
        "api_key",
        "secret",
        "token",
        "c:\\",
        "file://",
        "ssh://",
    )
    if any(marker in serialized for marker in forbidden):
        raise ValueError("adaptation prompt facts contain a forbidden field or value")


def adaptation_request(
    facts: AdaptationInputFacts,
) -> StructuredGenerationRequest[AdaptationPlanProposal]:
    """Build the strict, temperature-zero adaptation planning request."""
    return StructuredGenerationRequest[AdaptationPlanProposal](
        schema_version="1",
        task_name="adaptation_plan",
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_version=PROMPT_VERSION,
        response_schema=AdaptationPlanProposal,
        facts=adaptation_prompt_facts(facts),
        artifact_excerpts=[],
        model_parameters={"temperature": 0},
        budget_class="pipeline_adaptation_small",
    )


def validate_proposal_against_inputs(
    proposal: AdaptationPlanProposal,
    facts: AdaptationInputFacts,
) -> None:
    """Reject schema-valid LLM suggestions that conflict with deterministic facts."""
    expected_mapping = {name: index for index, name in enumerate(facts.label_names)}
    if proposal.channels != facts.channels:
        raise ValueError("proposal channels do not match the dataset profile")
    if proposal.num_classes != len(facts.label_names):
        raise ValueError("proposal num_classes does not match the dataset profile")
    if proposal.label_mapping != expected_mapping:
        raise ValueError("proposal label_mapping does not match the frozen label order")
    if proposal.group_split_key != facts.group_split_key:
        raise ValueError("proposal group split does not match the dataset policy")
    if tuple(proposal.metrics) != REQUIRED_METRICS:
        raise ValueError("proposal metrics do not match the fixed output contract")
    _assert_safe_generated_text(proposal)


def _assert_safe_generated_text(proposal: AdaptationPlanProposal) -> None:
    """Prevent hallucinated local paths or credential-like text entering artifacts."""
    generated_values = [
        proposal.rationale,
        *(gap.gap_id for gap in proposal.gaps),
        *(gap.current_state for gap in proposal.gaps),
        *(gap.required_state for gap in proposal.gaps),
        *(change.change_id for change in proposal.changes),
        *(change.reason for change in proposal.changes),
    ]
    if any(
        ord(character) < 32 or ord(character) == 127
        for value in generated_values
        for character in value
    ):
        raise ValueError("structured proposal contains forbidden control characters")
    encoded = json.dumps(generated_values, ensure_ascii=False).casefold()
    forbidden = (
        "file://",
        "ssh://",
        "\\\\",
        "/home/",
        "/private",
        "/users/",
        "/root/",
        "/data/",
        ".env",
        "api_key",
        "api-key",
        "apikey",
        "authorization",
        "bearer ",
        "credential",
        "secret",
        "token",
    )
    if _WINDOWS_ABSOLUTE_RE.search(encoded) or any(marker in encoded for marker in forbidden):
        raise ValueError("structured proposal contains forbidden path or credential-like text")


def compile_adaptation_plan(
    *,
    workflow_id: str,
    facts: AdaptationInputFacts,
    generation_result: StructuredGenerationResult[AdaptationPlanProposal],
    now: object,
) -> CompiledAdaptationPlan:
    """Revalidate provider output and bind it to immutable input provenance."""
    from datetime import datetime

    if not isinstance(now, datetime):
        raise TypeError("plan timestamp must be a datetime")
    try:
        proposal = AdaptationPlanProposal.model_validate(
            generation_result.value.model_dump(mode="json")
        )
    except (AttributeError, ValidationError) as error:
        raise ValueError("structured LLM result did not match AdaptationPlanProposal") from error
    validate_proposal_against_inputs(proposal, facts)
    generation = GenerationRecord(
        schema_version="1",
        provider_id=generation_result.provider_id,
        model_id=generation_result.model_id,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_version=PROMPT_VERSION,
        prompt_hash=generation_result.prompt_hash,
        output_hash=generation_result.output_hash,
    )
    return CompiledAdaptationPlan(
        workflow_id=workflow_id,
        plan_id=f"adaptation-plan-{workflow_id}",
        revision=1,
        repository_id=facts.repository_id,
        repository_url=facts.repository_url,
        base_commit_sha=facts.base_commit_sha,
        dataset_id=facts.dataset_id,
        dataset_version=facts.dataset_version,
        dataset_content_hash=facts.dataset_content_hash,
        dataset_kind=facts.dataset_kind,
        repository_kind=facts.repository_kind,
        proposal=proposal,
        generation=generation,
        origin="LLM_PROPOSAL",
        created_at=now,
        updated_at=now,
    )


def apply_human_plan_edits(
    plan: CompiledAdaptationPlan,
    approval: Approval,
    facts: AdaptationInputFacts,
    *,
    now: object,
) -> CompiledAdaptationPlan:
    """Apply only exact structured plan fields, then re-check dataset invariants."""
    from datetime import datetime

    if not isinstance(now, datetime):
        raise TypeError("edit timestamp must be a datetime")
    values = plan.proposal.model_dump(mode="python")
    for edit in approval.edits:
        if edit.op is not PatchOperationType.REPLACE or edit.path not in _EDITABLE_PLAN_PATHS:
            raise ValueError("patch EDIT may only replace a whitelisted structured plan field")
        key = edit.path.removeprefix("/")
        values[key] = edit.value
    try:
        proposal = AdaptationPlanProposal.model_validate(values)
    except ValidationError as error:
        raise ValueError("patch EDIT failed the strict adaptation plan schema") from error
    validate_proposal_against_inputs(proposal, facts)
    return plan.model_copy(
        update={
            "revision": plan.revision + 1,
            "proposal": proposal,
            "origin": "HUMAN_EDIT",
            "updated_at": now,
        }
    )


def public_plan_summary(plan: CompiledAdaptationPlan) -> Mapping[str, object]:
    """Return a compact presentation summary without local paths or raw content."""
    return cast(
        Mapping[str, object],
        {
            "revision": plan.revision,
            "channels": plan.proposal.channels,
            "num_classes": plan.proposal.num_classes,
            "label_mapping": dict(plan.proposal.label_mapping),
            "group_split_key": plan.proposal.group_split_key,
            "metrics": list(plan.proposal.metrics),
            "metrics_output_file": plan.proposal.metrics_output_file,
        },
    )


__all__ = [
    "ALLOWED_FIXTURE_REPOSITORY_URL",
    "adaptation_prompt_facts",
    "adaptation_request",
    "apply_human_plan_edits",
    "compile_adaptation_plan",
    "public_plan_summary",
    "validate_adaptation_inputs",
    "validate_proposal_against_inputs",
]
