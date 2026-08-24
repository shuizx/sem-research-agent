"""adaptation strict input, prompt, patch policy, and real fixture probe tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vision_research_ops.application.services.adaptation_models import (
    AdaptationChangeProposal,
    CompatibilityGapProposal,
)
from vision_research_ops.application.services.adaptation_patch import (
    PatchPolicyError,
    validate_patch_fields,
    validate_patch_path,
)
from vision_research_ops.application.services.adaptation_planning import (
    adaptation_prompt_facts,
    adaptation_request,
    validate_adaptation_inputs,
    validate_proposal_against_inputs,
)

from .conftest import AdaptationHarness, load_dataset_profile, repository_result


def test_generated_proposal_ids_are_canonical_and_reject_dangerous_values() -> None:
    """Persisted LLM identifiers allow fixed canonical IDs, never paths or secrets."""
    gap_values = {
        "area": "INPUT_CHANNELS",
        "current_state": "The fixture uses its repository default.",
        "required_state": "Use one grayscale channel.",
        "risk": "LOW",
    }
    change_values = {
        "area": "INPUT_CHANNELS",
        "target_template": "SEM_PLAIN_PYTORCH_CONFIG_V1",
        "target_field": "/input/channels",
        "action": "SET",
        "reason": "Match the de-identified dataset contract.",
    }

    safe_gap = CompatibilityGapProposal(gap_id="gap-input-channels", **gap_values)
    safe_change = AdaptationChangeProposal(change_id="change-input-channels", **change_values)
    assert safe_gap.gap_id == "gap-input-channels"
    assert safe_change.change_id == "change-input-channels"

    for gap_id in (
        r"C:\private",
        "/private",
        r"\\server\share",
        "gap-secret-token",
        "gap-authorization",
        "gap-control\ncharacter",
    ):
        with pytest.raises(ValidationError):
            CompatibilityGapProposal(gap_id=gap_id, **gap_values)
    for change_id in (
        r"C:\private",
        "/private",
        r"\\server\share",
        "change-credential",
        "change-api-key",
        "change-control\ncharacter",
    ):
        with pytest.raises(ValidationError):
            AdaptationChangeProposal(change_id=change_id, **change_values)


def test_sanitized_prompt_contains_only_deidentified_contract_facts() -> None:
    """The LLM sees no location, authorization, internal ID, image, path, or key."""
    facts = validate_adaptation_inputs(repository_result(), load_dataset_profile())
    prompt_facts = adaptation_prompt_facts(facts)
    request = adaptation_request(facts)
    encoded = json.dumps(prompt_facts, sort_keys=True)

    assert request.response_schema.__name__ == "AdaptationPlanProposal"
    assert request.model_parameters == {"temperature": 0}
    assert request.artifact_excerpts == []
    assert "UNTRUSTED_CONTENT" in encoded
    assert "location_ref" not in encoded
    assert "fixture-dataset-handle" not in encoded
    assert "authorization" not in encoded
    assert "dataset-synthetic-sem-1" not in encoded
    assert "workflow-" not in encoded
    assert "DASHSCOPE" not in encoded
    assert "api_key" not in encoded.casefold()
    assert "C:\\" not in encoded


@pytest.mark.parametrize(
    "path",
    [
        "../sem_adaptation.json",
        "/sem_adaptation.json",
        "C:\\fixture\\sem_adaptation.json",
        ".git/config",
        ".env",
        "requirements.txt",
        "model.pt",
        "nested/sem_adaptation.json",
        "sem%5fadaptation.json",
    ],
)
def test_patch_policy_rejects_out_of_scope_secret_binary_and_dependency_paths(path: str) -> None:
    """Only the exact text config target can reach the patch workspace."""
    with pytest.raises(PatchPolicyError):
        validate_patch_path(path)


def test_patch_policy_rejects_unknown_or_duplicate_fields() -> None:
    """Config compilation cannot add dependencies or arbitrary source edits."""
    with pytest.raises(PatchPolicyError):
        validate_patch_fields(["/dependencies/torch"])
    with pytest.raises(PatchPolicyError):
        validate_patch_fields(["/input/channels", "/input/channels"])


@pytest.mark.asyncio
@pytest.mark.integration
async def test_patch_and_smoke_are_actual_fixture_operations(
    make_adaptation_harness,
) -> None:
    """The fixture produces a real diff and four subprocess-backed stage records."""
    harness: AdaptationHarness = make_adaptation_harness()
    facts = validate_adaptation_inputs(
        harness.repository_store.load_result("workflow-repository-p3-fixture"),
        harness.dependencies.dataset_profile,
    )
    planner_output = await harness.llm.plan(facts, ctx=_operation_context())
    generation = planner_output.generation
    unsafe = generation.value.model_copy(
        update={"rationale": "Read C:\\private\\secret before applying the plan."}
    )
    with pytest.raises(ValueError, match="forbidden path"):
        validate_proposal_against_inputs(unsafe, facts)
    from vision_research_ops.application.services.adaptation_planning import (
        compile_adaptation_plan,
    )

    plan = compile_adaptation_plan(
        workflow_id="workflow-service-smoke",
        facts=facts,
        generation_result=generation,
        now=harness.dependencies.clock(),
    )
    harness.store.write_plan(plan)
    patch = await harness.patch_tool.apply(plan, ctx=_operation_context())
    smoke = await harness.smoke_tool.run(patch, ctx=_operation_context())

    assert patch.patch_hash.startswith("sha256:")
    diff = harness.store.resolve_ref(patch.patch_ref).read_text(encoding="utf-8")
    assert "--- a/sem_adaptation.json" in diff
    assert '"channels": 1' in diff
    assert '"num_classes": 4' in diff
    assert "requirements.txt" not in diff
    assert smoke.status == "PASSED"
    assert [stage.stage.value for stage in smoke.stages] == [
        "STATIC_POLICY",
        "IMPORT",
        "ONE_BATCH",
        "BOUNDED_OVERFIT",
    ]
    assert all(stage.exit_code == 0 for stage in smoke.stages)
    assert all(stage.command.shell is False for stage in smoke.stages)
    assert all(stage.command.argv[0] == "-I" for stage in smoke.stages)
    assert smoke.stages[0].evidence["network_guard"] == "STDLIB_SOCKET_BLOCKED"
    assert smoke.real_pytorch_training is False
    assert smoke.capability_boundary == "FIXTURE_CONTRACT_PROBE_NO_TORCH"
    metrics_path = harness.store.resolve_ref(patch.workspace_ref) / "outputs" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["real_pytorch_training"] is False
    assert set(metrics["metrics"]) == {
        "macro_f1",
        "balanced_accuracy",
        "per_class_recall",
    }


def _operation_context():
    from vision_research_ops.ports import OperationContext

    return OperationContext(
        schema_version="1",
        correlation_id="corr-adaptation-service",
        workflow_id="workflow-service-smoke",
        actor_id="pipeline-user",
        idempotency_key="adaptation-service-idempotency",
        sensitivity="INTERNAL",
    )


def test_dataset_fixture_is_relative_and_synthetic() -> None:
    """The committed profile is descriptive and never embeds an absolute data path."""
    profile = load_dataset_profile()
    assert profile.authorization == {
        "profile_use_allowed": True,
        "source_kind": "SYNTHETIC",
    }
    assert profile.location_ref.startswith("dataset-handle-")
    assert not Path(profile.location_ref).is_absolute()
