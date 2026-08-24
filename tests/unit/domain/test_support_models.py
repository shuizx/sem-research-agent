"""Support-model contracts: split policies, templates, budgets, and edits."""

from __future__ import annotations

import pydantic
import pytest

from vision_research_ops.domain import (
    AdaptationAttempt,
    ModelRunTemplate,
    PatchOperation,
    ResearchBudget,
    SplitPolicy,
)
from vision_research_ops.domain.enums import PatchOperationType, SplitStrategy

pytestmark = pytest.mark.unit

SHA256 = "sha256:" + "a" * 64
COMMIT40 = "a" * 40
COMMIT64 = "b" * 64


def test_adaptation_attempt_happy_path_and_positive_fields(make_adaptation_attempt) -> None:
    attempt = make_adaptation_attempt()
    assert attempt.attempt_number == 1
    assert attempt.status == "GENERATED"
    for field in ("attempt_number", "plan_revision"):
        with pytest.raises(pydantic.ValidationError):
            make_adaptation_attempt(**{field: 0})
    with pytest.raises(pydantic.ValidationError):
        make_adaptation_attempt(patch_hash="bad")


def test_git_commit_sha_accepts_complete_lowercase_sha_lengths(
    make_repository_snapshot, make_adaptation_plan, make_adaptation_attempt, make_model_run_template
) -> None:
    assert make_repository_snapshot(commit_sha=COMMIT40).commit_sha == COMMIT40
    assert make_adaptation_plan(commit_sha=COMMIT64).commit_sha == COMMIT64
    assert make_adaptation_attempt(base_commit_sha=COMMIT64).base_commit_sha == COMMIT64
    assert make_model_run_template(commit_sha=COMMIT40).commit_sha == COMMIT40


@pytest.mark.parametrize("bad", ["a" * 7, "A" * 40, "g" * 40, " " + "a" * 40, "a" * 41])
def test_all_commit_fields_reject_short_uppercase_nonhex_and_whitespace(
    make_repository_snapshot,
    make_adaptation_plan,
    make_adaptation_attempt,
    make_model_run_template,
    bad: str,
) -> None:
    for factory, field in (
        (make_repository_snapshot, "commit_sha"),
        (make_adaptation_plan, "commit_sha"),
        (make_adaptation_attempt, "base_commit_sha"),
        (make_model_run_template, "commit_sha"),
    ):
        with pytest.raises(pydantic.ValidationError):
            factory(**{field: bad})


def test_model_run_template_uses_frozen_entrypoint_and_no_provisional_fields(
    make_model_run_template,
) -> None:
    template = make_model_run_template()
    assert template.entrypoint.cwd_subpath == "."
    assert template.patch_hash is None
    for removed_field in ("code_ref", "argv", "env_digest", "resources"):
        with pytest.raises(pydantic.ValidationError):
            make_model_run_template(**{removed_field: "provisional"})


def test_model_run_template_json_roundtrip(make_model_run_template) -> None:
    template = make_model_run_template()
    assert ModelRunTemplate.model_validate_json(template.model_dump_json()) == template


def test_generation_record_has_only_credential_free_provenance_fields(
    make_generation_record,
) -> None:
    record = make_generation_record()
    assert record.provider_id == "fake-provider"
    assert record.prompt_hash == SHA256
    with pytest.raises(pydantic.ValidationError):
        make_generation_record(api_key="secret")


def test_split_policy_accepts_all_four_contract_strategies() -> None:
    time_policy = SplitPolicy(
        schema_version="1",
        strategy=SplitStrategy.TIME_EXTRAPOLATION,
        group_keys=["wafer_id"],
        time_key="captured_at",
        time_cutoff="2026-08-06T08:00:00+08:00",
    )
    group_policy = SplitPolicy(
        schema_version="1",
        strategy=SplitStrategy.GROUP_HOLDOUT,
        group_keys=["wafer_id"],
        test_fraction=0.2,
        seed=7,
    )
    domain_policy = SplitPolicy(
        schema_version="1",
        strategy=SplitStrategy.DOMAIN_HOLDOUT,
        group_keys=["machine_id"],
        holdout_values={"machine_id": ["tool-a"]},
    )
    sample_policy = SplitPolicy(
        schema_version="1",
        strategy=SplitStrategy.SAMPLE_STRATIFIED,
        test_fraction=0.2,
        seed=7,
    )
    assert time_policy.time_cutoff is not None and time_policy.time_cutoff.hour == 0
    assert group_policy.test_fraction == 0.2
    assert domain_policy.holdout_values == {"machine_id": ["tool-a"]}
    assert sample_policy.group_keys == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": SplitStrategy.TIME_EXTRAPOLATION, "time_cutoff": "2026-08-06T08:00:00Z"},
        {
            "strategy": SplitStrategy.TIME_EXTRAPOLATION,
            "time_key": "captured_at",
            "time_cutoff": "2026-08-06T08:00:00Z",
            "test_fraction": 0.2,
        },
        {
            "strategy": SplitStrategy.TIME_EXTRAPOLATION,
            "time_key": "captured_at",
            "time_cutoff": "2026-08-06T08:00:00Z",
            "holdout_values": {"wafer_id": ["w1"]},
            "group_keys": ["wafer_id"],
        },
        {"strategy": SplitStrategy.GROUP_HOLDOUT, "test_fraction": 0.2, "seed": 7},
        {
            "strategy": SplitStrategy.GROUP_HOLDOUT,
            "group_keys": ["wafer_id"],
            "time_key": "captured_at",
            "test_fraction": 0.2,
            "seed": 7,
        },
        {"strategy": SplitStrategy.GROUP_HOLDOUT, "group_keys": ["wafer_id"]},
        {
            "strategy": SplitStrategy.GROUP_HOLDOUT,
            "group_keys": ["wafer_id"],
            "test_fraction": 0.2,
            "holdout_values": {"wafer_id": ["w1"]},
            "seed": 7,
        },
        {
            "strategy": SplitStrategy.GROUP_HOLDOUT,
            "group_keys": ["wafer_id"],
            "test_fraction": 0.2,
        },
        {"strategy": SplitStrategy.DOMAIN_HOLDOUT, "group_keys": ["machine_id"]},
        {
            "strategy": SplitStrategy.DOMAIN_HOLDOUT,
            "holdout_values": {"machine_id": ["tool-a"]},
        },
        {
            "strategy": SplitStrategy.DOMAIN_HOLDOUT,
            "group_keys": ["machine_id"],
            "holdout_values": {"machine_id": ["tool-a"]},
            "test_fraction": 0.2,
        },
        {
            "strategy": SplitStrategy.DOMAIN_HOLDOUT,
            "group_keys": ["machine_id"],
            "holdout_values": {"machine_id": ["tool-a"]},
            "time_key": "captured_at",
        },
        {"strategy": SplitStrategy.SAMPLE_STRATIFIED, "test_fraction": 0.2},
        {"strategy": SplitStrategy.SAMPLE_STRATIFIED, "seed": 7},
        {
            "strategy": SplitStrategy.SAMPLE_STRATIFIED,
            "test_fraction": 0.2,
            "seed": 7,
            "group_keys": ["wafer_id"],
        },
        {
            "strategy": SplitStrategy.SAMPLE_STRATIFIED,
            "test_fraction": 0.2,
            "seed": 7,
            "time_key": "captured_at",
        },
    ],
)
def test_split_policy_rejects_each_strategy_invalid_combination(kwargs: dict[str, object]) -> None:
    with pytest.raises(pydantic.ValidationError):
        SplitPolicy(schema_version="1", **kwargs)


def test_split_policy_enforces_fraction_sum_seed_and_holdout_key_membership() -> None:
    with pytest.raises(pydantic.ValidationError):
        SplitPolicy(
            schema_version="1",
            strategy=SplitStrategy.SAMPLE_STRATIFIED,
            test_fraction=0.7,
            validation_fraction=0.3,
            seed=7,
        )
    with pytest.raises(pydantic.ValidationError):
        SplitPolicy(
            schema_version="1",
            strategy=SplitStrategy.TIME_EXTRAPOLATION,
            time_key="captured_at",
            time_cutoff="2026-08-06T08:00:00Z",
            validation_fraction=0.1,
        )
    with pytest.raises(pydantic.ValidationError):
        SplitPolicy(
            schema_version="1",
            strategy=SplitStrategy.DOMAIN_HOLDOUT,
            group_keys=["machine_id"],
            holdout_values={"other_key": ["tool-a"]},
        )


def test_split_policy_deduplicates_group_and_holdout_values_in_order() -> None:
    policy = SplitPolicy(
        schema_version="1",
        strategy=SplitStrategy.DOMAIN_HOLDOUT,
        group_keys=["machine_id", "machine_id", "lot_id"],
        holdout_values={"machine_id": ["tool-a", "tool-a", "tool-b"]},
    )
    assert policy.group_keys == ["machine_id", "lot_id"]
    assert policy.holdout_values == {"machine_id": ["tool-a", "tool-b"]}


def test_query_spec_uses_ordered_iso_date_window(make_query_spec) -> None:
    query = make_query_spec(date_from="2026-01-01", date_to="2026-12-31")
    assert query.date_from is not None
    with pytest.raises(pydantic.ValidationError):
        make_query_spec(date_from="2026-12-31", date_to="2026-01-01")


def test_research_budget_requires_all_eight_explicit_nonnegative_fields(
    make_research_budget,
) -> None:
    budget = make_research_budget()
    assert budget.max_provider_pages == 10
    for field in (
        "max_provider_pages",
        "max_provider_records",
        "max_llm_calls",
        "max_llm_tokens",
        "max_cost_estimate",
        "max_candidate_repositories",
        "max_adaptation_attempts",
        "max_workflow_walltime_seconds",
    ):
        payload = budget.model_dump()
        del payload[field]
        with pytest.raises(pydantic.ValidationError):
            ResearchBudget(**payload)
    with pytest.raises(pydantic.ValidationError):
        make_research_budget(max_provider_pages=-1)
    with pytest.raises(pydantic.ValidationError):
        make_research_budget(max_provider_records="100")
    with pytest.raises(pydantic.ValidationError):
        make_research_budget(max_cost_estimate=float("inf"))


def test_training_budget_requires_positive_hard_limits_and_epoch_or_step(
    make_training_budget,
) -> None:
    assert make_training_budget(max_steps=100).max_steps == 100
    with pytest.raises(pydantic.ValidationError):
        make_training_budget(max_epochs=None, max_steps=None)
    with pytest.raises(pydantic.ValidationError):
        make_training_budget(max_epochs=0)
    with pytest.raises(pydantic.ValidationError):
        make_training_budget(max_walltime_seconds="3600")
    with pytest.raises(pydantic.ValidationError):
        make_training_budget(max_test_evaluations=0)


def test_patch_operation_accepts_add_replace_remove_and_json_pointer_escapes(
    make_patch_operation,
) -> None:
    add = make_patch_operation(op=PatchOperationType.ADD, path="/items/0", value={"x": 1})
    replace = make_patch_operation(op=PatchOperationType.REPLACE, path="/a~1b/~0name", value=False)
    remove = make_patch_operation(op=PatchOperationType.REMOVE, path="/items/0", value=None)
    assert add.value == {"x": 1}
    assert replace.path == "/a~1b/~0name"
    assert remove.value is None
    assert PatchOperation.model_validate_json(remove.model_dump_json()) == remove


@pytest.mark.parametrize("path", ["", "not-a-pointer", "root", "/bad~2escape", "/bad~"])
def test_patch_operation_rejects_root_and_invalid_json_pointer(
    make_patch_operation, path: str
) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_patch_operation(path=path)


def test_patch_operation_enforces_operation_value_contract_and_json_bounds(
    make_patch_operation,
) -> None:
    with pytest.raises(pydantic.ValidationError):
        make_patch_operation(op=PatchOperationType.ADD, value=None)
    with pytest.raises(pydantic.ValidationError):
        make_patch_operation(op=PatchOperationType.REPLACE, value=None)
    with pytest.raises(pydantic.ValidationError):
        make_patch_operation(op=PatchOperationType.REMOVE, value="not-null")
    with pytest.raises(pydantic.ValidationError):
        make_patch_operation(value={"bad": float("nan")})
    with pytest.raises(pydantic.ValidationError):
        make_patch_operation(value={"payload": "x" * (16 * 1024)})


def test_metric_summary_enforces_spread_and_confidence_interval(make_metric_summary) -> None:
    assert make_metric_summary(spread=0.0, ci_lower=0.7, ci_upper=0.9)
    with pytest.raises(pydantic.ValidationError):
        make_metric_summary(spread=-0.1)
    with pytest.raises(pydantic.ValidationError):
        make_metric_summary(ci_lower=0.9, ci_upper=0.7)


def test_risk_finding_requires_typed_category_severity_and_bounded_description(
    make_risk_finding,
) -> None:
    assert make_risk_finding()
    with pytest.raises(pydantic.ValidationError):
        make_risk_finding(category="NETWORK")
    with pytest.raises(pydantic.ValidationError):
        make_risk_finding(severity="HIGH")
    with pytest.raises(pydantic.ValidationError):
        make_risk_finding(description="")
    with pytest.raises(pydantic.ValidationError):
        make_risk_finding(description="x" * 1025)


def test_nested_json_roundtrip_factory(make_repository_snapshot, make_risk_finding) -> None:
    snapshot = make_repository_snapshot(risk_findings=[make_risk_finding()])
    assert type(snapshot).model_validate_json(snapshot.model_dump_json()) == snapshot


def test_adaptation_attempt_json_roundtrip(make_adaptation_attempt) -> None:
    attempt = make_adaptation_attempt()
    assert AdaptationAttempt.model_validate_json(attempt.model_dump_json()) == attempt
