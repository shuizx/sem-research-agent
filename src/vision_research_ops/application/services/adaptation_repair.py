"""Single deterministic repair transformation for a failed fixture smoke."""

from __future__ import annotations

from datetime import datetime

from .adaptation_models import CompiledAdaptationPlan, RepairRecord, SmokeResultRecord


def repair_failed_fixture_plan(
    plan: CompiledAdaptationPlan,
    smoke: SmokeResultRecord,
    *,
    now: datetime,
) -> CompiledAdaptationPlan:
    """Advance the fixture adapter revision exactly once after an observed failure."""
    if smoke.status != "FAILED" or not smoke.retryable:
        raise ValueError("deterministic repair requires a retryable failed smoke result")
    if smoke.patch_hash == "":
        raise ValueError("repair requires exact failed patch provenance")
    if plan.repair_revision != 0 or plan.repair_history:
        raise ValueError("adaptation permits at most one automatic repair")
    next_revision = plan.revision + 1
    failure_stage = next(
        stage.stage.value for stage in smoke.stages if stage.status.value == "FAILED"
    )
    repair = RepairRecord(
        from_revision=plan.revision,
        to_revision=next_revision,
        reason_code=f"REPAIR_AFTER_{failure_stage}",
        repaired_at=now,
    )
    return plan.model_copy(
        update={
            "revision": next_revision,
            "repair_revision": 1,
            "repair_history": [repair],
            "origin": "DETERMINISTIC_REPAIR",
            "updated_at": now,
        }
    )


__all__ = ["repair_failed_fixture_plan"]
