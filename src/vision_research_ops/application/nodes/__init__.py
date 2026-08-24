"""Dependency-injected nodes for fixture and Research Agent graphs."""

from .adaptation import (
    generate_bounded_patch,
    load_and_compare_inputs,
    patch_acceptance_gate,
    plan_adaptation,
    repair_patch_once,
    run_bounded_smoke,
)
from .fixtures import (
    analyze_result_fixture,
    propose_adaptation_fixture,
    retrieve_research_fixture,
    submit_training_fixture,
    validate_request,
)
from .gates import human_gate

__all__ = [
    "analyze_result_fixture",
    "generate_bounded_patch",
    "human_gate",
    "load_and_compare_inputs",
    "patch_acceptance_gate",
    "plan_adaptation",
    "propose_adaptation_fixture",
    "repair_patch_once",
    "retrieve_research_fixture",
    "run_bounded_smoke",
    "submit_training_fixture",
    "validate_request",
]
