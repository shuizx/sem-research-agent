"""Reusable foundation port-contract assertions shared by fake and future real-adapter tests."""

from .assertions import (
    assert_concrete_protocol_signatures,
    assert_public_protocol_registry,
    assert_replay_and_conflict,
    assert_structured_failure,
)

__all__ = [
    "assert_concrete_protocol_signatures",
    "assert_public_protocol_registry",
    "assert_replay_and_conflict",
    "assert_structured_failure",
]
