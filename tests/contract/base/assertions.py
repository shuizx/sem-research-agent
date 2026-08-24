"""Shared assertions for deterministic concrete port-contract implementations."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, Self, TypeVar, get_args, get_origin, get_type_hints

import pytest

from vision_research_ops.ports import PortError

_PROTOCOL_DUNDER_METHODS = frozenset({"__aenter__", "__aexit__", "__aiter__", "__anext__"})
_INVARIANT_ORIGINS = frozenset({dict, frozenset, list, set})


@dataclass(frozen=True)
class ConcreteProtocolReport:
    """Count of automatically discovered Protocol methods checked against concrete fakes."""

    protocol_count: int
    implementation_count: int
    method_count: int


def discover_protocol_methods(protocol: type[object]) -> tuple[str, ...]:
    """Derive every public callable contract method from a Protocol and its Protocol MRO."""
    discovered: dict[str, None] = {}
    for base in reversed(protocol.__mro__):
        if base in (object, Protocol) or not getattr(base, "_is_protocol", False):
            continue
        for method_name, member in base.__dict__.items():
            if not callable(member):
                continue
            if method_name.startswith("_") and method_name not in _PROTOCOL_DUNDER_METHODS:
                continue
            discovered[method_name] = None
    return tuple(discovered)


def assert_concrete_protocol_signatures(
    protocol: type[object],
    implementation: object,
) -> int:
    """Assert a fake implements every automatically discovered Protocol method exactly."""
    method_names = discover_protocol_methods(protocol)
    assert method_names, f"{protocol.__name__} has no concrete methods to assert"
    for method_name in method_names:
        protocol_method = getattr(protocol, method_name)
        implementation_method = getattr(type(implementation), method_name, None)
        assert implementation_method is not None, (
            f"{type(implementation).__name__} is missing {protocol.__name__}.{method_name}"
        )
        assert inspect.iscoroutinefunction(implementation_method) is inspect.iscoroutinefunction(
            protocol_method
        ), f"{type(implementation).__name__}.{method_name} has the wrong async/sync shape"
        expected_signature = inspect.signature(protocol_method)
        actual_signature = inspect.signature(implementation_method)
        expected_parameters = tuple(expected_signature.parameters.values())
        actual_parameters = tuple(actual_signature.parameters.values())
        assert tuple(parameter.name for parameter in actual_parameters) == tuple(
            parameter.name for parameter in expected_parameters
        ), f"{type(implementation).__name__}.{method_name} has different parameter names"
        for expected, actual in zip(expected_parameters, actual_parameters, strict=True):
            assert actual.kind is expected.kind, (
                f"{type(implementation).__name__}.{method_name}.{expected.name} has "
                "different positional/keyword-only semantics"
            )
            assert (actual.default is inspect.Parameter.empty) is (
                expected.default is inspect.Parameter.empty
            ), f"{type(implementation).__name__}.{method_name}.{expected.name} changes requiredness"
            if expected.default is not inspect.Parameter.empty:
                assert actual.default == expected.default, (
                    f"{type(implementation).__name__}.{method_name}.{expected.name} "
                    "changes its default value"
                )
        if "ctx" in expected_signature.parameters:
            assert actual_signature.parameters["ctx"].kind is inspect.Parameter.KEYWORD_ONLY
            assert actual_signature.parameters["ctx"].default is inspect.Parameter.empty
        expected_return = get_type_hints(protocol_method).get("return", inspect.Signature.empty)
        actual_return = get_type_hints(implementation_method).get("return", inspect.Signature.empty)
        assert expected_return is not inspect.Signature.empty, (
            f"{protocol.__name__}.{method_name} is missing a return annotation"
        )
        assert actual_return is not inspect.Signature.empty, (
            f"{type(implementation).__name__}.{method_name} is missing a return annotation"
        )
        assert _return_annotation_is_compatible(
            expected_return,
            actual_return,
            implementation_type=type(implementation),
        ), (
            f"{type(implementation).__name__}.{method_name} has incompatible return annotation "
            f"{actual_return!r}; expected a covariant form of {expected_return!r}"
        )
    return len(method_names)


def assert_public_protocol_registry(
    registry: Mapping[type[object], Callable[[], object]],
    *,
    public_protocols: Iterable[type[object]],
    deferred_protocols: Iterable[type[object]] = (),
) -> ConcreteProtocolReport:
    """Check that every non-deferred exported Protocol has one concrete fake and all methods."""
    expected_protocols = set(public_protocols) - set(deferred_protocols)
    registered_protocols = set(registry)
    missing = sorted(protocol.__name__ for protocol in expected_protocols - registered_protocols)
    extra = sorted(protocol.__name__ for protocol in registered_protocols - expected_protocols)
    assert registered_protocols == expected_protocols, (
        f"public Protocol registry mismatch: missing={missing}, extra={extra}"
    )
    method_count = 0
    implementations: set[type[object]] = set()
    for protocol, factory in registry.items():
        implementation = factory()
        implementations.add(type(implementation))
        method_count += assert_concrete_protocol_signatures(protocol, implementation)
    return ConcreteProtocolReport(
        protocol_count=len(expected_protocols),
        implementation_count=len(implementations),
        method_count=method_count,
    )


def _return_annotation_is_compatible(
    expected: object,
    actual: object,
    *,
    implementation_type: type[object],
) -> bool:
    """Check resolved return annotations with explicit variance and TypeVar compatibility."""
    if expected is Any or actual is Any:
        return False
    if expected is Self:
        return isinstance(actual, type) and issubclass(actual, implementation_type)
    if isinstance(expected, TypeVar):
        return _type_satisfies_typevar(actual, expected, implementation_type=implementation_type)
    if isinstance(actual, TypeVar):
        return _typevar_satisfies_annotation(
            actual, expected, implementation_type=implementation_type
        )
    expected_pydantic = getattr(expected, "__pydantic_generic_metadata__", None)
    actual_pydantic = getattr(actual, "__pydantic_generic_metadata__", None)
    if isinstance(expected_pydantic, dict) or isinstance(actual_pydantic, dict):
        if not isinstance(expected_pydantic, dict) or not isinstance(actual_pydantic, dict):
            return False
        if expected_pydantic.get("origin") is not actual_pydantic.get("origin"):
            return False
        return _generic_arguments_compatible(
            expected_pydantic.get("args", ()),
            actual_pydantic.get("args", ()),
            origin=expected_pydantic.get("origin"),
            implementation_type=implementation_type,
        )
    expected_origin = get_origin(expected)
    actual_origin = get_origin(actual)
    if expected_origin is not None or actual_origin is not None:
        if expected_origin != actual_origin:
            return False
        return _generic_arguments_compatible(
            get_args(expected),
            get_args(actual),
            origin=expected_origin,
            implementation_type=implementation_type,
        )
    if expected == actual:
        return True
    if isinstance(expected, type) and isinstance(actual, type):
        try:
            return issubclass(actual, expected)
        except TypeError:
            return False
    return False


def _generic_arguments_compatible(
    expected_args: tuple[object, ...],
    actual_args: tuple[object, ...],
    *,
    origin: object,
    implementation_type: type[object],
) -> bool:
    """Apply strict invariant container rules and covariance only where the origin permits it."""
    if len(expected_args) != len(actual_args):
        return False
    if origin in _INVARIANT_ORIGINS:
        return all(
            _annotations_equal(expected, actual)
            for expected, actual in zip(expected_args, actual_args, strict=True)
        )
    return all(
        _return_annotation_is_compatible(
            expected,
            actual,
            implementation_type=implementation_type,
        )
        for expected, actual in zip(expected_args, actual_args, strict=True)
    )


def _annotations_equal(expected: object, actual: object) -> bool:
    """Compare invariant generic arguments without accepting subtype substitution."""
    if expected is Any or actual is Any:
        return False
    if isinstance(expected, TypeVar) or isinstance(actual, TypeVar):
        return expected is actual
    expected_origin = get_origin(expected)
    actual_origin = get_origin(actual)
    if expected_origin is not None or actual_origin is not None:
        return expected_origin == actual_origin and all(
            _annotations_equal(left, right)
            for left, right in zip(get_args(expected), get_args(actual), strict=True)
        )
    return expected == actual


def _type_satisfies_typevar(
    actual: object,
    expected: TypeVar,
    *,
    implementation_type: type[object],
) -> bool:
    """Accept only a matching or demonstrably bounded/constraint-compatible TypeVar."""
    if isinstance(actual, TypeVar):
        if actual is expected:
            return True
        if expected.__bound__ is None and not expected.__constraints__:
            return actual.__bound__ is not None or bool(actual.__constraints__)
        if expected.__bound__ is not None:
            return _typevar_satisfies_annotation(
                actual,
                expected.__bound__,
                implementation_type=implementation_type,
            )
        if actual.__bound__ is not None:
            return any(
                _return_annotation_is_compatible(
                    constraint,
                    actual.__bound__,
                    implementation_type=implementation_type,
                )
                for constraint in expected.__constraints__
            )
        if actual.__constraints__:
            return all(
                any(
                    _return_annotation_is_compatible(
                        constraint,
                        actual_constraint,
                        implementation_type=implementation_type,
                    )
                    for constraint in expected.__constraints__
                )
                for actual_constraint in actual.__constraints__
            )
        return False
    if expected.__constraints__:
        return any(
            _return_annotation_is_compatible(
                constraint,
                actual,
                implementation_type=implementation_type,
            )
            for constraint in expected.__constraints__
        )
    if expected.__bound__ is None:
        return True
    return _return_annotation_is_compatible(
        expected.__bound__,
        actual,
        implementation_type=implementation_type,
    )


def _typevar_satisfies_annotation(
    actual: TypeVar,
    expected: object,
    *,
    implementation_type: type[object],
) -> bool:
    """Require a TypeVar bound or constraint to prove compatibility with one concrete return."""
    if actual.__constraints__:
        return all(
            _return_annotation_is_compatible(
                expected,
                constraint,
                implementation_type=implementation_type,
            )
            for constraint in actual.__constraints__
        )
    if actual.__bound__ is None:
        return False
    return _return_annotation_is_compatible(
        expected,
        actual.__bound__,
        implementation_type=implementation_type,
    )


def assert_structured_failure(
    error: PortError,
    *,
    code: str,
    retryable: bool,
    correlation_id: str,
) -> None:
    """Assert stable failure semantics without inspecting a raw exception message."""
    assert error.failure.code == code
    assert error.failure.retryable is retryable
    assert error.failure.correlation_id == correlation_id


async def assert_replay_and_conflict[T](
    *,
    first_call: Callable[[], Awaitable[T]],
    replay_call: Callable[[], Awaitable[T]],
    conflicting_call: Callable[[], Awaitable[object]],
    error_type: type[PortError],
    expected_result: T,
) -> None:
    """Assert safe idempotent replay and an explicit conflicting-payload failure."""
    assert await first_call() == expected_result
    assert await replay_call() == expected_result
    with pytest.raises(error_type) as raised:
        await conflicting_call()
    assert raised.value.failure.code == "IDEMPOTENCY_KEY_REUSED_WITH_DIFFERENT_PAYLOAD"
    assert raised.value.failure.retryable is False
