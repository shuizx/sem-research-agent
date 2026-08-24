"""Strict scalar, wire-format, and bounded-JSON contract regression tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from vision_research_ops.domain import (
    ContentHash,
    FiniteFloat,
    GitCommitSha,
    ISODate,
    JsonObject,
    JsonValue,
    NetworkPolicy,
    NonNegativeInt,
    OpenUnitInterval,
    PositiveInt,
    ReasonCode,
    StrictBoolean,
    UnitInterval,
    UTCDateTime,
)
from vision_research_ops.domain.errors import DomainErrorModel
from vision_research_ops.domain.models import DomainModel

pytestmark = pytest.mark.unit

SHA256 = "sha256:" + "a" * 64


class _Scalars(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    finite_float: FiniteFloat
    nonnegative_int: NonNegativeInt
    positive_int: PositiveInt
    strict_bool: StrictBoolean


class _DateHolder(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    at: UTCDateTime
    day: ISODate


class _JsonValueHolder(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    value: JsonValue


class _JsonObjectHolder(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    value: JsonObject


class _WireHolder(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    number: FiniteFloat
    policy: NetworkPolicy
    at: UTCDateTime
    day: ISODate


class _HashHolder(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    content_hash: ContentHash
    commit_sha: GitCommitSha
    reason_code: ReasonCode


def _nested_object(depth: int) -> object:
    value: object = 0
    for _ in range(depth):
        value = {"nested": value}
    return value


def test_schema_version_is_required_and_literal(make_artifact_ref) -> None:
    valid = make_artifact_ref()
    assert valid.schema_version == "1"
    payload = valid.model_dump()
    del payload["schema_version"]
    with pytest.raises(ValidationError):
        type(valid)(**payload)
    with pytest.raises(ValidationError):
        make_artifact_ref(schema_version="2")


def test_unknown_fields_are_forbidden(make_artifact_ref) -> None:
    with pytest.raises(ValidationError):
        make_artifact_ref(unexpected_field=1)


@pytest.mark.parametrize("bad", ["2.5", True, False])
def test_float_rejects_strings_and_booleans(bad: object) -> None:
    with pytest.raises(ValidationError):
        _Scalars(finite_float=bad, nonnegative_int=0, positive_int=1, strict_bool=True)


@pytest.mark.parametrize("bad", ["4096", 1.0, True, False])
def test_integer_rejects_coercion_and_bool(bad: object) -> None:
    with pytest.raises(ValidationError):
        _Scalars(finite_float=1.0, nonnegative_int=bad, positive_int=1, strict_bool=True)


@pytest.mark.parametrize("bad", [0, 1, "true", "false"])
def test_boolean_rejects_numeric_and_text_coercion(bad: object) -> None:
    with pytest.raises(ValidationError):
        _Scalars(finite_float=1.0, nonnegative_int=0, positive_int=1, strict_bool=bad)


def test_strict_scalars_accept_exact_python_types() -> None:
    model = _Scalars(finite_float=1, nonnegative_int=0, positive_int=1, strict_bool=False)
    assert model.finite_float == 1.0
    assert model.nonnegative_int == 0
    assert model.positive_int == 1
    assert model.strict_bool is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_finite_float_rejects_non_finite_values(bad: float) -> None:
    with pytest.raises(ValidationError):
        _Scalars(finite_float=bad, nonnegative_int=0, positive_int=1, strict_bool=True)


def test_unit_intervals_use_closed_and_open_boundaries() -> None:
    class _Intervals(BaseModel):
        model_config = ConfigDict(strict=True, allow_inf_nan=False)

        closed: UnitInterval
        open: OpenUnitInterval

    assert _Intervals(closed=0.0, open=0.5)
    assert _Intervals(closed=1.0, open=0.5)
    for value in (0.0, 1.0):
        with pytest.raises(ValidationError):
            _Intervals(closed=value, open=value)
    with pytest.raises(ValidationError):
        _Intervals(closed=-0.1, open=0.5)


def test_datetime_accepts_aware_values_and_normalizes_utc() -> None:
    model = _DateHolder(at="2026-08-06T08:00:00+08:00", day="2026-08-06")
    assert model.at.tzinfo is not None
    assert model.at.utcoffset().total_seconds() == 0
    assert model.at.hour == 0
    assert model.day == date(2026, 8, 6)
    direct = _DateHolder(at=datetime(2026, 8, 6, tzinfo=UTC), day=date(2026, 8, 6))
    assert direct.at.tzinfo is UTC


@pytest.mark.parametrize(
    ("wire_value", "canonical_value"),
    [
        ("2026-08-06t08:00:00z", "2026-08-06T08:00:00Z"),
        ("2026-08-06t08:00:00+08:00", "2026-08-06T00:00:00Z"),
    ],
)
def test_datetime_python_mode_accepts_lowercase_rfc3339_profile(
    wire_value: str, canonical_value: str
) -> None:
    model = _DateHolder(at=wire_value, day="2026-08-06")
    assert json.loads(model.model_dump_json())["at"] == canonical_value


@pytest.mark.parametrize(
    "bad",
    [
        datetime(2026, 8, 6),
        12345,
        True,
        False,
        "2026-08-06 08:00:00+00:00",
        "2026-08-06t08:00:00",
        "2026-08-06T08:00:00.1234567Z",
        "2026-08-06T08:00:60Z",
    ],
)
def test_datetime_rejects_naive_numeric_bool_and_non_rfc3339(bad: object) -> None:
    with pytest.raises(ValidationError):
        _DateHolder(at=bad, day="2026-08-06")


@pytest.mark.parametrize(
    "bad",
    [datetime(2026, 8, 6, tzinfo=UTC), 12345, True, False, "2026-02-30", "20260806"],
)
def test_iso_date_rejects_datetime_numeric_bool_and_invalid_calendar_date(bad: object) -> None:
    with pytest.raises(ValidationError):
        _DateHolder(at="2026-08-06T08:00:00Z", day=bad)


@pytest.mark.parametrize(
    ("wire_value", "canonical_value"),
    [
        ("2026-08-06t08:00:00z", "2026-08-06T08:00:00Z"),
        ("2026-08-06t08:00:00+08:00", "2026-08-06T00:00:00Z"),
    ],
)
def test_datetime_and_iso_date_json_wire_roundtrip(wire_value: str, canonical_value: str) -> None:
    payload = json.dumps({"at": wire_value, "day": "2026-08-06"})
    model = _DateHolder.model_validate_json(payload)
    encoded = json.loads(model.model_dump_json())
    assert encoded == {"at": canonical_value, "day": "2026-08-06"}
    assert _DateHolder.model_validate_json(model.model_dump_json()) == model


def test_domain_model_validates_constrained_defaults_on_instantiation() -> None:
    class _InvalidDomainDefault(DomainModel):
        schema_version: Literal["1"] = "1"
        positive: PositiveInt = 0

    with pytest.raises(ValidationError):
        _InvalidDomainDefault()


def test_domain_error_model_validates_constrained_defaults_on_instantiation() -> None:
    class _InvalidErrorDefault(DomainErrorModel):
        schema_version: Literal["1"] = "1"
        positive: PositiveInt = 0

    with pytest.raises(ValidationError):
        _InvalidErrorDefault()


def test_legal_json_number_enum_datetime_and_iso_date_pass_in_json_mode() -> None:
    payload = json.dumps(
        {
            "number": 1,
            "policy": "ALLOWLIST",
            "at": "2026-08-06T08:00:00Z",
            "day": "2026-08-06",
        }
    )
    model = _WireHolder.model_validate_json(payload)
    assert model.number == 1.0
    assert model.policy is NetworkPolicy.ALLOWLIST
    assert model.day == date(2026, 8, 6)


def test_hash_reason_and_full_commit_constraints() -> None:
    valid = _HashHolder(
        content_hash=SHA256,
        commit_sha="a" * 40,
        reason_code="VALIDATION_IMPORT_FAILED",
    )
    assert valid.commit_sha == "a" * 40
    assert (
        _HashHolder(
            content_hash=SHA256,
            commit_sha="b" * 64,
            reason_code="VALIDATION_IMPORT_FAILED",
        ).commit_sha
        == "b" * 64
    )
    for bad in ["a" * 7, "A" * 40, "g" * 40, " " + "a" * 40, "a" * 41]:
        with pytest.raises(ValidationError):
            _HashHolder(content_hash=SHA256, commit_sha=bad, reason_code="VALIDATION_IMPORT_FAILED")


def test_json_value_accepts_standard_finite_json_and_roundtrips() -> None:
    value = {"a": [1, 2.5, False, None, {"b": "value"}]}
    model = _JsonValueHolder(value=value)
    assert _JsonValueHolder.model_validate_json(model.model_dump_json()) == model
    assert json.loads(model.model_dump_json())["value"] == value


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_json_value_rejects_non_finite_at_top_level(bad: float) -> None:
    with pytest.raises(ValidationError):
        _JsonValueHolder(value=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_json_value_rejects_non_finite_in_lists_and_dicts(bad: float) -> None:
    with pytest.raises(ValidationError):
        _JsonValueHolder(value=[{"bad": bad}])
    with pytest.raises(ValidationError):
        _JsonObjectHolder(value={"nested": [bad]})


def test_json_value_rejects_nonstandard_values_and_non_string_object_keys() -> None:
    with pytest.raises(ValidationError):
        _JsonValueHolder(value=(1, 2))
    with pytest.raises(ValidationError):
        _JsonObjectHolder(value={1: "not-a-string-key"})  # type: ignore[dict-item]


def test_json_value_depth_boundary_is_eight_containers() -> None:
    assert _JsonValueHolder(value=_nested_object(8))
    with pytest.raises(ValidationError):
        _JsonValueHolder(value=_nested_object(9))


def test_json_value_size_boundary_is_sixteen_kib() -> None:
    empty_payload_size = len(
        json.dumps({"payload": ""}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    boundary = "x" * (16 * 1024 - empty_payload_size)
    assert _JsonObjectHolder(value={"payload": boundary})
    with pytest.raises(ValidationError):
        _JsonObjectHolder(value={"payload": boundary + "x"})


def test_json_non_finite_input_never_serializes_as_null() -> None:
    with pytest.raises(ValidationError):
        _JsonValueHolder(value={"bad": float("nan")})
    valid = _JsonValueHolder(value={"finite": 1.0})
    assert json.loads(valid.model_dump_json()) == {"value": {"finite": 1.0}}
