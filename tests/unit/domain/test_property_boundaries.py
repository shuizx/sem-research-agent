"""Hypothesis-driven property tests for domain constraints.

These are pure, offline, deterministic-in-distribution property tests for
constrained scalars that are a natural fit for property-based testing. They do
not aim to exhaustively duplicate the explicit boundary tests elsewhere.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict, ValidationError

from vision_research_ops.domain.errors import (
    ContentHash,
    FiniteFloat,
    ReasonCode,
    UnitInterval,
    UTCDateTime,
)

pytestmark = pytest.mark.unit


class _DateHolder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    at: UTCDateTime


class _FloatHolder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    v: FiniteFloat


class _UnitHolder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    v: UnitInterval


class _HashHolder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    h: ContentHash


class _ReasonHolder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: ReasonCode


_seg_char = st.characters(min_codepoint=48, max_codepoint=57) | st.characters(
    min_codepoint=65, max_codepoint=90
)
segs = st.text(alphabet=_seg_char, min_size=1, max_size=8)

_offsets = st.integers(min_value=-12 * 60, max_value=14 * 60).map(
    lambda minutes: timedelta(minutes=minutes)
)


@given(
    dt=st.builds(
        datetime,
        year=st.integers(2000, 2099),
        month=st.integers(1, 12),
        day=st.integers(1, 28),
        hour=st.integers(0, 23),
        minute=st.integers(0, 59),
        second=st.integers(0, 59),
        tzinfo=st.just(UTC),
    ),
)
def test_utc_datetime_roundtrip_preserves_instant(dt: datetime) -> None:
    model = _DateHolder(at=dt)
    rebuilt = _DateHolder.model_validate_json(model.model_dump_json())
    assert rebuilt.at.timestamp() == dt.timestamp()
    assert rebuilt.at.utcoffset() == timedelta(0)


@given(
    dt=st.builds(
        datetime,
        year=st.integers(2000, 2099),
        month=st.integers(1, 12),
        day=st.integers(1, 28),
        hour=st.integers(0, 23),
        minute=st.integers(0, 59),
        second=st.integers(0, 59),
        tzinfo=st.just(UTC),
    ),
    offset=_offsets,
)
def test_utc_datetime_accepts_width_range_of_offsets(dt: datetime, offset: timedelta) -> None:
    value = dt.astimezone(timezone(offset))
    assert abs(value.utcoffset()) == abs(offset)
    model = _DateHolder(at=value)
    assert model.at.utcoffset() == timedelta(0)


@given(v=st.floats(allow_nan=False, allow_infinity=False))
def test_finite_float_accepts_only_finite(v: float) -> None:
    assert _FloatHolder(v=v).v == v


@given(v=st.floats(allow_nan=True, allow_infinity=True))
def test_finite_float_rejects_non_finite(v: float) -> None:
    if v != v or v in (float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            _FloatHolder(v=v)


@given(v=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False))
def test_unit_interval_bounds_are_enforced(v: float) -> None:
    if 0.0 <= v <= 1.0:
        assert _UnitHolder(v=v).v == v
    else:
        with pytest.raises(ValidationError):
            _UnitHolder(v=v)


@given(seg1=segs, seg2=segs, seg3=segs, extra=st.lists(segs, max_size=4))
def test_reason_code_multi_segment_uppercase(
    seg1: str, seg2: str, seg3: str, extra: list[str]
) -> None:
    code = "_".join([seg1, seg2, seg3, *extra])
    assert _ReasonHolder(code=code).code == code


@given(
    hex_chars=st.text(
        alphabet=st.characters(min_codepoint=48, max_codepoint=57)
        | st.characters(min_codepoint=97, max_codepoint=102),
        min_size=64,
        max_size=64,
    )
)
def test_sha256_lowercase_hex_accepted(hex_chars: str) -> None:
    candidate = "sha256:" + "".join(hex_chars)
    assert _HashHolder(h=candidate).h == candidate
