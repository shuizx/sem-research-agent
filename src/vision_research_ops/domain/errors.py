"""Strict shared constraints and serializable domain failures.

The aliases in this module are deliberately strict at the boundary.  Domain
objects never silently turn strings into numbers, booleans into integers, or
non-finite floating-point values into JSON ``null``.  Datetime and date values
are the narrow exceptions: their explicit parsers accept their documented JSON
wire forms and normalize them to canonical Python values.
"""

import json
import math
import re
from datetime import UTC, date, datetime
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
)
from pydantic import (
    JsonValue as PydanticJsonValue,
)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:[Zz]|[+-]\d{2}:\d{2})$"
)
_NETWORK_ALLOWLIST_REF_RE = re.compile(r"^netref_[a-z0-9][a-z0-9_-]{0,63}$")

_JSON_MAX_BYTES = 16 * 1024
_JSON_MAX_DEPTH = 8


def _validate_non_blank(value: str) -> str:
    """Reject empty or whitespace-only strings without trimming valid values."""
    if value.strip() == "":
        raise ValueError("string must not be empty or whitespace-only")
    return value


def _validate_finite(value: float) -> float:
    """Reject NaN and positive/negative infinity."""
    if not math.isfinite(value):
        raise ValueError("value must be a finite number")
    return value


def _validate_network_allowlist_ref(value: str) -> str:
    """Require the exact canonical network-policy handle grammar."""
    if _NETWORK_ALLOWLIST_REF_RE.fullmatch(value) is None:
        raise ValueError("network allowlist reference must use the canonical netref_* grammar")
    return value


def _coerce_utc(value: object) -> datetime:
    """Accept an aware datetime or RFC 3339 string and normalize it to UTC."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        if not _RFC3339_RE.fullmatch(value):
            raise ValueError("timestamp must be an RFC 3339 string with an explicit offset")
        try:
            normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError("timestamp must be a valid RFC 3339 datetime") from exc
    else:
        raise ValueError("expected an aware datetime or RFC 3339 string")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("naive datetime is not allowed; an explicit UTC offset is required")
    return parsed.astimezone(UTC)


def _serialize_utc(value: datetime) -> str:
    """Serialize UTC datetimes using the RFC 3339 ``Z`` spelling."""
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _coerce_iso_date(value: object) -> date:
    """Accept a ``date`` or exact ``YYYY-MM-DD`` string, never a datetime."""
    if isinstance(value, datetime):
        raise ValueError("datetime is not allowed where an ISO calendar date is required")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        if not _ISO_DATE_RE.fullmatch(value):
            raise ValueError("date must use the YYYY-MM-DD ISO calendar form")
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("date must be a valid ISO calendar date") from exc
    raise ValueError("expected a date or YYYY-MM-DD string")


def _serialize_iso_date(value: date) -> str:
    """Serialize dates in their exact ISO calendar representation."""
    return value.isoformat()


def _validate_json_node(value: object, depth: int) -> None:
    """Recursively validate the strict JSON subset and its nesting depth."""
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must not contain NaN or infinity")
        return

    next_depth = depth + 1
    if next_depth > _JSON_MAX_DEPTH:
        raise ValueError(f"JSON payload nesting depth must not exceed {_JSON_MAX_DEPTH}")

    if type(value) is list:
        for item in value:
            _validate_json_node(item, next_depth)
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            _validate_json_node(item, next_depth)
        return
    raise ValueError("value is not a standard JSON scalar, array, or object")


def _validate_json_payload(value: object) -> None:
    """Validate a whole payload, including deterministic UTF-8 size accounting."""
    _validate_json_node(value, depth=0)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ValueError("value is not serializable as a strict UTF-8 JSON payload") from exc
    if len(encoded) > _JSON_MAX_BYTES:
        raise ValueError(f"JSON payload must not exceed {_JSON_MAX_BYTES} bytes")


def _validate_json_value(value: object) -> object:
    """Pydantic hook for a bounded, finite JSON value."""
    _validate_json_payload(value)
    return value


def _validate_json_object(value: object) -> object:
    """Pydantic hook for a bounded JSON object with string keys."""
    if type(value) is not dict:
        raise ValueError("expected a JSON object")
    _validate_json_payload(value)
    return value


#: Stable `<AREA>_<CATEGORY>_<DETAIL>` identifier; it is extensible, not an enum.
type ReasonCode = Annotated[
    StrictStr,
    Field(pattern=r"^[A-Z0-9]+(?:_[A-Z0-9]+){2,}$"),
]

#: ``sha256:<64 lowercase hexadecimal characters>`` content digest.
type ContentHash = Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]

#: Complete lowercase hexadecimal Git object identifier (SHA-1 or SHA-256 length).
type GitCommitSha = Annotated[StrictStr, Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")]

#: Opaque network-policy handle resolved only by trusted settings/policy code.
type NetworkAllowlistRef = Annotated[
    StrictStr,
    Field(pattern=r"^netref_[a-z0-9][a-z0-9_-]{0,63}$"),
    AfterValidator(_validate_network_allowlist_ref),
]

#: Opaque identifier; never infer a semantic structure from it.
type OpaqueId = Annotated[StrictStr, Field(min_length=1), AfterValidator(_validate_non_blank)]

#: Strict non-empty string for names, URI references, and keys.
type NonBlankStr = Annotated[StrictStr, Field(min_length=1), AfterValidator(_validate_non_blank)]

#: Human-facing, de-sensitized text with an explicit contract cap.
type HumanText = Annotated[
    StrictStr,
    Field(min_length=1, max_length=1024),
    AfterValidator(_validate_non_blank),
]

#: Extensible internal failure family used for audit grouping, not API projection.
type FailureCategory = Annotated[
    StrictStr,
    Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$"),
]

#: Explicit UTC parser; JSON output is RFC 3339 with a trailing ``Z``.
type UTCDateTime = Annotated[
    datetime,
    BeforeValidator(_coerce_utc),
    PlainSerializer(_serialize_utc, return_type=str, when_used="json"),
]

#: ISO calendar date parser; JSON output is always ``YYYY-MM-DD``.
type ISODate = Annotated[
    date,
    BeforeValidator(_coerce_iso_date),
    PlainSerializer(_serialize_iso_date, return_type=str, when_used="json"),
]

#: Strict boolean; numeric 0/1 and text booleans are rejected.
type StrictBoolean = StrictBool

#: Strict finite number.
type FiniteFloat = Annotated[
    StrictFloat,
    Field(allow_inf_nan=False),
    AfterValidator(_validate_finite),
]

#: Strict finite number greater than zero.
type PositiveFiniteFloat = Annotated[
    StrictFloat,
    Field(gt=0.0, allow_inf_nan=False),
    AfterValidator(_validate_finite),
]

#: Strict finite number greater than or equal to zero.
type NonNegativeFiniteFloat = Annotated[
    StrictFloat,
    Field(ge=0.0, allow_inf_nan=False),
    AfterValidator(_validate_finite),
]

#: Strict finite number in the closed interval ``[0, 1]``.
type UnitInterval = Annotated[
    StrictFloat,
    Field(ge=0.0, le=1.0, allow_inf_nan=False),
    AfterValidator(_validate_finite),
]

#: Strict finite number in the open interval ``(0, 1)``.
type OpenUnitInterval = Annotated[
    StrictFloat,
    Field(gt=0.0, lt=1.0, allow_inf_nan=False),
    AfterValidator(_validate_finite),
]

#: Strict integer greater than or equal to zero.
type NonNegativeInt = Annotated[StrictInt, Field(ge=0)]

#: Strict integer greater than zero.
type PositiveInt = Annotated[StrictInt, Field(gt=0)]

#: Strict signed integer used where negative process exit values remain meaningful.
type StrictInteger = StrictInt

#: Strict finite, bounded standard JSON scalar/array/object value.
type JsonValue = Annotated[PydanticJsonValue, BeforeValidator(_validate_json_value)]

#: Strict finite, bounded standard JSON object with only string keys.
type JsonObject = Annotated[dict[str, JsonValue], BeforeValidator(_validate_json_object)]


class DomainErrorModel(BaseModel):
    """Base class for serializable domain errors with strict input semantics."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class StructuredFailure(DomainErrorModel):
    """Stable de-sensitized failure summary retained with domain records.

    ``category`` and ``message_hash`` remain internal-audit fields.  API layers
    project only the stable user-facing subset defined in the interface contract.
    Raw exception details belong in de-sensitized logs or immutable artifacts.
    """

    schema_version: Literal["1"]
    code: ReasonCode
    category: FailureCategory
    message: HumanText
    message_hash: ContentHash
    retryable: StrictBoolean
    correlation_id: OpaqueId | None = None
    details: JsonObject = Field(default_factory=dict)


__all__ = [
    "ContentHash",
    "DomainErrorModel",
    "FailureCategory",
    "FiniteFloat",
    "GitCommitSha",
    "HumanText",
    "ISODate",
    "JsonObject",
    "JsonValue",
    "NetworkAllowlistRef",
    "NonBlankStr",
    "NonNegativeFiniteFloat",
    "NonNegativeInt",
    "OpaqueId",
    "OpenUnitInterval",
    "PositiveFiniteFloat",
    "PositiveInt",
    "ReasonCode",
    "StrictBoolean",
    "StrictInteger",
    "StructuredFailure",
    "UTCDateTime",
    "UnitInterval",
]
