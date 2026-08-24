"""Private JSON safety helpers shared by port errors and deterministic fakes.

The module is intentionally private: it implements internal redaction and
fingerprinting mechanics, not an application-facing port contract.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from hashlib import sha256

from pydantic import BaseModel

from vision_research_ops.domain import JsonObject, JsonValue

_REDACTED = "[REDACTED]"
_NON_FINITE = "[NON_FINITE_FLOAT]"
_SENSITIVE_KEY_PARTS = (
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_MESSAGE_SECRET_PATTERN = re.compile(
    r"""(?ix)
    (?<![a-z0-9])
    (?P<key>
        ["']?
        (?:api[\s_-]*key|authorization|password|credential|secret|token)
        ["']?
    )
    (?P<separator>\s*(?::|=)\s*|\s+)
    (?:
        (?P<value_quote>["'])
        [^"'\r\n]*
        (?P=value_quote)
        |
        (?:bearer\s+)?[^\s,;}&\]\r\n#]+
    )
    """
)


def is_secret_like_key(key: str) -> bool:
    """Return whether a mapping key is secret-like, ignoring case and separators."""
    normalized = "".join(character for character in key.casefold() if character.isalnum())
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def redact_message(message: str, *, secret_values: object | None = None) -> str:
    """Remove recognized secret values and common credential assignments from text."""
    safe = message
    try:
        secrets = _collect_secret_strings(secret_values)
    except Exception:
        secrets = set()
    for secret in sorted(secrets, key=lambda item: (-len(item), item)):
        if secret:
            safe = safe.replace(secret, _REDACTED)
    return _MESSAGE_SECRET_PATTERN.sub(
        lambda match: f"{match.group('key')}{match.group('separator')}{_REDACTED}",
        safe,
    )


def redact_for_observation(value: object, *, key: str | None = None) -> JsonValue:
    """Return a recursively redacted, JSON-safe observation without object reprs."""
    if key is not None and is_secret_like_key(key):
        return _REDACTED
    if isinstance(value, BaseModel):
        return redact_for_observation(value.model_dump(mode="python"), key=key)
    if isinstance(value, type) and issubclass(value, BaseModel):
        return {"pydantic_model": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, Enum):
        return redact_for_observation(value.value, key=key)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"sha256": f"sha256:{sha256(value).hexdigest()}", "size_bytes": len(value)}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _NON_FINITE
    if isinstance(value, list | tuple):
        if len(value) == 2 and isinstance(value[0], str) and is_secret_like_key(value[0]):
            return [
                redact_for_observation(value[0]),
                redact_for_observation(value[1], key=value[0]),
            ]
        return [redact_for_observation(item) for item in value]
    if isinstance(value, Mapping):
        try:
            return {
                _safe_mapping_key(item_key): redact_for_observation(
                    item_value,
                    key=_safe_mapping_key(item_key),
                )
                for item_key, item_value in value.items()
            }
        except Exception:
            return {"type": type(value).__name__, "observation": "[UNAVAILABLE]"}
    return {"type": type(value).__name__}


def redact_json_object(value: Mapping[str, object]) -> JsonObject:
    """Return a recursively redacted object suitable for ``StructuredFailure.details``."""
    redacted = redact_for_observation(value)
    if not isinstance(redacted, dict):  # pragma: no cover - mapping inputs always become objects
        raise TypeError("redacted details must be a JSON object")
    return redacted


def canonical_json_bytes(value: object) -> bytes:
    """Encode a finite canonical JSON value without redacting its comparison semantics."""
    normalized = _canonicalize(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_hash(value: object) -> str:
    """Return a non-reversible hash of the complete, unredacted canonical payload."""
    return f"sha256:{sha256(canonical_json_bytes(value)).hexdigest()}"


def _safe_mapping_key(value: object) -> str:
    """Return an observable mapping key without evaluating an arbitrary object repr."""
    return value if isinstance(value, str) else "<non-string-key>"


def _collect_secret_strings(value: object | None, *, key: str | None = None) -> set[str]:
    """Collect scalar values paired with secret-like keys for message sanitization."""
    if value is None:
        return set()
    if key is not None and is_secret_like_key(key):
        return _scalar_strings(value)
    if isinstance(value, BaseModel):
        return _collect_secret_strings(value.model_dump(mode="python"), key=key)
    if isinstance(value, Mapping):
        mapping_values: set[str] = set()
        try:
            for item_key, item_value in value.items():
                safe_key = _safe_mapping_key(item_key)
                mapping_values.update(_collect_secret_strings(item_value, key=safe_key))
        except Exception:
            return set()
        return mapping_values
    if isinstance(value, list | tuple):
        if len(value) == 2 and isinstance(value[0], str) and is_secret_like_key(value[0]):
            return _scalar_strings(value[1])
        sequence_values: set[str] = set()
        for item in value:
            sequence_values.update(_collect_secret_strings(item))
        return sequence_values
    return set()


def _scalar_strings(value: object) -> set[str]:
    """Flatten supported credential values without using arbitrary object reprs."""
    if isinstance(value, BaseModel):
        return _scalar_strings(value.model_dump(mode="python"))
    if isinstance(value, str):
        return {value}
    if isinstance(value, bytes):
        return set()
    if value is None or isinstance(value, (bool, int, float, Enum, datetime)):
        return {str(value)}
    if isinstance(value, Mapping):
        mapping_values: set[str] = set()
        try:
            for nested in value.values():
                mapping_values.update(_scalar_strings(nested))
        except Exception:
            return set()
        return mapping_values
    if isinstance(value, list | tuple):
        sequence_values: set[str] = set()
        for nested in value:
            sequence_values.update(_scalar_strings(nested))
        return sequence_values
    return set()


def _canonicalize(value: object) -> object:
    """Convert supported values into finite standard JSON without retaining the result."""
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="python"))
    if isinstance(value, type) and issubclass(value, BaseModel):
        return {"pydantic_model": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"sha256": f"sha256:{sha256(value).hexdigest()}", "size_bytes": len(value)}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON rejects non-finite floats")
        return value
    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for item_key, item_value in value.items():
            if not isinstance(item_key, str):
                raise TypeError("canonical JSON requires string mapping keys")
            result[item_key] = _canonicalize(item_value)
        return result
    raise TypeError("canonical JSON accepts only supported JSON boundary values")
