"""Deterministic adaptation patch template and explicit filesystem policy."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Literal, cast

from vision_research_ops.domain import JsonObject

from .adaptation_models import REQUIRED_PATCH_FIELDS, CompiledAdaptationPlan

ADAPTATION_CONFIG_PATH: Literal["sem_adaptation.json"] = "sem_adaptation.json"
ALLOWED_PATCH_FILES: frozenset[str] = frozenset({ADAPTATION_CONFIG_PATH})
ALLOWED_PATCH_FIELDS: frozenset[str] = REQUIRED_PATCH_FIELDS | frozenset(
    {"/revision", "/repair_revision", "/fixture_contract"}
)
_SECRET_NAMES = frozenset(
    {
        ".env",
        ".git",
        "credentials",
        "id_rsa",
        "secrets",
        "token",
    }
)
_BINARY_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".dll", ".exe", ".jpg", ".onnx", ".png", ".pt", ".pth"}
)
_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class PatchPolicyError(ValueError):
    """Raised before any filesystem mutation when a patch target is unsafe."""


def validate_patch_path(path: str) -> str:
    """Accept only the one canonical text config path used by the fixture."""
    if not isinstance(path, str) or path != path.strip() or not path:
        raise PatchPolicyError("patch path must be a non-blank canonical string")
    if (
        "\\" in path
        or "%" in path
        or path.startswith("/")
        or path.startswith("//")
        or _DRIVE_RE.match(path)
    ):
        raise PatchPolicyError("absolute, encoded, UNC, drive, and backslash paths are forbidden")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PatchPolicyError("patch paths cannot traverse or contain ambiguous segments")
    if any(part.casefold() in _SECRET_NAMES for part in parts):
        raise PatchPolicyError("control and secret paths are forbidden")
    if PurePosixPath(path).suffix.casefold() in _BINARY_SUFFIXES:
        raise PatchPolicyError("binary patch targets are forbidden")
    if path not in ALLOWED_PATCH_FILES:
        raise PatchPolicyError("patch path is outside the adaptation fixture allowlist")
    return path


def validate_patch_fields(fields: Sequence[str]) -> list[str]:
    """Reject dependency or unrecognized config mutations before compilation."""
    if not fields or len(fields) != len(set(fields)):
        raise PatchPolicyError("patch fields must be a non-empty unique list")
    if any(field not in ALLOWED_PATCH_FIELDS for field in fields):
        raise PatchPolicyError("patch contains an unapproved config field")
    return list(fields)


def canonical_json_bytes(value: object) -> bytes:
    """Encode a deterministic UTF-8 JSON document."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def content_hash(value: bytes) -> str:
    """Return the repository-wide content hash spelling."""
    return f"sha256:{sha256(value).hexdigest()}"


def compile_adaptation_config(plan: CompiledAdaptationPlan) -> JsonObject:
    """Compile the validated proposal into the sole executable fixture contract."""
    validate_patch_path(ADAPTATION_CONFIG_PATH)
    proposal = plan.proposal
    return cast(
        JsonObject,
        {
            "schema_version": "1",
            "fixture_contract": "SEM_PLAIN_PYTORCH_CONFIG_V1",
            "revision": plan.revision,
            "repair_revision": plan.repair_revision,
            "input": {
                "modality": "GRAYSCALE",
                "channels": proposal.channels,
            },
            "model": {"num_classes": proposal.num_classes},
            "data": {
                "label_mapping": dict(proposal.label_mapping),
                "group_split_strategy": "GROUP_HOLDOUT",
                "group_split_key": proposal.group_split_key,
            },
            "metrics": {
                "names": list(proposal.metrics),
                "output_file": proposal.metrics_output_file,
            },
        },
    )


__all__ = [
    "ADAPTATION_CONFIG_PATH",
    "ALLOWED_PATCH_FIELDS",
    "ALLOWED_PATCH_FILES",
    "PatchPolicyError",
    "canonical_json_bytes",
    "compile_adaptation_config",
    "content_hash",
    "validate_patch_fields",
    "validate_patch_path",
]
