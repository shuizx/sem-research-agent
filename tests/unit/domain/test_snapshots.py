"""Deterministic public JSON Schema snapshots and semantic regression checks."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from pydantic import BaseModel

import vision_research_ops.domain as public_domain
import vision_research_ops.domain.errors as errors_module
import vision_research_ops.domain.models as models_module
from vision_research_ops.domain import (
    Approval,
    PerClassSummary,
    ResourceRequest,
    SplitPolicy,
    StructuredFailure,
)

pytestmark = pytest.mark.unit

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "snapshots" / "domain" / "v1"

_ABSTRACT_MODEL_BASES = {models_module.DomainModel, errors_module.DomainErrorModel}


def _defined_concrete_public_models(module: ModuleType) -> set[type[BaseModel]]:
    """Discover public Pydantic models actually defined by one production module."""
    discovered: set[type[BaseModel]] = set()
    for name, candidate in vars(module).items():
        if name.startswith("_") or not inspect.isclass(candidate):
            continue
        if not issubclass(candidate, BaseModel) or candidate in _ABSTRACT_MODEL_BASES:
            continue
        if candidate.__module__ == module.__name__:
            discovered.add(candidate)
    return discovered


def _exported_concrete_public_models() -> set[type[BaseModel]]:
    """Return concrete Pydantic models exposed through the package public surface."""
    exported: set[type[BaseModel]] = set()
    for name in public_domain.__all__:
        candidate = getattr(public_domain, name)
        if (
            inspect.isclass(candidate)
            and issubclass(candidate, BaseModel)
            and candidate not in _ABSTRACT_MODEL_BASES
        ):
            exported.add(candidate)
    return exported


DISCOVERED_PUBLIC_MODELS = tuple(
    sorted(
        _defined_concrete_public_models(models_module)
        | _defined_concrete_public_models(errors_module),
        key=lambda model: model.__name__,
    )
)
DISCOVERED_PUBLIC_MODEL_NAMES = {model.__name__ for model in DISCOVERED_PUBLIC_MODELS}
EXPORTED_PUBLIC_MODELS = _exported_concrete_public_models()


def _snapshot_text(schema: dict[str, object]) -> str:
    """Serialize schema with the project's deterministic snapshot format."""
    return json.dumps(schema, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _resolve_reference(schema: dict[str, object], node: dict[str, object]) -> dict[str, object]:
    """Resolve a local ``$defs`` reference for direct semantic assertions."""
    current = node
    while "$ref" in current:
        ref = current["$ref"]
        assert isinstance(ref, str) and ref.startswith("#/$defs/")
        defs = cast(dict[str, object], schema["$defs"])
        current = cast(dict[str, object], defs[ref.removeprefix("#/$defs/")])
    return current


@pytest.fixture
def snapshot_path_factory():
    def _path(model: type) -> Path:
        return SNAPSHOT_DIR / f"{model.__name__}.json"

    return _path


@pytest.mark.parametrize("model", DISCOVERED_PUBLIC_MODELS, ids=lambda model: model.__name__)
def test_public_model_schema_matches_snapshot(model: type, snapshot_path_factory) -> None:
    """Schema changes fail explicitly; snapshots are never auto-accepted."""
    path = snapshot_path_factory(model)
    assert path.exists(), f"missing snapshot: {path}"
    expected_text = path.read_text(encoding="utf-8")
    actual_text = _snapshot_text(model.model_json_schema())
    if expected_text != actual_text:
        pytest.fail(
            f"schema snapshot mismatch for {model.__name__}:\n"
            f"expected {path}\n"
            f"--- generated ---\n{actual_text}\n"
            f"--- committed ---\n{expected_text}"
        )


@pytest.mark.parametrize("model", DISCOVERED_PUBLIC_MODELS, ids=lambda model: model.__name__)
def test_schema_generation_is_deterministic(model: type) -> None:
    assert _snapshot_text(model.model_json_schema()) == _snapshot_text(model.model_json_schema())


def test_every_defined_concrete_public_model_is_exported() -> None:
    missing_exports = {
        model.__name__ for model in DISCOVERED_PUBLIC_MODELS if model not in EXPORTED_PUBLIC_MODELS
    }
    assert not missing_exports
    assert set(DISCOVERED_PUBLIC_MODELS) == EXPORTED_PUBLIC_MODELS


def test_exported_concrete_public_models_match_snapshot_files_exactly() -> None:
    files = {path.stem for path in SNAPSHOT_DIR.glob("*.json")}
    assert files == {model.__name__ for model in EXPORTED_PUBLIC_MODELS}


def test_public_schema_count_matches_accepted_contract() -> None:
    assert len(DISCOVERED_PUBLIC_MODELS) == len(DISCOVERED_PUBLIC_MODEL_NAMES) == 35


def test_resource_cpu_schema_uses_standard_exclusive_minimum() -> None:
    schema = ResourceRequest.model_json_schema()
    properties = cast(dict[str, object], schema["properties"])
    cpu_schema = properties["cpu_cores"]
    resolved = _resolve_reference(schema, cast(dict[str, object], cpu_schema))
    assert resolved["type"] == "number"
    assert resolved["exclusiveMinimum"] == 0.0

    allowlist_schema = cast(dict[str, object], properties["network_allowlist_refs"])
    items = cast(dict[str, object], allowlist_schema["items"])
    allowlist_ref = _resolve_reference(schema, items)
    assert allowlist_ref["type"] == "string"
    assert allowlist_ref["pattern"] == r"^netref_[a-z0-9][a-z0-9_-]{0,63}$"


def test_interval_and_string_schema_keywords_are_standard_and_semantic() -> None:
    per_class_schema = PerClassSummary.model_json_schema()
    precision = _resolve_reference(
        per_class_schema,
        cast(
            dict[str, object], cast(dict[str, object], per_class_schema["properties"])["precision"]
        ),
    )
    assert precision["type"] == "number"
    assert precision["minimum"] == 0.0
    assert precision["maximum"] == 1.0

    split_schema = SplitPolicy.model_json_schema()
    test_fraction = _resolve_reference(
        split_schema,
        cast(
            dict[str, object], cast(dict[str, object], split_schema["properties"])["test_fraction"]
        ),
    )
    open_branch = cast(list[object], test_fraction["anyOf"])[0]
    open_interval = _resolve_reference(split_schema, cast(dict[str, object], open_branch))
    assert open_interval["exclusiveMinimum"] == 0.0
    assert open_interval["exclusiveMaximum"] == 1.0

    failure_schema = StructuredFailure.model_json_schema()
    category = _resolve_reference(
        failure_schema,
        cast(dict[str, object], cast(dict[str, object], failure_schema["properties"])["category"]),
    )
    assert category["pattern"] == "^[A-Z][A-Z0-9_]{0,63}$"
    assert category["type"] == "string"

    approval_schema = Approval.model_json_schema()
    reason = _resolve_reference(
        approval_schema,
        cast(dict[str, object], cast(dict[str, object], approval_schema["properties"])["reason"]),
    )
    assert reason["minLength"] == 1
    assert reason["maxLength"] == 1024


def _walk_schema(value: object) -> list[dict[str, object]]:
    """Return all object nodes for a schema-keyword audit."""
    if isinstance(value, dict):
        nodes = [cast(dict[str, object], value)]
        for child in value.values():
            nodes.extend(_walk_schema(child))
        return nodes
    if isinstance(value, list):
        nodes: list[dict[str, object]] = []
        for child in value:
            nodes.extend(_walk_schema(child))
        return nodes
    return []


def test_schemas_do_not_emit_nonstandard_gt_ge_lt_le_keywords() -> None:
    forbidden = {"gt", "ge", "lt", "le"}
    for model in DISCOVERED_PUBLIC_MODELS:
        for node in _walk_schema(model.model_json_schema()):
            assert forbidden.isdisjoint(node), (
                f"{model.__name__} emitted nonstandard constraints: {node}"
            )
